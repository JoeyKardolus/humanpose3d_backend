#!/usr/bin/env python3
"""
Train depth refinement model on AIST++ data.

Uses:
- REAL MediaPipe errors (not synthetic)
- View angle from torso orientation
- Cross-joint attention for pose-aware correction
- Biomechanical losses (bone length, symmetry)

Usage:
    uv run --group neural python scripts/train_depth_model.py

Options:
    --data        Path to training data (default: data/training/aistpp_converted)
    --epochs      Number of epochs (default: 50)
    --batch-size  Batch size (default: 64)
    --lr          Learning rate (default: 1e-4)
    --workers     Dataloader workers (default: 4)
    --fp16        Use mixed precision training
    --checkpoint  Resume from checkpoint
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import argparse
import time
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from src.depth_refinement.model import create_model
from src.depth_refinement.losses import (
    DepthRefinementLoss, compute_torso_lengths, bone_variance_loss, apply_bone_locking
)
from src.depth_refinement.dataset import create_dataloaders, create_temporal_dataloaders


def compute_camera_loss(
    pred_camera_pos: torch.Tensor,
    gt_camera_pos: torch.Tensor,
) -> torch.Tensor:
    """Compute loss for camera position prediction.

    Uses both direction error and distance error.

    Args:
        pred_camera_pos: (batch, 3) predicted camera position relative to pelvis
        gt_camera_pos: (batch, 3) ground truth camera position

    Returns:
        Scalar loss tensor
    """
    # Skip if GT camera position is all zeros (old data format)
    if (gt_camera_pos.abs().sum() < 1e-6):
        return torch.tensor(0.0, device=pred_camera_pos.device)

    # Direction loss: cosine similarity between predicted and GT direction
    pred_dir = pred_camera_pos / (torch.norm(pred_camera_pos, dim=-1, keepdim=True) + 1e-8)
    gt_dir = gt_camera_pos / (torch.norm(gt_camera_pos, dim=-1, keepdim=True) + 1e-8)
    direction_loss = (1 - (pred_dir * gt_dir).sum(dim=-1)).mean()

    # Distance loss: relative error in distance
    pred_dist = torch.norm(pred_camera_pos, dim=-1)
    gt_dist = torch.norm(gt_camera_pos, dim=-1)
    distance_loss = ((pred_dist - gt_dist) / (gt_dist + 1e-8)).pow(2).mean()

    return direction_loss + 0.1 * distance_loss


def compute_angle_loss(
    pred_azimuth: torch.Tensor,
    pred_elevation: torch.Tensor,
    gt_azimuth: torch.Tensor,
    gt_elevation: torch.Tensor,
) -> torch.Tensor:
    """Compute loss for camera angle prediction (backward compat).

    Uses circular distance for azimuth (handles 0°/360° wraparound).
    Uses standard MSE for elevation.

    Args:
        pred_azimuth: (batch,) predicted azimuth 0-360°
        pred_elevation: (batch,) predicted elevation -90 to +90°
        gt_azimuth: (batch,) ground truth azimuth
        gt_elevation: (batch,) ground truth elevation

    Returns:
        Scalar loss tensor
    """
    # Circular distance for azimuth (handles 0°/360° wraparound)
    # Convert to radians for angular distance
    pred_az_rad = pred_azimuth * (3.14159265 / 180.0)
    gt_az_rad = gt_azimuth * (3.14159265 / 180.0)

    # Circular distance: 1 - cos(diff) is 0 when angles match, 2 when 180° apart
    az_loss = (1 - torch.cos(pred_az_rad - gt_az_rad)).mean()

    # Standard MSE for elevation (normalized to match model output scale ±30°)
    el_loss = ((pred_elevation - gt_elevation) / 30.0).pow(2).mean()

    return az_loss + el_loss


def batched_bone_variance_loss(poses: torch.Tensor) -> torch.Tensor:
    """Compute bone variance loss for batched temporal windows (vectorized).

    Args:
        poses: (batch, window, 17, 3) poses from multiple windows

    Returns:
        Mean bone variance across all windows
    """
    from src.depth_refinement.losses import COCO_BONES

    # Build index tensors for vectorized bone length computation
    bone_i = torch.tensor([b[0] for b in COCO_BONES], device=poses.device)
    bone_j = torch.tensor([b[1] for b in COCO_BONES], device=poses.device)

    # poses: (batch, window, 17, 3)
    # Get joint positions for all bones at once
    joint_i = poses[:, :, bone_i]  # (batch, window, num_bones, 3)
    joint_j = poses[:, :, bone_j]  # (batch, window, num_bones, 3)

    # Compute bone lengths: (batch, window, num_bones)
    bone_lengths = torch.norm(joint_i - joint_j, dim=-1)

    # Variance within each window (dim=1), then mean across bones and batches
    var_per_window = bone_lengths.var(dim=1)  # (batch, num_bones)

    return var_per_window.mean()


def train_epoch(
    model: nn.Module,
    train_loader,
    optimizer,
    loss_fn: DepthRefinementLoss,
    scaler: GradScaler = None,
    device: str = 'cuda',
    camera_loss_weight: float = 0.3,
    bone_locking: bool = False,
    use_predicted_angles: bool = True,
    angle_noise_std: float = 0.0,
) -> dict:
    """Train for one epoch.

    Args:
        bone_locking: If True, expects temporal windows (batch, window, 17, 3)
                      and computes per-window bone variance loss
        use_predicted_angles: If True (default), use predicted angles for depth correction
        angle_noise_std: Standard deviation of noise to add to angles (degrees)
    """
    model.train()

    total_loss = 0.0
    loss_components = {
        'pose': 0.0, 'bone': 0.0, 'symmetry': 0.0, 'confidence': 0.0,
        'camera': 0.0, 'torso_width': 0.0, 'torso_predictor': 0.0, 'bone_var': 0.0
    }
    num_batches = 0

    pbar = tqdm(train_loader, desc='Train', leave=False)

    for batch in pbar:
        corrupted = batch['corrupted'].to(device)
        ground_truth = batch['ground_truth'].to(device)
        visibility = batch['visibility'].to(device)
        pose_2d = batch['pose_2d'].to(device)
        azimuth = batch['azimuth'].to(device)
        elevation = batch['elevation'].to(device)

        optimizer.zero_grad()

        if bone_locking and corrupted.dim() == 4:
            # Batched temporal windows: (batch, window, 17, 3)
            batch_size, window_size = corrupted.shape[:2]

            # Flatten for model: (batch*window, 17, 3)
            corrupted_flat = corrupted.view(-1, 17, 3)
            ground_truth_flat = ground_truth.view(-1, 17, 3)
            visibility_flat = visibility.view(-1, 17)
            pose_2d_flat = pose_2d.view(-1, 17, 2)
            azimuth_flat = azimuth.view(-1)
            elevation_flat = elevation.view(-1)

            gt_torso_lengths = compute_torso_lengths(ground_truth_flat)

            if scaler is not None:
                with autocast():
                    output = model(corrupted_flat, visibility_flat, pose_2d=pose_2d_flat,
                                   azimuth=azimuth_flat, elevation=elevation_flat,
                                   use_predicted_angles=use_predicted_angles,
                                   angle_noise_std=angle_noise_std)
                    camera_loss = compute_angle_loss(
                        output['pred_azimuth'], output['pred_elevation'],
                        azimuth_flat, elevation_flat
                    )

                    corrected_flat = corrupted_flat + output['delta_xyz']

                    # Reshape back for per-window bone variance
                    corrected = corrected_flat.view(batch_size, window_size, 17, 3)

                    # Per-window bone variance (same person within each window)
                    bone_var = batched_bone_variance_loss(corrected)

                    # Standard losses on flattened data
                    output_for_loss = {**output, 'delta_xyz': corrected_flat - corrupted_flat}
                    losses = loss_fn(output_for_loss, corrupted_flat, ground_truth_flat,
                                     visibility_flat, gt_torso_lengths)
                    losses['camera'] = camera_loss
                    losses['bone_var'] = bone_var
                    losses['total'] = losses['total'] + camera_loss_weight * camera_loss + 0.1 * bone_var

                scaler.scale(losses['total']).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                output = model(corrupted_flat, visibility_flat, pose_2d=pose_2d_flat,
                               azimuth=azimuth_flat, elevation=elevation_flat,
                               use_predicted_angles=use_predicted_angles,
                               angle_noise_std=angle_noise_std)
                camera_loss = compute_angle_loss(
                    output['pred_azimuth'], output['pred_elevation'],
                    azimuth_flat, elevation_flat
                )

                corrected_flat = corrupted_flat + output['delta_xyz']
                corrected = corrected_flat.view(batch_size, window_size, 17, 3)
                bone_var = batched_bone_variance_loss(corrected)

                output_for_loss = {**output, 'delta_xyz': corrected_flat - corrupted_flat}
                losses = loss_fn(output_for_loss, corrupted_flat, ground_truth_flat,
                                 visibility_flat, gt_torso_lengths)
                losses['camera'] = camera_loss
                losses['bone_var'] = bone_var
                losses['total'] = losses['total'] + camera_loss_weight * camera_loss + 0.1 * bone_var

                losses['total'].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
        else:
            # Standard training: (batch, 17, 3)
            gt_torso_lengths = compute_torso_lengths(ground_truth)

            if scaler is not None:
                with autocast():
                    output = model(corrupted, visibility, pose_2d=pose_2d, azimuth=azimuth, elevation=elevation,
                                   use_predicted_angles=use_predicted_angles, angle_noise_std=angle_noise_std)
                    camera_loss = compute_angle_loss(
                        output['pred_azimuth'], output['pred_elevation'],
                        azimuth, elevation
                    )
                    corrected = corrupted + output['delta_xyz']
                    output_for_loss = {**output, 'delta_xyz': corrected - corrupted}
                    losses = loss_fn(output_for_loss, corrupted, ground_truth, visibility, gt_torso_lengths)
                    losses['camera'] = camera_loss
                    bone_var = bone_variance_loss(corrected)
                    losses['bone_var'] = bone_var
                    losses['total'] = losses['total'] + camera_loss_weight * camera_loss + 0.1 * bone_var

                scaler.scale(losses['total']).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                output = model(corrupted, visibility, pose_2d=pose_2d, azimuth=azimuth, elevation=elevation,
                               use_predicted_angles=use_predicted_angles, angle_noise_std=angle_noise_std)
                camera_loss = compute_angle_loss(
                    output['pred_azimuth'], output['pred_elevation'],
                    azimuth, elevation
                )
                corrected = corrupted + output['delta_xyz']
                output_for_loss = {**output, 'delta_xyz': corrected - corrupted}
                losses = loss_fn(output_for_loss, corrupted, ground_truth, visibility, gt_torso_lengths)
                losses['camera'] = camera_loss
                bone_var = bone_variance_loss(corrected)
                losses['bone_var'] = bone_var
                losses['total'] = losses['total'] + camera_loss_weight * camera_loss + 0.1 * bone_var

                losses['total'].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

        total_loss += losses['total'].item()
        for key in loss_components:
            if key in losses:
                loss_components[key] += losses[key].item()
        num_batches += 1

        pbar.set_postfix({
            'loss': f"{losses['total'].item():.4f}",
            'pose': f"{losses['pose'].item():.4f}",
            'cam': f"{losses['camera'].item():.4f}",
            'bvar': f"{losses['bone_var'].item():.4f}",
        })

    return {
        'total': total_loss / num_batches,
        **{k: v / num_batches for k, v in loss_components.items()},
    }


@torch.no_grad()
def validate(
    model: nn.Module,
    val_loader,
    loss_fn: DepthRefinementLoss,
    device: str = 'cuda',
    camera_loss_weight: float = 0.3,
    bone_locking: bool = False,
    use_predicted_angles: bool = True,
) -> dict:
    """Validate model.

    Args:
        use_predicted_angles: If True (default), use predicted angles for depth correction
    """
    model.eval()

    total_loss = 0.0
    loss_components = {
        'pose': 0.0, 'bone': 0.0, 'symmetry': 0.0, 'confidence': 0.0,
        'camera': 0.0, 'torso_width': 0.0, 'torso_predictor': 0.0, 'bone_var': 0.0
    }
    num_batches = 0

    # Additional metrics
    total_depth_error = 0.0
    total_azimuth_error = 0.0
    total_elevation_error = 0.0
    total_samples = 0

    for batch in tqdm(val_loader, desc='Val', leave=False):
        corrupted = batch['corrupted'].to(device)
        ground_truth = batch['ground_truth'].to(device)
        visibility = batch['visibility'].to(device)
        pose_2d = batch['pose_2d'].to(device)
        azimuth = batch['azimuth'].to(device)
        elevation = batch['elevation'].to(device)

        if bone_locking and corrupted.dim() == 4:
            # Batched temporal windows: (batch, window, 17, 3)
            batch_size, window_size = corrupted.shape[:2]

            # Flatten for model
            corrupted_flat = corrupted.view(-1, 17, 3)
            ground_truth_flat = ground_truth.view(-1, 17, 3)
            visibility_flat = visibility.view(-1, 17)
            pose_2d_flat = pose_2d.view(-1, 17, 2)
            azimuth_flat = azimuth.view(-1)
            elevation_flat = elevation.view(-1)

            gt_torso_lengths = compute_torso_lengths(ground_truth_flat)

            output = model(corrupted_flat, visibility_flat, pose_2d=pose_2d_flat,
                           azimuth=azimuth_flat, elevation=elevation_flat,
                           use_predicted_angles=use_predicted_angles)
            camera_loss = compute_angle_loss(
                output['pred_azimuth'], output['pred_elevation'],
                azimuth_flat, elevation_flat
            )

            corrected_flat = corrupted_flat + output['delta_xyz']
            corrected = corrected_flat.view(batch_size, window_size, 17, 3)
            bone_var = batched_bone_variance_loss(corrected)

            output_for_loss = {**output, 'delta_xyz': corrected_flat - corrupted_flat}
            losses = loss_fn(output_for_loss, corrupted_flat, ground_truth_flat,
                             visibility_flat, gt_torso_lengths)
            losses['camera'] = camera_loss
            losses['bone_var'] = bone_var
            losses['total'] = losses['total'] + camera_loss_weight * camera_loss + 0.1 * bone_var

            # Metrics on flattened data
            depth_error = (corrected_flat[:, :, 2] - ground_truth_flat[:, :, 2]).abs().mean()
            total_depth_error += depth_error.item() * corrupted_flat.size(0)
            az_diff = torch.abs(output['pred_azimuth'] - azimuth_flat)
            az_diff = torch.min(az_diff, 360.0 - az_diff)
            total_azimuth_error += az_diff.sum().item()
            el_diff = torch.abs(output['pred_elevation'] - elevation_flat)
            total_elevation_error += el_diff.sum().item()
            total_samples += corrupted_flat.size(0)
        else:
            # Standard validation
            gt_torso_lengths = compute_torso_lengths(ground_truth)

            output = model(corrupted, visibility, pose_2d=pose_2d, azimuth=azimuth, elevation=elevation,
                           use_predicted_angles=use_predicted_angles)
            camera_loss = compute_angle_loss(
                output['pred_azimuth'], output['pred_elevation'],
                azimuth, elevation
            )

            corrected = corrupted + output['delta_xyz']
            output_for_loss = {**output, 'delta_xyz': corrected - corrupted}
            losses = loss_fn(output_for_loss, corrupted, ground_truth, visibility, gt_torso_lengths)
            losses['camera'] = camera_loss
            bone_var = bone_variance_loss(corrected)
            losses['bone_var'] = bone_var
            losses['total'] = losses['total'] + camera_loss_weight * camera_loss + 0.1 * bone_var

            depth_error = (corrected[:, :, 2] - ground_truth[:, :, 2]).abs().mean()
            total_depth_error += depth_error.item() * corrupted.size(0)
            az_diff = torch.abs(output['pred_azimuth'] - azimuth)
            az_diff = torch.min(az_diff, 360.0 - az_diff)
            total_azimuth_error += az_diff.sum().item()
            el_diff = torch.abs(output['pred_elevation'] - elevation)
            total_elevation_error += el_diff.sum().item()
            total_samples += corrupted.size(0)

        total_loss += losses['total'].item()
        for key in loss_components:
            if key in losses:
                loss_components[key] += losses[key].item()
        num_batches += 1

    return {
        'total': total_loss / num_batches,
        **{k: v / num_batches for k, v in loss_components.items()},
        'depth_error_m': total_depth_error / total_samples,
        'azimuth_error_deg': total_azimuth_error / total_samples,
        'elevation_error_deg': total_elevation_error / total_samples,
    }


def main():
    parser = argparse.ArgumentParser(description='Train depth refinement model')
    parser.add_argument('--data', type=str, default='data/training/aistpp_converted',
                        help='Path to training data')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--workers', type=int, default=4, help='Dataloader workers')
    parser.add_argument('--fp16', action='store_true', help='Use mixed precision')
    parser.add_argument('--checkpoint', type=str, help='Resume from checkpoint')
    parser.add_argument('--save-dir', type=str, default='models/checkpoints',
                        help='Directory to save checkpoints')
    # Model architecture
    parser.add_argument('--d-model', type=int, default=64, help='Model hidden dimension')
    parser.add_argument('--num-layers', type=int, default=4, help='Number of transformer layers')
    parser.add_argument('--num-heads', type=int, default=4, help='Number of attention heads')
    parser.add_argument('--elepose', action='store_true',
                        help='Use ElePose backbone for hybrid angle prediction (fuses deep 2D features with existing features)')
    parser.add_argument('--elepose-hidden-dim', type=int, default=1024,
                        help='Hidden dimension for ElePose backbone')
    # Bone locking (temporal training)
    parser.add_argument('--bone-locking', action='store_true',
                        help='Use temporal windows with bone locking (same person per batch)')
    parser.add_argument('--window-size', type=int, default=50,
                        help='Temporal window size for bone locking (default: 50)')
    # Teacher forcing / angle usage
    parser.add_argument('--use-gt-angles', action='store_true',
                        help='Use GT angles for depth correction (legacy teacher forcing, not recommended)')
    parser.add_argument('--angle-noise-std', type=float, default=0.0,
                        help='Add noise to angles during training for robustness (degrees, 0 to disable)')
    parser.add_argument('--angle-noise-warmup', type=int, default=10,
                        help='Epochs before starting angle noise (let model learn basic patterns first)')
    args = parser.parse_args()

    # Setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Create save directory
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Data
    print(f"\nLoading data from {args.data}")
    if args.bone_locking:
        # For temporal training, batch multiple windows together
        # Each window has window_size frames, so effective batch = windows_per_batch * window_size
        # With 16 windows of 50 frames = 800 frames per batch (good GPU utilization)
        windows_per_batch = max(1, args.batch_size // args.window_size)
        print(f"Using temporal windows (window_size={args.window_size}, {windows_per_batch} windows/batch)")
        train_loader, val_loader = create_temporal_dataloaders(
            args.data,
            window_size=args.window_size,
            batch_size=windows_per_batch,
            num_workers=args.workers,
        )
    else:
        train_loader, val_loader = create_dataloaders(
            args.data,
            batch_size=args.batch_size,
            num_workers=args.workers,
        )
    print(f"Train: {len(train_loader)} batches, Val: {len(val_loader)} batches")

    # Model
    model = create_model(
        num_joints=17,
        d_model=args.d_model,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        use_elepose=args.elepose,
        elepose_hidden_dim=args.elepose_hidden_dim,
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel parameters: {num_params:,}")

    # Loss - now uses pose_weight for full 3D corrections
    loss_fn = DepthRefinementLoss(
        pose_weight=1.0,
        bone_weight=0.1,
        symmetry_weight=0.05,
        confidence_weight=0.1,
        depth_axis_weight=2.0,  # Extra weight on Z axis
    )

    # Optimizer
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # Mixed precision
    scaler = GradScaler() if args.fp16 and device == 'cuda' else None
    if scaler:
        print("Using mixed precision training (FP16)")

    # Resume from checkpoint
    start_epoch = 0
    best_val_loss = float('inf')

    if args.checkpoint:
        print(f"\nLoading checkpoint: {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        print(f"Resuming from epoch {start_epoch}")

    # Training loop
    print("\n" + "=" * 60)
    print("TRAINING")
    print("=" * 60)
    print(f"Using predicted angles for depth: {not args.use_gt_angles}")
    print(f"Angle noise std: {args.angle_noise_std}° (after epoch {args.angle_noise_warmup})")

    for epoch in range(start_epoch, args.epochs):
        epoch_start = time.time()

        # Compute angle noise for this epoch (warmup then full noise)
        current_angle_noise = 0.0
        if epoch >= args.angle_noise_warmup:
            current_angle_noise = args.angle_noise_std

        # Train
        train_losses = train_epoch(
            model, train_loader, optimizer, loss_fn, scaler, device,
            bone_locking=args.bone_locking,
            use_predicted_angles=not args.use_gt_angles,
            angle_noise_std=current_angle_noise,
        )

        # Validate (always use predicted angles, no noise)
        val_losses = validate(
            model, val_loader, loss_fn, device,
            bone_locking=args.bone_locking,
            use_predicted_angles=True,  # Always use predicted for validation (like inference)
        )

        # Update LR
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        epoch_time = time.time() - epoch_start

        # Print progress
        print(f"\nEpoch {epoch + 1}/{args.epochs} ({epoch_time:.1f}s) | LR: {current_lr:.2e}")
        print(f"  Train: loss={train_losses['total']:.4f} "
              f"(pose={train_losses['pose']:.4f}, bone={train_losses['bone']:.4f}, bvar={train_losses['bone_var']:.4f}, cam={train_losses['camera']:.4f})")
        print(f"  Val:   loss={val_losses['total']:.4f} "
              f"(pose={val_losses['pose']:.4f}, bone={val_losses['bone']:.4f}, bvar={val_losses['bone_var']:.4f}, cam={val_losses['camera']:.4f})")
        print(f"  Val depth error: {val_losses['depth_error_m']*100:.2f} cm")
        print(f"  Val angle error: azimuth={val_losses['azimuth_error_deg']:.1f}°, elevation={val_losses['elevation_error_deg']:.1f}°")

        # Save best model
        if val_losses['total'] < best_val_loss:
            best_val_loss = val_losses['total']
            checkpoint_path = save_dir / 'best_depth_model.pth'
            torch.save({
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'best_val_loss': best_val_loss,
                'config': {
                    'num_joints': 17,
                    'd_model': args.d_model,
                    'num_heads': args.num_heads,
                    'num_layers': args.num_layers,
                    'use_elepose': args.elepose,
                    'elepose_hidden_dim': args.elepose_hidden_dim,
                },
            }, checkpoint_path)
            print(f"  -> Saved best model (val_loss={best_val_loss:.4f})")

        # Save periodic checkpoint
        if (epoch + 1) % 10 == 0:
            checkpoint_path = save_dir / f'checkpoint_epoch{epoch + 1}.pth'
            torch.save({
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'best_val_loss': best_val_loss,
                'config': {
                    'num_joints': 17,
                    'd_model': args.d_model,
                    'num_heads': args.num_heads,
                    'num_layers': args.num_layers,
                    'use_elepose': args.elepose,
                    'elepose_hidden_dim': args.elepose_hidden_dim,
                },
            }, checkpoint_path)
            print(f"  -> Saved checkpoint: {checkpoint_path.name}")

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Model saved to: {save_dir / 'best_depth_model.pth'}")
    print("=" * 60)


if __name__ == '__main__':
    main()

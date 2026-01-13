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
from src.depth_refinement.losses import DepthRefinementLoss
from src.depth_refinement.dataset import create_dataloaders


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

    # Standard MSE for elevation (normalized to similar scale)
    el_loss = ((pred_elevation - gt_elevation) / 90.0).pow(2).mean()

    return az_loss + el_loss


def train_epoch(
    model: nn.Module,
    train_loader,
    optimizer,
    loss_fn: DepthRefinementLoss,
    scaler: GradScaler = None,
    device: str = 'cuda',
    camera_loss_weight: float = 0.1,
) -> dict:
    """Train for one epoch."""
    model.train()

    total_loss = 0.0
    loss_components = {'depth': 0.0, 'bone': 0.0, 'symmetry': 0.0, 'confidence': 0.0, 'camera': 0.0}
    num_batches = 0

    pbar = tqdm(train_loader, desc='Train', leave=False)

    for batch in pbar:
        corrupted = batch['corrupted'].to(device)
        ground_truth = batch['ground_truth'].to(device)
        visibility = batch['visibility'].to(device)
        pose_2d = batch['pose_2d'].to(device)  # 2D pose for camera prediction
        camera_pos = batch['camera_pos'].to(device)
        azimuth = batch['azimuth'].to(device)
        elevation = batch['elevation'].to(device)

        optimizer.zero_grad()

        # Check if we have valid camera position data (new format)
        has_camera_pos = camera_pos.abs().sum() > 1e-6

        if scaler is not None:
            with autocast():
                if has_camera_pos:
                    # New format: use camera position + 2D pose
                    output = model(corrupted, visibility, pose_2d=pose_2d, camera_pos=camera_pos)
                    camera_loss = compute_camera_loss(output['pred_camera_pos'], camera_pos)
                else:
                    # Old format: use azimuth/elevation directly
                    output = model(corrupted, visibility, pose_2d=pose_2d, azimuth=azimuth, elevation=elevation)
                    camera_loss = compute_angle_loss(
                        output['pred_azimuth'], output['pred_elevation'],
                        azimuth, elevation
                    )

                losses = loss_fn(output, corrupted, ground_truth, visibility)
                losses['camera'] = camera_loss
                losses['total'] = losses['total'] + camera_loss_weight * camera_loss

            scaler.scale(losses['total']).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            if has_camera_pos:
                output = model(corrupted, visibility, pose_2d=pose_2d, camera_pos=camera_pos)
                camera_loss = compute_camera_loss(output['pred_camera_pos'], camera_pos)
            else:
                output = model(corrupted, visibility, pose_2d=pose_2d, azimuth=azimuth, elevation=elevation)
                camera_loss = compute_angle_loss(
                    output['pred_azimuth'], output['pred_elevation'],
                    azimuth, elevation
                )

            losses = loss_fn(output, corrupted, ground_truth, visibility)
            losses['camera'] = camera_loss
            losses['total'] = losses['total'] + camera_loss_weight * camera_loss

            losses['total'].backward()
            optimizer.step()

        total_loss += losses['total'].item()
        for key in loss_components:
            if key in losses:
                loss_components[key] += losses[key].item()
        num_batches += 1

        pbar.set_postfix({
            'loss': f"{losses['total'].item():.4f}",
            'depth': f"{losses['depth'].item():.4f}",
            'cam': f"{losses['camera'].item():.4f}",
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
    camera_loss_weight: float = 0.1,
) -> dict:
    """Validate model."""
    model.eval()

    total_loss = 0.0
    loss_components = {'depth': 0.0, 'bone': 0.0, 'symmetry': 0.0, 'confidence': 0.0, 'camera': 0.0}
    num_batches = 0

    # Additional metrics
    total_depth_error = 0.0
    total_azimuth_error = 0.0
    total_elevation_error = 0.0
    total_camera_dir_error = 0.0
    total_samples = 0

    for batch in tqdm(val_loader, desc='Val', leave=False):
        corrupted = batch['corrupted'].to(device)
        ground_truth = batch['ground_truth'].to(device)
        visibility = batch['visibility'].to(device)
        pose_2d = batch['pose_2d'].to(device)  # 2D pose for camera prediction
        camera_pos = batch['camera_pos'].to(device)
        azimuth = batch['azimuth'].to(device)
        elevation = batch['elevation'].to(device)

        # Check if we have valid camera position data
        has_camera_pos = camera_pos.abs().sum() > 1e-6

        if has_camera_pos:
            output = model(corrupted, visibility, pose_2d=pose_2d, camera_pos=camera_pos)
            camera_loss = compute_camera_loss(output['pred_camera_pos'], camera_pos)
        else:
            output = model(corrupted, visibility, pose_2d=pose_2d, azimuth=azimuth, elevation=elevation)
            camera_loss = compute_angle_loss(
                output['pred_azimuth'], output['pred_elevation'],
                azimuth, elevation
            )

        losses = loss_fn(output, corrupted, ground_truth, visibility)
        losses['camera'] = camera_loss
        losses['total'] = losses['total'] + camera_loss_weight * camera_loss

        total_loss += losses['total'].item()
        for key in loss_components:
            if key in losses:
                loss_components[key] += losses[key].item()
        num_batches += 1

        # Compute actual depth error after correction
        pred_delta_z = output['delta_z']
        corrected_z = corrupted[:, :, 2] + pred_delta_z
        gt_z = ground_truth[:, :, 2]
        depth_error = (corrected_z - gt_z).abs().mean()
        total_depth_error += depth_error.item() * corrupted.size(0)

        # Compute angle prediction errors (from predicted camera position)
        az_diff = torch.abs(output['pred_azimuth'] - azimuth)
        az_diff = torch.min(az_diff, 360.0 - az_diff)  # Handle wraparound
        total_azimuth_error += az_diff.sum().item()

        el_diff = torch.abs(output['pred_elevation'] - elevation)
        total_elevation_error += el_diff.sum().item()

        # Camera direction error (if available)
        if has_camera_pos:
            pred_dir = output['pred_camera_pos'] / (torch.norm(output['pred_camera_pos'], dim=-1, keepdim=True) + 1e-8)
            gt_dir = camera_pos / (torch.norm(camera_pos, dim=-1, keepdim=True) + 1e-8)
            dir_error = torch.acos(torch.clamp((pred_dir * gt_dir).sum(dim=-1), -1, 1))
            total_camera_dir_error += torch.rad2deg(dir_error).sum().item()

        total_samples += corrupted.size(0)

    return {
        'total': total_loss / num_batches,
        **{k: v / num_batches for k, v in loss_components.items()},
        'depth_error_m': total_depth_error / total_samples,
        'azimuth_error_deg': total_azimuth_error / total_samples,
        'elevation_error_deg': total_elevation_error / total_samples,
        'camera_dir_error_deg': total_camera_dir_error / total_samples if total_camera_dir_error > 0 else 0,
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
    train_loader, val_loader = create_dataloaders(
        args.data,
        batch_size=args.batch_size,
        num_workers=args.workers,
    )
    print(f"Train: {len(train_loader)} batches, Val: {len(val_loader)} batches")

    # Model
    model = create_model(
        num_joints=17,
        d_model=64,
        num_heads=4,
        num_layers=4,
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel parameters: {num_params:,}")

    # Loss
    loss_fn = DepthRefinementLoss(
        depth_weight=1.0,
        bone_weight=0.1,
        symmetry_weight=0.05,
        confidence_weight=0.1,
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

    for epoch in range(start_epoch, args.epochs):
        epoch_start = time.time()

        # Train
        train_losses = train_epoch(model, train_loader, optimizer, loss_fn, scaler, device)

        # Validate
        val_losses = validate(model, val_loader, loss_fn, device)

        # Update LR
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        epoch_time = time.time() - epoch_start

        # Print progress
        print(f"\nEpoch {epoch + 1}/{args.epochs} ({epoch_time:.1f}s) | LR: {current_lr:.2e}")
        print(f"  Train: loss={train_losses['total']:.4f} "
              f"(depth={train_losses['depth']:.4f}, bone={train_losses['bone']:.4f}, cam={train_losses['camera']:.4f})")
        print(f"  Val:   loss={val_losses['total']:.4f} "
              f"(depth={val_losses['depth']:.4f}, bone={val_losses['bone']:.4f}, cam={val_losses['camera']:.4f})")
        print(f"  Val depth error: {val_losses['depth_error_m']*100:.2f} cm")
        print(f"  Val angle error: azimuth={val_losses['azimuth_error_deg']:.1f}°, elevation={val_losses['elevation_error_deg']:.1f}°")
        if val_losses['camera_dir_error_deg'] > 0:
            print(f"  Val camera direction error: {val_losses['camera_dir_error_deg']:.1f}°")

        # Save best model
        if val_losses['total'] < best_val_loss:
            best_val_loss = val_losses['total']
            checkpoint_path = save_dir / 'best_model.pth'
            torch.save({
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'best_val_loss': best_val_loss,
                'config': {
                    'num_joints': 17,
                    'd_model': 64,
                    'num_heads': 4,
                    'num_layers': 4,
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
            }, checkpoint_path)
            print(f"  -> Saved checkpoint: {checkpoint_path.name}")

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Model saved to: {save_dir / 'best_model.pth'}")
    print("=" * 60)


if __name__ == '__main__':
    main()

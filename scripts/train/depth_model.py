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
# scripts/train/depth_model.py -> scripts/train -> scripts -> project_root
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import argparse
import time
import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR
from tqdm import tqdm

# Compact logging mode (disable tqdm progress bars when piped to file)
IS_TTY = sys.stdout.isatty()

# Optional advanced optimizers - install with: pip install lion-pytorch pytorch-optimizer
ADVANCED_OPTIMIZERS_AVAILABLE = {}
try:
    from lion_pytorch import Lion
    ADVANCED_OPTIMIZERS_AVAILABLE['lion'] = Lion
except ImportError:
    pass

try:
    from pytorch_optimizer import AdEMAMix, Sophia, ScheduleFreeAdamW, SOAP
    ADVANCED_OPTIMIZERS_AVAILABLE['ademamix'] = AdEMAMix
    ADVANCED_OPTIMIZERS_AVAILABLE['sophia'] = Sophia
    ADVANCED_OPTIMIZERS_AVAILABLE['schedule_free'] = ScheduleFreeAdamW
    ADVANCED_OPTIMIZERS_AVAILABLE['soap'] = SOAP
except ImportError:
    pass


def create_optimizer(model, optimizer_name: str, lr: float, weight_decay: float = 0.01):
    """Create optimizer based on name.

    Optimizer recommendations:
    - adamw: Default, works well for most cases
    - lion: 50% less memory, good for large batches (64+), use 3-10x smaller LR
    - ademamix: Dual EMA for better convergence, good for long training
    - sophia: Second-order, good for transformers, use ~2x smaller LR
    - schedule_free: No LR schedule needed, simplifies training
    - soap: Second-order with Adam, very stable

    Args:
        model: PyTorch model
        optimizer_name: One of 'adamw', 'lion', 'ademamix', 'sophia', 'schedule_free', 'soap'
        lr: Base learning rate (will be adjusted for some optimizers)
        weight_decay: Weight decay (will be adjusted for some optimizers)

    Returns:
        optimizer, needs_scheduler (bool)
    """
    params = model.parameters()

    if optimizer_name == 'adamw':
        return AdamW(params, lr=lr, weight_decay=weight_decay, betas=(0.9, 0.999)), True

    elif optimizer_name == 'lion':
        if 'lion' not in ADVANCED_OPTIMIZERS_AVAILABLE:
            raise ImportError("Lion not available. Install with: pip install lion-pytorch")
        # Lion needs 3-10x smaller LR and 3-10x larger weight decay
        lion_lr = lr / 5
        lion_wd = weight_decay * 5
        print(f"  Lion: Adjusted LR {lr} -> {lion_lr}, WD {weight_decay} -> {lion_wd}")
        return Lion(params, lr=lion_lr, weight_decay=lion_wd, betas=(0.9, 0.99)), True

    elif optimizer_name == 'ademamix':
        if 'ademamix' not in ADVANCED_OPTIMIZERS_AVAILABLE:
            raise ImportError("AdEMAMix not available. Install with: pip install pytorch-optimizer")
        # AdEMAMix uses dual EMAs - works well with standard LR
        return AdEMAMix(params, lr=lr, weight_decay=weight_decay), True

    elif optimizer_name == 'sophia':
        if 'sophia' not in ADVANCED_OPTIMIZERS_AVAILABLE:
            raise ImportError("Sophia not available. Install with: pip install pytorch-optimizer")
        # Sophia is second-order - use slightly smaller LR
        sophia_lr = lr / 2
        print(f"  Sophia: Adjusted LR {lr} -> {sophia_lr}")
        return Sophia(params, lr=sophia_lr, weight_decay=weight_decay, rho=0.04), True

    elif optimizer_name == 'schedule_free':
        if 'schedule_free' not in ADVANCED_OPTIMIZERS_AVAILABLE:
            raise ImportError("ScheduleFreeAdamW not available. Install with: pip install pytorch-optimizer")
        # Schedule-free doesn't need LR scheduler
        return ScheduleFreeAdamW(params, lr=lr, weight_decay=weight_decay), False

    elif optimizer_name == 'soap':
        if 'soap' not in ADVANCED_OPTIMIZERS_AVAILABLE:
            raise ImportError("SOAP not available. Install with: pip install pytorch-optimizer")
        # SOAP is stabilized Shampoo+Adam - very stable training
        return SOAP(params, lr=lr, weight_decay=weight_decay), True

    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}. "
                        f"Available: adamw, lion, ademamix, sophia, schedule_free, soap")

from src.depth_refinement.model import create_model, solve_depth_least_squares
from src.depth_refinement.losses import (
    DepthRefinementLoss, compute_torso_lengths, bone_variance_loss, apply_bone_locking,
    limb_orientation_loss, limb_orientation_angle_error,
    solved_depth_loss, scale_factor_regularization
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
    use_limb_orientations: bool = False,
    limb_orientation_weight: float = 0.5,
    use_least_squares: bool = False,
    projection_loss_weight: float = 0.3,
    use_amp: bool = False,
    amp_dtype: torch.dtype = torch.float16,
) -> dict:
    """Train for one epoch.

    Args:
        bone_locking: If True, expects temporal windows (batch, window, 17, 3)
                      and computes per-window bone variance loss
        use_predicted_angles: If True (default), use predicted angles for depth correction
        angle_noise_std: Standard deviation of noise to add to angles (degrees)
        use_limb_orientations: If True, compute limb orientation loss (POF-inspired)
        limb_orientation_weight: Weight for limb orientation loss
        use_least_squares: If True, use MTC-style least-squares depth solver
        projection_loss_weight: Weight for projection consistency loss
        use_amp: If True, use automatic mixed precision
        amp_dtype: torch.float16 or torch.bfloat16
    """
    model.train()

    total_loss = 0.0
    loss_components = {
        'pose': 0.0, 'bone': 0.0, 'symmetry': 0.0, 'confidence': 0.0,
        'camera': 0.0, 'torso_width': 0.0, 'torso_predictor': 0.0, 'bone_var': 0.0,
        'limb_orient': 0.0, 'solved_depth': 0.0, 'scale_reg': 0.0
    }
    num_batches = 0
    skipped_batches = 0

    pbar = tqdm(train_loader, desc='Train', leave=False, disable=not IS_TTY)

    for batch in pbar:
        corrupted = batch['corrupted'].to(device)
        ground_truth = batch['ground_truth'].to(device)
        visibility = batch['visibility'].to(device)
        pose_2d = batch['pose_2d'].to(device)
        azimuth = batch['azimuth'].to(device)
        elevation = batch['elevation'].to(device)

        # Get projected 2D (GT 3D projected to 2D - CRITICAL for POF foreshortening)
        projected_2d = None
        if 'projected_2d' in batch:
            projected_2d = batch['projected_2d'].to(device)

        # Get GT limb orientations if available (for POF-inspired training)
        gt_limb_orientations = None
        if use_limb_orientations and 'gt_limb_orientations' in batch:
            gt_limb_orientations = batch['gt_limb_orientations'].to(device)

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
            projected_2d_flat = projected_2d.view(-1, 17, 2) if projected_2d is not None else None

            gt_torso_lengths = compute_torso_lengths(ground_truth_flat)

            if scaler is not None:
                with autocast():
                    output = model(corrupted_flat, visibility_flat, pose_2d=pose_2d_flat,
                                   azimuth=azimuth_flat, elevation=elevation_flat,
                                   use_predicted_angles=use_predicted_angles,
                                   angle_noise_std=angle_noise_std,
                                   projected_2d=projected_2d_flat)
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

                # Skip batch if NaN/Inf detected (FP16 overflow)
                if torch.isnan(losses['total']) or torch.isinf(losses['total']):
                    print(f"\nWarning: NaN/Inf loss detected in bone_locking FP16 path, skipping batch")
                    optimizer.zero_grad()
                    skipped_batches += 1
                    continue

                scaler.scale(losses['total']).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                output = model(corrupted_flat, visibility_flat, pose_2d=pose_2d_flat,
                               azimuth=azimuth_flat, elevation=elevation_flat,
                               use_predicted_angles=use_predicted_angles,
                               angle_noise_std=angle_noise_std,
                               projected_2d=projected_2d_flat)
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

                # Skip batch if NaN/Inf detected
                if torch.isnan(losses['total']) or torch.isinf(losses['total']):
                    print(f"\nWarning: NaN/Inf loss detected in bone_locking path, skipping batch")
                    optimizer.zero_grad()
                    skipped_batches += 1
                    continue

                losses['total'].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
        else:
            # Standard training: (batch, 17, 3)
            gt_torso_lengths = compute_torso_lengths(ground_truth)

            # Forward pass with optional AMP (supports both fp16 and bf16)
            with autocast('cuda', dtype=amp_dtype, enabled=use_amp):
                output = model(corrupted, visibility, pose_2d=pose_2d, azimuth=azimuth, elevation=elevation,
                               use_predicted_angles=use_predicted_angles, angle_noise_std=angle_noise_std,
                               projected_2d=projected_2d)
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

                # Limb orientation loss (POF-inspired)
                if use_limb_orientations and gt_limb_orientations is not None and 'pred_limb_orientations' in output:
                    limb_loss = limb_orientation_loss(output['pred_limb_orientations'], gt_limb_orientations, visibility)
                    losses['limb_orient'] = limb_loss
                    losses['total'] = losses['total'] + limb_orientation_weight * limb_loss

                    # Solved depth loss (MTC-style)
                    # Trains network to predict orientations that lead to correct depths
                    if use_least_squares and projection_loss_weight > 0:
                        solved_pose, scale_factors = solve_depth_least_squares(
                            pose_2d, output['pred_limb_orientations'], corrupted
                        )
                        depth_loss = solved_depth_loss(solved_pose, ground_truth, visibility)
                        scale_reg = scale_factor_regularization(scale_factors)
                        losses['solved_depth'] = depth_loss
                        losses['scale_reg'] = scale_reg
                        losses['total'] = losses['total'] + projection_loss_weight * depth_loss + 0.1 * scale_reg

            # Skip batch if NaN/Inf detected (can happen with FP16)
            if torch.isnan(losses['total']) or torch.isinf(losses['total']):
                print(f"\nWarning: NaN/Inf loss detected, skipping batch")
                optimizer.zero_grad()
                skipped_batches += 1
                continue

            # Backward pass - use scaler only for fp16 (bf16 doesn't need it)
            if scaler is not None:
                scaler.scale(losses['total']).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
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
    use_limb_orientations: bool = False,
    limb_orientation_weight: float = 0.5,
    use_least_squares: bool = False,
    projection_loss_weight: float = 0.3,
) -> dict:
    """Validate model.

    Args:
        use_predicted_angles: If True (default), use predicted angles for depth correction
        use_limb_orientations: If True, compute limb orientation loss (POF-inspired)
        limb_orientation_weight: Weight for limb orientation loss
        use_least_squares: If True, use MTC-style least-squares depth solver
        projection_loss_weight: Weight for projection consistency loss
    """
    model.eval()

    total_loss = 0.0
    loss_components = {
        'pose': 0.0, 'bone': 0.0, 'symmetry': 0.0, 'confidence': 0.0,
        'camera': 0.0, 'torso_width': 0.0, 'torso_predictor': 0.0, 'bone_var': 0.0,
        'limb_orient': 0.0, 'solved_depth': 0.0, 'scale_reg': 0.0
    }
    num_batches = 0

    # Additional metrics
    total_depth_error = 0.0
    total_azimuth_error = 0.0
    total_elevation_error = 0.0
    total_limb_angle_error = 0.0  # Limb orientation error in degrees
    total_samples = 0

    # Per-limb error tracking (for diagnostics)
    limb_errors = torch.zeros(14)  # Sum of errors per limb (14 limbs with cross-body diagonals)
    limb_counts = 0

    # Track azimuth distribution (for front/back disambiguation verification)
    all_gt_azimuths = []
    all_pred_azimuths = []

    for batch in tqdm(val_loader, desc='Val', leave=False, disable=not IS_TTY):
        corrupted = batch['corrupted'].to(device)
        ground_truth = batch['ground_truth'].to(device)
        visibility = batch['visibility'].to(device)
        pose_2d = batch['pose_2d'].to(device)
        azimuth = batch['azimuth'].to(device)
        elevation = batch['elevation'].to(device)

        # Get projected 2D (GT 3D projected to 2D - CRITICAL for POF foreshortening)
        projected_2d = None
        if 'projected_2d' in batch:
            projected_2d = batch['projected_2d'].to(device)

        # Get GT limb orientations if available
        gt_limb_orientations = None
        if use_limb_orientations and 'gt_limb_orientations' in batch:
            gt_limb_orientations = batch['gt_limb_orientations'].to(device)

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
            projected_2d_flat = projected_2d.view(-1, 17, 2) if projected_2d is not None else None

            gt_torso_lengths = compute_torso_lengths(ground_truth_flat)

            output = model(corrupted_flat, visibility_flat, pose_2d=pose_2d_flat,
                           azimuth=azimuth_flat, elevation=elevation_flat,
                           use_predicted_angles=use_predicted_angles,
                           projected_2d=projected_2d_flat)
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
                           use_predicted_angles=use_predicted_angles,
                           projected_2d=projected_2d)
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

            # Limb orientation loss (POF-inspired)
            if use_limb_orientations and gt_limb_orientations is not None and 'pred_limb_orientations' in output:
                limb_loss = limb_orientation_loss(output['pred_limb_orientations'], gt_limb_orientations, visibility)
                losses['limb_orient'] = limb_loss
                losses['total'] = losses['total'] + limb_orientation_weight * limb_loss
                # Track angular error in degrees (per-limb and mean)
                per_limb_err = limb_orientation_angle_error(output['pred_limb_orientations'], gt_limb_orientations)  # (batch, 14)
                total_limb_angle_error += per_limb_err.mean().item() * corrupted.size(0)
                # Track per-limb errors for diagnostics
                limb_errors += per_limb_err.sum(dim=0).cpu()
                limb_counts += corrupted.size(0)

                # Solved depth loss (MTC-style)
                if use_least_squares and projection_loss_weight > 0:
                    solved_pose, scale_factors = solve_depth_least_squares(
                        pose_2d, output['pred_limb_orientations'], corrupted
                    )
                    depth_loss = solved_depth_loss(solved_pose, ground_truth, visibility)
                    scale_reg = scale_factor_regularization(scale_factors)
                    losses['solved_depth'] = depth_loss
                    losses['scale_reg'] = scale_reg
                    losses['total'] = losses['total'] + projection_loss_weight * depth_loss + 0.1 * scale_reg

            depth_error = (corrected[:, :, 2] - ground_truth[:, :, 2]).abs().mean()
            total_depth_error += depth_error.item() * corrupted.size(0)
            az_diff = torch.abs(output['pred_azimuth'] - azimuth)
            az_diff = torch.min(az_diff, 360.0 - az_diff)
            total_azimuth_error += az_diff.sum().item()
            el_diff = torch.abs(output['pred_elevation'] - elevation)
            total_elevation_error += el_diff.sum().item()
            total_samples += corrupted.size(0)

            # Track azimuth for front/back analysis
            all_gt_azimuths.extend(azimuth.cpu().tolist())
            all_pred_azimuths.extend(output['pred_azimuth'].cpu().tolist())

        total_loss += losses['total'].item()
        for key in loss_components:
            if key in losses:
                loss_components[key] += losses[key].item()
        num_batches += 1

    metrics = {
        'total': total_loss / num_batches,
        **{k: v / num_batches for k, v in loss_components.items()},
        'depth_error_m': total_depth_error / total_samples,
        'azimuth_error_deg': total_azimuth_error / total_samples,
        'elevation_error_deg': total_elevation_error / total_samples,
    }

    # Add limb orientation error if computed
    if use_limb_orientations and total_limb_angle_error > 0:
        metrics['limb_angle_error_deg'] = total_limb_angle_error / total_samples
        # Per-limb breakdown
        if limb_counts > 0:
            metrics['per_limb_errors'] = (limb_errors / limb_counts).tolist()

    # Azimuth distribution analysis (front vs back)
    if all_gt_azimuths:
        import numpy as np
        gt_az = np.array(all_gt_azimuths)
        pred_az = np.array(all_pred_azimuths)
        # Front: 315-45°, Back: 135-225°
        front_mask = (gt_az >= 315) | (gt_az < 45)
        back_mask = (gt_az >= 135) & (gt_az < 225)
        metrics['front_count'] = int(front_mask.sum())
        metrics['back_count'] = int(back_mask.sum())
        # Predicted azimuth range
        metrics['pred_az_min'] = float(pred_az.min())
        metrics['pred_az_max'] = float(pred_az.max())
        metrics['pred_az_std'] = float(pred_az.std())

    return metrics


def main():
    parser = argparse.ArgumentParser(description='Train depth refinement model')
    parser.add_argument('--data', type=str, default='data/training/aistpp_converted',
                        help='Path to training data')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--workers', type=int, default=4, help='Dataloader workers')
    parser.add_argument('--fp16', action='store_true', help='Use FP16 mixed precision (may cause NaN with limb orientations)')
    parser.add_argument('--bf16', action='store_true', help='Use BF16 mixed precision (recommended for RTX 30xx+, more stable)')
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
    parser.add_argument('--use-limb-orientations', action='store_true',
                        help='Use POF-inspired per-limb orientation prediction (replaces global camera angles)')
    parser.add_argument('--limb-orientation-weight', type=float, default=0.5,
                        help='Weight for limb orientation loss (default: 0.5)')
    # MTC-style least-squares solver and projection consistency
    parser.add_argument('--use-least-squares', action='store_true',
                        help='Use MTC-style least-squares depth solver (requires --use-limb-orientations)')
    parser.add_argument('--projection-loss-weight', type=float, default=0.3,
                        help='Weight for projection consistency loss (default: 0.3)')
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
    # Optimizer selection
    parser.add_argument('--optimizer', type=str, default='adamw',
                        choices=['adamw', 'lion', 'ademamix', 'sophia', 'schedule_free', 'soap'],
                        help='Optimizer to use (default: adamw). '
                             'lion: 50%% less memory, good for large batches. '
                             'ademamix: dual EMA, better for long training. '
                             'sophia: second-order, good for transformers. '
                             'schedule_free: no LR schedule needed. '
                             'soap: stabilized Shampoo+Adam.')
    parser.add_argument('--weight-decay', type=float, default=0.01, help='Weight decay (default: 0.01)')
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
        use_limb_orientations=args.use_limb_orientations,
    ).to(device)

    if args.use_limb_orientations:
        print("\nUsing POF-inspired limb orientation prediction")
        if args.use_least_squares:
            print("  + MTC-style least-squares depth solver")
            print(f"  + Projection consistency loss weight: {args.projection_loss_weight}")

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
    print(f"\nOptimizer: {args.optimizer}")
    if args.optimizer != 'adamw':
        available = list(ADVANCED_OPTIMIZERS_AVAILABLE.keys())
        print(f"  Available advanced optimizers: {available if available else 'None (install lion-pytorch or pytorch-optimizer)'}")
    optimizer, needs_scheduler = create_optimizer(model, args.optimizer, args.lr, args.weight_decay)

    # LR Scheduler - only if optimizer needs one
    scheduler = None
    if needs_scheduler:
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
        print(f"  Using CosineAnnealingLR scheduler")
    else:
        print(f"  No scheduler needed (schedule-free optimizer)")

    # Mixed precision setup
    # BF16 is recommended for RTX 30xx+ (Ampere+) - same dynamic range as FP32, no overflow issues
    # FP16 can cause NaN with limb orientation loss due to gradient overflow
    use_amp = (args.fp16 or args.bf16) and device == 'cuda'
    amp_dtype = torch.bfloat16 if args.bf16 else torch.float16
    scaler = GradScaler() if args.fp16 and device == 'cuda' else None  # BF16 doesn't need scaler
    if args.bf16 and device == 'cuda':
        print("Using mixed precision training (BF16 - recommended)")
    elif scaler:
        print("Using mixed precision training (FP16 - may cause NaN)")

    # Resume from checkpoint
    start_epoch = 0
    best_val_loss = float('inf')

    if args.checkpoint:
        print(f"\nLoading checkpoint: {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        if scheduler is not None and 'scheduler' in checkpoint:
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
            use_limb_orientations=args.use_limb_orientations,
            limb_orientation_weight=args.limb_orientation_weight,
            use_least_squares=args.use_least_squares,
            projection_loss_weight=args.projection_loss_weight,
            use_amp=use_amp,
            amp_dtype=amp_dtype,
        )

        # Validate (always use predicted angles, no noise)
        val_losses = validate(
            model, val_loader, loss_fn, device,
            bone_locking=args.bone_locking,
            use_predicted_angles=True,  # Always use predicted for validation (like inference)
            use_limb_orientations=args.use_limb_orientations,
            limb_orientation_weight=args.limb_orientation_weight,
            use_least_squares=args.use_least_squares,
            projection_loss_weight=args.projection_loss_weight,
        )

        # Update LR (if using scheduler)
        if scheduler is not None:
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]
        else:
            current_lr = args.lr  # Schedule-free uses constant LR

        epoch_time = time.time() - epoch_start
        is_best = val_losses['total'] < best_val_loss

        # Print progress - compact mode for logs, verbose for terminal
        if IS_TTY:
            # Verbose output for interactive terminal
            print(f"\nEpoch {epoch + 1}/{args.epochs} ({epoch_time:.1f}s) | LR: {current_lr:.2e}")
            train_loss_str = (f"  Train: loss={train_losses['total']:.4f} "
                              f"(pose={train_losses['pose']:.4f}, bone={train_losses['bone']:.4f}, bvar={train_losses['bone_var']:.4f}, cam={train_losses['camera']:.4f}")
            if args.use_limb_orientations:
                train_loss_str += f", limb={train_losses.get('limb_orient', 0):.4f}"
            if args.use_least_squares:
                train_loss_str += f", sdepth={train_losses.get('solved_depth', 0):.4f}"
            train_loss_str += ")"
            print(train_loss_str)

            val_loss_str = (f"  Val:   loss={val_losses['total']:.4f} "
                            f"(pose={val_losses['pose']:.4f}, bone={val_losses['bone']:.4f}, bvar={val_losses['bone_var']:.4f}, cam={val_losses['camera']:.4f}")
            if args.use_limb_orientations:
                val_loss_str += f", limb={val_losses.get('limb_orient', 0):.4f}"
            if args.use_least_squares:
                val_loss_str += f", sdepth={val_losses.get('solved_depth', 0):.4f}"
            val_loss_str += ")"
            print(val_loss_str)

            print(f"  Val depth error: {val_losses['depth_error_m']*100:.2f} cm")
            print(f"  Val angle error: azimuth={val_losses['azimuth_error_deg']:.1f}°, elevation={val_losses['elevation_error_deg']:.1f}°")
            if args.use_limb_orientations and 'limb_angle_error_deg' in val_losses:
                print(f"  Val limb orientation error: {val_losses['limb_angle_error_deg']:.1f}°")
                if 'per_limb_errors' in val_losses:
                    errs = val_losses['per_limb_errors']
                    arms_mean = (errs[0] + errs[1] + errs[2] + errs[3]) / 4
                    legs_mean = (errs[4] + errs[5] + errs[6] + errs[7]) / 4
                    torso_mean = (errs[8] + errs[9] + errs[10] + errs[11]) / 4
                    print(f"    Arms: {arms_mean:.1f}° | Legs: {legs_mean:.1f}° | Torso: {torso_mean:.1f}°")
                if 'pred_az_std' in val_losses:
                    print(f"    Azimuth range: {val_losses['pred_az_min']:.0f}°-{val_losses['pred_az_max']:.0f}° (std={val_losses['pred_az_std']:.0f}°)")
        else:
            # Compact single-line output for log files
            depth_cm = val_losses['depth_error_m'] * 100
            az_err = val_losses['azimuth_error_deg']
            el_err = val_losses['elevation_error_deg']
            best_marker = " *BEST*" if is_best else ""
            limb_str = ""
            if args.use_limb_orientations and 'limb_angle_error_deg' in val_losses:
                limb_str = f" limb={val_losses['limb_angle_error_deg']:.1f}°"
            print(f"E{epoch+1:03d} | trn={train_losses['total']:.4f} val={val_losses['total']:.4f} | "
                  f"depth={depth_cm:.1f}cm az={az_err:.1f}° el={el_err:.1f}°{limb_str} | "
                  f"lr={current_lr:.1e} {epoch_time:.0f}s{best_marker}")

        # Save best model
        if val_losses['total'] < best_val_loss:
            best_val_loss = val_losses['total']
            checkpoint_path = save_dir / 'best_depth_model.pth'
            checkpoint_data = {
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_val_loss': best_val_loss,
                'config': {
                    'num_joints': 17,
                    'd_model': args.d_model,
                    'num_heads': args.num_heads,
                    'num_layers': args.num_layers,
                    'use_elepose': args.elepose,
                    'elepose_hidden_dim': args.elepose_hidden_dim,
                    'use_limb_orientations': args.use_limb_orientations,
                    'optimizer_name': args.optimizer,
                },
            }
            if scheduler is not None:
                checkpoint_data['scheduler'] = scheduler.state_dict()
            torch.save(checkpoint_data, checkpoint_path)
            if IS_TTY:
                print(f"  -> Saved best model (val_loss={best_val_loss:.4f})")

        # Save periodic checkpoint
        if (epoch + 1) % 10 == 0:
            checkpoint_path = save_dir / f'checkpoint_epoch{epoch + 1}.pth'
            checkpoint_data = {
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_val_loss': best_val_loss,
                'config': {
                    'num_joints': 17,
                    'd_model': args.d_model,
                    'num_heads': args.num_heads,
                    'num_layers': args.num_layers,
                    'use_elepose': args.elepose,
                    'elepose_hidden_dim': args.elepose_hidden_dim,
                    'use_limb_orientations': args.use_limb_orientations,
                    'optimizer_name': args.optimizer,
                },
            }
            if scheduler is not None:
                checkpoint_data['scheduler'] = scheduler.state_dict()
            torch.save(checkpoint_data, checkpoint_path)
            if IS_TTY:
                print(f"  -> Saved checkpoint: {checkpoint_path.name}")
            else:
                print(f"       [checkpoint saved: {checkpoint_path.name}]")

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Model saved to: {save_dir / 'best_depth_model.pth'}")
    print("=" * 60)


if __name__ == '__main__':
    main()

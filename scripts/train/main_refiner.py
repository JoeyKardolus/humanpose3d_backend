#!/usr/bin/env python3
"""
Training script for MainRefiner model.

The MainRefiner learns to optimally combine outputs from:
1. POF Model (CameraPOFModel) - provides 3D poses via Part Orientation Fields
2. Joint Constraint Refinement Model (JointConstraintRefiner)

Training modes:
1. Frozen constraints (default): Only train fusion network
2. End-to-end: Fine-tune all models together

Usage:
    # Frozen training (recommended first)
    uv run --group neural python scripts/train/main_refiner.py \
        --pof-checkpoint models/checkpoints/best_pof_model.pth \
        --joint-checkpoint models/checkpoints/best_joint_model.pth \
        --freeze-constraints \
        --epochs 50 --batch-size 256 --bf16

    # End-to-end fine-tuning (after frozen training)
    uv run --group neural python scripts/train/main_refiner.py \
        --checkpoint models/checkpoints/best_main_refiner.pth \
        --pof-checkpoint models/checkpoints/best_pof_model.pth \
        --joint-checkpoint models/checkpoints/best_joint_model.pth \
        --unfreeze-constraints --constraint-lr 1e-5 \
        --epochs 20 --batch-size 128
"""

import argparse
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.amp import autocast
from torch.cuda.amp import GradScaler
from pathlib import Path
from tqdm import tqdm
import json
from datetime import datetime
from typing import Dict, Optional
import numpy as np

# Add project root to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.main_refinement.model import MainRefiner, create_model
from src.main_refinement.losses import MainRefinerLoss
from src.main_refinement.dataset import MainRefinerDataset, create_dataloaders
from src.pof.model import CameraPOFModel, load_pof_model
from src.pof.dataset import normalize_pose_2d, compute_limb_features_2d
from src.pof.reconstruction import reconstruct_skeleton_least_squares
from src.joint_refinement.model import JointConstraintRefiner


def load_pof_model_frozen(
    checkpoint_path: Path,
    device: str,
    freeze: bool = True,
) -> CameraPOFModel:
    """Load pretrained POF model."""
    model = load_pof_model(str(checkpoint_path), device=device, verbose=True)

    if freeze:
        model.eval()
        for param in model.parameters():
            param.requires_grad = False
        print(f"Loaded frozen POF model: {sum(p.numel() for p in model.parameters()):,} params")
    else:
        model.train()
        print(f"Loaded trainable POF model: {sum(p.numel() for p in model.parameters()):,} params")

    return model


def load_joint_model(
    checkpoint_path: Path,
    device: str,
    freeze: bool = True,
) -> JointConstraintRefiner:
    """Load pretrained joint model."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint.get('config', {})

    model = JointConstraintRefiner(
        d_model=config.get('d_model', 128),
        n_heads=config.get('n_heads', 4),
        n_layers=config.get('n_layers', 4),
        dropout=0.0 if freeze else config.get('dropout', 0.1),
    )

    # Handle different checkpoint formats
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    elif 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
    else:
        raise KeyError(f"Checkpoint has no 'model_state_dict' or 'model' key. Keys: {list(checkpoint.keys())}")
    model.to(device)

    if freeze:
        model.eval()
        for param in model.parameters():
            param.requires_grad = False
        print(f"Loaded frozen joint model: {sum(p.numel() for p in model.parameters()):,} params")
    else:
        model.train()
        print(f"Loaded trainable joint model: {sum(p.numel() for p in model.parameters()):,} params")

    return model


def run_pof_model(
    pof_model: CameraPOFModel,
    pose_2d: torch.Tensor,
    visibility: torch.Tensor,
    device: str,
    freeze: bool = True,
) -> Dict[str, torch.Tensor]:
    """Run POF model to get limb orientations and reconstructed pose."""
    batch_size = pose_2d.size(0)

    # Process each sample in batch
    pof_vectors_list = []
    reconstructed_list = []

    pose_2d_np = pose_2d.cpu().numpy()
    visibility_np = visibility.cpu().numpy()

    for i in range(batch_size):
        # Normalize 2D and compute limb features
        pose_2d_norm, _, _ = normalize_pose_2d(pose_2d_np[i])
        limb_delta_2d, limb_length_2d = compute_limb_features_2d(pose_2d_norm)

        # Convert to tensors
        pose_2d_t = torch.from_numpy(pose_2d_norm.astype(np.float32)).unsqueeze(0).to(device)
        vis_t = torch.from_numpy(visibility_np[i].astype(np.float32)).unsqueeze(0).to(device)
        limb_delta_t = torch.from_numpy(limb_delta_2d.astype(np.float32)).unsqueeze(0).to(device)
        limb_length_t = torch.from_numpy(limb_length_2d.astype(np.float32)).unsqueeze(0).to(device)

        # Run POF model
        with torch.set_grad_enabled(not freeze):
            pof_vec = pof_model(pose_2d_t, vis_t, limb_delta_t, limb_length_t)

        pof_vectors_list.append(pof_vec)

        # Reconstruct 3D pose from POF
        pof_np = pof_vec.detach().cpu().numpy()
        reconstructed = reconstruct_skeleton_least_squares(
            pof_np, pose_2d_np[i:i+1], None,
            pelvis_depth=0.0, denormalize=False
        )
        reconstructed_list.append(torch.from_numpy(reconstructed.astype(np.float32)))

    pof_vectors = torch.cat(pof_vectors_list, dim=0)
    reconstructed_pose = torch.cat(reconstructed_list, dim=0).to(device)

    return {
        'pof_vectors': pof_vectors,
        'reconstructed_pose': reconstructed_pose,
        'confidence': torch.ones(batch_size, 17, device=device),
    }


def train_epoch(
    main_model: MainRefiner,
    pof_model: CameraPOFModel,
    joint_model: JointConstraintRefiner,
    train_loader,
    optimizer,
    loss_fn: MainRefinerLoss,
    device: str,
    scaler: Optional[GradScaler] = None,
    use_amp: bool = False,
    amp_dtype: torch.dtype = torch.float16,
    freeze_constraints: bool = True,
) -> Dict[str, float]:
    """Train for one epoch."""
    main_model.train()

    if not freeze_constraints:
        pof_model.train()
        joint_model.train()

    total_loss = 0.0
    loss_components = {
        'pose': 0.0,
        'bone': 0.0,
        'gate': 0.0,
        'improvement': 0.0,
        'confidence': 0.0,
    }
    num_batches = 0
    skipped_batches = 0

    pbar = tqdm(train_loader, desc='Train')

    for batch in pbar:
        # Move to device
        raw_pose = batch['raw_pose'].to(device)
        ground_truth = batch['ground_truth'].to(device)
        visibility = batch['visibility'].to(device)
        pose_2d = batch['pose_2d'].to(device)

        # Input validation
        if (torch.isnan(raw_pose).any() or torch.isinf(raw_pose).any() or
            torch.isnan(ground_truth).any() or torch.isinf(ground_truth).any()):
            print(f"\nWarning: NaN/Inf in input data, skipping batch")
            skipped_batches += 1
            continue

        optimizer.zero_grad()

        with autocast(device_type='cuda', dtype=amp_dtype, enabled=use_amp):
            # Run POF model
            if 'pof_outputs' in batch:
                # Pre-computed outputs
                pof_outputs = {k: v.to(device) for k, v in batch['pof_outputs'].items()}
            else:
                # Run model online
                pof_outputs = run_pof_model(pof_model, pose_2d, visibility, device, freeze_constraints)

            # Run joint model
            if 'joint_outputs' in batch:
                # Pre-computed outputs
                joint_outputs = {k: v.to(device) for k, v in batch['joint_outputs'].items()}
            else:
                # Need to compute joint angles first (simplified: use zeros)
                joint_outputs = {
                    'refined_angles': torch.zeros(raw_pose.size(0), 12, 3, device=device),
                    'delta_angles': torch.zeros(raw_pose.size(0), 12, 3, device=device),
                }

            # Run main refiner
            output = main_model(raw_pose, visibility, pof_outputs, joint_outputs)

            # Compute losses
            losses = loss_fn(output, raw_pose, ground_truth, pof_outputs, visibility)

        # Check for NaN
        if torch.isnan(losses['total']) or torch.isinf(losses['total']):
            print(f"\nWarning: NaN/Inf loss detected, skipping batch")
            optimizer.zero_grad()
            skipped_batches += 1
            continue

        # Backward pass
        if scaler is not None:
            scaler.scale(losses['total']).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(main_model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            losses['total'].backward()
            torch.nn.utils.clip_grad_norm_(main_model.parameters(), max_norm=1.0)
            optimizer.step()

        total_loss += losses['total'].item()
        for key in loss_components:
            if key in losses:
                loss_components[key] += losses[key].item()
        num_batches += 1

        pbar.set_postfix({
            'loss': f"{losses['total'].item():.4f}",
            'pose': f"{losses['pose'].item():.4f}",
            'gate': f"{losses['gate'].item():.4f}",
        })

    if num_batches == 0:
        print(f"\nWARNING: All batches were skipped!")
        return {'total': float('nan'), **{k: float('nan') for k in loss_components}}

    return {
        'total': total_loss / num_batches,
        **{k: v / num_batches for k, v in loss_components.items()},
    }


@torch.no_grad()
def validate(
    main_model: MainRefiner,
    pof_model: CameraPOFModel,
    joint_model: JointConstraintRefiner,
    val_loader,
    loss_fn: MainRefinerLoss,
    device: str,
    use_amp: bool = False,
    amp_dtype: torch.dtype = torch.float16,
) -> Dict[str, float]:
    """Validate model."""
    main_model.eval()
    pof_model.eval()
    joint_model.eval()

    total_loss = 0.0
    loss_components = {
        'pose': 0.0,
        'bone': 0.0,
        'gate': 0.0,
        'improvement': 0.0,
        'confidence': 0.0,
    }
    num_batches = 0

    # Also track improvement metrics
    pof_only_error = 0.0
    fusion_error = 0.0

    for batch in tqdm(val_loader, desc='Val'):
        raw_pose = batch['raw_pose'].to(device)
        ground_truth = batch['ground_truth'].to(device)
        visibility = batch['visibility'].to(device)
        pose_2d = batch['pose_2d'].to(device)

        with autocast(device_type='cuda', dtype=amp_dtype, enabled=use_amp):
            # Run POF model
            if 'pof_outputs' in batch:
                pof_outputs = {k: v.to(device) for k, v in batch['pof_outputs'].items()}
            else:
                pof_outputs = run_pof_model(pof_model, pose_2d, visibility, device, freeze=True)

            # Run joint model
            if 'joint_outputs' in batch:
                joint_outputs = {k: v.to(device) for k, v in batch['joint_outputs'].items()}
            else:
                joint_outputs = {
                    'refined_angles': torch.zeros(raw_pose.size(0), 12, 3, device=device),
                    'delta_angles': torch.zeros(raw_pose.size(0), 12, 3, device=device),
                }

            # Run main refiner
            output = main_model(raw_pose, visibility, pof_outputs, joint_outputs)

            # Compute losses
            losses = loss_fn(output, raw_pose, ground_truth, pof_outputs, visibility)

        if not torch.isnan(losses['total']):
            total_loss += losses['total'].item()
            for key in loss_components:
                if key in losses:
                    loss_components[key] += losses[key].item()
            num_batches += 1

            # Track improvement: POF-only vs fusion
            pof_reconstructed = pof_outputs['reconstructed_pose']
            pof_only_error += (pof_reconstructed - ground_truth).abs().mean().item()
            fusion_error += (output['refined_pose'] - ground_truth).abs().mean().item()

    if num_batches == 0:
        return {'total': float('nan'), **{k: float('nan') for k in loss_components}}

    results = {
        'total': total_loss / num_batches,
        **{k: v / num_batches for k, v in loss_components.items()},
        'pof_only_error': pof_only_error / num_batches,
        'fusion_error': fusion_error / num_batches,
    }

    # Improvement ratio (lower is better)
    if results['pof_only_error'] > 0:
        results['improvement_ratio'] = results['fusion_error'] / results['pof_only_error']
    else:
        results['improvement_ratio'] = 1.0

    return results


def main():
    parser = argparse.ArgumentParser(description='Train MainRefiner model')

    # Data
    parser.add_argument('--data', type=str, default='data/training/aistpp_converted',
                        help='Path to training data (comma-separated for multiple)')
    parser.add_argument('--val-ratio', type=float, default=0.1,
                        help='Validation split ratio')

    # Model checkpoints
    parser.add_argument('--pof-checkpoint', type=str, required=True,
                        help='Path to pretrained POF model')
    parser.add_argument('--joint-checkpoint', type=str, required=True,
                        help='Path to pretrained joint model')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Resume from main refiner checkpoint')

    # Model architecture
    parser.add_argument('--d-model', type=int, default=128,
                        help='Model hidden dimension')
    parser.add_argument('--num-heads', type=int, default=4,
                        help='Number of attention heads')
    parser.add_argument('--num-layers', type=int, default=2,
                        help='Number of transformer layers')

    # Training
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--warmup-epochs', type=int, default=2)

    # Constraint model training
    parser.add_argument('--freeze-constraints', action='store_true',
                        help='Freeze POF and joint models (default)')
    parser.add_argument('--unfreeze-constraints', action='store_true',
                        help='Fine-tune constraint models')
    parser.add_argument('--constraint-lr', type=float, default=1e-5,
                        help='Learning rate for constraint models (if unfrozen)')

    # Loss weights
    parser.add_argument('--pose-weight', type=float, default=1.0)
    parser.add_argument('--bone-weight', type=float, default=0.5)
    parser.add_argument('--gate-supervision-weight', type=float, default=0.3)
    parser.add_argument('--improvement-weight', type=float, default=0.2)
    parser.add_argument('--confidence-weight', type=float, default=0.1)

    # Mixed precision
    parser.add_argument('--fp16', action='store_true', help='Use FP16 mixed precision')
    parser.add_argument('--bf16', action='store_true', help='Use BF16 mixed precision')

    # Output
    parser.add_argument('--output-dir', type=str, default='models/checkpoints')
    parser.add_argument('--max-samples', type=int, default=None,
                        help='Limit training samples (for debugging)')

    args = parser.parse_args()

    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name()}")

    # Determine freeze mode
    freeze_constraints = args.freeze_constraints or not args.unfreeze_constraints

    # Load constraint models
    pof_model = load_pof_model_frozen(
        Path(args.pof_checkpoint),
        device,
        freeze=freeze_constraints,
    )
    joint_model = load_joint_model(
        Path(args.joint_checkpoint),
        device,
        freeze=freeze_constraints,
    )

    # Create main model
    main_model = create_model(
        d_model=args.d_model,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
    )
    main_model.to(device)

    # Resume from checkpoint if provided
    start_epoch = 0
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
        main_model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint.get('epoch', 0)
        print(f"Resumed from epoch {start_epoch}")

    # Create dataloaders
    print(f"\nLoading data from {args.data}")
    train_loader, val_loader = create_dataloaders(
        args.data,
        batch_size=args.batch_size,
        num_workers=args.workers,
        val_ratio=args.val_ratio,
        max_samples=args.max_samples,
        pof_model=pof_model if not freeze_constraints else None,
        joint_model=joint_model if not freeze_constraints else None,
        device=device,
    )
    print(f"Train: {len(train_loader)} batches, Val: {len(val_loader)} batches")

    # Optimizer - separate param groups for main vs constraint models
    if freeze_constraints:
        optimizer = AdamW(main_model.parameters(), lr=args.lr, weight_decay=0.01)
    else:
        optimizer = AdamW([
            {'params': main_model.parameters(), 'lr': args.lr},
            {'params': pof_model.parameters(), 'lr': args.constraint_lr},
            {'params': joint_model.parameters(), 'lr': args.constraint_lr},
        ], weight_decay=0.01)

    # Scheduler
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # Loss function
    loss_fn = MainRefinerLoss(
        pose_weight=args.pose_weight,
        bone_weight=args.bone_weight,
        gate_supervision_weight=args.gate_supervision_weight,
        improvement_weight=args.improvement_weight,
        confidence_weight=args.confidence_weight,
    )

    # Mixed precision
    use_amp = args.fp16 or args.bf16
    if args.bf16:
        amp_dtype = torch.bfloat16
        scaler = None  # BF16 doesn't need scaler
        print("Using BF16 mixed precision")
    elif args.fp16:
        amp_dtype = torch.float16
        scaler = GradScaler()
        print("Using FP16 mixed precision")
    else:
        amp_dtype = torch.float32
        scaler = None

    # Output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Training loop
    best_val_loss = float('inf')
    print(f"\n{'='*60}")
    print("TRAINING")
    print(f"{'='*60}")

    for epoch in range(start_epoch, args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")

        # Train
        train_losses = train_epoch(
            main_model, pof_model, joint_model,
            train_loader, optimizer, loss_fn, device,
            scaler=scaler, use_amp=use_amp, amp_dtype=amp_dtype,
            freeze_constraints=freeze_constraints,
        )

        # Validate
        val_losses = validate(
            main_model, pof_model, joint_model,
            val_loader, loss_fn, device,
            use_amp=use_amp, amp_dtype=amp_dtype,
        )

        # Update scheduler
        scheduler.step()

        # Log
        print(f"  Train Loss: {train_losses['total']:.4f}")
        print(f"  Val Loss: {val_losses['total']:.4f}")
        print(f"  Val Pose Loss: {val_losses['pose']:.4f}")
        if 'improvement_ratio' in val_losses:
            print(f"  Improvement Ratio: {val_losses['improvement_ratio']:.3f} "
                  f"({'better' if val_losses['improvement_ratio'] < 1 else 'worse'} than POF-only)")

        # Save best model
        if val_losses['total'] < best_val_loss and not torch.isnan(torch.tensor(val_losses['total'])):
            best_val_loss = val_losses['total']
            checkpoint_path = output_dir / 'best_main_refiner.pth'
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': main_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_losses['total'],
                'config': {
                    'd_model': args.d_model,
                    'num_heads': args.num_heads,
                    'num_layers': args.num_layers,
                },
            }, checkpoint_path)
            print(f"  Saved best model to {checkpoint_path}")

        # Save periodic checkpoint
        if (epoch + 1) % 10 == 0:
            checkpoint_path = output_dir / f'main_refiner_epoch{epoch+1}.pth'
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': main_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_losses['total'],
                'config': {
                    'd_model': args.d_model,
                    'num_heads': args.num_heads,
                    'num_layers': args.num_layers,
                },
            }, checkpoint_path)
            print(f"  Saved checkpoint to {checkpoint_path}")

    print(f"\n{'='*60}")
    print("TRAINING COMPLETE")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()

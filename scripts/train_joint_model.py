#!/usr/bin/env python3
"""
Train joint constraint refinement model on AIST++ data.

Uses:
- Joint angles computed by validated ISB kinematics
- Cross-joint attention for pose-aware correction
- Soft constraints learned from data distribution

Usage:
    uv run --group neural python scripts/train_joint_model.py

Options:
    --data        Path to training data (default: data/training/aistpp_joint_angles)
    --epochs      Number of epochs (default: 100)
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

from src.joint_refinement.model import create_model
from src.joint_refinement.losses import JointRefinementLoss
from src.joint_refinement.dataset import create_dataloaders


def train_epoch(
    model: nn.Module,
    train_loader,
    optimizer,
    loss_fn: JointRefinementLoss,
    scaler: GradScaler = None,
    device: str = 'cuda',
) -> dict:
    """Train for one epoch."""
    model.train()

    total_loss = 0.0
    loss_components = {'reconstruction': 0.0, 'symmetry': 0.0, 'delta_reg': 0.0}
    num_batches = 0

    pbar = tqdm(train_loader, desc='Train', leave=False)

    for batch in pbar:
        corrupted = batch['corrupted_angles'].to(device)
        ground_truth = batch['ground_truth_angles'].to(device)
        visibility = batch['joint_visibility'].to(device)

        optimizer.zero_grad()

        if scaler is not None:
            with autocast():
                refined, delta = model(corrupted, visibility)
                losses = loss_fn(refined, ground_truth, delta, visibility)
                loss = losses['total']

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            refined, delta = model(corrupted, visibility)
            losses = loss_fn(refined, ground_truth, delta, visibility)
            loss = losses['total']

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        # Track losses
        total_loss += loss.item()
        for key in loss_components:
            if key in losses:
                loss_components[key] += losses[key].item()
        num_batches += 1

        # Update progress bar
        pbar.set_postfix({'loss': loss.item()})

    # Average losses
    avg_loss = total_loss / num_batches
    for key in loss_components:
        loss_components[key] /= num_batches

    return {
        'total_loss': avg_loss,
        **loss_components,
    }


def validate(
    model: nn.Module,
    val_loader,
    loss_fn: JointRefinementLoss,
    device: str = 'cuda',
) -> dict:
    """Validate the model."""
    model.eval()

    total_loss = 0.0
    total_angle_error = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in tqdm(val_loader, desc='Val', leave=False):
            corrupted = batch['corrupted_angles'].to(device)
            ground_truth = batch['ground_truth_angles'].to(device)
            visibility = batch['joint_visibility'].to(device)

            refined, delta = model(corrupted, visibility)
            losses = loss_fn(refined, ground_truth, delta, visibility)

            total_loss += losses['total'].item()

            # Compute mean absolute angle error
            angle_error = (refined - ground_truth).abs().mean()
            total_angle_error += angle_error.item()

            num_batches += 1

    return {
        'val_loss': total_loss / num_batches,
        'mean_angle_error': total_angle_error / num_batches,
    }


def main():
    parser = argparse.ArgumentParser(description='Train joint constraint refinement model')
    parser.add_argument('--data', type=Path,
                        default=Path('data/training/aistpp_joint_angles'),
                        help='Path to training data')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--workers', type=int, default=4,
                        help='Dataloader workers')
    parser.add_argument('--fp16', action='store_true',
                        help='Use mixed precision training')
    parser.add_argument('--checkpoint', type=Path, default=None,
                        help='Resume from checkpoint')
    parser.add_argument('--d-model', type=int, default=128,
                        help='Model hidden dimension')
    parser.add_argument('--n-layers', type=int, default=4,
                        help='Number of transformer layers')
    parser.add_argument('--max-samples', type=int, default=None,
                        help='Limit training samples (for debugging)')
    args = parser.parse_args()

    print("=" * 60)
    print("Joint Constraint Refinement Training")
    print("=" * 60)
    print(f"Data: {args.data}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"FP16: {args.fp16}")
    print()

    # Check data directory
    if not args.data.exists():
        print(f"Error: Data directory not found: {args.data}")
        print("Run: uv run python scripts/generate_joint_angle_training.py")
        sys.exit(1)

    # Device setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Create dataloaders
    print("\nLoading data...")
    train_loader, val_loader = create_dataloaders(
        args.data,
        batch_size=args.batch_size,
        num_workers=args.workers,
        max_samples=args.max_samples,
    )
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # Create model
    print("\nCreating model...")
    model = create_model(
        d_model=args.d_model,
        n_layers=args.n_layers,
    )
    model = model.to(device)

    # Load checkpoint if provided
    start_epoch = 0
    if args.checkpoint:
        print(f"Loading checkpoint: {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint.get('epoch', 0)

    # Loss function
    # Reduced weights: symmetry is often violated in dance, model needs larger corrections
    loss_fn = JointRefinementLoss(
        symmetry_weight=0.01,   # Reduced 10x - asymmetric poses are valid in dance
        delta_weight=0.001,     # Reduced 10x - allow larger corrections when needed
        chain_weight=0.1,       # Kinematic chain consistency - connected joints have correlated errors
    )

    # Optimizer
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # Mixed precision
    scaler = GradScaler() if args.fp16 else None

    # Create checkpoint directory
    checkpoint_dir = Path('models/checkpoints')
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Training loop
    print("\nStarting training...")
    best_val_loss = float('inf')
    best_angle_error = float('inf')

    for epoch in range(start_epoch, args.epochs):
        start_time = time.time()

        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, loss_fn, scaler, device
        )

        # Validate
        val_metrics = validate(model, val_loader, loss_fn, device)

        # Update scheduler
        scheduler.step()

        # Track best model
        is_best = val_metrics['val_loss'] < best_val_loss
        if is_best:
            best_val_loss = val_metrics['val_loss']
            best_angle_error = val_metrics['mean_angle_error']

            # Save best model
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_metrics['val_loss'],
                'mean_angle_error': val_metrics['mean_angle_error'],
            }, checkpoint_dir / 'best_joint_model.pth')

        epoch_time = time.time() - start_time

        # Print progress
        print(f"\nEpoch {epoch + 1}/{args.epochs} ({epoch_time:.1f}s)")
        print(f"  Train: loss={train_metrics['total_loss']:.4f}, "
              f"recon={train_metrics['reconstruction']:.4f}, "
              f"sym={train_metrics['symmetry']:.4f}")
        print(f"  Val:   loss={val_metrics['val_loss']:.4f}, "
              f"angle_err={val_metrics['mean_angle_error']:.2f}° "
              f"{'*BEST*' if is_best else ''}")
        print(f"  LR: {scheduler.get_last_lr()[0]:.2e}")

    # Training complete
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Best mean angle error: {best_angle_error:.2f}°")
    print(f"Model saved to: {checkpoint_dir / 'best_joint_model.pth'}")


if __name__ == '__main__':
    main()

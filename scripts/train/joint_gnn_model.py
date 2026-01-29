#!/usr/bin/env python3
"""Training script for GNN-based joint angle refinement model.

Train a Graph Neural Network to refine joint angles using skeleton structure
as inductive bias. Alternative to the transformer-based model.

Model types:
- gcn: Basic Graph Convolutional Network with combined adjacency
- semgcn: Semantic GCN with separate edge types (kinematic, symmetry, hierarchy)
- semgcn-temporal: Semantic GCN with angle sign classification and temporal context

Usage:
    # Train SemGCN-Temporal (recommended)
    uv run --group neural python scripts/train/joint_gnn_model.py \
        --data data/training/aistpp_joint_angles \
        --model-type semgcn-temporal \
        --d-model 192 --num-layers 4 \
        --sign-weight 0.1 --temporal \
        --epochs 100 --batch-size 256 --workers 8 --bf16

    # Train basic SemGCN
    uv run --group neural python scripts/train/joint_gnn_model.py \
        --data data/training/aistpp_joint_angles \
        --model-type semgcn \
        --d-model 192 --num-layers 4 \
        --epochs 100 --batch-size 256 --workers 8 --bf16

    # Train basic GCN
    uv run --group neural python scripts/train/joint_gnn_model.py \
        --data data/training/aistpp_joint_angles \
        --model-type gcn \
        --epochs 100 --batch-size 256 --workers 8 --bf16
"""

import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast, GradScaler

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from src.joint_refinement.gnn_model import (
    create_gnn_joint_model,
    NUM_JOINTS,
)
from src.joint_refinement.dataset import create_dataloaders
from src.joint_refinement.losses import (
    GNNJointRefinementLoss,
    angular_distance,
    sign_accuracy,
)

# Compact logging mode (disable tqdm progress bars when piped to file)
IS_TTY = sys.stdout.isatty()


def parse_args():
    parser = argparse.ArgumentParser(description="Train GNN-based joint refinement model")

    # Data
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/training/aistpp_joint_angles"),
        help="Path to training data",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Limit samples for debugging",
    )

    # Training
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--val-ratio", type=float, default=0.1)

    # Model architecture
    parser.add_argument(
        "--model-type",
        type=str,
        default="semgcn-temporal",
        choices=["gcn", "semgcn", "semgcn-temporal"],
        help="GNN model type",
    )
    parser.add_argument("--d-model", type=int, default=192)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument(
        "--use-gat",
        action="store_true",
        help="Use Graph Attention instead of GCN (semgcn only)",
    )

    # Temporal mode
    parser.add_argument(
        "--temporal",
        action="store_true",
        help="Enable temporal training with consecutive frame pairs",
    )

    # Loss weights
    parser.add_argument("--symmetry-weight", type=float, default=0.01)
    parser.add_argument("--delta-weight", type=float, default=0.001)
    parser.add_argument("--chain-weight", type=float, default=0.1)
    parser.add_argument(
        "--sign-weight",
        type=float,
        default=0.1,
        help="Weight for angle sign classification loss (temporal model only)",
    )
    parser.add_argument(
        "--temporal-weight",
        type=float,
        default=0.05,
        help="Weight for temporal smoothness loss",
    )

    # Mixed precision
    parser.add_argument("--fp16", action="store_true", help="Use FP16")
    parser.add_argument("--bf16", action="store_true", help="Use BF16")

    # Checkpointing
    parser.add_argument("--checkpoint", type=Path, help="Resume from checkpoint")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("models/checkpoints"),
        help="Output directory for checkpoints",
    )
    parser.add_argument("--save-every", type=int, default=10, help="Save every N epochs")

    # Misc
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")

    return parser.parse_args()


def train_epoch(
    model: nn.Module,
    train_loader,
    optimizer,
    criterion: GNNJointRefinementLoss,
    scaler: GradScaler,
    device: str,
    use_amp: bool,
    amp_dtype: torch.dtype,
    temporal: bool = False,
    model_type: str = "semgcn",
) -> dict:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    total_reconstruction = 0.0
    total_symmetry = 0.0
    total_sign = 0.0
    total_angle_error = 0.0
    total_sign_acc = 0.0
    num_batches = 0

    for batch in train_loader:
        corrupted = batch['corrupted_angles'].to(device)
        ground_truth = batch['ground_truth_angles'].to(device)
        visibility = batch['joint_visibility'].to(device)

        # Temporal mode: get previous angles
        prev_angles = None
        prev_refined = None
        if temporal and 'prev_corrupted_angles' in batch:
            prev_angles = batch['prev_corrupted_angles'].to(device)
            has_prev = batch['has_prev']
            # Mask prev_angles for samples without previous frame
            mask = torch.tensor(has_prev, dtype=torch.float32, device=device)
            prev_angles = prev_angles * mask.unsqueeze(-1).unsqueeze(-1)

        optimizer.zero_grad()

        with autocast("cuda", enabled=use_amp, dtype=amp_dtype):
            if model_type == "semgcn-temporal":
                # Temporal model returns (refined, delta, sign_logits)
                refined, delta, sign_logits = model(corrupted, visibility, prev_angles)
                losses = criterion(
                    refined, ground_truth, delta, visibility,
                    sign_logits=sign_logits,
                    prev_refined_angles=prev_refined,
                )
            else:
                # Non-temporal models return (refined, delta)
                refined, delta = model(corrupted, visibility)
                losses = criterion(refined, ground_truth, delta, visibility)

            loss = losses['total']

        if use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        total_loss += loss.item()
        total_reconstruction += losses.get('reconstruction', torch.tensor(0.0)).item()
        total_symmetry += losses.get('symmetry', torch.tensor(0.0)).item()

        if model_type == "semgcn-temporal" and 'sign' in losses:
            total_sign += losses['sign'].item()
            # Compute sign accuracy
            with torch.no_grad():
                acc = sign_accuracy(sign_logits, ground_truth, visibility)
                total_sign_acc += acc['total_acc']

        # Angular error metric
        with torch.no_grad():
            angle_err = angular_distance(refined, ground_truth).mean().item()
            total_angle_error += angle_err

        num_batches += 1

    result = {
        'loss': total_loss / num_batches,
        'reconstruction': total_reconstruction / num_batches,
        'symmetry': total_symmetry / num_batches,
        'angle_error_deg': total_angle_error / num_batches,
    }
    if model_type == "semgcn-temporal":
        result['sign'] = total_sign / num_batches
        result['sign_acc'] = total_sign_acc / num_batches

    return result


@torch.no_grad()
def validate(
    model: nn.Module,
    val_loader,
    criterion: GNNJointRefinementLoss,
    device: str,
    temporal: bool = False,
    model_type: str = "semgcn",
) -> dict:
    """Validate model."""
    model.eval()
    total_loss = 0.0
    total_reconstruction = 0.0
    total_angle_error = 0.0
    total_sign_acc = 0.0
    num_batches = 0

    for batch in val_loader:
        corrupted = batch['corrupted_angles'].to(device)
        ground_truth = batch['ground_truth_angles'].to(device)
        visibility = batch['joint_visibility'].to(device)

        # Temporal mode
        prev_angles = None
        if temporal and 'prev_corrupted_angles' in batch:
            prev_angles = batch['prev_corrupted_angles'].to(device)
            has_prev = batch['has_prev']
            mask = torch.tensor(has_prev, dtype=torch.float32, device=device)
            prev_angles = prev_angles * mask.unsqueeze(-1).unsqueeze(-1)

        if model_type == "semgcn-temporal":
            refined, delta, sign_logits = model(corrupted, visibility, prev_angles)
            losses = criterion(refined, ground_truth, delta, visibility, sign_logits=sign_logits)
            acc = sign_accuracy(sign_logits, ground_truth, visibility)
            total_sign_acc += acc['total_acc']
        else:
            refined, delta = model(corrupted, visibility)
            losses = criterion(refined, ground_truth, delta, visibility)

        total_loss += losses['total'].item()
        total_reconstruction += losses.get('reconstruction', torch.tensor(0.0)).item()

        angle_err = angular_distance(refined, ground_truth).mean().item()
        total_angle_error += angle_err

        num_batches += 1

    result = {
        'loss': total_loss / num_batches,
        'reconstruction': total_reconstruction / num_batches,
        'angle_error_deg': total_angle_error / num_batches,
    }
    if model_type == "semgcn-temporal":
        result['sign_acc'] = total_sign_acc / num_batches

    return result


def save_checkpoint(
    model: nn.Module,
    optimizer,
    scheduler,
    epoch: int,
    best_val_loss: float,
    args,
    output_path: Path,
) -> None:
    """Save training checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'best_val_loss': best_val_loss,
        'config': model.get_config(),
    }
    torch.save(checkpoint, output_path)


def main():
    args = parse_args()

    # Auto-enable temporal mode for semgcn-temporal
    temporal = args.temporal or args.model_type == "semgcn-temporal"
    if args.model_type == "semgcn-temporal" and not args.temporal:
        print("Note: --temporal auto-enabled for semgcn-temporal model")

    # Setup
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    # Device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    print(f"Using device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Mixed precision
    use_amp = args.fp16 or args.bf16
    if args.bf16:
        amp_dtype = torch.bfloat16
        print("Using BF16 mixed precision")
    elif args.fp16:
        amp_dtype = torch.float16
        print("Using FP16 mixed precision")
    else:
        amp_dtype = torch.float32

    scaler = GradScaler("cuda", enabled=use_amp and amp_dtype == torch.float16)

    # Data
    print(f"\nLoading data from: {args.data}")
    print(f"Temporal mode: {temporal}")

    if not args.data.exists():
        print(f"Error: Data directory not found: {args.data}")
        print("Run: uv run python scripts/data/generate_joint_angles.py")
        sys.exit(1)

    train_loader, val_loader = create_dataloaders(
        args.data,
        batch_size=args.batch_size,
        num_workers=args.workers,
        val_ratio=args.val_ratio,
        max_samples=args.max_samples,
        temporal=temporal,
    )
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # Model
    print(f"\nCreating {args.model_type.upper()} model...")
    model = create_gnn_joint_model(
        model_type=args.model_type,
        d_model=args.d_model,
        num_layers=args.num_layers,
        dropout=args.dropout,
        use_gat=args.use_gat,
    )
    model = model.to(device)

    # Loss
    criterion = GNNJointRefinementLoss(
        symmetry_weight=args.symmetry_weight,
        delta_weight=args.delta_weight,
        chain_weight=args.chain_weight,
        sign_weight=args.sign_weight,
        temporal_weight=args.temporal_weight,
    )
    print(f"Loss weights: sym={args.symmetry_weight}, delta={args.delta_weight}, "
          f"chain={args.chain_weight}, sign={args.sign_weight}, temporal={args.temporal_weight}")

    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # Scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.lr * 0.01,
    )

    # Resume from checkpoint
    start_epoch = 0
    best_val_loss = float("inf")
    best_angle_error = float("inf")

    if args.checkpoint:
        if args.checkpoint.exists():
            print(f"Loading checkpoint: {args.checkpoint}")
            checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if checkpoint.get('scheduler_state_dict'):
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            best_val_loss = checkpoint.get('best_val_loss', float("inf"))
            print(f"Resuming from epoch {start_epoch}")

    # Output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Model name prefix for checkpoints
    model_prefix = f"joint_{args.model_type}"
    if args.use_gat:
        model_prefix += "_gat"

    # Training loop
    print(f"\nStarting training for {args.epochs} epochs...")
    print("=" * 80)

    for epoch in range(start_epoch, args.epochs):
        epoch_start = time.time()

        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, criterion,
            scaler, device, use_amp, amp_dtype,
            temporal=temporal, model_type=args.model_type,
        )

        # Validate
        val_metrics = validate(
            model, val_loader, criterion, device,
            temporal=temporal, model_type=args.model_type,
        )

        # Update scheduler
        scheduler.step()

        epoch_time = time.time() - epoch_start

        # Track best model
        is_best = val_metrics['loss'] < best_val_loss
        if is_best:
            best_val_loss = val_metrics['loss']
            best_angle_error = val_metrics['angle_error_deg']

        # Print progress
        if args.model_type == "semgcn-temporal":
            if IS_TTY:
                print(
                    f"Epoch {epoch+1:3d}/{args.epochs} | "
                    f"Train Loss: {train_metrics['loss']:.4f} | "
                    f"Val Loss: {val_metrics['loss']:.4f} | "
                    f"Train Err: {train_metrics['angle_error_deg']:.2f}° | "
                    f"Val Err: {val_metrics['angle_error_deg']:.2f}° | "
                    f"Sign Acc: {val_metrics['sign_acc']:.1%} | "
                    f"LR: {scheduler.get_last_lr()[0]:.2e} | "
                    f"Time: {epoch_time:.1f}s"
                    f"{' *BEST*' if is_best else ''}"
                )
            else:
                best_marker = " *BEST*" if is_best else ""
                print(
                    f"E{epoch+1:03d} | "
                    f"trn={train_metrics['loss']:.4f} val={val_metrics['loss']:.4f} | "
                    f"err={val_metrics['angle_error_deg']:.2f}° | "
                    f"sign={val_metrics['sign_acc']:.1%} | "
                    f"lr={scheduler.get_last_lr()[0]:.1e} {epoch_time:.0f}s{best_marker}"
                )
        else:
            if IS_TTY:
                print(
                    f"Epoch {epoch+1:3d}/{args.epochs} | "
                    f"Train Loss: {train_metrics['loss']:.4f} | "
                    f"Val Loss: {val_metrics['loss']:.4f} | "
                    f"Train Err: {train_metrics['angle_error_deg']:.2f}° | "
                    f"Val Err: {val_metrics['angle_error_deg']:.2f}° | "
                    f"LR: {scheduler.get_last_lr()[0]:.2e} | "
                    f"Time: {epoch_time:.1f}s"
                    f"{' *BEST*' if is_best else ''}"
                )
            else:
                best_marker = " *BEST*" if is_best else ""
                print(
                    f"E{epoch+1:03d} | "
                    f"trn={train_metrics['loss']:.4f} val={val_metrics['loss']:.4f} | "
                    f"err={val_metrics['angle_error_deg']:.2f}° | "
                    f"lr={scheduler.get_last_lr()[0]:.1e} {epoch_time:.0f}s{best_marker}"
                )

        # Save best model
        if is_best:
            save_checkpoint(
                model, optimizer, scheduler, epoch, best_val_loss, args,
                args.output_dir / f"best_{model_prefix}_model.pth"
            )
            if IS_TTY:
                print(f"  -> New best model saved (val_loss: {best_val_loss:.4f})")

        # Save periodic checkpoint
        if (epoch + 1) % args.save_every == 0:
            save_checkpoint(
                model, optimizer, scheduler, epoch, best_val_loss, args,
                args.output_dir / f"{model_prefix}_epoch_{epoch+1}.pth"
            )

    # Save final model
    save_checkpoint(
        model, optimizer, scheduler, args.epochs - 1, best_val_loss, args,
        args.output_dir / f"{model_prefix}_final.pth"
    )

    print("=" * 80)
    print("Training Complete!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Best mean angle error: {best_angle_error:.2f}°")
    print(f"Best model saved to: {args.output_dir / f'best_{model_prefix}_model.pth'}")


if __name__ == "__main__":
    main()

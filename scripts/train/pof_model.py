#!/usr/bin/env python3
"""Training script for camera-space POF model.

Train a model to predict Part Orientation Fields (POF) from 2D keypoints
in camera space.

Usage:
    uv run --group neural python scripts/train/pof_model.py \
        --data "data/training/aistpp_converted" \
        --epochs 50 --batch-size 256 --workers 8 --bf16

    # With multiple data sources:
    uv run --group neural python scripts/train/pof_model.py \
        --data "data/training/aistpp_converted,data/training/mtc_converted" \
        --epochs 50 --batch-size 256 --workers 8 --bf16

    # Resume from checkpoint:
    uv run --group neural python scripts/train/pof_model.py \
        --data "data/training/aistpp_converted" \
        --checkpoint models/checkpoints/pof_epoch_25.pth \
        --epochs 50
"""

import argparse
import sys
import time
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast, GradScaler

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from src.pof import (
    CameraPOFModel,
    create_pof_model,
    CameraPOFDataset,
    create_pof_dataloaders,
    CameraPOFLoss,
    pof_angular_error,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train camera-space POF model")

    # Data
    parser.add_argument(
        "--data",
        type=str,
        default="data/training/aistpp_converted",
        help="Path to training data (comma-separated for multiple)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Limit samples for debugging",
    )

    # Training
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--val-ratio", type=float, default=0.1)

    # Model architecture
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--dim-feedforward", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)

    # Loss weights
    parser.add_argument("--cosine-weight", type=float, default=1.0)
    parser.add_argument("--symmetry-weight", type=float, default=0.1)

    # Mixed precision
    parser.add_argument("--fp16", action="store_true", help="Use FP16")
    parser.add_argument("--bf16", action="store_true", help="Use BF16")

    # Checkpointing
    parser.add_argument("--checkpoint", type=str, help="Resume from checkpoint")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models/checkpoints",
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
    criterion: CameraPOFLoss,
    scaler: GradScaler,
    device: str,
    use_amp: bool,
    amp_dtype: torch.dtype,
) -> dict:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    total_cosine = 0.0
    total_symmetry = 0.0
    total_angular_error = 0.0
    num_batches = 0

    for batch in train_loader:
        pose_2d = batch["pose_2d"].to(device)
        visibility = batch["visibility"].to(device)
        limb_delta_2d = batch["limb_delta_2d"].to(device)
        limb_length_2d = batch["limb_length_2d"].to(device)
        gt_pof = batch["gt_pof"].to(device)

        optimizer.zero_grad()

        with autocast("cuda", enabled=use_amp, dtype=amp_dtype):
            pred_pof = model(pose_2d, visibility, limb_delta_2d, limb_length_2d)
            losses = criterion(pred_pof, gt_pof, visibility)
            loss = losses["total"]

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
        total_cosine += losses["cosine"].item()
        total_symmetry += losses["symmetry"].item()

        # Angular error metric
        with torch.no_grad():
            angular_err = pof_angular_error(pred_pof, gt_pof).mean().item()
            total_angular_error += angular_err

        num_batches += 1

    return {
        "loss": total_loss / num_batches,
        "cosine": total_cosine / num_batches,
        "symmetry": total_symmetry / num_batches,
        "angular_error_deg": total_angular_error / num_batches,
    }


@torch.no_grad()
def validate(
    model: nn.Module,
    val_loader,
    criterion: CameraPOFLoss,
    device: str,
) -> dict:
    """Validate model."""
    model.eval()
    total_loss = 0.0
    total_cosine = 0.0
    total_angular_error = 0.0
    num_batches = 0

    for batch in val_loader:
        pose_2d = batch["pose_2d"].to(device)
        visibility = batch["visibility"].to(device)
        limb_delta_2d = batch["limb_delta_2d"].to(device)
        limb_length_2d = batch["limb_length_2d"].to(device)
        gt_pof = batch["gt_pof"].to(device)

        pred_pof = model(pose_2d, visibility, limb_delta_2d, limb_length_2d)
        losses = criterion(pred_pof, gt_pof, visibility)

        total_loss += losses["total"].item()
        total_cosine += losses["cosine"].item()

        angular_err = pof_angular_error(pred_pof, gt_pof).mean().item()
        total_angular_error += angular_err

        num_batches += 1

    return {
        "loss": total_loss / num_batches,
        "cosine": total_cosine / num_batches,
        "angular_error_deg": total_angular_error / num_batches,
    }


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
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "best_val_loss": best_val_loss,
        "config": {
            "d_model": args.d_model,
            "num_heads": args.num_heads,
            "num_layers": args.num_layers,
            "dim_feedforward": args.dim_feedforward,
            "dropout": args.dropout,
        },
    }
    torch.save(checkpoint, output_path)


def main():
    args = parse_args()

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
    print(f"Loading data from: {args.data}")
    train_loader, val_loader = create_pof_dataloaders(
        args.data,
        batch_size=args.batch_size,
        num_workers=args.workers,
        val_ratio=args.val_ratio,
        seed=args.seed,
        max_samples=args.max_samples,
    )
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # Model
    model = create_pof_model(
        d_model=args.d_model,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
    )
    model = model.to(device)

    # Loss
    criterion = CameraPOFLoss(
        cosine_weight=args.cosine_weight,
        symmetry_weight=args.symmetry_weight,
    )

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

    if args.checkpoint:
        checkpoint_path = Path(args.checkpoint)
        if checkpoint_path.exists():
            print(f"Loading checkpoint: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            if checkpoint.get("scheduler_state_dict"):
                scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            start_epoch = checkpoint["epoch"] + 1
            best_val_loss = checkpoint.get("best_val_loss", float("inf"))
            print(f"Resuming from epoch {start_epoch}")

    # Output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Training loop
    print(f"\nStarting training for {args.epochs} epochs...")
    print("=" * 60)

    for epoch in range(start_epoch, args.epochs):
        epoch_start = time.time()

        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, criterion,
            scaler, device, use_amp, amp_dtype
        )

        # Validate
        val_metrics = validate(model, val_loader, criterion, device)

        # Update scheduler
        scheduler.step()

        epoch_time = time.time() - epoch_start

        # Print progress
        print(
            f"Epoch {epoch+1:3d}/{args.epochs} | "
            f"Train Loss: {train_metrics['loss']:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f} | "
            f"Train Err: {train_metrics['angular_error_deg']:.2f}° | "
            f"Val Err: {val_metrics['angular_error_deg']:.2f}° | "
            f"LR: {scheduler.get_last_lr()[0]:.2e} | "
            f"Time: {epoch_time:.1f}s"
        )

        # Save best model
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            save_checkpoint(
                model, optimizer, scheduler, epoch, best_val_loss, args,
                output_dir / "best_pof_model.pth"
            )
            print(f"  -> New best model saved (val_loss: {best_val_loss:.4f})")

        # Save periodic checkpoint
        if (epoch + 1) % args.save_every == 0:
            save_checkpoint(
                model, optimizer, scheduler, epoch, best_val_loss, args,
                output_dir / f"pof_epoch_{epoch+1}.pth"
            )

    # Save final model
    save_checkpoint(
        model, optimizer, scheduler, args.epochs - 1, best_val_loss, args,
        output_dir / "pof_final.pth"
    )

    print("=" * 60)
    print(f"Training complete. Best val loss: {best_val_loss:.4f}")
    print(f"Best model saved to: {output_dir / 'best_pof_model.pth'}")


if __name__ == "__main__":
    main()

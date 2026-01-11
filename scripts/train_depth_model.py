#!/usr/bin/env python3
"""
Train PoseFormer depth refinement model on RTX 5080.

GPU-optimized training:
- FP16 mixed precision
- Batch size 256
- DataLoader with 12 workers
- Tensorboard logging
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler
import numpy as np
from tqdm import tqdm
import argparse
from typing import Dict, List, Tuple

from anatomical.depth_model import PoseFormerDepthRefiner
from anatomical.biomechanical_losses import BiomechanicalLoss, EnhancedBiomechanicalLoss


class DepthRefinementDataset(Dataset):
    """Dataset for depth refinement training pairs with temporal sequences."""

    def __init__(self, data_dir: Path, temporal_window: int = 11):
        self.data_dir = data_dir
        self.temporal_window = temporal_window
        self.half_window = temporal_window // 2

        # Load all NPZ files and group by sequence
        all_files = sorted(data_dir.glob("*.npz"))

        if len(all_files) == 0:
            raise ValueError(f"No NPZ files found in {data_dir}")

        # Group files by (sequence, angle, noise) to get temporal sequences
        from collections import defaultdict
        sequences = defaultdict(list)

        for filepath in all_files:
            # Parse filename: 01_01_f0000_a00_n030.npz
            parts = filepath.stem.split('_')
            seq_id = f"{parts[0]}_{parts[1]}"  # e.g., "01_01"
            frame_num = int(parts[2][1:])  # f0000 -> 0
            angle = parts[3]  # a00
            noise = parts[4]  # n030

            key = (seq_id, angle, noise)
            sequences[key].append((frame_num, filepath))

        # Sort each sequence by frame number and create temporal windows
        self.temporal_sequences = []

        for key, frames in sequences.items():
            frames.sort(key=lambda x: x[0])  # Sort by frame number

            # Create sliding windows
            for i in range(len(frames) - temporal_window + 1):
                window_files = [f[1] for f in frames[i:i+temporal_window]]
                self.temporal_sequences.append(window_files)

        # Load first file to get marker info
        sample = np.load(all_files[0])
        self.marker_names = sample["marker_names"].tolist()
        self.num_markers = len(self.marker_names)

        print(f"Dataset: {len(all_files)} total files")
        print(f"Created {len(self.temporal_sequences)} temporal sequences ({temporal_window} frames each)")
        print(f"Markers: {self.num_markers}")

    def __len__(self) -> int:
        return len(self.temporal_sequences)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
        """Get training example.

        Returns:
            features: (frames, markers, feature_dim) input features
            ground_truth: (frames, markers, 3) ground truth positions
            marker_names: List of marker names
        """
        # Load temporal sequence of frames
        window_files = self.temporal_sequences[idx]

        frames_corrupted = []
        frames_gt = []
        camera_angle = None

        for filepath in window_files:
            data = np.load(filepath)
            frames_corrupted.append(data["corrupted"])  # (markers, 3)
            frames_gt.append(data["ground_truth"])  # (markers, 3)

            # Get camera angle (same for all frames in sequence)
            if camera_angle is None:
                camera_angle = float(data["camera_angle"])

        # Stack into temporal sequences
        frames_corrupted = np.stack(frames_corrupted, axis=0)  # (frames, markers, 3)
        frames_gt = np.stack(frames_gt, axis=0)  # (frames, markers, 3)

        # Add small temporal jitter for augmentation
        jitter = np.random.randn(self.temporal_window, self.num_markers, 3) * 0.001  # 1mm noise
        frames_corrupted = frames_corrupted + jitter

        # Build features: x, y, z, visibility, variance, is_augmented, marker_type, camera_angle
        features = np.zeros((self.temporal_window, self.num_markers, 8))
        features[:, :, :3] = frames_corrupted  # x, y, z
        features[:, :, 3] = 0.8  # Default visibility
        features[:, :, 4] = 0.01  # Low variance
        features[:, :, 5] = 0.0  # Not augmented (CMU mocap)
        features[:, :, 6] = 0.0  # Marker type (generic)
        features[:, :, 7] = camera_angle / 90.0  # Normalized camera angle (0-90° → 0-1)

        return (
            torch.from_numpy(features).float(),
            torch.from_numpy(frames_gt).float(),
            self.marker_names
        )


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    biomech_loss: BiomechanicalLoss,
    scaler: GradScaler,
    device: str,
    epoch: int,
) -> Dict[str, float]:
    """Train for one epoch.

    Args:
        model: PoseFormer model
        dataloader: Training data loader
        optimizer: Optimizer
        biomech_loss: Biomechanical loss function
        scaler: GradScaler for FP16
        device: Device
        epoch: Current epoch number

    Returns:
        Dict of average losses
    """
    model.train()

    total_losses = {"total": 0.0, "mse": 0.0, "bone_length": 0.0, "ground_plane": 0.0,
                    "symmetry": 0.0, "smoothness": 0.0, "joint_angle": 0.0}
    num_batches = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")

    for batch_idx, (features, ground_truth, marker_names) in enumerate(pbar):
        features = features.to(device)  # (batch, frames, markers, feature_dim)
        ground_truth = ground_truth.to(device)  # (batch, frames, markers, 3)

        optimizer.zero_grad()

        # Forward pass with FP16
        with autocast('cuda'):
            delta_z, confidence = model(features)

            # Apply depth corrections
            corrupted = features[:, :, :, :3]  # Extract x, y, z
            refined = corrupted.clone()
            refined[:, :, :, 2] += delta_z  # Add depth correction

            # Compute biomechanical loss (self-supervised)
            biomech_total, biomech_dict = biomech_loss(refined, marker_names[0])

            # Compute supervised loss (MSE to ground truth)
            mse_loss = nn.functional.mse_loss(refined, ground_truth)

            # Combined loss: supervised (primary) + biomechanical constraints
            loss = args.mse_weight * mse_loss + biomech_total

            # Update loss dict (extract scalar values, ignore diagnostics for now)
            loss_dict = {
                'mse': mse_loss.item(),
                'total': loss.item(),
            }
            # Add biomechanical losses (skip diagnostics in training loop)
            for key, value in biomech_dict.items():
                if not isinstance(value, dict):  # Skip nested diagnostics
                    loss_dict[key] = value

        # Backward pass with gradient scaling
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # Accumulate losses
        for key, value in loss_dict.items():
            total_losses[key] += value
        num_batches += 1

        # Update progress bar
        pbar.set_postfix({
            "loss": f"{loss_dict['total']:.4f}",
            "mse": f"{loss_dict['mse']:.4f}",
        })

    # Average losses
    avg_losses = {key: value / num_batches for key, value in total_losses.items()}

    return avg_losses


def validate(
    model: nn.Module,
    dataloader: DataLoader,
    biomech_loss: BiomechanicalLoss,
    device: str,
) -> Dict[str, float]:
    """Validate model.

    Args:
        model: PoseFormer model
        dataloader: Validation data loader
        biomech_loss: Biomechanical loss function
        device: Device

    Returns:
        Dict of average losses
    """
    model.eval()

    total_losses = {"total": 0.0, "mse": 0.0, "bone_length": 0.0, "ground_plane": 0.0,
                    "symmetry": 0.0, "smoothness": 0.0, "joint_angle": 0.0}
    num_batches = 0

    with torch.no_grad():
        for features, ground_truth, marker_names in tqdm(dataloader, desc="Validation"):
            features = features.to(device)
            ground_truth = ground_truth.to(device)

            # Forward pass
            delta_z, confidence = model(features)

            # Apply corrections
            corrupted = features[:, :, :, :3]
            refined = corrupted.clone()
            refined[:, :, :, 2] += delta_z

            # Compute biomechanical loss
            biomech_total, biomech_dict = biomech_loss(refined, marker_names[0])

            # Compute supervised loss
            mse_loss = nn.functional.mse_loss(refined, ground_truth)

            # Combined loss
            loss = args.mse_weight * mse_loss + biomech_total

            # Update loss dict (extract scalar values, ignore diagnostics)
            loss_dict = {
                'mse': mse_loss.item(),
                'total': loss.item(),
            }
            # Add biomechanical losses (skip nested diagnostics)
            for key, value in biomech_dict.items():
                if not isinstance(value, dict):
                    loss_dict[key] = value

            # Accumulate
            for key, value in loss_dict.items():
                total_losses[key] += value
            num_batches += 1

    # Average losses
    avg_losses = {key: value / num_batches for key, value in total_losses.items()}

    return avg_losses


def main():
    parser = argparse.ArgumentParser(description="Train PoseFormer depth refinement")
    parser.add_argument("--data", type=str, default="data/training/cmu_converted",
                        help="Path to training data directory")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Batch size (default: 64, balanced for speed/memory)")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument("--fp16", action="store_true", default=True,
                        help="Use FP16 mixed precision")
    parser.add_argument("--workers", type=int, default=8,
                        help="DataLoader workers")
    parser.add_argument("--checkpoint-dir", type=str, default="models/checkpoints",
                        help="Checkpoint directory")
    parser.add_argument("--val-split", type=float, default=0.1,
                        help="Validation split")

    # Loss weight tuning arguments
    parser.add_argument("--mse-weight", type=float, default=10.0,
                        help="Weight for supervised MSE loss (default: 10.0)")
    parser.add_argument("--bone-weight", type=float, default=1.0,
                        help="Weight for bone length CV loss (default: 1.0)")
    parser.add_argument("--rom-weight", type=float, default=0.5,
                        help="Weight for joint angle ROM loss (default: 0.5)")
    parser.add_argument("--ground-weight", type=float, default=0.3,
                        help="Weight for ground plane loss (default: 0.3)")
    parser.add_argument("--symmetry-weight", type=float, default=0.2,
                        help="Weight for symmetry loss (default: 0.2)")
    parser.add_argument("--smoothness-weight", type=float, default=0.1,
                        help="Weight for temporal smoothness loss (default: 0.1)")

    args = parser.parse_args()

    # Setup device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    print()

    # Create model
    print("Creating PoseFormer model...")
    model = PoseFormerDepthRefiner(
        num_markers=59,
        num_frames=11,
        feature_dim=8,  # x, y, z, visibility, variance, is_augmented, marker_type, camera_angle
        d_model=512,
        num_heads=8,
        num_layers=6,
        d_ff=2048,
        dropout=0.1,
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")
    print()

    # Create enhanced loss function with CLI-tunable weights
    biomech_loss = EnhancedBiomechanicalLoss(
        bone_cv_weight=args.bone_weight,
        rom_weight=args.rom_weight,
        ground_weight=args.ground_weight,
        symmetry_weight=args.symmetry_weight,
        smoothness_weight=args.smoothness_weight,
    ).to(device)

    print("Loss weights:")
    print(f"  MSE (supervised):     {args.mse_weight}")
    print(f"  Bone CV (primary):    {args.bone_weight}")
    print(f"  ROM (secondary):      {args.rom_weight}")
    print(f"  Ground plane:         {args.ground_weight}")
    print(f"  Symmetry:             {args.symmetry_weight}")
    print(f"  Smoothness:           {args.smoothness_weight}")
    print()

    # Load dataset
    print(f"Loading dataset from {args.data}...")
    full_dataset = DepthRefinementDataset(Path(args.data))

    # Split train/val
    val_size = int(len(full_dataset) * args.val_split)
    train_size = len(full_dataset) - val_size

    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size]
    )

    print(f"Train: {len(train_dataset)} examples")
    print(f"Val: {len(val_dataset)} examples")
    print()

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )

    # Optimizer and scheduler
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    # GradScaler for FP16
    scaler = GradScaler('cuda') if args.fp16 else None

    # Checkpoint directory
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Training loop
    best_val_loss = float("inf")

    print("Starting training...")
    print("=" * 80)
    print()

    for epoch in range(1, args.epochs + 1):
        # Train
        train_losses = train_epoch(
            model, train_loader, optimizer, biomech_loss, scaler, device, epoch
        )

        # Validate
        val_losses = validate(model, val_loader, biomech_loss, device)

        # Update scheduler
        scheduler.step(val_losses["total"])

        # Print epoch summary
        print(f"\nEpoch {epoch}/{args.epochs}")
        print(f"  Train Loss: {train_losses['total']:.6f}")
        print(f"    MSE: {train_losses['mse']:.6f}")
        print(f"    Bone: {train_losses['bone_length']:.6f}")
        print(f"    Ground: {train_losses['ground_plane']:.6f}")
        print(f"  Val Loss: {val_losses['total']:.6f}")
        print(f"    MSE: {val_losses['mse']:.6f}")
        print()

        # Save checkpoint
        if val_losses["total"] < best_val_loss:
            best_val_loss = val_losses["total"]
            checkpoint_path = checkpoint_dir / "best_model.pth"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": best_val_loss,
            }, checkpoint_path)
            print(f"✓ Saved best model (val_loss={best_val_loss:.6f})")
            print()

    print("=" * 80)
    print("Training complete!")
    print(f"Best validation loss: {best_val_loss:.6f}")
    print(f"Model saved to: {checkpoint_dir / 'best_model.pth'}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Quick test script for the depth refinement model.

Tests:
1. Data loading from generated AIST++ training data
2. Train/val split
3. Model forward pass
4. Loss computation
5. Short training run (3 epochs)
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import numpy as np

from src.depth_refinement.model import PoseAwareDepthRefiner
from src.depth_refinement.dataset import AISTPPDepthDataset
from src.depth_refinement.losses import DepthRefinementLoss


def main():
    print("=" * 60)
    print("DEPTH REFINEMENT MODEL TEST")
    print("=" * 60)

    # Config
    data_dir = Path("data/training/aistpp_converted")
    batch_size = 64
    num_epochs = 3
    num_batches_per_epoch = 50
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\nDevice: {device}")
    print(f"Data directory: {data_dir}")

    # 1. Load dataset
    print("\n=== LOADING DATA ===")
    npz_files = list(data_dir.glob("*.npz"))
    print(f"Found {len(npz_files)} training samples")

    if len(npz_files) < 100:
        print("ERROR: Not enough training data. Need at least 100 samples.")
        return

    # Create dataset (no split - we'll split manually with random_split)
    dataset = AISTPPDepthDataset(data_dir, split='train', val_ratio=0.0, augment=False)
    print(f"Dataset size: {len(dataset)}")

    # Check a sample
    sample = dataset[0]
    print(f"Sample keys: {list(sample.keys())}")
    print(f"  corrupted shape: {sample['corrupted'].shape}")
    print(f"  ground_truth shape: {sample['ground_truth'].shape}")
    print(f"  visibility shape: {sample['visibility'].shape}")
    if 'pose_2d' in sample:
        print(f"  pose_2d shape: {sample['pose_2d'].shape}")
    print(f"  azimuth: {sample['azimuth']:.2f}°")
    print(f"  elevation: {sample['elevation']:.2f}°")

    # 2. Train/Val split
    print("\n=== CREATING TRAIN/VAL SPLIT ===")
    val_size = int(0.2 * len(dataset))
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    print(f"Train size: {train_size}")
    print(f"Val size: {val_size}")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # 3. Create model
    print("\n=== CREATING MODEL ===")
    model = PoseAwareDepthRefiner(
        num_joints=17,
        d_model=64,
        num_heads=4,
        num_layers=4,
        dim_feedforward=256,
        dropout=0.1,
        output_confidence=True,
        use_2d_pose=True,
    )
    model = model.to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")

    # 4. Test forward pass
    print("\n=== TESTING FORWARD PASS ===")
    model.eval()
    with torch.no_grad():
        batch = next(iter(train_loader))
        corrupted = batch['corrupted'].to(device)
        visibility = batch['visibility'].to(device)
        pose_2d = batch['pose_2d'].to(device) if 'pose_2d' in batch else None

        output = model(corrupted, visibility, pose_2d=pose_2d)
        print(f"Output keys: {list(output.keys())}")
        print(f"  delta_xyz shape: {output['delta_xyz'].shape}")
        print(f"  pred_azimuth shape: {output['pred_azimuth'].shape}")
        print(f"  pred_elevation shape: {output['pred_elevation'].shape}")
        if 'confidence' in output:
            print(f"  confidence shape: {output['confidence'].shape}")
    print("Forward pass: OK")

    # 5. Setup training
    print("\n=== SETTING UP TRAINING ===")
    loss_fn = DepthRefinementLoss(
        depth_weight=1.0,
        bone_weight=0.1,
        symmetry_weight=0.05,
        confidence_weight=0.1,
    )
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)

    # 6. Training loop
    print(f"\n=== TRAINING ({num_epochs} epochs, {num_batches_per_epoch} batches each) ===")
    model.train()

    for epoch in range(num_epochs):
        epoch_losses = {'total': 0, 'depth': 0, 'bone': 0, 'symmetry': 0}

        for batch_idx, batch in enumerate(train_loader):
            if batch_idx >= num_batches_per_epoch:
                break

            # Move to device
            corrupted = batch['corrupted'].to(device)
            ground_truth = batch['ground_truth'].to(device)
            visibility = batch['visibility'].to(device)
            pose_2d = batch['pose_2d'].to(device) if 'pose_2d' in batch else None

            # Forward pass
            output = model(corrupted, visibility, pose_2d=pose_2d)

            # Compute loss - pass model output dict, not refined pose
            loss_dict = loss_fn(output, corrupted, ground_truth, visibility)

            # Backward pass
            optimizer.zero_grad()
            loss_dict['total'].backward()
            optimizer.step()

            # Accumulate losses
            for key in epoch_losses:
                epoch_losses[key] += loss_dict[key].item()

        # Print epoch summary
        avg_losses = {k: v / num_batches_per_epoch for k, v in epoch_losses.items()}
        print(f"Epoch {epoch+1}/{num_epochs}: "
              f"total={avg_losses['total']:.4f}, "
              f"depth={avg_losses['depth']:.4f}, "
              f"bone={avg_losses['bone']:.4f}, "
              f"sym={avg_losses['symmetry']:.4f}")

    print("Training: OK")

    # 7. Validation
    print("\n=== VALIDATION ===")
    model.eval()
    val_losses = {'total': 0, 'depth': 0, 'bone': 0, 'symmetry': 0}
    num_val_batches = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            if batch_idx >= 20:  # Limit validation batches
                break

            corrupted = batch['corrupted'].to(device)
            ground_truth = batch['ground_truth'].to(device)
            visibility = batch['visibility'].to(device)
            pose_2d = batch['pose_2d'].to(device) if 'pose_2d' in batch else None

            output = model(corrupted, visibility, pose_2d=pose_2d)
            loss_dict = loss_fn(output, corrupted, ground_truth, visibility)

            for key in val_losses:
                val_losses[key] += loss_dict[key].item()
            num_val_batches += 1

    avg_val_losses = {k: v / max(num_val_batches, 1) for k, v in val_losses.items()}
    print(f"Validation: "
          f"total={avg_val_losses['total']:.4f}, "
          f"depth={avg_val_losses['depth']:.4f}, "
          f"bone={avg_val_losses['bone']:.4f}, "
          f"sym={avg_val_losses['symmetry']:.4f}")

    # 8. Camera prediction test
    print("\n=== CAMERA PREDICTION TEST ===")
    with torch.no_grad():
        batch = next(iter(val_loader))
        corrupted = batch['corrupted'].to(device)
        visibility = batch['visibility'].to(device)
        pose_2d = batch['pose_2d'].to(device) if 'pose_2d' in batch else None
        gt_azimuth = batch['azimuth'].to(device)
        gt_elevation = batch['elevation'].to(device)

        output = model(corrupted, visibility, pose_2d=pose_2d)

        # Compute camera prediction error
        azimuth_error = torch.abs(output['pred_azimuth'] - gt_azimuth).mean()
        elevation_error = torch.abs(output['pred_elevation'] - gt_elevation).mean()

        print(f"Azimuth error: {azimuth_error.item():.2f}°")
        print(f"Elevation error: {elevation_error.item():.2f}°")

    # 9. Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"Data samples: {len(dataset)}")
    print(f"Model params: {num_params:,}")
    print(f"Device: {device}")
    print(f"Final train loss: {avg_losses['total']:.4f}")
    print(f"Final val loss: {avg_val_losses['total']:.4f}")
    print(f"Camera prediction: azimuth ±{azimuth_error.item():.1f}°, elevation ±{elevation_error.item():.1f}°")
    print("\nALL TESTS PASSED!")


if __name__ == '__main__':
    main()

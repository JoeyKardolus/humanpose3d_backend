"""
PyTorch Dataset for AIST++ joint angle refinement training pairs.

Each sample contains:
- corrupted_angles: (12, 3) joint angles from MediaPipe (with depth errors)
- ground_truth_angles: (12, 3) joint angles from AIST++ ground truth
- visibility: (17,) per-joint visibility scores from MediaPipe

Joint order (12 joints):
    pelvis, hip_R, hip_L, knee_R, knee_L, ankle_R, ankle_L,
    trunk, shoulder_R, shoulder_L, elbow_R, elbow_L

Each joint has 3 DOF: flex, abd, rot (except elbow which is 1 DOF)
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Optional, Tuple
import random

from src.application.config.paths import StoragePaths

# Joint names and their indices
JOINT_NAMES = [
    'pelvis', 'hip_R', 'hip_L', 'knee_R', 'knee_L',
    'ankle_R', 'ankle_L', 'trunk', 'shoulder_R', 'shoulder_L',
    'elbow_R', 'elbow_L',
]

# Mapping from joint index to relevant visibility indices (COCO 17)
# Used to determine which joints were well-observed
JOINT_TO_VISIBILITY = {
    0: [11, 12],  # pelvis -> hips
    1: [12, 14],  # hip_R -> R hip, R knee
    2: [11, 13],  # hip_L -> L hip, L knee
    3: [14, 16],  # knee_R -> R knee, R ankle
    4: [13, 15],  # knee_L -> L knee, L ankle
    5: [16],      # ankle_R -> R ankle
    6: [15],      # ankle_L -> L ankle
    7: [5, 6, 11, 12],  # trunk -> shoulders, hips
    8: [6, 8],    # shoulder_R -> R shoulder, R elbow
    9: [5, 7],    # shoulder_L -> L shoulder, L elbow
    10: [8, 10],  # elbow_R -> R elbow, R wrist
    11: [7, 9],   # elbow_L -> L elbow, L wrist
}


class AISTPPJointDataset(Dataset):
    """Dataset for AIST++ joint angle refinement training pairs."""

    def __init__(
        self,
        data_dir: str | Path,
        split: str = 'train',
        val_ratio: float = 0.1,
        seed: int = 42,
        augment: bool = True,
        max_samples: Optional[int] = None,
    ):
        """
        Initialize dataset.

        Args:
            data_dir: Path to joint angle training data (NPZ files)
            split: 'train' or 'val'
            val_ratio: Fraction of data for validation
            seed: Random seed for train/val split
            augment: Whether to apply data augmentation
            max_samples: Limit number of samples (for debugging)
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.augment = augment and (split == 'train')

        # Find all NPZ files
        all_files = sorted(self.data_dir.glob('*.npz'))

        if max_samples:
            all_files = all_files[:max_samples]

        # Deterministic train/val split by sequence
        sequences = set(self._get_sequence(f) for f in all_files)
        sequences = sorted(sequences)

        rng = random.Random(seed)
        rng.shuffle(sequences)

        n_val = max(1, int(len(sequences) * val_ratio))
        val_sequences = set(sequences[:n_val])
        train_sequences = set(sequences[n_val:])

        # Filter files by split
        if split == 'train':
            self.files = [f for f in all_files if self._get_sequence(f) in train_sequences]
        else:
            self.files = [f for f in all_files if self._get_sequence(f) in val_sequences]

        print(f"[{split}] Loaded {len(self.files)} samples from {len(sequences)} sequences")

    @staticmethod
    def _get_sequence(path: Path) -> str:
        """Extract sequence name from filename."""
        # e.g., gBR_sBM_cAll_d04_mBR0_ch01_c01_f000000.npz -> gBR_sBM_cAll_d04_mBR0_ch01_c01
        return '_'.join(path.stem.split('_')[:-1])

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> dict:
        """Load a training sample.

        Returns:
            dict with:
                'corrupted_angles': (12, 3) tensor - joint angles from MediaPipe
                'ground_truth_angles': (12, 3) tensor - joint angles from GT
                'joint_visibility': (12,) tensor - visibility per joint
                'visibility': (17,) tensor - raw COCO visibility
        """
        # Try multiple samples to skip gimbal lock cases
        max_attempts = 10
        for attempt in range(max_attempts):
            actual_idx = (idx + attempt) % len(self.files)
            data = np.load(self.files[actual_idx])

            corrupted_angles_np = data['corrupted_angles'].astype(np.float32)
            ground_truth_angles_np = data['ground_truth_angles'].astype(np.float32)

            # Skip gimbal lock cases (angles near ±180° from Euler decomposition)
            if np.any(np.abs(ground_truth_angles_np) > 170):
                continue
            if np.any(np.abs(corrupted_angles_np) > 170):
                continue

            # Valid sample - proceed with loading
            corrupted_angles = torch.from_numpy(corrupted_angles_np)
            ground_truth_angles = torch.from_numpy(ground_truth_angles_np)
            visibility = torch.from_numpy(data['visibility'].astype(np.float32))

            # Compute per-joint visibility from COCO visibility
            joint_visibility = self._compute_joint_visibility(visibility)

            # Optional augmentation
            if self.augment:
                corrupted_angles, ground_truth_angles = self._augment(
                    corrupted_angles, ground_truth_angles
                )

            return {
                'corrupted_angles': corrupted_angles,
                'ground_truth_angles': ground_truth_angles,
                'joint_visibility': joint_visibility,
                'visibility': visibility,
            }

        # Fallback: return original sample if no valid sample found after max_attempts
        data = np.load(self.files[idx])
        corrupted_angles = torch.from_numpy(data['corrupted_angles'].astype(np.float32))
        ground_truth_angles = torch.from_numpy(data['ground_truth_angles'].astype(np.float32))
        visibility = torch.from_numpy(data['visibility'].astype(np.float32))
        joint_visibility = self._compute_joint_visibility(visibility)

        if self.augment:
            corrupted_angles, ground_truth_angles = self._augment(
                corrupted_angles, ground_truth_angles
            )

        return {
            'corrupted_angles': corrupted_angles,
            'ground_truth_angles': ground_truth_angles,
            'joint_visibility': joint_visibility,
            'visibility': visibility,
        }

    def _compute_joint_visibility(self, visibility: torch.Tensor) -> torch.Tensor:
        """Compute per-joint visibility from COCO visibility.

        Uses minimum visibility of relevant COCO joints for each joint angle.
        """
        joint_vis = torch.zeros(12, dtype=torch.float32)
        for i, indices in JOINT_TO_VISIBILITY.items():
            # Take minimum visibility of relevant joints
            joint_vis[i] = visibility[indices].min()
        return joint_vis

    def _augment(
        self,
        corrupted: torch.Tensor,
        ground_truth: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply data augmentation.

        Augmentations for joint angles:
        - Random left/right swap (mirror)
        - Small angle noise on corrupted (not GT)
        """
        # Random left/right swap (swap L/R joints)
        if random.random() > 0.5:
            # Swap hip, knee, ankle, shoulder, elbow (R <-> L)
            swap_pairs = [(1, 2), (3, 4), (5, 6), (8, 9), (10, 11)]
            for i, j in swap_pairs:
                corrupted[[i, j]] = corrupted[[j, i]]
                ground_truth[[i, j]] = ground_truth[[j, i]]

            # Mirror abd and rot signs (they flip with L/R)
            # flex stays same, abd and rot flip sign
            corrupted[:, 1] = -corrupted[:, 1]  # abd
            corrupted[:, 2] = -corrupted[:, 2]  # rot
            ground_truth[:, 1] = -ground_truth[:, 1]
            ground_truth[:, 2] = -ground_truth[:, 2]

        # Small noise on corrupted angles (simulate additional MediaPipe noise)
        if random.random() > 0.5:
            noise = torch.randn_like(corrupted) * 2.0  # ±2° std
            corrupted = corrupted + noise

        return corrupted, ground_truth


def create_dataloaders(
    data_dir: str | Path,
    batch_size: int = 64,
    num_workers: int = 4,
    val_ratio: float = 0.1,
    seed: int = 42,
    max_samples: Optional[int] = None,
) -> Tuple[DataLoader, DataLoader]:
    """Create train and validation dataloaders.

    Args:
        data_dir: Path to joint angle training data
        batch_size: Batch size
        num_workers: Number of data loading workers
        val_ratio: Validation split ratio
        seed: Random seed
        max_samples: Limit samples (for debugging)

    Returns:
        (train_loader, val_loader)
    """
    train_dataset = AISTPPJointDataset(
        data_dir,
        split='train',
        val_ratio=val_ratio,
        seed=seed,
        augment=True,
        max_samples=max_samples,
    )

    val_dataset = AISTPPJointDataset(
        data_dir,
        split='val',
        val_ratio=val_ratio,
        seed=seed,
        augment=False,
        max_samples=max_samples,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader


if __name__ == '__main__':
    # Test dataset
    data_dir = StoragePaths.load().training_root / "aistpp_joint_angles"

    if not data_dir.exists():
        print(f"Data directory not found: {data_dir}")
        exit(1)

    train_loader, val_loader = create_dataloaders(data_dir, batch_size=4, num_workers=0)

    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")

    # Check a sample
    batch = next(iter(train_loader))
    print(f"\nBatch shapes:")
    for key, value in batch.items():
        print(f"  {key}: {value.shape}")

    # Check angle errors
    angle_errors = batch['corrupted_angles'] - batch['ground_truth_angles']
    print(f"\nAngle error stats:")
    print(f"  Mean: {angle_errors.abs().mean():.2f}°")
    print(f"  Std: {angle_errors.std():.2f}°")
    print(f"  Max: {angle_errors.abs().max():.2f}°")

    print(f"\nJoint visibility: {batch['joint_visibility'][0]}")

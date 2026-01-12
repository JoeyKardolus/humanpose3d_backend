"""
PyTorch Dataset for AIST++ depth refinement training pairs.

Each sample contains:
- corrupted: (17, 3) MediaPipe 3D pose with REAL depth errors
- ground_truth: (17, 3) AIST++ mocap ground truth
- visibility: (17,) per-joint visibility scores
- view_angle: scalar view angle in degrees (0-90)
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Optional, Tuple, List
import random


class AISTPPDepthDataset(Dataset):
    """Dataset for AIST++ depth refinement training pairs."""

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
            data_dir: Path to training data (NPZ files)
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
        # e.g., gBR_sBM_cAll_d04_mBR0_ch01_f000000.npz -> gBR_sBM_cAll_d04_mBR0_ch01
        return '_'.join(path.stem.split('_')[:-1])

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> dict:
        """Load a training sample.

        Returns:
            dict with:
                'corrupted': (17, 3) tensor
                'ground_truth': (17, 3) tensor
                'visibility': (17,) tensor
                'view_angle': scalar tensor
        """
        data = np.load(self.files[idx])

        corrupted = torch.from_numpy(data['corrupted'].astype(np.float32))
        ground_truth = torch.from_numpy(data['ground_truth'].astype(np.float32))
        visibility = torch.from_numpy(data['visibility'].astype(np.float32))
        view_angle = torch.tensor(float(data['view_angle']), dtype=torch.float32)

        # Optional augmentation
        if self.augment:
            corrupted, ground_truth = self._augment(corrupted, ground_truth)

        return {
            'corrupted': corrupted,
            'ground_truth': ground_truth,
            'visibility': visibility,
            'view_angle': view_angle,
        }

    def _augment(
        self,
        corrupted: torch.Tensor,
        ground_truth: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply data augmentation.

        Augmentations that preserve depth error patterns:
        - Random rotation around Y axis (vertical)
        - Random X/Z flip (mirror left/right)
        - Small random scale
        """
        # Random Y rotation (doesn't affect depth errors much)
        if random.random() > 0.5:
            angle = random.uniform(-0.3, 0.3)  # ±17 degrees
            cos_a, sin_a = np.cos(angle), np.sin(angle)
            rot = torch.tensor([
                [cos_a, 0, sin_a],
                [0, 1, 0],
                [-sin_a, 0, cos_a],
            ], dtype=torch.float32)

            corrupted = torch.matmul(corrupted, rot)
            ground_truth = torch.matmul(ground_truth, rot)

        # Random X flip (left/right swap)
        if random.random() > 0.5:
            corrupted[:, 0] = -corrupted[:, 0]
            ground_truth[:, 0] = -ground_truth[:, 0]

            # Swap left/right joints
            # COCO order: 0=nose, 1=Leye, 2=Reye, 3=Lear, 4=Rear, 5=Lshoulder, 6=Rshoulder,
            # 7=Lelbow, 8=Relbow, 9=Lwrist, 10=Rwrist, 11=Lhip, 12=Rhip,
            # 13=Lknee, 14=Rknee, 15=Lankle, 16=Rankle
            swap_pairs = [(1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16)]
            for i, j in swap_pairs:
                corrupted[[i, j]] = corrupted[[j, i]]
                ground_truth[[i, j]] = ground_truth[[j, i]]

        # Small random scale
        if random.random() > 0.5:
            scale = random.uniform(0.9, 1.1)
            corrupted = corrupted * scale
            ground_truth = ground_truth * scale

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
        data_dir: Path to training data
        batch_size: Batch size
        num_workers: Number of data loading workers
        val_ratio: Validation split ratio
        seed: Random seed
        max_samples: Limit samples (for debugging)

    Returns:
        (train_loader, val_loader)
    """
    train_dataset = AISTPPDepthDataset(
        data_dir,
        split='train',
        val_ratio=val_ratio,
        seed=seed,
        augment=True,
        max_samples=max_samples,
    )

    val_dataset = AISTPPDepthDataset(
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
    data_dir = Path('data/training/aistpp_converted')

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

    print(f"\nView angles: {batch['view_angle']}")
    print(f"Visibility range: {batch['visibility'].min():.2f} - {batch['visibility'].max():.2f}")

    # Check depth errors
    depth_errors = batch['corrupted'][:, :, 2] - batch['ground_truth'][:, :, 2]
    print(f"\nDepth error stats:")
    print(f"  Mean: {depth_errors.mean():.4f} m")
    print(f"  Std: {depth_errors.std():.4f} m")
    print(f"  Max: {depth_errors.abs().max():.4f} m")

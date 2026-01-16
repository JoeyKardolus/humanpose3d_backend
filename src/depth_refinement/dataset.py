"""
PyTorch Dataset for AIST++ depth refinement training pairs.

Each sample contains:
- corrupted: (17, 3) MediaPipe 3D pose with REAL depth errors
- ground_truth: (17, 3) AIST++ mocap ground truth
- visibility: (17,) per-joint visibility scores
- pose_2d: (17, 2) MediaPipe 2D pose (normalized image coordinates) - key for camera prediction!
- camera_pos: (3,) camera position relative to pelvis (for training camera predictor)
- azimuth: scalar azimuth angle in degrees (0-360)
- elevation: scalar elevation angle in degrees (-90 to +90)

The 2D pose is crucial for camera viewpoint estimation (ElePose CVPR 2022 insight):
- Foreshortening patterns directly encode camera angle
- Left/right asymmetry encodes azimuth
- Relative joint positions encode elevation
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Optional, Tuple, List, Dict
import random
import re
from collections import defaultdict


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
        """Extract sequence name from filename (contains dancer ID)."""
        # e.g., gBR_sBM_cAll_d04_mBR0_ch01_f0001_a30_n050.npz -> gBR_sBM_cAll_d04_mBR0_ch01
        # The sequence name is everything before _f{frame}
        stem = path.stem
        match = re.match(r'^(.+)_f\d+', stem)
        if match:
            return match.group(1)
        # Fallback for old format
        return '_'.join(stem.split('_')[:-1])

    @staticmethod
    def _get_frame_num(path: Path) -> int:
        """Extract frame number from filename."""
        # e.g., gBR_sBM_cAll_d04_mBR0_ch01_f0001_a30_n050.npz -> 1
        match = re.search(r'_f(\d+)', path.stem)
        if match:
            return int(match.group(1))
        return 0

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> dict:
        """Load a training sample.

        Returns:
            dict with:
                'corrupted': (17, 3) tensor
                'ground_truth': (17, 3) tensor
                'visibility': (17,) tensor
                'pose_2d': (17, 2) tensor - 2D pose (normalized image coords)
                'camera_pos': (3,) tensor - camera position relative to pelvis
                'azimuth': scalar tensor (0-360 degrees)
                'elevation': scalar tensor (-90 to +90 degrees)
        """
        data = np.load(self.files[idx])

        corrupted = torch.from_numpy(data['corrupted'].astype(np.float32))
        ground_truth = torch.from_numpy(data['ground_truth'].astype(np.float32))
        visibility = torch.from_numpy(data['visibility'].astype(np.float32))

        # Load 2D pose (key for camera prediction - foreshortening encodes viewpoint!)
        if 'pose_2d' in data:
            pose_2d = torch.from_numpy(data['pose_2d'].astype(np.float32))
        else:
            # Backward compatibility: generate synthetic 2D from 3D if not available
            # This is less accurate but allows training on old data
            # Project 3D to 2D using perspective (assume f=1, center at origin)
            pose_2d = corrupted[:, :2] / (corrupted[:, 2:3] + 1e-6)  # x/z, y/z
            # Normalize to [0, 1] range (rough approximation)
            pose_2d = (pose_2d + 1) / 2

        # Load camera position (new format) or create dummy
        if 'camera_relative' in data:
            camera_pos = torch.from_numpy(data['camera_relative'].astype(np.float32))
        else:
            # Backward compatibility: no camera position in old format
            # Create dummy position (will use azimuth/elevation instead)
            camera_pos = torch.zeros(3, dtype=torch.float32)

        # Load azimuth/elevation (new format) or fall back to view_angle (old format)
        if 'azimuth' in data:
            azimuth = torch.tensor(float(data['azimuth']), dtype=torch.float32)
            elevation = torch.tensor(float(data['elevation']), dtype=torch.float32)
        else:
            # Backward compatibility: old format had single view_angle (0-90)
            # Convert to azimuth (assume frontal view) and elevation (0)
            view_angle = float(data['view_angle'])
            azimuth = torch.tensor(view_angle, dtype=torch.float32)
            elevation = torch.tensor(0.0, dtype=torch.float32)

        # Optional augmentation
        if self.augment:
            corrupted, ground_truth, azimuth, camera_pos, pose_2d = self._augment(
                corrupted, ground_truth, azimuth, camera_pos, pose_2d
            )

        return {
            'corrupted': corrupted,
            'ground_truth': ground_truth,
            'visibility': visibility,
            'pose_2d': pose_2d,
            'camera_pos': camera_pos,
            'azimuth': azimuth,
            'elevation': elevation,
        }

    def _augment(
        self,
        corrupted: torch.Tensor,
        ground_truth: torch.Tensor,
        azimuth: torch.Tensor,
        camera_pos: torch.Tensor,
        pose_2d: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply data augmentation.

        Augmentations that preserve depth error patterns:
        - Random rotation around Y axis (vertical) - adjusts azimuth, camera_pos, and 2D pose
        - Random X flip (mirror left/right) - mirrors azimuth, camera_pos, and 2D pose
        - Small random scale
        """
        # Random Y rotation (rotates the pose, adjusts azimuth and camera position)
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

            # Rotate camera position (inverse rotation - if we rotate pose CW, camera appears CCW)
            rot_inv = rot.T
            camera_pos = torch.matmul(camera_pos, rot_inv)

            # Adjust azimuth: rotating pose clockwise = camera appears to move counter-clockwise
            angle_deg = np.degrees(angle)
            azimuth = (azimuth - angle_deg) % 360.0

            # 2D pose: small rotations in world don't significantly change 2D appearance
            # (it's a projection), but we could apply a slight rotation in image space
            # For simplicity, we keep 2D unchanged for Y rotation (rotation around vertical)

        # Random X flip (left/right swap) - mirrors the view
        if random.random() > 0.5:
            corrupted[:, 0] = -corrupted[:, 0]
            ground_truth[:, 0] = -ground_truth[:, 0]

            # Mirror camera position X coordinate
            camera_pos[0] = -camera_pos[0]

            # Mirror 2D pose X coordinate (flip horizontally in image)
            pose_2d[:, 0] = 1.0 - pose_2d[:, 0]  # Assuming normalized [0, 1] coordinates

            # Swap left/right joints in 3D
            # COCO order: 0=nose, 1=Leye, 2=Reye, 3=Lear, 4=Rear, 5=Lshoulder, 6=Rshoulder,
            # 7=Lelbow, 8=Relbow, 9=Lwrist, 10=Rwrist, 11=Lhip, 12=Rhip,
            # 13=Lknee, 14=Rknee, 15=Lankle, 16=Rankle
            swap_pairs = [(1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16)]
            for i, j in swap_pairs:
                corrupted[[i, j]] = corrupted[[j, i]]
                ground_truth[[i, j]] = ground_truth[[j, i]]
                pose_2d[[i, j]] = pose_2d[[j, i]]  # Swap 2D joints too!

            # Mirror azimuth: 90° (right) becomes 270° (left), etc.
            # Formula: new_az = (360 - az) % 360
            azimuth = (360.0 - azimuth) % 360.0

        # Small random scale (only affects pose, not camera position direction)
        if random.random() > 0.5:
            scale = random.uniform(0.9, 1.1)
            corrupted = corrupted * scale
            ground_truth = ground_truth * scale
            # Camera position distance scales too
            camera_pos = camera_pos * scale
            # 2D pose: scale around center (0.5, 0.5)
            pose_2d = 0.5 + (pose_2d - 0.5) * scale

        return corrupted, ground_truth, azimuth, camera_pos, pose_2d


class TemporalWindowDataset(Dataset):
    """Dataset that returns temporal windows from same sequence (same person).

    This enables bone locking during training - bone lengths should be consistent
    within a window since it's the same person.
    """

    def __init__(
        self,
        data_dir: str | Path,
        window_size: int = 50,
        split: str = 'train',
        val_ratio: float = 0.1,
        seed: int = 42,
        max_sequences: Optional[int] = None,
    ):
        """
        Initialize temporal window dataset.

        Args:
            data_dir: Path to training data (NPZ files)
            window_size: Number of consecutive frames per window
            split: 'train' or 'val'
            val_ratio: Fraction of sequences for validation
            seed: Random seed for train/val split
            max_sequences: Limit number of sequences (for debugging)
        """
        self.data_dir = Path(data_dir)
        self.window_size = window_size
        self.split = split

        # Find all NPZ files and group by sequence
        all_files = sorted(self.data_dir.glob('*.npz'))

        # Group files by sequence (same dancer)
        sequence_files: Dict[str, List[Path]] = defaultdict(list)
        for f in all_files:
            seq = self._get_sequence(f)
            sequence_files[seq].append(f)

        # Sort files within each sequence by frame number
        for seq in sequence_files:
            sequence_files[seq].sort(key=lambda f: self._get_frame_num(f))

        # Filter sequences that have enough frames for a window
        valid_sequences = {
            seq: files for seq, files in sequence_files.items()
            if len(files) >= window_size
        }

        # Split sequences into train/val
        sequences = sorted(valid_sequences.keys())
        rng = random.Random(seed)
        rng.shuffle(sequences)

        if max_sequences:
            sequences = sequences[:max_sequences]

        n_val = max(1, int(len(sequences) * val_ratio))
        val_sequences = set(sequences[:n_val])
        train_sequences = set(sequences[n_val:])

        # Keep only sequences for this split
        if split == 'train':
            self.sequences = {s: valid_sequences[s] for s in train_sequences if s in valid_sequences}
        else:
            self.sequences = {s: valid_sequences[s] for s in val_sequences if s in valid_sequences}

        # Create list of (sequence_name, start_idx) pairs for all valid windows
        self.windows = []
        for seq_name, files in self.sequences.items():
            # How many windows can we extract from this sequence?
            n_windows = len(files) - window_size + 1
            for start_idx in range(n_windows):
                self.windows.append((seq_name, start_idx))

        print(f"[{split}] {len(self.windows)} windows from {len(self.sequences)} sequences (window_size={window_size})")

    @staticmethod
    def _get_sequence(path: Path) -> str:
        """Extract sequence name from filename (contains dancer ID)."""
        stem = path.stem
        match = re.match(r'^(.+)_f\d+', stem)
        if match:
            return match.group(1)
        return '_'.join(stem.split('_')[:-1])

    @staticmethod
    def _get_frame_num(path: Path) -> int:
        """Extract frame number from filename."""
        match = re.search(r'_f(\d+)', path.stem)
        if match:
            return int(match.group(1))
        return 0

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> dict:
        """Load a temporal window.

        Returns:
            dict with tensors of shape (window_size, ...):
                'corrupted': (W, 17, 3)
                'ground_truth': (W, 17, 3)
                'visibility': (W, 17)
                'pose_2d': (W, 17, 2)
                'azimuth': (W,)
                'elevation': (W,)
        """
        seq_name, start_idx = self.windows[idx]
        files = self.sequences[seq_name][start_idx:start_idx + self.window_size]

        # Load all frames in window
        corrupted_list = []
        ground_truth_list = []
        visibility_list = []
        pose_2d_list = []
        azimuth_list = []
        elevation_list = []

        for f in files:
            data = np.load(f)

            corrupted_list.append(data['corrupted'].astype(np.float32))
            ground_truth_list.append(data['ground_truth'].astype(np.float32))
            visibility_list.append(data['visibility'].astype(np.float32))

            if 'pose_2d' in data:
                pose_2d_list.append(data['pose_2d'].astype(np.float32))
            else:
                # Backward compatibility
                pose_2d = corrupted_list[-1][:, :2] / (corrupted_list[-1][:, 2:3] + 1e-6)
                pose_2d = (pose_2d + 1) / 2
                pose_2d_list.append(pose_2d)

            if 'azimuth' in data:
                azimuth_list.append(float(data['azimuth']))
                elevation_list.append(float(data['elevation']))
            else:
                azimuth_list.append(float(data['view_angle']))
                elevation_list.append(0.0)

        return {
            'corrupted': torch.from_numpy(np.stack(corrupted_list)),
            'ground_truth': torch.from_numpy(np.stack(ground_truth_list)),
            'visibility': torch.from_numpy(np.stack(visibility_list)),
            'pose_2d': torch.from_numpy(np.stack(pose_2d_list)),
            'azimuth': torch.tensor(azimuth_list, dtype=torch.float32),
            'elevation': torch.tensor(elevation_list, dtype=torch.float32),
        }


def create_temporal_dataloaders(
    data_dir: str | Path,
    window_size: int = 50,
    batch_size: int = 1,  # Each "batch" is actually a window
    num_workers: int = 4,
    val_ratio: float = 0.1,
    seed: int = 42,
    max_sequences: Optional[int] = None,
) -> Tuple[DataLoader, DataLoader]:
    """Create train and validation dataloaders with temporal windows.

    Each batch contains window_size consecutive frames from the same sequence
    (same person), enabling bone locking during training.

    Args:
        data_dir: Path to training data
        window_size: Frames per window (default: 50)
        batch_size: Number of windows per batch (default: 1)
        num_workers: Number of data loading workers
        val_ratio: Validation split ratio
        seed: Random seed
        max_sequences: Limit sequences (for debugging)

    Returns:
        (train_loader, val_loader)
    """
    train_dataset = TemporalWindowDataset(
        data_dir,
        window_size=window_size,
        split='train',
        val_ratio=val_ratio,
        seed=seed,
        max_sequences=max_sequences,
    )

    val_dataset = TemporalWindowDataset(
        data_dir,
        window_size=window_size,
        split='val',
        val_ratio=val_ratio,
        seed=seed,
        max_sequences=max_sequences,
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

    print(f"\n2D Pose (for camera prediction):")
    print(f"  Shape: {batch['pose_2d'].shape}")
    print(f"  X range: {batch['pose_2d'][:, :, 0].min():.3f} - {batch['pose_2d'][:, :, 0].max():.3f}")
    print(f"  Y range: {batch['pose_2d'][:, :, 1].min():.3f} - {batch['pose_2d'][:, :, 1].max():.3f}")

    print(f"\nCamera position (relative to pelvis): {batch['camera_pos']}")
    print(f"Azimuth (0-360°): {batch['azimuth']}")
    print(f"Elevation (-90 to +90°): {batch['elevation']}")
    print(f"Visibility range: {batch['visibility'].min():.2f} - {batch['visibility'].max():.2f}")

    # Check depth errors
    depth_errors = batch['corrupted'][:, :, 2] - batch['ground_truth'][:, :, 2]
    print(f"\nDepth error stats:")
    print(f"  Mean: {depth_errors.mean():.4f} m")
    print(f"  Std: {depth_errors.std():.4f} m")
    print(f"  Max: {depth_errors.abs().max():.4f} m")

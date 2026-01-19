"""
Dataset for MainRefiner training.

Provides outputs from both constraint models (depth + joint) for training
the fusion network. Supports two modes:

1. Pre-computed mode: Load pre-computed model outputs from disk (faster)
2. Online mode: Run models on-the-fly during training (more flexible)

The dataset uses the same underlying AIST++ data as the individual models.
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Optional, Tuple, Dict, Union
import random
import re


class MainRefinerDataset(Dataset):
    """Dataset that provides constraint model outputs for fusion training.

    This dataset wraps the depth refinement dataset and runs both constraint
    models to generate their outputs. For efficiency, model outputs can be
    pre-computed and cached.
    """

    def __init__(
        self,
        data_dir: Union[str, Path, list],
        depth_model: Optional[torch.nn.Module] = None,
        joint_model: Optional[torch.nn.Module] = None,
        split: str = 'train',
        val_ratio: float = 0.1,
        seed: int = 42,
        max_samples: Optional[int] = None,
        device: str = 'cpu',
    ):
        """
        Initialize dataset.

        Args:
            data_dir: Path(s) to AIST++ training data (depth format)
            depth_model: PoseAwareDepthRefiner model (optional, for online mode)
            joint_model: JointConstraintRefiner model (optional, for online mode)
            split: 'train' or 'val'
            val_ratio: Fraction of data for validation
            seed: Random seed for train/val split
            max_samples: Limit number of samples (for debugging)
            device: Device for running models ('cpu' or 'cuda')
        """
        self.split = split
        self.depth_model = depth_model
        self.joint_model = joint_model
        self.device = device

        # Set models to eval mode if provided
        if self.depth_model is not None:
            self.depth_model.eval()
        if self.joint_model is not None:
            self.joint_model.eval()

        # Support multiple data directories
        if isinstance(data_dir, str) and ',' in data_dir:
            data_dirs = [Path(d.strip()) for d in data_dir.split(',')]
        elif isinstance(data_dir, list):
            data_dirs = [Path(d) for d in data_dir]
        else:
            data_dirs = [Path(data_dir)]

        # Find all NPZ files, filtering out empty/corrupted files
        all_files = []
        skipped = 0
        for d in data_dirs:
            if d.exists():
                for f in sorted(d.glob('*.npz')):
                    if f.stat().st_size == 0:
                        skipped += 1
                        continue
                    all_files.append(f)
            else:
                print(f"Warning: data directory not found: {d}")

        if skipped > 0:
            print(f"Warning: skipped {skipped} empty/corrupted files")

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
        stem = path.stem
        match = re.match(r'^(.+)_f\d+', stem)
        if match:
            return match.group(1)
        return '_'.join(stem.split('_')[:-1])

    def __len__(self) -> int:
        return len(self.files)

    @torch.no_grad()
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Load a training sample with constraint model outputs.

        Returns:
            dict with:
                'raw_pose': (17, 3) corrupted MediaPipe pose
                'ground_truth': (17, 3) AIST++ ground truth
                'visibility': (17,) per-joint visibility
                'pose_2d': (17, 2) 2D pose (for depth model)
                'depth_outputs': dict with depth model outputs (if model provided)
                'joint_outputs': dict with joint model outputs (if model provided)
                'azimuth': scalar camera azimuth
                'elevation': scalar camera elevation
        """
        try:
            data = np.load(self.files[idx])
        except (EOFError, ValueError, OSError):
            # Handle corrupted files by returning a random valid sample
            alt_idx = random.randint(0, len(self.files) - 1)
            if alt_idx == idx:
                alt_idx = (idx + 1) % len(self.files)
            return self.__getitem__(alt_idx)

        # Load base data
        corrupted = torch.from_numpy(data['corrupted'].astype(np.float32))
        ground_truth = torch.from_numpy(data['ground_truth'].astype(np.float32))
        visibility = torch.from_numpy(data['visibility'].astype(np.float32))

        # Load 2D pose
        if 'pose_2d' in data:
            pose_2d = torch.from_numpy(data['pose_2d'].astype(np.float32))
        else:
            # Fallback: generate synthetic 2D from 3D
            pose_2d = corrupted[:, :2] / (corrupted[:, 2:3].clamp(min=1e-4))
            pose_2d = (pose_2d + 1) / 2

        # Load projected 2D for POF
        if 'projected_2d' in data:
            projected_2d = torch.from_numpy(data['projected_2d'].astype(np.float32))
        else:
            projected_2d = pose_2d.clone()

        # Load camera angles
        if 'azimuth' in data:
            azimuth = torch.tensor(float(data['azimuth']), dtype=torch.float32)
            elevation = torch.tensor(float(data['elevation']), dtype=torch.float32)
        else:
            azimuth = torch.tensor(0.0, dtype=torch.float32)
            elevation = torch.tensor(0.0, dtype=torch.float32)

        result = {
            'raw_pose': corrupted,
            'ground_truth': ground_truth,
            'visibility': visibility,
            'pose_2d': pose_2d,
            'projected_2d': projected_2d,
            'azimuth': azimuth,
            'elevation': elevation,
        }

        # Run depth model if provided
        if self.depth_model is not None:
            depth_outputs = self._run_depth_model(
                corrupted, visibility, pose_2d, projected_2d, azimuth, elevation
            )
            result['depth_outputs'] = depth_outputs

        # Run joint model if provided
        if self.joint_model is not None:
            joint_outputs = self._run_joint_model(corrupted, visibility)
            result['joint_outputs'] = joint_outputs

        return result

    def _run_depth_model(
        self,
        corrupted: torch.Tensor,
        visibility: torch.Tensor,
        pose_2d: torch.Tensor,
        projected_2d: torch.Tensor,
        azimuth: torch.Tensor,
        elevation: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Run depth model on a single sample."""
        # Add batch dimension
        corrupted_b = corrupted.unsqueeze(0).to(self.device)
        visibility_b = visibility.unsqueeze(0).to(self.device)
        pose_2d_b = pose_2d.unsqueeze(0).to(self.device)
        projected_2d_b = projected_2d.unsqueeze(0).to(self.device)

        output = self.depth_model(
            corrupted_b,
            visibility_b,
            pose_2d=pose_2d_b,
            projected_2d=projected_2d_b,
        )

        # Remove batch dimension and move to CPU
        return {k: v.squeeze(0).cpu() for k, v in output.items()}

    def _run_joint_model(
        self,
        corrupted: torch.Tensor,
        visibility: torch.Tensor,
        precomputed_angles: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Run joint model on a single sample.

        Joint angles must be pre-computed from augmented TRC data using
        comprehensive_joint_angles.py. Cannot compute from COCO-17 directly.

        Args:
            corrupted: (17, 3) pose tensor (unused if precomputed_angles provided)
            visibility: (17,) visibility tensor
            precomputed_angles: (12, 3) pre-computed joint angles from TRC

        Returns:
            Dict with 'refined_angles' and 'delta_angles' tensors
        """
        if precomputed_angles is None:
            # No pre-computed angles available - return zeros
            # Joint angles require augmented TRC (64 markers) which isn't available here
            return {
                'refined_angles': torch.zeros(12, 3),
                'delta_angles': torch.zeros(12, 3),
            }

        # Run joint model on pre-computed angles
        angles_b = precomputed_angles.unsqueeze(0).to(self.device)
        vis_12 = visibility[:12].unsqueeze(0).to(self.device)

        refined_angles, delta_angles = self.joint_model(angles_b, vis_12)

        return {
            'refined_angles': refined_angles.squeeze(0).cpu(),
            'delta_angles': delta_angles.squeeze(0).cpu(),
        }


class PrecomputedMainRefinerDataset(Dataset):
    """Dataset with pre-computed constraint model outputs.

    For efficiency, model outputs can be pre-computed once and saved to disk.
    This dataset loads those pre-computed outputs.
    """

    def __init__(
        self,
        data_dir: Union[str, Path],
        split: str = 'train',
        val_ratio: float = 0.1,
        seed: int = 42,
        max_samples: Optional[int] = None,
    ):
        """
        Initialize dataset.

        Args:
            data_dir: Path to pre-computed training data
            split: 'train' or 'val'
            val_ratio: Fraction of data for validation
            seed: Random seed for train/val split
            max_samples: Limit number of samples (for debugging)
        """
        self.data_dir = Path(data_dir)
        self.split = split

        # Find all NPZ files
        all_files = sorted(self.data_dir.glob('*.npz'))

        if max_samples:
            all_files = all_files[:max_samples]

        # Simple split (by file index, not sequence for pre-computed)
        rng = random.Random(seed)
        indices = list(range(len(all_files)))
        rng.shuffle(indices)

        n_val = max(1, int(len(indices) * val_ratio))

        if split == 'train':
            self.files = [all_files[i] for i in indices[n_val:]]
        else:
            self.files = [all_files[i] for i in indices[:n_val]]

        print(f"[{split}] Loaded {len(self.files)} pre-computed samples")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Load a pre-computed training sample."""
        data = np.load(self.files[idx])

        # Required fields
        result = {
            'raw_pose': torch.from_numpy(data['raw_pose'].astype(np.float32)),
            'ground_truth': torch.from_numpy(data['ground_truth'].astype(np.float32)),
            'visibility': torch.from_numpy(data['visibility'].astype(np.float32)),
        }

        # Depth model outputs
        result['depth_outputs'] = {
            'delta_xyz': torch.from_numpy(data['depth_delta_xyz'].astype(np.float32)),
            'confidence': torch.from_numpy(data['depth_confidence'].astype(np.float32)),
        }

        if 'depth_limb_orientations' in data:
            result['depth_outputs']['pred_limb_orientations'] = torch.from_numpy(
                data['depth_limb_orientations'].astype(np.float32)
            )

        if 'depth_azimuth' in data:
            result['depth_outputs']['pred_azimuth'] = torch.tensor(
                float(data['depth_azimuth']), dtype=torch.float32
            )
            result['depth_outputs']['pred_elevation'] = torch.tensor(
                float(data['depth_elevation']), dtype=torch.float32
            )

        # Joint model outputs
        result['joint_outputs'] = {
            'refined_angles': torch.from_numpy(data['joint_refined_angles'].astype(np.float32)),
            'delta_angles': torch.from_numpy(data['joint_delta_angles'].astype(np.float32)),
        }

        return result


def create_dataloaders(
    data_dir: Union[str, Path],
    batch_size: int = 64,
    num_workers: int = 4,
    val_ratio: float = 0.1,
    seed: int = 42,
    max_samples: Optional[int] = None,
    depth_model: Optional[torch.nn.Module] = None,
    joint_model: Optional[torch.nn.Module] = None,
    device: str = 'cpu',
) -> Tuple[DataLoader, DataLoader]:
    """Create train and validation dataloaders.

    Args:
        data_dir: Path to training data
        batch_size: Batch size
        num_workers: Number of data loading workers
        val_ratio: Validation split ratio
        seed: Random seed
        max_samples: Limit samples (for debugging)
        depth_model: Depth model for online inference
        joint_model: Joint model for online inference
        device: Device for model inference

    Returns:
        (train_loader, val_loader)
    """
    train_dataset = MainRefinerDataset(
        data_dir,
        depth_model=depth_model,
        joint_model=joint_model,
        split='train',
        val_ratio=val_ratio,
        seed=seed,
        max_samples=max_samples,
        device=device,
    )

    val_dataset = MainRefinerDataset(
        data_dir,
        depth_model=depth_model,
        joint_model=joint_model,
        split='val',
        val_ratio=val_ratio,
        seed=seed,
        max_samples=max_samples,
        device=device,
    )

    # Collate function to handle nested dicts
    def collate_fn(batch):
        result = {}
        for key in batch[0].keys():
            if isinstance(batch[0][key], dict):
                # Nested dict (depth_outputs, joint_outputs)
                result[key] = {}
                for subkey in batch[0][key].keys():
                    result[key][subkey] = torch.stack([b[key][subkey] for b in batch])
            else:
                result[key] = torch.stack([b[key] for b in batch])
        return result

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_fn,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    return train_loader, val_loader


if __name__ == '__main__':
    # Test dataset (without models)
    data_dir = Path('data/training/aistpp_converted')

    if not data_dir.exists():
        print(f"Data directory not found: {data_dir}")
        exit(1)

    # Test without models (just base data)
    dataset = MainRefinerDataset(data_dir, max_samples=100)

    print(f"\nDataset size: {len(dataset)}")

    sample = dataset[0]
    print("\nSample keys:")
    for key, value in sample.items():
        if isinstance(value, dict):
            print(f"  {key}:")
            for subkey, subvalue in value.items():
                print(f"    {subkey}: {subvalue.shape if hasattr(subvalue, 'shape') else subvalue}")
        else:
            print(f"  {key}: {value.shape if hasattr(value, 'shape') else value}")

    print("\nDataset test passed!")

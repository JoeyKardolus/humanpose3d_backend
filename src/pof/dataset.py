"""Training data loading for camera-space POF.

Uses existing AIST++ training data but extracts camera-space features:
- pose_2d: MediaPipe 2D detections (normalized to pelvis-centered, unit-torso)
- visibility: Per-joint confidence scores
- limb_delta_2d: 2D displacement vectors for each limb (foreshortening)
- limb_length_2d: 2D lengths for each limb (foreshortening magnitude)
- ground_truth: For computing GT POF vectors in camera space

COORDINATE SYSTEMS:
- AIST++ world: Y-up, Z-away, X-right (right-handed)
- Camera space: Y-down, Z-toward camera, X-right in image (OpenCV convention)
- The GT is stored in AIST++ world coords, rotated to camera view using camera_R
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Optional, Tuple, List, Union
import random
import re

from .constants import (
    LIMB_DEFINITIONS,
    NUM_LIMBS,
    NUM_JOINTS,
    JOINT_SWAP_PAIRS,
    LIMB_SWAP_PAIRS,
    LEFT_HIP_IDX,
    RIGHT_HIP_IDX,
    LEFT_SHOULDER_IDX,
    RIGHT_SHOULDER_IDX,
)


def world_to_camera_space(pose_3d: np.ndarray, camera_R: np.ndarray) -> np.ndarray:
    """Transform pose from AIST++ world coordinates to camera space.

    Uses the camera rotation matrix directly for accurate transformation.

    Args:
        pose_3d: (17, 3) pose in AIST++ world coords
        camera_R: (3, 3) camera rotation matrix (world → camera)

    Returns:
        (17, 3) pose in camera space
    """
    # Apply camera rotation: world → camera
    pose_camera = (camera_R @ pose_3d.T).T
    return pose_camera


def normalize_pose_2d(pose_2d: np.ndarray, eps: float = 1e-6) -> Tuple[np.ndarray, np.ndarray, float]:
    """Normalize 2D pose to pelvis-centered, unit-torso scale.

    This makes the model position-invariant and scale-invariant.

    Args:
        pose_2d: (17, 2) raw 2D coordinates (e.g., [0,1] range)
        eps: Small value to prevent division by zero

    Returns:
        pose_2d_norm: (17, 2) normalized pose
        pelvis_2d: (2,) pelvis position for denormalization
        torso_scale: scalar torso scale for denormalization
    """
    # Compute pelvis center (midpoint of hips)
    pelvis_2d = (pose_2d[LEFT_HIP_IDX] + pose_2d[RIGHT_HIP_IDX]) / 2

    # Center on pelvis
    centered = pose_2d - pelvis_2d

    # Compute torso scale (average of L/R shoulder-to-hip distance)
    l_torso = np.linalg.norm(pose_2d[LEFT_SHOULDER_IDX] - pose_2d[LEFT_HIP_IDX])
    r_torso = np.linalg.norm(pose_2d[RIGHT_SHOULDER_IDX] - pose_2d[RIGHT_HIP_IDX])
    torso_scale = (l_torso + r_torso) / 2

    # Scale to unit torso
    safe_scale = max(torso_scale, eps)
    pose_2d_norm = centered / safe_scale

    return pose_2d_norm.astype(np.float32), pelvis_2d.astype(np.float32), float(torso_scale)


def compute_limb_features_2d(pose_2d: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute 2D limb features from normalized pose.

    For each limb, compute:
    - delta_2d: 2D displacement vector from parent to child (normalized)
    - length_2d: 2D length (foreshortening indicator)

    Args:
        pose_2d: (17, 2) normalized 2D pose

    Returns:
        limb_delta_2d: (14, 2) normalized 2D displacement unit vectors
        limb_length_2d: (14,) 2D lengths
    """
    limb_delta_2d = np.zeros((NUM_LIMBS, 2), dtype=np.float32)
    limb_length_2d = np.zeros(NUM_LIMBS, dtype=np.float32)

    for limb_idx, (parent, child) in enumerate(LIMB_DEFINITIONS):
        delta = pose_2d[child] - pose_2d[parent]
        length = np.linalg.norm(delta)
        limb_length_2d[limb_idx] = length

        # Normalize to unit vector (handle zero length)
        if length > 1e-6:
            limb_delta_2d[limb_idx] = delta / length
        else:
            limb_delta_2d[limb_idx] = np.array([0.0, 0.0])

    return limb_delta_2d, limb_length_2d


def compute_gt_pof_from_3d(pose_3d: np.ndarray) -> np.ndarray:
    """Compute ground truth POF unit vectors from 3D pose.

    POF is the unit vector from parent joint to child joint
    for each limb, computed in camera space.

    Args:
        pose_3d: (17, 3) or (batch, 17, 3) COCO joint positions

    Returns:
        (14, 3) or (batch, 14, 3) unit vectors
    """
    single_pose = pose_3d.ndim == 2
    if single_pose:
        pose_3d = pose_3d[np.newaxis, ...]

    batch_size = pose_3d.shape[0]
    pof = np.zeros((batch_size, NUM_LIMBS, 3), dtype=np.float32)

    for limb_idx, (parent, child) in enumerate(LIMB_DEFINITIONS):
        vec = pose_3d[:, child] - pose_3d[:, parent]  # (batch, 3)
        length = np.linalg.norm(vec, axis=-1, keepdims=True)  # (batch, 1)
        # Normalize to unit vector, avoid division by zero
        pof[:, limb_idx] = vec / np.maximum(length, 1e-6)

    if single_pose:
        pof = pof[0]

    return pof


def compute_gt_pof_from_3d_torch(pose_3d: torch.Tensor) -> torch.Tensor:
    """PyTorch version of compute_gt_pof_from_3d.

    Args:
        pose_3d: (batch, 17, 3) COCO joint positions

    Returns:
        (batch, 14, 3) unit vectors
    """
    batch_size = pose_3d.size(0)
    device = pose_3d.device
    dtype = pose_3d.dtype

    pof = torch.zeros(batch_size, NUM_LIMBS, 3, device=device, dtype=dtype)

    for limb_idx, (parent, child) in enumerate(LIMB_DEFINITIONS):
        vec = pose_3d[:, child] - pose_3d[:, parent]  # (batch, 3)
        pof[:, limb_idx] = torch.nn.functional.normalize(vec, dim=-1, eps=1e-6)

    return pof


class CameraPOFDataset(Dataset):
    """Dataset for camera-space POF training.

    Uses existing AIST++ training data (from depth refinement).
    All 2D inputs are normalized to pelvis-centered, unit-torso scale.

    Returns:
    - pose_2d: (17, 2) normalized 2D coordinates (pelvis-centered, unit-torso)
    - visibility: (17,) per-joint confidence
    - limb_delta_2d: (14, 2) normalized 2D displacement vectors (foreshortening direction)
    - limb_length_2d: (14,) 2D lengths (foreshortening magnitude)
    - gt_pof: (14, 3) ground truth POF unit vectors in camera space
    """

    def __init__(
        self,
        data_dir: Union[str, Path, List[str]],
        split: str = "train",
        val_ratio: float = 0.1,
        seed: int = 42,
        augment: bool = True,
        max_samples: Optional[int] = None,
    ):
        """Initialize dataset.

        Args:
            data_dir: Path or comma-separated paths to training data
            split: 'train' or 'val'
            val_ratio: Fraction of sequences for validation
            seed: Random seed for reproducible splits
            augment: Enable data augmentation (train only)
            max_samples: Limit total samples (for debugging)
        """
        self.split = split
        self.augment = augment and (split == "train")

        # Support multiple data directories
        if isinstance(data_dir, str) and "," in data_dir:
            data_dirs = [Path(d.strip()) for d in data_dir.split(",")]
        elif isinstance(data_dir, list):
            data_dirs = [Path(d) for d in data_dir]
        else:
            data_dirs = [Path(data_dir)]

        # Find all NPZ files
        all_files = []
        for d in data_dirs:
            if d.exists():
                for f in sorted(d.glob("*.npz")):
                    if f.stat().st_size > 0:
                        all_files.append(f)

        if max_samples:
            all_files = all_files[:max_samples]

        # Split by sequence to avoid data leakage
        sequences = set(self._get_sequence(f) for f in all_files)
        sequences = sorted(sequences)

        rng = random.Random(seed)
        rng.shuffle(sequences)

        n_val = max(1, int(len(sequences) * val_ratio))
        val_sequences = set(sequences[:n_val])
        train_sequences = set(sequences[n_val:])

        if split == "train":
            self.files = [
                f for f in all_files
                if self._get_sequence(f) in train_sequences
            ]
        else:
            self.files = [
                f for f in all_files
                if self._get_sequence(f) in val_sequences
            ]

        print(f"[CameraPOFDataset {split}] Loaded {len(self.files)} samples")

    @staticmethod
    def _get_sequence(path: Path) -> str:
        """Extract sequence identifier from filename."""
        stem = path.stem
        match = re.match(r"^(.+)_f\d+", stem)
        if match:
            return match.group(1)
        return "_".join(stem.split("_")[:-1])

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> dict:
        """Load and process a single sample.

        Returns dict with:
        - pose_2d: (17, 2) normalized 2D coordinates (pelvis-centered, unit-torso)
        - visibility: (17,) confidence scores
        - limb_delta_2d: (14, 2) normalized 2D displacement unit vectors
        - limb_length_2d: (14,) 2D lengths (foreshortening)
        - gt_pof: (14, 3) ground truth POF unit vectors in camera space
        """
        data = np.load(self.files[idx])

        # Load required fields
        pose_2d_raw = data["pose_2d"].astype(np.float32)
        visibility = data["visibility"].astype(np.float32)
        ground_truth = data["ground_truth"].astype(np.float32)
        camera_R = data["camera_R"].astype(np.float32)

        # Normalize 2D pose (pelvis-centered, unit-torso)
        pose_2d, _, _ = normalize_pose_2d(pose_2d_raw)

        # Compute 2D limb features (foreshortening)
        limb_delta_2d, limb_length_2d = compute_limb_features_2d(pose_2d)

        # Transform GT from AIST++ world coords to camera space
        ground_truth_camera = world_to_camera_space(ground_truth, camera_R)

        # Compute GT POF from 3D ground truth in camera space
        gt_pof = compute_gt_pof_from_3d(ground_truth_camera)

        # Data augmentation (horizontal flip)
        if self.augment and random.random() > 0.5:
            pose_2d, visibility, limb_delta_2d, limb_length_2d, gt_pof = self._flip_augment(
                pose_2d, visibility, limb_delta_2d, limb_length_2d, gt_pof
            )

        return {
            "pose_2d": torch.from_numpy(pose_2d),
            "visibility": torch.from_numpy(visibility),
            "limb_delta_2d": torch.from_numpy(limb_delta_2d),
            "limb_length_2d": torch.from_numpy(limb_length_2d),
            "gt_pof": torch.from_numpy(gt_pof),
        }

    def _flip_augment(
        self,
        pose_2d: np.ndarray,
        visibility: np.ndarray,
        limb_delta_2d: np.ndarray,
        limb_length_2d: np.ndarray,
        gt_pof: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Horizontal flip augmentation.

        Flips X coordinates and swaps left/right joints and limbs.
        Since pose is pelvis-centered, we just negate X, no need to shift.
        """
        # Flip X coordinate (negate since pelvis-centered)
        pose_2d = pose_2d.copy()
        pose_2d[:, 0] = -pose_2d[:, 0]

        # Swap left/right joints
        visibility = visibility.copy()
        for i, j in JOINT_SWAP_PAIRS:
            pose_2d[[i, j]] = pose_2d[[j, i]]
            visibility[[i, j]] = visibility[[j, i]]

        # Flip limb 2D features
        limb_delta_2d = limb_delta_2d.copy()
        limb_delta_2d[:, 0] = -limb_delta_2d[:, 0]  # Flip X direction

        limb_length_2d = limb_length_2d.copy()

        # Swap left/right limbs
        for i, j in LIMB_SWAP_PAIRS:
            limb_delta_2d[[i, j]] = limb_delta_2d[[j, i]]
            limb_length_2d[[i, j]] = limb_length_2d[[j, i]]

        # Flip POF X component and swap left/right limbs
        gt_pof = gt_pof.copy()
        gt_pof[:, 0] = -gt_pof[:, 0]  # Flip X direction
        for i, j in LIMB_SWAP_PAIRS:
            gt_pof[[i, j]] = gt_pof[[j, i]]

        return pose_2d, visibility, limb_delta_2d, limb_length_2d, gt_pof


def create_pof_dataloaders(
    data_dir: Union[str, Path],
    batch_size: int = 256,
    num_workers: int = 4,
    val_ratio: float = 0.1,
    seed: int = 42,
    max_samples: Optional[int] = None,
) -> Tuple[DataLoader, DataLoader]:
    """Create train and validation dataloaders.

    Args:
        data_dir: Path to training data directory
        batch_size: Batch size for both loaders
        num_workers: Number of data loading workers
        val_ratio: Fraction of sequences for validation
        seed: Random seed for splits
        max_samples: Limit samples (for debugging)

    Returns:
        (train_loader, val_loader) tuple
    """
    train_dataset = CameraPOFDataset(
        data_dir,
        split="train",
        val_ratio=val_ratio,
        seed=seed,
        augment=True,
        max_samples=max_samples,
    )

    val_dataset = CameraPOFDataset(
        data_dir,
        split="val",
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

"""Anatomical bone length estimation from body height.

Estimates bone lengths using anthropometric proportions based on
total body height. This eliminates the need for MediaPipe 3D
coordinates to determine bone lengths.

References:
- Winter, D.A. (2009). Biomechanics and Motor Control of Human Movement
- NASA Anthropometric Source Book (1978)
- Drillis & Contini (1966). Body Segment Parameters
"""

from dataclasses import dataclass
from typing import Dict, Optional
import numpy as np

from .constants import NUM_LIMBS, LIMB_TO_BONE


@dataclass
class AnatomicalProportions:
    """Bone lengths as proportions of total body height.

    Default values are based on anthropometric studies and represent
    average adult proportions. Values are fractions of standing height.

    Attributes:
        upper_arm: Shoulder to elbow length
        forearm: Elbow to wrist length
        thigh: Hip to knee length
        shin: Knee to ankle length
        torso_side: Same-side shoulder to hip distance
        shoulder_width: Left to right shoulder distance
        hip_width: Left to right hip distance
        cross_torso: Diagonal shoulder to opposite hip distance
    """
    # Upper limb proportions
    upper_arm: float = 0.172      # ~0.17 * height
    forearm: float = 0.157        # ~0.16 * height

    # Lower limb proportions
    thigh: float = 0.245          # ~0.25 * height
    shin: float = 0.246           # ~0.25 * height

    # Torso proportions
    torso_side: float = 0.288     # ~0.29 * height (shoulder to hip, same side)
    shoulder_width: float = 0.259 # ~0.26 * height (biacromial width)
    hip_width: float = 0.191      # ~0.19 * height (bi-iliac width)
    cross_torso: float = 0.345    # ~0.35 * height (diagonal)


# Default proportions for general use
DEFAULT_PROPORTIONS = AnatomicalProportions()


def estimate_bone_lengths_from_height(
    height_m: float,
    proportions: Optional[AnatomicalProportions] = None,
) -> Dict[int, float]:
    """Estimate bone lengths for all 14 limbs from body height.

    Args:
        height_m: Subject standing height in meters
        proportions: Custom proportions (uses defaults if None)

    Returns:
        Dictionary mapping limb index (0-13) to bone length in meters
    """
    if proportions is None:
        proportions = DEFAULT_PROPORTIONS

    # Map limb indices to corresponding bone proportions
    # Order matches LIMB_DEFINITIONS in constants.py
    return {
        0: height_m * proportions.upper_arm,      # L upper arm
        1: height_m * proportions.forearm,        # L forearm
        2: height_m * proportions.upper_arm,      # R upper arm (symmetric)
        3: height_m * proportions.forearm,        # R forearm (symmetric)
        4: height_m * proportions.thigh,          # L thigh
        5: height_m * proportions.shin,           # L shin
        6: height_m * proportions.thigh,          # R thigh (symmetric)
        7: height_m * proportions.shin,           # R shin (symmetric)
        8: height_m * proportions.shoulder_width, # Shoulder width
        9: height_m * proportions.hip_width,      # Hip width
        10: height_m * proportions.torso_side,    # L torso
        11: height_m * proportions.torso_side,    # R torso (symmetric)
        12: height_m * proportions.cross_torso,   # L cross-body diagonal
        13: height_m * proportions.cross_torso,   # R cross-body diagonal (symmetric)
    }


def bone_lengths_to_array(bone_lengths: Dict[int, float]) -> np.ndarray:
    """Convert bone lengths dictionary to numpy array.

    Args:
        bone_lengths: Dictionary mapping limb index to length

    Returns:
        (14,) array of bone lengths
    """
    result = np.zeros(NUM_LIMBS, dtype=np.float32)
    for limb_idx, length in bone_lengths.items():
        result[limb_idx] = length
    return result


def array_to_bone_lengths(bone_array: np.ndarray) -> Dict[int, float]:
    """Convert numpy array to bone lengths dictionary.

    Args:
        bone_array: (14,) array of bone lengths

    Returns:
        Dictionary mapping limb index to length
    """
    return {i: float(bone_array[i]) for i in range(NUM_LIMBS)}


def estimate_bone_lengths_array(
    height_m: float,
    proportions: Optional[AnatomicalProportions] = None,
) -> np.ndarray:
    """Convenience function to get bone lengths as numpy array.

    Args:
        height_m: Subject standing height in meters
        proportions: Custom proportions (uses defaults if None)

    Returns:
        (14,) array of bone lengths in meters
    """
    bone_dict = estimate_bone_lengths_from_height(height_m, proportions)
    return bone_lengths_to_array(bone_dict)


def compute_bone_lengths_from_pose(
    pose_3d: np.ndarray,
    window_size: int = 50,
) -> np.ndarray:
    """Compute bone lengths from actual 3D pose data.

    Useful for comparison or when actual pose data is available.
    Uses median over window_size frames for stability.

    Args:
        pose_3d: (N, 17, 3) or (17, 3) 3D joint positions
        window_size: Number of frames to median over (if N > 1)

    Returns:
        (14,) array of bone lengths
    """
    from .constants import LIMB_DEFINITIONS

    single_frame = pose_3d.ndim == 2
    if single_frame:
        pose_3d = pose_3d[np.newaxis, ...]

    n_frames = pose_3d.shape[0]
    bone_lengths = np.zeros((n_frames, NUM_LIMBS), dtype=np.float32)

    for limb_idx, (parent, child) in enumerate(LIMB_DEFINITIONS):
        vec = pose_3d[:, child] - pose_3d[:, parent]
        bone_lengths[:, limb_idx] = np.linalg.norm(vec, axis=-1)

    # Use median for stability
    if n_frames > 1:
        use_frames = min(window_size, n_frames)
        result = np.median(bone_lengths[:use_frames], axis=0)
    else:
        result = bone_lengths[0]

    return result


def validate_bone_lengths(
    bone_lengths: np.ndarray,
    height_m: float,
    tolerance: float = 0.3,
) -> bool:
    """Validate that bone lengths are anatomically reasonable.

    Checks if bone lengths are within tolerance of expected
    anthropometric proportions.

    Args:
        bone_lengths: (14,) array of bone lengths in meters
        height_m: Expected body height in meters
        tolerance: Maximum allowed fractional deviation (0.3 = 30%)

    Returns:
        True if all bone lengths are within tolerance
    """
    expected = estimate_bone_lengths_array(height_m)

    for i in range(NUM_LIMBS):
        if expected[i] > 0:
            deviation = abs(bone_lengths[i] - expected[i]) / expected[i]
            if deviation > tolerance:
                return False

    return True


def scale_bone_lengths(
    bone_lengths: np.ndarray,
    scale_factor: float,
) -> np.ndarray:
    """Scale all bone lengths by a factor.

    Useful for adjusting bone lengths while maintaining proportions.

    Args:
        bone_lengths: (14,) array of bone lengths
        scale_factor: Multiplicative scale factor

    Returns:
        (14,) scaled bone lengths
    """
    return bone_lengths * scale_factor


def estimate_height_from_bone_lengths(
    bone_lengths: np.ndarray,
    proportions: Optional[AnatomicalProportions] = None,
) -> float:
    """Estimate body height from measured bone lengths.

    Uses average of multiple bone length estimates for robustness.

    Args:
        bone_lengths: (14,) array of bone lengths in meters
        proportions: Proportions to use for estimation

    Returns:
        Estimated body height in meters
    """
    if proportions is None:
        proportions = DEFAULT_PROPORTIONS

    # Estimate height from different bones and average
    estimates = []

    # From thighs (most reliable)
    if bone_lengths[4] > 0:
        estimates.append(bone_lengths[4] / proportions.thigh)
    if bone_lengths[6] > 0:
        estimates.append(bone_lengths[6] / proportions.thigh)

    # From shins
    if bone_lengths[5] > 0:
        estimates.append(bone_lengths[5] / proportions.shin)
    if bone_lengths[7] > 0:
        estimates.append(bone_lengths[7] / proportions.shin)

    # From torso
    if bone_lengths[10] > 0:
        estimates.append(bone_lengths[10] / proportions.torso_side)
    if bone_lengths[11] > 0:
        estimates.append(bone_lengths[11] / proportions.torso_side)

    if not estimates:
        return 1.7  # Default fallback

    return float(np.median(estimates))

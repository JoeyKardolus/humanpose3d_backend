"""Facing direction detection from visibility scores.

Determines the facing direction of a person based on ear/nose visibility
from MediaPipe detections. This replaces the need for azimuth/elevation
prediction in camera-space POF.
"""

from enum import Enum
from typing import Union
import numpy as np

try:
    import torch
except ImportError:
    torch = None

from .constants import NOSE_IDX, LEFT_EAR_IDX, RIGHT_EAR_IDX


class FacingDirection(Enum):
    """Facing direction of a person relative to the camera.

    Values correspond to one-hot encoding indices.
    """
    FRONTAL = 0       # Person facing toward camera
    BACK = 1          # Person facing away from camera
    CAMERA_LEFT = 2   # Person's right side visible (facing camera's left)
    CAMERA_RIGHT = 3  # Person's left side visible (facing camera's right)


def detect_facing_direction(
    visibility: np.ndarray,
    nose_threshold: float = 0.5,
    ear_threshold: float = 0.3,
) -> FacingDirection:
    """Detect facing direction from nose and ear visibility scores.

    NOTE: This visibility-based detection is unreliable because MediaPipe
    often gives high visibility even for back views. Consider using
    detect_facing_from_pose() instead for more reliable results.

    Logic:
    - Nose not visible -> BACK (person facing away)
    - Left ear visible, right ear not -> CAMERA_RIGHT (person's left side toward camera)
    - Right ear visible, left ear not -> CAMERA_LEFT (person's right side toward camera)
    - Both ears visible or neither clear -> FRONTAL (default)

    Args:
        visibility: (17,) per-joint visibility/confidence scores (0-1)
        nose_threshold: Minimum visibility for nose to be considered visible
        ear_threshold: Minimum visibility for ears to be considered visible

    Returns:
        FacingDirection enum value
    """
    nose_vis = visibility[NOSE_IDX]
    left_ear_vis = visibility[LEFT_EAR_IDX]
    right_ear_vis = visibility[RIGHT_EAR_IDX]

    nose_visible = nose_vis > nose_threshold
    left_ear_visible = left_ear_vis > ear_threshold
    right_ear_visible = right_ear_vis > ear_threshold

    # If nose not visible, person is facing away
    if not nose_visible:
        return FacingDirection.BACK

    # Check ear visibility for profile views
    if left_ear_visible and not right_ear_visible:
        # Left ear visible means person's left side is toward camera
        # From camera's perspective, person is facing camera's right
        return FacingDirection.CAMERA_RIGHT

    if right_ear_visible and not left_ear_visible:
        # Right ear visible means person's right side is toward camera
        # From camera's perspective, person is facing camera's left
        return FacingDirection.CAMERA_LEFT

    # Default to frontal (nose visible, both ears similar)
    return FacingDirection.FRONTAL


def detect_facing_from_pose(
    pose_2d: np.ndarray,
    shoulder_threshold: float = 0.02,
) -> FacingDirection:
    """Detect facing direction from 2D pose shoulder positions.

    More reliable than visibility-based detection because it uses actual
    joint positions rather than MediaPipe's unreliable visibility scores.

    Logic:
    - L_shoulder.x > R_shoulder.x by threshold -> BACK (we see their back)
    - L_shoulder.x < R_shoulder.x by threshold -> FRONTAL (we see their front)
    - Similar x positions -> check nose position for profile views

    In standard image coordinates:
    - FRONTAL: person's left shoulder appears on RIGHT of image (lower x)
    - BACK: person's left shoulder appears on LEFT of image (higher x)

    Args:
        pose_2d: (17, 2) normalized 2D keypoint positions
        shoulder_threshold: Minimum x difference to determine front/back

    Returns:
        FacingDirection enum value
    """
    from .constants import LEFT_SHOULDER_IDX, RIGHT_SHOULDER_IDX

    l_shoulder_x = pose_2d[LEFT_SHOULDER_IDX, 0]
    r_shoulder_x = pose_2d[RIGHT_SHOULDER_IDX, 0]

    x_diff = l_shoulder_x - r_shoulder_x

    if x_diff > shoulder_threshold:
        # Left shoulder is to the RIGHT of right shoulder in image
        # This means we see the person's back
        return FacingDirection.BACK
    elif x_diff < -shoulder_threshold:
        # Left shoulder is to the LEFT of right shoulder in image
        # This means we see the person's front
        return FacingDirection.FRONTAL
    else:
        # Shoulders at similar x -> profile view
        # Check if nose is to the left or right of shoulder midpoint
        nose_x = pose_2d[NOSE_IDX, 0]
        shoulder_mid_x = (l_shoulder_x + r_shoulder_x) / 2

        if nose_x < shoulder_mid_x - 0.02:
            return FacingDirection.CAMERA_LEFT
        elif nose_x > shoulder_mid_x + 0.02:
            return FacingDirection.CAMERA_RIGHT
        else:
            # Can't determine clearly, default to frontal
            return FacingDirection.FRONTAL


def facing_to_one_hot(facing: FacingDirection) -> np.ndarray:
    """Convert facing direction enum to one-hot encoding.

    Args:
        facing: FacingDirection enum value

    Returns:
        (4,) one-hot encoded array
    """
    one_hot = np.zeros(4, dtype=np.float32)
    one_hot[facing.value] = 1.0
    return one_hot


def one_hot_to_facing(one_hot: np.ndarray) -> FacingDirection:
    """Convert one-hot encoding back to FacingDirection enum.

    Args:
        one_hot: (4,) one-hot encoded array

    Returns:
        FacingDirection enum value
    """
    idx = int(np.argmax(one_hot))
    return FacingDirection(idx)


def detect_facing_direction_batch(
    visibility: np.ndarray,
    nose_threshold: float = 0.5,
    ear_threshold: float = 0.3,
) -> np.ndarray:
    """Batch version of detect_facing_direction.

    Args:
        visibility: (batch, 17) per-joint visibility scores
        nose_threshold: Minimum visibility for nose
        ear_threshold: Minimum visibility for ears

    Returns:
        (batch,) integer array of facing direction indices
    """
    batch_size = visibility.shape[0]
    result = np.zeros(batch_size, dtype=np.int64)

    for i in range(batch_size):
        facing = detect_facing_direction(
            visibility[i], nose_threshold, ear_threshold
        )
        result[i] = facing.value

    return result


def facing_batch_to_one_hot(facing_indices: np.ndarray) -> np.ndarray:
    """Convert batch of facing direction indices to one-hot encodings.

    Args:
        facing_indices: (batch,) integer array of facing direction indices

    Returns:
        (batch, 4) one-hot encoded array
    """
    batch_size = facing_indices.shape[0]
    one_hot = np.zeros((batch_size, 4), dtype=np.float32)
    for i in range(batch_size):
        one_hot[i, facing_indices[i]] = 1.0
    return one_hot


if torch is not None:
    def facing_to_one_hot_torch(facing: "torch.Tensor") -> "torch.Tensor":
        """Convert batch of facing direction indices to one-hot (PyTorch).

        Args:
            facing: (batch,) integer tensor of facing direction indices

        Returns:
            (batch, 4) one-hot encoded tensor
        """
        batch_size = facing.size(0)
        one_hot = torch.zeros(
            batch_size, 4, device=facing.device, dtype=torch.float32
        )
        one_hot.scatter_(1, facing.unsqueeze(1), 1.0)
        return one_hot


def interpret_forward_direction(facing: FacingDirection) -> np.ndarray:
    """Get the 'forward' direction vector in camera space for a facing direction.

    This tells us which way the person is facing in camera coordinates.
    Useful for understanding POF predictions in context.

    Camera space convention:
    - X: positive = right in image
    - Y: positive = down in image
    - Z: positive = away from camera (into scene)

    Args:
        facing: FacingDirection enum value

    Returns:
        (3,) unit vector representing forward direction in camera space
    """
    if facing == FacingDirection.FRONTAL:
        # Person facing toward camera (negative Z)
        return np.array([0.0, 0.0, -1.0], dtype=np.float32)
    elif facing == FacingDirection.BACK:
        # Person facing away from camera (positive Z)
        return np.array([0.0, 0.0, 1.0], dtype=np.float32)
    elif facing == FacingDirection.CAMERA_LEFT:
        # Person facing camera's left (negative X)
        return np.array([-1.0, 0.0, 0.0], dtype=np.float32)
    elif facing == FacingDirection.CAMERA_RIGHT:
        # Person facing camera's right (positive X)
        return np.array([1.0, 0.0, 0.0], dtype=np.float32)
    else:
        # Default to frontal
        return np.array([0.0, 0.0, -1.0], dtype=np.float32)

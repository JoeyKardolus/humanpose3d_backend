"""Anatomical segment coordinate system builders.

Builds right-handed orthonormal coordinate systems for body segments
following ISB (International Society of Biomechanics) recommendations
where applicable.

Coordinate system conventions:
- Pelvis: X=anterior, Y=superior, Z=right
- Femur: X=anterior, Y=proximal->distal, Z=lateral
- Tibia: X=anterior, Y=proximal->distal, Z=lateral
- Foot: X=anterior (heel->toe), Y=superior, Z=lateral
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np


def normalize(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Normalize a vector with safe handling of near-zero norms.

    Args:
        v: Input vector (3,)
        eps: Minimum norm threshold

    Returns:
        Normalized vector or NaN array if norm too small
    """
    v = np.asarray(v, dtype=float)
    norm = np.linalg.norm(v)
    if not np.isfinite(norm) or norm < eps:
        return np.full(3, np.nan, dtype=float)
    return v / norm


def build_orthonormal_frame(x_hint: np.ndarray, y_hint: np.ndarray) -> np.ndarray:
    """Build right-handed orthonormal coordinate system.

    Strategy:
    1. X axis aligned with x_hint
    2. Y axis in plane of x_hint and y_hint (orthogonalized)
    3. Z = X × Y (right-handed)
    4. Re-orthogonalize Y = Z × X for numerical stability
    5. Verify determinant = +1 (right-handed)

    Args:
        x_hint: Desired X axis direction (3,)
        y_hint: Hint for Y axis direction (3,)

    Returns:
        Rotation matrix (3x3) with axes as columns: [X, Y, Z]
    """
    x = normalize(x_hint)

    # Y orthogonal to X
    y_temp = y_hint - np.dot(y_hint, x) * x
    y_temp = normalize(y_temp)

    # Z perpendicular to X-Y plane
    z = normalize(np.cross(x, y_temp))

    # Re-orthogonalize Y for numerical stability
    y = normalize(np.cross(z, x))

    # Check for NaN axes
    if np.isnan(x).any() or np.isnan(y).any() or np.isnan(z).any():
        return np.full((3, 3), np.nan, dtype=float)

    result = np.column_stack([x, y, z])

    # Verify right-handedness (det = +1)
    det = np.linalg.det(result)
    if det < 0:
        # Left-handed system - flip Z axis to fix
        z = -z
        y = normalize(np.cross(z, x))
        result = np.column_stack([x, y, z])

    return result


def ensure_continuity(
    current_axes: np.ndarray,
    previous_axes: Optional[np.ndarray]
) -> np.ndarray:
    """Ensure coordinate system continuity across frames.

    DISABLED: The original implementation flipped all axes when score < 0,
    which changed det(R) from +1 to -1, creating left-handed coordinate
    systems. This caused scipy's Rotation.from_matrix to fail or produce
    incorrect Euler angles.

    Instead, we now return the axes unchanged and rely on Euler angle
    unwrapping (in angle_processing.py) to handle any discontinuities.
    This maintains valid right-handed coordinate systems throughout.

    Args:
        current_axes: Current frame axes (3x3)
        previous_axes: Previous frame axes (3x3) or None

    Returns:
        Current axes unchanged (3x3)
    """
    # Simply return current axes - rely on Euler angle unwrapping instead
    return current_axes


def markers_valid(*markers) -> bool:
    """Check all markers exist and are finite.

    Args:
        *markers: Variable number of marker arrays to check

    Returns:
        True if all markers are not None and contain finite values
    """
    return all(m is not None and np.isfinite(m).all() for m in markers)


def align_with_reference(axis: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Flip axis if pointing opposite to reference direction.

    Args:
        axis: Vector to potentially flip (3,)
        reference: Reference direction (3,)

    Returns:
        Axis pointing in same general direction as reference
    """
    return -axis if np.dot(axis, reference) < 0 else axis


def build_segment_frame(
    long_axis: np.ndarray,
    lateral_hint: np.ndarray,
    reference_lateral: np.ndarray,
    negate_long_axis: bool = True,
    previous: Optional[np.ndarray] = None,
) -> Optional[np.ndarray]:
    """Build standard segment frame: X=anterior, Y=along bone (distal), Z=lateral.

    This is the common pattern for limb segments (femur, tibia, humerus, forearm, trunk).
    Uses build_orthonormal_frame() with long_axis as primary, then reorders axes to
    standard ISB convention.

    Args:
        long_axis: Vector along the bone (typically proximal→distal direction)
        lateral_hint: Vector pointing roughly lateral (e.g., epicondyle line)
        reference_lateral: Reference lateral direction (e.g., pelvis Z) for alignment
        negate_long_axis: If True, negate Y axis so it points distally (default for limbs).
                          Set False for trunk where Y should point superiorly.
        previous: Previous frame for continuity check (3x3) or None

    Returns:
        Rotation matrix (3x3) with columns [X, Y, Z] or None if invalid
    """
    # Align lateral hint with reference direction
    z_hint = align_with_reference(lateral_hint, reference_lateral)

    # Build orthonormal frame with long axis as primary (X), lateral as secondary (Y)
    axes = build_orthonormal_frame(long_axis, z_hint)

    if np.isnan(axes).any():
        return None

    # Reorder to ISB convention: X=anterior, Y=long axis, Z=lateral
    # build_orthonormal_frame returns: col0=X (long_axis), col1=Y (orthog lateral), col2=Z (cross)
    x = axes[:, 2]  # anterior (cross product result)
    y = -axes[:, 0] if negate_long_axis else axes[:, 0]  # along bone
    z = axes[:, 1]  # lateral (re-orthogonalized)

    result = np.column_stack([x, y, z])

    # Verify right-handedness after reordering
    # Negating Y can create left-handed system - fix by flipping X
    det = np.linalg.det(result)
    if det < 0:
        x = -x
        result = np.column_stack([x, y, z])

    return ensure_continuity(result, previous)


def pelvis_axes(
    rasis: Optional[np.ndarray],
    lasis: Optional[np.ndarray],
    rpsis: Optional[np.ndarray],
    lpsis: Optional[np.ndarray],
    previous: Optional[np.ndarray] = None,
) -> Optional[np.ndarray]:
    """Build pelvis anatomical coordinate system.

    ISB-inspired pelvis frame following Wu et al. 2002:
    - X: Anterior (cross product of Y×Z, points forward)
    - Y: Superior (upward direction)
    - Z: Right (LASIS -> RASIS, medial-lateral axis)

    ROBUSTNESS FIX: Pose2Sim LSTM can output ASIS/PSIS markers at incorrect
    relative heights, which would make the ISB-standard PSIS→ASIS direction
    point downward instead of upward. To handle this, we use world-up (0,1,0)
    as the primary Y reference and only use ASIS markers for the lateral axis.

    Args:
        rasis: Right ASIS position (3,) or None
        lasis: Left ASIS position (3,) or None
        rpsis: Right PSIS position (3,) or None (used for anterior/posterior)
        lpsis: Left PSIS position (3,) or None (used for anterior/posterior)
        previous: Previous frame axes for continuity (3x3) or None

    Returns:
        Rotation matrix (3x3) with columns [X, Y, Z] or None if invalid markers

    Note: pelvis_axes uses a custom construction that prioritizes correct
    orientation over strict ISB marker-based definition to handle noisy
    marker data from ML-based augmentation.
    """
    if not markers_valid(rasis, lasis, rpsis, lpsis):
        return None

    asis_mid = 0.5 * (rasis + lasis)
    psis_mid = 0.5 * (rpsis + lpsis)

    # Z axis: medial-lateral (pointing to subject's right)
    # From left ASIS to right ASIS
    z_raw = normalize(rasis - lasis)

    # World up reference (in Y-up coordinate system)
    world_up = np.array([0.0, 1.0, 0.0])

    # X axis: anterior direction
    # Use ASIS/PSIS for anterior hint, then orthogonalize
    # ASIS should be anterior to PSIS (more negative Z in camera space for person facing camera)
    anterior_hint = normalize(asis_mid - psis_mid)

    # Build coordinate system with world-up as Y reference
    # X = component of anterior_hint orthogonal to world_up, then orthogonal to Z
    # First, get a robust Z by making it orthogonal to world_up
    z_temp = z_raw - np.dot(z_raw, world_up) * world_up
    z = normalize(z_temp)

    # Y = world_up component orthogonal to Z
    y_temp = world_up - np.dot(world_up, z) * z
    y = normalize(y_temp)

    # X = Y × Z (right-handed)
    x = normalize(np.cross(y, z))

    # Check for NaN axes
    if np.isnan(x).any() or np.isnan(y).any() or np.isnan(z).any():
        return None

    # Build rotation matrix with axes as columns
    result = np.column_stack([x, y, z])

    # Verify right-handedness (det = +1)
    det = np.linalg.det(result)
    if det < 0:
        # Left-handed system - flip Z to fix
        z = -z
        x = normalize(np.cross(y, z))
        result = np.column_stack([x, y, z])

    # Ensure X points forward (anterior) using ASIS/PSIS hint
    # For person facing camera, anterior = negative Z direction (toward camera)
    # If X·anterior_hint < 0, X is pointing backward, flip X and Z
    if np.dot(x, anterior_hint) < 0:
        x = -x
        z = -z
        result = np.column_stack([x, y, z])

    # Axis continuity check (matching reference implementation)
    # If all axes point in opposite direction from previous frame, flip them
    # This prevents 180° discontinuities in Euler angles
    if previous is not None and not np.isnan(previous).any():
        score = (
            np.dot(result[:, 0], previous[:, 0]) +  # X similarity
            np.dot(result[:, 1], previous[:, 1]) +  # Y similarity
            np.dot(result[:, 2], previous[:, 2])    # Z similarity
        )
        # If score < 0, all axes point opposite direction -> flip all
        if score < 0:
            result = -result

    return result


def femur_axes(
    hjc: Optional[np.ndarray],
    lateral_knee: Optional[np.ndarray],
    medial_knee: Optional[np.ndarray],
    pelvis_z: np.ndarray,
    previous: Optional[np.ndarray] = None,
) -> Optional[np.ndarray]:
    """Build femur anatomical coordinate system.

    Femur frame: X=anterior, Y=distal, Z=lateral (ISB convention)

    Args:
        hjc: Hip joint center position (3,) or None
        lateral_knee: Lateral knee condyle position (3,) or None
        medial_knee: Medial knee condyle position (3,) or None
        pelvis_z: Pelvis Z axis for lateral alignment (3,)
        previous: Previous frame axes (3x3) or None

    Returns:
        Rotation matrix (3x3) or None if invalid
    """
    if not markers_valid(hjc, lateral_knee, medial_knee):
        return None

    knee_mid = 0.5 * (lateral_knee + medial_knee)
    return build_segment_frame(
        long_axis=hjc - knee_mid,
        lateral_hint=lateral_knee - medial_knee,
        reference_lateral=pelvis_z,
        negate_long_axis=True,
        previous=previous,
    )


def tibia_axes(
    lateral_knee: Optional[np.ndarray],
    medial_knee: Optional[np.ndarray],
    lateral_ankle: Optional[np.ndarray],
    medial_ankle: Optional[np.ndarray],
    pelvis_z: np.ndarray,
    previous: Optional[np.ndarray] = None,
) -> Optional[np.ndarray]:
    """Build tibia anatomical coordinate system.

    Tibia frame: X=anterior, Y=distal, Z=lateral (ISB convention)

    Args:
        lateral_knee: Lateral knee condyle (3,) or None
        medial_knee: Medial knee condyle (3,) or None
        lateral_ankle: Lateral malleolus (3,) or None
        medial_ankle: Medial malleolus (3,) or None
        pelvis_z: Pelvis Z axis for alignment (3,)
        previous: Previous frame axes (3x3) or None

    Returns:
        Rotation matrix (3x3) or None if invalid
    """
    if not markers_valid(lateral_knee, medial_knee, lateral_ankle, medial_ankle):
        return None

    knee_mid = 0.5 * (lateral_knee + medial_knee)
    ankle_mid = 0.5 * (lateral_ankle + medial_ankle)
    return build_segment_frame(
        long_axis=knee_mid - ankle_mid,
        lateral_hint=lateral_ankle - medial_ankle,
        reference_lateral=pelvis_z,
        negate_long_axis=True,
        previous=previous,
    )


def foot_axes(
    calcaneus: Optional[np.ndarray],
    toe: Optional[np.ndarray],
    fifth_meta: Optional[np.ndarray],
    pelvis_z: np.ndarray,
    previous: Optional[np.ndarray] = None,
) -> Optional[np.ndarray]:
    """Build foot anatomical coordinate system.

    Foot frame:
    - X: Anterior (heel -> toe)
    - Y: Superior (perpendicular to forefoot plane)
    - Z: Lateral (5th metatarsal -> toe, aligned)

    Args:
        calcaneus: Heel marker (3,) or None
        toe: Toe marker (3,) or None
        fifth_meta: 5th metatarsal marker (3,) or None
        pelvis_z: Pelvis Z for alignment (3,)
        previous: Previous frame axes (3x3) or None

    Returns:
        Rotation matrix (3x3) or None if invalid

    Note: foot_axes uses a custom construction (X primary, Z secondary) because
    the foot's anterior direction is the primary axis (unlike limbs where Y is primary).
    """
    if not markers_valid(calcaneus, toe, fifth_meta):
        return None

    # X axis: heel -> toe (anterior) - primary axis for foot
    x = normalize(toe - calcaneus)

    # Z axis: forefoot width (lateral), aligned with pelvis
    z_temp = normalize(align_with_reference(fifth_meta - toe, pelvis_z))

    # Y axis: dorsal (up), from cross(Z, X)
    y = normalize(np.cross(z_temp, x))

    # Re-orthogonalize Z = cross(X, Y)
    z = normalize(np.cross(x, y))

    # Check for NaN
    if np.isnan(x).any() or np.isnan(y).any() or np.isnan(z).any():
        return None

    # X=anterior, Y=superior/dorsal, Z=lateral
    result = np.column_stack([x, y, z])
    return ensure_continuity(result, previous)


def trunk_axes(
    c7: Optional[np.ndarray],
    r_shoulder: Optional[np.ndarray],
    l_shoulder: Optional[np.ndarray],
    pelvis_origin: Optional[np.ndarray],
    pelvis_z: np.ndarray,
    previous: Optional[np.ndarray] = None,
) -> Optional[np.ndarray]:
    """Build trunk (thorax) anatomical coordinate system.

    Trunk frame: X=anterior, Y=superior, Z=lateral (ISB convention)

    Args:
        c7: C7 vertebra marker (3,) or None
        r_shoulder: Right shoulder marker (3,) or None
        l_shoulder: Left shoulder marker (3,) or None
        pelvis_origin: Pelvis center position (3,) or None
        pelvis_z: Pelvis Z axis for lateral alignment (3,)
        previous: Previous frame axes (3x3) or None

    Returns:
        Rotation matrix (3x3) or None if invalid
    """
    if not markers_valid(r_shoulder, l_shoulder, pelvis_origin):
        return None

    shoulder_mid = 0.5 * (r_shoulder + l_shoulder)
    # Use C7 if available, otherwise use shoulder midpoint
    trunk_top = c7 if markers_valid(c7) else shoulder_mid

    return build_segment_frame(
        long_axis=trunk_top - pelvis_origin,
        lateral_hint=r_shoulder - l_shoulder,
        reference_lateral=pelvis_z,
        negate_long_axis=False,  # Y points superiorly (not distally)
        previous=previous,
    )


def humerus_axes(
    shoulder: Optional[np.ndarray],
    elbow: Optional[np.ndarray],
    wrist: Optional[np.ndarray],
    trunk_z: np.ndarray,
    previous: Optional[np.ndarray] = None,
) -> Optional[np.ndarray]:
    """Build humerus (upper arm) anatomical coordinate system.

    Humerus frame: X=anterior, Y=distal, Z=lateral (ISB convention)
    Z axis is perpendicular to the shoulder-elbow-wrist plane.

    Args:
        shoulder: Shoulder marker (3,) or None
        elbow: Elbow marker (3,) or None
        wrist: Wrist marker (3,) or None
        trunk_z: Trunk Z axis for lateral alignment (3,)
        previous: Previous frame axes (3x3) or None

    Returns:
        Rotation matrix (3x3) or None if invalid
    """
    if not markers_valid(shoulder, elbow, wrist):
        return None

    # Z hint: perpendicular to arm plane (shoulder-elbow-wrist)
    lateral_hint = np.cross(elbow - shoulder, wrist - elbow)

    return build_segment_frame(
        long_axis=elbow - shoulder,
        lateral_hint=lateral_hint,
        reference_lateral=trunk_z,
        negate_long_axis=True,
        previous=previous,
    )


def forearm_axes(
    elbow: Optional[np.ndarray],
    wrist: Optional[np.ndarray],
    humerus_z: np.ndarray,
    previous: Optional[np.ndarray] = None,
) -> Optional[np.ndarray]:
    """Build forearm anatomical coordinate system.

    Forearm frame: X=anterior, Y=distal, Z=lateral (ISB convention)
    Z axis is aligned with the humerus lateral axis.

    Args:
        elbow: Elbow marker (3,) or None
        wrist: Wrist marker (3,) or None
        humerus_z: Humerus Z axis for alignment (3,)
        previous: Previous frame axes (3x3) or None

    Returns:
        Rotation matrix (3x3) or None if invalid
    """
    if not markers_valid(elbow, wrist):
        return None

    return build_segment_frame(
        long_axis=wrist - elbow,
        lateral_hint=humerus_z,
        reference_lateral=humerus_z,  # Already aligned, this is a no-op
        negate_long_axis=True,
        previous=previous,
    )

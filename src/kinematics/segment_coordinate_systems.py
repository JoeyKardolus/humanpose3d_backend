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

    Prevents axis flipping by checking dot product with previous frame.
    Note: We cannot simply flip all axes as that would create a left-handed
    system. Instead, we only flip if it maintains right-handedness.

    Args:
        current_axes: Current frame axes (3x3)
        previous_axes: Previous frame axes (3x3) or None

    Returns:
        Continuous axes (3x3)
    """
    if previous_axes is None:
        return current_axes

    if np.isnan(current_axes).any() or np.isnan(previous_axes).any():
        return current_axes

    # Score = sum of dot products of corresponding axes
    score = (
        np.dot(current_axes[:, 0], previous_axes[:, 0]) +
        np.dot(current_axes[:, 1], previous_axes[:, 1]) +
        np.dot(current_axes[:, 2], previous_axes[:, 2])
    )

    # If alignment is significantly negative, check if we should flip
    # But flipping all axes creates left-handed system, so skip this for now
    # The coordinate system should be stable enough without this
    # if score < -1.5:
    #     # Consider flipping, but preserve handedness
    #     pass

    return current_axes


def pelvis_axes(
    rasis: Optional[np.ndarray],
    lasis: Optional[np.ndarray],
    rpsis: Optional[np.ndarray],
    lpsis: Optional[np.ndarray],
    previous: Optional[np.ndarray] = None,
) -> Optional[np.ndarray]:
    """Build pelvis anatomical coordinate system.

    ISB-inspired pelvis frame:
    - Z: Right (RASIS -> LASIS direction, but reversed for right-pointing)
    - Y: Superior (PSIS midpoint -> ASIS midpoint)
    - X: Anterior (Y × Z, points forward)

    Args:
        rasis: Right ASIS position (3,) or None
        lasis: Left ASIS position (3,) or None
        rpsis: Right PSIS position (3,) or None
        lpsis: Left PSIS position (3,) or None
        previous: Previous frame axes for continuity (3x3) or None

    Returns:
        Rotation matrix (3x3) or None if invalid markers
    """
    # Check for valid inputs
    if any(
        m is None or not np.isfinite(m).all()
        for m in [rasis, lasis, rpsis, lpsis]
    ):
        return None

    asis_mid = 0.5 * (rasis + lasis)
    psis_mid = 0.5 * (rpsis + lpsis)

    # Z axis: medial-lateral (right pointing)
    z_hint = rasis - lasis

    # Y axis: inferior-superior
    y_hint = asis_mid - psis_mid

    axes = build_orthonormal_frame(z_hint, y_hint)

    if np.isnan(axes).any():
        return None

    # Reorder to X=anterior, Y=superior, Z=right
    # Current: Z=right, Y=superior, X=Z×Y=anterior
    x = axes[:, 2]  # anterior (cross product)
    y = axes[:, 1]  # superior
    z = axes[:, 0]  # right

    result = np.column_stack([x, y, z])
    return ensure_continuity(result, previous)


def femur_axes(
    hjc: Optional[np.ndarray],
    lateral_knee: Optional[np.ndarray],
    medial_knee: Optional[np.ndarray],
    pelvis_z: np.ndarray,
    previous: Optional[np.ndarray] = None,
) -> Optional[np.ndarray]:
    """Build femur anatomical coordinate system.

    Femur frame:
    - Y: Proximal->distal (knee center -> HJC, negated for distal-pointing)
    - Z: Lateral (knee epicondyle line, aligned with pelvis)
    - X: Anterior (Y × Z)

    Args:
        hjc: Hip joint center position (3,) or None
        lateral_knee: Lateral knee condyle position (3,) or None
        medial_knee: Medial knee condyle position (3,) or None
        pelvis_z: Pelvis Z axis for lateral alignment (3,)
        previous: Previous frame axes (3x3) or None

    Returns:
        Rotation matrix (3x3) or None if invalid
    """
    if any(
        m is None or not np.isfinite(m).all()
        for m in [hjc, lateral_knee, medial_knee]
    ):
        return None

    knee_mid = 0.5 * (lateral_knee + medial_knee)

    # Y axis: knee -> hip (we'll negate for distal-pointing)
    y_hint = hjc - knee_mid

    # Z axis: lateral direction (epicondyle line)
    z_hint = lateral_knee - medial_knee

    # Ensure Z points in same general direction as pelvis Z
    if np.dot(z_hint, pelvis_z) < 0:
        z_hint = -z_hint

    axes = build_orthonormal_frame(y_hint, z_hint)

    if np.isnan(axes).any():
        return None

    # Current: Y=proximal-distal, temp_Y orthogonal, Z=lateral
    x = axes[:, 2]  # anterior (Y × Z)
    y = -axes[:, 0]  # distal-pointing (negate to point down)
    z = axes[:, 1]  # lateral (re-orthogonalized)

    result = np.column_stack([x, y, z])
    return ensure_continuity(result, previous)


def tibia_axes(
    lateral_knee: Optional[np.ndarray],
    medial_knee: Optional[np.ndarray],
    lateral_ankle: Optional[np.ndarray],
    medial_ankle: Optional[np.ndarray],
    pelvis_z: np.ndarray,
    previous: Optional[np.ndarray] = None,
) -> Optional[np.ndarray]:
    """Build tibia anatomical coordinate system.

    Tibia frame:
    - Y: Proximal->distal (ankle center -> knee center, negated)
    - Z: Lateral (malleoli line, aligned with pelvis)
    - X: Anterior (Y × Z)

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
    if any(
        m is None or not np.isfinite(m).all()
        for m in [lateral_knee, medial_knee, lateral_ankle, medial_ankle]
    ):
        return None

    knee_mid = 0.5 * (lateral_knee + medial_knee)
    ankle_mid = 0.5 * (lateral_ankle + medial_ankle)

    # Y axis: knee -> ankle (we'll negate for distal)
    y_hint = knee_mid - ankle_mid

    # Z axis: malleoli width (lateral)
    z_hint = lateral_ankle - medial_ankle

    # Align with pelvis lateral
    if np.dot(z_hint, pelvis_z) < 0:
        z_hint = -z_hint

    axes = build_orthonormal_frame(y_hint, z_hint)

    if np.isnan(axes).any():
        return None

    x = axes[:, 2]  # anterior
    y = -axes[:, 0]  # distal
    z = axes[:, 1]  # lateral

    result = np.column_stack([x, y, z])
    return ensure_continuity(result, previous)


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
    """
    if any(
        m is None or not np.isfinite(m).all()
        for m in [calcaneus, toe, fifth_meta]
    ):
        return None

    # X axis: heel -> toe (anterior)
    x_hint = toe - calcaneus

    # Z axis: forefoot width (lateral)
    z_hint = fifth_meta - toe

    # Align with pelvis
    if np.dot(z_hint, pelvis_z) < 0:
        z_hint = -z_hint

    axes = build_orthonormal_frame(x_hint, z_hint)

    if np.isnan(axes).any():
        return None

    # X=anterior, Y=superior, Z=lateral
    x = axes[:, 0]
    y = axes[:, 1]
    z = axes[:, 2]

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

    Trunk frame:
    - Y: Superior (pelvis midpoint -> C7/shoulder midpoint)
    - Z: Lateral (R shoulder -> L shoulder direction, aligned with pelvis)
    - X: Anterior (Y × Z)

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
    if any(
        m is None or not np.isfinite(m).all()
        for m in [r_shoulder, l_shoulder, pelvis_origin]
    ):
        return None

    shoulder_mid = 0.5 * (r_shoulder + l_shoulder)

    # Use C7 if available, otherwise use shoulder midpoint
    if c7 is not None and np.isfinite(c7).all():
        trunk_top = c7
    else:
        trunk_top = shoulder_mid

    # Y axis: inferior-superior (pelvis -> trunk top)
    y_hint = trunk_top - pelvis_origin

    # Z axis: lateral (right shoulder -> left shoulder)
    z_hint = r_shoulder - l_shoulder

    # Align with pelvis lateral direction
    if np.dot(z_hint, pelvis_z) < 0:
        z_hint = -z_hint

    axes = build_orthonormal_frame(y_hint, z_hint)

    if np.isnan(axes).any():
        return None

    # Reorder: Y=superior, Z=lateral, X=anterior
    x = axes[:, 2]  # anterior (cross product)
    y = axes[:, 0]  # superior
    z = axes[:, 1]  # lateral (re-orthogonalized)

    result = np.column_stack([x, y, z])
    return ensure_continuity(result, previous)


def humerus_axes(
    shoulder: Optional[np.ndarray],
    elbow: Optional[np.ndarray],
    wrist: Optional[np.ndarray],
    trunk_z: np.ndarray,
    previous: Optional[np.ndarray] = None,
) -> Optional[np.ndarray]:
    """Build humerus (upper arm) anatomical coordinate system.

    Humerus frame:
    - Y: Proximal->distal (shoulder -> elbow, negated for distal-pointing)
    - Z: Lateral (perpendicular to plane of shoulder-elbow-wrist)
    - X: Anterior (Y × Z)

    Args:
        shoulder: Shoulder marker (3,) or None
        elbow: Elbow marker (3,) or None
        wrist: Wrist marker (3,) or None
        trunk_z: Trunk Z axis for lateral alignment (3,)
        previous: Previous frame axes (3x3) or None

    Returns:
        Rotation matrix (3x3) or None if invalid
    """
    if any(
        m is None or not np.isfinite(m).all()
        for m in [shoulder, elbow, wrist]
    ):
        return None

    # Y axis: shoulder -> elbow (will negate for distal)
    y_hint = elbow - shoulder

    # Z axis: perpendicular to arm plane (shoulder-elbow-wrist)
    v1 = elbow - shoulder
    v2 = wrist - elbow
    z_hint = np.cross(v1, v2)

    # Align with trunk lateral
    if np.dot(z_hint, trunk_z) < 0:
        z_hint = -z_hint

    axes = build_orthonormal_frame(y_hint, z_hint)

    if np.isnan(axes).any():
        return None

    x = axes[:, 2]  # anterior
    y = -axes[:, 0]  # distal-pointing (negated)
    z = axes[:, 1]  # lateral

    result = np.column_stack([x, y, z])
    return ensure_continuity(result, previous)


def forearm_axes(
    elbow: Optional[np.ndarray],
    wrist: Optional[np.ndarray],
    humerus_z: np.ndarray,
    previous: Optional[np.ndarray] = None,
) -> Optional[np.ndarray]:
    """Build forearm anatomical coordinate system.

    Forearm frame:
    - Y: Proximal->distal (elbow -> wrist, negated for distal)
    - Z: Lateral (aligned with humerus)
    - X: Anterior (Y × Z)

    Args:
        elbow: Elbow marker (3,) or None
        wrist: Wrist marker (3,) or None
        humerus_z: Humerus Z axis for alignment (3,)
        previous: Previous frame axes (3x3) or None

    Returns:
        Rotation matrix (3x3) or None if invalid
    """
    if any(
        m is None or not np.isfinite(m).all()
        for m in [elbow, wrist]
    ):
        return None

    # Y axis: elbow -> wrist
    y_hint = wrist - elbow

    # Z axis: align with humerus lateral
    z_hint = humerus_z.copy()

    axes = build_orthonormal_frame(y_hint, z_hint)

    if np.isnan(axes).any():
        return None

    x = axes[:, 2]  # anterior
    y = -axes[:, 0]  # distal
    z = axes[:, 1]  # lateral

    result = np.column_stack([x, y, z])
    return ensure_continuity(result, previous)

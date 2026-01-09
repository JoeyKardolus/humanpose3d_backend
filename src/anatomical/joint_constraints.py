"""Joint angle constraints for depth refinement.

Enforces anatomically plausible joint angle limits to disambiguate depth
in 3D pose estimation. Based on research showing that joint constraints
significantly reduce impossible poses and improve depth accuracy.

References:
- Akhter & Black (CVPR 2015): Pose-Conditioned Joint Angle Limits for 3D HPE
- MANIKIN (2024): Biomechanically Accurate Neural Inverse Kinematics
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple

import numpy as np


# Anatomical joint angle limits (degrees)
# Based on biomechanical literature and motion capture data
# Format: (min, max) for each degree of freedom

JOINT_LIMITS = {
    # Hip: (flexion/extension, abduction/adduction, internal/external rotation)
    "hip": {
        "flex": (-30, 130),   # Extension (-) to flexion (+)
        "abd": (-30, 50),     # Adduction (-) to abduction (+)
        "rot": (-45, 45),     # Internal (-) to external (+) rotation
    },

    # Knee: (flexion/extension, abduction/adduction, internal/external rotation)
    "knee": {
        "flex": (0, 160),     # Full extension (0) to flexion (+)
        "abd": (-15, 15),     # Limited abd/add in knee
        "rot": (-40, 40),     # Tibial rotation
    },

    # Ankle: (dorsi/plantar flexion, inversion/eversion, rotation)
    "ankle": {
        "flex": (-45, 30),    # Plantarflexion (-) to dorsiflexion (+)
        "abd": (-30, 30),     # Inversion/eversion
        "rot": (-30, 30),     # Limited ankle rotation
    },

    # Elbow: (flexion/extension, carrying angle, pronation/supination)
    "elbow": {
        "flex": (0, 150),     # Full extension to flexion
        "abd": (-10, 10),     # Limited varus/valgus
        "rot": (-90, 90),     # Forearm pronation/supination
    },

    # Shoulder: (flexion/extension, abduction/adduction, rotation)
    "shoulder": {
        "flex": (-60, 180),   # Extension to flexion
        "abd": (-30, 180),    # Adduction to abduction
        "rot": (-90, 90),     # Internal to external rotation
    },
}


# Pose-dependent limits: how limits vary with joint configuration
# Example: Knee abduction range decreases as knee flexes
POSE_DEPENDENT_LIMITS = {
    "knee_abd_vs_flex": {
        # (knee_flex_deg): (abd_min, abd_max)
        0: (-15, 15),       # Extended knee
        90: (-10, 10),      # 90° flexion
        140: (-5, 5),       # Deep flexion
    },
}


@dataclass
class ConstraintViolation:
    """Records a joint angle constraint violation."""
    joint: str
    dof: str  # "flex", "abd", or "rot"
    frame: int
    angle_deg: float
    limit_min: float
    limit_max: float
    violation_amount: float  # How far outside limits


def check_angle_violations(
    angles: Dict[str, np.ndarray],
    limits: Dict[str, Dict[str, Tuple[float, float]]] = JOINT_LIMITS,
) -> List[ConstraintViolation]:
    """Check for joint angle limit violations.

    Args:
        angles: Dict mapping joint names to angle arrays (N, 3) for flex/abd/rot
        limits: Joint limit definitions

    Returns:
        List of constraint violations
    """
    violations = []

    for joint, angle_array in angles.items():
        if joint not in limits:
            continue

        joint_limits = limits[joint]
        num_frames = angle_array.shape[0]

        for dof_idx, dof in enumerate(["flex", "abd", "rot"]):
            if dof not in joint_limits:
                continue

            min_limit, max_limit = joint_limits[dof]

            for frame in range(num_frames):
                angle = angle_array[frame, dof_idx]

                if not np.isfinite(angle):
                    continue

                # Check violation
                if angle < min_limit:
                    violations.append(ConstraintViolation(
                        joint=joint,
                        dof=dof,
                        frame=frame,
                        angle_deg=angle,
                        limit_min=min_limit,
                        limit_max=max_limit,
                        violation_amount=min_limit - angle,
                    ))
                elif angle > max_limit:
                    violations.append(ConstraintViolation(
                        joint=joint,
                        dof=dof,
                        frame=frame,
                        angle_deg=angle,
                        limit_min=min_limit,
                        limit_max=max_limit,
                        violation_amount=angle - max_limit,
                    ))

    return violations


def soft_clamp_angles(
    angles: Dict[str, np.ndarray],
    limits: Dict[str, Dict[str, Tuple[float, float]]] = JOINT_LIMITS,
    margin_deg: float = 5.0,
    strength: float = 0.5,
) -> Dict[str, np.ndarray]:
    """Apply soft clamping to joint angles near limits.

    Uses a smooth penalty function that gradually pulls angles back within
    limits without hard discontinuities.

    Args:
        angles: Joint angles dict (modified in-place)
        limits: Joint limit definitions
        margin_deg: Distance from limit where clamping starts
        strength: Clamping strength (0-1), higher = stronger pull toward limits

    Returns:
        Clamped angles (same dict, modified)
    """
    clamped = {}

    for joint, angle_array in angles.items():
        if joint not in limits:
            clamped[joint] = angle_array.copy()
            continue

        joint_limits = limits[joint]
        result = angle_array.copy()

        for dof_idx, dof in enumerate(["flex", "abd", "rot"]):
            if dof not in joint_limits:
                continue

            min_limit, max_limit = joint_limits[dof]

            for frame in range(result.shape[0]):
                angle = result[frame, dof_idx]

                if not np.isfinite(angle):
                    continue

                # Soft clamp near lower limit
                if angle < min_limit + margin_deg:
                    deficit = min_limit - angle
                    if deficit > 0:  # Violation
                        # Exponential pull-back
                        correction = deficit * strength
                        result[frame, dof_idx] = angle + correction
                    else:  # Within margin
                        # Smooth transition using sigmoid-like function
                        ratio = -deficit / margin_deg  # 0 at limit, 1 at margin
                        pull = (1 - ratio) * strength * 2
                        result[frame, dof_idx] = angle + pull

                # Soft clamp near upper limit
                elif angle > max_limit - margin_deg:
                    excess = angle - max_limit
                    if excess > 0:  # Violation
                        correction = excess * strength
                        result[frame, dof_idx] = angle - correction
                    else:  # Within margin
                        ratio = -excess / margin_deg
                        pull = (1 - ratio) * strength * 2
                        result[frame, dof_idx] = angle - pull

        clamped[joint] = result

    return clamped


def estimate_depth_correction_from_constraints(
    coords: np.ndarray,
    marker_index: Dict[str, int],
    side: Literal["R", "L"],
    target_limits: Dict[str, Dict[str, Tuple[float, float]]] = JOINT_LIMITS,
    max_correction_m: float = 0.05,
) -> np.ndarray:
    """Estimate depth corrections to satisfy joint angle constraints.

    This is an experimental function that adjusts Z coordinates (depth) to
    bring joint angles within biomechanical limits. It's a lightweight
    heuristic approach.

    Args:
        coords: Marker coordinates (N, M, 3)
        marker_index: Marker name to index mapping
        side: "R" or "L"
        target_limits: Joint angle limits to satisfy
        max_correction_m: Maximum allowed depth adjustment per marker

    Returns:
        Corrected coordinates (N, M, 3)

    Note:
        This is a simplified heuristic. For production use, consider:
        - Full inverse kinematics optimization
        - Temporal smoothness constraints
        - Integration with existing bone length constraints
    """
    # This is a placeholder for future implementation
    # Full IK optimization would go here

    # For now, return unchanged coordinates
    # TODO: Implement gradient-based optimization or RANSAC-style search
    return coords.copy()


def print_violation_summary(violations: List[ConstraintViolation]) -> None:
    """Print human-readable summary of constraint violations.

    Args:
        violations: List of detected violations
    """
    if not violations:
        print("✓ No joint angle constraint violations detected")
        return

    print(f"⚠ Found {len(violations)} joint angle violations:\n")

    # Group by joint and DOF
    by_joint_dof: Dict[Tuple[str, str], List[ConstraintViolation]] = {}

    for v in violations:
        key = (v.joint, v.dof)
        by_joint_dof.setdefault(key, []).append(v)

    for (joint, dof), joint_violations in sorted(by_joint_dof.items()):
        print(f"  {joint.upper()} {dof}:")
        print(f"    Violations: {len(joint_violations)} frames")

        violations_sorted = sorted(joint_violations, key=lambda x: abs(x.violation_amount), reverse=True)
        worst = violations_sorted[0]

        print(f"    Worst: frame {worst.frame}, angle={worst.angle_deg:.1f}°, "
              f"limit=[{worst.limit_min:.1f}, {worst.limit_max:.1f}]°, "
              f"violation={worst.violation_amount:.1f}°")
        print()

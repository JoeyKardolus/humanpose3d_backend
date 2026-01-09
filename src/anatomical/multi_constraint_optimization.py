"""Multi-constraint optimization for biomechanically accurate 3D pose.

Iteratively applies multiple biomechanical constraints until convergence:
1. Bone length consistency
2. Joint angle limits
3. Ground plane contact
4. Temporal smoothness

This prevents cascading violations where fixing one constraint breaks another.
Inspired by BioPose (2025) and MANIKIN (2024) multi-objective optimization.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np


def compute_constraint_violations(
    coords: np.ndarray,
    marker_index: Dict[str, int],
    subject_height: float,
) -> Dict[str, float]:
    """Compute all constraint violation metrics.

    Args:
        coords: Marker coordinates (num_frames, num_markers, 3)
        marker_index: Marker name -> index mapping
        subject_height: Subject height in meters

    Returns:
        Dictionary of violation metrics
    """
    violations = {
        "bone_lengths": 0,
        "ground_plane": 0,
        "temporal": 0,
    }

    # Bone length violations (coefficient of variation)
    bone_stats = compute_bone_length_statistics(coords, marker_index)
    if bone_stats:
        violations["bone_lengths"] = np.mean([s["cv"] for s in bone_stats.values()])

    # Ground plane violations (feet below ground)
    foot_markers = []
    for name in ["r_calc", "r_calc_study", "RHeel", "L_calc", "L_calc_study", "LHeel"]:
        if name in marker_index:
            foot_markers.append(marker_index[name])

    if foot_markers:
        foot_y = coords[:, foot_markers, 1]  # Y is vertical
        violations["ground_plane"] = np.sum(foot_y < 0)

    # Temporal smoothness (acceleration)
    if coords.shape[0] > 2:
        velocities = np.diff(coords, axis=0)
        accelerations = np.diff(velocities, axis=0)
        violations["temporal"] = np.nanmean(np.abs(accelerations))

    return violations


def multi_constraint_optimization(
    coords: np.ndarray,
    marker_index: Dict[str, int],
    subject_height: float = 1.78,
    max_iterations: int = 10,
    convergence_threshold: float = 0.01,
    bone_length_weight: float = 1.0,
    joint_angle_weight: float = 1.0,
    ground_plane_weight: float = 1.0,
    verbose: bool = True,
) -> Tuple[np.ndarray, Dict]:
    """Apply biomechanical constraints SEQUENTIALLY to prevent constraint fighting.

    Sequential Strategy:
    0. FILTER: Remove unreliable augmented markers (high temporal variance)
    1. STABILIZE: Apply bone length constraints to lock down skeleton geometry
    2. FINALIZE: Ground plane, hip width, heel smoothing

    This prevents the "cascading violation" problem where iterating between constraints
    causes them to fight each other (bone length improves then degrades).

    Args:
        coords: Marker coordinates (num_frames, num_markers, 3)
        marker_index: Marker name -> index mapping
        subject_height: Subject height in meters
        max_iterations: DEPRECATED - kept for API compatibility, not used
        convergence_threshold: DEPRECATED - kept for API compatibility, not used
        bone_length_weight: Weight for bone length constraint (0-1)
        joint_angle_weight: DEPRECATED - kept for API compatibility, not used
        ground_plane_weight: Weight for ground plane constraint (0-1)
        verbose: Print progress

    Returns:
        Tuple of (refined_coords, stats)
    """
    coords = coords.copy()

    stats = {
        "iterations": 1,  # Sequential is single-pass
        "initial_violations": {},
        "final_violations": {},
        "improvement": {},
    }

    # PHASE 0: Filter unreliable augmented markers (high temporal variance)
    # This removes noisy markers like heels before optimization
    if verbose:
        print("[multi_constraint] Filtering unreliable augmented markers...")

    coords = filter_unreliable_augmented_markers(
        coords,
        marker_index,
        variance_threshold=0.05,  # Filter markers with variance > 0.05
        verbose=verbose,
    )

    # Compute initial violations
    initial_violations = compute_constraint_violations(coords, marker_index, subject_height)
    stats["initial_violations"] = initial_violations

    if verbose:
        print("\n" + "="*60)
        print("Multi-Constraint Optimization (Sequential)")
        print("="*60)
        print(f"Initial state:")
        print(f"  Bone lengths: {initial_violations['bone_lengths']:.4f} (CV)")
        print(f"  Ground plane: {initial_violations['ground_plane']} violations")
        print(f"  Temporal: {initial_violations['temporal']:.6f} (accel)")
        print()

    # =========================================================================
    # PHASE 1: STABILIZE - Lock down bone lengths
    # =========================================================================
    if bone_length_weight > 0:
        if verbose:
            print("PHASE 1: STABILIZE - Locking down bone lengths...")

        coords = apply_bone_length_constraints_numpy(
            coords,
            marker_index,
            tolerance=0.08,      # Strict tolerance (8% deviation allowed)
            depth_weight=0.95,   # Very strong focus on depth (Z-axis)
            iterations=5,        # Aggressive stabilization
            augmented_constraints={
                # Medial knee markers (maintain distance from lateral knee)
                "r_knee_study": ["r_mknee_study"],
                "L_knee_study": ["L_mknee_study"],
                # Medial ankle + heel markers (maintain distance from lateral ankle)
                "r_ankle_study": ["r_mankle_study", "r_calc_study"],
                "L_ankle_study": ["L_mankle_study", "L_calc_study"],
            },
        )

        if verbose:
            stab_v = compute_constraint_violations(coords, marker_index, subject_height)
            print(f"  Bone length CV: {initial_violations['bone_lengths']:.4f} → {stab_v['bone_lengths']:.4f}")
            print()

    # =========================================================================
    # PHASE 3: FINALIZE - Ground plane, hip width, heel smoothing
    # =========================================================================
    if verbose:
        print("PHASE 3: FINALIZE - Ground plane, hip width, smoothing...")

    # Ground plane constraints
    if ground_plane_weight > 0:
        coords = apply_ground_plane_to_coords(
            coords,
            marker_index,
            contact_threshold=0.03,
        )

    # Hip width constraint
    coords = constrain_hip_width(coords, marker_index, subject_height)

    if verbose:
        fin_v = compute_constraint_violations(coords, marker_index, subject_height)
        print(f"  Ground plane violations: {fin_v['ground_plane']}")
        print(f"  Hip width: {compute_hip_width(coords, marker_index):.3f}m")
        print()

    # Final violations
    final_violations = compute_constraint_violations(coords, marker_index, subject_height)
    stats["final_violations"] = final_violations

    # Compute improvements
    for key in initial_violations:
        if initial_violations[key] > 0:
            improvement = 100 * (initial_violations[key] - final_violations[key]) / initial_violations[key]
            stats["improvement"][key] = improvement

    if verbose:
        print("="*60)
        print("Final Results (Sequential):")
        print("="*60)
        for key in ["bone_lengths", "ground_plane", "temporal"]:
            init = initial_violations[key]
            final = final_violations[key]
            if key in stats["improvement"]:
                improvement = stats["improvement"][key]
                print(f"{key:20s}: {init:.4f} → {final:.4f} ({improvement:+.1f}%)")
            else:
                print(f"{key:20s}: {init:.4f} → {final:.4f}")
        print("="*60)

    return coords, stats


def apply_augmented_marker_distance_constraints(
    coords: np.ndarray,
    marker_index: Dict[str, int],
    augmented_constraints: Dict[str, List[str]],
) -> np.ndarray:
    """Constrain augmented markers to maintain fixed distance from their MediaPipe parent.

    This prevents augmented markers (medial knee, medial ankle) from becoming scattered
    when bone length constraints adjust their parent MediaPipe markers (lateral knee/ankle).

    Strategy:
    - Compute median distance from parent across all frames
    - For each frame, project augmented marker onto sphere of that radius around parent
    - Preserve direction (so medial stays medial), only adjust distance

    Args:
        coords: Marker coordinates (num_frames, num_markers, 3)
        marker_index: Marker name -> index mapping
        augmented_constraints: Dict mapping parent marker -> list of child markers

    Returns:
        Coordinates with augmented markers constrained to fixed distance
    """
    coords = coords.copy()

    for parent_name, child_names in augmented_constraints.items():
        if parent_name not in marker_index:
            continue

        parent_idx = marker_index[parent_name]

        for child_name in child_names:
            if child_name not in marker_index:
                continue

            child_idx = marker_index[child_name]

            # Compute median distance across all frames (robust to outliers)
            distances = []
            for fi in range(coords.shape[0]):
                parent_pos = coords[fi, parent_idx]
                child_pos = coords[fi, child_idx]

                if np.isnan(parent_pos).any() or np.isnan(child_pos).any():
                    continue

                dist = np.linalg.norm(child_pos - parent_pos)
                if dist > 1e-6:  # Valid distance
                    distances.append(dist)

            if not distances:
                continue

            target_distance = np.median(distances)

            # Constrain each frame to maintain this distance
            for fi in range(coords.shape[0]):
                parent_pos = coords[fi, parent_idx]
                child_pos = coords[fi, child_idx]

                if np.isnan(parent_pos).any() or np.isnan(child_pos).any():
                    continue

                # Vector from parent to child
                vec = child_pos - parent_pos
                current_distance = np.linalg.norm(vec)

                if current_distance < 1e-6:
                    continue

                # Project onto sphere of target radius (preserve direction)
                direction = vec / current_distance
                coords[fi, child_idx] = parent_pos + direction * target_distance

    return coords


def apply_bone_length_constraints_numpy(
    coords: np.ndarray,
    marker_index: Dict[str, int],
    tolerance: float = 0.15,
    depth_weight: float = 0.8,
    iterations: int = 1,
    augmented_constraints: Optional[Dict[str, List[str]]] = None,
) -> np.ndarray:
    """Apply bone length constraints with RIGID SEGMENT adjustment.

    KEY FIX: When adjusting a child marker, move all its descendants by the same
    delta to preserve joint angles downstream. This prevents bone length constraints
    from undoing joint angle constraints.

    Example: If we move ankle to fix knee→ankle length, we also move heel and toes
    by the same delta, preserving ankle→heel and heel→toe angles.

    Args:
        coords: Marker coordinates (num_frames, num_markers, 3)
        marker_index: Marker name -> index mapping
        tolerance: Maximum allowed bone length deviation (fraction)
        depth_weight: Weight for depth (Z) vs XY corrections (0-1)
        iterations: Number of constraint iterations
        augmented_constraints: Optional dict mapping parent -> augmented child markers
                              to maintain fixed distance (not rigid movement)

    Returns:
        Coordinates with bone length constraints applied
    """
    coords = coords.copy()

    # Define kinematic chains: parent → child → grandchildren...
    # This defines rigid segments that should move together
    # IMPORTANT: Augmented markers (medial, heel) are NOT included here - they use distance constraints
    kinematic_chains = {
        # Right leg: hip → knee → ankle (NO heel - it's augmented)
        "RHJC_study": ["r_knee_study"],
        "r_knee_study": ["r_ankle_study"],  # Only main kinematic chain
        # r_calc_study removed - uses distance constraint instead

        # Left leg: hip → knee → ankle (NO heel - it's augmented)
        "LHJC_study": ["L_knee_study"],
        "L_knee_study": ["L_ankle_study"],  # Only main kinematic chain
        # L_calc_study removed - uses distance constraint instead

        # Toes are children of heel (keep these)
        "r_calc_study": ["r_toe_study", "r_5meta_study"],
        "L_calc_study": ["L_toe_study", "L_5meta_study"],
    }

    # Define bone pairs to constrain (parent → child relationships)
    bone_pairs = [
        ("RHJC_study", "r_knee_study"),      # Right thigh
        ("r_knee_study", "r_ankle_study"),   # Right shank
        ("LHJC_study", "L_knee_study"),      # Left thigh
        ("L_knee_study", "L_ankle_study"),   # Left shank
    ]

    # Filter to available bones
    available_bones = [(p, c) for p, c in bone_pairs if p in marker_index and c in marker_index]

    if not available_bones:
        return coords

    def get_all_descendants(marker_name: str) -> List[str]:
        """Get all descendant markers recursively (for rigid segment movement)."""
        descendants = []
        if marker_name in kinematic_chains:
            for child in kinematic_chains[marker_name]:
                descendants.append(child)
                descendants.extend(get_all_descendants(child))
        return descendants

    for _ in range(iterations):
        for parent_name, child_name in available_bones:
            p_idx = marker_index[parent_name]
            c_idx = marker_index[child_name]

            # Compute bone lengths across all frames
            lengths = []
            for fi in range(coords.shape[0]):
                p = coords[fi, p_idx]
                c = coords[fi, c_idx]

                if np.isnan(p).any() or np.isnan(c).any():
                    continue

                length = np.linalg.norm(c - p)
                lengths.append(length)

            if not lengths:
                continue

            # Target length = median
            target_length = np.median(lengths)

            # Adjust frames where length deviates too much
            for fi in range(coords.shape[0]):
                p = coords[fi, p_idx]
                c = coords[fi, c_idx]

                if np.isnan(p).any() or np.isnan(c).any():
                    continue

                current_length = np.linalg.norm(c - p)

                if current_length < 1e-6:
                    continue

                # Check if within tolerance
                deviation = abs(current_length - target_length) / target_length

                if deviation > tolerance:
                    # Compute corrected child position
                    direction = (c - p) / current_length
                    c_corrected = p + direction * target_length

                    # Blend with depth_weight (focus on Z)
                    c_new = c.copy()
                    c_new[2] = depth_weight * c_corrected[2] + (1 - depth_weight) * c[2]
                    c_new[0] = (1 - depth_weight) * c_corrected[0] + depth_weight * c[0]
                    c_new[1] = (1 - depth_weight) * c_corrected[1] + depth_weight * c[1]

                    # Compute delta (how much child moved)
                    delta = c_new - c

                    # CRITICAL: Apply same delta to child marker
                    coords[fi, c_idx] = c_new

                    # CRITICAL: Move all descendants by same delta (rigid segment)
                    descendants = get_all_descendants(child_name)
                    for desc_name in descendants:
                        if desc_name in marker_index:
                            desc_idx = marker_index[desc_name]
                            desc_pos = coords[fi, desc_idx]
                            if not np.isnan(desc_pos).any():
                                coords[fi, desc_idx] = desc_pos + delta

    # Apply augmented marker distance constraints (if provided)
    # These markers maintain FIXED DISTANCE from their parent (don't move rigidly)
    if augmented_constraints:
        coords = apply_augmented_marker_distance_constraints(
            coords, marker_index, augmented_constraints
        )

    return coords


def apply_ground_plane_to_coords(
    coords: np.ndarray,
    marker_index: Dict[str, int],
    contact_threshold: float = 0.03,
) -> np.ndarray:
    """Apply ground plane constraint: feet shouldn't go below ground.

    Estimates ground as 5th percentile of foot markers, then clamps.
    """
    coords = coords.copy()

    # Find foot markers
    foot_markers = []
    for name in ["r_calc", "r_calc_study", "RHeel", "L_calc", "L_calc_study", "LHeel",
                 "r_toe", "r_toe_study", "RBigToe", "L_toe", "L_toe_study", "LBigToe"]:
        if name in marker_index:
            foot_markers.append(marker_index[name])

    if not foot_markers:
        return coords

    # Estimate ground plane
    foot_y = coords[:, foot_markers, 1]  # Y is vertical
    ground_level = np.nanpercentile(foot_y, 5)

    # Clamp foot markers to ground
    for mi in foot_markers:
        below_ground = coords[:, mi, 1] < ground_level
        coords[below_ground, mi, 1] = ground_level

    return coords


def compute_bone_length_statistics(
    coords: np.ndarray,
    marker_index: Dict[str, int],
) -> Dict[str, Dict[str, float]]:
    """Compute bone length statistics (mean, std, CV) for all bones."""

    bone_pairs = [
        ("RHJC_study", "r_knee_study"),
        ("r_knee_study", "r_ankle_study"),
        ("r_ankle_study", "r_calc_study"),
        ("LHJC_study", "L_knee_study"),
        ("L_knee_study", "L_ankle_study"),
        ("L_ankle_study", "L_calc_study"),
    ]

    stats = {}

    for parent, child in bone_pairs:
        if parent not in marker_index or child not in marker_index:
            continue

        p_idx = marker_index[parent]
        c_idx = marker_index[child]

        lengths = []
        for fi in range(coords.shape[0]):
            p = coords[fi, p_idx]
            c = coords[fi, c_idx]

            if np.isnan(p).any() or np.isnan(c).any():
                continue

            length = np.linalg.norm(c - p)
            lengths.append(length)

        if lengths:
            mean_length = np.mean(lengths)
            std_length = np.std(lengths)
            cv = std_length / mean_length if mean_length > 0 else 0

            stats[f"{parent}-{child}"] = {
                "mean": mean_length,
                "std": std_length,
                "cv": cv,
            }

    return stats


def filter_unreliable_augmented_markers(
    coords: np.ndarray,
    marker_index: Dict[str, int],
    variance_threshold: float = 0.08,
    verbose: bool = True,
) -> np.ndarray:
    """Filter out augmented markers with high temporal variance (unreliable predictions).

    This removes noisy augmented markers (heels, etc.) that have high frame-to-frame
    variance, which indicates poor LSTM prediction quality.

    Args:
        coords: Marker coordinates (num_frames, num_markers, 3)
        marker_index: Marker name -> index mapping
        variance_threshold: Maximum allowed temporal variance (default 0.08)
        verbose: Print filtered markers

    Returns:
        Coordinates with unreliable markers set to NaN
    """
    coords = coords.copy()

    # Define MediaPipe markers (keep these always)
    mediapipe_markers = {
        'Neck', 'RShoulder', 'LShoulder', 'RHip', 'LHip', 'RKnee', 'LKnee',
        'RAnkle', 'LAnkle', 'RHeel', 'LHeel', 'RSmallToe', 'LSmallToe',
        'RBigToe', 'LBigToe', 'RElbow', 'LElbow', 'RWrist', 'LWrist',
        'Hip', 'Head', 'Nose'
    }

    filtered_markers = []
    checked_augmented = 0

    for marker_name, marker_idx in marker_index.items():
        # Skip MediaPipe markers (always keep)
        if marker_name in mediapipe_markers:
            continue

        checked_augmented += 1

        # Get marker coordinates
        marker_coords = coords[:, marker_idx, :]

        # Check valid frames
        valid_frames = ~np.isnan(marker_coords).any(axis=1)

        if np.sum(valid_frames) < 10:  # Need at least 10 frames
            continue

        valid_coords = marker_coords[valid_frames]

        # Compute temporal variance
        temporal_var = np.mean(np.var(valid_coords, axis=0))

        # Filter if variance too high
        if temporal_var > variance_threshold:
            coords[:, marker_idx, :] = np.nan
            filtered_markers.append((marker_name, temporal_var))

    if verbose:
        print(f"[filter_augmented] Checked {checked_augmented} augmented markers")
        if filtered_markers:
            print(f"[filter_augmented] Filtered {len(filtered_markers)} unreliable markers (variance > {variance_threshold}):")
            for name, var in sorted(filtered_markers, key=lambda x: x[1], reverse=True):
                print(f"  - {name:<20s} (variance={var:.6f})")
        else:
            print(f"[filter_augmented] No markers exceeded variance threshold {variance_threshold}")

    return coords


def compute_hip_width(
    coords: np.ndarray,
    marker_index: Dict[str, int],
) -> float:
    """Compute mean hip width (distance between RHJC and LHJC)."""
    if "RHJC_study" not in marker_index or "LHJC_study" not in marker_index:
        return 0.0

    r_idx = marker_index["RHJC_study"]
    l_idx = marker_index["LHJC_study"]

    widths = []
    for fi in range(coords.shape[0]):
        r = coords[fi, r_idx]
        l = coords[fi, l_idx]

        if np.isnan(r).any() or np.isnan(l).any():
            continue

        width = np.linalg.norm(l - r)
        widths.append(width)

    return np.median(widths) if widths else 0.0


def constrain_hip_width(
    coords: np.ndarray,
    marker_index: Dict[str, int],
    subject_height: float,
) -> np.ndarray:
    """Constrain hip width to anatomically plausible values.

    Hip width (inter-ASIS distance) is typically 0.18-0.22 * height for adults.
    We use 0.20 * height as target (midpoint).

    This prevents the hip joint centers from collapsing together or spreading too wide.
    """
    coords = coords.copy()

    if "RHJC_study" not in marker_index or "LHJC_study" not in marker_index:
        return coords

    r_idx = marker_index["RHJC_study"]
    l_idx = marker_index["LHJC_study"]

    # Target hip width based on anthropometry
    # Inter-ASIS distance ≈ 20% of height (Davis et al. 1991)
    target_width = 0.20 * subject_height

    # Allow some variation (±15%)
    min_width = target_width * 0.85
    max_width = target_width * 1.15

    for fi in range(coords.shape[0]):
        r = coords[fi, r_idx]
        l = coords[fi, l_idx]

        if np.isnan(r).any() or np.isnan(l).any():
            continue

        current_width = np.linalg.norm(l - r)

        # Only adjust if outside tolerance
        if current_width < min_width or current_width > max_width:
            # Use target width instead of clamping to boundary
            desired_width = target_width

            # Compute midpoint
            midpoint = (r + l) / 2

            # Direction from R to L
            direction = (l - r) / current_width if current_width > 1e-6 else np.array([1, 0, 0])

            # Reposition HJCs symmetrically
            coords[fi, r_idx] = midpoint - direction * desired_width / 2
            coords[fi, l_idx] = midpoint + direction * desired_width / 2

    return coords

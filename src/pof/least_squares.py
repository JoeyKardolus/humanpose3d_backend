"""Least-squares depth solver for POF-based 3D reconstruction.

Implements the MTC (MonocularTotalCapture) style least-squares solver that
keeps X,Y fixed from 2D observations and only solves for Z depths.

Key insight from MTC: Under orthographic projection, 3D X,Y ≈ 2D positions.
By normalizing both 2D input and POF predictions to the same scale (pelvis-
centered, unit torso), the least-squares solution becomes valid:

    scale = dot(delta_2d, orient_xy) / ||orient_xy||^2
    child_depth = parent_depth + scale * orient_z

This ensures the reconstructed 3D skeleton projects back to the observed 2D
positions (reprojection error ≈ 0 by construction).

Clean separation design (new architecture):
- X, Y direction comes from 2D observations (delta_2d)
- Z magnitude (|Z|) comes from ZMagnitudeHead
- Z sign comes from Z-sign classification head
- Full orientation is reconstructed by combining these components
"""

import torch
import torch.nn.functional as F
from typing import Tuple, Optional

from .constants import (
    LIMB_DEFINITIONS,
    NUM_JOINTS,
    NUM_LIMBS,
    LEFT_HIP_IDX,
    RIGHT_HIP_IDX,
    LEFT_SHOULDER_IDX,
    RIGHT_SHOULDER_IDX,
)

# NASA anthropometric proportions (fraction of body height)
# From bone_lengths.py AnatomicalProportions
NASA_TORSO_SIDE = 0.288      # shoulder to hip, same side
NASA_SHOULDER_WIDTH = 0.259  # biacromial width
NASA_HIP_WIDTH = 0.191       # bi-iliac width
NASA_UPPER_ARM = 0.172
NASA_FOREARM = 0.157
NASA_THIGH = 0.245
NASA_SHIN = 0.246
NASA_CROSS_TORSO = 0.345

# Normalized to unit-torso space (divide by torso proportion)
DEFAULT_SHOULDER_WIDTH = NASA_SHOULDER_WIDTH / NASA_TORSO_SIDE  # ~0.90
DEFAULT_HIP_WIDTH = NASA_HIP_WIDTH / NASA_TORSO_SIDE            # ~0.66


def get_default_bone_lengths_normalized() -> torch.Tensor:
    """Get NASA bone lengths in normalized (unit-torso) space.

    Returns:
        (14,) tensor of bone lengths normalized to unit-torso scale.
    """
    # Normalize all by torso proportion to get unit-torso scale
    t = NASA_TORSO_SIDE
    return torch.tensor([
        NASA_UPPER_ARM / t,      # 0: L_upper_arm
        NASA_FOREARM / t,        # 1: L_forearm
        NASA_UPPER_ARM / t,      # 2: R_upper_arm
        NASA_FOREARM / t,        # 3: R_forearm
        NASA_THIGH / t,          # 4: L_thigh
        NASA_SHIN / t,           # 5: L_shin
        NASA_THIGH / t,          # 6: R_thigh
        NASA_SHIN / t,           # 7: R_shin
        NASA_SHOULDER_WIDTH / t, # 8: shoulder_width
        NASA_HIP_WIDTH / t,      # 9: hip_width
        NASA_TORSO_SIDE / t,     # 10: L_torso (= 1.0)
        NASA_TORSO_SIDE / t,     # 11: R_torso (= 1.0)
        NASA_CROSS_TORSO / t,    # 12: L_cross
        NASA_CROSS_TORSO / t,    # 13: R_cross
    ], dtype=torch.float32)


def compute_limb_delta_2d(
    keypoints_2d: torch.Tensor,
) -> torch.Tensor:
    """Compute 2D limb displacement vectors from 2D keypoints.

    Args:
        keypoints_2d: (batch, 17, 2) 2D keypoints

    Returns:
        (batch, 14, 2) 2D limb displacement vectors (child - parent)
    """
    batch_size = keypoints_2d.size(0)
    device = keypoints_2d.device
    dtype = keypoints_2d.dtype

    delta_2d = torch.zeros(batch_size, NUM_LIMBS, 2, device=device, dtype=dtype)

    for limb_idx, (parent, child) in enumerate(LIMB_DEFINITIONS):
        delta_2d[:, limb_idx] = keypoints_2d[:, child] - keypoints_2d[:, parent]

    return delta_2d


def build_orientation_from_z_magnitude(
    z_magnitudes: torch.Tensor,
    z_signs: torch.Tensor,
    delta_2d: torch.Tensor,
) -> torch.Tensor:
    """Build full 3D orientation from Z magnitude, Z sign, and 2D observations.

    Clean separation design: Each component provides ONE piece of information:
    - delta_2d: XY direction from 2D observations
    - z_magnitudes: |Z| foreshortening from model
    - z_signs: Z direction (+1 or -1)

    Mathematical basis: For a unit orientation vector:
        ||orient_xy||² + Z² = 1

    Given:
    - delta_2d from observations -> provides direction of orient_xy
    - |Z| from z_magnitudes -> determines ||orient_xy|| = sqrt(1 - |Z|²)

    Reconstruction:
        xy_magnitude = sqrt(1 - z_magnitude²)
        orient_xy = xy_magnitude * normalize(delta_2d)
        orient_z = z_sign * z_magnitude

    Args:
        z_magnitudes: (batch, 14) predicted |Z| magnitudes in [0, 1]
        z_signs: (batch, 14) Z direction signs (+1 or -1)
        delta_2d: (batch, 14, 2) observed 2D limb directions

    Returns:
        (batch, 14, 3) unit orientation vectors
    """
    # Clamp z_magnitudes to valid range
    z_magnitudes = torch.clamp(z_magnitudes, 0.0, 1.0)

    # Compute XY magnitude from unit vector constraint: ||xy||² + |z|² = 1
    xy_magnitude_sq = 1.0 - z_magnitudes ** 2
    xy_magnitude_sq = torch.clamp(xy_magnitude_sq, min=0.0)  # numerical safety
    xy_magnitude = torch.sqrt(xy_magnitude_sq)  # (batch, 14)

    # Normalize observed 2D direction
    delta_2d_norm = F.normalize(delta_2d, dim=-1, eps=1e-6)  # (batch, 14, 2)

    # Scale by XY magnitude
    orient_xy = delta_2d_norm * xy_magnitude.unsqueeze(-1)  # (batch, 14, 2)

    # Apply sign to Z magnitude
    orient_z = z_magnitudes * z_signs  # (batch, 14)

    # Combine into unit vector [X, Y, Z]
    orientation = torch.cat([orient_xy, orient_z.unsqueeze(-1)], dim=-1)  # (batch, 14, 3)

    # Re-normalize for numerical safety (should already be close to unit)
    orientation = F.normalize(orientation, dim=-1, eps=1e-6)

    return orientation


def build_orientation_from_z_magnitude_and_logits(
    z_magnitudes: torch.Tensor,
    z_sign_logits: torch.Tensor,
    delta_2d: torch.Tensor,
    z_sign_threshold: float = 0.5,
) -> torch.Tensor:
    """Build orientation from Z magnitudes and Z-sign logits.

    Convenience wrapper that converts logits to signs before reconstruction.

    Args:
        z_magnitudes: (batch, 14) predicted |Z| magnitudes in [0, 1]
        z_sign_logits: (batch, 14) logits for P(Z > 0)
        delta_2d: (batch, 14, 2) observed 2D limb directions
        z_sign_threshold: Threshold for Z-sign classification (default 0.5)

    Returns:
        (batch, 14, 3) unit orientation vectors
    """
    # Convert logits to signs
    z_sign_prob = torch.sigmoid(z_sign_logits)  # (batch, 14)
    z_signs = torch.where(
        z_sign_prob > z_sign_threshold,
        torch.ones_like(z_sign_prob),
        -torch.ones_like(z_sign_prob),
    )  # (batch, 14)

    return build_orientation_from_z_magnitude(z_magnitudes, z_signs, delta_2d)


# Hierarchical solve order: hip width -> torso -> extremities
# Hip width MUST come first to establish L/R depth separation for side views
SOLVE_ORDER = [
    # First: establish L/R hip depth separation (critical for side views!)
    9,   # Hip width (11-12): L_hip → R_hip depth offset
    # From hips: solve shoulders via torso
    10,  # L torso (5-11): L_hip → L_shoulder
    11,  # R torso (6-12): R_hip → R_shoulder
    # From shoulders: solve arms
    0,   # L upper arm (5-7): L_shoulder → L_elbow
    2,   # R upper arm (6-8): R_shoulder → R_elbow
    1,   # L forearm (7-9): L_elbow → L_wrist
    3,   # R forearm (8-10): R_elbow → R_wrist
    # From hips: solve legs
    4,   # L thigh (11-13): L_hip → L_knee
    6,   # R thigh (12-14): R_hip → R_knee
    5,   # L shin (13-15): L_knee → L_ankle
    7,   # R shin (14-16): R_knee → R_ankle
]

# Kinematic parent→child for bone enforcement (respects reconstruction hierarchy)
# Different from LIMB_DEFINITIONS which defines POF direction (may be opposite!)
# For torso limbs, POF is shoulder→hip but kinematic is hip→shoulder
LIMB_KINEMATIC_PARENTS = {
    # Torso: hip is parent, shoulder is child (OPPOSITE of LIMB_DEFINITIONS!)
    10: (11, 5),   # L_hip → L_shoulder
    11: (12, 6),   # R_hip → R_shoulder
    # Arms: same as LIMB_DEFINITIONS
    0: (5, 7),     # L_shoulder → L_elbow
    1: (7, 9),     # L_elbow → L_wrist
    2: (6, 8),     # R_shoulder → R_elbow
    3: (8, 10),    # R_elbow → R_wrist
    # Legs: same as LIMB_DEFINITIONS
    4: (11, 13),   # L_hip → L_knee
    5: (13, 15),   # L_knee → L_ankle
    6: (12, 14),   # R_hip → R_knee
    7: (14, 16),   # R_knee → R_ankle
    # Width limbs: used for L/R depth separation in side views
    9: (11, 12),   # L_hip → R_hip (hip width, L_hip is parent/reference)
    # Unused width and cross-body limbs
    8: None,       # Shoulder width (derived from torso, not solved directly)
    12: None,      # L cross-body diagonal
    13: None,      # R cross-body diagonal
}


def normalize_2d_for_pof(
    keypoints_2d: torch.Tensor,
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Normalize 2D keypoints to POF training scale.

    Transforms [0,1] image coordinates to pelvis-centered, unit-torso scale.
    This matches the coordinate system used during POF training.

    Args:
        keypoints_2d: (batch, 17, 2) normalized [0,1] image coordinates
        eps: Small value to prevent division by zero

    Returns:
        normalized_2d: (batch, 17, 2) pelvis-centered, unit torso scale
        pelvis_2d: (batch, 2) pelvis position for denormalization
        torso_scale: (batch,) torso scale for denormalization
    """
    # Compute pelvis center (midpoint of hips)
    pelvis_2d = (
        keypoints_2d[:, LEFT_HIP_IDX] + keypoints_2d[:, RIGHT_HIP_IDX]
    ) / 2  # (batch, 2)

    # Center on pelvis
    centered = keypoints_2d - pelvis_2d.unsqueeze(1)  # (batch, 17, 2)

    # Compute torso scale (average of L/R shoulder-to-hip)
    l_torso = torch.norm(
        keypoints_2d[:, LEFT_SHOULDER_IDX] - keypoints_2d[:, LEFT_HIP_IDX],
        dim=-1
    )  # (batch,)
    r_torso = torch.norm(
        keypoints_2d[:, RIGHT_SHOULDER_IDX] - keypoints_2d[:, RIGHT_HIP_IDX],
        dim=-1
    )  # (batch,)
    torso_scale = (l_torso + r_torso) / 2  # (batch,)

    # Prevent division by zero
    safe_scale = torch.clamp(torso_scale, min=eps)

    # Scale to unit torso
    normalized_2d = centered / safe_scale.unsqueeze(-1).unsqueeze(-1)

    return normalized_2d, pelvis_2d, torso_scale


def solve_depth_least_squares_pof(
    pof: torch.Tensor,
    keypoints_2d: torch.Tensor,
    bone_lengths: Optional[torch.Tensor] = None,
    pelvis_depth: float = 0.0,
    normalize_input: bool = True,
    return_scale_factors: bool = False,
    use_2d_xy: bool = False,
    enforce_width: bool = False,
    enforce_all_bones: bool = False,
) -> torch.Tensor:
    """MTC-style depth solver using 2D keypoints and POF orientations.

    Pipeline:
    1. Normalize 2D: center on pelvis, scale to unit torso
    2. Copy normalized 2D X,Y as 3D X,Y (MTC insight)
    3. For each limb: solve scale = dot(delta_2d, orient_xy) / ||orient_xy||^2
    4. Compute depth: child_z = parent_z + scale * orient_z
    5. Return 3D pose in normalized camera space

    Same code used for training (differentiable) and inference.

    Args:
        pof: (batch, 14, 3) POF unit vectors from model
        keypoints_2d: (batch, 17, 2) 2D keypoints (normalized [0,1] if
                     normalize_input=True, or already normalized if False)
        bone_lengths: (14,) or (batch, 14) optional bone lengths for scale
                     clamping. If None, uses default clamp of [-1.5, 1.5]
        pelvis_depth: Initial Z for pelvis (default 0.0)
        normalize_input: If True, normalize 2D to pelvis-centered unit torso
        return_scale_factors: If True, also return scale factors for each limb
        use_2d_xy: If True, derive orient_xy from 2D delta instead of using
                  POF's XY directly. Only uses |Z| and sign(Z) from POF.
                  This ensures orient_xy is always parallel to observed 2D.
        enforce_width: If True, enforce hip and shoulder width bone lengths
                      using default normalized values. Fixes collapsed width
                      in side views.

    Returns:
        pose_3d: (batch, 17, 3) reconstructed 3D pose in normalized space
                 (pelvis-centered, unit torso scale)
        scale_factors: (batch, 14) scale factors if return_scale_factors=True
    """
    batch_size = pof.size(0)
    device = pof.device
    dtype = pof.dtype

    # Step 1: Normalize 2D if needed
    if normalize_input:
        normalized_2d, _, _ = normalize_2d_for_pof(keypoints_2d)
    else:
        normalized_2d = keypoints_2d

    # Step 2: Initialize 3D pose with 2D X,Y (MTC insight)
    pose_3d = torch.zeros(batch_size, NUM_JOINTS, 3, device=device, dtype=dtype)
    pose_3d[:, :, :2] = normalized_2d  # Copy X,Y from 2D

    # Initialize L_hip depth (reference point)
    # R_hip will be solved from hip_width POF (limb 9) - critical for side views!
    pose_3d[:, LEFT_HIP_IDX, 2] = pelvis_depth
    # R_hip initialized to same but will be overwritten by limb 9 solver
    pose_3d[:, RIGHT_HIP_IDX, 2] = pelvis_depth

    # Track scale factors for diagnostics/losses
    scale_factors = torch.zeros(batch_size, NUM_LIMBS, device=device, dtype=dtype)

    # Step 3-4: Solve depths hierarchically
    for limb_idx in SOLVE_ORDER:
        pof_parent, pof_child = LIMB_DEFINITIONS[limb_idx]
        orientation = pof[:, limb_idx]  # (batch, 3)

        # Use kinematic parent/child direction (may differ from POF direction)
        # For torso limbs, POF points shoulder→hip but kinematic is hip→shoulder
        kinematic = LIMB_KINEMATIC_PARENTS.get(limb_idx)
        if kinematic is not None:
            parent_idx, child_idx = kinematic
            # If kinematic direction is reversed from POF, negate orientation
            if (parent_idx, child_idx) != (pof_parent, pof_child):
                orientation = -orientation
        else:
            parent_idx, child_idx = pof_parent, pof_child

        # Get 2D positions (already normalized)
        parent_2d = normalized_2d[:, parent_idx]  # (batch, 2)
        child_2d = normalized_2d[:, child_idx]    # (batch, 2)
        parent_depth = pose_3d[:, parent_idx, 2]  # (batch,)

        # 2D displacement
        delta_2d = child_2d - parent_2d  # (batch, 2)
        delta_2d_len = torch.norm(delta_2d, dim=-1)  # (batch,)

        # Orientation components
        orient_z = orientation[:, 2]    # (batch,)

        if use_2d_xy:
            # Derive orient_xy from 2D delta - only use |Z| from POF
            # For unit vector: ||xy||² + |z|² = 1, so ||xy|| = sqrt(1 - |z|²)
            z_mag = torch.abs(orient_z)
            xy_mag = torch.sqrt(torch.clamp(1.0 - z_mag ** 2, min=0.0))
            delta_2d_norm = F.normalize(delta_2d, dim=-1, eps=1e-6)
            orient_xy = delta_2d_norm * xy_mag.unsqueeze(-1)  # (batch, 2)
        else:
            # Use POF's XY directly (original behavior)
            orient_xy = orientation[:, :2]  # (batch, 2)

        # Least-squares solution for scale
        # scale = dot(delta_2d, orient_xy) / ||orient_xy||^2
        orient_xy_norm_sq = (orient_xy ** 2).sum(dim=-1)  # (batch,)

        # Edge case: when orient_xy is small (limb pointing at camera)
        valid_solve = orient_xy_norm_sq > 0.05

        # Safe division
        safe_norm_sq = torch.where(
            valid_solve,
            orient_xy_norm_sq,
            torch.ones_like(orient_xy_norm_sq)
        )
        scale_from_lstsq = (delta_2d * orient_xy).sum(dim=-1) / safe_norm_sq

        # Fallback: when limb points at camera, use bone length directly
        # Since orientation is a unit vector, scale = bone_length gives correct 3D length
        # The sign is determined by the Z component of the orientation
        if bone_lengths is not None:
            if bone_lengths.dim() == 1:
                bl = bone_lengths[limb_idx]
            else:
                bl = bone_lengths[:, limb_idx]
            fallback_scale = bl * torch.sign(orient_z + 1e-8)
        else:
            # No bone length info, fall back to 2D length (poor estimate for side views)
            fallback_scale = delta_2d_len * torch.sign(orient_z + 1e-8)

        # Choose between lstsq and fallback
        scale = torch.where(valid_solve, scale_from_lstsq, fallback_scale)

        # Clamp scale to reasonable bounds
        if bone_lengths is not None:
            # Clamp to [-1.5, 1.5] * bone_length
            if bone_lengths.dim() == 1:
                max_scale = 1.5 * bone_lengths[limb_idx]
            else:
                max_scale = 1.5 * bone_lengths[:, limb_idx]
            scale = scale.clamp(-max_scale, max_scale)
        else:
            # Default clamp for unit torso scale
            scale = scale.clamp(-1.5, 1.5)

        # Store scale factor
        scale_factors[:, limb_idx] = scale

        # Compute child depth
        child_depth = parent_depth + scale * orient_z
        pose_3d[:, child_idx, 2] = child_depth

    # Handle head joints (not in kinematic chain)
    shoulder_center_depth = (
        pose_3d[:, LEFT_SHOULDER_IDX, 2] + pose_3d[:, RIGHT_SHOULDER_IDX, 2]
    ) / 2

    # Nose is typically slightly in front of shoulders
    pose_3d[:, 0, 2] = shoulder_center_depth + 0.05

    # Eyes and ears: same depth as nose
    for idx in [1, 2, 3, 4]:
        pose_3d[:, idx, 2] = pose_3d[:, 0, 2]

    # NaN safety
    nan_mask = torch.isnan(pose_3d).any(dim=-1, keepdim=True)
    if nan_mask.any():
        # Replace NaN with zeros
        pose_3d = torch.where(nan_mask, torch.zeros_like(pose_3d), pose_3d)

    # Enforce width bone lengths (hip and shoulder)
    if enforce_width:
        # Create default width bone lengths in normalized space
        width_bl = torch.zeros(NUM_LIMBS, device=device, dtype=dtype)
        width_bl[8] = DEFAULT_SHOULDER_WIDTH  # shoulder width
        width_bl[9] = DEFAULT_HIP_WIDTH       # hip width
        pose_3d = enforce_width_bone_lengths(pose_3d, width_bl, pof, strength=1.0)

    if return_scale_factors:
        return pose_3d, scale_factors
    return pose_3d


def enforce_bone_lengths(
    pose_3d: torch.Tensor,
    bone_lengths: torch.Tensor,
    strength: float = 1.0,
) -> torch.Tensor:
    """Enforce bone length constraints by adjusting Z only (preserve X,Y from 2D).

    For side views where limbs overlap in 2D, the LS solver may produce
    limbs that are too short (since X,Y are locked to 2D positions).
    This function adjusts only the Z coordinate to achieve the required bone length,
    preserving the 2D projection (X,Y).

    Math: Given fixed dx, dy from 2D, solve for dz such that:
        sqrt(dx² + dy² + dz²) = bone_length
        dz = ±sqrt(bone_length² - dx² - dy²)

    The sign is preserved from the original POF-derived Z direction.

    IMPORTANT: Uses LIMB_KINEMATIC_PARENTS (not LIMB_DEFINITIONS) for parent/child
    direction. For torso limbs, POF direction is shoulder→hip but kinematic
    reconstruction is hip→shoulder. Using wrong direction would overwrite hip Z
    (the root) based on shoulder, destroying the skeleton hierarchy.

    Args:
        pose_3d: (batch, 17, 3) solved pose
        bone_lengths: (14,) or (batch, 14) expected bone lengths
        strength: 0-1, how strongly to enforce (1.0 = exact, 0.5 = halfway)

    Returns:
        (batch, 17, 3) pose with corrected bone lengths (only Z modified)
    """
    batch_size = pose_3d.size(0)
    pose_out = pose_3d.clone()

    # Handle single bone_lengths array
    if bone_lengths.dim() == 1:
        bone_lengths = bone_lengths.unsqueeze(0).expand(batch_size, -1)

    # Process limbs in kinematic order (parents before children)
    # This ensures corrections propagate correctly from root (hips) to leaves
    # Skip width limbs (8, 9) and cross-body (12, 13) - no kinematic parent/child
    ENFORCE_ORDER = [
        # Torso first (hips are root, shoulders are children)
        10, 11,  # L/R torso: hip → shoulder
        # Arms (from shoulders)
        0, 2,    # Upper arms: shoulder → elbow
        1, 3,    # Forearms: elbow → wrist
        # Legs (from hips)
        4, 6,    # Thighs: hip → knee
        5, 7,    # Shins: knee → ankle
    ]

    for limb_idx in ENFORCE_ORDER:
        # Use kinematic parent/child direction (not POF direction!)
        kinematic = LIMB_KINEMATIC_PARENTS.get(limb_idx)
        if kinematic is None:
            continue  # Skip limbs without kinematic hierarchy
        parent_idx, child_idx = kinematic

        # Current positions
        parent_pos = pose_out[:, parent_idx]  # (batch, 3)
        child_pos = pose_out[:, child_idx]    # (batch, 3)

        # XY displacement is FIXED (from 2D projection)
        dx = child_pos[:, 0] - parent_pos[:, 0]  # (batch,)
        dy = child_pos[:, 1] - parent_pos[:, 1]  # (batch,)
        dz_original = child_pos[:, 2] - parent_pos[:, 2]  # (batch,)

        # Current XY distance squared
        xy_dist_sq = dx * dx + dy * dy  # (batch,)

        # Expected bone length
        L = bone_lengths[:, limb_idx]  # (batch,)
        L_sq = L * L

        # Required Z displacement: dz² = L² - dx² - dy²
        dz_sq_required = L_sq - xy_dist_sq

        # If XY distance already exceeds bone length, we can't fix it
        # (would require imaginary Z). In this case, just use minimal Z.
        dz_sq_required = torch.clamp(dz_sq_required, min=0.0)

        # Compute required |dz|
        dz_magnitude = torch.sqrt(dz_sq_required)

        # Preserve sign from original Z direction (from POF prediction)
        dz_sign = torch.sign(dz_original)
        dz_sign = torch.where(dz_sign == 0, torch.ones_like(dz_sign), dz_sign)

        # New Z displacement
        dz_new = dz_sign * dz_magnitude

        # Apply strength (blend between original and corrected)
        dz_final = dz_original + strength * (dz_new - dz_original)

        # Update only the Z coordinate of child
        pose_out[:, child_idx, 2] = parent_pos[:, 2] + dz_final

    return pose_out


def enforce_width_bone_lengths(
    pose_3d: torch.Tensor,
    bone_lengths: torch.Tensor,
    pof: torch.Tensor,
    strength: float = 1.0,
    foreshorten_threshold: float = 0.7,
) -> torch.Tensor:
    """Enforce bone lengths for width limbs (hip width, shoulder width).

    Width limbs connect L/R joints at the same hierarchy level, so they need
    special handling. We use L joint as reference and adjust R joint's Z.

    Only enforces when the 2D distance is significantly foreshortened (< threshold
    of expected bone length), to avoid adding unnecessary depth when facing camera.

    Args:
        pose_3d: (batch, 17, 3) pose
        bone_lengths: (14,) or (batch, 14) bone lengths
        pof: (batch, 14, 3) POF vectors (for Z sign)
        strength: 0-1, how strongly to enforce
        foreshorten_threshold: Only enforce if 2D dist < threshold * bone_length

    Returns:
        (batch, 17, 3) pose with corrected width bone lengths
    """
    batch_size = pose_3d.size(0)
    pose_out = pose_3d.clone()

    if bone_lengths.dim() == 1:
        bone_lengths = bone_lengths.unsqueeze(0).expand(batch_size, -1)

    # Width limbs: (limb_idx, L_joint, R_joint)
    WIDTH_LIMBS = [
        (9, LEFT_HIP_IDX, RIGHT_HIP_IDX),       # hip width
        (8, LEFT_SHOULDER_IDX, RIGHT_SHOULDER_IDX),  # shoulder width
    ]

    for limb_idx, l_joint, r_joint in WIDTH_LIMBS:
        # L joint is reference, adjust R joint
        l_pos = pose_out[:, l_joint]  # (batch, 3)
        r_pos = pose_out[:, r_joint]  # (batch, 3)

        # XY displacement is FIXED from 2D
        dx = r_pos[:, 0] - l_pos[:, 0]
        dy = r_pos[:, 1] - l_pos[:, 1]
        dz_original = r_pos[:, 2] - l_pos[:, 2]

        xy_dist = torch.sqrt(dx * dx + dy * dy)
        xy_dist_sq = xy_dist * xy_dist

        # Expected bone length
        L = bone_lengths[:, limb_idx]
        L_sq = L * L

        # Only enforce if significantly foreshortened (person is turned)
        # If 2D distance is close to bone length, person is facing camera
        needs_enforcement = xy_dist < (foreshorten_threshold * L)

        # Required Z: dz² = L² - dx² - dy²
        dz_sq_required = torch.clamp(L_sq - xy_dist_sq, min=0.0)
        dz_magnitude = torch.sqrt(dz_sq_required)

        # Use POF Z sign for direction
        pof_z = pof[:, limb_idx, 2]
        dz_sign = torch.sign(pof_z)
        dz_sign = torch.where(dz_sign == 0, torch.sign(dz_original), dz_sign)
        dz_sign = torch.where(dz_sign == 0, torch.ones_like(dz_sign), dz_sign)

        dz_new = dz_sign * dz_magnitude

        # Only apply enforcement where needed
        dz_final = torch.where(
            needs_enforcement,
            dz_original + strength * (dz_new - dz_original),
            dz_original
        )

        pose_out[:, r_joint, 2] = l_pos[:, 2] + dz_final

    return pose_out


def denormalize_pose_3d(
    pose_3d: torch.Tensor,
    pelvis_2d: torch.Tensor,
    torso_scale: torch.Tensor,
    output_depth: float = 2.0,
    metric_torso_scale: Optional[float] = None,
) -> torch.Tensor:
    """Denormalize pose from unit torso scale to meters.

    When metric_torso_scale is provided (computed from known subject height),
    the output is in true metric scale. Otherwise, uses the 2D-derived torso
    scale which gives approximate but not true metric output.

    Args:
        pose_3d: (batch, 17, 3) normalized pose (pelvis-centered, unit torso)
        pelvis_2d: (batch, 2) original pelvis position in image coords
        torso_scale: (batch,) original torso scale from 2D (used for X,Y only
                    when metric_torso_scale is provided)
        output_depth: Target depth for pelvis in output
        metric_torso_scale: If provided, use this for true metric scale output.
                           Computed as: subject_height / HEIGHT_TO_TORSO_RATIO

    Returns:
        (batch, 17, 3) denormalized pose in meter scale
    """
    batch_size = pose_3d.size(0)
    device = pose_3d.device
    dtype = pose_3d.dtype

    # Use metric scale if provided, otherwise fall back to 2D-derived scale
    if metric_torso_scale is not None:
        # True metric output using known subject height
        scale = torch.full((batch_size,), metric_torso_scale, device=device, dtype=dtype)
    else:
        # Approximate scale from 2D observations
        scale = torso_scale

    # Scale back
    output = pose_3d * scale.unsqueeze(-1).unsqueeze(-1)

    # Add back pelvis position (X,Y) scaled appropriately
    # When using metric scale, we still use 2D pelvis position but scale it
    # to maintain correct image-space relationship
    pelvis_3d = torch.zeros(batch_size, 1, 3, device=device, dtype=dtype)
    if metric_torso_scale is not None:
        # Scale pelvis X,Y by ratio of metric to 2D scale
        scale_ratio = scale / torch.clamp(torso_scale, min=1e-6)
        pelvis_3d[:, 0, :2] = pelvis_2d * scale_ratio.unsqueeze(-1)
    else:
        pelvis_3d[:, 0, :2] = pelvis_2d
    pelvis_3d[:, 0, 2] = output_depth
    output = output + pelvis_3d

    return output


def solve_with_denormalization(
    pof: torch.Tensor,
    keypoints_2d: torch.Tensor,
    bone_lengths: Optional[torch.Tensor] = None,
    output_depth: float = 2.0,
    metric_torso_scale: Optional[float] = None,
    enforce_bones: bool = False,
    use_2d_xy: bool = True,
) -> torch.Tensor:
    """Solve depths and denormalize to meter scale.

    Convenience function that combines solving and denormalization.

    Args:
        pof: (batch, 14, 3) POF unit vectors
        keypoints_2d: (batch, 17, 2) normalized [0,1] image coordinates
        bone_lengths: Optional bone lengths for scale clamping (in meters)
        output_depth: Target pelvis depth in output
        metric_torso_scale: If provided, use for true metric scale output.
                           Computed as: subject_height / HEIGHT_TO_TORSO_RATIO
        enforce_bones: If True, enforce bone length constraints by adjusting Z.
                      Default False - rely on POF predictions for bone lengths.
        use_2d_xy: If True (default), derive orient_xy from 2D observations.
                  Only uses |Z| from POF. This ensures correct reconstruction.

    Returns:
        (batch, 17, 3) 3D pose in meter scale
    """
    # Normalize and remember original scale
    normalized_2d, pelvis_2d, torso_scale = normalize_2d_for_pof(keypoints_2d)

    # Normalize bone_lengths to unit-torso scale (to match normalized 2D space)
    # In normalized space, torso = 1 unit, so divide by metric torso length
    # This fixes scale clamping: e.g., 0.5m / 0.52m ≈ 0.96 units instead of 0.5
    bl_normalized = None
    if bone_lengths is not None:
        if metric_torso_scale is not None:
            # Use known metric torso scale from subject height
            bl_normalized = bone_lengths / metric_torso_scale
        else:
            # Use average torso bone length as normalizer
            # Limbs 10, 11 are L/R torso (shoulder→hip)
            avg_torso = (bone_lengths[10] + bone_lengths[11]) / 2
            avg_torso = torch.clamp(avg_torso, min=1e-6)
            bl_normalized = bone_lengths / avg_torso

    # Solve depths with NORMALIZED bone_lengths
    pose_3d = solve_depth_least_squares_pof(
        pof, normalized_2d,
        bone_lengths=bl_normalized,  # Now in unit-torso scale
        pelvis_depth=0.0,
        normalize_input=False,
        use_2d_xy=use_2d_xy,
    )

    # Denormalize with metric scale if provided
    pose_3d = denormalize_pose_3d(
        pose_3d, pelvis_2d, torso_scale, output_depth,
        metric_torso_scale=metric_torso_scale
    )

    # Optionally enforce bone lengths (in meter space - use original bone_lengths)
    if enforce_bones and bone_lengths is not None:
        pose_3d = enforce_bone_lengths(pose_3d, bone_lengths, strength=1.0)

    return pose_3d

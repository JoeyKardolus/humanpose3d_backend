"""Graph construction utilities for GNN-based POF.

Builds adjacency matrices for different edge types:
- Joint-sharing: Limbs that share at least one joint
- Symmetry: Left-right symmetric limb pairs
- Kinematic: Parent-child limb dependencies in the kinematic chain
"""

import torch
from typing import Tuple

from .constants import LIMB_DEFINITIONS, LIMB_SWAP_PAIRS, NUM_LIMBS


def build_joint_sharing_adj() -> torch.Tensor:
    """Build adjacency matrix for limbs sharing at least one joint.

    Two limbs are connected if they share a common joint.
    For example:
    - L_upper_arm (5→7) and L_forearm (7→9) share elbow (joint 7)
    - L_upper_arm (5→7) and L_torso (5→11) share L shoulder (joint 5)

    Returns:
        (14, 14) adjacency matrix
    """
    adj = torch.zeros(NUM_LIMBS, NUM_LIMBS)

    for i, (p1, c1) in enumerate(LIMB_DEFINITIONS):
        for j, (p2, c2) in enumerate(LIMB_DEFINITIONS):
            if i != j:
                # Check if any joint is shared
                joints_i = {p1, c1}
                joints_j = {p2, c2}
                if joints_i & joints_j:  # Intersection is non-empty
                    adj[i, j] = 1.0

    return adj


def build_symmetry_adj() -> torch.Tensor:
    """Build adjacency matrix for left-right symmetric limb pairs.

    Connects symmetric limbs:
    - L_upper_arm (0) ↔ R_upper_arm (2)
    - L_forearm (1) ↔ R_forearm (3)
    - L_thigh (4) ↔ R_thigh (6)
    - L_shin (5) ↔ R_shin (7)
    - L_torso (10) ↔ R_torso (11)
    - L_cross (12) ↔ R_cross (13)

    Returns:
        (14, 14) adjacency matrix
    """
    adj = torch.zeros(NUM_LIMBS, NUM_LIMBS)

    for i, j in LIMB_SWAP_PAIRS:
        adj[i, j] = 1.0
        adj[j, i] = 1.0

    return adj


def build_kinematic_adj() -> torch.Tensor:
    """Build adjacency matrix for kinematic chain dependencies.

    Connects limbs based on the reconstruction order:
    - Torso limbs (L/R torso) influence shoulder position
    - Upper arm limbs influence forearm
    - Thigh limbs influence shin
    - Hip width influences thighs
    - Shoulder width influences upper arms

    Returns:
        (14, 14) adjacency matrix
    """
    adj = torch.zeros(NUM_LIMBS, NUM_LIMBS)

    # Kinematic dependencies (parent_limb, child_limb)
    # Limb indices from constants.py:
    # 0: L_upper_arm, 1: L_forearm, 2: R_upper_arm, 3: R_forearm
    # 4: L_thigh, 5: L_shin, 6: R_thigh, 7: R_shin
    # 8: shoulder_width, 9: hip_width
    # 10: L_torso, 11: R_torso, 12: L_cross, 13: R_cross

    kinematic_deps = [
        # Torso to upper arms
        (10, 0),  # L_torso → L_upper_arm (shares L shoulder)
        (11, 2),  # R_torso → R_upper_arm (shares R shoulder)
        (8, 0),   # shoulder_width → L_upper_arm
        (8, 2),   # shoulder_width → R_upper_arm

        # Upper arms to forearms
        (0, 1),   # L_upper_arm → L_forearm (shares elbow)
        (2, 3),   # R_upper_arm → R_forearm (shares elbow)

        # Hip to thighs
        (9, 4),   # hip_width → L_thigh
        (9, 6),   # hip_width → R_thigh
        (10, 4),  # L_torso → L_thigh (shares L hip)
        (11, 6),  # R_torso → R_thigh (shares R hip)

        # Thighs to shins
        (4, 5),   # L_thigh → L_shin (shares knee)
        (6, 7),   # R_thigh → R_shin (shares knee)

        # Cross-body diagonals connect torso regions
        (12, 10), # L_cross connects to L_torso
        (12, 11), # L_cross connects to R_torso
        (13, 10), # R_cross connects to L_torso
        (13, 11), # R_cross connects to R_torso
    ]

    for parent, child in kinematic_deps:
        adj[parent, child] = 1.0
        adj[child, parent] = 1.0  # Bidirectional for message passing

    return adj


def build_all_adjacencies() -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build all three adjacency matrices.

    Returns:
        (adj_joint, adj_kinematic, adj_symmetry) tuple of (14, 14) matrices
    """
    return (
        build_joint_sharing_adj(),
        build_kinematic_adj(),
        build_symmetry_adj(),
    )


def get_combined_adj(include_self_loops: bool = True) -> torch.Tensor:
    """Get a combined adjacency matrix (union of all edge types).

    Useful for basic GCN that uses a single adjacency.

    Args:
        include_self_loops: Whether to add self-connections

    Returns:
        (14, 14) combined adjacency matrix
    """
    adj_joint = build_joint_sharing_adj()
    adj_kin = build_kinematic_adj()
    adj_sym = build_symmetry_adj()

    # Union of all edges
    adj = (adj_joint + adj_kin + adj_sym).clamp(max=1.0)

    if include_self_loops:
        adj = adj + torch.eye(NUM_LIMBS)

    return adj


def visualize_adjacency(adj: torch.Tensor, name: str = "Adjacency") -> str:
    """Create ASCII visualization of adjacency matrix.

    Args:
        adj: (14, 14) adjacency matrix
        name: Name for the visualization

    Returns:
        String representation of the matrix
    """
    from .constants import LIMB_NAMES

    lines = [f"\n{name} Matrix (14×14):"]

    # Header
    header = "            " + " ".join(f"{i:2d}" for i in range(NUM_LIMBS))
    lines.append(header)

    # Rows
    for i, limb_name in enumerate(LIMB_NAMES):
        row_str = f"{limb_name:12s}"
        for j in range(NUM_LIMBS):
            val = adj[i, j].item()
            char = "█" if val > 0.5 else "·"
            row_str += f" {char} "
        lines.append(row_str)

    return "\n".join(lines)


if __name__ == "__main__":
    # Test and visualize adjacency matrices
    print("Building adjacency matrices...")

    adj_joint = build_joint_sharing_adj()
    adj_kin = build_kinematic_adj()
    adj_sym = build_symmetry_adj()
    adj_combined = get_combined_adj()

    print(f"Joint-sharing edges: {int(adj_joint.sum().item())}")
    print(f"Kinematic edges: {int(adj_kin.sum().item())}")
    print(f"Symmetry edges: {int(adj_sym.sum().item())}")
    print(f"Combined edges: {int(adj_combined.sum().item() - NUM_LIMBS)}")  # Exclude self-loops

    print(visualize_adjacency(adj_joint, "Joint-Sharing"))
    print(visualize_adjacency(adj_sym, "Symmetry"))
    print(visualize_adjacency(adj_kin, "Kinematic"))

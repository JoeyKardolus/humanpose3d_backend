"""Graph construction utilities for GNN-based joint refinement.

Builds adjacency matrices for different edge types over 12 joints:
- Kinematic: Parent-child skeleton hierarchy
- Symmetry: Left-right symmetric joint pairs
- Hierarchy: Biomechanical coupling (core stability, pelvis-trunk coordination)

Joint order (12 joints):
    0: pelvis, 1: hip_R, 2: hip_L, 3: knee_R, 4: knee_L,
    5: ankle_R, 6: ankle_L, 7: trunk, 8: shoulder_R, 9: shoulder_L,
    10: elbow_R, 11: elbow_L
"""

import torch
from typing import Tuple

# Number of joints in the model
NUM_JOINTS = 12

# Joint names (same order as model.py)
JOINT_NAMES = [
    'pelvis', 'hip_R', 'hip_L', 'knee_R', 'knee_L',
    'ankle_R', 'ankle_L', 'trunk', 'shoulder_R', 'shoulder_L',
    'elbow_R', 'elbow_L',
]

# Joint name to index mapping
JOINT_TO_IDX = {name: i for i, name in enumerate(JOINT_NAMES)}

# Kinematic edges (parent -> child skeleton hierarchy)
# These represent the actual bone connections in the kinematic chain
KINEMATIC_EDGES = [
    (0, 1), (0, 2), (0, 7),  # pelvis -> hip_R, hip_L, trunk
    (1, 3), (2, 4),          # hip -> knee
    (3, 5), (4, 6),          # knee -> ankle
    (7, 8), (7, 9),          # trunk -> shoulders
    (8, 10), (9, 11),        # shoulder -> elbow
]

# Symmetry edges (left-right pairs)
# These joints should have mirrored constraints
SYMMETRY_PAIRS = [
    (1, 2),   # hip_R <-> hip_L
    (3, 4),   # knee_R <-> knee_L
    (5, 6),   # ankle_R <-> ankle_L
    (8, 9),   # shoulder_R <-> shoulder_L
    (10, 11), # elbow_R <-> elbow_L
]

# Hierarchy edges (biomechanical coupling)
# These represent functional coupling beyond direct bone connections
HIERARCHY_EDGES = [
    (0, 7),   # pelvis <-> trunk (core stability)
    (1, 2),   # hip_R <-> hip_L (pelvis rotation affects both)
    (8, 9),   # shoulder_R <-> shoulder_L (trunk rotation affects both)
    (1, 7),   # hip_R <-> trunk (torso side-bend)
    (2, 7),   # hip_L <-> trunk (torso side-bend)
]


def build_kinematic_adj() -> torch.Tensor:
    """Build adjacency matrix for kinematic chain connections.

    Connects joints based on the skeleton hierarchy (parent-child bones).

    Returns:
        (12, 12) adjacency matrix with self-loops
    """
    adj = torch.zeros(NUM_JOINTS, NUM_JOINTS)

    for parent, child in KINEMATIC_EDGES:
        adj[parent, child] = 1.0
        adj[child, parent] = 1.0  # Bidirectional for message passing

    # Add self-loops
    adj = adj + torch.eye(NUM_JOINTS)

    return adj


def build_symmetry_adj() -> torch.Tensor:
    """Build adjacency matrix for left-right symmetric joint pairs.

    Connects symmetric joints that should have mirrored constraints.

    Returns:
        (12, 12) adjacency matrix with self-loops
    """
    adj = torch.zeros(NUM_JOINTS, NUM_JOINTS)

    for left, right in SYMMETRY_PAIRS:
        adj[left, right] = 1.0
        adj[right, left] = 1.0

    # Add self-loops
    adj = adj + torch.eye(NUM_JOINTS)

    return adj


def build_hierarchy_adj() -> torch.Tensor:
    """Build adjacency matrix for biomechanical hierarchy connections.

    Connects joints based on functional coupling (core stability,
    pelvis-trunk coordination, etc.).

    Returns:
        (12, 12) adjacency matrix with self-loops
    """
    adj = torch.zeros(NUM_JOINTS, NUM_JOINTS)

    for i, j in HIERARCHY_EDGES:
        adj[i, j] = 1.0
        adj[j, i] = 1.0

    # Add self-loops
    adj = adj + torch.eye(NUM_JOINTS)

    return adj


def build_all_adjacencies() -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build all three adjacency matrices.

    Returns:
        (adj_kinematic, adj_symmetry, adj_hierarchy) tuple of (12, 12) matrices
    """
    return (
        build_kinematic_adj(),
        build_symmetry_adj(),
        build_hierarchy_adj(),
    )


def get_combined_adj(include_self_loops: bool = True) -> torch.Tensor:
    """Get a combined adjacency matrix (union of all edge types).

    Useful for basic GCN that uses a single adjacency.

    Args:
        include_self_loops: Whether to add self-connections

    Returns:
        (12, 12) combined adjacency matrix
    """
    adj_kin = build_kinematic_adj()
    adj_sym = build_symmetry_adj()
    adj_hier = build_hierarchy_adj()

    # Union of all edges (remove self-loops first to avoid double-counting)
    adj = (adj_kin + adj_sym + adj_hier).clamp(max=1.0)

    if include_self_loops:
        # Self-loops already included in individual matrices
        pass
    else:
        # Remove self-loops
        adj = adj - torch.eye(NUM_JOINTS)
        adj = adj.clamp(min=0.0)

    return adj


def visualize_adjacency(adj: torch.Tensor, name: str = "Adjacency") -> str:
    """Create ASCII visualization of adjacency matrix.

    Args:
        adj: (12, 12) adjacency matrix
        name: Name for the visualization

    Returns:
        String representation of the matrix
    """
    lines = [f"\n{name} Matrix (12x12):"]

    # Header with joint indices
    header = "          " + " ".join(f"{i:2d}" for i in range(NUM_JOINTS))
    lines.append(header)

    # Rows with joint names
    for i, joint_name in enumerate(JOINT_NAMES):
        # Truncate joint name to 10 chars
        row_str = f"{joint_name:10s}"
        for j in range(NUM_JOINTS):
            val = adj[i, j].item()
            # Use different chars for self-loops vs edges
            if i == j and val > 0.5:
                char = "o"  # self-loop
            elif val > 0.5:
                char = "#"  # edge
            else:
                char = "."  # no edge
            row_str += f" {char} "
        lines.append(row_str)

    return "\n".join(lines)


def count_edges(adj: torch.Tensor, include_self_loops: bool = False) -> int:
    """Count number of edges in adjacency matrix.

    Args:
        adj: Adjacency matrix
        include_self_loops: Whether to count self-loops

    Returns:
        Number of edges (each undirected edge counted once)
    """
    if include_self_loops:
        # Count all non-zero entries, divide by 2 for undirected
        # Self-loops are on diagonal, so we need to handle them separately
        off_diag = adj.sum() - adj.diag().sum()
        on_diag = (adj.diag() > 0).sum()
        return int(off_diag / 2) + int(on_diag)
    else:
        # Count only off-diagonal edges
        off_diag = adj.sum() - adj.diag().sum()
        return int(off_diag / 2)


if __name__ == "__main__":
    # Test and visualize adjacency matrices
    print("Building adjacency matrices for joint refinement...")
    print(f"Number of joints: {NUM_JOINTS}")
    print(f"Joint names: {JOINT_NAMES}")
    print()

    adj_kin = build_kinematic_adj()
    adj_sym = build_symmetry_adj()
    adj_hier = build_hierarchy_adj()
    adj_combined = get_combined_adj()

    print(f"Kinematic edges: {count_edges(adj_kin)}")
    print(f"Symmetry edges: {count_edges(adj_sym)}")
    print(f"Hierarchy edges: {count_edges(adj_hier)}")
    print(f"Combined edges: {count_edges(adj_combined)}")

    print(visualize_adjacency(adj_kin, "Kinematic"))
    print(visualize_adjacency(adj_sym, "Symmetry"))
    print(visualize_adjacency(adj_hier, "Hierarchy"))
    print(visualize_adjacency(adj_combined, "Combined"))

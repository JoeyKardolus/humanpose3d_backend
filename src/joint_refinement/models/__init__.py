"""Joint refinement GNN model variants.

This subpackage contains different GNN architectures for joint refinement:
- gcn: Basic GCN model (JointGCNRefiner)
- semgcn: Semantic GCN with separate edge types (SemGCNJointRefiner)
- temporal: SemGCN with temporal context and sign (SemGCNTemporalJointRefiner)

All models use the same input format for drop-in replacement.
"""

# Import from parent gnn_model for backward compatibility
from ..gnn_model import (
    JointGCNRefiner,
    SemGCNJointRefiner,
    SemGCNTemporalJointRefiner,
    AngleEncoder,
    JointRefinementHead,
    TemporalJointRefinerInference,
    create_gnn_joint_model,
    load_gnn_joint_model,
)

__all__ = [
    "JointGCNRefiner",
    "SemGCNJointRefiner",
    "SemGCNTemporalJointRefiner",
    "AngleEncoder",
    "JointRefinementHead",
    "TemporalJointRefinerInference",
    "create_gnn_joint_model",
    "load_gnn_joint_model",
]

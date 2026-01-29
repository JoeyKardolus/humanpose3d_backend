"""POF GNN model variants.

This subpackage contains different GNN architectures for POF prediction:
- gcn: Basic GCN model (POFGraphModel)
- semgcn: Semantic GCN with separate edge types (SemGCNPOFModel)
- temporal: SemGCN with temporal context and z-sign (SemGCNTemporalZSign)

All models use the same input format for drop-in replacement.
"""

# Import from parent gnn_model for backward compatibility
# The models are defined in the main gnn_model.py and re-exported here
from ..gnn_model import (
    POFGraphModel,
    SemGCNPOFModel,
    SemGCNTemporalZSign,
    LimbFeatureBuilder,
    POFHead,
    TemporalPOFInference,
    create_gnn_pof_model,
    load_gnn_pof_model,
)

__all__ = [
    "POFGraphModel",
    "SemGCNPOFModel",
    "SemGCNTemporalZSign",
    "LimbFeatureBuilder",
    "POFHead",
    "TemporalPOFInference",
    "create_gnn_pof_model",
    "load_gnn_pof_model",
]

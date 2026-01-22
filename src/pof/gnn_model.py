"""Graph Neural Network models for POF prediction.

Alternative to the transformer-based model in model.py.
Uses skeleton graph structure as inductive bias.

Variants:
- POFGraphModel: Basic GCN with combined adjacency
- SemGCNPOFModel: Semantic GCN with separate edge types (joint-sharing, kinematic, symmetry)

Both use the same input format as CameraPOFModel for drop-in replacement.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional

from .constants import NUM_JOINTS, NUM_LIMBS, LIMB_DEFINITIONS
from .graph_utils import (
    build_joint_sharing_adj,
    build_kinematic_adj,
    build_symmetry_adj,
    get_combined_adj,
)


class GCNLayer(nn.Module):
    """Graph Convolutional layer with residual connection.

    Uses normalized adjacency (D^-1 A) for message passing.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        dropout: float = 0.1,
        use_residual: bool = True,
    ):
        super().__init__()
        self.use_residual = use_residual and (in_dim == out_dim)

        self.linear = nn.Linear(in_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, num_nodes, in_dim) node features
            adj: (num_nodes, num_nodes) adjacency matrix

        Returns:
            (batch, num_nodes, out_dim) updated node features
        """
        # Normalize adjacency: D^-1 A (row normalization)
        deg = adj.sum(dim=-1, keepdim=True).clamp(min=1)
        adj_norm = adj / deg

        # Message passing: aggregate neighbor features
        h = torch.matmul(adj_norm, x)  # (batch, num_nodes, in_dim)

        # Transform
        h = self.linear(h)
        h = self.norm(h)
        h = F.gelu(h)
        h = self.dropout(h)

        # Residual connection
        if self.use_residual:
            h = h + x

        return h


class GATLayer(nn.Module):
    """Graph Attention layer (simplified version).

    Learns attention weights for neighbors instead of using fixed adjacency.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = out_dim // num_heads
        assert out_dim % num_heads == 0

        self.q_proj = nn.Linear(in_dim, out_dim)
        self.k_proj = nn.Linear(in_dim, out_dim)
        self.v_proj = nn.Linear(in_dim, out_dim)
        self.out_proj = nn.Linear(out_dim, out_dim)

        self.norm = nn.LayerNorm(out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, num_nodes, in_dim)
            adj: (num_nodes, num_nodes) - used as attention mask

        Returns:
            (batch, num_nodes, out_dim)
        """
        batch_size, num_nodes, _ = x.shape

        # Project to Q, K, V
        q = self.q_proj(x).view(batch_size, num_nodes, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(batch_size, num_nodes, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(batch_size, num_nodes, self.num_heads, self.head_dim)

        # (batch, heads, nodes, head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)

        # Mask non-neighbors (where adj == 0)
        mask = (adj == 0).unsqueeze(0).unsqueeze(0)  # (1, 1, nodes, nodes)
        scores = scores.masked_fill(mask, float('-inf'))

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        # Aggregate
        out = torch.matmul(attn, v)  # (batch, heads, nodes, head_dim)
        out = out.transpose(1, 2).contiguous().view(batch_size, num_nodes, -1)
        out = self.out_proj(out)

        # Residual + norm
        out = self.norm(out + x)

        return out


class LimbFeatureBuilder(nn.Module):
    """Build per-limb features from 2D pose and visibility.

    Same feature extraction as LimbEncoder in model.py but as a standalone module.
    """

    def __init__(self):
        super().__init__()
        # Per-limb features: 9 total
        # - limb_delta_2d: (2,) normalized 2D displacement direction
        # - limb_length_2d: (1,) 2D length (foreshortening)
        # - parent_pos: (2,) parent joint 2D position
        # - child_pos: (2,) child joint 2D position
        # - parent_vis: (1,) parent visibility
        # - child_vis: (1,) child visibility
        self.limb_input_dim = 9

    def forward(
        self,
        pose_2d: torch.Tensor,        # (batch, 17, 2)
        visibility: torch.Tensor,      # (batch, 17)
        limb_delta_2d: torch.Tensor,   # (batch, 14, 2)
        limb_length_2d: torch.Tensor,  # (batch, 14)
    ) -> torch.Tensor:
        """Build per-limb features.

        Returns:
            (batch, 14, 9) per-limb feature vectors
        """
        batch_size = pose_2d.size(0)
        device = pose_2d.device

        # Extract per-limb joint features
        parent_pos = torch.zeros(batch_size, NUM_LIMBS, 2, device=device)
        child_pos = torch.zeros(batch_size, NUM_LIMBS, 2, device=device)
        parent_vis = torch.zeros(batch_size, NUM_LIMBS, device=device)
        child_vis = torch.zeros(batch_size, NUM_LIMBS, device=device)

        for limb_idx, (parent, child) in enumerate(LIMB_DEFINITIONS):
            parent_pos[:, limb_idx] = pose_2d[:, parent]
            child_pos[:, limb_idx] = pose_2d[:, child]
            parent_vis[:, limb_idx] = visibility[:, parent]
            child_vis[:, limb_idx] = visibility[:, child]

        # Concatenate all features: (batch, 14, 9)
        limb_features = torch.cat([
            limb_delta_2d,                     # (batch, 14, 2)
            limb_length_2d.unsqueeze(-1),      # (batch, 14, 1)
            parent_pos,                        # (batch, 14, 2)
            child_pos,                         # (batch, 14, 2)
            parent_vis.unsqueeze(-1),          # (batch, 14, 1)
            child_vis.unsqueeze(-1),           # (batch, 14, 1)
        ], dim=-1)

        return limb_features


class POFHead(nn.Module):
    """Predict POF unit vectors for each limb.

    Shared initial layer + per-limb specialized heads.
    Output is normalized to unit vectors.
    """

    def __init__(self, d_model: int = 128):
        super().__init__()

        # Shared processing
        self.shared = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
        )

        # Per-limb heads
        self.limb_heads = nn.ModuleList([
            nn.Linear(d_model // 2, 3)
            for _ in range(NUM_LIMBS)
        ])

        # Initialize to small weights
        for head in self.limb_heads:
            nn.init.normal_(head.weight, std=0.01)
            nn.init.zeros_(head.bias)

    def forward(self, limb_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            limb_features: (batch, 14, d_model)

        Returns:
            (batch, 14, 3) unit vectors (normalized)
        """
        shared_out = self.shared(limb_features)  # (batch, 14, d_model//2)

        predictions = []
        for i, head in enumerate(self.limb_heads):
            vec = head(shared_out[:, i, :])  # (batch, 3)
            predictions.append(vec)

        pof = torch.stack(predictions, dim=1)  # (batch, 14, 3)

        # Normalize to unit vectors
        pof = F.normalize(pof, dim=-1, eps=1e-6)

        return pof


class POFGraphModel(nn.Module):
    """Basic GCN model for POF prediction.

    Uses combined adjacency matrix (union of all edge types).
    Simpler than SemGCN but may be sufficient for the task.
    """

    def __init__(
        self,
        d_model: int = 128,
        num_layers: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_layers = num_layers

        # Feature extraction
        self.limb_builder = LimbFeatureBuilder()
        self.input_proj = nn.Linear(9, d_model)

        # Build adjacency matrix (combined)
        self.register_buffer('adj', get_combined_adj(include_self_loops=True))

        # GCN layers
        self.gcn_layers = nn.ModuleList([
            GCNLayer(d_model, d_model, dropout=dropout)
            for _ in range(num_layers)
        ])

        # Global context from full pose
        self.global_encoder = nn.Sequential(
            nn.Linear(NUM_JOINTS * 2 + NUM_JOINTS, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )
        self.combine = nn.Linear(d_model * 2, d_model)

        # POF prediction
        self.pof_head = POFHead(d_model)

    def forward(
        self,
        pose_2d: torch.Tensor,         # (batch, 17, 2)
        visibility: torch.Tensor,       # (batch, 17)
        limb_delta_2d: torch.Tensor,    # (batch, 14, 2)
        limb_length_2d: torch.Tensor,   # (batch, 14)
    ) -> torch.Tensor:
        """Predict POF unit vectors.

        Returns:
            (batch, 14, 3) unit vectors for each limb
        """
        batch_size = pose_2d.size(0)

        # Build per-limb features
        limb_features = self.limb_builder(pose_2d, visibility, limb_delta_2d, limb_length_2d)
        h = self.input_proj(limb_features)  # (batch, 14, d_model)

        # GCN message passing with residual connections
        for gcn in self.gcn_layers:
            h = gcn(h, self.adj)

        # Global context
        pose_flat = pose_2d.reshape(batch_size, -1)
        global_features = torch.cat([pose_flat, visibility], dim=-1)
        global_encoded = self.global_encoder(global_features)

        # Combine local (GCN) and global features
        global_broadcast = global_encoded.unsqueeze(1).expand(-1, NUM_LIMBS, -1)
        h = self.combine(torch.cat([h, global_broadcast], dim=-1))

        # Predict POF
        return self.pof_head(h)

    def num_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_config(self) -> Dict[str, Any]:
        """Get model configuration for checkpointing."""
        return {
            "model_type": "gcn",
            "d_model": self.d_model,
            "num_layers": self.num_layers,
        }


class SemGCNPOFModel(nn.Module):
    """Semantic Graph Convolutional Network for POF prediction.

    Uses separate GCN stacks for different edge types:
    - Joint-sharing: Limbs that share at least one joint
    - Kinematic: Parent-child limb dependencies
    - Symmetry: Left-right symmetric pairs

    This provides stronger inductive bias by treating different
    relationships differently.
    """

    def __init__(
        self,
        d_model: int = 128,
        num_layers: int = 4,
        dropout: float = 0.1,
        use_gat: bool = False,
        num_heads: int = 4,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.use_gat = use_gat

        # Feature extraction
        self.limb_builder = LimbFeatureBuilder()
        self.input_proj = nn.Linear(9, d_model)

        # Register adjacency matrices
        self.register_buffer('adj_joint', build_joint_sharing_adj() + torch.eye(NUM_LIMBS))
        self.register_buffer('adj_kinematic', build_kinematic_adj() + torch.eye(NUM_LIMBS))
        self.register_buffer('adj_symmetry', build_symmetry_adj() + torch.eye(NUM_LIMBS))

        # Create layer type
        def make_layer():
            if use_gat:
                return GATLayer(d_model, d_model, num_heads=num_heads, dropout=dropout)
            else:
                return GCNLayer(d_model, d_model, dropout=dropout)

        # Separate GCN/GAT stacks per edge type
        self.gcn_joint = nn.ModuleList([make_layer() for _ in range(num_layers)])
        self.gcn_kinematic = nn.ModuleList([make_layer() for _ in range(num_layers)])
        self.gcn_symmetry = nn.ModuleList([make_layer() for _ in range(num_layers)])

        # Fusion: combine outputs from all edge types
        self.fusion = nn.Sequential(
            nn.Linear(d_model * 3, d_model * 2),
            nn.LayerNorm(d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
        )

        # Global context
        self.global_encoder = nn.Sequential(
            nn.Linear(NUM_JOINTS * 2 + NUM_JOINTS, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )
        self.combine = nn.Linear(d_model * 2, d_model)

        # POF prediction
        self.pof_head = POFHead(d_model)

    def forward(
        self,
        pose_2d: torch.Tensor,         # (batch, 17, 2)
        visibility: torch.Tensor,       # (batch, 17)
        limb_delta_2d: torch.Tensor,    # (batch, 14, 2)
        limb_length_2d: torch.Tensor,   # (batch, 14)
    ) -> torch.Tensor:
        """Predict POF unit vectors using semantic graph convolutions.

        Returns:
            (batch, 14, 3) unit vectors for each limb
        """
        batch_size = pose_2d.size(0)

        # Build per-limb features
        limb_features = self.limb_builder(pose_2d, visibility, limb_delta_2d, limb_length_2d)
        h = self.input_proj(limb_features)  # (batch, 14, d_model)

        # Process through each edge type's GCN stack
        h_joint = h
        h_kin = h
        h_sym = h

        for i in range(self.num_layers):
            h_joint = self.gcn_joint[i](h_joint, self.adj_joint)
            h_kin = self.gcn_kinematic[i](h_kin, self.adj_kinematic)
            h_sym = self.gcn_symmetry[i](h_sym, self.adj_symmetry)

        # Fuse all edge type outputs
        h_fused = self.fusion(torch.cat([h_joint, h_kin, h_sym], dim=-1))

        # Global context
        pose_flat = pose_2d.reshape(batch_size, -1)
        global_features = torch.cat([pose_flat, visibility], dim=-1)
        global_encoded = self.global_encoder(global_features)

        # Combine local (GCN) and global features
        global_broadcast = global_encoded.unsqueeze(1).expand(-1, NUM_LIMBS, -1)
        h = self.combine(torch.cat([h_fused, global_broadcast], dim=-1))

        # Predict POF
        return self.pof_head(h)

    def num_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_config(self) -> Dict[str, Any]:
        """Get model configuration for checkpointing."""
        return {
            "model_type": "semgcn",
            "d_model": self.d_model,
            "num_layers": self.num_layers,
            "use_gat": self.use_gat,
        }


class SemGCNTemporalZSign(nn.Module):
    """Semantic GCN with Z-sign classification head and temporal context.

    Extends SemGCN with:
    1. Z-sign auxiliary head: Binary classification for each limb's Z direction
       (toward camera = 0, away from camera = 1)
    2. Temporal context: Previous frame's POF vectors inform current prediction
       to resolve depth ambiguity in foreshortened limbs

    Key insight: When a limb appears short in 2D (foreshortened), it could be
    pointing toward (+Z) or away (-Z) from camera. Both project identically.
    The temporal context helps because arms don't teleport: if frame 10 shows
    arm forward, frame 11 is probably still forward.
    """

    def __init__(
        self,
        d_model: int = 256,
        num_layers: int = 4,
        dropout: float = 0.1,
        use_gat: bool = False,
        num_heads: int = 4,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.use_gat = use_gat

        # Feature extraction
        self.limb_builder = LimbFeatureBuilder()
        self.input_proj = nn.Linear(9, d_model)

        # Register adjacency matrices
        self.register_buffer('adj_joint', build_joint_sharing_adj() + torch.eye(NUM_LIMBS))
        self.register_buffer('adj_kinematic', build_kinematic_adj() + torch.eye(NUM_LIMBS))
        self.register_buffer('adj_symmetry', build_symmetry_adj() + torch.eye(NUM_LIMBS))

        # Create layer type
        def make_layer():
            if use_gat:
                return GATLayer(d_model, d_model, num_heads=num_heads, dropout=dropout)
            else:
                return GCNLayer(d_model, d_model, dropout=dropout)

        # Separate GCN/GAT stacks per edge type
        self.gcn_joint = nn.ModuleList([make_layer() for _ in range(num_layers)])
        self.gcn_kinematic = nn.ModuleList([make_layer() for _ in range(num_layers)])
        self.gcn_symmetry = nn.ModuleList([make_layer() for _ in range(num_layers)])

        # Fusion: combine outputs from all edge types
        self.fusion = nn.Sequential(
            nn.Linear(d_model * 3, d_model * 2),
            nn.LayerNorm(d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
        )

        # Global context
        self.global_encoder = nn.Sequential(
            nn.Linear(NUM_JOINTS * 2 + NUM_JOINTS, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )
        self.combine = nn.Linear(d_model * 2, d_model)

        # Temporal context: encode previous frame's POF (14×3=42 features)
        self.prev_encoder = nn.Sequential(
            nn.Linear(NUM_LIMBS * 3, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )

        # Gate to combine current + temporal features
        self.temporal_combine = nn.Linear(d_model * 2, d_model)

        # Z-sign classification head (14 limbs)
        # Predicts probability that limb Z > 0 (pointing away from camera)
        self.z_sign_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, NUM_LIMBS),
        )

        # POF prediction head
        self.pof_head = POFHead(d_model)

    def _encode_with_semgcn(
        self,
        pose_2d: torch.Tensor,
        visibility: torch.Tensor,
        limb_delta_2d: torch.Tensor,
        limb_length_2d: torch.Tensor,
    ) -> torch.Tensor:
        """Encode input through SemGCN layers.

        Returns:
            (batch, 14, d_model) encoded limb features
        """
        batch_size = pose_2d.size(0)

        # Build per-limb features
        limb_features = self.limb_builder(pose_2d, visibility, limb_delta_2d, limb_length_2d)
        h = self.input_proj(limb_features)  # (batch, 14, d_model)

        # Process through each edge type's GCN stack
        h_joint = h
        h_kin = h
        h_sym = h

        for i in range(self.num_layers):
            h_joint = self.gcn_joint[i](h_joint, self.adj_joint)
            h_kin = self.gcn_kinematic[i](h_kin, self.adj_kinematic)
            h_sym = self.gcn_symmetry[i](h_sym, self.adj_symmetry)

        # Fuse all edge type outputs
        h_fused = self.fusion(torch.cat([h_joint, h_kin, h_sym], dim=-1))

        # Global context
        pose_flat = pose_2d.reshape(batch_size, -1)
        global_features = torch.cat([pose_flat, visibility], dim=-1)
        global_encoded = self.global_encoder(global_features)

        # Combine local (GCN) and global features
        global_broadcast = global_encoded.unsqueeze(1).expand(-1, NUM_LIMBS, -1)
        h = self.combine(torch.cat([h_fused, global_broadcast], dim=-1))

        return h

    def forward(
        self,
        pose_2d: torch.Tensor,         # (batch, 17, 2)
        visibility: torch.Tensor,       # (batch, 17)
        limb_delta_2d: torch.Tensor,    # (batch, 14, 2)
        limb_length_2d: torch.Tensor,   # (batch, 14)
        prev_pof: Optional[torch.Tensor] = None,  # (batch, 14, 3) previous frame POF
    ) -> tuple:
        """Predict POF unit vectors and Z-sign logits.

        Args:
            pose_2d: (batch, 17, 2) normalized 2D joint positions
            visibility: (batch, 17) per-joint visibility scores
            limb_delta_2d: (batch, 14, 2) normalized 2D displacement vectors
            limb_length_2d: (batch, 14) 2D lengths (foreshortening indicator)
            prev_pof: (batch, 14, 3) previous frame's POF vectors (optional)

        Returns:
            tuple of:
            - pof: (batch, 14, 3) predicted unit vectors for each limb
            - z_sign_logits: (batch, 14) logits for Z > 0 classification
        """
        # Get SemGCN features
        h = self._encode_with_semgcn(pose_2d, visibility, limb_delta_2d, limb_length_2d)
        # h: (batch, 14, d_model)

        # Add temporal context if available
        if prev_pof is not None:
            prev_flat = prev_pof.reshape(prev_pof.size(0), -1)  # (batch, 42)
            prev_feat = self.prev_encoder(prev_flat)  # (batch, d_model)
            prev_broadcast = prev_feat.unsqueeze(1).expand(-1, NUM_LIMBS, -1)
            h = self.temporal_combine(torch.cat([h, prev_broadcast], dim=-1))

        # Predict POF
        pof = self.pof_head(h)  # (batch, 14, 3)

        # Predict Z-sign (global pooling over limbs for classification)
        z_sign_logits = self.z_sign_head(h.mean(dim=1))  # (batch, 14)

        return pof, z_sign_logits

    def forward_pof_only(
        self,
        pose_2d: torch.Tensor,
        visibility: torch.Tensor,
        limb_delta_2d: torch.Tensor,
        limb_length_2d: torch.Tensor,
        prev_pof: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass returning only POF (for compatibility with standard interface).

        Returns:
            (batch, 14, 3) predicted unit vectors
        """
        pof, _ = self.forward(pose_2d, visibility, limb_delta_2d, limb_length_2d, prev_pof)
        return pof

    def num_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_config(self) -> Dict[str, Any]:
        """Get model configuration for checkpointing."""
        return {
            "model_type": "semgcn-temporal",
            "d_model": self.d_model,
            "num_layers": self.num_layers,
            "use_gat": self.use_gat,
        }


class TemporalPOFInference:
    """Inference wrapper for temporal POF model.

    Maintains state (previous frame's POF) across frames
    for video sequences.
    """

    def __init__(self, model: SemGCNTemporalZSign, device: str = "cpu"):
        self.model = model
        self.model.eval()
        self.device = device
        self.prev_pof: Optional[torch.Tensor] = None

    def predict(
        self,
        pose_2d: torch.Tensor,
        visibility: torch.Tensor,
        limb_delta_2d: torch.Tensor,
        limb_length_2d: torch.Tensor,
    ) -> torch.Tensor:
        """Predict POF for current frame.

        Args:
            pose_2d: (batch, 17, 2) normalized 2D positions
            visibility: (batch, 17) visibility scores
            limb_delta_2d: (batch, 14, 2) 2D limb directions
            limb_length_2d: (batch, 14) 2D limb lengths

        Returns:
            (batch, 14, 3) predicted POF vectors
        """
        with torch.no_grad():
            pof, _ = self.model(
                pose_2d.to(self.device),
                visibility.to(self.device),
                limb_delta_2d.to(self.device),
                limb_length_2d.to(self.device),
                prev_pof=self.prev_pof,
            )
            self.prev_pof = pof.detach()  # Store for next frame
            return pof

    def reset(self):
        """Reset temporal state. Call at start of new video."""
        self.prev_pof = None


def create_gnn_pof_model(
    model_type: str = "semgcn",
    d_model: int = 128,
    num_layers: int = 4,
    dropout: float = 0.1,
    use_gat: bool = False,
    verbose: bool = True,
) -> nn.Module:
    """Create GNN-based POF model.

    Args:
        model_type: "gcn", "semgcn", or "semgcn-temporal"
        d_model: Hidden dimension
        num_layers: Number of GCN layers
        dropout: Dropout rate
        use_gat: Use graph attention instead of GCN (SemGCN only)
        verbose: Print model info

    Returns:
        POFGraphModel, SemGCNPOFModel, or SemGCNTemporalZSign
    """
    if model_type == "gcn":
        model = POFGraphModel(
            d_model=d_model,
            num_layers=num_layers,
            dropout=dropout,
        )
    elif model_type == "semgcn":
        model = SemGCNPOFModel(
            d_model=d_model,
            num_layers=num_layers,
            dropout=dropout,
            use_gat=use_gat,
        )
    elif model_type == "semgcn-temporal":
        model = SemGCNTemporalZSign(
            d_model=d_model,
            num_layers=num_layers,
            dropout=dropout,
            use_gat=use_gat,
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    if verbose:
        print(f"Created {model_type.upper()} POF Model:")
        print(f"  d_model={d_model}, layers={num_layers}")
        print(f"  Parameters: {model.num_parameters():,}")

    return model


def load_gnn_pof_model(
    checkpoint_path: str,
    device: str = "cpu",
    verbose: bool = True,
) -> nn.Module:
    """Load GNN POF model from checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file
        device: Device to load model on
        verbose: Print loading info

    Returns:
        Loaded model
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Get config from checkpoint
    config = checkpoint.get("config", {})
    model_type = config.get("model_type", "semgcn")

    model = create_gnn_pof_model(
        model_type=model_type,
        d_model=config.get("d_model", 128),
        num_layers=config.get("num_layers", 4),
        use_gat=config.get("use_gat", False),
        verbose=False,
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)

    if verbose:
        print(f"Loaded {model_type.upper()} POF Model from {checkpoint_path}")
        print(f"  Parameters: {model.num_parameters():,}")

    return model

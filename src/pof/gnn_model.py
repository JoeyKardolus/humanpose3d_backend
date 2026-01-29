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
from typing import Dict, Any, Optional, Tuple

from .constants import NUM_JOINTS, NUM_LIMBS, LIMB_DEFINITIONS
from .graph_utils import (
    build_joint_sharing_adj,
    build_kinematic_adj,
    build_symmetry_adj,
    get_combined_adj,
)

# Import shared NN layers (GCNLayer, GATLayer defined in shared module)
from src.shared.nn_layers import GCNLayer, GATLayer


def build_orientation_from_geometry(
    z_sign_logits: torch.Tensor,
    delta_2d: torch.Tensor,
    bone_lengths: torch.Tensor,
    z_sign_threshold: float = 0.5,
) -> torch.Tensor:
    """DEPRECATED: This approach is flawed.

    This function assumed |Z| could be computed from geometry:
        |Z| = sqrt(bone_length² - ||delta_2d||²)

    This is WRONG because delta_2d from 2D keypoints ≠ 3D (Δx, Δy) projection.
    Perspective distorts: x_2d = f * X_3d / Z_3d
    So ||delta_2d|| ≠ ||(Δx, Δy)_3d|| - you cannot compute |Z| geometrically from 2D.

    The model must LEARN the full 3D orientation from 2D appearance.
    Use the POFHead output directly instead.

    Args:
        z_sign_logits: (batch, 14) logits for P(Z > 0)
        delta_2d: (batch, 14, 2) observed 2D limb displacements (normalized scale)
        bone_lengths: (14,) or (batch, 14) bone lengths (same scale as delta_2d)
        z_sign_threshold: Threshold for Z-sign classification (default 0.5)

    Returns:
        (batch, 14, 3) unit orientation vectors
    """
    import warnings
    warnings.warn(
        "build_orientation_from_geometry is deprecated. "
        "The geometric approach is flawed due to perspective distortion. "
        "Use the model's POF output directly instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    batch_size = z_sign_logits.size(0)
    device = z_sign_logits.device
    dtype = z_sign_logits.dtype

    # Get Z signs from logits
    z_sign_prob = torch.sigmoid(z_sign_logits)  # (batch, 14)
    z_signs = torch.where(
        z_sign_prob > z_sign_threshold,
        torch.ones_like(z_sign_prob),
        -torch.ones_like(z_sign_prob),
    )  # (batch, 14)

    # Handle bone_lengths shape
    if bone_lengths.dim() == 1:
        bone_lengths = bone_lengths.unsqueeze(0).expand(batch_size, -1)

    # Compute ||delta_2d||²
    delta_2d_len_sq = (delta_2d ** 2).sum(dim=-1)  # (batch, 14)

    # Compute |delta_z|² = bone_length² - ||delta_2d||²
    bone_len_sq = bone_lengths ** 2
    delta_z_sq = bone_len_sq - delta_2d_len_sq

    # Handle case where 2D displacement exceeds bone length (noise/error)
    # In this case, limb is nearly perpendicular to camera, |Z| ≈ 0
    delta_z_sq = torch.clamp(delta_z_sq, min=0.0)

    # |delta_z|
    delta_z_mag = torch.sqrt(delta_z_sq)  # (batch, 14)

    # Apply sign
    delta_z = z_signs * delta_z_mag  # (batch, 14)

    # Build 3D displacement: [delta_x, delta_y, delta_z]
    delta_3d = torch.cat([delta_2d, delta_z.unsqueeze(-1)], dim=-1)  # (batch, 14, 3)

    # Convert to unit orientation by dividing by bone length
    # (with safety for zero bone lengths)
    safe_bone_len = torch.clamp(bone_lengths, min=1e-6).unsqueeze(-1)  # (batch, 14, 1)
    orientation = delta_3d / safe_bone_len

    # Re-normalize for numerical safety
    orientation = F.normalize(orientation, dim=-1, eps=1e-6)

    return orientation


# Keep old name as alias for backward compatibility during transition
def build_orientation_from_z_magnitude(
    z_magnitudes: torch.Tensor,
    z_sign_logits: torch.Tensor,
    delta_2d: torch.Tensor,
    z_sign_threshold: float = 0.5,
) -> torch.Tensor:
    """DEPRECATED: Use build_orientation_from_geometry instead.

    This function is kept for backward compatibility but |Z| should be
    computed from geometry, not predicted by a model.
    """
    # Get Z signs from logits
    z_sign_prob = torch.sigmoid(z_sign_logits)
    z_signs = torch.where(
        z_sign_prob > z_sign_threshold,
        torch.ones_like(z_sign_prob),
        -torch.ones_like(z_sign_prob),
    )

    z_magnitudes = torch.clamp(z_magnitudes, 0.0, 1.0)
    xy_magnitude_sq = torch.clamp(1.0 - z_magnitudes ** 2, min=0.0)
    xy_magnitude = torch.sqrt(xy_magnitude_sq)

    delta_2d_norm = F.normalize(delta_2d, dim=-1, eps=1e-6)
    orient_xy = delta_2d_norm * xy_magnitude.unsqueeze(-1)
    orient_z = z_magnitudes * z_signs

    orientation = torch.cat([orient_xy, orient_z.unsqueeze(-1)], dim=-1)
    orientation = F.normalize(orientation, dim=-1, eps=1e-6)

    return orientation



# GCNLayer and GATLayer imported from src.shared.nn_layers


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
    """Predict POF unit vectors for each limb (legacy).

    Shared initial layer + per-limb specialized heads.
    Output is normalized to unit vectors.

    NOTE: This is the legacy head that predicts full 3D vectors.
    For new models, use ZMagnitudeHead which only predicts |Z| foreshortening.
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
    """Semantic GCN Z-sign classifier with temporal context.

    This model predicts ONLY the Z sign (depth direction) for each limb.
    Everything else is determined by observations + geometry:

    | Component | Source |
    |-----------|--------|
    | XY position | 2D observations |
    | |Z| magnitude | geometry: sqrt(bone_length² - ||delta_2d||²) |
    | Z sign | THIS MODEL (the only ambiguity) |

    Key insight: When a limb appears short in 2D (foreshortened), it could be
    pointing toward (+Z) or away (-Z) from camera. Both project identically.
    The Z sign is the ONLY thing we can't derive from observations.

    Temporal context helps resolve ambiguity (arms don't teleport between frames).
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

        # Temporal context: encode previous frame's reconstructed POF (14×3=42 features)
        # We use full POF for temporal context because that's what we reconstruct
        self.prev_encoder = nn.Sequential(
            nn.Linear(NUM_LIMBS * 3, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )

        # Gate to combine current + temporal features
        self.temporal_combine = nn.Linear(d_model * 2, d_model)

        # POF prediction head (predicts full 3D unit vectors)
        # The model learns to predict orientation from 2D appearance
        self.pof_head = POFHead(d_model)

        # Z-sign classification head (auxiliary task)
        # Predicts probability that limb Z > 0 (pointing away from camera)
        # This provides explicit supervision for depth direction disambiguation
        self.z_sign_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, NUM_LIMBS),
        )

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
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict POF unit vectors and Z-sign logits.

        The model learns to predict full 3D orientation from 2D appearance.
        Z-sign is an auxiliary task that helps disambiguate depth direction.

        Args:
            pose_2d: (batch, 17, 2) normalized 2D joint positions
            visibility: (batch, 17) per-joint visibility scores
            limb_delta_2d: (batch, 14, 2) normalized 2D displacement vectors
            limb_length_2d: (batch, 14) 2D lengths (foreshortening indicator)
            prev_pof: (batch, 14, 3) previous frame's POF vectors (temporal context)

        Returns:
            Tuple of:
            - pof: (batch, 14, 3) predicted unit orientation vectors
            - z_sign_logits: (batch, 14) logits for Z > 0 classification
        """
        # Get SemGCN features
        h = self._encode_with_semgcn(pose_2d, visibility, limb_delta_2d, limb_length_2d)
        # h: (batch, 14, d_model)

        # ALWAYS apply temporal context (use zeros if prev_pof is None)
        # This is critical because the model learned weights expecting temporal_combine
        # to always transform the hidden features. Skipping it produces garbage.
        if prev_pof is None:
            prev_pof = torch.zeros(pose_2d.size(0), NUM_LIMBS, 3, device=pose_2d.device)

        prev_flat = prev_pof.reshape(prev_pof.size(0), -1)  # (batch, 42)
        prev_feat = self.prev_encoder(prev_flat)  # (batch, d_model)
        prev_broadcast = prev_feat.unsqueeze(1).expand(-1, NUM_LIMBS, -1)
        h = self.temporal_combine(torch.cat([h, prev_broadcast], dim=-1))

        # Predict POF unit vectors (the main output)
        pof = self.pof_head(h)  # (batch, 14, 3)

        # Predict Z-sign as auxiliary task (global pooling over limbs)
        z_sign_logits = self.z_sign_head(h.mean(dim=1))  # (batch, 14)

        return pof, z_sign_logits

    def forward_with_zsign_correction(
        self,
        pose_2d: torch.Tensor,
        visibility: torch.Tensor,
        limb_delta_2d: torch.Tensor,
        limb_length_2d: torch.Tensor,
        prev_pof: Optional[torch.Tensor] = None,
        z_sign_threshold: float = 0.5,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with optional Z-sign correction.

        Predicts POF and optionally corrects the Z direction if the POF
        prediction disagrees with the Z-sign classification head.

        Args:
            pose_2d: (batch, 17, 2) normalized 2D joint positions
            visibility: (batch, 17) per-joint visibility scores
            limb_delta_2d: (batch, 14, 2) normalized 2D displacement vectors
            limb_length_2d: (batch, 14) 2D lengths
            prev_pof: (batch, 14, 3) previous frame's POF vectors
            z_sign_threshold: Threshold for Z-sign correction (default 0.5)

        Returns:
            tuple of:
            - pof: (batch, 14, 3) POF unit vectors (optionally Z-corrected)
            - z_sign_logits: (batch, 14) logits for Z > 0 classification
        """
        pof, z_sign_logits = self.forward(
            pose_2d, visibility, limb_delta_2d, limb_length_2d, prev_pof
        )

        # Optional Z-sign correction: flip POF.z if it disagrees with z_sign_head
        z_sign_prob = torch.sigmoid(z_sign_logits)
        z_should_be_positive = z_sign_prob > z_sign_threshold
        z_is_positive = pof[:, :, 2] > 0
        needs_flip = z_should_be_positive != z_is_positive

        pof_corrected = pof.clone()
        pof_corrected[:, :, 2] = torch.where(
            needs_flip, -pof[:, :, 2], pof[:, :, 2]
        )
        pof_corrected = F.normalize(pof_corrected, dim=-1, eps=1e-6)

        return pof_corrected, z_sign_logits

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

    def __init__(
        self,
        model: SemGCNTemporalZSign,
        device: str = "cpu",
        use_zsign_correction: bool = True,
    ):
        """Initialize inference wrapper.

        Args:
            model: Trained SemGCNTemporalZSign model
            device: Device to run inference on
            use_zsign_correction: If True, correct POF.z based on z_sign_head
        """
        self.model = model
        self.model.eval()
        self.device = device
        self.use_zsign_correction = use_zsign_correction
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
            pof, z_sign_logits = self.model(
                pose_2d.to(self.device),
                visibility.to(self.device),
                limb_delta_2d.to(self.device),
                limb_length_2d.to(self.device),
                prev_pof=self.prev_pof,
            )

            # Optional Z-sign correction
            if self.use_zsign_correction:
                z_sign_prob = torch.sigmoid(z_sign_logits)
                z_should_be_positive = z_sign_prob > 0.5
                z_is_positive = pof[:, :, 2] > 0
                needs_flip = z_should_be_positive != z_is_positive
                pof = pof.clone()
                pof[:, :, 2] = torch.where(needs_flip, -pof[:, :, 2], pof[:, :, 2])
                pof = F.normalize(pof, dim=-1, eps=1e-6)

            self.prev_pof = pof.detach()  # Store for next frame
            return pof

    def predict_with_zsign(
        self,
        pose_2d: torch.Tensor,
        visibility: torch.Tensor,
        limb_delta_2d: torch.Tensor,
        limb_length_2d: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict POF and Z-sign logits for current frame.

        Returns:
            Tuple of:
            - pof: (batch, 14, 3) predicted POF vectors
            - z_sign_logits: (batch, 14) Z-sign classification logits
        """
        with torch.no_grad():
            pof, z_sign_logits = self.model(
                pose_2d.to(self.device),
                visibility.to(self.device),
                limb_delta_2d.to(self.device),
                limb_length_2d.to(self.device),
                prev_pof=self.prev_pof,
            )
            self.prev_pof = pof.detach()
            return pof, z_sign_logits

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

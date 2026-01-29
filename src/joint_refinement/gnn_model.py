"""Graph Neural Network models for joint angle refinement.

Alternative to the transformer-based model in model.py.
Uses skeleton graph structure as inductive bias for joint angles.

Variants:
- JointGCNRefiner: Basic GCN with combined adjacency
- SemGCNJointRefiner: Semantic GCN with separate edge types (kinematic, symmetry, hierarchy)
- SemGCNTemporalJointRefiner: SemGCN with angle sign classification and temporal context

All use the same input format as JointConstraintRefiner for drop-in replacement.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, Tuple

from .graph_utils import (
    NUM_JOINTS,
    JOINT_NAMES,
    build_kinematic_adj,
    build_symmetry_adj,
    build_hierarchy_adj,
    get_combined_adj,
)

# Import shared NN layers (GCNLayer, GATLayer defined in shared module)
from src.shared.nn_layers import GCNLayer, GATLayer


class AngleEncoder(nn.Module):
    """Encode joint angles with periodic features.

    Angles are periodic, so we use sin/cos encoding to capture this.
    Reused from model.py for consistency.
    """

    def __init__(self, d_model: int = 128):
        super().__init__()
        self.d_model = d_model

        # Project 3 angles (flex, abd, rot) to d_model
        # Use both raw angles and sin/cos for periodicity
        # Input: 3 angles + 6 sin/cos = 9 features per joint
        self.input_proj = nn.Linear(9, d_model)

    def forward(self, angles: torch.Tensor) -> torch.Tensor:
        """
        Args:
            angles: (B, 12, 3) joint angles in degrees

        Returns:
            (B, 12, d_model) encoded angles
        """
        # Convert to radians for sin/cos
        angles_rad = angles * (math.pi / 180.0)

        # Compute sin/cos for periodicity
        sin_angles = torch.sin(angles_rad)
        cos_angles = torch.cos(angles_rad)

        # Concatenate: [raw_angles, sin, cos]
        # Scale raw angles to roughly [-1, 1] range (divide by 180)
        scaled_angles = angles / 180.0
        features = torch.cat([scaled_angles, sin_angles, cos_angles], dim=-1)

        # Project to d_model
        return self.input_proj(features)



# GCNLayer and GATLayer imported from src.shared.nn_layers


class JointRefinementHead(nn.Module):
    """Per-joint refinement head that predicts angle corrections.

    Uses shared initial layer + per-joint specialized heads.
    """

    def __init__(self, d_model: int = 192, n_dof: int = 3):
        super().__init__()
        self.n_joints = NUM_JOINTS
        self.n_dof = n_dof

        # Shared processing
        self.shared = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
        )

        # Per-joint heads
        self.joint_heads = nn.ModuleList([
            nn.Linear(d_model // 2, n_dof)
            for _ in range(self.n_joints)
        ])

        # Initialize to small weights (start near identity)
        for head in self.joint_heads:
            nn.init.normal_(head.weight, std=0.01)
            nn.init.zeros_(head.bias)

    def forward(self, joint_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            joint_features: (batch, 12, d_model)

        Returns:
            (batch, 12, 3) angle corrections (delta)
        """
        shared_out = self.shared(joint_features)  # (batch, 12, d_model//2)

        predictions = []
        for i, head in enumerate(self.joint_heads):
            delta = head(shared_out[:, i, :])  # (batch, 3)
            predictions.append(delta)

        return torch.stack(predictions, dim=1)  # (batch, 12, 3)


class JointGCNRefiner(nn.Module):
    """Basic GCN model for joint angle refinement.

    Uses combined adjacency matrix (union of all edge types).
    Simpler than SemGCN but may be sufficient for the task.
    """

    def __init__(
        self,
        d_model: int = 192,
        num_layers: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.n_joints = NUM_JOINTS
        self.n_dof = 3

        # Angle encoder
        self.angle_encoder = AngleEncoder(d_model)

        # Visibility embedding
        self.vis_embedding = nn.Linear(1, d_model // 4)
        self.feature_proj = nn.Linear(d_model + d_model // 4, d_model)

        # Build adjacency matrix (combined)
        self.register_buffer('adj', get_combined_adj(include_self_loops=True))

        # GCN layers
        self.gcn_layers = nn.ModuleList([
            GCNLayer(d_model, d_model, dropout=dropout)
            for _ in range(num_layers)
        ])

        # Refinement head
        self.refinement_head = JointRefinementHead(d_model, n_dof=3)

    def forward(
        self,
        angles: torch.Tensor,
        visibility: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Refine joint angles using GCN.

        Args:
            angles: (B, 12, 3) joint angles in degrees
            visibility: (B, 12) optional per-joint visibility

        Returns:
            refined_angles: (B, 12, 3) refined joint angles
            delta: (B, 12, 3) predicted corrections
        """
        B = angles.shape[0]

        # Encode angles
        h = self.angle_encoder(angles)  # (B, 12, d_model)

        # Add visibility features
        if visibility is not None:
            vis_feat = self.vis_embedding(visibility.unsqueeze(-1))
            h = torch.cat([h, vis_feat], dim=-1)
            h = self.feature_proj(h)
        else:
            vis_feat = torch.zeros(B, self.n_joints, self.d_model // 4, device=h.device)
            h = torch.cat([h, vis_feat], dim=-1)
            h = self.feature_proj(h)

        # GCN message passing
        for gcn in self.gcn_layers:
            h = gcn(h, self.adj)

        # Predict corrections
        delta = self.refinement_head(h)

        # Refined angles = input + delta
        refined_angles = angles + delta

        return refined_angles, delta

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


class SemGCNJointRefiner(nn.Module):
    """Semantic Graph Convolutional Network for joint angle refinement.

    Uses separate GCN stacks for different edge types:
    - Kinematic: Parent-child skeleton hierarchy
    - Symmetry: Left-right symmetric pairs
    - Hierarchy: Biomechanical coupling

    This provides stronger inductive bias by treating different
    relationships differently.
    """

    def __init__(
        self,
        d_model: int = 192,
        num_layers: int = 4,
        dropout: float = 0.1,
        use_gat: bool = False,
        num_heads: int = 4,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.use_gat = use_gat
        self.n_joints = NUM_JOINTS
        self.n_dof = 3

        # Angle encoder
        self.angle_encoder = AngleEncoder(d_model)

        # Visibility embedding
        self.vis_embedding = nn.Linear(1, d_model // 4)
        self.feature_proj = nn.Linear(d_model + d_model // 4, d_model)

        # Register adjacency matrices
        self.register_buffer('adj_kinematic', build_kinematic_adj())
        self.register_buffer('adj_symmetry', build_symmetry_adj())
        self.register_buffer('adj_hierarchy', build_hierarchy_adj())

        # Create layer type
        def make_layer():
            if use_gat:
                return GATLayer(d_model, d_model, num_heads=num_heads, dropout=dropout)
            else:
                return GCNLayer(d_model, d_model, dropout=dropout)

        # Separate GCN/GAT stacks per edge type
        self.gcn_kinematic = nn.ModuleList([make_layer() for _ in range(num_layers)])
        self.gcn_symmetry = nn.ModuleList([make_layer() for _ in range(num_layers)])
        self.gcn_hierarchy = nn.ModuleList([make_layer() for _ in range(num_layers)])

        # Fusion: combine outputs from all edge types
        self.fusion = nn.Sequential(
            nn.Linear(d_model * 3, d_model * 2),
            nn.LayerNorm(d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
        )

        # Refinement head
        self.refinement_head = JointRefinementHead(d_model, n_dof=3)

    def forward(
        self,
        angles: torch.Tensor,
        visibility: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Refine joint angles using semantic GCN.

        Args:
            angles: (B, 12, 3) joint angles in degrees
            visibility: (B, 12) optional per-joint visibility

        Returns:
            refined_angles: (B, 12, 3) refined joint angles
            delta: (B, 12, 3) predicted corrections
        """
        B = angles.shape[0]

        # Encode angles
        h = self.angle_encoder(angles)  # (B, 12, d_model)

        # Add visibility features
        if visibility is not None:
            vis_feat = self.vis_embedding(visibility.unsqueeze(-1))
            h = torch.cat([h, vis_feat], dim=-1)
            h = self.feature_proj(h)
        else:
            vis_feat = torch.zeros(B, self.n_joints, self.d_model // 4, device=h.device)
            h = torch.cat([h, vis_feat], dim=-1)
            h = self.feature_proj(h)

        # Process through each edge type's GCN stack
        h_kin = h
        h_sym = h
        h_hier = h

        for i in range(self.num_layers):
            h_kin = self.gcn_kinematic[i](h_kin, self.adj_kinematic)
            h_sym = self.gcn_symmetry[i](h_sym, self.adj_symmetry)
            h_hier = self.gcn_hierarchy[i](h_hier, self.adj_hierarchy)

        # Fuse all edge type outputs
        h_fused = self.fusion(torch.cat([h_kin, h_sym, h_hier], dim=-1))

        # Predict corrections
        delta = self.refinement_head(h_fused)

        # Refined angles = input + delta
        refined_angles = angles + delta

        return refined_angles, delta

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


class SemGCNTemporalJointRefiner(nn.Module):
    """Semantic GCN with angle sign classification and temporal context.

    Extends SemGCN with:
    1. Sign auxiliary head: Binary classification for abd/rot sign per joint
       Helps resolve ambiguity when joint is near neutral position
    2. Temporal context: Previous frame's angles inform current prediction
       to maintain consistency across frames

    Key insight: Like POF's Z-sign disambiguation, joint angles near zero
    have sign ambiguity. The temporal context helps because joints don't
    flip sign between frames arbitrarily.
    """

    def __init__(
        self,
        d_model: int = 192,
        num_layers: int = 4,
        dropout: float = 0.1,
        use_gat: bool = False,
        num_heads: int = 4,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.use_gat = use_gat
        self.n_joints = NUM_JOINTS
        self.n_dof = 3

        # Angle encoder
        self.angle_encoder = AngleEncoder(d_model)

        # Visibility embedding
        self.vis_embedding = nn.Linear(1, d_model // 4)
        self.feature_proj = nn.Linear(d_model + d_model // 4, d_model)

        # Register adjacency matrices
        self.register_buffer('adj_kinematic', build_kinematic_adj())
        self.register_buffer('adj_symmetry', build_symmetry_adj())
        self.register_buffer('adj_hierarchy', build_hierarchy_adj())

        # Create layer type
        def make_layer():
            if use_gat:
                return GATLayer(d_model, d_model, num_heads=num_heads, dropout=dropout)
            else:
                return GCNLayer(d_model, d_model, dropout=dropout)

        # Separate GCN/GAT stacks per edge type
        self.gcn_kinematic = nn.ModuleList([make_layer() for _ in range(num_layers)])
        self.gcn_symmetry = nn.ModuleList([make_layer() for _ in range(num_layers)])
        self.gcn_hierarchy = nn.ModuleList([make_layer() for _ in range(num_layers)])

        # Fusion: combine outputs from all edge types
        self.fusion = nn.Sequential(
            nn.Linear(d_model * 3, d_model * 2),
            nn.LayerNorm(d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
        )

        # Temporal context: encode previous frame's angles (12 joints x 3 DOF = 36 features)
        self.prev_encoder = nn.Sequential(
            nn.Linear(NUM_JOINTS * 3, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )

        # Gate to combine current + temporal features
        self.temporal_combine = nn.Linear(d_model * 2, d_model)

        # Sign classification head (12 joints x 2 = abd and rot signs)
        # Predicts probability that abd > 0 and rot > 0 for each joint
        self.sign_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, NUM_JOINTS * 2),  # 12 joints x 2 (abd, rot)
        )

        # Refinement head
        self.refinement_head = JointRefinementHead(d_model, n_dof=3)

    def _encode_with_semgcn(
        self,
        angles: torch.Tensor,
        visibility: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Encode input through SemGCN layers.

        Returns:
            (batch, 12, d_model) encoded joint features
        """
        B = angles.shape[0]

        # Encode angles
        h = self.angle_encoder(angles)  # (B, 12, d_model)

        # Add visibility features
        if visibility is not None:
            vis_feat = self.vis_embedding(visibility.unsqueeze(-1))
            h = torch.cat([h, vis_feat], dim=-1)
            h = self.feature_proj(h)
        else:
            vis_feat = torch.zeros(B, self.n_joints, self.d_model // 4, device=h.device)
            h = torch.cat([h, vis_feat], dim=-1)
            h = self.feature_proj(h)

        # Process through each edge type's GCN stack
        h_kin = h
        h_sym = h
        h_hier = h

        for i in range(self.num_layers):
            h_kin = self.gcn_kinematic[i](h_kin, self.adj_kinematic)
            h_sym = self.gcn_symmetry[i](h_sym, self.adj_symmetry)
            h_hier = self.gcn_hierarchy[i](h_hier, self.adj_hierarchy)

        # Fuse all edge type outputs
        h_fused = self.fusion(torch.cat([h_kin, h_sym, h_hier], dim=-1))

        return h_fused

    def forward(
        self,
        angles: torch.Tensor,
        visibility: Optional[torch.Tensor] = None,
        prev_angles: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Refine joint angles with temporal context and sign prediction.

        Args:
            angles: (B, 12, 3) joint angles in degrees
            visibility: (B, 12) optional per-joint visibility
            prev_angles: (B, 12, 3) previous frame's angles (optional)

        Returns:
            refined_angles: (B, 12, 3) refined joint angles
            delta: (B, 12, 3) predicted corrections
            sign_logits: (B, 12, 2) logits for abd > 0 and rot > 0
        """
        # Get SemGCN features
        h = self._encode_with_semgcn(angles, visibility)
        # h: (batch, 12, d_model)

        # Add temporal context if available
        if prev_angles is not None:
            prev_flat = prev_angles.reshape(prev_angles.size(0), -1)  # (batch, 36)
            prev_feat = self.prev_encoder(prev_flat)  # (batch, d_model)
            prev_broadcast = prev_feat.unsqueeze(1).expand(-1, self.n_joints, -1)
            h = self.temporal_combine(torch.cat([h, prev_broadcast], dim=-1))

        # Predict corrections
        delta = self.refinement_head(h)  # (batch, 12, 3)

        # Predict sign (global pooling over joints for classification)
        sign_logits = self.sign_head(h.mean(dim=1))  # (batch, 24)
        sign_logits = sign_logits.view(-1, NUM_JOINTS, 2)  # (batch, 12, 2)

        # Refined angles = input + delta
        refined_angles = angles + delta

        return refined_angles, delta, sign_logits

    def forward_no_sign(
        self,
        angles: torch.Tensor,
        visibility: Optional[torch.Tensor] = None,
        prev_angles: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning only refined angles and delta (for compatibility).

        Returns:
            refined_angles: (B, 12, 3) refined joint angles
            delta: (B, 12, 3) predicted corrections
        """
        refined, delta, _ = self.forward(angles, visibility, prev_angles)
        return refined, delta

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


class TemporalJointRefinerInference:
    """Inference wrapper for temporal joint refinement model.

    Maintains state (previous frame's angles) across frames
    for video sequences.
    """

    def __init__(self, model: SemGCNTemporalJointRefiner, device: str = "cpu"):
        self.model = model
        self.model.eval()
        self.device = device
        self.prev_angles: Optional[torch.Tensor] = None

    def refine(
        self,
        angles: torch.Tensor,
        visibility: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Refine angles for current frame.

        Args:
            angles: (batch, 12, 3) or (12, 3) joint angles in degrees
            visibility: (batch, 12) or (12,) visibility scores

        Returns:
            (batch, 12, 3) or (12, 3) refined angles
        """
        # Handle single frame
        single_frame = angles.dim() == 2
        if single_frame:
            angles = angles.unsqueeze(0)
            if visibility is not None:
                visibility = visibility.unsqueeze(0)

        with torch.no_grad():
            refined, _, _ = self.model(
                angles.to(self.device),
                visibility.to(self.device) if visibility is not None else None,
                prev_angles=self.prev_angles,
            )
            self.prev_angles = refined.detach()  # Store for next frame

        if single_frame:
            return refined.squeeze(0)
        return refined

    def reset(self):
        """Reset temporal state. Call at start of new video."""
        self.prev_angles = None


def create_gnn_joint_model(
    model_type: str = "semgcn",
    d_model: int = 192,
    num_layers: int = 4,
    dropout: float = 0.1,
    use_gat: bool = False,
    verbose: bool = True,
) -> nn.Module:
    """Create GNN-based joint refinement model.

    Args:
        model_type: "gcn", "semgcn", or "semgcn-temporal"
        d_model: Hidden dimension (default 192 for ~1M params target)
        num_layers: Number of GCN layers
        dropout: Dropout rate
        use_gat: Use graph attention instead of GCN (SemGCN only)
        verbose: Print model info

    Returns:
        JointGCNRefiner, SemGCNJointRefiner, or SemGCNTemporalJointRefiner
    """
    if model_type == "gcn":
        model = JointGCNRefiner(
            d_model=d_model,
            num_layers=num_layers,
            dropout=dropout,
        )
    elif model_type == "semgcn":
        model = SemGCNJointRefiner(
            d_model=d_model,
            num_layers=num_layers,
            dropout=dropout,
            use_gat=use_gat,
        )
    elif model_type == "semgcn-temporal":
        model = SemGCNTemporalJointRefiner(
            d_model=d_model,
            num_layers=num_layers,
            dropout=dropout,
            use_gat=use_gat,
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    if verbose:
        print(f"Created {model_type.upper()} Joint Refiner:")
        print(f"  d_model={d_model}, layers={num_layers}")
        print(f"  Parameters: {model.num_parameters():,}")

    return model


def load_gnn_joint_model(
    checkpoint_path: str,
    device: str = "cpu",
    verbose: bool = True,
) -> nn.Module:
    """Load GNN joint refinement model from checkpoint.

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

    model = create_gnn_joint_model(
        model_type=model_type,
        d_model=config.get("d_model", 192),
        num_layers=config.get("num_layers", 4),
        use_gat=config.get("use_gat", False),
        verbose=False,
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)

    if verbose:
        print(f"Loaded {model_type.upper()} Joint Refiner from {checkpoint_path}")
        print(f"  Parameters: {model.num_parameters():,}")

    return model


if __name__ == "__main__":
    # Test model creation
    print("Testing GNN Joint Refinement Models")
    print("=" * 50)

    for model_type in ["gcn", "semgcn", "semgcn-temporal"]:
        print(f"\n{model_type.upper()}:")
        model = create_gnn_joint_model(model_type=model_type, d_model=192, num_layers=4)

        # Test forward pass
        B = 4
        angles = torch.randn(B, 12, 3) * 30  # Random angles
        visibility = torch.rand(B, 12)  # Random visibility

        if model_type == "semgcn-temporal":
            # Test with and without temporal context
            refined, delta, sign_logits = model(angles, visibility)
            print(f"  Output shapes: refined={refined.shape}, delta={delta.shape}, sign={sign_logits.shape}")

            # Test with previous angles
            prev_angles = torch.randn(B, 12, 3) * 30
            refined2, delta2, sign_logits2 = model(angles, visibility, prev_angles)
            print(f"  With prev_angles: refined={refined2.shape}")
        else:
            refined, delta = model(angles, visibility)
            print(f"  Output shapes: refined={refined.shape}, delta={delta.shape}")

        print(f"  Delta mean: {delta.abs().mean():.4f} (should be ~0 initially)")

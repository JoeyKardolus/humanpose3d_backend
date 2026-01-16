"""
Neural joint constraint refinement model.

Architecture:
- Input: joint angles (12, 3) computed by validated ISB kinematics
- Cross-joint transformer attention learns pose context
- Per-joint refinement heads predict angle corrections
- Output: refined angles (12, 3)

The model learns realistic joint constraints from AIST++ data,
not hard-coded limits. It understands that:
- Bent knee allows different hip ROM
- Arms up changes trunk constraints
- Joint angles are interdependent

Uses the validated ISB kinematics for angle computation (src/kinematics/).
This model only learns to REFINE those angles, not compute them.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


# Joint order (must match dataset.py and training data)
JOINT_NAMES = [
    'pelvis', 'hip_R', 'hip_L', 'knee_R', 'knee_L',
    'ankle_R', 'ankle_L', 'trunk', 'shoulder_R', 'shoulder_L',
    'elbow_R', 'elbow_L',
]

# Kinematic chain connections (for attention bias)
# Each joint is connected to its parent and children
KINEMATIC_CHAINS = {
    'pelvis': ['hip_R', 'hip_L', 'trunk'],
    'hip_R': ['pelvis', 'knee_R'],
    'hip_L': ['pelvis', 'knee_L'],
    'knee_R': ['hip_R', 'ankle_R'],
    'knee_L': ['hip_L', 'ankle_L'],
    'ankle_R': ['knee_R'],
    'ankle_L': ['knee_L'],
    'trunk': ['pelvis', 'shoulder_R', 'shoulder_L'],
    'shoulder_R': ['trunk', 'elbow_R'],
    'shoulder_L': ['trunk', 'elbow_L'],
    'elbow_R': ['shoulder_R'],
    'elbow_L': ['shoulder_L'],
}


def build_adjacency_matrix() -> torch.Tensor:
    """Build adjacency matrix for kinematic chain connections."""
    n_joints = len(JOINT_NAMES)
    adj = torch.zeros(n_joints, n_joints)

    name_to_idx = {name: i for i, name in enumerate(JOINT_NAMES)}

    for joint, neighbors in KINEMATIC_CHAINS.items():
        i = name_to_idx[joint]
        for neighbor in neighbors:
            j = name_to_idx[neighbor]
            adj[i, j] = 1.0
            adj[j, i] = 1.0

    # Self connections
    adj = adj + torch.eye(n_joints)

    return adj


class AngleEncoder(nn.Module):
    """Encode joint angles with periodic features.

    Angles are periodic, so we use sin/cos encoding to capture this.
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


class CrossJointAttention(nn.Module):
    """Cross-joint attention layer.

    Allows each joint to attend to other joints based on:
    1. Kinematic chain proximity (bias toward neighbors)
    2. Learned attention patterns
    """

    def __init__(
        self,
        d_model: int = 128,
        n_heads: int = 4,
        dropout: float = 0.1,
        use_kinematic_bias: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.use_kinematic_bias = use_kinematic_bias

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

        # Kinematic adjacency bias (learnable scaling)
        if use_kinematic_bias:
            self.register_buffer('adjacency', build_adjacency_matrix())
            self.kinematic_bias_scale = nn.Parameter(torch.tensor(1.0))

    def forward(
        self,
        x: torch.Tensor,
        visibility: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (B, 12, d_model) joint features
            visibility: (B, 12) optional visibility mask

        Returns:
            (B, 12, d_model) attended features
        """
        B, N, D = x.shape

        # Project to Q, K, V
        q = self.q_proj(x).view(B, N, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, N, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, N, self.n_heads, self.head_dim).transpose(1, 2)

        # Attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # Add kinematic chain bias
        if self.use_kinematic_bias:
            # Adjacency is (12, 12), expand to (1, 1, 12, 12)
            bias = self.adjacency.unsqueeze(0).unsqueeze(0) * self.kinematic_bias_scale
            scores = scores + bias

        # Visibility mask (optional)
        if visibility is not None:
            # Mask out attention to low-visibility joints
            # visibility: (B, 12) -> (B, 1, 1, 12)
            vis_mask = visibility.unsqueeze(1).unsqueeze(2)
            scores = scores - 10.0 * (1.0 - vis_mask)

        # Softmax attention
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        # Apply attention to values
        out = torch.matmul(attn, v)

        # Reshape and project
        out = out.transpose(1, 2).contiguous().view(B, N, D)
        out = self.out_proj(out)

        return out


class TransformerBlock(nn.Module):
    """Transformer block with cross-joint attention."""

    def __init__(
        self,
        d_model: int = 128,
        n_heads: int = 4,
        dropout: float = 0.1,
        use_kinematic_bias: bool = True,
    ):
        super().__init__()

        self.attention = CrossJointAttention(
            d_model, n_heads, dropout, use_kinematic_bias
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        visibility: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Self-attention with residual
        x = x + self.attention(self.norm1(x), visibility)

        # FFN with residual
        x = x + self.ffn(self.norm2(x))

        return x


class JointConstraintRefiner(nn.Module):
    """Neural joint constraint refinement model.

    Takes joint angles (12, 3) and refines them using learned constraints.

    Architecture:
    1. Encode angles with periodic features (sin/cos)
    2. Cross-joint attention captures pose context
    3. Per-joint heads predict angle corrections (delta)

    The model learns soft constraints from data distribution,
    not hard-coded joint limits.
    """

    def __init__(
        self,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 4,
        dropout: float = 0.1,
        use_kinematic_bias: bool = True,
    ):
        super().__init__()

        self.d_model = d_model
        self.n_joints = 12
        self.n_dof = 3

        # Angle encoder
        self.angle_encoder = AngleEncoder(d_model)

        # Visibility embedding (optional)
        self.vis_embedding = nn.Linear(1, d_model // 4)

        # Combine angle and visibility features
        self.feature_proj = nn.Linear(d_model + d_model // 4, d_model)

        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads, dropout, use_kinematic_bias)
            for _ in range(n_layers)
        ])

        # Per-joint refinement heads
        # Each joint gets a small MLP to predict delta angles
        self.joint_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.GELU(),
                nn.Linear(d_model // 2, 3),  # 3 DOF per joint
            )
            for _ in range(self.n_joints)
        ])

        # Initialize outputs to zero (start with identity)
        for head in self.joint_heads:
            nn.init.zeros_(head[-1].weight)
            nn.init.zeros_(head[-1].bias)

    def forward(
        self,
        angles: torch.Tensor,
        visibility: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            angles: (B, 12, 3) joint angles in degrees
            visibility: (B, 12) optional per-joint visibility (0-1)

        Returns:
            refined_angles: (B, 12, 3) refined joint angles
            delta: (B, 12, 3) predicted corrections
        """
        B = angles.shape[0]

        # Encode angles
        x = self.angle_encoder(angles)  # (B, 12, d_model)

        # Add visibility features if provided
        if visibility is not None:
            vis_feat = self.vis_embedding(visibility.unsqueeze(-1))  # (B, 12, d_model//4)
            x = torch.cat([x, vis_feat], dim=-1)
            x = self.feature_proj(x)
        else:
            # No visibility - use learned default
            vis_feat = torch.zeros(B, self.n_joints, self.d_model // 4, device=x.device)
            x = torch.cat([x, vis_feat], dim=-1)
            x = self.feature_proj(x)

        # Transformer layers
        for layer in self.layers:
            x = layer(x, visibility)

        # Per-joint refinement
        delta = torch.zeros(B, self.n_joints, self.n_dof, device=angles.device)
        for i, head in enumerate(self.joint_heads):
            delta[:, i, :] = head(x[:, i, :])

        # Refined angles = input + delta
        refined_angles = angles + delta

        return refined_angles, delta

    def num_parameters(self) -> int:
        """Return number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_model(
    d_model: int = 128,
    n_heads: int = 4,
    n_layers: int = 4,
    dropout: float = 0.1,
    use_kinematic_bias: bool = True,
) -> JointConstraintRefiner:
    """Create joint constraint refiner model.

    Args:
        d_model: Hidden dimension
        n_heads: Number of attention heads
        n_layers: Number of transformer layers
        dropout: Dropout rate
        use_kinematic_bias: Whether to use kinematic chain attention bias

    Returns:
        JointConstraintRefiner model
    """
    model = JointConstraintRefiner(
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        dropout=dropout,
        use_kinematic_bias=use_kinematic_bias,
    )

    print(f"Created JointConstraintRefiner with {model.num_parameters():,} parameters")
    print(f"  d_model: {d_model}, n_heads: {n_heads}, n_layers: {n_layers}")

    return model


if __name__ == '__main__':
    # Test model
    model = create_model()

    # Test forward pass
    B = 4
    angles = torch.randn(B, 12, 3) * 30  # Random angles in degrees
    visibility = torch.rand(B, 12)  # Random visibility

    refined, delta = model(angles, visibility)

    print(f"\nInput angles shape: {angles.shape}")
    print(f"Refined angles shape: {refined.shape}")
    print(f"Delta shape: {delta.shape}")

    print(f"\nDelta stats:")
    print(f"  Mean: {delta.abs().mean():.4f}")
    print(f"  Max: {delta.abs().max():.4f}")
    print(f"  (should be ~0 with fresh initialization)")

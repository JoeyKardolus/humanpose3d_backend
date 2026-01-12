"""
Pose-Aware Depth Refinement Network.

Learns to correct MediaPipe depth errors using:
1. Cross-joint attention (which joints inform which for depth)
2. View angle conditioning (different patterns for frontal vs profile)
3. Visibility-weighted learning (low confidence joints need more correction)

Architecture:
- Joint encoder: Embed each joint's (x, y, z, visibility) -> 64 features
- Cross-joint transformer: Learn inter-joint depth relationships
- View-conditioned MLP: Apply angle-specific correction patterns
- Output: Per-joint depth correction (delta_z)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for joint identity."""

    def __init__(self, d_model: int, max_joints: int = 17):
        super().__init__()
        pe = torch.zeros(max_joints, d_model)
        position = torch.arange(0, max_joints, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to joint features.

        Args:
            x: (batch, num_joints, d_model)

        Returns:
            x + positional encoding
        """
        return x + self.pe[:x.size(1)]


class ViewAngleEncoder(nn.Module):
    """Encode view angle into a feature vector.

    Uses Fourier features to capture periodic angle patterns:
    - 0° (frontal) has symmetric depth errors
    - 45° (angled) has asymmetric patterns
    - 90° (profile) has completely different error distribution
    """

    def __init__(self, d_model: int, num_frequencies: int = 8):
        super().__init__()
        self.num_frequencies = num_frequencies

        # MLP to process Fourier features
        self.mlp = nn.Sequential(
            nn.Linear(2 * num_frequencies + 1, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )

    def forward(self, angle: torch.Tensor) -> torch.Tensor:
        """Encode view angle to feature vector.

        Args:
            angle: (batch,) view angle in degrees (0-90)

        Returns:
            (batch, d_model) angle embedding
        """
        # Normalize to [0, 1]
        angle_norm = angle / 90.0

        # Fourier features for periodic patterns
        freqs = 2 ** torch.arange(
            self.num_frequencies, device=angle.device, dtype=angle.dtype
        )
        scaled = angle_norm.unsqueeze(-1) * freqs * math.pi

        # Concat [angle, sin(2^k * pi * angle), cos(2^k * pi * angle)]
        features = torch.cat([
            angle_norm.unsqueeze(-1),
            torch.sin(scaled),
            torch.cos(scaled),
        ], dim=-1)

        return self.mlp(features)


class CrossJointAttention(nn.Module):
    """Transformer encoder for cross-joint depth inference.

    Key insight: To correct depth of joint A, we need information from
    joints B, C, D that are anatomically connected or have reliable estimates.

    Example learned patterns:
    - Wrist depth inferred from elbow + shoulder configuration
    - Far-side ankle depth inferred from near-side ankle (at 45° view)
    - Occluded hip depth inferred from visible knee + ankle chain
    """

    def __init__(
        self,
        d_model: int = 64,
        num_heads: int = 4,
        num_layers: int = 4,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,  # Pre-norm for stability
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

    def forward(
        self,
        joint_features: torch.Tensor,
        visibility_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """Apply cross-joint attention.

        Args:
            joint_features: (batch, num_joints, d_model)
            visibility_mask: (batch, num_joints) low visibility = less reliable

        Returns:
            (batch, num_joints, d_model) attended features
        """
        # Optional: could use visibility to mask attention weights
        # But for now, let the network learn this implicitly
        return self.transformer(joint_features)


class PoseAwareDepthRefiner(nn.Module):
    """
    Main network: Learns to correct depth using pose context and view angle.

    The key insight is that depth errors are SYSTEMATIC:
    - They correlate with view angle
    - They can be inferred from other visible joints
    - They follow anatomical constraints

    Input:
        pose: (batch, 17, 3) - x, y, z coordinates per joint
        visibility: (batch, 17) - MediaPipe visibility scores (0-1)
        view_angle: (batch,) - viewing angle in degrees (0-90)

    Output:
        delta_z: (batch, 17) - depth corrections per joint
        confidence: (batch, 17) - confidence in corrections (optional)
    """

    def __init__(
        self,
        num_joints: int = 17,
        d_model: int = 64,
        num_heads: int = 4,
        num_layers: int = 4,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        output_confidence: bool = True,
    ):
        super().__init__()

        self.num_joints = num_joints
        self.d_model = d_model
        self.output_confidence = output_confidence

        # Joint encoder: (x, y, z, visibility) -> d_model
        # Input: 4 features per joint
        self.joint_encoder = nn.Sequential(
            nn.Linear(4, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )

        # Positional encoding for joint identity
        self.pos_encoder = PositionalEncoding(d_model, max_joints=num_joints)

        # View angle encoder
        self.view_encoder = ViewAngleEncoder(d_model, num_frequencies=8)

        # Cross-joint attention transformer
        self.cross_joint_attn = CrossJointAttention(
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )

        # Output heads
        self.depth_head = nn.Sequential(
            nn.Linear(d_model * 2, d_model),  # *2 for view concat
            nn.ReLU(),
            nn.Linear(d_model, 1),
        )

        if output_confidence:
            self.confidence_head = nn.Sequential(
                nn.Linear(d_model * 2, d_model),
                nn.ReLU(),
                nn.Linear(d_model, 1),
                nn.Sigmoid(),
            )

    def forward(
        self,
        pose: torch.Tensor,
        visibility: torch.Tensor,
        view_angle: torch.Tensor,
    ) -> dict:
        """
        Forward pass.

        Args:
            pose: (batch, 17, 3) joint positions
            visibility: (batch, 17) per-joint visibility
            view_angle: (batch,) view angle in degrees

        Returns:
            dict with:
                'delta_z': (batch, 17) depth corrections
                'confidence': (batch, 17) correction confidence (if enabled)
        """
        batch_size = pose.size(0)

        # 1. Combine pose + visibility as joint features
        joint_input = torch.cat([
            pose,
            visibility.unsqueeze(-1),
        ], dim=-1)  # (batch, 17, 4)

        # 2. Encode each joint
        joint_features = self.joint_encoder(joint_input)  # (batch, 17, d_model)

        # 3. Add positional encoding (joint identity)
        joint_features = self.pos_encoder(joint_features)

        # 4. Cross-joint attention
        attended = self.cross_joint_attn(joint_features, visibility)  # (batch, 17, d_model)

        # 5. Encode view angle
        view_features = self.view_encoder(view_angle)  # (batch, d_model)

        # 6. Concatenate view features to each joint
        view_expanded = view_features.unsqueeze(1).expand(-1, self.num_joints, -1)
        combined = torch.cat([attended, view_expanded], dim=-1)  # (batch, 17, d_model*2)

        # 7. Predict depth corrections
        delta_z = self.depth_head(combined).squeeze(-1)  # (batch, 17)

        output = {'delta_z': delta_z}

        if self.output_confidence:
            confidence = self.confidence_head(combined).squeeze(-1)  # (batch, 17)
            output['confidence'] = confidence

        return output


def create_model(
    num_joints: int = 17,
    d_model: int = 64,
    num_heads: int = 4,
    num_layers: int = 4,
    **kwargs,
) -> PoseAwareDepthRefiner:
    """Create depth refinement model with default settings.

    Default config is optimized for AIST++ COCO 17 joints:
    - d_model=64: Compact but expressive
    - num_layers=4: Enough for cross-joint reasoning
    - num_heads=4: 16-dim per head

    Total params: ~500K (lightweight for real-time inference)
    """
    return PoseAwareDepthRefiner(
        num_joints=num_joints,
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
        **kwargs,
    )


if __name__ == '__main__':
    # Quick test
    model = create_model()
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Test forward pass
    batch_size = 4
    pose = torch.randn(batch_size, 17, 3)
    visibility = torch.rand(batch_size, 17)
    view_angle = torch.rand(batch_size) * 90

    output = model(pose, visibility, view_angle)
    print(f"delta_z shape: {output['delta_z'].shape}")
    print(f"confidence shape: {output['confidence'].shape}")

"""
Main Refiner Model with Learned Constraint Fusion.

This model learns to optimally combine outputs from:
1. Depth Refinement Model (PoseAwareDepthRefiner) - provides depth corrections
2. Joint Angle Refinement Model (JointConstraintRefiner) - provides angle corrections

Architecture:
- Encodes outputs from each constraint model into a unified feature space
- Cross-attention allows depth and joint features to inform each other
- Gating network learns when to trust each constraint source per-joint
- Fusion head combines weighted outputs into final pose

Key insight: Different joints benefit from different constraints:
- Limb endpoints (wrists, ankles) benefit from depth model (occlusion handling)
- Core joints (pelvis, hips) benefit from joint model (anatomical consistency)
- The model learns this from data rather than hard-coded rules.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, Optional


# COCO-17 joint indices
COCO_JOINTS = {
    'nose': 0, 'l_eye': 1, 'r_eye': 2, 'l_ear': 3, 'r_ear': 4,
    'l_shoulder': 5, 'r_shoulder': 6, 'l_elbow': 7, 'r_elbow': 8,
    'l_wrist': 9, 'r_wrist': 10, 'l_hip': 11, 'r_hip': 12,
    'l_knee': 13, 'r_knee': 14, 'l_ankle': 15, 'r_ankle': 16,
}

# Joint refinement model joint order (12 joints)
JOINT_MODEL_ORDER = [
    'pelvis', 'hip_R', 'hip_L', 'knee_R', 'knee_L',
    'ankle_R', 'ankle_L', 'trunk', 'shoulder_R', 'shoulder_L',
    'elbow_R', 'elbow_L',
]

# Mapping from 12 joint model joints to 17 COCO joints
# -1 means no direct mapping (model uses learned interpolation)
JOINT12_TO_COCO17 = {
    0: -1,     # pelvis -> no direct COCO equivalent (use hip center)
    1: 12,     # hip_R -> r_hip
    2: 11,     # hip_L -> l_hip
    3: 14,     # knee_R -> r_knee
    4: 13,     # knee_L -> l_knee
    5: 16,     # ankle_R -> r_ankle
    6: 15,     # ankle_L -> l_ankle
    7: -1,     # trunk -> no direct COCO equivalent (use shoulder center)
    8: 6,      # shoulder_R -> r_shoulder
    9: 5,      # shoulder_L -> l_shoulder
    10: 8,     # elbow_R -> r_elbow
    11: 7,     # elbow_L -> l_elbow
}

# Limb definitions (from depth model) - 14 limbs
LIMBS = [
    (5, 7),    # 0: L shoulder → elbow
    (7, 9),    # 1: L elbow → wrist
    (6, 8),    # 2: R shoulder → elbow
    (8, 10),   # 3: R elbow → wrist
    (11, 13),  # 4: L hip → knee
    (13, 15),  # 5: L knee → ankle
    (12, 14),  # 6: R hip → knee
    (14, 16),  # 7: R knee → ankle
    (5, 6),    # 8: Shoulder width
    (11, 12),  # 9: Hip width
    (5, 11),   # 10: L torso
    (6, 12),   # 11: R torso
    (5, 12),   # 12: L shoulder → R hip
    (6, 11),   # 13: R shoulder → L hip
]


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
        return x + self.pe[:x.size(1)]


class LimbToJointMapper(nn.Module):
    """Map 14 limb orientations to 17 per-joint features.

    Each joint receives features from limbs that connect to it.
    Joints not connected to any limb get a learned default.
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model

        # Encode limb orientations
        self.limb_encoder = nn.Linear(3, d_model)

        # Build mapping: for each COCO joint, which limbs connect to it?
        # A limb connects to a joint if that joint is parent or child
        self.joint_to_limbs = {}
        for joint_idx in range(17):
            connected_limbs = []
            for limb_idx, (parent, child) in enumerate(LIMBS):
                if joint_idx == parent or joint_idx == child:
                    connected_limbs.append(limb_idx)
            self.joint_to_limbs[joint_idx] = connected_limbs

        # For joints with no connected limbs (face joints), use learned embedding
        self.default_embedding = nn.Parameter(torch.zeros(d_model))

        # Combine multiple limb features per joint
        self.combiner = nn.Linear(d_model, d_model)

    def forward(self, limb_orientations: torch.Tensor) -> torch.Tensor:
        """
        Args:
            limb_orientations: (batch, 14, 3) unit vectors per limb

        Returns:
            (batch, 17, d_model) per-joint features from limb orientations
        """
        batch_size = limb_orientations.size(0)
        device = limb_orientations.device

        # Encode all limb orientations
        limb_features = self.limb_encoder(limb_orientations)  # (batch, 14, d_model)

        # Map to 17 joints
        joint_features = torch.zeros(batch_size, 17, self.d_model, device=device)

        for joint_idx in range(17):
            connected_limbs = self.joint_to_limbs[joint_idx]
            if len(connected_limbs) > 0:
                # Average features from connected limbs
                limb_indices = torch.tensor(connected_limbs, device=device)
                connected_features = limb_features[:, limb_indices, :]  # (batch, n_limbs, d_model)
                joint_features[:, joint_idx] = connected_features.mean(dim=1)
            else:
                # Use learned default for unconnected joints (face)
                joint_features[:, joint_idx] = self.default_embedding

        # Final projection
        return self.combiner(joint_features)


class Joint12To17Mapper(nn.Module):
    """Map 12 joint model outputs to 17 COCO joint features.

    Direct mappings for joints that exist in both systems.
    Learned interpolation for joints without direct mapping.
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model

        # For joints without direct mapping, learn to interpolate
        # Pelvis (index 0 in joint model) -> interpolate from hips
        # Trunk (index 7 in joint model) -> interpolate from shoulders

        # Learned embedding for face joints (no angle info)
        self.face_embedding = nn.Parameter(torch.zeros(d_model))

        # Learned embedding for wrists (no direct joint model output)
        self.wrist_embedding = nn.Parameter(torch.zeros(d_model))

        # Project features
        self.output_proj = nn.Linear(d_model, d_model)

    def forward(self, joint_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            joint_features: (batch, 12, d_model) from JointOutputEncoder

        Returns:
            (batch, 17, d_model) mapped to COCO joint order
        """
        batch_size = joint_features.size(0)
        device = joint_features.device

        output = torch.zeros(batch_size, 17, self.d_model, device=device)

        # Direct mappings (12 joint model -> COCO 17)
        for j12_idx, coco_idx in JOINT12_TO_COCO17.items():
            if coco_idx >= 0:
                output[:, coco_idx] = joint_features[:, j12_idx]

        # Interpolated mappings:
        # Nose (0): average of shoulders
        output[:, 0] = (joint_features[:, 8] + joint_features[:, 9]) / 2  # shoulders

        # Eyes (1, 2): use face embedding
        output[:, 1] = self.face_embedding
        output[:, 2] = self.face_embedding

        # Ears (3, 4): use face embedding
        output[:, 3] = self.face_embedding
        output[:, 4] = self.face_embedding

        # Wrists (9, 10): use wrist embedding + elbow info
        output[:, 9] = joint_features[:, 11] + self.wrist_embedding  # l_wrist from l_elbow
        output[:, 10] = joint_features[:, 10] + self.wrist_embedding  # r_wrist from r_elbow

        return self.output_proj(output)


class DepthOutputEncoder(nn.Module):
    """Encode depth model outputs into feature space."""

    def __init__(self, d_model: int):
        super().__init__()

        # Per-joint features: delta_xyz (3) + confidence (1) = 4
        self.joint_encoder = nn.Sequential(
            nn.Linear(4, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
        )

        # Limb orientation features: 14 limbs -> per-joint
        self.limb_to_joint = LimbToJointMapper(d_model)

        # Camera angle embedding: azimuth + elevation
        self.camera_encoder = nn.Sequential(
            nn.Linear(4, d_model // 2),  # sin/cos of az, sin/cos of el
            nn.ReLU(),
            nn.Linear(d_model // 2, d_model),
        )

        # Combine all features: joint (d_model) + limb (d_model) + camera (d_model)
        self.fusion = nn.Sequential(
            nn.Linear(d_model * 3, d_model * 2),
            nn.LayerNorm(d_model * 2),
            nn.ReLU(),
            nn.Linear(d_model * 2, d_model),
        )

    def forward(
        self,
        depth_outputs: Dict[str, torch.Tensor],
        raw_pose: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            depth_outputs: dict with:
                - 'delta_xyz': (B, 17, 3)
                - 'confidence': (B, 17)
                - 'pred_limb_orientations': (B, 14, 3) optional
                - 'pred_azimuth': (B,)
                - 'pred_elevation': (B,)
            raw_pose: (B, 17, 3) input pose

        Returns:
            (B, 17, d_model) encoded depth features
        """
        batch_size = raw_pose.size(0)
        device = raw_pose.device

        # Encode per-joint corrections
        delta_xyz = depth_outputs['delta_xyz']
        confidence = depth_outputs.get('confidence', torch.ones(batch_size, 17, device=device))

        joint_input = torch.cat([
            delta_xyz,
            confidence.unsqueeze(-1),
        ], dim=-1)  # (B, 17, 4)
        joint_feat = self.joint_encoder(joint_input)  # (B, 17, d_model)

        # Encode limb orientations -> per-joint
        if 'pred_limb_orientations' in depth_outputs:
            limb_feat = self.limb_to_joint(depth_outputs['pred_limb_orientations'])
        else:
            # If no limb orientations, use zeros
            limb_feat = torch.zeros_like(joint_feat)

        # Encode camera angles -> broadcast to all joints
        azimuth = depth_outputs.get('pred_azimuth', torch.zeros(batch_size, device=device))
        elevation = depth_outputs.get('pred_elevation', torch.zeros(batch_size, device=device))

        # Convert to sin/cos for smooth encoding
        az_rad = azimuth * (math.pi / 180.0)
        el_rad = elevation * (math.pi / 180.0)
        camera_input = torch.stack([
            torch.sin(az_rad), torch.cos(az_rad),
            torch.sin(el_rad), torch.cos(el_rad),
        ], dim=-1)  # (B, 4)

        camera_feat = self.camera_encoder(camera_input)  # (B, d_model)
        camera_feat = camera_feat.unsqueeze(1).expand(-1, 17, -1)  # (B, 17, d_model)

        # Fuse all features
        combined = torch.cat([joint_feat, limb_feat, camera_feat], dim=-1)
        return self.fusion(combined)  # (B, 17, d_model)


class JointOutputEncoder(nn.Module):
    """Encode joint model outputs into feature space."""

    def __init__(self, d_model: int):
        super().__init__()

        # Per-joint features: delta_angles (3) + refined_angles (3) = 6
        # Angles are in degrees, normalize by dividing by 180
        self.angle_encoder = nn.Sequential(
            nn.Linear(6, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
        )

        # Map 12 joint model joints to 17 COCO joints
        self.joint_12_to_17 = Joint12To17Mapper(d_model)

    def forward(self, joint_outputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Args:
            joint_outputs: dict with:
                - 'refined_angles': (B, 12, 3)
                - 'delta_angles': (B, 12, 3)

        Returns:
            (B, 17, d_model) encoded joint features
        """
        # Normalize angles to roughly [-1, 1] range
        delta_angles = joint_outputs['delta_angles'] / 180.0
        refined_angles = joint_outputs['refined_angles'] / 180.0

        # Encode per-joint angles
        joint_input = torch.cat([delta_angles, refined_angles], dim=-1)  # (B, 12, 6)
        joint_feat = self.angle_encoder(joint_input)  # (B, 12, d_model)

        # Map to 17 joints
        return self.joint_12_to_17(joint_feat)  # (B, 17, d_model)


class CrossModelAttention(nn.Module):
    """Cross-attention between depth and joint model features.

    Allows each source to attend to the other, learning which
    features from the other model are most informative.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()

        # Self-attention layers for each modality
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )

        self.depth_self_attn = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        encoder_layer2 = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.joint_self_attn = nn.TransformerEncoder(encoder_layer2, num_layers=num_layers)

        # Cross-attention: depth attends to joint
        self.depth_cross_attn = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True
        )
        self.depth_cross_norm = nn.LayerNorm(d_model)

        # Cross-attention: joint attends to depth
        self.joint_cross_attn = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True
        )
        self.joint_cross_norm = nn.LayerNorm(d_model)

        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
        )

    def forward(
        self,
        depth_features: torch.Tensor,
        joint_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            depth_features: (B, 17, d_model)
            joint_features: (B, 17, d_model)

        Returns:
            (B, 17, d_model) fused features
        """
        # Self-attention within each modality
        depth_feat = self.depth_self_attn(depth_features)
        joint_feat = self.joint_self_attn(joint_features)

        # Cross-attention: depth attends to joint
        depth_cross, _ = self.depth_cross_attn(
            query=depth_feat,
            key=joint_feat,
            value=joint_feat,
        )
        depth_feat = self.depth_cross_norm(depth_feat + depth_cross)

        # Cross-attention: joint attends to depth
        joint_cross, _ = self.joint_cross_attn(
            query=joint_feat,
            key=depth_feat,
            value=depth_feat,
        )
        joint_feat = self.joint_cross_norm(joint_feat + joint_cross)

        # Fuse both streams
        combined = torch.cat([depth_feat, joint_feat], dim=-1)
        return self.fusion(combined)


class GatingNetwork(nn.Module):
    """Learn per-joint weights for each constraint source.

    Outputs weights that indicate how much to trust depth vs joint
    model for each joint. Uses softmax so weights sum to 1.
    """

    def __init__(self, d_model: int):
        super().__init__()

        # Input: fused features after cross-attention
        # Output: weights for [depth, joint, residual] per joint
        self.gate = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 3),  # depth_weight, joint_weight, residual_weight
        )

    def forward(self, fused_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            fused_features: (B, 17, d_model)

        Returns:
            dict with:
                'depth_weights': (B, 17) how much to trust depth model
                'joint_weights': (B, 17) how much to trust joint model
                'residual_weights': (B, 17) how much to add learned correction
        """
        raw_weights = self.gate(fused_features)  # (B, 17, 3)

        # Softmax so weights sum to 1
        weights = F.softmax(raw_weights, dim=-1)

        return {
            'depth_weights': weights[:, :, 0],
            'joint_weights': weights[:, :, 1],
            'residual_weights': weights[:, :, 2],
        }


class FusionHead(nn.Module):
    """Combine weighted constraint outputs into final pose."""

    def __init__(self, d_model: int):
        super().__init__()

        # Residual correction head (learned additional corrections)
        self.correction_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 3),  # x, y, z correction
        )

        # Initialize with small weights for small initial corrections
        nn.init.normal_(self.correction_head[-1].weight, std=0.01)
        nn.init.zeros_(self.correction_head[-1].bias)

    def forward(
        self,
        fused_features: torch.Tensor,
        depth_delta: torch.Tensor,
        gating_weights: Dict[str, torch.Tensor],
        raw_pose: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            fused_features: (B, 17, d_model)
            depth_delta: (B, 17, 3) depth model corrections
            gating_weights: dict with depth_weights, joint_weights, residual_weights
            raw_pose: (B, 17, 3) original input pose

        Returns:
            (B, 17, 3) refined pose
        """
        # Weighted depth correction
        depth_weight = gating_weights['depth_weights'].unsqueeze(-1)  # (B, 17, 1)
        weighted_depth = depth_delta * depth_weight

        # Learned residual correction
        residual_weight = gating_weights['residual_weights'].unsqueeze(-1)
        learned_correction = self.correction_head(fused_features)
        weighted_residual = learned_correction * residual_weight

        # Final pose = raw + weighted corrections
        # Note: joint model provides angle corrections, not position corrections
        # So we only use depth_delta and learned residual for position
        refined_pose = raw_pose + weighted_depth + weighted_residual

        return refined_pose


class MainRefiner(nn.Module):
    """
    Main refiner that learns to combine depth and joint constraint models.

    Uses learned gating to decide when to trust each constraint source.
    Trained with GT to learn optimal fusion strategy.

    Architecture:
    1. Encode outputs from depth model (delta_xyz, confidence, limb orientations, camera angles)
    2. Encode outputs from joint model (delta_angles, refined_angles)
    3. Cross-attention between both feature streams
    4. Gating network learns per-joint weights for each source
    5. Fusion head combines weighted outputs into final pose
    """

    def __init__(
        self,
        d_model: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.d_model = d_model

        # Constraint Model Encoders
        self.depth_encoder = DepthOutputEncoder(d_model)
        self.joint_encoder = JointOutputEncoder(d_model)

        # Raw pose encoder (for residual learning)
        self.pose_encoder = nn.Sequential(
            nn.Linear(4, d_model),  # x, y, z, visibility
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )

        # Positional encoding for joint identity
        self.pos_encoder = PositionalEncoding(d_model, max_joints=17)

        # Cross-attention fusion
        self.cross_attention = CrossModelAttention(
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
        )

        # Combine with raw pose features
        self.feature_combiner = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
        )

        # Gating Network
        self.gating_network = GatingNetwork(d_model)

        # Fusion Head
        self.fusion_head = FusionHead(d_model)

        # Confidence Head
        self.confidence_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        raw_pose: torch.Tensor,
        visibility: torch.Tensor,
        depth_outputs: Dict[str, torch.Tensor],
        joint_outputs: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            raw_pose: (B, 17, 3) raw 3D pose
            visibility: (B, 17) per-joint visibility
            depth_outputs: dict from PoseAwareDepthRefiner
            joint_outputs: dict from JointConstraintRefiner

        Returns:
            dict with:
                'refined_pose': (B, 17, 3) final fused pose
                'depth_weights': (B, 17) how much depth model was used
                'joint_weights': (B, 17) how much joint model was used
                'residual_weights': (B, 17) how much learned correction was used
                'confidence': (B, 17) overall confidence
        """
        batch_size = raw_pose.size(0)

        # Encode raw pose with visibility
        pose_input = torch.cat([
            raw_pose,
            visibility.unsqueeze(-1),
        ], dim=-1)  # (B, 17, 4)
        pose_features = self.pose_encoder(pose_input)  # (B, 17, d_model)
        pose_features = self.pos_encoder(pose_features)

        # Encode constraint model outputs
        depth_features = self.depth_encoder(depth_outputs, raw_pose)  # (B, 17, d_model)
        joint_features = self.joint_encoder(joint_outputs)  # (B, 17, d_model)

        # Cross-attention between depth and joint
        fused_features = self.cross_attention(depth_features, joint_features)

        # Combine with raw pose features
        combined = torch.cat([fused_features, pose_features], dim=-1)
        final_features = self.feature_combiner(combined)

        # Gating network learns which source to trust
        gating_weights = self.gating_network(final_features)

        # Fusion head produces final pose
        refined_pose = self.fusion_head(
            final_features,
            depth_outputs['delta_xyz'],
            gating_weights,
            raw_pose,
        )

        # Confidence
        confidence = self.confidence_head(final_features).squeeze(-1)

        return {
            'refined_pose': refined_pose,
            'depth_weights': gating_weights['depth_weights'],
            'joint_weights': gating_weights['joint_weights'],
            'residual_weights': gating_weights['residual_weights'],
            'confidence': confidence,
        }

    def num_parameters(self) -> int:
        """Return number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_model(
    d_model: int = 128,
    num_heads: int = 4,
    num_layers: int = 2,
    dropout: float = 0.1,
) -> MainRefiner:
    """Create main refiner model with default settings.

    Args:
        d_model: Hidden dimension (default: 128)
        num_heads: Number of attention heads (default: 4)
        num_layers: Number of transformer layers (default: 2)
        dropout: Dropout rate (default: 0.1)

    Returns:
        MainRefiner model
    """
    model = MainRefiner(
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
        dropout=dropout,
    )

    print(f"Created MainRefiner with {model.num_parameters():,} parameters")
    print(f"  d_model: {d_model}, num_heads: {num_heads}, num_layers: {num_layers}")

    return model


if __name__ == '__main__':
    # Test model
    print("Testing MainRefiner model...")

    model = create_model()

    # Test forward pass with mock data
    B = 4

    raw_pose = torch.randn(B, 17, 3)
    visibility = torch.rand(B, 17)

    # Mock depth model outputs
    depth_outputs = {
        'delta_xyz': torch.randn(B, 17, 3) * 0.1,
        'confidence': torch.rand(B, 17),
        'pred_limb_orientations': F.normalize(torch.randn(B, 14, 3), dim=-1),
        'pred_azimuth': torch.rand(B) * 360,
        'pred_elevation': (torch.rand(B) - 0.5) * 180,
    }

    # Mock joint model outputs
    joint_outputs = {
        'refined_angles': torch.randn(B, 12, 3) * 30,
        'delta_angles': torch.randn(B, 12, 3) * 5,
    }

    output = model(raw_pose, visibility, depth_outputs, joint_outputs)

    print(f"\nOutput shapes:")
    for key, value in output.items():
        print(f"  {key}: {value.shape}")

    print(f"\nGating weights (should sum to ~1.0 per joint):")
    total_weights = (
        output['depth_weights'] +
        output['joint_weights'] +
        output['residual_weights']
    )
    print(f"  Sum of weights: {total_weights.mean():.4f} (should be ~1.0)")

    print(f"\nPose correction magnitude:")
    delta = output['refined_pose'] - raw_pose
    print(f"  Mean: {delta.abs().mean():.4f}")
    print(f"  Max: {delta.abs().max():.4f}")

    print(f"\nConfidence stats:")
    print(f"  Mean: {output['confidence'].mean():.4f}")
    print(f"  Range: [{output['confidence'].min():.4f}, {output['confidence'].max():.4f}]")

    print("\nModel test passed!")

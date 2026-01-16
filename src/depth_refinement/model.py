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


class ResidualBlock(nn.Module):
    """Residual block as used in ElePose.

    Simple skip connection: out = block(x) + x
    """

    def __init__(self, dim: int, use_batchnorm: bool = True):
        super().__init__()
        layers = [nn.Linear(dim, dim)]
        if use_batchnorm:
            layers.append(nn.BatchNorm1d(dim))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        layers.append(nn.Linear(dim, dim))
        if use_batchnorm:
            layers.append(nn.BatchNorm1d(dim))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class ElePoseBackbone(nn.Module):
    """ElePose-style 2D pose feature extractor (shared backbone).

    Based on ElePose (CVPR 2022) architecture:
    - Input: 2D pose only (2 * num_joints features)
    - Upscale to hidden_dim (1024 in original)
    - Residual blocks for feature extraction
    - Returns features (no prediction head) for fusion with other features

    Key insight: 2D pose foreshortening directly encodes camera viewpoint.
    - Shoulder width ratio encodes azimuth
    - Torso compression encodes elevation
    - Limb length ratios encode both

    This backbone extracts deep 2D foreshortening features that can be
    combined with other features (transformer context, 3D pose, visibility)
    for improved angle prediction.

    Reference: https://github.com/bastianwandt/ElePose
    """

    def __init__(
        self,
        num_joints: int = 17,
        hidden_dim: int = 1024,
        num_blocks: int = 2,
        use_batchnorm: bool = True,
    ):
        super().__init__()
        self.num_joints = num_joints
        self.hidden_dim = hidden_dim
        input_dim = 2 * num_joints  # Flattened 2D pose

        # Upscale from 2D pose to hidden dimension
        self.upscale = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim) if use_batchnorm else nn.Identity(),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Residual blocks for feature extraction
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(hidden_dim, use_batchnorm) for _ in range(num_blocks)]
        )

    def forward(self, pose_2d: torch.Tensor) -> torch.Tensor:
        """Extract features from 2D pose.

        Args:
            pose_2d: (batch, num_joints, 2) 2D joint positions

        Returns:
            features: (batch, hidden_dim) deep 2D foreshortening features
        """
        batch_size = pose_2d.size(0)

        # Flatten 2D pose
        x = pose_2d.view(batch_size, -1)  # (batch, 2*num_joints)

        # Upscale and process
        x = self.upscale(x)  # (batch, hidden_dim)
        x = self.res_blocks(x)  # (batch, hidden_dim)

        return x


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
    """Encode azimuth + elevation into a feature vector.

    Uses Fourier features to capture periodic angle patterns:
    - Azimuth 0-360°: 0°=front, 90°=right, 180°=back, 270°=left
    - Elevation: angle above/below horizontal

    Depth error patterns vary with viewing angle:
    - 0° (frontal) has symmetric depth errors
    - 90° (profile) has maximum depth error on far side
    - Elevation affects vertical depth ambiguity
    """

    def __init__(self, d_model: int, num_frequencies: int = 8):
        super().__init__()
        self.num_frequencies = num_frequencies

        # Input: azimuth (1) + elevation (1) + Fourier features for each
        # Azimuth: 2*num_freq (sin+cos for full 360° periodicity)
        # Elevation: 2*num_freq (sin+cos)
        input_dim = 2 + 4 * num_frequencies

        # MLP to process combined features
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )

    def forward(self, azimuth: torch.Tensor, elevation: torch.Tensor) -> torch.Tensor:
        """Encode azimuth + elevation to feature vector.

        Args:
            azimuth: (batch,) azimuth in degrees (0-360)
            elevation: (batch,) elevation in degrees (-90 to +90)

        Returns:
            (batch, d_model) angle embedding
        """
        # Normalize azimuth to [0, 1] for full circle
        az_norm = azimuth / 360.0
        # Normalize elevation to [-1, 1]
        el_norm = elevation / 90.0

        # Fourier features for periodic patterns
        freqs = 2 ** torch.arange(
            self.num_frequencies, device=azimuth.device, dtype=azimuth.dtype
        )

        # Azimuth Fourier features (full 360° periodicity)
        az_scaled = az_norm.unsqueeze(-1) * freqs * 2 * math.pi
        az_fourier = torch.cat([torch.sin(az_scaled), torch.cos(az_scaled)], dim=-1)

        # Elevation Fourier features
        el_scaled = el_norm.unsqueeze(-1) * freqs * math.pi
        el_fourier = torch.cat([torch.sin(el_scaled), torch.cos(el_scaled)], dim=-1)

        # Concat all features
        features = torch.cat([
            az_norm.unsqueeze(-1),
            el_norm.unsqueeze(-1),
            az_fourier,
            el_fourier,
        ], dim=-1)

        return self.mlp(features)


class Pose2DEncoder(nn.Module):
    """Encode 2D pose features that are informative for camera viewpoint.

    Based on ElePose (CVPR 2022) insight: 2D appearance directly encodes camera viewpoint.

    Key 2D features that encode viewpoint:
    1. Foreshortening: limb appears shorter when pointing toward/away from camera
    2. Relative positions: shoulder heights equal = frontal view
    3. Left/right asymmetry: encodes azimuth angle
    4. Limb length ratios: perspective distortion
    """

    def __init__(self, d_model: int = 64, num_joints: int = 17):
        super().__init__()
        self.num_joints = num_joints

        # Encode raw 2D coordinates (foreshortening visible here)
        self.coord_encoder = nn.Sequential(
            nn.Linear(num_joints * 2, d_model * 2),
            nn.ReLU(),
            nn.Linear(d_model * 2, d_model),
        )

        # Hand-crafted viewpoint features
        # Indices in COCO format for key joints
        self.feature_dim = 15  # Number of hand-crafted features

        self.feature_encoder = nn.Sequential(
            nn.Linear(self.feature_dim, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )

        # Combine both
        self.combiner = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
        )

    def _extract_viewpoint_features(self, pose_2d: torch.Tensor) -> torch.Tensor:
        """Extract hand-crafted features that encode camera viewpoint.

        Args:
            pose_2d: (batch, 17, 2) normalized 2D coordinates

        Returns:
            (batch, feature_dim) viewpoint-informative features
        """
        # COCO joint indices
        L_SHOULDER, R_SHOULDER = 5, 6
        L_HIP, R_HIP = 11, 12
        L_ELBOW, R_ELBOW = 7, 8
        L_WRIST, R_WRIST = 9, 10
        L_KNEE, R_KNEE = 13, 14
        L_ANKLE, R_ANKLE = 15, 16
        NOSE = 0

        features = []

        # 1. Shoulder height difference (encodes body rotation around vertical axis)
        shoulder_height_diff = pose_2d[:, L_SHOULDER, 1] - pose_2d[:, R_SHOULDER, 1]
        features.append(shoulder_height_diff.unsqueeze(-1))

        # 2. Hip height difference
        hip_height_diff = pose_2d[:, L_HIP, 1] - pose_2d[:, R_HIP, 1]
        features.append(hip_height_diff.unsqueeze(-1))

        # 3. Shoulder width (foreshortening: narrower when viewed from side)
        shoulder_width = torch.norm(pose_2d[:, L_SHOULDER] - pose_2d[:, R_SHOULDER], dim=-1)
        features.append(shoulder_width.unsqueeze(-1))

        # 4. Hip width (foreshortening)
        hip_width = torch.norm(pose_2d[:, L_HIP] - pose_2d[:, R_HIP], dim=-1)
        features.append(hip_width.unsqueeze(-1))

        # 5-6. Left vs Right arm length ratio (encodes azimuth - one arm closer to camera)
        l_arm_len = (
            torch.norm(pose_2d[:, L_SHOULDER] - pose_2d[:, L_ELBOW], dim=-1) +
            torch.norm(pose_2d[:, L_ELBOW] - pose_2d[:, L_WRIST], dim=-1)
        )
        r_arm_len = (
            torch.norm(pose_2d[:, R_SHOULDER] - pose_2d[:, R_ELBOW], dim=-1) +
            torch.norm(pose_2d[:, R_ELBOW] - pose_2d[:, R_WRIST], dim=-1)
        )
        arm_ratio = l_arm_len / (r_arm_len + 1e-6)
        features.append(arm_ratio.unsqueeze(-1))
        features.append((l_arm_len - r_arm_len).unsqueeze(-1))

        # 7-8. Left vs Right leg length ratio
        l_leg_len = (
            torch.norm(pose_2d[:, L_HIP] - pose_2d[:, L_KNEE], dim=-1) +
            torch.norm(pose_2d[:, L_KNEE] - pose_2d[:, L_ANKLE], dim=-1)
        )
        r_leg_len = (
            torch.norm(pose_2d[:, R_HIP] - pose_2d[:, R_KNEE], dim=-1) +
            torch.norm(pose_2d[:, R_KNEE] - pose_2d[:, R_ANKLE], dim=-1)
        )
        leg_ratio = l_leg_len / (r_leg_len + 1e-6)
        features.append(leg_ratio.unsqueeze(-1))
        features.append((l_leg_len - r_leg_len).unsqueeze(-1))

        # 9. Torso height (hip to shoulder center)
        torso_top = (pose_2d[:, L_SHOULDER] + pose_2d[:, R_SHOULDER]) / 2
        torso_bottom = (pose_2d[:, L_HIP] + pose_2d[:, R_HIP]) / 2
        torso_height = torch.norm(torso_top - torso_bottom, dim=-1)
        features.append(torso_height.unsqueeze(-1))

        # 10. Torso aspect ratio (height / width) - encodes forward/backward lean
        torso_aspect = torso_height / (shoulder_width + 1e-6)
        features.append(torso_aspect.unsqueeze(-1))

        # 11. Nose horizontal offset from torso center (encodes head turn / body rotation)
        torso_center = (torso_top + torso_bottom) / 2
        nose_offset_x = pose_2d[:, NOSE, 0] - torso_center[:, 0]
        features.append(nose_offset_x.unsqueeze(-1))

        # 12. Nose vertical offset (encodes camera elevation)
        nose_offset_y = pose_2d[:, NOSE, 1] - torso_top[:, 1]
        features.append(nose_offset_y.unsqueeze(-1))

        # 13-14. Horizontal center of left vs right side (asymmetry encodes rotation)
        left_center_x = (pose_2d[:, L_SHOULDER, 0] + pose_2d[:, L_HIP, 0]) / 2
        right_center_x = (pose_2d[:, R_SHOULDER, 0] + pose_2d[:, R_HIP, 0]) / 2
        side_asymmetry = left_center_x - right_center_x
        features.append(side_asymmetry.unsqueeze(-1))

        # 15. Overall body center X (encodes if person is facing left or right of camera)
        body_center_x = pose_2d.mean(dim=1)[:, 0]
        features.append(body_center_x.unsqueeze(-1))

        # 16. Overall body center Y (encodes camera height relative to person)
        body_center_y = pose_2d.mean(dim=1)[:, 1]
        features.append(body_center_y.unsqueeze(-1))

        return torch.cat(features, dim=-1)  # (batch, feature_dim)

    def forward(self, pose_2d: torch.Tensor) -> torch.Tensor:
        """Encode 2D pose for viewpoint estimation.

        Args:
            pose_2d: (batch, 17, 2) normalized 2D coordinates

        Returns:
            (batch, d_model) 2D pose embedding
        """
        batch_size = pose_2d.size(0)

        # Encode raw coordinates (captures all foreshortening patterns)
        coord_flat = pose_2d.view(batch_size, -1)  # (batch, 34)
        coord_features = self.coord_encoder(coord_flat)  # (batch, d_model)

        # Extract and encode hand-crafted viewpoint features
        viewpoint_features = self._extract_viewpoint_features(pose_2d)  # (batch, 16)
        feature_encoded = self.feature_encoder(viewpoint_features)  # (batch, d_model)

        # Combine both
        combined = torch.cat([coord_features, feature_encoded], dim=-1)  # (batch, d_model*2)
        return self.combiner(combined)  # (batch, d_model)


class DirectAnglePredictor(nn.Module):
    """Predict camera azimuth and elevation DIRECTLY from pose features.

    This avoids the body-frame mismatch problem where:
    - Training labels are computed from GT pose body frame
    - Inference computes angles from corrupted pose body frame
    - These frames differ by ~60-80°, causing huge prediction errors

    By predicting angles directly from (2D pose, 3D features, visibility),
    we learn a consistent mapping that works at both training and inference.

    Architecture (Hybrid Fusion when use_elepose=True):
    - Existing branch: joint_features + 3D + visibility + Pose2DEncoder → features
    - ElePose branch: 2D pose → ElePoseBackbone → deep 2D foreshortening features (1024)
    - Fusion: concat(existing, elepose) → fusion_head → [az_sin, az_cos, elevation]

    The hybrid fusion combines:
    - Transformer context from cross-joint attention (existing)
    - Deep 2D foreshortening patterns from ElePose backbone (new)

    Based on:
    - ElePose (CVPR 2022) - camera elevation from 2D pose foreshortening
    - Empirical testing showed 8.9° azimuth error vs 61° with body-frame approach
    """

    def __init__(
        self,
        d_model: int = 64,
        num_joints: int = 17,
        use_2d_pose: bool = True,
        use_elepose: bool = False,
        elepose_hidden_dim: int = 1024,
    ):
        super().__init__()
        self.num_joints = num_joints
        self.use_elepose = use_elepose
        self.elepose_hidden_dim = elepose_hidden_dim

        # When using ElePose, it REPLACES Pose2DEncoder (not adds to it)
        # This avoids redundant 2D pose processing that wastes parameters
        if use_elepose:
            # ElePose backbone for deep 2D foreshortening features
            self.elepose_backbone = ElePoseBackbone(
                num_joints=num_joints,
                hidden_dim=elepose_hidden_dim,
                num_blocks=2,
                use_batchnorm=True,
            )
            self.use_2d_pose = False  # ElePose replaces Pose2DEncoder
        else:
            # Fallback: use lightweight Pose2DEncoder when ElePose disabled
            self.use_2d_pose = use_2d_pose
            if use_2d_pose:
                self.pose_2d_encoder = Pose2DEncoder(d_model, num_joints)

        # Input dimension for existing features branch:
        # joint_features: d_model * num_joints
        # raw 3D: 3 * num_joints (x,y,z per joint)
        # visibility: num_joints
        # (NO Pose2DEncoder when using ElePose - it's redundant)
        existing_dim = d_model * num_joints  # From transformer joint features
        existing_dim += 3 * num_joints  # Raw 3D pose (important for viewpoint!)
        existing_dim += num_joints  # Visibility
        # Only add Pose2DEncoder dim when NOT using ElePose
        if self.use_2d_pose and not use_elepose:
            existing_dim += d_model  # 2D pose encoder output

        self.existing_dim = existing_dim

        # Fusion dimension: existing + elepose (if enabled)
        if use_elepose:
            fusion_dim = existing_dim + elepose_hidden_dim
        else:
            fusion_dim = existing_dim

        # Deeper fusion head for better utilization of ElePose features
        # Old: 512 → 256 → 3 (too shallow, can't learn which features matter)
        # New: Gradual compression with depth, but efficient width
        # Width scaled to fusion_dim to avoid oversized first layer
        hidden1 = min(512, fusion_dim // 4)  # ~800 for ElePose, ~500 for base
        hidden2 = hidden1 // 2  # 256-400
        hidden3 = hidden2 // 2  # 128-200

        # NOTE: LeakyReLU instead of ReLU - ReLU blocks negative values needed
        # for negative elevation predictions (AIST++ elevation range: -9° to +13°)
        self.fusion_head = nn.Sequential(
            nn.Linear(fusion_dim, hidden1),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.1),
            nn.Linear(hidden1, hidden2),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.1),
            nn.Linear(hidden2, hidden3),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden3, 3),  # [az_sin, az_cos, elevation]
        )

    def forward(
        self,
        joint_features: torch.Tensor,
        pose_3d: torch.Tensor,
        visibility: torch.Tensor,
        pose_2d: torch.Tensor = None,
        elepose_features: torch.Tensor = None,
    ) -> tuple:
        """Predict azimuth and elevation directly using hybrid fusion.

        Args:
            joint_features: (batch, num_joints, d_model) encoded joint features
            pose_3d: (batch, 17, 3) 3D pose (corrupted/input pose)
            visibility: (batch, 17) joint visibility scores
            pose_2d: (batch, 17, 2) normalized 2D coordinates
            elepose_features: (batch, elepose_hidden_dim) pre-computed ElePose features
                             If None and use_elepose=True, will compute from pose_2d

        Returns:
            azimuth: (batch,) predicted azimuth 0-360°
            elevation: (batch,) predicted elevation -90 to +90°
            elepose_features: (batch, elepose_hidden_dim) computed features (for reuse)
        """
        batch_size = joint_features.size(0)

        # === Existing features branch ===
        # Flatten all features
        flat_joint = joint_features.view(batch_size, -1)  # (batch, num_joints * d_model)
        flat_3d = pose_3d.view(batch_size, -1)  # (batch, num_joints * 3)
        flat_vis = visibility.view(batch_size, -1)  # (batch, num_joints)

        existing_features = [flat_joint, flat_3d, flat_vis]

        # Add 2D pose features (lightweight encoder) - ONLY when NOT using ElePose
        if self.use_2d_pose and pose_2d is not None:
            pose_2d_features = self.pose_2d_encoder(pose_2d)  # (batch, d_model)
            existing_features.append(pose_2d_features)

        existing = torch.cat(existing_features, dim=-1)

        # === Normalize existing features for proper gradient flow ===
        existing = F.layer_norm(existing, (existing.shape[-1],))

        # === ElePose features branch (replaces Pose2DEncoder when enabled) ===
        computed_elepose = None
        if self.use_elepose and pose_2d is not None:
            # Use pre-computed features if available, otherwise compute
            if elepose_features is not None:
                computed_elepose = elepose_features
            else:
                computed_elepose = self.elepose_backbone(pose_2d)  # (batch, elepose_hidden_dim)

            # Normalize ElePose features to match scale of existing features
            computed_elepose_norm = F.layer_norm(computed_elepose, (self.elepose_hidden_dim,))
            fused = torch.cat([existing, computed_elepose_norm], dim=-1)
        else:
            fused = existing

        # === Fusion head → angles ===
        pred = self.fusion_head(fused)
        az_sin = pred[:, 0]
        az_cos = pred[:, 1]

        # Convert sin/cos to azimuth degrees
        azimuth = torch.atan2(az_sin, az_cos) * (180.0 / 3.14159265)
        azimuth = torch.where(azimuth < 0, azimuth + 360.0, azimuth)

        # Elevation: tanh scaling to -30 to +30 range
        # NOTE: AIST++ elevation is only -9° to +13°, so ±30° gives better gradient
        # signal than ±90° (which compressed the actual range to tiny tanh outputs)
        elevation = torch.tanh(pred[:, 2]) * 30.0

        return azimuth, elevation, computed_elepose


# Keep old class for backward compatibility with saved checkpoints
class CameraPositionPredictor(nn.Module):
    """DEPRECATED: Predict camera position relative to subject from pose features.

    This approach has a fundamental flaw: angles computed from predicted position
    use the corrupted pose's body frame, which differs from GT body frame by ~60-80°.

    Use DirectAnglePredictor instead for new models.
    """

    def __init__(self, d_model: int = 64, num_joints: int = 17, use_2d_pose: bool = True):
        super().__init__()
        self.use_2d_pose = use_2d_pose

        if use_2d_pose:
            self.pose_2d_encoder = Pose2DEncoder(d_model, num_joints)

        input_dim = d_model * num_joints
        if use_2d_pose:
            input_dim += d_model

        self.position_head = nn.Sequential(
            nn.Linear(input_dim, d_model * 4),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model * 4, d_model * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model * 2, 3),
        )

        self.distance_head = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1),
            nn.Softplus(),
        )

    def forward(self, joint_features: torch.Tensor, pose_2d: torch.Tensor = None) -> torch.Tensor:
        batch_size = joint_features.size(0)
        flat_3d = joint_features.view(batch_size, -1)

        if self.use_2d_pose and pose_2d is not None:
            pose_2d_features = self.pose_2d_encoder(pose_2d)
            flat = torch.cat([flat_3d, pose_2d_features], dim=-1)
        else:
            flat = flat_3d

        direction = self.position_head(flat)
        direction = direction / (torch.norm(direction, dim=-1, keepdim=True) + 1e-8)
        distance = self.distance_head(flat) + 2.0
        camera_pos = direction * distance

        return camera_pos


def compute_angles_from_camera_position(
    pose: torch.Tensor,
    camera_pos: torch.Tensor,
) -> tuple:
    """Compute azimuth and elevation from camera position relative to subject.

    Uses body-derived coordinate frame from torso plane (4 points: 2 hips + 2 shoulders).
    This is robust to subject leaning/bending unlike world-Y approach.

    Args:
        pose: (batch, 17, 3) joint positions (used to get torso orientation)
        camera_pos: (batch, 3) camera position relative to pelvis

    Returns:
        azimuth: (batch,) azimuth in degrees (0-360)
        elevation: (batch,) elevation in degrees (-90 to +90)
    """
    # Get torso points - COCO format:
    # 5=left_shoulder, 6=right_shoulder, 11=left_hip, 12=right_hip, 0=nose
    left_shoulder = pose[:, 5, :]    # (batch, 3)
    right_shoulder = pose[:, 6, :]   # (batch, 3)
    left_hip = pose[:, 11, :]        # (batch, 3)
    right_hip = pose[:, 12, :]       # (batch, 3)
    nose = pose[:, 0, :]             # (batch, 3)

    # Torso center (origin of body frame)
    torso_center = (left_hip + right_hip + left_shoulder + right_shoulder) / 4

    # Right axis: from left side to right side of body
    left_side = (left_hip + left_shoulder) / 2
    right_side = (right_hip + right_shoulder) / 2
    right_axis = right_side - left_side
    right_axis = right_axis / (torch.norm(right_axis, dim=-1, keepdim=True) + 1e-8)

    # Up axis: from hip center to shoulder center (body's actual "up")
    hip_center = (left_hip + right_hip) / 2
    shoulder_center = (left_shoulder + right_shoulder) / 2
    up_axis = shoulder_center - hip_center
    up_axis = up_axis / (torch.norm(up_axis, dim=-1, keepdim=True) + 1e-8)

    # Forward axis: perpendicular to torso plane
    forward_axis = torch.cross(right_axis, up_axis, dim=-1)
    forward_axis = forward_axis / (torch.norm(forward_axis, dim=-1, keepdim=True) + 1e-8)

    # Verify forward direction using nose (nose should be in front)
    nose_dir = nose - torso_center
    nose_dot = (forward_axis * nose_dir).sum(dim=-1)  # (batch,)
    # Flip forward if pointing away from nose
    flip_mask = nose_dot < 0
    forward_axis = torch.where(flip_mask.unsqueeze(-1), -forward_axis, forward_axis)

    # Re-orthogonalize up axis for numerical stability
    # Use right × forward (not forward × right) to get correct upward direction
    up_axis = torch.cross(right_axis, forward_axis, dim=-1)
    up_axis = up_axis / (torch.norm(up_axis, dim=-1, keepdim=True) + 1e-8)

    # Project camera position onto subject's local frame
    # camera_pos is relative to pelvis, convert to relative to torso_center
    pelvis = (left_hip + right_hip) / 2
    # torso_center - pelvis = offset from pelvis to torso_center
    cam_from_torso = camera_pos - (torso_center - pelvis)

    forward_component = (cam_from_torso * forward_axis).sum(dim=-1)  # (batch,)
    right_component = (cam_from_torso * right_axis).sum(dim=-1)      # (batch,)
    up_component = (cam_from_torso * up_axis).sum(dim=-1)            # (batch,)

    # Compute azimuth: angle in horizontal plane
    # 0° = front, 90° = right, 180° = back, 270° = left
    azimuth_rad = torch.atan2(right_component, forward_component)
    azimuth = torch.rad2deg(azimuth_rad)
    azimuth = torch.where(azimuth < 0, azimuth + 360.0, azimuth)

    # Compute elevation: angle above/below horizontal
    horizontal_dist = torch.sqrt(forward_component**2 + right_component**2 + 1e-8)
    elevation_rad = torch.atan2(up_component, horizontal_dist)
    elevation = torch.rad2deg(elevation_rad)

    return azimuth, elevation


# Keep old name as alias for backward compatibility
CameraAnglePredictor = CameraPositionPredictor


class TorsoWidthPredictor(nn.Module):
    """Predict expected 3D torso bone lengths from 2D foreshortening + view angle.

    Key insight: 2D bone length encodes 3D length + viewing angle:
        2D_length ≈ 3D_length × cos(angle_to_camera)

    From 2D measurements and predicted view angle, we can infer expected 3D lengths.

    Predicts 4 bone lengths:
    - shoulder_width (5-6): left_shoulder - right_shoulder
    - hip_width (11-12): left_hip - right_hip
    - left_torso (5-11): left_shoulder - left_hip
    - right_torso (6-12): right_shoulder - right_hip

    These can rotate independently (anatomically correct) while maintaining lengths.
    """

    def __init__(self, d_model: int = 64):
        super().__init__()
        # Input: pose_2d_features (d_model) + view_features (d_model)
        # Output: 4 bone lengths [shoulder_width, hip_width, left_torso, right_torso]
        self.bone_head = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model, 4),
            nn.Softplus(),  # Ensures positive lengths
        )

    def forward(
        self,
        pose_2d_features: torch.Tensor,
        view_features: torch.Tensor,
    ) -> torch.Tensor:
        """Predict expected 3D torso bone lengths.

        Args:
            pose_2d_features: (batch, d_model) from Pose2DEncoder
            view_features: (batch, d_model) from ViewAngleEncoder

        Returns:
            (batch, 4) predicted bone lengths [shoulder, hip, left_torso, right_torso]
        """
        combined = torch.cat([pose_2d_features, view_features], dim=-1)
        return self.bone_head(combined)


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
    - They correlate with view angle (azimuth + elevation)
    - They can be inferred from other visible joints
    - They follow anatomical constraints

    NEW ARCHITECTURE: Predicts camera POSITION, then computes angles from it.
    This ensures the same angle calculation is used for training and inference.

    IMPROVED: Uses 2D pose features for camera prediction (ElePose CVPR 2022 insight).
    2D appearance directly encodes camera viewpoint through foreshortening patterns.

    Input:
        pose: (batch, 17, 3) - x, y, z coordinates per joint
        visibility: (batch, 17) - MediaPipe visibility scores (0-1)
        pose_2d: (batch, 17, 2) - optional 2D pose (normalized image coords) for camera prediction
        camera_pos: (batch, 3) - optional GT camera position relative to pelvis
        azimuth: (batch,) - optional GT azimuth (0-360°), computed from camera_pos if not provided
        elevation: (batch,) - optional GT elevation (-90 to +90°), computed from camera_pos if not provided

    Output:
        delta_z: (batch, 17) - depth corrections per joint
        confidence: (batch, 17) - confidence in corrections
        pred_camera_pos: (batch, 3) - predicted camera position
        pred_azimuth: (batch,) - predicted/computed azimuth
        pred_elevation: (batch,) - predicted/computed elevation
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
        use_2d_pose: bool = True,
        use_elepose: bool = False,
        elepose_hidden_dim: int = 1024,
    ):
        super().__init__()

        self.num_joints = num_joints
        self.d_model = d_model
        self.output_confidence = output_confidence
        self.use_2d_pose = use_2d_pose
        self.use_elepose = use_elepose

        # Joint encoder: (x, y, z, visibility) -> d_model
        # Input: 4 features per joint
        self.joint_encoder = nn.Sequential(
            nn.Linear(4, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )

        # Positional encoding for joint identity
        self.pos_encoder = PositionalEncoding(d_model, max_joints=num_joints)

        # View angle encoder (takes azimuth + elevation)
        self.view_encoder = ViewAngleEncoder(d_model, num_frequencies=8)

        # Direct angle predictor (predicts azimuth/elevation directly)
        # Avoids body-frame mismatch between GT and corrupted poses
        # If use_elepose=True, uses hybrid fusion (ElePose backbone + existing features)
        self.angle_predictor = DirectAnglePredictor(
            d_model=d_model,
            num_joints=num_joints,
            use_2d_pose=use_2d_pose,
            use_elepose=use_elepose,
            elepose_hidden_dim=elepose_hidden_dim,
        )

        # Torso width predictor (predicts expected 3D bone lengths from 2D foreshortening)
        # When using ElePose, we SHARE features instead of having a separate encoder
        if use_elepose:
            # Project ElePose features (1024) down to d_model for torso predictor
            self.elepose_to_torso = nn.Sequential(
                nn.Linear(elepose_hidden_dim, d_model * 2),
                nn.ReLU(),
                nn.Linear(d_model * 2, d_model),
            )
            self.pose_2d_encoder_for_torso = None  # Not needed - using ElePose
            self.torso_predictor = TorsoWidthPredictor(d_model)
        elif use_2d_pose:
            # Fallback: separate Pose2DEncoder when ElePose disabled
            self.elepose_to_torso = None
            self.pose_2d_encoder_for_torso = Pose2DEncoder(d_model, num_joints)
            self.torso_predictor = TorsoWidthPredictor(d_model)
        else:
            self.elepose_to_torso = None
            self.pose_2d_encoder_for_torso = None
            self.torso_predictor = None

        # Cross-joint attention transformer
        self.cross_joint_attn = CrossJointAttention(
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )

        # Output heads - outputs delta_xyz (3 values per joint)
        self.depth_head = nn.Sequential(
            nn.Linear(d_model * 2, d_model),  # *2 for view concat
            nn.ReLU(),
            nn.Linear(d_model, 3),  # x, y, z corrections
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
        pose_2d: torch.Tensor = None,
        camera_pos: torch.Tensor = None,
        azimuth: torch.Tensor = None,
        elevation: torch.Tensor = None,
        use_predicted_angles: bool = True,
        angle_noise_std: float = 0.0,
    ) -> dict:
        """
        Forward pass.

        Args:
            pose: (batch, 17, 3) joint positions
            visibility: (batch, 17) per-joint visibility
            pose_2d: (batch, 17, 2) normalized 2D coordinates for camera prediction (optional but recommended)
            camera_pos: (batch, 3) GT camera position relative to pelvis, None for inference
            azimuth: (batch,) GT azimuth (0-360), used for angle loss computation
            elevation: (batch,) GT elevation (-90 to +90), used for angle loss computation
            use_predicted_angles: If True (default), use predicted angles for depth correction.
                                  This ensures train/inference consistency. Set False only for
                                  ablation studies comparing GT vs predicted angles.
            angle_noise_std: Standard deviation of noise to add to angles during training (degrees).
                            Helps model be robust to angle prediction errors. Only applied during training.

        Returns:
            dict with:
                'delta_xyz': (batch, 17, 3) 3D corrections per joint
                'confidence': (batch, 17) correction confidence (if enabled)
                'pred_azimuth': (batch,) directly predicted azimuth (0-360°)
                'pred_elevation': (batch,) directly predicted elevation (-90 to +90°)
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

        # 4. Predict angles DIRECTLY from features (avoids body-frame mismatch!)
        # When using ElePose, the angle_predictor also returns the computed ElePose features
        # for reuse by the torso predictor (computed once, used twice)
        pred_azimuth, pred_elevation, elepose_features = self.angle_predictor(
            joint_features, pose, visibility, pose_2d
        )

        # 5. Cross-joint attention
        attended = self.cross_joint_attn(joint_features, visibility)  # (batch, 17, d_model)

        # 6. Determine which angles to use for depth correction
        if use_predicted_angles or (azimuth is None and elevation is None):
            # Use predicted angles (default: ensures train/inference consistency)
            # IMPORTANT: .detach() prevents depth loss from backpropagating through angle predictor
            # This avoids gradient interference: camera_loss trains angles, pose_loss trains depth
            use_azimuth = pred_azimuth.detach()
            use_elevation = pred_elevation.detach()
        else:
            # Ablation: use GT angles (only for comparing GT vs predicted performance)
            use_azimuth = azimuth
            use_elevation = elevation

        # 7. Optional angle noise for robustness training
        if angle_noise_std > 0 and self.training:
            az_noise = torch.randn_like(use_azimuth) * angle_noise_std
            el_noise = torch.randn_like(use_elevation) * angle_noise_std * 0.5  # Less noise for elevation
            use_azimuth = (use_azimuth + az_noise) % 360.0
            use_elevation = torch.clamp(use_elevation + el_noise, -90.0, 90.0)

        # 8. Encode view angle (azimuth + elevation)
        view_features = self.view_encoder(use_azimuth, use_elevation)  # (batch, d_model)

        # 8.5. Predict expected torso bone lengths from 2D foreshortening + view angle
        # When using ElePose, reuse the features computed in angle_predictor (no redundant computation)
        pred_torso_lengths = None
        if self.torso_predictor is not None and pose_2d is not None:
            if self.elepose_to_torso is not None and elepose_features is not None:
                # SHARED: Project ElePose features to d_model for torso predictor
                pose_2d_features = self.elepose_to_torso(elepose_features)  # (batch, d_model)
            elif self.pose_2d_encoder_for_torso is not None:
                # FALLBACK: Use separate Pose2DEncoder (when ElePose disabled)
                pose_2d_features = self.pose_2d_encoder_for_torso(pose_2d)  # (batch, d_model)
            else:
                pose_2d_features = None

            if pose_2d_features is not None:
                pred_torso_lengths = self.torso_predictor(pose_2d_features, view_features)  # (batch, 4)

        # 9. Concatenate view features to each joint
        view_expanded = view_features.unsqueeze(1).expand(-1, self.num_joints, -1)
        combined = torch.cat([attended, view_expanded], dim=-1)  # (batch, 17, d_model*2)

        # 10. Predict 3D corrections (x, y, z per joint)
        delta_xyz = self.depth_head(combined)  # (batch, 17, 3)

        output = {
            'delta_xyz': delta_xyz,
            'pred_azimuth': pred_azimuth,
            'pred_elevation': pred_elevation,
        }

        # Add predicted torso lengths (for torso width constraint loss)
        if pred_torso_lengths is not None:
            output['pred_torso_lengths'] = pred_torso_lengths

        if self.output_confidence:
            confidence = self.confidence_head(combined).squeeze(-1)  # (batch, 17)
            output['confidence'] = confidence

        return output


def create_model(
    num_joints: int = 17,
    d_model: int = 64,
    num_heads: int = 4,
    num_layers: int = 4,
    use_2d_pose: bool = True,
    **kwargs,
) -> PoseAwareDepthRefiner:
    """Create depth refinement model with default settings.

    Default config is optimized for AIST++ COCO 17 joints:
    - d_model=64: Compact but expressive
    - num_layers=4: Enough for cross-joint reasoning
    - num_heads=4: 16-dim per head
    - use_2d_pose=True: Use 2D pose for camera prediction (ElePose insight)

    Total params: ~530K with 2D pose, ~500K without (lightweight for real-time inference)
    """
    return PoseAwareDepthRefiner(
        num_joints=num_joints,
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
        use_2d_pose=use_2d_pose,
        **kwargs,
    )


if __name__ == '__main__':
    # Quick test - base model (no ElePose)
    model = create_model()
    print(f"Base model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Test forward pass
    batch_size = 4
    pose = torch.randn(batch_size, 17, 3)
    visibility = torch.rand(batch_size, 17)

    # 2D pose - normalized image coordinates
    pose_2d = torch.rand(batch_size, 17, 2)  # x, y in [0, 1]

    print("\n=== Base model (Pose2DEncoder, no ElePose) ===")
    output = model(pose, visibility, pose_2d=pose_2d)
    print(f"delta_xyz shape: {output['delta_xyz'].shape}")
    print(f"confidence shape: {output['confidence'].shape}")
    print(f"pred_azimuth: {output['pred_azimuth']}")
    print(f"pred_elevation: {output['pred_elevation']}")
    if 'pred_torso_lengths' in output:
        print(f"pred_torso_lengths: {output['pred_torso_lengths'].shape}")

    # Test with GT angles (backward compatibility)
    azimuth = torch.rand(batch_size) * 360  # 0-360°
    elevation = (torch.rand(batch_size) - 0.5) * 180  # -90 to +90°

    print("\n=== Base model (with GT angles - backward compat) ===")
    output2 = model(pose, visibility, pose_2d=pose_2d, azimuth=azimuth, elevation=elevation)
    print(f"delta_xyz shape: {output2['delta_xyz'].shape}")
    if 'pred_torso_lengths' in output2:
        print(f"pred_torso_lengths shape: {output2['pred_torso_lengths'].shape}")
        print(f"pred_torso_lengths: {output2['pred_torso_lengths']}")

    # Test without 2D pose (fallback - less accurate)
    print("\n=== Model without 2D pose (fallback) ===")
    model_no_2d = create_model(use_2d_pose=False)
    print(f"Model (no 2D) parameters: {sum(p.numel() for p in model_no_2d.parameters()):,}")
    output_no_2d = model_no_2d(pose, visibility)
    print(f"delta_xyz shape: {output_no_2d['delta_xyz'].shape}")
    print(f"pred_azimuth: {output_no_2d['pred_azimuth']}")
    print(f"pred_elevation: {output_no_2d['pred_elevation']}")

    # Test ElePose hybrid model (STREAMLINED - no redundant Pose2DEncoder)
    print("\n=== ElePose hybrid model (streamlined fusion) ===")
    model_elepose = create_model(use_elepose=True, d_model=128)
    print(f"ElePose model parameters: {sum(p.numel() for p in model_elepose.parameters()):,}")

    # Verify architecture: ElePose replaces Pose2DEncoder
    ap = model_elepose.angle_predictor
    print(f"  Has pose_2d_encoder: {hasattr(ap, 'pose_2d_encoder')}")
    print(f"  Has elepose_backbone: {hasattr(ap, 'elepose_backbone')}")
    print(f"  Has elepose_to_torso projection: {model_elepose.elepose_to_torso is not None}")
    print(f"  Has pose_2d_encoder_for_torso: {model_elepose.pose_2d_encoder_for_torso is not None}")

    output_elepose = model_elepose(pose, visibility, pose_2d=pose_2d)
    print(f"delta_xyz shape: {output_elepose['delta_xyz'].shape}")
    print(f"pred_azimuth: {output_elepose['pred_azimuth']}")
    print(f"pred_elevation: {output_elepose['pred_elevation']}")
    if 'pred_torso_lengths' in output_elepose:
        print(f"pred_torso_lengths shape: {output_elepose['pred_torso_lengths'].shape}")

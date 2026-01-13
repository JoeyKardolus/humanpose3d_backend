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


class CameraPositionPredictor(nn.Module):
    """Predict camera position relative to subject from pose features.

    At inference time, we don't have ground truth camera position.
    This head learns to predict the camera's position relative to the subject's pelvis.

    The predicted position is then used to compute azimuth/elevation using the
    same geometric calculation as during training - ensuring consistency.

    IMPROVED: Now uses 2D pose features which directly encode camera viewpoint
    through foreshortening and perspective patterns (ElePose CVPR 2022 insight).

    Based on:
    - ElePose (CVPR 2022) - camera elevation from 2D pose
    - Map-Relative Pose Regression (CVPR 2024)

    Output is in subject-relative coordinates:
    - x: left/right (positive = camera to subject's right)
    - y: up/down (positive = camera above subject)
    - z: front/back (positive = camera in front of subject)
    """

    def __init__(self, d_model: int = 64, num_joints: int = 17, use_2d_pose: bool = True):
        super().__init__()
        self.use_2d_pose = use_2d_pose

        # 2D pose encoder (primary source for viewpoint - foreshortening encodes angle!)
        if use_2d_pose:
            self.pose_2d_encoder = Pose2DEncoder(d_model, num_joints)

        # Input dim depends on whether we use 2D pose
        # If 2D pose: d_model (from 2D encoder) + d_model * num_joints (from 3D)
        # If no 2D: just d_model * num_joints (from 3D)
        input_dim = d_model * num_joints
        if use_2d_pose:
            input_dim += d_model  # Add 2D pose features

        # Pool joint features and predict camera position
        self.position_head = nn.Sequential(
            nn.Linear(input_dim, d_model * 4),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model * 4, d_model * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model * 2, 3),  # Output: [x, y, z] relative position
        )

        # Typical camera distance range: 2-10 meters from subject
        # We predict a unit direction and a distance scale
        self.distance_head = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1),
            nn.Softplus(),  # Ensures positive distance
        )

    def forward(
        self,
        joint_features: torch.Tensor,
        pose_2d: torch.Tensor = None,
    ) -> torch.Tensor:
        """Predict camera position from joint features and optionally 2D pose.

        Args:
            joint_features: (batch, num_joints, d_model) encoded joint features
            pose_2d: (batch, 17, 2) normalized 2D coordinates, optional but recommended

        Returns:
            camera_pos: (batch, 3) predicted camera position relative to pelvis
        """
        batch_size = joint_features.size(0)

        # Flatten 3D joint features
        flat_3d = joint_features.view(batch_size, -1)  # (batch, num_joints * d_model)

        # Combine with 2D pose features if available
        if self.use_2d_pose and pose_2d is not None:
            pose_2d_features = self.pose_2d_encoder(pose_2d)  # (batch, d_model)
            flat = torch.cat([flat_3d, pose_2d_features], dim=-1)
        else:
            flat = flat_3d

        # Predict direction (normalized) and distance separately
        direction = self.position_head(flat)  # (batch, 3)
        direction = direction / (torch.norm(direction, dim=-1, keepdim=True) + 1e-8)

        distance = self.distance_head(flat) + 2.0  # Minimum 2m distance

        # Scale direction by distance
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
    up_axis = torch.cross(forward_axis, right_axis, dim=-1)
    up_axis = up_axis / (torch.norm(up_axis, dim=-1, keepdim=True) + 1e-8)

    # Project camera position onto subject's local frame
    # camera_pos is relative to pelvis, adjust to torso center
    pelvis = (left_hip + right_hip) / 2
    cam_from_torso = camera_pos + pelvis - torso_center  # Adjust reference point

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
    ):
        super().__init__()

        self.num_joints = num_joints
        self.d_model = d_model
        self.output_confidence = output_confidence
        self.use_2d_pose = use_2d_pose

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

        # Camera POSITION predictor (predicts x,y,z relative to pelvis)
        # Angles are then computed from this position - same as training!
        # Now with 2D pose features for better viewpoint estimation
        self.camera_predictor = CameraPositionPredictor(d_model, num_joints, use_2d_pose=use_2d_pose)

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
        pose_2d: torch.Tensor = None,
        camera_pos: torch.Tensor = None,
        azimuth: torch.Tensor = None,
        elevation: torch.Tensor = None,
    ) -> dict:
        """
        Forward pass.

        Args:
            pose: (batch, 17, 3) joint positions
            visibility: (batch, 17) per-joint visibility
            pose_2d: (batch, 17, 2) normalized 2D coordinates for camera prediction (optional but recommended)
            camera_pos: (batch, 3) GT camera position relative to pelvis, None for inference
            azimuth: (batch,) GT azimuth (0-360), used if camera_pos not provided
            elevation: (batch,) GT elevation (-90 to +90), used if camera_pos not provided

        Returns:
            dict with:
                'delta_z': (batch, 17) depth corrections
                'confidence': (batch, 17) correction confidence (if enabled)
                'pred_camera_pos': (batch, 3) predicted camera position
                'pred_azimuth': (batch,) computed azimuth from predicted position
                'pred_elevation': (batch,) computed elevation from predicted position
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

        # 4. Predict camera POSITION from pose (always, for training the predictor)
        # Uses 2D pose features if available (more informative for viewpoint!)
        pred_camera_pos = self.camera_predictor(joint_features, pose_2d)  # (batch, 3)

        # 5. Compute angles from predicted camera position
        # This is the SAME calculation used during training data generation!
        pred_azimuth, pred_elevation = compute_angles_from_camera_position(pose, pred_camera_pos)

        # 6. Cross-joint attention
        attended = self.cross_joint_attn(joint_features, visibility)  # (batch, 17, d_model)

        # 7. Determine which angles to use for depth correction
        if camera_pos is not None:
            # Training with GT camera position: compute angles from GT position
            use_azimuth, use_elevation = compute_angles_from_camera_position(pose, camera_pos)
        elif azimuth is not None and elevation is not None:
            # Training with GT angles directly (backward compatibility)
            use_azimuth = azimuth
            use_elevation = elevation
        else:
            # Inference: use predicted angles (computed from predicted camera position)
            use_azimuth = pred_azimuth
            use_elevation = pred_elevation

        # 8. Encode view angle (azimuth + elevation)
        view_features = self.view_encoder(use_azimuth, use_elevation)  # (batch, d_model)

        # 9. Concatenate view features to each joint
        view_expanded = view_features.unsqueeze(1).expand(-1, self.num_joints, -1)
        combined = torch.cat([attended, view_expanded], dim=-1)  # (batch, 17, d_model*2)

        # 10. Predict depth corrections
        delta_z = self.depth_head(combined).squeeze(-1)  # (batch, 17)

        output = {
            'delta_z': delta_z,
            'pred_camera_pos': pred_camera_pos,
            'pred_azimuth': pred_azimuth,
            'pred_elevation': pred_elevation,
        }

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
    # Quick test
    model = create_model()
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Test forward pass with GT camera position (training mode - preferred)
    batch_size = 4
    pose = torch.randn(batch_size, 17, 3)
    visibility = torch.rand(batch_size, 17)

    # 2D pose - normalized image coordinates
    pose_2d = torch.rand(batch_size, 17, 2)  # x, y in [0, 1]

    # Simulate camera positions: 3-8m away from subject
    camera_pos = torch.randn(batch_size, 3)
    camera_pos = camera_pos / torch.norm(camera_pos, dim=-1, keepdim=True)  # Normalize direction
    camera_pos = camera_pos * (torch.rand(batch_size, 1) * 5 + 3)  # 3-8m distance

    print("\n=== Training mode (with GT camera position + 2D pose) ===")
    output = model(pose, visibility, pose_2d=pose_2d, camera_pos=camera_pos)
    print(f"delta_z shape: {output['delta_z'].shape}")
    print(f"confidence shape: {output['confidence'].shape}")
    print(f"pred_camera_pos shape: {output['pred_camera_pos'].shape}")
    print(f"pred_camera_pos: {output['pred_camera_pos']}")
    print(f"pred_azimuth: {output['pred_azimuth']}")
    print(f"pred_elevation: {output['pred_elevation']}")

    # Test with GT angles (backward compatibility)
    azimuth = torch.rand(batch_size) * 360  # 0-360°
    elevation = (torch.rand(batch_size) - 0.5) * 180  # -90 to +90°

    print("\n=== Training mode (with GT angles - backward compat) ===")
    output2 = model(pose, visibility, pose_2d=pose_2d, azimuth=azimuth, elevation=elevation)
    print(f"delta_z shape: {output2['delta_z'].shape}")

    # Test forward pass without GT but WITH 2D pose (inference mode - RECOMMENDED)
    print("\n=== Inference mode with 2D pose (recommended) ===")
    output_inf = model(pose, visibility, pose_2d=pose_2d)
    print(f"delta_z shape: {output_inf['delta_z'].shape}")
    print(f"pred_camera_pos: {output_inf['pred_camera_pos']}")
    print(f"pred_azimuth: {output_inf['pred_azimuth']}")
    print(f"pred_elevation: {output_inf['pred_elevation']}")

    # Test without 2D pose (fallback - less accurate)
    print("\n=== Inference mode without 2D pose (fallback) ===")
    model_no_2d = create_model(use_2d_pose=False)
    print(f"Model (no 2D) parameters: {sum(p.numel() for p in model_no_2d.parameters()):,}")
    output_no_2d = model_no_2d(pose, visibility)
    print(f"delta_z shape: {output_no_2d['delta_z'].shape}")
    print(f"pred_camera_pos: {output_no_2d['pred_camera_pos']}")

#!/usr/bin/env python3
"""Diagnose torso collapse issue in POF reconstruction."""

import numpy as np
import torch
import sys
sys.path.insert(0, ".")

from src.pof.inference import CameraPOFInference
from src.pof.bone_lengths import estimate_bone_lengths_array
from src.pof.constants import (
    LIMB_DEFINITIONS, LIMB_NAMES, NUM_LIMBS, NUM_JOINTS,
    LEFT_SHOULDER_IDX, RIGHT_SHOULDER_IDX, LEFT_HIP_IDX, RIGHT_HIP_IDX,
    HEIGHT_TO_TORSO_RATIO, COCO_JOINT_NAMES,
)
from src.pof.least_squares import (
    normalize_2d_for_pof, solve_depth_least_squares_pof,
    SOLVE_ORDER, LIMB_KINEMATIC_PARENTS,
)
from src.pof.dataset import normalize_pose_2d, compute_limb_features_2d


def load_trc_coco17(trc_path: str, frame_idx: int = 100):
    """Load COCO-17 keypoints from initial TRC file."""
    import pandas as pd

    # Read TRC (tab-separated, skip header lines)
    with open(trc_path, 'r') as f:
        lines = f.readlines()

    # Find data start (after header)
    data_start = 0
    for i, line in enumerate(lines):
        if line.strip().startswith("Frame#"):
            # Next line is units, data starts after
            data_start = i + 2
            # Parse marker names from this line
            header_parts = line.strip().split('\t')
            # Markers start at index 2 (Frame#, Time, then X1, Y1, Z1, X2...)
            marker_names = []
            for j in range(2, len(header_parts), 3):
                if header_parts[j]:
                    marker_names.append(header_parts[j])
            break

    print(f"Found {len(marker_names)} markers: {marker_names[:10]}...")

    # Parse data
    data_lines = lines[data_start:]
    frame_data = data_lines[frame_idx].strip().split('\t')

    # Extract positions for each marker
    positions = {}
    for i, name in enumerate(marker_names):
        base_idx = 2 + i * 3
        if base_idx + 2 < len(frame_data):
            x = float(frame_data[base_idx]) if frame_data[base_idx] else np.nan
            y = float(frame_data[base_idx + 1]) if frame_data[base_idx + 1] else np.nan
            z = float(frame_data[base_idx + 2]) if frame_data[base_idx + 2] else np.nan
            positions[name] = np.array([x, y, z])

    return positions, marker_names


def diagnose_pof_reconstruction(model_path: str, trc_path: str, height_m: float = 1.78):
    """Diagnose torso collapse by inspecting POF values and LS solve."""

    # Load model
    print(f"Loading model: {model_path}")
    inference = CameraPOFInference(model_path, device="cpu")

    # Load from TRC
    frame_idx = 100
    positions, marker_names = load_trc_coco17(trc_path, frame_idx)

    # Map TRC marker names to COCO-17
    # TRC uses Pose2Sim names, need to map to indices
    pose2sim_to_coco = {
        'Nose': 0,
        'LEye': 1, 'REye': 2,
        'LEar': 3, 'REar': 4,
        'LShoulder': 5, 'RShoulder': 6,
        'LElbow': 7, 'RElbow': 8,
        'LWrist': 9, 'RWrist': 10,
        'LHip': 11, 'RHip': 12,
        'LKnee': 13, 'RKnee': 14,
        'LAnkle': 15, 'RAnkle': 16,
    }

    keypoints_3d = np.zeros((17, 3), dtype=np.float32)
    visibility = np.ones(17, dtype=np.float32)

    for name, coco_idx in pose2sim_to_coco.items():
        if name in positions:
            keypoints_3d[coco_idx] = positions[name]
        else:
            visibility[coco_idx] = 0.0

    # TRC is already in meters, create pseudo-2D by projecting X,Z (assuming Y is up)
    # Actually TRC coordinates: X=right, Y=up, Z=forward (away from camera)
    # For 2D projection, we want X,Y in image space
    # Assuming camera looks along +Z, 2D = (X, -Y) normalized

    # Get bounding box for normalization
    valid_mask = visibility > 0
    valid_kp = keypoints_3d[valid_mask]
    x_min, x_max = valid_kp[:, 0].min(), valid_kp[:, 0].max()
    y_min, y_max = valid_kp[:, 1].min(), valid_kp[:, 1].max()

    # Normalize to [0, 1] with Y flipped (image convention: Y down)
    keypoints_2d = np.zeros((17, 2), dtype=np.float32)
    keypoints_2d[:, 0] = (keypoints_3d[:, 0] - x_min) / (x_max - x_min + 1e-6)
    keypoints_2d[:, 1] = 1.0 - (keypoints_3d[:, 1] - y_min) / (y_max - y_min + 1e-6)  # Flip Y

    print(f"\n=== Frame {frame_idx} Keypoints ===")
    print(f"3D bounding box: X=[{x_min:.3f}, {x_max:.3f}], Y=[{y_min:.3f}, {y_max:.3f}]")
    print(f"\nL_shoulder (5): 3D={keypoints_3d[5]}, 2D={keypoints_2d[5]}")
    print(f"R_shoulder (6): 3D={keypoints_3d[6]}, 2D={keypoints_2d[6]}")
    print(f"L_hip (11): 3D={keypoints_3d[11]}, 2D={keypoints_2d[11]}")
    print(f"R_hip (12): 3D={keypoints_3d[12]}, 2D={keypoints_2d[12]}")

    # Compute actual 3D torso lengths from TRC
    l_torso_3d = np.linalg.norm(keypoints_3d[5] - keypoints_3d[11])
    r_torso_3d = np.linalg.norm(keypoints_3d[6] - keypoints_3d[12])
    print(f"\nActual 3D torso lengths from TRC: L={l_torso_3d:.4f}m, R={r_torso_3d:.4f}m")

    # Compute 2D torso vectors
    l_torso_2d = keypoints_2d[5] - keypoints_2d[11]  # L_shoulder - L_hip
    r_torso_2d = keypoints_2d[6] - keypoints_2d[12]  # R_shoulder - R_hip
    print(f"\n2D torso delta (shoulder-hip):")
    print(f"  L: [{l_torso_2d[0]:.4f}, {l_torso_2d[1]:.4f}], len={np.linalg.norm(l_torso_2d):.4f}")
    print(f"  R: [{r_torso_2d[0]:.4f}, {r_torso_2d[1]:.4f}], len={np.linalg.norm(r_torso_2d):.4f}")

    # Predict POF
    pof, zsign_info = inference.predict_pof_with_zsign(keypoints_2d, visibility)

    print(f"\n=== POF Predictions ===")
    for limb_idx in [10, 11, 9, 8]:  # L_torso, R_torso, hip_width, shoulder_width
        name = LIMB_NAMES[limb_idx]
        parent, child = LIMB_DEFINITIONS[limb_idx]
        pof_vec = pof[limb_idx]
        z_prob = zsign_info['probs'][limb_idx]
        was_corrected = zsign_info['corrections'][limb_idx]

        print(f"\n{name} (limb {limb_idx}): POF points {COCO_JOINT_NAMES[parent]}->{COCO_JOINT_NAMES[child]}")
        print(f"  POF: [{pof_vec[0]:.4f}, {pof_vec[1]:.4f}, {pof_vec[2]:.4f}]")
        print(f"  ||POF_xy|| = {np.linalg.norm(pof_vec[:2]):.4f}")
        print(f"  Z_sign prob(Z>0): {z_prob:.3f}, corrected: {was_corrected}")

        # For torso, analyze alignment with 2D delta
        if limb_idx in [10, 11]:
            kinematic = LIMB_KINEMATIC_PARENTS[limb_idx]
            parent_kin, child_kin = kinematic

            # POF as defined points shoulder->hip
            # Kinematic is hip->shoulder, so we negate
            pof_kinematic = -pof_vec

            if limb_idx == 10:
                delta_2d = l_torso_2d
            else:
                delta_2d = r_torso_2d

            delta_2d_norm = delta_2d / (np.linalg.norm(delta_2d) + 1e-6)

            print(f"  Kinematic direction: {COCO_JOINT_NAMES[parent_kin]}->{COCO_JOINT_NAMES[child_kin]}")
            print(f"  POF (negated for kinematic): [{pof_kinematic[0]:.4f}, {pof_kinematic[1]:.4f}, {pof_kinematic[2]:.4f}]")
            print(f"  2D delta (normalized): [{delta_2d_norm[0]:.4f}, {delta_2d_norm[1]:.4f}]")

            dot_xy = np.dot(delta_2d_norm, pof_kinematic[:2])
            print(f"  dot(2D_norm, POF_xy): {dot_xy:.4f}")

            # Expected: POF_xy should be parallel to 2D_delta (dot ≈ 1 or -1)
            # If dot ≈ 0, POF_xy is perpendicular to 2D, which would give scale ≈ 0

    # Trace LS solver
    print(f"\n=== LS Solver Trace ===")

    # Normalize 2D
    pose_2d_norm, _, _ = normalize_pose_2d(keypoints_2d)
    kp_t = torch.from_numpy(pose_2d_norm).unsqueeze(0).float()
    pof_t = torch.from_numpy(pof).unsqueeze(0).float()

    normalized_2d, pelvis_2d, torso_scale = normalize_2d_for_pof(kp_t)
    print(f"Torso scale (from 2D): {torso_scale.item():.4f}")

    # Bone lengths
    bone_lengths = estimate_bone_lengths_array(height_m)
    metric_torso_scale = height_m / HEIGHT_TO_TORSO_RATIO
    bl_normalized = bone_lengths / metric_torso_scale
    print(f"\nExpected bone lengths:")
    print(f"  L_torso (10): {bone_lengths[10]:.4f}m (normalized: {bl_normalized[10]:.4f})")
    print(f"  R_torso (11): {bone_lengths[11]:.4f}m (normalized: {bl_normalized[11]:.4f})")

    # Manual solve trace - WITH use_2d_xy fix
    pose_3d = torch.zeros(1, NUM_JOINTS, 3)
    pose_3d[:, :, :2] = normalized_2d
    pose_3d[:, LEFT_HIP_IDX, 2] = 0.0
    pose_3d[:, RIGHT_HIP_IDX, 2] = 0.0

    for limb_idx in SOLVE_ORDER[:3]:  # hip_width, L_torso, R_torso
        pof_parent, pof_child = LIMB_DEFINITIONS[limb_idx]
        orientation = pof_t[:, limb_idx].clone()

        kinematic = LIMB_KINEMATIC_PARENTS.get(limb_idx)
        if kinematic is not None:
            parent_idx, child_idx = kinematic
            if (parent_idx, child_idx) != (pof_parent, pof_child):
                orientation = -orientation
        else:
            parent_idx, child_idx = pof_parent, pof_child

        parent_2d = normalized_2d[:, parent_idx]
        child_2d = normalized_2d[:, child_idx]
        parent_depth = pose_3d[:, parent_idx, 2]

        delta_2d = child_2d - parent_2d
        delta_2d_len = torch.norm(delta_2d, dim=-1)

        # NEW: derive orient_xy from 2D delta (use_2d_xy=True)
        orient_z = orientation[:, 2]
        z_mag = torch.abs(orient_z)
        xy_mag = torch.sqrt(torch.clamp(1.0 - z_mag ** 2, min=0.0))
        delta_2d_norm = torch.nn.functional.normalize(delta_2d, dim=-1, eps=1e-6)
        orient_xy = delta_2d_norm * xy_mag.unsqueeze(-1)  # Derived from 2D!

        orient_xy_norm_sq = (orient_xy ** 2).sum(dim=-1)

        scale_from_lstsq = (delta_2d * orient_xy).sum(dim=-1) / (orient_xy_norm_sq + 1e-6)
        child_depth = parent_depth + scale_from_lstsq * orient_z

        pose_3d[:, child_idx, 2] = child_depth

        print(f"\n--- Limb {limb_idx} ({LIMB_NAMES[limb_idx]}): {COCO_JOINT_NAMES[parent_idx]} -> {COCO_JOINT_NAMES[child_idx]} ---")
        print(f"  orient_z (from POF): {orient_z.item():.4f}, |z|={z_mag.item():.4f}")
        print(f"  xy_mag (derived): {xy_mag.item():.4f}")
        print(f"  orient_xy (from 2D): [{orient_xy[0,0]:.4f}, {orient_xy[0,1]:.4f}]")
        print(f"  delta_2d: [{delta_2d[0,0]:.4f}, {delta_2d[0,1]:.4f}], |delta_2d|={delta_2d_len.item():.4f}")
        print(f"  ||orient_xy||^2 = {orient_xy_norm_sq.item():.4f}")
        print(f"  scale = {scale_from_lstsq.item():.4f}")
        print(f"  child_depth = {child_depth.item():.4f}")

    # Check resulting torso lengths
    print(f"\n=== Resulting 3D Skeleton (normalized) ===")
    print(f"L_hip Z: {pose_3d[0, 11, 2].item():.4f}")
    print(f"R_hip Z: {pose_3d[0, 12, 2].item():.4f}")
    print(f"L_shoulder Z: {pose_3d[0, 5, 2].item():.4f}")
    print(f"R_shoulder Z: {pose_3d[0, 6, 2].item():.4f}")

    l_torso_recon = torch.norm(pose_3d[0, 5] - pose_3d[0, 11]).item()
    r_torso_recon = torch.norm(pose_3d[0, 6] - pose_3d[0, 12]).item()
    print(f"\nReconstructed torso lengths: L={l_torso_recon:.4f}, R={r_torso_recon:.4f}")
    print(f"Expected (normalized): L={bl_normalized[10]:.4f}, R={bl_normalized[11]:.4f}")
    print(f"Ratio (recon/expected): L={l_torso_recon/bl_normalized[10]:.3f}, R={r_torso_recon/bl_normalized[11]:.3f}")

    # Metric scale
    l_torso_m = l_torso_recon * metric_torso_scale
    r_torso_m = r_torso_recon * metric_torso_scale
    print(f"\nReconstructed torso (meters): L={l_torso_m:.4f}m, R={r_torso_m:.4f}m")
    print(f"Expected (meters): L={bone_lengths[10]:.4f}m, R={bone_lengths[11]:.4f}m")


if __name__ == "__main__":
    model_path = "models/checkpoints/best_pof_semgcn-temporal_model.pth"
    trc_path = "data/output/joey/joey_initial.trc"

    diagnose_pof_reconstruction(model_path, trc_path)

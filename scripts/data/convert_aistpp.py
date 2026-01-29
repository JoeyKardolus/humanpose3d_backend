#!/usr/bin/env python3
"""
Convert AIST++ dataset to training pairs for depth refinement.

AIST++ provides:
- Multi-view dance videos (60fps)
- 3D keypoints in COCO format (17 joints)
- Camera calibration data

This script:
1. Runs MediaPipe on video frames -> noisy 3D pose (REAL errors!)
2. Loads AIST++ keypoints3d -> ground truth 3D pose
3. Computes view angle from ground truth torso plane
4. Creates training pairs for depth refinement model

Key insight: We use REAL MediaPipe errors, not synthetic noise.
"""

import os
# Suppress TensorFlow and MediaPipe warnings BEFORE imports
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['GLOG_minloglevel'] = '2'
os.environ['ABSL_MIN_LOG_LEVEL'] = '2'

import sys
import argparse
from pathlib import Path
from multiprocessing import Pool

import warnings
warnings.filterwarnings('ignore')

import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)
logging.getLogger('absl').setLevel(logging.ERROR)

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import json
from src.application.config.paths import StoragePaths
import pickle
import numpy as np
from tqdm import tqdm
import cv2
import mediapipe as mp
from scipy.spatial.transform import Rotation

from src.depth_refinement.data_utils import align_body_frames


# COCO 17 keypoints (AIST++ format)
COCO_KEYPOINTS = [
    'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
    'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
    'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
    'left_knee', 'right_knee', 'left_ankle', 'right_ankle',
]

# MediaPipe landmark indices for COCO 17 joints
MEDIAPIPE_TO_COCO = {
    0: 0,    # nose
    1: 2,    # left_eye
    2: 5,    # right_eye
    3: 7,    # left_ear
    4: 8,    # right_ear
    5: 11,   # left_shoulder
    6: 12,   # right_shoulder
    7: 13,   # left_elbow
    8: 14,   # right_elbow
    9: 15,   # left_wrist
    10: 16,  # right_wrist
    11: 23,  # left_hip
    12: 24,  # right_hip
    13: 25,  # left_knee
    14: 26,  # right_knee
    15: 27,  # left_ankle
    16: 28,  # right_ankle
}


def load_camera_params(cameras_dir: Path, setting_name: str, camera_name: str = 'c01') -> dict:
    """
    Load full camera calibration from AIST++ calibration.

    Args:
        cameras_dir: Path to annotations/cameras/
        setting_name: Camera setting (e.g., 'setting1')
        camera_name: Camera identifier (e.g., 'c01')

    Returns:
        Dict with:
            'position': camera position in world coords (meters)
            'rotation': 3x3 rotation matrix (world to camera)
            'translation': translation vector (meters)
            'intrinsic': 3x3 intrinsic matrix
            'name': camera identifier
    """
    setting_file = cameras_dir / f"{setting_name}.json"
    with open(setting_file, 'r') as f:
        cameras = json.load(f)

    for cam in cameras:
        if cam['name'] == camera_name:
            # Rodrigues to rotation matrix
            rvec = np.array(cam['rotation'])
            R = Rotation.from_rotvec(rvec).as_matrix()

            # Translation (in cm -> meters)
            t = np.array(cam['translation']) / 100.0

            # Camera position in world coords: C = -R^T @ t
            cam_pos = -R.T @ t

            # Intrinsic matrix (3x3)
            K = np.array(cam['matrix']).reshape(3, 3)

            return {
                'position': cam_pos,
                'rotation': R,
                'translation': t,
                'intrinsic': K,
                'name': camera_name,
            }

    raise ValueError(f"Camera {camera_name} not found in {setting_file}")


def get_camera_setting(seq_name: str, mapping_file: Path) -> str:
    """Get camera setting name for sequence from mapping file."""
    with open(mapping_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2 and parts[0] == seq_name:
                return parts[1]
    return 'setting1'  # Default


def compute_view_angles(pose_3d: np.ndarray, camera_pos: np.ndarray) -> tuple:
    """
    Compute camera viewing angles in subject's BODY-RELATIVE coordinate frame.

    Uses body-derived axes (not world axes) to match inference code exactly.
    This ensures training labels match what the model computes during inference.

    Returns full 0-360° azimuth range:
      0° = camera directly in front of subject
     90° = camera to subject's right side (profile)
    180° = camera directly behind subject
    270° = camera to subject's left side (profile)

    Args:
        pose_3d: (17, 3) COCO keypoints (in meters)
        camera_pos: (3,) Camera position in world coords (meters)

    Returns:
        (azimuth, elevation) in degrees
        - azimuth: 0-360° around subject (0°=front, 90°=right, 180°=back, 270°=left)
        - elevation: angle above/below horizontal (+ve = camera above)
    """
    # COCO indices
    left_shoulder = pose_3d[5]
    right_shoulder = pose_3d[6]
    left_hip = pose_3d[11]
    right_hip = pose_3d[12]
    nose = pose_3d[0]

    # Torso center
    torso_center = (left_shoulder + right_shoulder + left_hip + right_hip) / 4

    # === BUILD BODY-RELATIVE COORDINATE FRAME (matches inference exactly) ===

    # Right axis: averaged sides (more stable than just shoulders)
    left_side = (left_hip + left_shoulder) / 2
    right_side = (right_hip + right_shoulder) / 2
    right_axis = right_side - left_side
    right_norm = np.linalg.norm(right_axis)
    if right_norm < 1e-6:
        return 0.0, 0.0
    right_axis = right_axis / right_norm

    # Up axis: body spine direction (hip center to shoulder center)
    hip_center = (left_hip + right_hip) / 2
    shoulder_center = (left_shoulder + right_shoulder) / 2
    up_axis = shoulder_center - hip_center
    up_norm = np.linalg.norm(up_axis)
    if up_norm < 1e-6:
        return 0.0, 0.0
    up_axis = up_axis / up_norm

    # Forward axis: right × up (matches inference cross product order!)
    forward_axis = np.cross(right_axis, up_axis)
    forward_norm = np.linalg.norm(forward_axis)
    if forward_norm < 1e-6:
        forward_axis = np.array([0.0, 0.0, 1.0])
    else:
        forward_axis = forward_axis / forward_norm

    # Nose verification (matches inference): flip if forward points away from nose
    nose_dir = nose - torso_center
    if np.dot(forward_axis, nose_dir) < 0:
        forward_axis = -forward_axis

    # Re-orthogonalize up axis for numerical stability
    # Use right × forward (not forward × right) to get correct upward direction
    up_axis = np.cross(right_axis, forward_axis)
    up_axis = up_axis / (np.linalg.norm(up_axis) + 1e-8)

    # === CAMERA DIRECTION IN SUBJECT'S FRAME ===
    # Vector from subject TO camera (so 0° = camera in front)
    subj_to_cam = camera_pos - torso_center
    cam_dist = np.linalg.norm(subj_to_cam)
    if cam_dist < 1e-6:
        return 0.0, 0.0

    # Project onto subject's coordinate frame
    forward_component = np.dot(subj_to_cam, forward_axis)
    right_component = np.dot(subj_to_cam, right_axis)
    up_component = np.dot(subj_to_cam, up_axis)

    # === AZIMUTH: full 0-360° ===
    # atan2 gives -180 to +180, convert to 0-360
    azimuth = np.degrees(np.arctan2(right_component, forward_component))
    if azimuth < 0:
        azimuth += 360.0

    # === ELEVATION ===
    horiz_dist = np.sqrt(forward_component**2 + right_component**2)
    if horiz_dist < 1e-6:
        elevation = 90.0 if up_component > 0 else -90.0
    else:
        elevation = np.degrees(np.arctan2(up_component, horiz_dist))

    return float(azimuth), float(elevation)


def extract_mediapipe_coco(results) -> tuple:
    """
    Extract 17 COCO joints, visibility, and 2D pose from MediaPipe results.

    Args:
        results: MediaPipe pose results

    Returns:
        (joints_3d, visibility, joints_2d) where:
        - joints_3d: (17, 3) array of 3D joint positions (world coords)
        - visibility: (17,) array of visibility scores
        - joints_2d: (17, 2) array of 2D joint positions (normalized image coords)
    """
    joints_3d = np.zeros((17, 3))
    joints_2d = np.zeros((17, 2))
    visibility = np.zeros(17)

    # 3D world landmarks (in meters)
    world_landmarks = results.pose_world_landmarks.landmark

    # 2D image landmarks (normalized 0-1)
    image_landmarks = results.pose_landmarks.landmark

    for coco_idx, mp_idx in MEDIAPIPE_TO_COCO.items():
        # 3D pose (world coordinates)
        lm_3d = world_landmarks[mp_idx]
        joints_3d[coco_idx] = [lm_3d.x, lm_3d.y, lm_3d.z]
        visibility[coco_idx] = lm_3d.visibility

        # 2D pose (normalized image coordinates)
        lm_2d = image_landmarks[mp_idx]
        joints_2d[coco_idx] = [lm_2d.x, lm_2d.y]  # x, y in [0, 1]

    return joints_3d, visibility, joints_2d


def center_on_pelvis(joints: np.ndarray) -> np.ndarray:
    """Center skeleton on pelvis (midpoint of hips)."""
    # COCO: left_hip=11, right_hip=12
    pelvis = (joints[11] + joints[12]) / 2
    return joints - pelvis


def compute_torso_scale(joints: np.ndarray) -> float:
    """
    Compute torso scale for normalization.

    Uses hip-to-shoulder distance (average of left and right).
    This is robust and consistent across different body positions.

    Args:
        joints: (17, 3) COCO keypoints

    Returns:
        Torso scale (distance in meters), or 0 if invalid
    """
    # COCO indices: left_shoulder=5, right_shoulder=6, left_hip=11, right_hip=12
    left_torso = np.linalg.norm(joints[5] - joints[11])   # left shoulder to left hip
    right_torso = np.linalg.norm(joints[6] - joints[12])  # right shoulder to right hip

    # Average of both sides for robustness
    torso_scale = (left_torso + right_torso) / 2

    # Sanity check: typical human torso is 0.4-0.7m
    if torso_scale < 0.1 or torso_scale > 2.0:
        return 0.0

    return torso_scale


def normalize_pose_scale(joints: np.ndarray, target_scale: float = 1.0) -> tuple:
    """
    Normalize pose to a consistent scale.

    Args:
        joints: (17, 3) COCO keypoints (already centered on pelvis)
        target_scale: Target torso scale (default 1.0 for unit scale)

    Returns:
        (normalized_joints, original_scale) tuple
        - normalized_joints: scaled to target_scale
        - original_scale: original torso scale for potential denormalization
    """
    scale = compute_torso_scale(joints)
    if scale < 0.01:
        return joints, 0.0  # Invalid scale, return as-is

    scale_factor = target_scale / scale
    return joints * scale_factor, scale


def load_keypoints3d(pkl_path: Path, use_optim: bool = True) -> np.ndarray:
    """Load 3D keypoints from AIST++ pickle file.

    AIST++ keypoints are in centimeters - we convert to meters.
    """
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)

    key = 'keypoints3d_optim' if use_optim else 'keypoints3d'
    keypoints = data[key]  # (n_frames, 17, 3)

    # Convert cm to meters
    return keypoints / 100.0


def get_video_name(seq_name: str, view: str = 'c01') -> str:
    """Convert AIST++ sequence name to video filename."""
    # seq_name like: gBR_sBM_cAll_d04_mBR0_ch01
    # video like: gBR_sBM_c01_d04_mBR0_ch01
    return seq_name.replace('cAll', view)


def project_3d_to_2d(points_3d: np.ndarray, R: np.ndarray, t: np.ndarray, K: np.ndarray,
                     image_width: int = 1920, image_height: int = 1080) -> np.ndarray:
    """
    Project 3D world points to normalized 2D image coordinates.

    Args:
        points_3d: (N, 3) 3D points in world coordinates
        R: (3, 3) rotation matrix (world to camera)
        t: (3,) translation vector
        K: (3, 3) intrinsic camera matrix
        image_width: Image width for normalization
        image_height: Image height for normalization

    Returns:
        (N, 2) normalized 2D coordinates in [0, 1] range
    """
    # Transform to camera frame: X_cam = R @ X_world + t
    points_cam = (R @ points_3d.T).T + t

    # Perspective projection
    points_2d_hom = (K @ points_cam.T).T

    # Perspective divide
    points_2d = points_2d_hom[:, :2] / points_2d_hom[:, 2:3]

    # Normalize to [0, 1] range
    points_2d[:, 0] = points_2d[:, 0] / image_width
    points_2d[:, 1] = points_2d[:, 1] / image_height

    return points_2d


def process_sequence_worker(args):
    """Worker function for multiprocessing - unpacks args and calls process_sequence."""
    return process_sequence(*args)


def process_sequence(
    video_path: Path,
    keypoints3d: np.ndarray,
    output_dir: Path,
    sequence_name: str,
    camera_params: dict,
    max_frames: int = 500,
    frame_skip: int = 2,
) -> int:
    """
    Process single AIST++ sequence.

    Args:
        video_path: Path to video file
        keypoints3d: (n_frames, 17, 3) ground truth keypoints (in meters)
        output_dir: Output directory
        sequence_name: Name for output files
        camera_params: Camera parameters dict with 'position', 'rotation', 'translation', 'intrinsic'
        max_frames: Max frames to process
        frame_skip: Process every N frames (60fps -> 30fps effective)

    Returns:
        Number of training examples generated
    """
    # Extract camera parameters
    camera_pos = camera_params['position']
    camera_R = camera_params['rotation']
    camera_t = camera_params['translation']
    camera_K = camera_params['intrinsic']

    # Skip if sequence already processed (check if ANY frame exists)
    existing = list(output_dir.glob(f"{sequence_name}_f*.npz"))
    if len(existing) >= max_frames // 2:
        return len(existing)

    # Create MediaPipe Pose instance for this worker
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        static_image_mode=True,
        model_complexity=2,
        enable_segmentation=False,
        min_detection_confidence=0.5,
    )

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        pose.close()
        return 0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    gt_frames = len(keypoints3d)

    # Use minimum of video frames and GT frames
    n_frames = min(total_frames, gt_frames, max_frames * frame_skip)

    examples_generated = 0
    frame_idx = 0
    while frame_idx < n_frames:
        ret, frame = cap.read()
        if not ret:
            break

        # Skip frames (60fps -> effective 30fps)
        if frame_idx % frame_skip != 0:
            frame_idx += 1
            continue

        # Get ground truth for this frame (already in meters)
        gt_pose = keypoints3d[frame_idx]  # (17, 3)

        # Check for invalid GT
        if np.isnan(gt_pose).any() or np.allclose(gt_pose, 0):
            frame_idx += 1
            continue

        # Run MediaPipe on frame
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_frame)

        # Need both 2D and 3D landmarks
        if not results.pose_world_landmarks or not results.pose_landmarks:
            frame_idx += 1
            continue

        # Extract MediaPipe 3D pose, visibility, and 2D pose
        mp_pose_data, visibility, pose_2d = extract_mediapipe_coco(results)

        # Project GT 3D pose to 2D using camera parameters
        projected_2d = project_3d_to_2d(gt_pose, camera_R, camera_t, camera_K)

        # c05 videos appear to be horizontally flipped in AIST++
        # Flip projected_2d X and swap left/right joints to match MediaPipe's view
        if '_c05' in sequence_name:
            projected_2d[:, 0] = 1.0 - projected_2d[:, 0]
            # Swap left/right joint pairs: (1,2), (3,4), (5,6), (7,8), (9,10), (11,12), (13,14), (15,16)
            swap_pairs = [(1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16)]
            for i, j in swap_pairs:
                projected_2d[[i, j]] = projected_2d[[j, i]]

        # Compute view angles from GT pose and actual camera position
        # BEFORE centering - we need world coordinates
        azimuth, elevation = compute_view_angles(gt_pose, camera_pos)

        # Compute camera position relative to subject's pelvis
        # This is what the model will learn to predict at inference time
        gt_pelvis = (gt_pose[11] + gt_pose[12]) / 2  # COCO: left_hip=11, right_hip=12
        camera_relative = camera_pos - gt_pelvis  # Camera pos in subject-centered coords

        # Center both on pelvis (for training, we want pelvis-centered)
        mp_centered = center_on_pelvis(mp_pose_data)
        gt_centered = center_on_pelvis(gt_pose)

        # Fix MediaPipe coordinate system to match AIST++
        # MediaPipe: Y down, Z toward camera
        # AIST++: Y up, Z away from camera (standard)
        mp_centered[:, 1] = -mp_centered[:, 1]  # Y: down -> up
        mp_centered[:, 2] = -mp_centered[:, 2]  # Z: toward camera -> away

        # === BODY FRAME ALIGNMENT (CRITICAL) ===
        # MediaPipe world landmarks are body-centric (aligned to torso).
        # AIST++ ground truth is in world coordinates.
        # Rotate MediaPipe to match GT's body frame orientation.
        mp_centered = align_body_frames(mp_centered, gt_centered)

        # === SCALE NORMALIZATION (CRITICAL) ===
        # MediaPipe and AIST++ may have different body size estimates.
        # Normalize BOTH to unit torso scale so model learns depth corrections,
        # not scale corrections.
        mp_normalized, mp_scale = normalize_pose_scale(mp_centered, target_scale=1.0)
        gt_normalized, gt_scale = normalize_pose_scale(gt_centered, target_scale=1.0)

        # Skip if either scale is invalid
        if mp_scale < 0.01 or gt_scale < 0.01:
            frame_idx += 1
            continue

        # Basic validation: nose should be above pelvis (use normalized poses)
        # In normalized space (torso=1.0), nose should be ~1.0-1.5 above pelvis
        # Use conservative threshold of 0.5 to allow for some variation
        if mp_normalized[0, 1] < 0.5 or gt_normalized[0, 1] < 0.5:
            frame_idx += 1
            continue

        # Save training pair
        output_path = output_dir / f"{sequence_name}_f{frame_idx:06d}.npz"

        # Skip if already exists (resume support)
        if output_path.exists():
            examples_generated += 1
            frame_idx += 1
            continue

        np.savez_compressed(
            output_path,
            corrupted=mp_normalized.astype(np.float32),    # (17, 3) MediaPipe, scale-normalized
            ground_truth=gt_normalized.astype(np.float32), # (17, 3) AIST++ GT, scale-normalized
            visibility=visibility.astype(np.float32),      # (17,) MediaPipe visibility
            pose_2d=pose_2d.astype(np.float32),            # (17, 2) MediaPipe 2D pose (key for camera!)
            projected_2d=projected_2d.astype(np.float32),  # (17, 2) GT projected to 2D via camera
            azimuth=np.float32(azimuth),                   # Horizontal angle: 0-360° (from camera pos!)
            elevation=np.float32(elevation),               # Vertical angle: -90 to +90° (from camera pos!)
            camera_relative=camera_relative.astype(np.float32),  # (3,) Camera pos relative to pelvis
            mp_scale=np.float32(mp_scale),                 # Original MediaPipe torso scale (for denorm)
            gt_scale=np.float32(gt_scale),                 # Original AIST++ torso scale (for denorm)
            sequence=sequence_name,
            frame_idx=frame_idx,
        )

        examples_generated += 1
        frame_idx += 1

    cap.release()
    pose.close()

    return examples_generated


def main():
    """Convert AIST++ dataset to training pairs."""
    storage_paths = StoragePaths.load()
    parser = argparse.ArgumentParser(description='Convert AIST++ to depth refinement training data')
    parser.add_argument('--workers', type=int, default=1, help='Number of parallel workers (default: 1)')
    parser.add_argument('--max-frames', type=int, default=180, help='Max frames per video (default: 180)')
    parser.add_argument('--frame-skip', type=int, default=3, help='Frame skip rate (default: 3)')
    args = parser.parse_args()

    print("=" * 80)
    print("AIST++ -> DEPTH REFINEMENT TRAINING DATA")
    print("=" * 80)
    print()
    print("Key: Using REAL MediaPipe errors from video frames!")
    print("     View angle computed from ACTUAL camera position (not torso normal)")
    print()

    # Paths
    aistpp_dir = storage_paths.training_root / "AIST++"
    annotations_dir = aistpp_dir / "annotations"
    videos_dir = aistpp_dir / "videos"
    cameras_dir = annotations_dir / "cameras"
    mapping_file = cameras_dir / "mapping.txt"
    output_dir = storage_paths.training_root / "aistpp_converted"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check directories
    keypoints_dir = annotations_dir / "keypoints3d"
    if not keypoints_dir.exists():
        print(f"ERROR: keypoints3d not found at {keypoints_dir}")
        return

    if not videos_dir.exists():
        print(f"ERROR: videos not found at {videos_dir}")
        return

    if not cameras_dir.exists():
        print(f"ERROR: cameras not found at {cameras_dir}")
        return

    # Find sequences
    kp_files = sorted(keypoints_dir.glob("*.pkl"))
    print(f"Found {len(kp_files)} keypoint sequences")

    # Check for videos
    video_files = list(videos_dir.glob("*.mp4"))
    print(f"Found {len(video_files)} videos")

    if len(video_files) == 0:
        print("No videos found.")
        return

    video_dict = {v.stem: v for v in video_files}
    print()

    # Cache camera params per setting
    camera_cache = {}

    # Camera views: 6 views for diverse angles
    camera_views = ['c01', 'c02', 'c03', 'c05', 'c07', 'c09']

    # Build task list
    tasks = []
    for kp_file in kp_files:
        seq_name = kp_file.stem

        try:
            gt_keypoints = load_keypoints3d(kp_file)
        except Exception:
            continue

        setting = get_camera_setting(seq_name, mapping_file)

        for cam_view in camera_views:
            video_name = get_video_name(seq_name, cam_view)
            video_path = video_dict.get(video_name)

            if video_path is None:
                continue

            cache_key = f"{setting}_{cam_view}"
            if cache_key not in camera_cache:
                try:
                    camera_cache[cache_key] = load_camera_params(cameras_dir, setting, cam_view)
                except Exception:
                    continue

            camera_params = camera_cache[cache_key]
            seq_name_cam = f"{seq_name}_{cam_view}"

            tasks.append((
                video_path,
                gt_keypoints,
                output_dir,
                seq_name_cam,
                camera_params,
                args.max_frames,
                args.frame_skip,
            ))

    # Process tasks
    total_examples = 0
    if args.workers > 1:
        print(f"Processing {len(tasks)} tasks with {args.workers} parallel workers...")
        print()
        with Pool(args.workers) as pool:
            for count in tqdm(pool.imap_unordered(process_sequence_worker, tasks), total=len(tasks)):
                total_examples += count
    else:
        # Single-threaded: verbose output like before
        print(f"Processing {len(tasks)} sequences...")
        print()
        for task in tasks:
            video_path, gt_keypoints, out_dir, seq_name_cam, cam_params, max_frames, frame_skip = task
            n_frames = len(gt_keypoints)
            print(f"Processing: {seq_name_cam} ({n_frames} frames)")
            count = process_sequence_worker(task)
            total_examples += count
            print(f"  -> {count} training pairs")

    # Final count
    final_count = len(list(output_dir.glob("*.npz")))
    print()
    print("=" * 80)
    print(f"Completed: {final_count} training samples")
    print(f"Saved to: {output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()

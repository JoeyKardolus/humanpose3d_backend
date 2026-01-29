#!/usr/bin/env python3
"""
Convert AIST++ dataset to training pairs for POF model using RTMPose 2D.

This is the RTMPose variant of convert_aistpp.py. Key differences:
- Uses RTMPose for 2D pose detection (higher accuracy than MediaPipe)
- RTMPose outputs COCO-17 natively (no mapping needed)
- RTMPose provides only 2D - view angles computed from GT only
- Training pairs: RTMPose 2D + AIST++ 3D GT for POF learning

Produces training data for the camera-space POF model that learns
to reconstruct 3D from 2D observations.
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
import pickle
import numpy as np
from tqdm import tqdm
import cv2
from scipy.spatial.transform import Rotation


# COCO 17 keypoints (standard order)
COCO_KEYPOINTS = [
    'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
    'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
    'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
    'left_knee', 'right_knee', 'left_ankle', 'right_ankle',
]


def load_camera_params(cameras_dir: Path, setting_name: str, camera_name: str = 'c01') -> dict:
    """Load camera calibration from AIST++."""
    setting_file = cameras_dir / f"{setting_name}.json"
    with open(setting_file, 'r') as f:
        cameras = json.load(f)

    for cam in cameras:
        if cam['name'] == camera_name:
            rvec = np.array(cam['rotation'])
            R = Rotation.from_rotvec(rvec).as_matrix()
            t = np.array(cam['translation']) / 100.0  # cm -> meters
            cam_pos = -R.T @ t
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
    return 'setting1'


def compute_view_angles(pose_3d: np.ndarray, camera_pos: np.ndarray) -> tuple:
    """
    Compute camera viewing angles in subject's body-relative frame.

    Returns:
        (azimuth, elevation) in degrees
        - azimuth: 0-360° (0°=front, 90°=right, 180°=back, 270°=left)
        - elevation: -90 to +90° (above/below horizontal)
    """
    # COCO indices
    left_shoulder = pose_3d[5]
    right_shoulder = pose_3d[6]
    left_hip = pose_3d[11]
    right_hip = pose_3d[12]
    nose = pose_3d[0]

    torso_center = (left_shoulder + right_shoulder + left_hip + right_hip) / 4

    # Build body frame
    left_side = (left_hip + left_shoulder) / 2
    right_side = (right_hip + right_shoulder) / 2
    right_axis = right_side - left_side
    right_norm = np.linalg.norm(right_axis)
    if right_norm < 1e-6:
        return 0.0, 0.0
    right_axis = right_axis / right_norm

    hip_center = (left_hip + right_hip) / 2
    shoulder_center = (left_shoulder + right_shoulder) / 2
    up_axis = shoulder_center - hip_center
    up_norm = np.linalg.norm(up_axis)
    if up_norm < 1e-6:
        return 0.0, 0.0
    up_axis = up_axis / up_norm

    forward_axis = np.cross(right_axis, up_axis)
    forward_norm = np.linalg.norm(forward_axis)
    if forward_norm < 1e-6:
        forward_axis = np.array([0.0, 0.0, 1.0])
    else:
        forward_axis = forward_axis / forward_norm

    nose_dir = nose - torso_center
    if np.dot(forward_axis, nose_dir) < 0:
        forward_axis = -forward_axis

    up_axis = np.cross(right_axis, forward_axis)
    up_axis = up_axis / (np.linalg.norm(up_axis) + 1e-8)

    # Camera direction in subject's frame
    subj_to_cam = camera_pos - torso_center
    cam_dist = np.linalg.norm(subj_to_cam)
    if cam_dist < 1e-6:
        return 0.0, 0.0

    forward_component = np.dot(subj_to_cam, forward_axis)
    right_component = np.dot(subj_to_cam, right_axis)
    up_component = np.dot(subj_to_cam, up_axis)

    azimuth = np.degrees(np.arctan2(right_component, forward_component))
    if azimuth < 0:
        azimuth += 360.0

    horiz_dist = np.sqrt(forward_component**2 + right_component**2)
    if horiz_dist < 1e-6:
        elevation = 90.0 if up_component > 0 else -90.0
    else:
        elevation = np.degrees(np.arctan2(up_component, horiz_dist))

    return float(azimuth), float(elevation)


def center_on_pelvis(joints: np.ndarray) -> np.ndarray:
    """Center skeleton on pelvis (midpoint of hips)."""
    pelvis = (joints[11] + joints[12]) / 2
    return joints - pelvis


def compute_torso_scale(joints: np.ndarray) -> float:
    """Compute torso scale (hip-shoulder distance)."""
    left_torso = np.linalg.norm(joints[5] - joints[11])
    right_torso = np.linalg.norm(joints[6] - joints[12])
    torso_scale = (left_torso + right_torso) / 2
    if torso_scale < 0.1 or torso_scale > 2.0:
        return 0.0
    return torso_scale


def normalize_pose_scale(joints: np.ndarray, target_scale: float = 1.0) -> tuple:
    """Normalize pose to consistent scale."""
    scale = compute_torso_scale(joints)
    if scale < 0.01:
        return joints, 0.0
    scale_factor = target_scale / scale
    return joints * scale_factor, scale


def load_keypoints3d(pkl_path: Path, use_optim: bool = True) -> np.ndarray:
    """Load 3D keypoints from AIST++ pickle file (returns meters)."""
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    key = 'keypoints3d_optim' if use_optim else 'keypoints3d'
    return data[key] / 100.0  # cm -> meters


def get_video_name(seq_name: str, view: str = 'c01') -> str:
    """Convert AIST++ sequence name to video filename."""
    return seq_name.replace('cAll', view)


def project_3d_to_2d(points_3d: np.ndarray, R: np.ndarray, t: np.ndarray, K: np.ndarray,
                     image_width: int = 1920, image_height: int = 1080) -> np.ndarray:
    """Project 3D points to normalized 2D image coordinates."""
    points_cam = (R @ points_3d.T).T + t
    points_2d_hom = (K @ points_cam.T).T
    points_2d = points_2d_hom[:, :2] / points_2d_hom[:, 2:3]
    points_2d[:, 0] = points_2d[:, 0] / image_width
    points_2d[:, 1] = points_2d[:, 1] / image_height
    return points_2d


def process_sequence_worker(args):
    """Worker function for multiprocessing."""
    return process_sequence(*args)


def process_sequence(
    video_path: Path,
    keypoints3d: np.ndarray,
    output_dir: Path,
    sequence_name: str,
    camera_params: dict,
    max_frames: int = 500,
    frame_skip: int = 2,
    rtmpose_model: str = 'm',
) -> int:
    """
    Process single AIST++ sequence using RTMPose.

    Args:
        video_path: Path to video file
        keypoints3d: (n_frames, 17, 3) ground truth keypoints (meters)
        output_dir: Output directory
        sequence_name: Name for output files
        camera_params: Camera parameters dict
        max_frames: Max frames to process
        frame_skip: Process every N frames
        rtmpose_model: RTMPose model size (s/m/l)

    Returns:
        Number of training examples generated
    """
    # Extract camera parameters
    camera_pos = camera_params['position']
    camera_R = camera_params['rotation']
    camera_t = camera_params['translation']
    camera_K = camera_params['intrinsic']

    # Skip if sequence already processed
    existing = list(output_dir.glob(f"{sequence_name}_f*.npz"))
    if len(existing) >= max_frames // 2:
        return len(existing)

    # Create RTMPose instance
    mode_map = {'s': 'lightweight', 'm': 'balanced', 'l': 'performance'}
    mode = mode_map.get(rtmpose_model, 'balanced')
    try:
        from rtmlib import Wholebody
        rtmpose = Wholebody(
            mode=mode,
            backend='onnxruntime',
            device='cuda',
        )
    except ImportError:
        print("ERROR: rtmlib not installed. Install with: pip install rtmlib")
        return 0
    except Exception as e:
        print(f"ERROR initializing RTMPose: {e}")
        return 0

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return 0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    image_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    image_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    gt_frames = len(keypoints3d)

    n_frames = min(total_frames, gt_frames, max_frames * frame_skip)

    examples_generated = 0
    frame_idx = 0
    while frame_idx < n_frames:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_skip != 0:
            frame_idx += 1
            continue

        gt_pose = keypoints3d[frame_idx]

        if np.isnan(gt_pose).any() or np.allclose(gt_pose, 0):
            frame_idx += 1
            continue

        # Run RTMPose (expects BGR)
        keypoints, scores = rtmpose(frame)

        if len(keypoints) == 0:
            frame_idx += 1
            continue

        # Get first person, COCO-17 (first 17 keypoints from Wholebody)
        pose_2d_pixels = keypoints[0][:17]  # (17, 2) pixel coords
        visibility = scores[0][:17]  # (17,) confidence

        # Normalize to [0, 1]
        pose_2d = np.zeros((17, 2), dtype=np.float32)
        pose_2d[:, 0] = pose_2d_pixels[:, 0] / image_width
        pose_2d[:, 1] = pose_2d_pixels[:, 1] / image_height

        # Project GT 3D to 2D
        projected_2d = project_3d_to_2d(gt_pose, camera_R, camera_t, camera_K,
                                        image_width, image_height)

        # c05 flip compensation
        if '_c05' in sequence_name:
            projected_2d[:, 0] = 1.0 - projected_2d[:, 0]
            swap_pairs = [(1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16)]
            for i, j in swap_pairs:
                projected_2d[[i, j]] = projected_2d[[j, i]]
            # Also flip RTMPose 2D for consistency
            pose_2d[:, 0] = 1.0 - pose_2d[:, 0]
            for i, j in swap_pairs:
                pose_2d[[i, j]] = pose_2d[[j, i]]

        # Compute view angles from GT (since RTMPose only gives 2D)
        azimuth, elevation = compute_view_angles(gt_pose, camera_pos)

        gt_pelvis = (gt_pose[11] + gt_pose[12]) / 2
        camera_relative = camera_pos - gt_pelvis

        # Center and normalize GT pose
        gt_centered = center_on_pelvis(gt_pose)
        gt_normalized, gt_scale = normalize_pose_scale(gt_centered, target_scale=1.0)

        if gt_scale < 0.01:
            frame_idx += 1
            continue

        # Validation: nose above pelvis
        if gt_normalized[0, 1] < 0.5:
            frame_idx += 1
            continue

        output_path = output_dir / f"{sequence_name}_f{frame_idx:06d}.npz"

        if output_path.exists():
            examples_generated += 1
            frame_idx += 1
            continue

        np.savez_compressed(
            output_path,
            # RTMPose provides 2D only - no corrupted 3D
            pose_2d=pose_2d.astype(np.float32),              # (17, 2) RTMPose 2D (key input!)
            projected_2d=projected_2d.astype(np.float32),    # (17, 2) GT projected to 2D
            ground_truth=gt_normalized.astype(np.float32),   # (17, 3) AIST++ GT, normalized
            visibility=visibility.astype(np.float32),        # (17,) RTMPose confidence
            azimuth=np.float32(azimuth),
            elevation=np.float32(elevation),
            camera_relative=camera_relative.astype(np.float32),
            camera_R=camera_R.astype(np.float32),
            gt_scale=np.float32(gt_scale),
            sequence=sequence_name,
            frame_idx=frame_idx,
            estimator='rtmpose',
        )

        examples_generated += 1
        frame_idx += 1

    cap.release()
    return examples_generated


def main():
    """Convert AIST++ dataset to training pairs using RTMPose."""
    parser = argparse.ArgumentParser(
        description='Convert AIST++ to POF training data using RTMPose 2D'
    )
    parser.add_argument('--workers', type=int, default=1,
                        help='Number of parallel workers (default: 1)')
    parser.add_argument('--max-frames', type=int, default=180,
                        help='Max frames per video (default: 180)')
    parser.add_argument('--frame-skip', type=int, default=3,
                        help='Frame skip rate (default: 3)')
    parser.add_argument('--rtmpose-model', type=str, default='m',
                        choices=['s', 'm', 'l'],
                        help='RTMPose model size (default: m)')
    args = parser.parse_args()

    print("=" * 80)
    print("AIST++ -> POF TRAINING DATA (RTMPose)")
    print("=" * 80)
    print()
    print(f"Using RTMPose-{args.rtmpose_model} for 2D pose detection")
    print("RTMPose provides ~2x better 2D accuracy than MediaPipe")
    print()

    # Paths
    aistpp_dir = Path("data/AIST++")
    annotations_dir = aistpp_dir / "annotations"
    videos_dir = aistpp_dir / "videos"
    cameras_dir = annotations_dir / "cameras"
    mapping_file = cameras_dir / "mapping.txt"
    output_dir = Path("data/training/aistpp_rtmpose")
    output_dir.mkdir(parents=True, exist_ok=True)

    keypoints_dir = annotations_dir / "keypoints3d"
    if not keypoints_dir.exists():
        print(f"ERROR: keypoints3d not found at {keypoints_dir}")
        return

    if not videos_dir.exists():
        print(f"ERROR: videos not found at {videos_dir}")
        return

    kp_files = sorted(keypoints_dir.glob("*.pkl"))
    print(f"Found {len(kp_files)} keypoint sequences")

    video_files = list(videos_dir.glob("*.mp4"))
    print(f"Found {len(video_files)} videos")

    if len(video_files) == 0:
        print("No videos found.")
        return

    video_dict = {v.stem: v for v in video_files}
    print()

    camera_cache = {}
    camera_views = ['c01', 'c02', 'c03', 'c05', 'c07', 'c09']

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
                args.rtmpose_model,
            ))

    total_examples = 0
    if args.workers > 1:
        print(f"Processing {len(tasks)} tasks with {args.workers} parallel workers...")
        print("Note: Each worker initializes its own RTMPose model")
        print()
        with Pool(args.workers) as pool:
            for count in tqdm(pool.imap_unordered(process_sequence_worker, tasks), total=len(tasks)):
                total_examples += count
    else:
        print(f"Processing {len(tasks)} sequences...")
        print()
        for task in tasks:
            video_path, gt_keypoints, out_dir, seq_name_cam, cam_params, max_frames, frame_skip, model = task
            n_frames = len(gt_keypoints)
            print(f"Processing: {seq_name_cam} ({n_frames} frames)")
            count = process_sequence_worker(task)
            total_examples += count
            print(f"  -> {count} training pairs")

    final_count = len(list(output_dir.glob("*.npz")))
    print()
    print("=" * 80)
    print(f"Completed: {final_count} training samples")
    print(f"Saved to: {output_dir}")
    print()
    print("To train POF model with RTMPose data:")
    print("  uv run --group neural python scripts/train/pof_model.py \\")
    print("    --data data/training/aistpp_rtmpose --epochs 50 --bf16")
    print("=" * 80)


if __name__ == "__main__":
    main()

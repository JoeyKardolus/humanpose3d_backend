#!/usr/bin/env python3
"""
Convert CMU Panoptic MTC (Monocular Total Capture) dataset to training pairs.

Dataset: http://domedb.perception.cs.cmu.edu/mtc.html
Size: ~270GB

CMU Panoptic MTC provides:
- Multi-view HD videos (31 cameras, 1920x1080, 30fps)
- 3D keypoints in COCO19 format (19 joints)
- Camera calibration data

This script:
1. Downloads the dataset (or uses existing)
2. Runs MediaPipe on video frames -> noisy 3D pose (REAL errors!)
3. Loads MTC keypoints -> ground truth 3D pose
4. Converts COCO19 to COCO17 format
5. Computes view angle from ground truth torso plane
6. Creates training pairs for depth refinement model

Usage:
    # First, explore the dataset structure
    python scripts/convert_cmu_mtc_to_training.py --explore --mtc-dir ~/.humanpose3d/training/mtc

    # Download and convert (full dataset ~270GB)
    python scripts/convert_cmu_mtc_to_training.py --download --mtc-dir ~/.humanpose3d/training/mtc

    # Convert already downloaded dataset
    python scripts/convert_cmu_mtc_to_training.py --mtc-dir /path/to/mtc_dataset

    # Quick test with limited sequences
    python scripts/convert_cmu_mtc_to_training.py --mtc-dir ~/.humanpose3d/training/mtc --max-sequences 5

Download manually (recommended for 270GB):
    wget -c http://domedb.perception.cs.cmu.edu/data/mtc/mtc_dataset.tar.gz
    tar -xzf mtc_dataset.tar.gz -C ~/.humanpose3d/training/mtc
"""

import os
import sys

# Suppress ALL TensorFlow, MediaPipe, and absl warnings BEFORE any imports
# This must be done before importing anything that might trigger these libraries
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['GLOG_minloglevel'] = '3'  # 3 = FATAL only
os.environ['GLOG_logtostderr'] = '0'
os.environ['GLOG_log_dir'] = '/tmp'
os.environ['ABSL_MIN_LOG_LEVEL'] = '3'
os.environ['MEDIAPIPE_DISABLE_GPU'] = '1'  # Suppress GPU/GL messages
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = ''  # Force CPU for MediaPipe (faster anyway)

# Redirect stderr to suppress C++ level warnings that Python can't catch
import io
import contextlib

class SuppressMediaPipeOutput:
    """Context manager to suppress MediaPipe's C++ stderr output."""
    def __init__(self):
        self._stderr = None
        self._devnull = None

    def __enter__(self):
        # Only suppress in worker processes, not main process
        return self

    def __exit__(self, *args):
        pass

from pathlib import Path

import warnings
warnings.filterwarnings('ignore')

import logging
logging.getLogger('tensorflow').setLevel(logging.FATAL)
logging.getLogger('absl').setLevel(logging.FATAL)
logging.getLogger('mediapipe').setLevel(logging.FATAL)

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.application.config.paths import StoragePaths
from src.depth_refinement.data_utils import align_body_frames

import argparse
import json
import pickle
import subprocess
import tarfile
import numpy as np
from tqdm import tqdm
import cv2
import mediapipe as mp
from scipy.spatial.transform import Rotation
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import Pool
import multiprocessing


# ============================================================================
# JOINT DEFINITIONS
# ============================================================================

# CMU Panoptic COCO19 keypoints
COCO19_KEYPOINTS = [
    'neck',           # 0
    'nose',           # 1
    'body_center',    # 2 (hip center/pelvis)
    'left_shoulder',  # 3
    'left_elbow',     # 4
    'left_wrist',     # 5
    'left_hip',       # 6
    'left_knee',      # 7
    'left_ankle',     # 8
    'right_shoulder', # 9
    'right_elbow',    # 10
    'right_wrist',    # 11
    'right_hip',      # 12
    'right_knee',     # 13
    'right_ankle',    # 14
    'left_eye',       # 15
    'right_eye',      # 16
    'left_ear',       # 17
    'right_ear',      # 18
]

# Standard COCO 17 keypoints (our training format)
COCO17_KEYPOINTS = [
    'nose',           # 0
    'left_eye',       # 1
    'right_eye',      # 2
    'left_ear',       # 3
    'right_ear',      # 4
    'left_shoulder',  # 5
    'right_shoulder', # 6
    'left_elbow',     # 7
    'right_elbow',    # 8
    'left_wrist',     # 9
    'right_wrist',    # 10
    'left_hip',       # 11
    'right_hip',      # 12
    'left_knee',      # 13
    'right_knee',     # 14
    'left_ankle',     # 15
    'right_ankle',    # 16
]

# Mapping from COCO19 index to COCO17 index
# Some COCO19 joints (neck, body_center) don't exist in COCO17
COCO19_TO_COCO17 = {
    1: 0,   # nose
    15: 1,  # left_eye
    16: 2,  # right_eye
    17: 3,  # left_ear
    18: 4,  # right_ear
    3: 5,   # left_shoulder
    9: 6,   # right_shoulder
    4: 7,   # left_elbow
    10: 8,  # right_elbow
    5: 9,   # left_wrist
    11: 10, # right_wrist
    6: 11,  # left_hip
    12: 12, # right_hip
    7: 13,  # left_knee
    13: 14, # right_knee
    8: 15,  # left_ankle
    14: 16, # right_ankle
}

# MediaPipe landmark indices for COCO 17 joints
MEDIAPIPE_TO_COCO17 = {
    0: 0,    # nose
    1: 2,    # left_eye (MediaPipe 2 = left eye inner, close enough)
    2: 5,    # right_eye (MediaPipe 5 = right eye inner)
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


# ============================================================================
# DOWNLOAD FUNCTIONS
# ============================================================================

def download_mtc_dataset(output_dir: Path, url: str = None):
    """
    Download CMU MTC dataset.

    Args:
        output_dir: Directory to save dataset
        url: Override download URL
    """
    if url is None:
        url = "http://domedb.perception.cs.cmu.edu/data/mtc/mtc_dataset.tar.gz"

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tar_path = output_dir / "mtc_dataset.tar.gz"

    if tar_path.exists():
        print(f"Archive already exists: {tar_path}")
    else:
        print(f"Downloading MTC dataset (~270GB) to {tar_path}")
        print("This will take a while...")

        # Use wget for resumable download
        cmd = [
            "wget", "-c",  # Continue partial downloads
            "--progress=bar:force",
            "-O", str(tar_path),
            url
        ]
        subprocess.run(cmd, check=True)

    # Extract if not already extracted
    extracted_marker = output_dir / ".extracted"
    if not extracted_marker.exists():
        print(f"Extracting {tar_path}...")
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(output_dir)
        extracted_marker.touch()
        print("Extraction complete!")
    else:
        print("Dataset already extracted")

    return output_dir


# ============================================================================
# DATA LOADING
# ============================================================================

def load_mtc_annotations(pkl_file: Path) -> dict:
    """
    Load MTC annotations from pickle file.

    Returns dict with 'training_data' and 'testing_data' lists.
    Each item has: seqName, frame_str, id, body (with landmarks).
    """
    with open(pkl_file, 'rb') as f:
        data = pickle.load(f)
    return data


def load_mtc_camera_data(pkl_file: Path) -> dict:
    """
    Load MTC camera calibration from pickle file.

    Returns dict mapping seq_name -> camera_id -> {K, R, t, distCoef}.
    """
    with open(pkl_file, 'rb') as f:
        data = pickle.load(f)
    return data


def get_camera_params(camera_data: dict, seq_name: str, cam_id: int) -> dict:
    """
    Get camera parameters for a specific sequence and camera.

    Returns dict with K, R, t, position or None if not found.
    """
    if seq_name not in camera_data:
        return None

    seq_cameras = camera_data[seq_name]
    if cam_id not in seq_cameras:
        return None

    cam = seq_cameras[cam_id]

    # Get intrinsics and extrinsics
    K = np.array(cam['K']) if 'K' in cam else None
    R = np.array(cam['R']) if 'R' in cam else None
    t = np.array(cam['t']).flatten() if 't' in cam else None

    # Camera position in world coords: C = -R^T @ t
    position = None
    if R is not None and t is not None:
        position = -R.T @ t

    return {
        'K': K,
        'R': R,
        't': t,
        'position': position,
    }


def load_cmu_calibration(calib_file: Path) -> dict:
    """
    Load CMU Panoptic camera calibration from JSON (legacy format).

    Returns dict mapping camera_id -> {K, R, t, position}
    """
    with open(calib_file, 'r') as f:
        calib_data = json.load(f)

    cameras = {}
    for cam in calib_data.get('cameras', []):
        cam_id = cam.get('name', cam.get('id'))

        # Intrinsics
        K = np.array(cam['K']).reshape(3, 3) if 'K' in cam else None

        # Extrinsics
        R = np.array(cam['R']).reshape(3, 3) if 'R' in cam else None
        t = np.array(cam['t']).reshape(3) if 't' in cam else None

        # Camera position in world coords
        if R is not None and t is not None:
            position = -R.T @ t
        else:
            position = None

        cameras[cam_id] = {
            'K': K,
            'R': R,
            't': t,
            'position': position,
        }

    return cameras


def load_skeleton_json(json_file: Path) -> list:
    """
    Load 3D skeleton data from CMU Panoptic JSON format (legacy).

    Returns list of bodies, each with 'id' and 'joints19' array.
    """
    with open(json_file, 'r') as f:
        data = json.load(f)

    bodies = []
    for body in data.get('bodies', []):
        body_id = body.get('id', 0)

        # Joints are stored as flat array: [x1,y1,z1,c1, x2,y2,z2,c2, ...]
        joints_flat = np.array(body.get('joints19', body.get('joints', [])))

        if len(joints_flat) >= 19 * 4:
            joints_flat = joints_flat[:19*4]
            joints = joints_flat.reshape(19, 4)  # (19, 4) with x,y,z,confidence

            bodies.append({
                'id': body_id,
                'joints19': joints[:, :3],  # (19, 3) xyz only
                'confidence': joints[:, 3],  # (19,) confidence
            })

    return bodies


def parse_mtc_landmarks(landmarks: list) -> tuple:
    """
    Parse MTC body landmarks from flat list to (19, 3) array.

    MTC landmarks are stored as: [x0, y0, z0, x1, y1, z1, ...]

    MTC coordinate system: Y points DOWN, Z points toward camera
    MediaPipe (after flip): Y points UP, Z points away from camera
    We convert MTC to match MediaPipe convention for consistency.

    Returns:
        joints19: (19, 3) array of joint positions (Y-up, Z-away convention)
        confidence: (19,) array of ones (MTC doesn't have per-joint confidence)
    """
    landmarks = np.array(landmarks)
    n_joints = len(landmarks) // 3
    if n_joints < 19:
        return None, None

    joints = landmarks[:19*3].reshape(19, 3)

    # Flip Y and Z to convert coordinate systems
    # MTC: Y-down, Z-toward -> Standard: Y-up, Z-away
    joints[:, 1] = -joints[:, 1]
    joints[:, 2] = -joints[:, 2]

    confidence = np.ones(19, dtype=np.float32)

    return joints, confidence


def convert_coco19_to_coco17(joints19: np.ndarray, confidence19: np.ndarray = None) -> tuple:
    """
    Convert COCO19 joints to COCO17 format.

    Args:
        joints19: (19, 3) array of 3D joint positions
        confidence19: (19,) array of confidence scores (optional)

    Returns:
        joints17: (17, 3) array in COCO17 format
        confidence17: (17,) array of confidence scores
    """
    joints17 = np.zeros((17, 3), dtype=np.float32)
    confidence17 = np.ones(17, dtype=np.float32)

    for coco19_idx, coco17_idx in COCO19_TO_COCO17.items():
        joints17[coco17_idx] = joints19[coco19_idx]
        if confidence19 is not None:
            confidence17[coco17_idx] = confidence19[coco19_idx]

    return joints17, confidence17


# ============================================================================
# MEDIAPIPE PROCESSING
# ============================================================================

def init_mediapipe():
    """Initialize MediaPipe pose detector."""
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        static_image_mode=True,
        model_complexity=2,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return pose


def process_frame_mediapipe(frame_bgr: np.ndarray, pose_detector) -> tuple:
    """
    Run MediaPipe on a single frame.

    Returns:
        pose_3d: (17, 3) COCO17 format or None if detection failed
        pose_2d: (17, 2) normalized image coords or None
        visibility: (17,) visibility scores or None
    """
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    results = pose_detector.process(frame_rgb)

    if not results.pose_world_landmarks:
        return None, None, None

    # Extract world landmarks (in meters, hip-centered)
    world_lm = results.pose_world_landmarks.landmark

    # Extract image landmarks (normalized 0-1)
    image_lm = results.pose_landmarks.landmark if results.pose_landmarks else None

    pose_3d = np.zeros((17, 3), dtype=np.float32)
    pose_2d = np.zeros((17, 2), dtype=np.float32)
    visibility = np.zeros(17, dtype=np.float32)

    for coco_idx, mp_idx in MEDIAPIPE_TO_COCO17.items():
        lm = world_lm[mp_idx]
        # MediaPipe: Y down, Z toward camera -> flip Y and Z
        pose_3d[coco_idx] = [lm.x, -lm.y, -lm.z]
        visibility[coco_idx] = lm.visibility

        if image_lm:
            img_lm = image_lm[mp_idx]
            pose_2d[coco_idx] = [img_lm.x, img_lm.y]

    return pose_3d, pose_2d, visibility


# ============================================================================
# VIEW ANGLE COMPUTATION
# ============================================================================

def compute_view_angles(pose_3d: np.ndarray, camera_pos: np.ndarray) -> tuple:
    """
    Compute camera viewing angles in subject's body-relative coordinate frame.

    Args:
        pose_3d: (17, 3) COCO17 keypoints
        camera_pos: (3,) Camera position in world coords

    Returns:
        (azimuth, elevation) in degrees
    """
    left_shoulder = pose_3d[5]
    right_shoulder = pose_3d[6]
    left_hip = pose_3d[11]
    right_hip = pose_3d[12]

    torso_center = (left_shoulder + right_shoulder + left_hip + right_hip) / 4

    # Build body-relative coordinate frame
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
        return 0.0, 0.0
    forward_axis = forward_axis / forward_norm

    # Re-orthogonalize
    right_axis = np.cross(up_axis, forward_axis)
    right_axis = right_axis / np.linalg.norm(right_axis)

    # Camera vector in body frame
    cam_vec = camera_pos - torso_center
    cam_dist = np.linalg.norm(cam_vec)
    if cam_dist < 1e-6:
        return 0.0, 0.0
    cam_vec = cam_vec / cam_dist

    # Project to body frame
    cam_forward = np.dot(cam_vec, forward_axis)
    cam_right = np.dot(cam_vec, right_axis)
    cam_up = np.dot(cam_vec, up_axis)

    # Azimuth: angle in horizontal plane (0=front, 90=right, 180=back, 270=left)
    azimuth = np.degrees(np.arctan2(cam_right, cam_forward))
    if azimuth < 0:
        azimuth += 360

    # Elevation: angle above/below horizontal
    horizontal_dist = np.sqrt(cam_forward**2 + cam_right**2)
    elevation = np.degrees(np.arctan2(cam_up, horizontal_dist))

    return float(azimuth), float(elevation)


def compute_torso_scale(pose_3d: np.ndarray) -> float:
    """Compute torso scale as average hip-to-shoulder distance."""
    left_shoulder = pose_3d[5]
    right_shoulder = pose_3d[6]
    left_hip = pose_3d[11]
    right_hip = pose_3d[12]

    left_torso = np.linalg.norm(left_shoulder - left_hip)
    right_torso = np.linalg.norm(right_shoulder - right_hip)

    return (left_torso + right_torso) / 2


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


# ============================================================================
# MTC PICKLE FORMAT PROCESSING (a4_release structure)
# ============================================================================

def process_mtc_worker(args):
    """Worker function for multiprocessing - unpacks args and calls process_mtc_sample."""
    # Suppress stderr in worker processes to hide MediaPipe C++ warnings
    # Must redirect at file descriptor level to catch C++ output
    import os
    import sys

    # Re-apply environment variables in worker process
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ['GLOG_minloglevel'] = '3'
    os.environ['GLOG_logtostderr'] = '0'
    os.environ['ABSL_MIN_LOG_LEVEL'] = '3'

    # Redirect stderr at file descriptor level (catches C++ output)
    old_stderr_fd = os.dup(2)
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull_fd, 2)

    try:
        result = process_mtc_sample(*args)
    finally:
        # Restore stderr
        os.dup2(old_stderr_fd, 2)
        os.close(old_stderr_fd)
        os.close(devnull_fd)

    return result


def process_mtc_sample(
    img_path: Path,
    gt_joints19: np.ndarray,
    cam_params: dict,
    output_dir: Path,
    seq_name: str,
    frame_str: str,
    cam_id: int,
) -> int:
    """
    Process a single MTC sample (image + GT pose).

    Returns 1 if sample created, 0 otherwise.
    """
    # Build output filename
    sample_name = f"{seq_name}_c{cam_id:02d}_f{frame_str}"
    sample_path = output_dir / f"{sample_name}.npz"

    # Resume support: skip if file already exists
    if sample_path.exists():
        return 1

    # Load image
    if not img_path.exists():
        return 0

    frame = cv2.imread(str(img_path))
    if frame is None:
        return 0

    # Initialize MediaPipe (per-worker) - stderr already redirected in worker wrapper
    mp_pose = mp.solutions.pose
    pose_detector = mp_pose.Pose(
        static_image_mode=True,
        model_complexity=2,
        enable_segmentation=False,
        min_detection_confidence=0.5,
    )

    # Run MediaPipe
    mp_pose_3d, mp_2d, mp_vis = process_frame_mediapipe(frame, pose_detector)
    pose_detector.close()

    if mp_pose_3d is None:
        return 0

    # Convert GT COCO19 to COCO17
    gt_pose, gt_conf = convert_coco19_to_coco17(gt_joints19)

    # Center both poses on pelvis
    gt_pelvis = (gt_pose[11] + gt_pose[12]) / 2
    mp_pelvis = (mp_pose_3d[11] + mp_pose_3d[12]) / 2

    gt_centered = gt_pose - gt_pelvis
    mp_centered = mp_pose_3d - mp_pelvis

    # Align body frames
    mp_centered = align_body_frames(mp_centered, gt_centered)

    # Compute scales
    gt_scale = compute_torso_scale(gt_centered)
    mp_scale = compute_torso_scale(mp_centered)

    if gt_scale < 0.01 or mp_scale < 0.01:
        return 0

    # Normalize to unit torso
    gt_normalized = gt_centered / gt_scale
    mp_normalized = mp_centered / mp_scale

    # Nose validation
    if mp_normalized[0, 1] < 0.5 or gt_normalized[0, 1] < 0.5:
        return 0

    # Compute view angles
    cam_pos = cam_params.get('position')
    if cam_pos is not None:
        cam_relative = cam_pos - gt_pelvis
        azimuth, elevation = compute_view_angles(gt_centered, cam_pos)
    else:
        cam_relative = np.array([0., 0., 2.])
        azimuth, elevation = 0.0, 0.0

    # Compute projected_2d
    # NOTE: MTC camera params (t) are in cm, so convert gt_pose back to cm for projection
    cam_K = cam_params.get('K')
    cam_R = cam_params.get('R')
    cam_t = cam_params.get('t')
    if cam_K is not None and cam_R is not None and cam_t is not None:
        # gt_pose is in meters (after /100 conversion) with Y-up, Z-away convention
        # Convert back to MTC convention (Y-down, Z-toward) and cm for projection
        gt_pose_cm = gt_pose.copy()
        gt_pose_cm[:, 1] = -gt_pose_cm[:, 1]  # Flip Y back to MTC Y-down
        gt_pose_cm[:, 2] = -gt_pose_cm[:, 2]  # Flip Z back to MTC Z-toward
        gt_pose_cm = gt_pose_cm * 100.0  # Convert meters to cm
        projected_2d = project_3d_to_2d(gt_pose_cm, cam_R, cam_t, cam_K)
    else:
        projected_2d = mp_2d

    # Save training sample
    np.savez_compressed(
        sample_path,
        corrupted=mp_normalized.astype(np.float32),
        ground_truth=gt_normalized.astype(np.float32),
        visibility=mp_vis.astype(np.float32),
        pose_2d=mp_2d.astype(np.float32),
        projected_2d=projected_2d.astype(np.float32),
        azimuth=np.float32(azimuth),
        elevation=np.float32(elevation),
        camera_relative=cam_relative.astype(np.float32),
        mp_scale=np.float32(mp_scale),
        gt_scale=np.float32(gt_scale),
        sequence=sample_name,
        frame_idx=int(frame_str),
    )

    return 1


def process_mtc_pickle_format(
    mtc_dir: Path,
    output_dir: Path,
    frame_skip: int = 3,
    max_frames: int = None,
    cameras: list = None,
    workers: int = 1,
    split: str = 'training_data',
) -> int:
    """
    Process MTC dataset in pickle format (a4_release structure).

    Args:
        mtc_dir: Path to a4_release directory
        output_dir: Output directory for NPZ files
        frame_skip: Process every Nth frame
        max_frames: Max frames total
        cameras: List of camera IDs to use (0-30)
        workers: Number of parallel workers
        split: 'training_data' or 'testing_data'

    Returns:
        Number of samples created.
    """
    mtc_dir = Path(mtc_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load annotations and camera data
    annotation_file = mtc_dir / "annotation.pkl"
    camera_file = mtc_dir / "camera_data.pkl"
    img_dir = mtc_dir / "hdImgs"

    if not annotation_file.exists():
        print(f"ERROR: annotation.pkl not found in {mtc_dir}")
        return 0

    if not camera_file.exists():
        print(f"ERROR: camera_data.pkl not found in {mtc_dir}")
        return 0

    print(f"Loading annotations from {annotation_file}...")
    annotations = load_mtc_annotations(annotation_file)

    print(f"Loading camera data from {camera_file}...")
    camera_data = load_mtc_camera_data(camera_file)

    # Get data split
    data = annotations.get(split, [])
    print(f"Found {len(data)} frames in {split}")

    if not data:
        print(f"No data found in {split}")
        return 0

    # Default cameras: use ALL 31 views (0-30) for maximum viewpoint diversity
    # This is the main value of the MTC dataset - comprehensive multi-view coverage
    if cameras is None:
        cameras = list(range(31))

    # Build task list
    tasks = []
    frame_count = 0

    for i, sample in enumerate(data):
        # Apply frame skip
        if i % frame_skip != 0:
            continue

        if max_frames and frame_count >= max_frames:
            break

        seq_name = sample['seqName']
        frame_str = sample['frame_str']
        body = sample.get('body', {})
        landmarks = body.get('landmarks', [])

        if not landmarks:
            continue

        # Parse landmarks to (19, 3) array
        gt_joints19, _ = parse_mtc_landmarks(landmarks)
        if gt_joints19 is None:
            continue

        # MTC landmarks are in cm, convert to meters
        gt_joints19 = gt_joints19 / 100.0

        # Get 2D visibility info
        visibility_2d = body.get('2D', {})

        for cam_id in cameras:
            # Check if this joint is visible in this camera
            cam_vis = visibility_2d.get(cam_id, {})
            inside_img = cam_vis.get('insideImg', [])

            # Skip if too many joints are out of view
            if inside_img and sum(inside_img) < 10:
                continue

            # Build image path
            img_path = img_dir / seq_name / frame_str / f"00_{cam_id:02d}_{frame_str}.jpg"

            # Get camera parameters
            cam_params = get_camera_params(camera_data, seq_name, cam_id)
            if cam_params is None:
                continue

            tasks.append((
                img_path,
                gt_joints19.copy(),
                cam_params,
                output_dir,
                seq_name,
                frame_str,
                cam_id,
            ))

        frame_count += 1

    print(f"Built {len(tasks)} tasks for {frame_count} frames x {len(cameras)} cameras")

    # Process tasks
    total_samples = 0
    if workers > 1:
        print(f"Processing with {workers} parallel workers...")
        with Pool(workers) as pool:
            for count in tqdm(pool.imap_unordered(process_mtc_worker, tasks), total=len(tasks)):
                total_samples += count
    else:
        print("Processing sequentially...")
        for task in tqdm(tasks):
            count = process_mtc_worker(task)
            total_samples += count

    return total_samples


# ============================================================================
# LEGACY VIDEO FORMAT PROCESSING
# ============================================================================

def find_sequences(mtc_dir: Path) -> list:
    """
    Find all sequences in MTC dataset.

    Returns list of dicts with sequence info.
    """
    mtc_dir = Path(mtc_dir)
    sequences = []

    # Look for sequence directories
    # MTC structure: mtc_dir/sequence_name/hdVideos/, hdPose3d_stage1_coco19/
    for seq_dir in sorted(mtc_dir.iterdir()):
        if not seq_dir.is_dir():
            continue

        video_dir = seq_dir / "hdVideos"
        pose_dir = seq_dir / "hdPose3d_stage1_coco19"
        calib_file = seq_dir / "calibration.json"

        # Also check alternative structures
        if not pose_dir.exists():
            pose_dir = seq_dir / "hdPose3d"
        if not pose_dir.exists():
            pose_dir = seq_dir / "body3DScene"

        if video_dir.exists() or pose_dir.exists():
            sequences.append({
                'name': seq_dir.name,
                'dir': seq_dir,
                'video_dir': video_dir if video_dir.exists() else None,
                'pose_dir': pose_dir if pose_dir.exists() else None,
                'calib_file': calib_file if calib_file.exists() else None,
            })

    return sequences


def process_sequence(
    seq_info: dict,
    output_dir: Path,
    frame_skip: int = 3,
    max_frames: int = None,
    cameras: list = None,
) -> int:
    """
    Process a single sequence.

    Returns number of samples created.
    """
    seq_name = seq_info['name']
    seq_dir = seq_info['dir']
    pose_dir = seq_info['pose_dir']
    video_dir = seq_info['video_dir']
    calib_file = seq_info['calib_file']

    if pose_dir is None:
        print(f"  Skipping {seq_name}: no pose data")
        return 0

    # Load calibration
    cam_params = {}
    if calib_file and calib_file.exists():
        cam_params = load_cmu_calibration(calib_file)

    # Find pose JSON files
    pose_files = sorted(pose_dir.glob("*.json"))
    if not pose_files:
        pose_files = sorted(pose_dir.glob("body3DScene_*.json"))

    if not pose_files:
        print(f"  Skipping {seq_name}: no pose files found")
        return 0

    # Find video files
    video_files = {}
    if video_dir and video_dir.exists():
        for vf in video_dir.glob("*.mp4"):
            # Extract camera ID from filename (e.g., "hd_00_00.mp4" -> "00_00")
            cam_id = vf.stem.replace("hd_", "")
            video_files[cam_id] = vf

    # Initialize MediaPipe
    pose_detector = init_mediapipe()

    # Select cameras to process
    if cameras is None:
        # Use up to 6 cameras for diversity
        cameras = list(video_files.keys())[:6]

    samples_created = 0
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for cam_id in cameras:
        if cam_id not in video_files:
            continue

        video_path = video_files[cam_id]
        cap = cv2.VideoCapture(str(video_path))

        if not cap.isOpened():
            continue

        # Get camera position
        cam_pos = None
        if cam_id in cam_params and cam_params[cam_id]['position'] is not None:
            cam_pos = cam_params[cam_id]['position']

        frame_idx = 0
        processed = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Skip frames for temporal diversity
            if frame_idx % frame_skip != 0:
                frame_idx += 1
                continue

            if max_frames and processed >= max_frames:
                break

            # Find corresponding pose file
            pose_file = None
            for pf in pose_files:
                # Match frame number in filename
                if f"_{frame_idx:08d}" in pf.name or f"_{frame_idx:06d}" in pf.name:
                    pose_file = pf
                    break

            if pose_file is None:
                frame_idx += 1
                continue

            # Load ground truth pose
            bodies = load_skeleton_json(pose_file)
            if not bodies:
                frame_idx += 1
                continue

            # Use first body (main subject)
            body = bodies[0]
            gt_joints19 = body['joints19']
            gt_conf19 = body['confidence']

            # Convert to COCO17
            gt_pose, gt_conf = convert_coco19_to_coco17(gt_joints19, gt_conf19)

            # Run MediaPipe
            mp_pose, mp_2d, mp_vis = process_frame_mediapipe(frame, pose_detector)

            if mp_pose is None:
                frame_idx += 1
                continue

            # Center both poses on pelvis
            gt_pelvis = (gt_pose[11] + gt_pose[12]) / 2
            mp_pelvis = (mp_pose[11] + mp_pose[12]) / 2

            gt_centered = gt_pose - gt_pelvis
            mp_centered = mp_pose - mp_pelvis

            # Align body frames to remove rotation differences
            # Without this, torso errors are ~85° due to rotation misalignment
            mp_centered = align_body_frames(mp_centered, gt_centered)

            # Compute scales
            gt_scale = compute_torso_scale(gt_centered)
            mp_scale = compute_torso_scale(mp_centered)

            if gt_scale < 0.01 or mp_scale < 0.01:
                frame_idx += 1
                continue

            # Normalize to unit torso
            gt_normalized = gt_centered / gt_scale
            mp_normalized = mp_centered / mp_scale

            # Nose validation: nose should be above pelvis (use normalized poses)
            # In normalized space (torso=1.0), nose should be ~1.0-1.5 above pelvis
            if mp_normalized[0, 1] < 0.5 or gt_normalized[0, 1] < 0.5:
                frame_idx += 1
                continue

            # Compute view angles
            if cam_pos is not None:
                # Adjust camera position relative to subject
                cam_relative = cam_pos - gt_pelvis
                azimuth, elevation = compute_view_angles(gt_centered, cam_pos)
            else:
                # Estimate from pose if no calibration
                cam_relative = np.array([0., 0., 2.])  # Assume frontal
                azimuth, elevation = 0.0, 0.0

            # Save training sample
            sample_name = f"{seq_name}_c{cam_id}_f{frame_idx:06d}"
            sample_path = output_dir / f"{sample_name}.npz"

            # Resume support: skip if file already exists
            if sample_path.exists():
                samples_created += 1
                processed += 1
                frame_idx += 1
                continue

            # Compute projected_2d (GT 3D -> 2D via camera params)
            # This is critical for POF (Part Orientation Fields)
            # NOTE: MTC camera params (t) are in cm, so use gt_pose_cm (before meter conversion)
            if cam_id in cam_params and cam_params[cam_id]['K'] is not None:
                cam_K = cam_params[cam_id]['K']
                cam_R = cam_params[cam_id]['R']
                cam_t = cam_params[cam_id]['t']
                if cam_R is not None and cam_t is not None:
                    # Use GT pose in original cm units for projection (matches camera t units)
                    projected_2d = project_3d_to_2d(gt_pose_cm, cam_R, cam_t, cam_K)
                else:
                    # Fall back to MediaPipe 2D if no extrinsics
                    projected_2d = mp_2d
            else:
                # Fall back to MediaPipe 2D if no calibration
                projected_2d = mp_2d

            np.savez_compressed(
                sample_path,
                corrupted=mp_normalized.astype(np.float32),
                ground_truth=gt_normalized.astype(np.float32),
                visibility=mp_vis.astype(np.float32),
                pose_2d=mp_2d.astype(np.float32),
                projected_2d=projected_2d.astype(np.float32),  # GT projected to 2D via camera
                azimuth=np.float32(azimuth),
                elevation=np.float32(elevation),
                camera_relative=cam_relative.astype(np.float32),
                mp_scale=np.float32(mp_scale),
                gt_scale=np.float32(gt_scale),
                sequence=sample_name,
                frame_idx=frame_idx,
            )

            samples_created += 1
            processed += 1
            frame_idx += 1

        cap.release()

    pose_detector.close()
    return samples_created


def detect_dataset_format(mtc_dir: Path) -> str:
    """
    Detect MTC dataset format.

    Returns:
        'pickle' if a4_release format (annotation.pkl, camera_data.pkl, hdImgs/)
        'legacy' if legacy format (hdVideos/, hdPose3d/)
        'unknown' otherwise
    """
    mtc_dir = Path(mtc_dir)

    # Check for pickle format (a4_release)
    # Could be directly in mtc_dir or in a subdirectory like a4_release
    for check_dir in [mtc_dir, mtc_dir / "a4_release"]:
        if (check_dir / "annotation.pkl").exists() and (check_dir / "camera_data.pkl").exists():
            return 'pickle', check_dir

    # Check for legacy format (sequence directories with hdVideos/)
    for item in mtc_dir.iterdir():
        if item.is_dir():
            if (item / "hdVideos").exists() or (item / "hdPose3d_stage1_coco19").exists():
                return 'legacy', mtc_dir

    return 'unknown', mtc_dir


def explore_dataset(mtc_dir: Path):
    """
    Explore and report on the dataset structure.

    Helps understand unfamiliar dataset layouts before conversion.
    """
    mtc_dir = Path(mtc_dir)

    print(f"\n{'='*60}")
    print(f"EXPLORING DATASET: {mtc_dir}")
    print(f"{'='*60}\n")

    if not mtc_dir.exists():
        print(f"ERROR: Directory does not exist: {mtc_dir}")
        return

    # Detect format
    fmt, data_dir = detect_dataset_format(mtc_dir)
    print(f"DETECTED FORMAT: {fmt}")
    if data_dir != mtc_dir:
        print(f"DATA DIRECTORY: {data_dir}")
    print("-" * 40)

    if fmt == 'pickle':
        # Explore pickle format (a4_release)
        annotation_file = data_dir / "annotation.pkl"
        camera_file = data_dir / "camera_data.pkl"
        img_dir = data_dir / "hdImgs"

        print(f"\n  annotation.pkl: {annotation_file.stat().st_size / (1024*1024):.1f} MB")
        print(f"  camera_data.pkl: {camera_file.stat().st_size / (1024*1024):.1f} MB")

        # Load and explore annotations
        try:
            annotations = load_mtc_annotations(annotation_file)
            train_data = annotations.get('training_data', [])
            test_data = annotations.get('testing_data', [])
            print(f"\n  Training samples: {len(train_data)}")
            print(f"  Testing samples: {len(test_data)}")

            if train_data:
                sample = train_data[0]
                print(f"\n  Sample keys: {list(sample.keys())}")
                print(f"  Sample seqName: {sample.get('seqName')}")
                print(f"  Sample frame_str: {sample.get('frame_str')}")
                body = sample.get('body', {})
                if body:
                    print(f"  Body keys: {list(body.keys())}")
                    landmarks = body.get('landmarks', [])
                    print(f"  Landmarks count: {len(landmarks)} ({len(landmarks)//3} joints)")

            # Count unique sequences
            seq_names = set(s.get('seqName') for s in train_data + test_data)
            print(f"\n  Unique sequences: {len(seq_names)}")
            for seq in sorted(seq_names)[:5]:
                print(f"    - {seq}")
            if len(seq_names) > 5:
                print(f"    ... and {len(seq_names) - 5} more")

        except Exception as e:
            print(f"  Error loading annotations: {e}")

        # Load and explore camera data
        try:
            camera_data = load_mtc_camera_data(camera_file)
            print(f"\n  Camera data sequences: {len(camera_data)}")
            for seq_name in list(camera_data.keys())[:2]:
                cams = camera_data[seq_name]
                print(f"    {seq_name}: {len(cams)} cameras")
                if cams:
                    cam_id = list(cams.keys())[0]
                    cam = cams[cam_id]
                    print(f"      Camera {cam_id} keys: {list(cam.keys())}")
        except Exception as e:
            print(f"  Error loading camera data: {e}")

        # Check image directory
        if img_dir.exists():
            seq_dirs = [d for d in img_dir.iterdir() if d.is_dir()]
            print(f"\n  Image sequences: {len(seq_dirs)}")
            if seq_dirs:
                first_seq = seq_dirs[0]
                frame_dirs = [d for d in first_seq.iterdir() if d.is_dir()]
                print(f"    {first_seq.name}: {len(frame_dirs)} frames")
                if frame_dirs:
                    first_frame = frame_dirs[0]
                    imgs = list(first_frame.glob("*.jpg"))
                    print(f"      {first_frame.name}: {len(imgs)} images (cameras)")

        print(f"\n{'='*60}")
        print("RECOMMENDATION:")
        print("-" * 40)
        print("  Dataset is in pickle format (a4_release structure).")
        print("  Ready to convert! Use:")
        print(f"    python {__file__} --mtc-dir {data_dir} --max-frames 1000 --workers 4")
        print(f"{'='*60}\n")
        return

    # Legacy exploration for other formats
    print("TOP-LEVEL CONTENTS:")
    print("-" * 40)
    dirs = []
    files = []
    for item in sorted(mtc_dir.iterdir()):
        if item.is_dir():
            dirs.append(item.name)
        else:
            size_mb = item.stat().st_size / (1024 * 1024)
            files.append(f"{item.name} ({size_mb:.1f} MB)")

    for d in dirs[:20]:
        print(f"  [DIR]  {d}")
    if len(dirs) > 20:
        print(f"  ... and {len(dirs) - 20} more directories")

    for f in files[:10]:
        print(f"  [FILE] {f}")
    if len(files) > 10:
        print(f"  ... and {len(files) - 10} more files")

    # Explore first sequence directory
    if dirs:
        print(f"\n\nFIRST SEQUENCE STRUCTURE: {dirs[0]}")
        print("-" * 40)
        first_seq = mtc_dir / dirs[0]
        _explore_dir_recursive(first_seq, depth=0, max_depth=3)

    # Look for common patterns
    print(f"\n\nDETECTED PATTERNS:")
    print("-" * 40)

    # Check for JSON files (poses)
    json_files = list(mtc_dir.rglob("*.json"))[:5]
    if json_files:
        print(f"  Found {len(list(mtc_dir.rglob('*.json')))} JSON files")
        print(f"  Example: {json_files[0].relative_to(mtc_dir)}")

        # Try to read first JSON
        try:
            with open(json_files[0], 'r') as f:
                data = json.load(f)
            print(f"  JSON keys: {list(data.keys())[:10]}")
            if 'bodies' in data:
                print(f"  Contains 'bodies' key - likely pose data!")
                if data['bodies']:
                    body = data['bodies'][0]
                    print(f"  Body keys: {list(body.keys())}")
        except Exception as e:
            print(f"  Could not read JSON: {e}")

    # Check for video files
    video_files = list(mtc_dir.rglob("*.mp4"))[:5]
    if video_files:
        print(f"\n  Found {len(list(mtc_dir.rglob('*.mp4')))} MP4 files")
        print(f"  Example: {video_files[0].relative_to(mtc_dir)}")

    # Check for image files
    img_files = list(mtc_dir.rglob("*.jpg"))[:5] + list(mtc_dir.rglob("*.png"))[:5]
    if img_files:
        total_imgs = len(list(mtc_dir.rglob("*.jpg"))) + len(list(mtc_dir.rglob("*.png")))
        print(f"\n  Found {total_imgs} image files (jpg/png)")
        print(f"  Example: {img_files[0].relative_to(mtc_dir)}")

    # Check for calibration
    calib_files = list(mtc_dir.rglob("*calib*.json"))
    if calib_files:
        print(f"\n  Found {len(calib_files)} calibration files")
        print(f"  Example: {calib_files[0].relative_to(mtc_dir)}")

    print(f"\n{'='*60}")
    print("RECOMMENDATION:")
    print("-" * 40)
    if json_files and (video_files or img_files):
        print("  Dataset appears to have both poses and videos/images.")
        print("  Ready to convert! Use:")
        print(f"    python {__file__} --mtc-dir {mtc_dir} --max-sequences 3")
    else:
        print("  Dataset structure unclear. Please check manually.")
    print(f"{'='*60}\n")


def _explore_dir_recursive(path: Path, depth: int = 0, max_depth: int = 3):
    """Helper to print directory tree."""
    if depth > max_depth:
        return

    indent = "  " * depth
    items = sorted(path.iterdir())

    dirs = [i for i in items if i.is_dir()]
    files = [i for i in items if i.is_file()]

    # Show directories
    for d in dirs[:5]:
        print(f"{indent}[DIR] {d.name}/")
        _explore_dir_recursive(d, depth + 1, max_depth)
    if len(dirs) > 5:
        print(f"{indent}... and {len(dirs) - 5} more directories")

    # Show files (sample)
    for f in files[:5]:
        size = f.stat().st_size
        if size > 1024 * 1024:
            size_str = f"{size / (1024*1024):.1f} MB"
        elif size > 1024:
            size_str = f"{size / 1024:.1f} KB"
        else:
            size_str = f"{size} B"
        print(f"{indent}[FILE] {f.name} ({size_str})")
    if len(files) > 5:
        print(f"{indent}... and {len(files) - 5} more files")


def main():
    storage_paths = StoragePaths.load()
    parser = argparse.ArgumentParser(description="Convert CMU MTC dataset to training data")
    parser.add_argument("--mtc-dir", type=str, default=str(storage_paths.training_root / "mtc"),
                        help="Path to MTC dataset directory")
    parser.add_argument("--output-dir", type=str, default=str(storage_paths.training_root / "mtc_converted"),
                        help="Output directory for training samples")
    parser.add_argument("--download", action="store_true",
                        help="Download dataset if not present")
    parser.add_argument("--explore", action="store_true",
                        help="Explore dataset structure without converting")
    parser.add_argument("--max-sequences", type=int, default=None,
                        help="Limit number of sequences to process (legacy format only)")
    parser.add_argument("--max-frames", type=int, default=None,
                        help="Max frames to process total")
    parser.add_argument("--frame-skip", type=int, default=3,
                        help="Process every Nth frame (default: 3)")
    parser.add_argument("--workers", type=int, default=1,
                        help="Number of parallel workers")
    parser.add_argument("--split", type=str, default="training_data",
                        choices=["training_data", "testing_data"],
                        help="Data split to process (pickle format only)")

    args = parser.parse_args()

    mtc_dir = Path(args.mtc_dir)
    output_dir = Path(args.output_dir)

    # Download if requested
    if args.download:
        download_mtc_dataset(mtc_dir)

    # Explore mode
    if args.explore:
        explore_dataset(mtc_dir)
        return

    if not mtc_dir.exists():
        print(f"Error: MTC directory not found: {mtc_dir}")
        print("Use --download to download the dataset, or specify --mtc-dir")
        print("Or use --explore to check the structure first")
        sys.exit(1)

    # Detect dataset format
    fmt, data_dir = detect_dataset_format(mtc_dir)

    if fmt == 'pickle':
        print(f"Detected pickle format (a4_release) at {data_dir}")
        print(f"Processing {args.split}...")
        total_samples = process_mtc_pickle_format(
            data_dir,
            output_dir,
            frame_skip=args.frame_skip,
            max_frames=args.max_frames,
            workers=args.workers,
            split=args.split,
        )
        print(f"\nConversion complete!")
        print(f"Total samples: {total_samples}")
        print(f"Output directory: {output_dir}")
        return

    elif fmt == 'legacy':
        # Find sequences (legacy format)
        sequences = find_sequences(mtc_dir)
        print(f"Found {len(sequences)} sequences")

        if len(sequences) == 0:
            print("\nNo sequences found! Try --explore to check dataset structure.")
            print("The script expects: mtc_dir/sequence_name/hdVideos/ and hdPose3d*/")
            sys.exit(1)

        if args.max_sequences:
            sequences = sequences[:args.max_sequences]
            print(f"Processing first {len(sequences)} sequences")

        # Process sequences
        output_dir.mkdir(parents=True, exist_ok=True)
        total_samples = 0

        for seq_info in tqdm(sequences, desc="Processing sequences"):
            n_samples = process_sequence(
                seq_info,
                output_dir,
                frame_skip=args.frame_skip,
                max_frames=args.max_frames,
            )
            total_samples += n_samples

            if n_samples > 0:
                tqdm.write(f"  {seq_info['name']}: {n_samples} samples")

        print(f"\nConversion complete!")
        print(f"Total samples: {total_samples}")
        print(f"Output directory: {output_dir}")

    else:
        print(f"Error: Unknown dataset format in {mtc_dir}")
        print("Use --explore to check the dataset structure.")
        sys.exit(1)
    print(f"Total samples: {total_samples}")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()

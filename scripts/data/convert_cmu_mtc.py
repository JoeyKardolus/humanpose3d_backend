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
    python scripts/convert_cmu_mtc_to_training.py --explore --mtc-dir data/mtc

    # Download and convert (full dataset ~270GB)
    python scripts/convert_cmu_mtc_to_training.py --download --mtc-dir data/mtc

    # Convert already downloaded dataset
    python scripts/convert_cmu_mtc_to_training.py --mtc-dir /path/to/mtc_dataset

    # Quick test with limited sequences
    python scripts/convert_cmu_mtc_to_training.py --mtc-dir data/mtc --max-sequences 5

Download manually (recommended for 270GB):
    wget -c http://domedb.perception.cs.cmu.edu/data/mtc/mtc_dataset.tar.gz
    tar -xzf mtc_dataset.tar.gz -C data/mtc
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.depth_refinement.data_utils import align_body_frames

import argparse
import json
import subprocess
import tarfile
import numpy as np
from tqdm import tqdm
import cv2
import mediapipe as mp
from scipy.spatial.transform import Rotation
from concurrent.futures import ProcessPoolExecutor, as_completed
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

def load_cmu_calibration(calib_file: Path) -> dict:
    """
    Load CMU Panoptic camera calibration.

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
            # Different conventions - try both
            try:
                # Convention 1: t is translation from world to camera
                position = -R.T @ t
            except:
                position = t
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
    Load 3D skeleton data from CMU Panoptic JSON format.

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


# ============================================================================
# MAIN CONVERSION
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

            np.savez_compressed(
                sample_path,
                corrupted=mp_normalized.astype(np.float32),
                ground_truth=gt_normalized.astype(np.float32),
                visibility=mp_vis.astype(np.float32),
                pose_2d=mp_2d.astype(np.float32),
                azimuth=np.float32(azimuth),
                elevation=np.float32(elevation),
                camera_relative=cam_relative.astype(np.float32),
                mp_scale=np.float32(mp_scale),
                gt_scale=np.float32(gt_scale),
            )

            samples_created += 1
            processed += 1
            frame_idx += 1

        cap.release()

    pose_detector.close()
    return samples_created


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

    # List top-level contents
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
    parser = argparse.ArgumentParser(description="Convert CMU MTC dataset to training data")
    parser.add_argument("--mtc-dir", type=str, default="data/mtc",
                        help="Path to MTC dataset directory")
    parser.add_argument("--output-dir", type=str, default="data/training/mtc_converted",
                        help="Output directory for training samples")
    parser.add_argument("--download", action="store_true",
                        help="Download dataset if not present")
    parser.add_argument("--explore", action="store_true",
                        help="Explore dataset structure without converting")
    parser.add_argument("--max-sequences", type=int, default=None,
                        help="Limit number of sequences to process")
    parser.add_argument("--max-frames", type=int, default=200,
                        help="Max frames per camera per sequence")
    parser.add_argument("--frame-skip", type=int, default=3,
                        help="Process every Nth frame (default: 3)")
    parser.add_argument("--workers", type=int, default=1,
                        help="Number of parallel workers")

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

    # Find sequences
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


if __name__ == "__main__":
    main()

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

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pickle
import numpy as np
from tqdm import tqdm
import cv2
import mediapipe as mp


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


def compute_view_angle_from_torso(pose_3d: np.ndarray) -> float:
    """
    Compute camera viewing angle from torso plane orientation.

    The torso plane is defined by shoulders and hips.
    View angle = angle between torso normal and camera Z-axis.

    Args:
        pose_3d: (17, 3) COCO keypoints

    Returns:
        View angle in degrees (0=frontal, 90=profile)
    """
    # COCO indices: left_shoulder=5, right_shoulder=6, left_hip=11, right_hip=12
    left_shoulder = pose_3d[5]
    right_shoulder = pose_3d[6]
    left_hip = pose_3d[11]
    right_hip = pose_3d[12]

    # Vectors defining torso plane
    shoulder_vec = right_shoulder - left_shoulder
    hip_vec = right_hip - left_hip

    # Normal to torso plane (points forward from body)
    torso_normal = np.cross(shoulder_vec, hip_vec)
    norm = np.linalg.norm(torso_normal)
    if norm < 1e-6:
        return 45.0  # Default if degenerate

    torso_normal = torso_normal / norm

    # Camera looks down +Z axis (in our coordinate system)
    camera_dir = np.array([0, 0, 1])
    cos_angle = np.dot(torso_normal, camera_dir)

    # Return absolute angle (0° = frontal, 90° = profile)
    angle = np.degrees(np.arccos(np.clip(np.abs(cos_angle), -1, 1)))
    return angle


def extract_mediapipe_coco(results) -> tuple:
    """
    Extract 17 COCO joints and visibility from MediaPipe results.

    Args:
        results: MediaPipe pose results

    Returns:
        (joints, visibility) where:
        - joints: (17, 3) array of joint positions
        - visibility: (17,) array of visibility scores
    """
    joints = np.zeros((17, 3))
    visibility = np.zeros(17)

    landmarks = results.pose_world_landmarks.landmark

    for coco_idx, mp_idx in MEDIAPIPE_TO_COCO.items():
        lm = landmarks[mp_idx]
        joints[coco_idx] = [lm.x, lm.y, lm.z]
        visibility[coco_idx] = lm.visibility

    return joints, visibility


def center_on_pelvis(joints: np.ndarray) -> np.ndarray:
    """Center skeleton on pelvis (midpoint of hips)."""
    # COCO: left_hip=11, right_hip=12
    pelvis = (joints[11] + joints[12]) / 2
    return joints - pelvis


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


def process_sequence(
    video_path: Path,
    keypoints3d: np.ndarray,
    output_dir: Path,
    sequence_name: str,
    pose,
    max_frames: int = 500,
    frame_skip: int = 2,
) -> int:
    """
    Process single AIST++ sequence.

    Args:
        video_path: Path to video file
        keypoints3d: (n_frames, 17, 3) ground truth keypoints
        output_dir: Output directory
        sequence_name: Name for output files
        pose: MediaPipe Pose object
        max_frames: Max frames to process
        frame_skip: Process every N frames (60fps -> 30fps effective)

    Returns:
        Number of training examples generated
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  Failed to open video: {video_path}")
        return 0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    gt_frames = len(keypoints3d)

    # Use minimum of video frames and GT frames
    n_frames = min(total_frames, gt_frames, max_frames * frame_skip)

    examples_generated = 0
    pbar = tqdm(total=n_frames // frame_skip, desc=f"  {sequence_name[:25]}", leave=False)

    frame_idx = 0
    while frame_idx < n_frames:
        ret, frame = cap.read()
        if not ret:
            break

        # Skip frames (60fps -> effective 30fps)
        if frame_idx % frame_skip != 0:
            frame_idx += 1
            continue

        # Get ground truth for this frame
        gt_pose = keypoints3d[frame_idx]  # (17, 3)

        # Check for invalid GT
        if np.isnan(gt_pose).any() or np.allclose(gt_pose, 0):
            frame_idx += 1
            continue

        # Run MediaPipe on frame
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_frame)

        if not results.pose_world_landmarks:
            frame_idx += 1
            continue

        # Extract MediaPipe pose and visibility
        mp_pose, visibility = extract_mediapipe_coco(results)

        # Center both on pelvis
        mp_centered = center_on_pelvis(mp_pose)
        gt_centered = center_on_pelvis(gt_pose)

        # Fix MediaPipe coordinate system to match AIST++
        # MediaPipe: Y down, Z toward camera
        # AIST++: Y up, Z away from camera (standard)
        mp_centered[:, 1] = -mp_centered[:, 1]  # Y: down -> up
        mp_centered[:, 2] = -mp_centered[:, 2]  # Z: toward camera -> away

        # Compute view angle from GT torso
        view_angle = compute_view_angle_from_torso(gt_centered)

        # Basic validation: nose should be above pelvis
        if mp_centered[0, 1] < 0.1 or gt_centered[0, 1] < 0.1:
            frame_idx += 1
            continue

        # Save training pair
        output_path = output_dir / f"{sequence_name}_f{frame_idx:06d}.npz"

        np.savez_compressed(
            output_path,
            corrupted=mp_centered.astype(np.float32),      # (17, 3) MediaPipe (REAL errors!)
            ground_truth=gt_centered.astype(np.float32),   # (17, 3) AIST++ GT
            visibility=visibility.astype(np.float32),      # (17,) MediaPipe visibility
            view_angle=np.float32(view_angle),             # Scalar: 0-90 degrees
            sequence=sequence_name,
            frame_idx=frame_idx,
        )

        examples_generated += 1
        pbar.update(1)
        frame_idx += 1

    pbar.close()
    cap.release()

    return examples_generated


def main():
    """Convert AIST++ dataset to training pairs."""

    print("=" * 80)
    print("AIST++ -> DEPTH REFINEMENT TRAINING DATA")
    print("=" * 80)
    print()
    print("Key: Using REAL MediaPipe errors from video frames!")
    print()

    # Paths
    aistpp_dir = Path("data/AIST++")
    annotations_dir = aistpp_dir / "annotations"
    videos_dir = aistpp_dir / "videos"
    output_dir = Path("data/training/aistpp_converted")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check directories
    keypoints_dir = annotations_dir / "keypoints3d"
    if not keypoints_dir.exists():
        print(f"ERROR: keypoints3d not found at {keypoints_dir}")
        return

    if not videos_dir.exists():
        print(f"ERROR: videos not found at {videos_dir}")
        print("Run: python data/AIST++/api/downloader.py --download_folder data/AIST++/videos --accept_terms")
        return

    # Find sequences
    kp_files = sorted(keypoints_dir.glob("*.pkl"))
    print(f"Found {len(kp_files)} keypoint sequences")

    # Check for videos
    video_files = list(videos_dir.glob("*.mp4"))
    print(f"Found {len(video_files)} videos")

    if len(video_files) == 0:
        print("No videos found. Download in progress or not started.")
        print("Run: python data/AIST++/api/downloader.py --download_folder data/AIST++/videos --accept_terms --num_processes 8")
        return

    video_dict = {v.stem: v for v in video_files}
    print()

    # Initialize MediaPipe
    print("Initializing MediaPipe Pose...")
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        static_image_mode=True,  # Per-frame for consistency
        model_complexity=2,
        enable_segmentation=False,
        min_detection_confidence=0.5,
    )
    print("Ready")
    print()

    # Process sequences
    total_examples = 0
    processed = 0
    max_sequences = 100  # Start with subset

    for kp_file in kp_files[:max_sequences]:
        seq_name = kp_file.stem  # e.g., gBR_sBM_cAll_d04_mBR0_ch01

        # Find video for camera c01 (frontal)
        video_name = get_video_name(seq_name, 'c01')
        video_path = video_dict.get(video_name)

        if video_path is None:
            continue

        # Load ground truth
        try:
            gt_keypoints = load_keypoints3d(kp_file)
        except Exception as e:
            print(f"  Failed to load {kp_file.name}: {e}")
            continue

        print(f"Processing: {seq_name} ({len(gt_keypoints)} frames)")

        num_examples = process_sequence(
            video_path,
            gt_keypoints,
            output_dir,
            seq_name,
            pose,
            max_frames=200,
            frame_skip=2,
        )

        total_examples += num_examples
        processed += 1
        print(f"  -> {num_examples} training pairs")

    print()
    print("=" * 80)
    print(f"Processed {processed} sequences")
    print(f"Generated {total_examples} training examples")
    print(f"Saved to: {output_dir}")
    print()
    print("Training data includes:")
    print("  - corrupted: MediaPipe 3D pose (REAL depth errors)")
    print("  - ground_truth: AIST++ mocap (clean depth)")
    print("  - view_angle: Camera angle from torso (0-90 deg)")
    print("  - visibility: Per-joint visibility from MediaPipe")
    print("=" * 80)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Convert HumanEva dataset to training pairs for depth refinement.

HumanEva has:
- Real video footage (JPEG frames)
- 3D mocap ground truth (15 body joints)
- Multiple camera views

This script:
1. Loads HumanEva video frames
2. Runs MediaPipe on frames → Get REAL noisy 3D poses
3. Loads mocap ground truth
4. Creates training pairs: (MediaPipe output, Mocap ground truth)
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

import numpy as np
from tqdm import tqdm
import cv2

# Import our existing MediaPipe detector
from posedetector.pose_detector import PoseDetector


# HumanEva joint names (15 joints)
HUMANEVA_JOINTS = [
    'Hip', 'RHip', 'RKnee', 'RAnkle', 'LHip', 'LKnee', 'LAnkle',
    'Spine', 'Thorax', 'Neck', 'Head',
    'LShoulder', 'LElbow', 'LWrist', 'RShoulder', 'RElbow', 'RWrist'
]


def load_mocap_frame(mocap_file: Path, frame_idx: int) -> np.ndarray:
    """Load single frame from HumanEva mocap file.

    Args:
        mocap_file: Path to mocap.txt file
        frame_idx: Frame number to load

    Returns:
        joints: (15, 3) array of 3D joint positions
    """
    # HumanEva mocap format: space-separated text file
    # Each line: frame_num x1 y1 z1 x2 y2 z2 ... (15 joints * 3 coords = 45 values)

    with open(mocap_file, 'r') as f:
        lines = f.readlines()

    if frame_idx >= len(lines):
        return None

    # Parse line
    values = lines[frame_idx].strip().split()
    frame_num = int(values[0])
    coords = [float(v) for v in values[1:46]]  # 15 joints * 3 coords

    # Reshape to (15, 3)
    joints = np.array(coords).reshape(15, 3)

    return joints


def process_sequence(
    sequence_dir: Path,
    output_dir: Path,
    detector: PoseDetector,
    max_frames: int = 100
) -> int:
    """Process single HumanEva sequence.

    Args:
        sequence_dir: Path to sequence directory (e.g., S1/Walking_1)
        output_dir: Output directory for training pairs
        detector: MediaPipe pose detector
        max_frames: Maximum frames to process (for testing)

    Returns:
        Number of training examples generated
    """
    # Find frames and mocap file
    frame_files = sorted(sequence_dir.glob("frame_*.jpg"))
    mocap_file = sequence_dir / "mocap.txt"

    if not mocap_file.exists():
        print(f"  ⚠️  No mocap.txt found in {sequence_dir}")
        return 0

    if len(frame_files) == 0:
        print(f"  ⚠️  No frame files found in {sequence_dir}")
        return 0

    print(f"  Found {len(frame_files)} frames, processing first {max_frames}...")

    examples_generated = 0
    sequence_name = sequence_dir.name

    for frame_idx, frame_file in enumerate(tqdm(frame_files[:max_frames], desc=f"  {sequence_name}")):
        # Load video frame
        frame = cv2.imread(str(frame_file))
        if frame is None:
            continue

        # Run MediaPipe on frame
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = detector.process_frame(rgb_frame)

        if not results or not results.pose_world_landmarks:
            continue  # MediaPipe failed

        # Extract MediaPipe 3D landmarks (33 points)
        mediapipe_landmarks = []
        for landmark in results.pose_world_landmarks.landmark:
            mediapipe_landmarks.append([landmark.x, landmark.y, landmark.z])

        mediapipe_poses = np.array(mediapipe_landmarks)  # (33, 3)

        # Load mocap ground truth for this frame
        mocap_gt = load_mocap_frame(mocap_file, frame_idx)
        if mocap_gt is None:
            continue

        # Center both on pelvis (Hip is joint 0 in HumanEva)
        pelvis_mp = mediapipe_poses[0]  # MediaPipe's first landmark
        pelvis_gt = mocap_gt[0]  # HumanEva's Hip

        mediapipe_centered = mediapipe_poses - pelvis_mp
        mocap_centered = mocap_gt - pelvis_gt

        # Save training pair
        example_name = f"{sequence_name}_f{frame_idx:06d}"
        output_path = output_dir / f"{example_name}.npz"

        np.savez_compressed(
            output_path,
            corrupted=mediapipe_centered,  # MediaPipe output (noisy)
            ground_truth=mocap_centered,   # Mocap ground truth (clean)
            mp_joint_names=['MediaPipe_' + str(i) for i in range(33)],
            gt_joint_names=HUMANEVA_JOINTS,
            source="HumanEva-I",
        )

        examples_generated += 1

    return examples_generated


def main():
    """Convert HumanEva dataset to training pairs."""

    print("="*80)
    print("HUMANEVA → TRAINING DATA CONVERSION")
    print("="*80)
    print()

    # Paths
    humaneva_dir = Path("data/humaneva")
    output_dir = Path("data/training/humaneva_converted")
    output_dir.mkdir(parents=True, exist_ok=True)

    if not humaneva_dir.exists():
        print(f"ERROR: HumanEva data not found at {humaneva_dir}")
        print()
        print("Please:")
        print("1. Register at http://humaneva.is.tue.mpg.de/datasets_human_1")
        print("2. Download HumanEva-I dataset")
        print("3. Extract to data/humaneva/")
        return

    # Find all sequences
    sequences = []
    for subject_dir in sorted(humaneva_dir.glob("S*")):
        if subject_dir.is_dir():
            for seq_dir in sorted(subject_dir.glob("*_*")):
                if seq_dir.is_dir() and (seq_dir / "mocap.txt").exists():
                    sequences.append(seq_dir)

    if len(sequences) == 0:
        print(f"ERROR: No sequences found in {humaneva_dir}")
        print("Expected structure: humaneva/S1/Walking_1/")
        return

    print(f"Found {len(sequences)} sequences:")
    for seq in sequences[:5]:
        print(f"  - {seq.parent.name}/{seq.name}")
    if len(sequences) > 5:
        print(f"  ... and {len(sequences) - 5} more")
    print()

    # Initialize MediaPipe detector
    print("Initializing MediaPipe detector...")
    detector = PoseDetector(
        static_image_mode=False,
        model_complexity=2,
        enable_segmentation=False
    )
    print("✓ MediaPipe ready")
    print()

    # Process sequences (start with first 2 for testing)
    total_examples = 0
    for sequence_dir in sequences[:2]:
        print(f"Processing: {sequence_dir.parent.name}/{sequence_dir.name}")
        num_examples = process_sequence(
            sequence_dir,
            output_dir,
            detector,
            max_frames=50  # Process first 50 frames per sequence
        )
        total_examples += num_examples
        print(f"  Generated {num_examples} training examples")
        print()

    print("="*80)
    print(f"✓ Generated {total_examples} training examples")
    print(f"✓ Saved to: {output_dir}")
    print()
    print("These are REAL training pairs:")
    print("  - Corrupted: Actual MediaPipe detections from video")
    print("  - Ground truth: Mocap 3D positions")
    print()
    print("Next steps:")
    print("1. Validate training data quality")
    print("2. Train depth refinement model on REAL data!")
    print("="*80)


if __name__ == "__main__":
    main()

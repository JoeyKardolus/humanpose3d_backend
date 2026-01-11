#!/usr/bin/env python3
"""
Apply trained PoseFormer depth refinement to TRC files.

Usage:
    python scripts/apply_depth_refinement.py --input data/output/pose-3d/joey/joey_final.trc
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

import torch
import numpy as np
import argparse
from typing import Dict, List, Tuple
from tqdm import tqdm

from anatomical.depth_model import PoseFormerDepthRefiner
from visualizedata.visualize_data import VisualizeData


def estimate_camera_angle(markers: Dict[str, np.ndarray], frame_idx: int) -> float:
    """Estimate camera viewing angle from body orientation using torso plane.

    Args:
        markers: Dict of marker_name → (num_frames, 3) positions
        frame_idx: Frame index to analyze

    Returns:
        Camera angle in degrees (0° = frontal, 90° = profile)
    """
    # Get shoulder and hip markers for this frame
    try:
        l_shoulder = markers['LShoulder'][frame_idx]
        r_shoulder = markers['RShoulder'][frame_idx]
        l_hip = markers['LHip'][frame_idx]
        r_hip = markers['RHip'][frame_idx]
    except KeyError:
        # Fallback to default if markers missing
        return 45.0

    # Check for NaN values
    if np.any(np.isnan([l_shoulder, r_shoulder, l_hip, r_hip])):
        return 45.0

    # Calculate torso plane vectors
    shoulder_vector = r_shoulder - l_shoulder  # Left to right
    hip_vector = r_hip - l_hip  # Left to right

    # Torso normal (perpendicular to torso plane, points forward)
    torso_normal = np.cross(shoulder_vector, hip_vector)
    torso_normal_norm = np.linalg.norm(torso_normal)

    if torso_normal_norm < 1e-6:
        return 45.0  # Degenerate case

    torso_normal = torso_normal / torso_normal_norm

    # Camera looks down +Z axis into the scene
    # Angle = deviation of torso normal from camera direction
    camera_direction = np.array([0, 0, 1])

    # Dot product gives cos(angle)
    cos_angle = np.dot(torso_normal, camera_direction)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)  # Numerical stability

    angle_rad = np.arccos(np.abs(cos_angle))  # Absolute for symmetry
    angle_deg = np.degrees(angle_rad)

    return float(angle_deg)


def load_trc_as_dict(trc_path: Path) -> Tuple[Dict[str, np.ndarray], float, int]:
    """Load TRC file into dictionary format.

    Returns:
        markers: Dict of marker_name → (num_frames, 3) array
        fps: Frame rate
        num_frames: Number of frames
    """
    viz = VisualizeData()
    marker_names, frames = viz.load_trc_frames(trc_path)

    num_frames = len(frames)
    num_markers = len(marker_names)

    # Convert frames to dict
    markers = {}
    for marker_idx, marker_name in enumerate(marker_names):
        positions = np.zeros((num_frames, 3))
        for frame_idx, frame in enumerate(frames):
            positions[frame_idx] = frame[marker_idx]
        markers[marker_name] = positions

    # Extract FPS from TRC header
    with open(trc_path, 'r') as f:
        lines = f.readlines()
        # Line 3: DataRate CameraRate NumFrames NumMarkers Units OrigDataRate OrigDataStartFrame OrigNumFrames
        header_line = lines[2].strip().split('\t')
        fps = float(header_line[0])

    return markers, fps, num_frames


def prepare_features(
    markers: Dict[str, np.ndarray],
    marker_names: List[str],
    frame_idx: int,
    temporal_window: int = 11,
) -> Tuple[np.ndarray, float]:
    """Prepare model input features for a single center frame.

    Args:
        markers: Dict of marker_name → (num_frames, 3) positions
        marker_names: Ordered list of marker names
        frame_idx: Center frame index
        temporal_window: Number of frames (must match training: 11)

    Returns:
        features: (1, frames, markers, 8) tensor
        camera_angle: Estimated camera angle
    """
    num_frames = markers[marker_names[0]].shape[0]
    num_markers = len(marker_names)
    half_window = temporal_window // 2

    # Get temporal window indices (pad if at boundaries)
    start_idx = max(0, frame_idx - half_window)
    end_idx = min(num_frames, frame_idx + half_window + 1)

    # Build temporal sequence
    window_positions = np.zeros((temporal_window, num_markers, 3))

    for t in range(temporal_window):
        actual_idx = frame_idx - half_window + t
        actual_idx = np.clip(actual_idx, 0, num_frames - 1)  # Clamp to valid range

        for marker_idx, marker_name in enumerate(marker_names):
            window_positions[t, marker_idx] = markers[marker_name][actual_idx]

    # Estimate camera angle from center frame
    camera_angle = estimate_camera_angle(markers, frame_idx)

    # Build features: x, y, z, visibility, variance, is_augmented, marker_type, camera_angle
    features = np.zeros((temporal_window, num_markers, 8))
    features[:, :, :3] = window_positions  # x, y, z
    features[:, :, 3] = 0.8  # Default visibility
    features[:, :, 4] = 0.01  # Low variance
    features[:, :, 5] = 1.0  # Is augmented (TRC from Pose2Sim)
    features[:, :, 6] = 0.0  # Marker type (generic)
    features[:, :, 7] = camera_angle / 90.0  # Normalized camera angle (0-90° → 0-1)

    # Add batch dimension
    features = features[np.newaxis, :, :, :]  # (1, frames, markers, 8)

    return features, camera_angle


def apply_depth_refinement(
    trc_path: Path,
    model_path: Path,
    output_path: Path,
    device: str = "cuda",
) -> None:
    """Apply depth refinement to TRC file.

    Args:
        trc_path: Input TRC file
        model_path: Trained model checkpoint
        output_path: Output refined TRC file
        device: Device for inference
    """
    print(f"Loading TRC: {trc_path.name}")
    markers, fps, num_frames = load_trc_as_dict(trc_path)
    marker_names = sorted(markers.keys())
    num_markers = len(marker_names)

    print(f"  Frames: {num_frames}")
    print(f"  Markers: {num_markers}")
    print(f"  FPS: {fps}")
    print()

    # Load model
    print("Loading depth refinement model...")
    model = PoseFormerDepthRefiner(
        num_markers=num_markers,
        num_frames=11,
        feature_dim=8,
        d_model=512,
        num_heads=8,
        num_layers=6,
        d_ff=2048,
        dropout=0.1,
    ).to(device)

    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"✓ Loaded checkpoint from epoch {checkpoint['epoch']}")
    print(f"  Validation loss: {checkpoint['val_loss']:.6f}")
    print()

    # Apply refinement to each frame
    print("Refining depth...")
    refined_markers = {name: positions.copy() for name, positions in markers.items()}

    camera_angles = []

    with torch.no_grad():
        for frame_idx in tqdm(range(num_frames), desc="Processing frames"):
            # Prepare features
            features, camera_angle = prepare_features(
                markers, marker_names, frame_idx, temporal_window=11
            )
            camera_angles.append(camera_angle)

            # Run inference
            features_tensor = torch.from_numpy(features).float().to(device)
            delta_z, confidence = model(features_tensor)

            # Apply depth corrections to center frame
            delta_z_np = delta_z[0, 5].cpu().numpy()  # Center frame (index 5 of 11)

            for marker_idx, marker_name in enumerate(marker_names):
                refined_markers[marker_name][frame_idx, 2] += delta_z_np[marker_idx]

    print()
    print(f"Average camera angle: {np.mean(camera_angles):.1f}° (range: {np.min(camera_angles):.1f}° - {np.max(camera_angles):.1f}°)")
    print()

    # Write refined TRC
    print(f"Writing refined TRC: {output_path.name}")
    write_trc(output_path, refined_markers, marker_names, fps)
    print("✓ Done!")


def write_trc(
    output_path: Path,
    markers: Dict[str, np.ndarray],
    marker_names: List[str],
    fps: float,
) -> None:
    """Write markers to TRC file.

    Args:
        output_path: Output TRC path
        markers: Dict of marker_name → (num_frames, 3) positions
        marker_names: Ordered marker names
        fps: Frame rate
    """
    num_frames = markers[marker_names[0]].shape[0]
    num_markers = len(marker_names)

    with open(output_path, 'w') as f:
        # Header line 1
        f.write("PathFileType\t4\t(X/Y/Z)\t" + output_path.name + "\n")

        # Header line 2
        f.write(f"DataRate\tCameraRate\tNumFrames\tNumMarkers\tUnits\tOrigDataRate\tOrigDataStartFrame\tOrigNumFrames\n")

        # Header line 3
        f.write(f"{fps:.2f}\t{fps:.2f}\t{num_frames}\t{num_markers}\tm\t{fps:.2f}\t1\t{num_frames}\n")

        # Header line 4: Frame# Time <marker1> <marker2> ...
        f.write("Frame#\tTime\t")
        f.write("\t".join([f"{name}\t\t" for name in marker_names]))
        f.write("\n")

        # Header line 5: X Y Z for each marker
        f.write("\t\t")
        for _ in marker_names:
            f.write("X\tY\tZ\t")
        f.write("\n")

        # Header line 6: Empty line
        f.write("\n")

        # Data rows
        for frame_idx in range(num_frames):
            time = frame_idx / fps
            f.write(f"{frame_idx + 1}\t{time:.4f}\t")

            for marker_name in marker_names:
                pos = markers[marker_name][frame_idx]
                if np.isnan(pos).any():
                    f.write("\t\t\t")
                else:
                    f.write(f"{pos[0]:.6f}\t{pos[1]:.6f}\t{pos[2]:.6f}\t")

            f.write("\n")


def main():
    parser = argparse.ArgumentParser(description="Apply depth refinement to TRC files")
    parser.add_argument("--input", type=str, required=True,
                        help="Input TRC file")
    parser.add_argument("--model", type=str, default="models/checkpoints/best_model.pth",
                        help="Trained model checkpoint")
    parser.add_argument("--output", type=str, default=None,
                        help="Output TRC file (default: <input>_refined.trc)")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device: cuda or cpu")

    args = parser.parse_args()

    # Setup paths
    input_path = Path(args.input)
    model_path = Path(args.model)

    if not input_path.exists():
        print(f"ERROR: Input TRC not found: {input_path}")
        return

    if not model_path.exists():
        print(f"ERROR: Model checkpoint not found: {model_path}")
        print("Train the model first with: python scripts/train_depth_model.py")
        return

    if args.output:
        output_path = Path(args.output)
    else:
        output_path = input_path.parent / f"{input_path.stem}_refined.trc"

    # Check device
    if args.device == "cuda" and not torch.cuda.is_available():
        print("WARNING: CUDA not available, falling back to CPU")
        args.device = "cpu"

    print("=" * 80)
    print("DEPTH REFINEMENT - TRC FILE")
    print("=" * 80)
    print()

    apply_depth_refinement(input_path, model_path, output_path, args.device)

    print()
    print("=" * 80)
    print(f"✓ Refined TRC saved to: {output_path}")
    print()
    print("Compare before/after:")
    print(f"  Original: {input_path}")
    print(f"  Refined:  {output_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()

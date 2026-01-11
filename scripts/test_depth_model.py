#!/usr/bin/env python3
"""
Test and visualize depth refinement model.

Shows:
- Corrupted positions (with noise)
- Model-refined positions
- Ground truth positions
- Depth error statistics
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import argparse

from anatomical.depth_model import PoseFormerDepthRefiner


def load_model(checkpoint_path: Path, device: str):
    """Load trained model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model = PoseFormerDepthRefiner(
        num_markers=59,
        num_frames=11,
        feature_dim=8,  # Updated to match training
        d_model=512,
        num_heads=8,
        num_layers=6,
        d_ff=2048,
        dropout=0.1,
    ).to(device)

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"Loaded model from epoch {checkpoint['epoch']}")
    print(f"Validation loss: {checkpoint['val_loss']:.6f}")

    return model


def test_sample(model, sample_path: Path, device: str):
    """Test model on a single sample."""
    # Load data
    data = np.load(sample_path)

    corrupted = data["corrupted"]  # (markers, 3)
    ground_truth = data["ground_truth"]  # (markers, 3)
    marker_names = data["marker_names"].tolist()
    camera_angle = float(data["camera_angle"])
    noise_std = float(data["noise_std"])

    # Create temporal window by repeating frame (for inference)
    temporal_window = 11
    frames_corrupted = np.repeat(corrupted[np.newaxis, :, :], temporal_window, axis=0)

    # Build features (matching training)
    features = np.zeros((temporal_window, len(marker_names), 8))
    features[:, :, :3] = frames_corrupted  # x, y, z
    features[:, :, 3] = 0.8  # Visibility
    features[:, :, 4] = 0.01  # Variance
    features[:, :, 5] = 0.0  # Not augmented
    features[:, :, 6] = 0.0  # Marker type
    features[:, :, 7] = camera_angle / 90.0  # Normalized angle

    # Add batch dimension
    features_tensor = torch.from_numpy(features).float().unsqueeze(0).to(device)

    # Run model
    with torch.no_grad():
        delta_z, confidence = model(features_tensor)

    # Apply corrections (use middle frame)
    mid_frame = temporal_window // 2
    delta_z_np = delta_z[0, mid_frame, :].cpu().numpy()
    confidence_np = confidence[0, mid_frame, :].cpu().numpy()

    refined = corrupted.copy()
    refined[:, 2] += delta_z_np  # Apply depth correction

    return corrupted, refined, ground_truth, marker_names, delta_z_np, confidence_np, camera_angle, noise_std


def visualize_results(corrupted, refined, ground_truth, marker_names, delta_z, confidence, camera_angle, noise_std, sample_name):
    """Create comprehensive visualization."""

    # Create figure with 4 subplots
    fig = plt.figure(figsize=(20, 10))

    # 1. 3D view: All three versions
    ax1 = fig.add_subplot(221, projection='3d')
    ax1.scatter(corrupted[:, 0], corrupted[:, 1], corrupted[:, 2],
                c='red', marker='o', s=30, alpha=0.4, label='Corrupted (input)')
    ax1.scatter(refined[:, 0], refined[:, 1], refined[:, 2],
                c='green', marker='^', s=40, alpha=0.6, label='Refined (model)')
    ax1.scatter(ground_truth[:, 0], ground_truth[:, 1], ground_truth[:, 2],
                c='blue', marker='s', s=30, alpha=0.4, label='Ground Truth')
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_zlabel('Z (m)')
    ax1.set_title(f'3D Positions (camera={camera_angle}°, noise={noise_std}m)')
    ax1.legend()

    # Set equal aspect ratio
    all_points = np.vstack([corrupted, refined, ground_truth])
    max_range = np.array([
        all_points[:, 0].max() - all_points[:, 0].min(),
        all_points[:, 1].max() - all_points[:, 1].min(),
        all_points[:, 2].max() - all_points[:, 2].min()
    ]).max() / 2.0

    mid_x = (all_points[:, 0].max() + all_points[:, 0].min()) * 0.5
    mid_y = (all_points[:, 1].max() + all_points[:, 1].min()) * 0.5
    mid_z = (all_points[:, 2].max() + all_points[:, 2].min()) * 0.5

    ax1.set_xlim(mid_x - max_range, mid_x + max_range)
    ax1.set_ylim(mid_y - max_range, mid_y + max_range)
    ax1.set_zlim(mid_z - max_range, mid_z + max_range)

    # 2. Depth comparison (Z-axis only)
    ax2 = fig.add_subplot(222)
    marker_indices = np.arange(len(marker_names))
    ax2.scatter(marker_indices, corrupted[:, 2], c='red', alpha=0.6, s=30, label='Corrupted')
    ax2.scatter(marker_indices, refined[:, 2], c='green', alpha=0.6, s=40, marker='^', label='Refined')
    ax2.scatter(marker_indices, ground_truth[:, 2], c='blue', alpha=0.6, s=30, marker='s', label='Ground Truth')
    ax2.set_xlabel('Marker Index')
    ax2.set_ylabel('Z Depth (m)')
    ax2.set_title('Depth (Z-axis) Comparison')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. Depth corrections applied
    ax3 = fig.add_subplot(223)
    ax3.bar(marker_indices, delta_z * 1000, color='purple', alpha=0.6)  # Convert to mm
    ax3.set_xlabel('Marker Index')
    ax3.set_ylabel('Depth Correction (mm)')
    ax3.set_title('Model Depth Corrections (Δz)')
    ax3.grid(True, alpha=0.3, axis='y')
    ax3.axhline(y=0, color='k', linestyle='-', linewidth=0.5)

    # 4. Error statistics
    ax4 = fig.add_subplot(224)

    # Compute errors
    corrupted_error = np.linalg.norm(corrupted - ground_truth, axis=1) * 1000  # mm
    refined_error = np.linalg.norm(refined - ground_truth, axis=1) * 1000  # mm

    # Compute depth-only errors
    corrupted_depth_error = np.abs(corrupted[:, 2] - ground_truth[:, 2]) * 1000  # mm
    refined_depth_error = np.abs(refined[:, 2] - ground_truth[:, 2]) * 1000  # mm

    bar_width = 0.35
    x = np.arange(2)

    means_3d = [corrupted_error.mean(), refined_error.mean()]
    means_depth = [corrupted_depth_error.mean(), refined_depth_error.mean()]

    ax4.bar(x - bar_width/2, means_3d, bar_width, label='3D Error', alpha=0.7, color='orange')
    ax4.bar(x + bar_width/2, means_depth, bar_width, label='Depth Error (Z only)', alpha=0.7, color='purple')

    ax4.set_ylabel('Mean Error (mm)')
    ax4.set_title('Error Comparison')
    ax4.set_xticks(x)
    ax4.set_xticklabels(['Corrupted', 'Refined'])
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')

    # Add text with improvement
    improvement_3d = (1 - refined_error.mean() / corrupted_error.mean()) * 100
    improvement_depth = (1 - refined_depth_error.mean() / corrupted_depth_error.mean()) * 100

    textstr = f'3D Error: {corrupted_error.mean():.1f}mm → {refined_error.mean():.1f}mm ({improvement_3d:+.1f}%)\n'
    textstr += f'Depth Error: {corrupted_depth_error.mean():.1f}mm → {refined_depth_error.mean():.1f}mm ({improvement_depth:+.1f}%)'

    ax4.text(0.5, 0.95, textstr, transform=ax4.transAxes,
             fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()

    # Print detailed statistics
    print(f"\n{'='*80}")
    print(f"Sample: {sample_name}")
    print(f"Camera angle: {camera_angle}°, Noise std: {noise_std}m")
    print(f"\n3D Position Error (mm):")
    print(f"  Corrupted: mean={corrupted_error.mean():.2f}, std={corrupted_error.std():.2f}, max={corrupted_error.max():.2f}")
    print(f"  Refined:   mean={refined_error.mean():.2f}, std={refined_error.std():.2f}, max={refined_error.max():.2f}")
    print(f"  Improvement: {improvement_3d:+.1f}%")
    print(f"\nDepth Error (mm):")
    print(f"  Corrupted: mean={corrupted_depth_error.mean():.2f}, std={corrupted_depth_error.std():.2f}, max={corrupted_depth_error.max():.2f}")
    print(f"  Refined:   mean={refined_depth_error.mean():.2f}, std={refined_depth_error.std():.2f}, max={refined_depth_error.max():.2f}")
    print(f"  Improvement: {improvement_depth:+.1f}%")
    print(f"\nModel Corrections:")
    print(f"  Mean Δz: {delta_z.mean()*1000:.2f}mm")
    print(f"  Std Δz:  {delta_z.std()*1000:.2f}mm")
    print(f"  Max |Δz|: {np.abs(delta_z).max()*1000:.2f}mm")
    print(f"  Mean confidence: {confidence.mean():.3f}")
    print(f"{'='*80}\n")

    return fig


def main():
    parser = argparse.ArgumentParser(description="Test depth refinement model")
    parser.add_argument("--checkpoint", type=str, default="models/checkpoints/best_model.pth",
                        help="Path to model checkpoint")
    parser.add_argument("--data-dir", type=str, default="data/training/cmu_converted",
                        help="Path to training data")
    parser.add_argument("--num-samples", type=int, default=5,
                        help="Number of samples to test")
    parser.add_argument("--output-dir", type=str, default=".",
                        help="Output directory for visualizations")

    args = parser.parse_args()

    # Setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}\n")

    checkpoint_path = Path(args.checkpoint)
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    model = load_model(checkpoint_path, device)

    # Get test samples
    samples = sorted(data_dir.glob("*.npz"))[:args.num_samples]

    if not samples:
        print(f"No samples found in {data_dir}")
        return

    print(f"\nTesting on {len(samples)} samples...\n")

    # Test each sample
    for i, sample_path in enumerate(samples):
        print(f"Testing {i+1}/{len(samples)}: {sample_path.name}")

        corrupted, refined, ground_truth, marker_names, delta_z, confidence, camera_angle, noise_std = \
            test_sample(model, sample_path, device)

        fig = visualize_results(
            corrupted, refined, ground_truth, marker_names,
            delta_z, confidence, camera_angle, noise_std,
            sample_path.stem
        )

        output_path = output_dir / f"depth_test_{i+1}_{sample_path.stem}.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_path}")
        plt.close()

    print("\n✓ Testing complete!")


if __name__ == "__main__":
    main()

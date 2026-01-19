from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.application.build_log import append_build_log
from src.datastream.data_stream import (
    ORDER_22,
    csv_to_trc_strict,
    write_landmark_csv,
)
from src.datastream.marker_estimation import estimate_missing_markers
from src.datastream.post_augmentation_estimation import (
    estimate_shoulder_clusters_and_hjc,
)
from src.kinematics.comprehensive_joint_angles import compute_all_joint_angles
from src.kinematics.visualize_comprehensive_angles import (
    plot_comprehensive_joint_angles,
    plot_side_by_side_comparison,
    save_comprehensive_angles_csv,
)
from src.markeraugmentation.markeraugmentation import run_pose2sim_augment
from src.markeraugmentation.gpu_config import patch_pose2sim_gpu
from src.mediastream.media_stream import probe_video_rotation, read_video_rgb
from src.posedetector.pose_detector import extract_world_landmarks
from src.visualizedata.visualize_data import VisualizeData
from src.pipeline.refinement import (
    apply_neural_depth_refinement,
    apply_neural_joint_refinement,
)
from src.pipeline.cleanup import cleanup_output_directory
from src.postprocessing.temporal_smoothing import smooth_trc, hide_markers_in_trc

OUTPUT_ROOT = Path("data/output")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="3D human pose estimation pipeline using MediaPipe + Pose2Sim."
    )
    # Required
    parser.add_argument("--video", required=True, help="Input video file")

    # Subject parameters
    parser.add_argument("--height", type=float, default=1.78, help="Subject height in meters")
    parser.add_argument("--mass", type=float, default=75.0, help="Subject mass in kg")

    # Detection settings
    parser.add_argument(
        "--visibility-min",
        type=float,
        default=0.3,
        help="Minimum landmark visibility threshold (default 0.3, use 0.1 for better coverage)",
    )

    # Pipeline options
    parser.add_argument(
        "--estimate-missing",
        action="store_true",
        help="Estimate missing markers using symmetry mirroring",
    )
    parser.add_argument(
        "--force-complete",
        action="store_true",
        help="Estimate shoulder clusters and hip joint centers after augmentation",
    )
    parser.add_argument(
        "--augmentation-cycles",
        type=int,
        default=20,
        help="Number of Pose2Sim augmentation cycles to average (default 20)",
    )

    # Neural refinement (recommended)
    parser.add_argument(
        "--main-refiner",
        action="store_true",
        help="Apply MainRefiner neural pipeline (depth + joint refinement) - RECOMMENDED",
    )
    parser.add_argument(
        "--depth-model-path",
        type=str,
        default="models/checkpoints/best_depth_model.pth",
        help="Path to depth refinement model checkpoint",
    )
    parser.add_argument(
        "--joint-model-path",
        type=str,
        default="models/checkpoints/best_joint_model.pth",
        help="Path to joint refinement model checkpoint",
    )
    parser.add_argument(
        "--main-refiner-path",
        type=str,
        default="models/checkpoints/best_main_refiner.pth",
        help="Path to MainRefiner model checkpoint",
    )

    # Joint angle computation
    parser.add_argument(
        "--joint-angle-smooth-window",
        type=int,
        default=9,
        help="Smoothing window for joint angle computation (default 9, 0=disabled)",
    )
    parser.add_argument(
        "--compute-all-joint-angles",
        action="store_true",
        help="Compute all joint angles (auto-enabled with --main-refiner)",
    )
    parser.add_argument(
        "--plot-all-joint-angles",
        action="store_true",
        help="Generate comprehensive joint angle visualization",
    )
    parser.add_argument(
        "--save-angle-comparison",
        action="store_true",
        help="Save side-by-side right vs left comparison plot",
    )

    # Post-processing
    parser.add_argument(
        "--temporal-smoothing",
        type=int,
        default=0,
        metavar="WINDOW",
        help="Apply temporal smoothing with given window size (0=disabled, recommended: 3 or 5)",
    )
    parser.add_argument(
        "--show-all-markers",
        action="store_true",
        help="Show all markers including low-visibility ones (by default, markers with avg visibility <0.5 are hidden)",
    )

    # Visualization
    parser.add_argument(
        "--show-video",
        action="store_true",
        help="Show annotated preview while processing",
    )
    parser.add_argument(
        "--export-preview",
        action="store_true",
        help="Save annotated preview video without display window",
    )
    parser.add_argument(
        "--plot-landmarks",
        action="store_true",
        help="Display 3D landmark viewer",
    )
    parser.add_argument(
        "--plot-augmented",
        action="store_true",
        help="Visualize augmented TRC output",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    video_path = Path(args.video)
    if not video_path.exists():
        print(f"[main] video not found: {video_path}", file=sys.stderr)
        sys.exit(1)

    run_dir = OUTPUT_ROOT / video_path.stem
    run_dir.mkdir(parents=True, exist_ok=True)

    try:
        visualizer: VisualizeData | None = (
            VisualizeData() if args.plot_landmarks or args.plot_augmented else None
        )

        # Step 1: Extract landmarks from video
        frames, fps = read_video_rgb(video_path)
        preview_rotation = probe_video_rotation(video_path)
        preview_path = None
        if args.show_video or args.export_preview:
            preview_path = run_dir / f"{video_path.stem}_preview.mp4"

        detection_output = extract_world_landmarks(
            frames,
            fps,
            Path("models/pose_landmarker_heavy.task"),
            args.visibility_min,
            display=args.show_video,
            return_raw_landmarks=args.plot_landmarks,
            return_2d_landmarks=args.main_refiner,  # Need 2D coords for depth refinement
            preview_output=preview_path,
            preview_rotation_degrees=preview_rotation,
        )

        # Unpack detection output based on flags
        landmarks_2d = {}
        raw_landmarks = []
        if args.plot_landmarks and args.main_refiner:
            records, raw_landmarks, landmarks_2d = detection_output
        elif args.plot_landmarks:
            records, raw_landmarks = detection_output
        elif args.main_refiner:
            records, landmarks_2d = detection_output
        else:
            records = detection_output

        # Estimate missing markers using symmetry
        if args.estimate_missing:
            original_count = len(records)
            records = estimate_missing_markers(records)
            estimated_count = len(records) - original_count
            print(f"[main] estimated {estimated_count} missing markers using symmetry")

        # Step 1.5: Apply neural depth refinement (pre-augmentation, COCO-17)
        if args.main_refiner:
            records = apply_neural_depth_refinement(records, args.depth_model_path, landmarks_2d)
            append_build_log("main step1.5 depth refinement applied")

        # Step 1: Write CSV
        csv_path = run_dir / f"{video_path.stem}.csv"
        row_count = write_landmark_csv(csv_path, records)
        append_build_log(f"main step1 CSV {csv_path} ({len(frames)} frames, {row_count} rows)")
        print(f"[main] step1 CSV -> {csv_path}")

        if args.plot_landmarks and raw_landmarks:
            (visualizer or VisualizeData()).plot_landmarks(raw_landmarks)
        if preview_path and preview_path.exists():
            print(f"[main] preview video saved to {preview_path}")

        # Step 2: Convert to TRC
        trc_path = run_dir / f"{video_path.stem}.trc"
        frames_written, markers_written = csv_to_trc_strict(csv_path, trc_path, ORDER_22)
        append_build_log(f"main step2 TRC {trc_path} ({frames_written} frames, {markers_written} markers)")
        print(f"[main] step2 TRC -> {trc_path}")

        # Step 3: Pose2Sim augmentation
        patch_pose2sim_gpu()
        lstm_path = run_pose2sim_augment(
            trc_path,
            run_dir,
            height=args.height,
            mass=args.mass,
            augmentation_cycles=args.augmentation_cycles,
        )
        append_build_log(f"main step3 augment {lstm_path}")
        print(f"[main] step3 augment -> {lstm_path}")

        # Step 3.5: Force complete markers
        final_output = lstm_path
        if args.force_complete:
            final_output = estimate_shoulder_clusters_and_hjc(lstm_path)
            append_build_log(f"main step3.5 complete {final_output}")
            print(f"[main] step3.5 force-complete -> {final_output}")

        # Step 4: Compute joint angles (auto-enabled with --main-refiner)
        if args.compute_all_joint_angles or args.main_refiner:
            try:
                print("\n" + "=" * 60)
                print("Computing Comprehensive Joint Angles (ISB Standards)")
                print("=" * 60)

                angle_results = compute_all_joint_angles(
                    final_output,
                    smooth_window=args.joint_angle_smooth_window,
                    unwrap=True,
                    zero_mode="first_n_seconds",
                    zero_window_s=0.5,
                    verbose=True,
                )

                # Apply neural joint constraint refinement
                if args.main_refiner:
                    angle_results = apply_neural_joint_refinement(
                        angle_results,
                        args.joint_model_path,
                    )

                # Save angle CSVs
                save_comprehensive_angles_csv(
                    angle_results,
                    output_dir=run_dir,
                    basename=video_path.stem,
                )

                # Create visualizations
                if args.plot_all_joint_angles:
                    plot_path = run_dir / f"{video_path.stem}_all_joint_angles.png"
                    plot_comprehensive_joint_angles(
                        angle_results,
                        output_path=plot_path,
                        title_prefix=video_path.stem,
                        dpi=150,
                    )

                if args.save_angle_comparison:
                    comparison_path = run_dir / f"{video_path.stem}_joint_angles_comparison.png"
                    plot_side_by_side_comparison(
                        angle_results,
                        output_path=comparison_path,
                        title_prefix=video_path.stem,
                        dpi=150,
                    )

                print("=" * 60)
                print("Joint Angle Computation Complete")
                print("=" * 60)

            except Exception as e:
                print(f"[main] WARNING: Joint angle computation failed - {e}", file=sys.stderr)
                import traceback
                traceback.print_exc()

        # Visualize augmented output
        if args.plot_augmented:
            aug_preview = run_dir / f"{final_output.stem}_preview.mp4"
            (visualizer or VisualizeData()).plot_trc_file(
                final_output, export_path=aug_preview, block=True
            )

        # Cleanup and organize
        cleanup_output_directory(run_dir, video_path.stem)

        # Update final_output path if renamed
        final_trc = run_dir / f"{video_path.stem}_final.trc"
        if final_trc.exists():
            final_output = final_trc

        # Apply temporal smoothing if requested
        if args.temporal_smoothing > 0:
            final_output = smooth_trc(final_output, final_output, window=args.temporal_smoothing)

        # Hide low-visibility markers by default (unless --show-all-markers)
        if not args.show_all_markers:
            # Marker hierarchy: parent -> children (hiding parent hides all descendants)
            MARKER_CHILDREN = {
                'RShoulder': ['RElbow'],
                'RElbow': ['RWrist'],
                'LShoulder': ['LElbow'],
                'LElbow': ['LWrist'],
                'RKnee': ['RAnkle'],
                'RAnkle': ['RHeel', 'RBigToe', 'RSmallToe'],
                'LKnee': ['LAnkle'],
                'LAnkle': ['LHeel', 'LBigToe', 'LSmallToe'],
            }

            def get_all_descendants(marker):
                """Get marker and all its descendants."""
                result = [marker]
                if marker in MARKER_CHILDREN:
                    for child in MARKER_CHILDREN[marker]:
                        result.extend(get_all_descendants(child))
                return result

            # Compute average visibility per marker from detection records
            from collections import defaultdict
            vis_sums = defaultdict(float)
            vis_counts = defaultdict(int)
            for rec in records:
                vis_sums[rec.landmark] += rec.visibility
                vis_counts[rec.landmark] += 1

            # Find markers with avg visibility < 0.5
            low_vis_markers = set()
            for marker, total in vis_sums.items():
                avg_vis = total / vis_counts[marker] if vis_counts[marker] > 0 else 0
                if avg_vis < 0.5:
                    # Add this marker and all its descendants
                    descendants = get_all_descendants(marker)
                    for m in descendants:
                        if m not in low_vis_markers:
                            low_vis_markers.add(m)
                            desc_vis = vis_sums.get(m, 0) / vis_counts.get(m, 1) if vis_counts.get(m, 0) > 0 else 0
                            print(f"[main] Hiding marker: {m} (avg_vis={desc_vis:.2f})")

            if low_vis_markers:
                hide_markers_in_trc(final_output, list(low_vis_markers))

        print(f"[main] finished pipeline. Output: {final_output}")

    except Exception as exc:
        print(f"[main] error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

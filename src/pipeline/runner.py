"""Pipeline runner shared by CLI and management commands."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.application.build_log import append_build_log
from src.datastream.data_stream import ORDER_22, csv_to_trc_strict, write_landmark_csv
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
from src.markeraugmentation.gpu_config import patch_pose2sim_gpu
from src.markeraugmentation.markeraugmentation import run_pose2sim_augment
from src.mediastream.media_stream import probe_video_rotation, read_video_rgb
from src.posedetector.pose_detector import extract_world_landmarks
from src.postprocessing.temporal_smoothing import hide_markers_in_trc, smooth_trc
from src.pipeline.cleanup import cleanup_output_directory
from src.pipeline.refinement import (
    apply_neural_depth_refinement,
    apply_neural_joint_refinement,
)
from src.visualizedata.visualize_data import VisualizeData

OUTPUT_ROOT = Path("data/output")


def add_pipeline_arguments(parser: argparse.ArgumentParser) -> None:
    """Register pipeline CLI arguments on the provided parser."""
    parser.add_argument("--video", required=True, help="Input video file")

    parser.add_argument("--height", type=float, default=1.78, help="Subject height in meters")
    parser.add_argument("--weight", type=float, default=75.0, help="Subject weight in kg")

    parser.add_argument(
        "--visibility-min",
        type=float,
        default=0.3,
        help="Minimum landmark visibility threshold (default 0.3, use 0.1 for better coverage)",
    )

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


def build_parser() -> argparse.ArgumentParser:
    """Create the pipeline argument parser."""
    parser = argparse.ArgumentParser(
        description="3D human pose estimation pipeline using MediaPipe + Pose2Sim."
    )
    add_pipeline_arguments(parser)
    return parser


def run_pipeline(args: argparse.Namespace) -> None:
    """Run the full pipeline with the provided CLI arguments."""
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
            return_2d_landmarks=args.main_refiner,
            preview_output=preview_path,
            preview_rotation_degrees=preview_rotation,
        )

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

        if args.estimate_missing:
            original_count = len(records)
            records = estimate_missing_markers(records)
            estimated_count = len(records) - original_count
            print(f"[main] estimated {estimated_count} missing markers using symmetry")

        if args.main_refiner:
            records = apply_neural_depth_refinement(records, args.depth_model_path, landmarks_2d)
            append_build_log("main step1.5 depth refinement applied")

        csv_path = run_dir / f"{video_path.stem}.csv"
        row_count = write_landmark_csv(csv_path, records)
        append_build_log(f"main step1 CSV {csv_path} ({len(frames)} frames, {row_count} rows)")
        print(f"[main] step1 CSV -> {csv_path}")

        if args.plot_landmarks and raw_landmarks:
            (visualizer or VisualizeData()).plot_landmarks(raw_landmarks)
        if preview_path and preview_path.exists():
            print(f"[main] preview video saved to {preview_path}")

        trc_path = run_dir / f"{video_path.stem}.trc"
        frames_written, markers_written = csv_to_trc_strict(csv_path, trc_path, ORDER_22)
        append_build_log(f"main step2 TRC {trc_path} ({frames_written} frames, {markers_written} markers)")
        print(f"[main] step2 TRC -> {trc_path}")

        patch_pose2sim_gpu()
        lstm_path = run_pose2sim_augment(
            trc_path,
            run_dir,
            height=args.height,
            weight=args.weight,
            augmentation_cycles=args.augmentation_cycles,
        )
        append_build_log(f"main step3 augment {lstm_path}")
        print(f"[main] step3 augment -> {lstm_path}")

        final_output = lstm_path
        if args.force_complete:
            final_output = estimate_shoulder_clusters_and_hjc(lstm_path)
            append_build_log(f"main step3.5 complete {final_output}")
            print(f"[main] step3.5 force-complete -> {final_output}")

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

                if args.main_refiner:
                    angle_results = apply_neural_joint_refinement(
                        angle_results,
                        args.joint_model_path,
                    )

                save_comprehensive_angles_csv(
                    angle_results,
                    output_dir=run_dir,
                    basename=video_path.stem,
                )

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

            except Exception as exc:
                print(f"[main] WARNING: Joint angle computation failed - {exc}", file=sys.stderr)
                import traceback

                traceback.print_exc()

        if args.plot_augmented:
            aug_preview = run_dir / f"{final_output.stem}_preview.mp4"
            (visualizer or VisualizeData()).plot_trc_file(
                final_output, export_path=aug_preview, block=True
            )

        cleanup_output_directory(run_dir, video_path.stem)

        final_trc = run_dir / f"{video_path.stem}_final.trc"
        if final_trc.exists():
            final_output = final_trc

        if args.temporal_smoothing > 0:
            final_output = smooth_trc(final_output, final_output, window=args.temporal_smoothing)

        if not args.show_all_markers:
            marker_children = {
                "RShoulder": ["RElbow"],
                "RElbow": ["RWrist"],
                "LShoulder": ["LElbow"],
                "LElbow": ["LWrist"],
                "RKnee": ["RAnkle"],
                "RAnkle": ["RHeel", "RBigToe", "RSmallToe"],
                "LKnee": ["LAnkle"],
                "LAnkle": ["LHeel", "LBigToe", "LSmallToe"],
            }

            def get_all_descendants(marker: str) -> list[str]:
                result = [marker]
                if marker in marker_children:
                    for child in marker_children[marker]:
                        result.extend(get_all_descendants(child))
                return result

            from collections import defaultdict

            vis_sums = defaultdict(float)
            vis_counts = defaultdict(int)
            for rec in records:
                vis_sums[rec.landmark] += rec.visibility
                vis_counts[rec.landmark] += 1

            low_vis_markers = set()
            for marker, total in vis_sums.items():
                avg_vis = total / vis_counts[marker] if vis_counts[marker] > 0 else 0
                if avg_vis < 0.5:
                    for marker_name in get_all_descendants(marker):
                        if marker_name not in low_vis_markers:
                            low_vis_markers.add(marker_name)
                            desc_vis = (
                                vis_sums.get(marker_name, 0)
                                / vis_counts.get(marker_name, 1)
                                if vis_counts.get(marker_name, 0) > 0
                                else 0
                            )
                            print(
                                f"[main] Hiding marker: {marker_name} (avg_vis={desc_vis:.2f})"
                            )

            if low_vis_markers:
                hide_markers_in_trc(final_output, list(low_vis_markers))

        print(f"[main] finished pipeline. Output: {final_output}")

    except Exception as exc:
        print(f"[main] error: {exc}", file=sys.stderr)
        sys.exit(1)

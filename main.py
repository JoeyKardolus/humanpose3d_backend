from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.anatomical.anatomical_constraints import apply_anatomical_constraints
from src.application.build_log import append_build_log
from src.datastream.data_stream import (
    ORDER_22,
    csv_to_trc_strict,
    header_fix_strict,
    write_landmark_csv,
)
from src.datastream.marker_estimation import estimate_missing_markers
from src.datastream.post_augmentation_estimation import estimate_shoulder_clusters_and_hjc
from src.filtering.flk_filter import apply_flk_filter, apply_gaussian_smoothing
from src.markeraugmentation.markeraugmentation import run_pose2sim_augment
from src.mediastream.media_stream import read_video_rgb
from src.posedetector.pose_detector import extract_world_landmarks
from src.visualizedata.visualize_data import VisualizeData

OUTPUT_ROOT = Path("data/output/pose-3d")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Strict-mode augmented markers pipeline orchestrator."
    )
    parser.add_argument("--video", required=True, help="Input video file")
    parser.add_argument("--height", type=float, default=1.78, help="Subject height (m)")
    parser.add_argument("--mass", type=float, default=75.0, help="Subject mass (kg)")
    parser.add_argument("--age", type=int, default=30, help="Subject age (years)")
    parser.add_argument(
        "--sex",
        choices=["male", "female"],
        default="male",
        help="Subject sex for augmentations",
    )
    parser.add_argument(
        "--visibility-min",
        type=float,
        default=0.3,
        help="Minimum visibility required to export a landmark (default 0.3 for better coverage)",
    )
    parser.add_argument(
        "--fix-header",
        action="store_true",
        help="Run header-fix on the augmented TRC output",
    )
    parser.add_argument(
        "--show-video",
        action="store_true",
        help="Render an annotated preview video while extracting landmarks",
    )
    parser.add_argument(
        "--plot-landmarks",
        dest="plot_landmarks",
        action="store_true",
        help="Display the 3D Matplotlib viewer for extracted CSV landmarks",
    )
    parser.add_argument(
        "--no-plot",
        dest="plot_landmarks",
        action="store_false",
        help="Skip the 3D landmark visualization even if --plot-landmarks was set",
    )
    parser.set_defaults(plot_landmarks=False)
    parser.add_argument(
        "--plot-augmented",
        action="store_true",
        help="Visualize the augmented TRC output and export an animation",
    )
    parser.add_argument(
        "--estimate-missing",
        action="store_true",
        help="Estimate missing markers using symmetry to improve augmentation (experimental)",
    )
    parser.add_argument(
        "--force-complete",
        action="store_true",
        help="Estimate all missing augmented markers to reach 100%% (experimental)",
    )
    parser.add_argument(
        "--augmentation-cycles",
        type=int,
        default=20,
        help="Number of augmentation cycles to run and average (default 20 for best results)",
    )
    parser.add_argument(
        "--flk-filter",
        action="store_true",
        help="Apply FLK spatio-temporal filtering to smooth pose landmarks (experimental)",
    )
    parser.add_argument(
        "--flk-model",
        type=str,
        default=None,
        help="Path to FLK GRU model file (default: auto-detect from models/GRU.h5)",
    )
    parser.add_argument(
        "--flk-enable-rnn",
        action="store_true",
        help="Enable FLK RNN component for better motion prediction (slower)",
    )
    parser.add_argument(
        "--flk-passes",
        type=int,
        default=1,
        help="Number of FLK filtering passes (default 1, higher = smoother but slower)",
    )
    parser.add_argument(
        "--gaussian-smooth",
        type=float,
        default=0.0,
        help="Apply Gaussian smoothing with given sigma (0 = disabled, 2.0 recommended)",
    )
    parser.add_argument(
        "--anatomical-constraints",
        action="store_true",
        help="Apply anatomical constraints (bone lengths, pelvis smoothing, ground plane)",
    )
    parser.add_argument(
        "--bone-smooth-window",
        type=int,
        default=21,
        help="Smoothing window for pelvis Z-depth (frames, must be odd, default 21)",
    )
    parser.add_argument(
        "--ground-percentile",
        type=float,
        default=5.0,
        help="Percentile for ground plane estimation (0-100, default 5.0)",
    )
    parser.add_argument(
        "--ground-margin",
        type=float,
        default=0.02,
        help="Tolerance margin for foot contact in meters (default 0.02)",
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
        frames, fps = read_video_rgb(video_path)
        preview_path = None
        if args.show_video:
            preview_path = run_dir / f"{video_path.stem}_preview.mp4"

        detection_output = extract_world_landmarks(
            frames,
            fps,
            Path("models/pose_landmarker_heavy.task"),
            args.visibility_min,
            display=args.show_video,
            return_raw_landmarks=args.plot_landmarks,
            preview_output=preview_path,
        )
        if args.plot_landmarks:
            records, raw_landmarks = detection_output
        else:
            records = detection_output
            raw_landmarks = []

        # Apply Gaussian smoothing if requested (before FLK)
        if args.gaussian_smooth > 0:
            records = apply_gaussian_smoothing(records, sigma=args.gaussian_smooth)
            print(f"[main] applied Gaussian smoothing (sigma={args.gaussian_smooth}) to {len(records)} records")

        # Apply FLK filtering if requested
        if args.flk_filter:
            try:
                model_path = Path(args.flk_model) if args.flk_model else None
                records = apply_flk_filter(
                    records,
                    model_path=model_path,
                    enable_rnn=args.flk_enable_rnn,
                    latency=0,
                    num_passes=args.flk_passes,
                )
                print(f"[main] applied FLK filtering ({args.flk_passes} passes) to {len(records)} landmark records")
            except ImportError as e:
                print(f"[main] WARNING: Skipping FLK filter - {e}", file=sys.stderr)
            except Exception as e:
                print(
                    f"[main] WARNING: FLK filtering failed - {e}", file=sys.stderr
                )

        # Estimate missing markers if requested
        if args.estimate_missing:
            original_count = len(records)
            records = estimate_missing_markers(records)
            estimated_count = len(records) - original_count
            print(f"[main] estimated {estimated_count} missing markers using symmetry")

        # Apply anatomical constraints if requested
        if args.anatomical_constraints:
            records = apply_anatomical_constraints(
                records,
                smooth_window=args.bone_smooth_window,
                ground_percentile=args.ground_percentile,
                ground_margin=args.ground_margin,
            )
            print(f"[main] applied anatomical constraints to {len(records)} landmark records")

        csv_path = run_dir / f"{video_path.stem}.csv"
        row_count = write_landmark_csv(csv_path, records)
        append_build_log(
            f"main step1 CSV {csv_path} ({len(frames)} frames, {row_count} rows)"
        )
        print(f"[main] step1 CSV -> {csv_path}")

        if args.plot_landmarks and raw_landmarks:
            (visualizer or VisualizeData()).plot_landmarks(raw_landmarks)
        if preview_path and preview_path.exists():
            print(f"[main] preview video saved to {preview_path}")

        trc_path = run_dir / f"{video_path.stem}.trc"
        frames_written, markers_written = csv_to_trc_strict(
            csv_path, trc_path, ORDER_22
        )
        append_build_log(
            f"main step2 TRC {trc_path} ({frames_written} frames, {markers_written} markers)"
        )
        print(f"[main] step2 TRC -> {trc_path}")

        lstm_path = run_pose2sim_augment(
            trc_path,
            run_dir,
            height=args.height,
            mass=args.mass,
            age=args.age,
            sex=args.sex,
            augmentation_cycles=args.augmentation_cycles,
        )
        append_build_log(f"main step3 augment {lstm_path}")
        print(f"[main] step3 augment -> {lstm_path}")

        final_output = lstm_path
        if args.force_complete:
            final_output = estimate_shoulder_clusters_and_hjc(lstm_path)
            append_build_log(f"main step3.5 complete {final_output}")
            print(f"[main] step3.5 force-complete -> {final_output}")
        if args.fix_header:
            final_output = header_fix_strict(lstm_path)
            append_build_log(f"main step4 header-fix {final_output}")
            print(f"[main] step4 header-fix -> {final_output}")

        if args.plot_augmented:
            aug_preview = run_dir / f"{final_output.stem}_preview.mp4"
            (visualizer or VisualizeData()).plot_trc_file(
                final_output, export_path=aug_preview, block=True
            )

        print(f"[main] finished pipeline. Output: {final_output}")
    except Exception as exc:
        print(f"[main] error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

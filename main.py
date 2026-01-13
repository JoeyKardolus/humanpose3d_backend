from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.anatomical.anatomical_constraints import apply_anatomical_constraints
from src.anatomical.bone_length_constraints import (
    apply_bone_length_constraints,
    report_bone_length_statistics,
)
from src.anatomical.ground_plane_refinement import (
    apply_enhanced_ground_plane_refinement,
)
from src.anatomical.multi_constraint_optimization import (
    multi_constraint_optimization,
)
from src.application.build_log import append_build_log
from src.datastream.data_stream import (
    ORDER_22,
    csv_to_trc_strict,
    header_fix_strict,
    write_landmark_csv,
)
from src.datastream.marker_estimation import estimate_missing_markers
from src.datastream.post_augmentation_estimation import estimate_shoulder_clusters_and_hjc
from src.kinematics.joint_angles_euler import compute_lower_limb_angles
from src.kinematics.joint_angles_upper_body import compute_upper_body_angles
from src.kinematics.visualize_angles import plot_joint_angles_time_series, plot_upper_body_angles
from src.kinematics.comprehensive_joint_angles import compute_all_joint_angles
from src.kinematics.visualize_comprehensive_angles import (
    plot_comprehensive_joint_angles,
    plot_side_by_side_comparison,
    save_comprehensive_angles_csv,
)
from src.anatomical.joint_constraints import (
    check_angle_violations,
    print_violation_summary,
    soft_clamp_angles,
)
from src.markeraugmentation.markeraugmentation import run_pose2sim_augment
from src.markeraugmentation.gpu_config import patch_pose2sim_gpu, get_gpu_info
from src.depth_refinement.inference import DepthRefiner
from src.mediastream.media_stream import read_video_rgb
from src.posedetector.pose_detector import extract_world_landmarks
from src.visualizedata.visualize_data import VisualizeData

OUTPUT_ROOT = Path("data/output/pose-3d")


def cleanup_output_directory(run_dir: Path, video_stem: str) -> None:
    """Organize output directory by removing intermediate files and organizing results.

    Cleanup actions:
    - Remove intermediate augmentation cycle files
    - Remove temporary Pose2Sim project directories
    - Remove Config files
    - Organize joint angle files into joint_angles/ subdirectory
    - Rename main files for clarity
    """
    import shutil

    # Remove intermediate augmentation files
    for f in run_dir.glob(f"{video_stem}_LSTM_cycle*.trc"):
        if "_complete_multi_refined" not in f.name and "_multi_refined" not in f.name:
            f.unlink(missing_ok=True)

    # Remove temporary Pose2Sim projects
    for d in run_dir.glob("pose2sim_project_cycle*"):
        if d.is_dir():
            shutil.rmtree(d, ignore_errors=True)

    # Remove Config files
    for f in run_dir.glob("Config_cycle*.toml"):
        f.unlink(missing_ok=True)

    # Remove Zone.Identifier files (WSL metadata)
    for f in run_dir.glob("*.Zone.Identifier"):
        f.unlink(missing_ok=True)

    # Organize joint angle files
    angle_dir = run_dir / "joint_angles"
    angle_files = list(run_dir.glob(f"{video_stem}_angles_*.csv")) + \
                  list(run_dir.glob(f"{video_stem}_all_joint_angles.png")) + \
                  list(run_dir.glob(f"{video_stem}_joint_angles_comparison.png"))

    if angle_files:
        angle_dir.mkdir(exist_ok=True)
        for f in angle_files:
            dest = angle_dir / f.name
            # Remove destination if it exists to avoid duplicates
            dest.unlink(missing_ok=True)
            f.rename(dest)

    # Rename main files for clarity (only if not already renamed)
    csv_file = run_dir / f"{video_stem}.csv"
    if csv_file.exists():
        (run_dir / f"{video_stem}_raw_landmarks.csv").unlink(missing_ok=True)
        csv_file.rename(run_dir / f"{video_stem}_raw_landmarks.csv")

    trc_file = run_dir / f"{video_stem}.trc"
    if trc_file.exists():
        (run_dir / f"{video_stem}_initial.trc").unlink(missing_ok=True)
        trc_file.rename(run_dir / f"{video_stem}_initial.trc")

    # Rename final output (multi_refined)
    for pattern in [f"{video_stem}_LSTM_cycle*_complete_multi_refined.trc",
                    f"{video_stem}_LSTM_complete_multi_refined.trc"]:
        for f in run_dir.glob(pattern):
            if f.exists():
                final_file = run_dir / f"{video_stem}_final.trc"
                final_file.unlink(missing_ok=True)
                f.rename(final_file)
                break


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
        "--show-video",
        action="store_true",
        help="Render an annotated preview video while extracting landmarks",
    )
    parser.add_argument(
        "--plot-landmarks",
        dest="plot_landmarks",
        action="store_true",
        default=False,
        help="Display the 3D Matplotlib viewer for extracted CSV landmarks",
    )
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
    parser.add_argument(
        "--bone-length-constraints",
        action="store_true",
        help="Enforce consistent bone lengths across frames to improve depth accuracy",
    )
    parser.add_argument(
        "--bone-length-tolerance",
        type=float,
        default=0.15,
        help="Acceptable bone length deviation as fraction (default 0.15 = 15%%)",
    )
    parser.add_argument(
        "--bone-depth-weight",
        type=float,
        default=0.8,
        help="Weight for depth correction vs xy (0-1, default 0.8 focuses on depth)",
    )
    parser.add_argument(
        "--bone-length-iterations",
        type=int,
        default=3,
        help="Number of bone constraint enforcement passes (default 3)",
    )
    parser.add_argument(
        "--bone-length-report",
        action="store_true",
        help="Print detailed bone length consistency statistics",
    )
    parser.add_argument(
        "--ground-plane-refinement",
        action="store_true",
        help="Enhanced ground plane with stance detection and depth propagation",
    )
    parser.add_argument(
        "--ground-contact-threshold",
        type=float,
        default=0.03,
        help="Max distance from ground for foot contact detection (default 0.03m)",
    )
    parser.add_argument(
        "--min-contact-frames",
        type=int,
        default=3,
        help="Minimum consecutive frames for valid stance phase (default 3)",
    )
    parser.add_argument(
        "--depth-propagation-weight",
        type=float,
        default=0.7,
        help="Weight decay for depth correction propagation up kinematic chain (default 0.7)",
    )
    parser.add_argument(
        "--multi-constraint-optimization",
        action="store_true",
        help="Apply iterative multi-constraint optimization (bone lengths + joint angles + ground plane)",
    )
    parser.add_argument(
        "--multi-constraint-iterations",
        type=int,
        default=10,
        help="Max iterations for multi-constraint optimization (default 10)",
    )
    parser.add_argument(
        "--compute-joint-angles",
        action="store_true",
        help="Compute hip/knee/ankle joint angles from augmented TRC (Flex/Ext, Abd/Add, Rotation)",
    )
    parser.add_argument(
        "--joint-angle-side",
        choices=["R", "L"],
        default="R",
        help="Side for joint angle computation (default R)",
    )
    parser.add_argument(
        "--joint-angle-smooth-window",
        type=int,
        default=9,
        help="Smoothing window for joint angle computation (default 9, 0=no smoothing)",
    )
    parser.add_argument(
        "--plot-joint-angles",
        action="store_true",
        help="Visualize computed joint angles (3-panel time series plot)",
    )
    parser.add_argument(
        "--check-joint-constraints",
        action="store_true",
        help="Check for biomechanical joint angle constraint violations",
    )
    parser.add_argument(
        "--compute-upper-body-angles",
        action="store_true",
        help="Compute trunk/shoulder/elbow joint angles (requires augmented TRC)",
    )
    parser.add_argument(
        "--upper-body-side",
        choices=["R", "L"],
        default="R",
        help="Side for upper body angle computation (default R)",
    )
    parser.add_argument(
        "--compute-all-joint-angles",
        action="store_true",
        help="Compute ALL joint angles (pelvis, lower body, trunk, upper body) using ISB standards",
    )
    parser.add_argument(
        "--plot-all-joint-angles",
        action="store_true",
        help="Visualize ALL joint angles in comprehensive multi-panel plot",
    )
    parser.add_argument(
        "--save-angle-comparison",
        action="store_true",
        help="Save side-by-side comparison plot (right vs left)",
    )
    parser.add_argument(
        "--plot-upper-body-angles",
        action="store_true",
        help="Visualize upper body angles (3-panel: trunk/shoulder/elbow)",
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

        # Apply bone length constraints if requested
        if args.bone_length_constraints:
            original_records = records.copy() if args.bone_length_report else []
            records = apply_bone_length_constraints(
                records,
                tolerance=args.bone_length_tolerance,
                depth_weight=args.bone_depth_weight,
                iterations=args.bone_length_iterations,
            )
            print(f"[main] applied bone length constraints to {len(records)} landmark records")

            if args.bone_length_report and original_records:
                report_bone_length_statistics(original_records, records)

        # Apply enhanced ground plane refinement if requested
        if args.ground_plane_refinement:
            records = apply_enhanced_ground_plane_refinement(
                records,
                subject_height=args.height,
                contact_threshold=args.ground_contact_threshold,
                min_contact_frames=args.min_contact_frames,
                propagation_weight=args.depth_propagation_weight,
            )
            print(f"[main] applied enhanced ground plane refinement to {len(records)} landmark records")

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

        # Enable GPU acceleration for Pose2Sim LSTM inference (after MediaPipe)
        patch_pose2sim_gpu()

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

        # Step 4: Multi-constraint optimization
        if args.multi_constraint_optimization:
            from src.kinematics.joint_angles_euler import read_trc

            # Read TRC as numpy array
            marker_idx, frame_nums, times, coords = read_trc(final_output)

            # Apply multi-constraint optimization
            coords_refined, mco_stats = multi_constraint_optimization(
                coords,
                marker_idx,
                subject_height=args.height,
                max_iterations=args.multi_constraint_iterations,
                convergence_threshold=0.01,
                verbose=True,
            )

            # Write refined TRC back
            refined_path = final_output.parent / f"{final_output.stem}_multi_refined.trc"

            import shutil
            shutil.copy(final_output, refined_path)

            # Update data section
            with open(refined_path, 'r') as f:
                lines = f.readlines()

            # Find where data starts
            data_start_idx = None
            for i, line in enumerate(lines):
                if line.strip() and not line.startswith('#') and '\t' in line:
                    parts = line.strip().split('\t')
                    try:
                        float(parts[0])
                        data_start_idx = i
                        break
                    except (ValueError, IndexError):
                        continue

            if data_start_idx is not None:
                # Rewrite data section
                new_data_lines = []
                for fi in range(len(frame_nums)):
                    row_parts = [str(int(frame_nums[fi])), f"{times[fi]:.6f}"]
                    for mi in range(coords_refined.shape[1]):
                        x, y, z = coords_refined[fi, mi]
                        row_parts.extend([f"{x:.6f}", f"{y:.6f}", f"{z:.6f}"])
                    new_data_lines.append('\t'.join(row_parts) + '\n')

                lines = lines[:data_start_idx] + new_data_lines

                with open(refined_path, 'w') as f:
                    f.writelines(lines)

                final_output = refined_path
                append_build_log(f"main step3.6 multi-constraint optimization {final_output}")
                print(f"[main] step3.6 multi-constraint-optimization -> {final_output}")

        # Step 5: Compute joint angles if requested
        if args.compute_joint_angles:
            try:
                angles_df = compute_lower_limb_angles(
                    final_output,
                    side=args.joint_angle_side,
                    smooth_window=args.joint_angle_smooth_window,
                )

                angles_csv = run_dir / f"{video_path.stem}_angles_{args.joint_angle_side}.csv"
                angles_df.to_csv(angles_csv, index=False)
                append_build_log(f"main step5 joint angles {angles_csv}")
                print(f"[main] step5 joint angles -> {angles_csv}")

                # Check for constraint violations if requested
                if args.check_joint_constraints:
                    angles_dict = {
                        "hip": angles_df[["hip_flex_deg", "hip_abd_deg", "hip_rot_deg"]].to_numpy(),
                        "knee": angles_df[["knee_flex_deg", "knee_abd_deg", "knee_rot_deg"]].to_numpy(),
                        "ankle": angles_df[["ankle_flex_deg", "ankle_abd_deg", "ankle_rot_deg"]].to_numpy(),
                    }
                    violations = check_angle_violations(angles_dict)
                    print_violation_summary(violations)

                # Visualize angles if requested
                if args.plot_joint_angles:
                    angles_plot = run_dir / f"{video_path.stem}_angles_{args.joint_angle_side}.png"
                    plot_joint_angles_time_series(
                        angles_df,
                        side=args.joint_angle_side,
                        output_path=angles_plot,
                        title=f"{video_path.stem} — Joint Angles ({args.joint_angle_side} Leg)",
                    )
                    print(f"[main] joint angle plot -> {angles_plot}")

            except Exception as e:
                print(f"[main] WARNING: Joint angle computation failed - {e}", file=sys.stderr)
                import traceback
                traceback.print_exc()

        # Step 6: Compute upper body angles if requested
        if args.compute_upper_body_angles:
            try:
                upper_angles_df = compute_upper_body_angles(
                    final_output,
                    side=args.upper_body_side,
                    smooth_window=args.joint_angle_smooth_window,
                )

                upper_angles_csv = run_dir / f"{video_path.stem}_upper_angles_{args.upper_body_side}.csv"
                upper_angles_df.to_csv(upper_angles_csv, index=False)
                append_build_log(f"main step6 upper body angles {upper_angles_csv}")
                print(f"[main] step6 upper body angles -> {upper_angles_csv}")

                # Visualize upper body angles if requested
                if args.plot_upper_body_angles:
                    upper_angles_plot = run_dir / f"{video_path.stem}_upper_angles_{args.upper_body_side}.png"
                    plot_upper_body_angles(
                        upper_angles_df,
                        side=args.upper_body_side,
                        output_path=upper_angles_plot,
                        title=f"{video_path.stem} — Upper Body Angles ({args.upper_body_side} Arm)",
                    )
                    print(f"[main] upper body angle plot -> {upper_angles_plot}")

            except Exception as e:
                print(f"[main] WARNING: Upper body angle computation failed - {e}", file=sys.stderr)
                import traceback
                traceback.print_exc()

        # Step 7: Compute ALL joint angles (comprehensive ISB-compliant)
        if args.compute_all_joint_angles:
            try:
                print("\n" + "="*60)
                print("Computing Comprehensive Joint Angles (ISB Standards)")
                print("="*60)

                angle_results = compute_all_joint_angles(
                    final_output,
                    smooth_window=args.joint_angle_smooth_window,
                    unwrap=True,
                    zero_mode="first_n_seconds",
                    zero_window_s=0.5,
                    verbose=True,
                )

                # Save all angle CSVs
                save_comprehensive_angles_csv(
                    angle_results,
                    output_dir=run_dir,
                    basename=video_path.stem,
                )

                # Create comprehensive visualization
                if args.plot_all_joint_angles:
                    comprehensive_plot = run_dir / f"{video_path.stem}_all_joint_angles.png"
                    plot_comprehensive_joint_angles(
                        angle_results,
                        output_path=comprehensive_plot,
                        title_prefix=video_path.stem,
                        dpi=150,
                    )

                # Create side-by-side comparison
                if args.save_angle_comparison:
                    comparison_plot = run_dir / f"{video_path.stem}_joint_angles_comparison.png"
                    plot_side_by_side_comparison(
                        angle_results,
                        output_path=comparison_plot,
                        title_prefix=video_path.stem,
                        dpi=150,
                    )

                print("="*60)
                print("Comprehensive Joint Angle Computation Complete")
                print("="*60)

            except Exception as e:
                print(f"[main] WARNING: Comprehensive joint angle computation failed - {e}", file=sys.stderr)
                import traceback
                traceback.print_exc()

        if args.plot_augmented:
            aug_preview = run_dir / f"{final_output.stem}_preview.mp4"
            (visualizer or VisualizeData()).plot_trc_file(
                final_output, export_path=aug_preview, block=True
            )

        # Cleanup and organize output directory
        cleanup_output_directory(run_dir, video_path.stem)

        # Update final_output path if it was renamed
        final_trc = run_dir / f"{video_path.stem}_final.trc"
        if final_trc.exists():
            final_output = final_trc

        print(f"[main] finished pipeline. Output: {final_output}")
    except Exception as exc:
        print(f"[main] error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

import argparse
import os
import platform
import shutil
import subprocess
import sys
import time
import socket
import webbrowser
from pathlib import Path

# Lightweight imports only - heavy deps (mediapipe, torch, matplotlib) deferred to main()
from src.posedetector.base import COCO_TO_MARKER_NAME
from src.datastream.data_stream import LandmarkRecord

def run(cmd, shell=False, check=True, **kwargs):
    print(f"> {cmd}")
    return subprocess.run(cmd, shell=shell, check=check, **kwargs)


def pose_result_to_records(result, marker_names=COCO_TO_MARKER_NAME):
    """Convert PoseDetectionResult to LandmarkRecords.

    For 2D-only results (like RTMPose), sets z=0 as placeholder.
    The actual 3D reconstruction happens via camera-pof.

    Args:
        result: PoseDetectionResult from pose estimator.
        marker_names: Dict mapping COCO-17 indices to marker names.

    Returns:
        List of LandmarkRecord for downstream processing.
    """
    records = []
    has_3d = result.keypoints_3d is not None

    for frame_idx in range(result.num_frames):
        timestamp = float(result.timestamps[frame_idx])

        for joint_idx, marker_name in marker_names.items():
            vis = float(result.visibility[frame_idx, joint_idx])

            # Get 2D position (normalized [0,1])
            x_2d = float(result.keypoints_2d[frame_idx, joint_idx, 0])
            y_2d = float(result.keypoints_2d[frame_idx, joint_idx, 1])

            if has_3d:
                # Use 3D coordinates if available
                x_m = float(result.keypoints_3d[frame_idx, joint_idx, 0])
                y_m = float(result.keypoints_3d[frame_idx, joint_idx, 1])
                z_m = float(result.keypoints_3d[frame_idx, joint_idx, 2])
            else:
                # Placeholder - 3D will be reconstructed via camera-pof
                # Use 2D positions as X,Y; Z=0 as placeholder
                x_m = x_2d
                y_m = y_2d
                z_m = 0.0

            records.append(LandmarkRecord(
                timestamp_s=timestamp,
                landmark=marker_name,
                x_m=x_m,
                y_m=y_m,
                z_m=z_m,
                visibility=vis,
            ))

    return records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="3D human pose estimation pipeline using MediaPipe + Pose2Sim."
    )
    # Required
    parser.add_argument("--video", required=True, help="Input video file")

    # Subject parameters
    parser.add_argument("--height", type=float, default=1.78, help="Subject height in meters")
    parser.add_argument("--mass", type=float, default=75.0, help="Subject mass in kg")

    # Pose estimator selection
    parser.add_argument(
        "--pose-estimator",
        type=str,
        choices=["mediapipe", "rtmpose"],
        default="mediapipe",
        help="Pose estimator to use (default: mediapipe). RTMPose provides better 2D accuracy but no 3D.",
    )
    parser.add_argument(
        "--rtmpose-model",
        type=str,
        choices=["s", "m", "l"],
        default="m",
        help="RTMPose model size: s=small/fast, m=medium/balanced (default), l=large/accurate",
    )

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

    parser.add_argument(
        "--joint-model-path",
        type=str,
        default="models/checkpoints/best_joint_model.pth",
        help="Path to joint refinement model checkpoint",
    )

    # Camera-space POF for 3D reconstruction (experimental)
    parser.add_argument(
        "--camera-pof",
        action="store_true",
        help="Use camera-space POF for 3D reconstruction instead of MediaPipe depth (experimental)",
    )

    # Joint constraint refinement (experimental)
    parser.add_argument(
        "--joint-refinement",
        action="store_true",
        help="Apply neural joint constraint refinement (experimental, requires --compute-all-joint-angles)",
    )
    parser.add_argument(
        "--pof-model-path",
        type=str,
        default="models/checkpoints/best_pof_semgcn-temporal_model.pth",
        help="Path to camera-space POF model checkpoint",
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
        help="Compute all ISB-compliant joint angles (12 joint groups)",
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


def install_uv_windows():
    run(
        [
            "powershell",
            "-ExecutionPolicy",
            "Bypass",
            "-Command",
            "irm https://astral.sh/uv/install.ps1 | iex",
        ]
    )


def install_uv_macos_linux():
    run("curl -LsSf https://astral.sh/uv/install.sh | sh", shell=True)


def ensure_uv_available_in_path():
    """Add common uv installation directories to PATH if they exist."""
    home = os.path.expanduser("~")
    current_path = os.environ.get("PATH", "")

    # Common uv installation paths by platform
    candidate_paths = [
        os.path.join(home, ".cargo", "bin"),  # macOS/Linux (older)
        os.path.join(home, ".local", "bin"),  # Windows and modern uv
    ]

    for candidate in candidate_paths:
        if os.path.isdir(candidate) and candidate not in current_path:
            os.environ["PATH"] = current_path + os.pathsep + candidate
            current_path = os.environ["PATH"]


def get_uv_executable() -> str | None:
    """Return the path to the uv executable, or None if not found."""
    which_result = shutil.which("uv")
    if which_result:
        return which_result

    # Direct check for common installation locations
    home = os.path.expanduser("~")
    candidates = [
        os.path.join(home, ".local", "bin", "uv.exe"),  # Windows
        os.path.join(home, ".local", "bin", "uv"),  # Unix
        os.path.join(home, ".cargo", "bin", "uv.exe"),  # Windows (older)
        os.path.join(home, ".cargo", "bin", "uv"),  # Unix (older)
    ]
    for candidate in candidates:
        if os.path.isfile(candidate):
            return candidate
    return None


def fail_if_uv_missing():
    if not uv_exists():
        print(
            "\n❌ uv installed, but not found in PATH.\n"
            "Restart your terminal (or log out/in) and run again.\n"
        )
        sys.exit(1)

    # Import heavy dependencies only when actually running (not for --help)
    from src.application.build_log import append_build_log
    from src.datastream.data_stream import ORDER_22, csv_to_trc_strict, write_landmark_csv
    from src.datastream.marker_estimation import estimate_missing_markers
    from src.datastream.post_augmentation_estimation import estimate_shoulder_clusters_and_hjc
    from src.datastream.trc_processing import smooth_trc, hide_markers_in_trc
    from src.markeraugmentation.markeraugmentation import run_pose2sim_augment
    from src.markeraugmentation.gpu_config import patch_pose2sim_gpu
    from src.mediastream.media_stream import probe_video_rotation, read_video_rgb
    from src.posedetector.pose_detector import extract_world_landmarks
    from src.posedetector import create_pose_estimator
    from src.pipeline.refinement import apply_neural_joint_refinement, apply_camera_pof_reconstruction
    from src.pipeline.cleanup import cleanup_output_directory

    run_dir = OUTPUT_ROOT / video_path.stem
    run_dir.mkdir(parents=True, exist_ok=True)

def ensure_uv_installed():
    system = platform.system()
    if uv_exists():
        print("uv already installed.")
        return

    print(f"uv not found. Installing for {system}...")
    if system == "Windows":
        install_uv_windows()
    elif system in ("Darwin", "Linux"):
        install_uv_macos_linux()
    else:
        raise RuntimeError(f"Unsupported OS: {system}")

    ensure_uv_available_in_path()
    fail_if_uv_missing()
    print("✅ uv installed and available.")


def get_uv_command() -> str:
    """Return the uv command to use (full path if not in PATH)."""
    uv_path = get_uv_executable()
    if uv_path is None:
        raise RuntimeError("uv executable not found")
    return uv_path


def uv_sync():
    print("\nRunning uv sync...")
    uv = get_uv_command()
    run([uv, "sync"])


def is_port_open(host: str, port: int, timeout: float = 0.3) -> bool:
    try:
        # Lazy import for visualization (only if needed)
        visualizer = None
        if args.plot_landmarks or args.plot_augmented:
            from src.visualizedata.visualize_data import VisualizeData
            visualizer = VisualizeData()

        # Step 1: Extract landmarks from video
        frames, fps = read_video_rgb(video_path)
        preview_rotation = probe_video_rotation(video_path)
        preview_path = None
        if args.show_video or args.export_preview:
            preview_path = run_dir / f"{video_path.stem}_preview.mp4"

        # Handle RTMPose vs MediaPipe
        landmarks_2d = {}
        raw_landmarks = []

        use_rtmpose = args.pose_estimator == "rtmpose"

        if use_rtmpose:
            # RTMPose provides 2D only - force camera-pof for 3D reconstruction
            args.camera_pof = True

            print(f"[main] Using RTMPose-{args.rtmpose_model} (2D only, 3D via camera-pof)")
            estimator = create_pose_estimator(
                "rtmpose",
                rtmpose_model_size=args.rtmpose_model,
            )

            # Run RTMPose detection
            if args.show_video or args.export_preview:
                result = estimator.detect_with_preview(
                    frames, fps, args.visibility_min,
                    preview_output=preview_path,
                    display=args.show_video,
                )
            else:
                result = estimator.detect(frames, fps, args.visibility_min)

            # Convert to LandmarkRecords (with placeholder 3D)
            records = pose_result_to_records(result)

            # Store 2D landmarks for camera-pof (keyed by marker name)
            for frame_idx in range(result.num_frames):
                timestamp = float(result.timestamps[frame_idx])
                for joint_idx, marker_name in COCO_TO_MARKER_NAME.items():
                    landmarks_2d.setdefault(marker_name, []).append({
                        "timestamp_s": timestamp,
                        "x": float(result.keypoints_2d[frame_idx, joint_idx, 0]),
                        "y": float(result.keypoints_2d[frame_idx, joint_idx, 1]),
                    })

            print(f"[main] RTMPose detected {result.num_frames} frames")

        else:
            # MediaPipe path (original behavior)
            # Determine if we need 2D landmarks (for camera-pof 3D reconstruction)
            use_2d_landmarks = args.camera_pof

            detection_output = extract_world_landmarks(
                frames,
                fps,
                Path("models/pose_landmarker_heavy.task"),
                args.visibility_min,
                display=args.show_video,
                return_raw_landmarks=args.plot_landmarks,
                return_2d_landmarks=use_2d_landmarks,
                preview_output=preview_path,
                preview_rotation_degrees=preview_rotation,
            )

            # Unpack detection output based on flags
            if args.plot_landmarks and use_2d_landmarks:
                records, raw_landmarks, landmarks_2d = detection_output
            elif args.plot_landmarks:
                records, raw_landmarks = detection_output
            elif use_2d_landmarks:
                records, landmarks_2d = detection_output
            else:
                records = detection_output

        # Estimate missing markers using symmetry
        if args.estimate_missing:
            original_count = len(records)
            records = estimate_missing_markers(records)
            estimated_count = len(records) - original_count
            print(f"[main] estimated {estimated_count} missing markers using symmetry")

        # Step 1.5: Apply camera-space POF reconstruction (pre-augmentation, COCO-17)
        if args.camera_pof:
            # Camera-space POF reconstruction - used as PRIMARY 3D source
            # Ignores any existing 3D (MediaPipe world coords) and reconstructs from 2D + POF
            image_size = (frames.shape[1], frames.shape[2])  # (H, W)
            records = apply_camera_pof_reconstruction(
                records, args.pof_model_path, landmarks_2d, args.height,
                image_size=image_size, is_primary_3d=True
            )
            append_build_log("main step1.5 camera-space POF reconstruction applied")

        # Step 1: Write CSV
        csv_path = run_dir / f"{video_path.stem}.csv"
        row_count = write_landmark_csv(csv_path, records)
        append_build_log(f"main step1 CSV {csv_path} ({len(frames)} frames, {row_count} rows)")
        print(f"[main] step1 CSV -> {csv_path}")

        if args.plot_landmarks and raw_landmarks:
            if visualizer is None:
                from src.visualizedata.visualize_data import VisualizeData
                visualizer = VisualizeData()
            visualizer.plot_landmarks(raw_landmarks)
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
            camera_space=args.camera_pof,  # Transform coords for POF input
        )
        append_build_log(f"main step3 augment {lstm_path}")
        print(f"[main] step3 augment -> {lstm_path}")

        # Step 3.5: Force complete markers
        final_output = lstm_path
        if args.force_complete:
            final_output = estimate_shoulder_clusters_and_hjc(lstm_path)
            append_build_log(f"main step3.5 complete {final_output}")
            print(f"[main] step3.5 force-complete -> {final_output}")

        # Step 4: Compute joint angles
        if args.compute_all_joint_angles:
            # Lazy import for joint angle computation (loads pandas, scipy)
            from src.kinematics.comprehensive_joint_angles import compute_all_joint_angles
            from src.kinematics.visualize_comprehensive_angles import save_comprehensive_angles_csv

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

                # Apply neural joint constraint refinement (experimental)
                if args.joint_refinement:
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

                # Create visualizations (lazy import matplotlib only when needed)
                if args.plot_all_joint_angles:
                    from src.kinematics.visualize_comprehensive_angles import plot_comprehensive_joint_angles
                    plot_path = run_dir / f"{video_path.stem}_all_joint_angles.png"
                    plot_comprehensive_joint_angles(
                        angle_results,
                        output_path=plot_path,
                        title_prefix=video_path.stem,
                        dpi=150,
                    )

                if args.save_angle_comparison:
                    from src.kinematics.visualize_comprehensive_angles import plot_side_by_side_comparison
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
            if visualizer is None:
                from src.visualizedata.visualize_data import VisualizeData
                visualizer = VisualizeData()
            aug_preview = run_dir / f"{final_output.stem}_preview.mp4"
            visualizer.plot_trc_file(final_output, export_path=aug_preview, block=True)

    elif args.mode == "log":
        server_proc = runserver_log(host, port, Path(args.logfile))

    # Wait until it’s reachable, then open browser
    if args.open_browser:
        if wait_for_server(host, port, max_wait_s=args.wait_seconds):
            open_server_page(host, port)
        else:
            print("Not opening browser because server is not reachable.")

    # Keep the .exe alive when needed
    if args.mode == "spawn":
        # In spawn mode, our process would exit instantly otherwise (bad for windowed exe).
        # So we just keep running while the server is reachable.
        print("\nSpawn mode: keeping launcher alive. Close this app to stop launcher.")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            pass

    if server_proc is not None:
        try:
            server_proc.wait()
        except KeyboardInterrupt:
            # If user Ctrl+C in console mode, propagate
            try:
                server_proc.terminate()
            except Exception:
                pass


if __name__ == "__main__":
    main()

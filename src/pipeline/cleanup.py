"""Output directory cleanup utilities."""

from pathlib import Path
import shutil


def cleanup_output_directory(run_dir: Path, video_stem: str) -> None:
    """Organize output directory by removing intermediate files and organizing results.

    Cleanup actions:
    - Remove intermediate augmentation cycle files
    - Remove temporary Pose2Sim project directories
    - Remove Config files
    - Organize joint angle files into joint_angles/ subdirectory
    - Rename main files for clarity
    """
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
    angle_files = (
        list(run_dir.glob(f"{video_stem}_angles_*.csv"))
        + list(run_dir.glob(f"{video_stem}_all_joint_angles.png"))
        + list(run_dir.glob(f"{video_stem}_joint_angles_comparison.png"))
    )

    if angle_files:
        angle_dir.mkdir(exist_ok=True)
        for f in angle_files:
            dest = angle_dir / f.name
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
    for pattern in [
        f"{video_stem}_LSTM_cycle*_complete_multi_refined.trc",
        f"{video_stem}_LSTM_complete_multi_refined.trc",
    ]:
        for f in run_dir.glob(pattern):
            if f.exists():
                final_file = run_dir / f"{video_stem}_final.trc"
                final_file.unlink(missing_ok=True)
                f.rename(final_file)
                break

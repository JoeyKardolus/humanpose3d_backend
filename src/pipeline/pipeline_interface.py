"""Pipeline interface for HumanPose3D processing.

Provides clean interfaces for running the full pose estimation pipeline:
- PipelineConfig: Configuration dataclass for pipeline parameters
- PipelineResult: Result container with output paths
- run_pipeline(): Main entry point for pipeline execution
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List


@dataclass
class PipelineConfig:
    """Configuration for pipeline execution.

    Attributes:
        video_path: Path to input video file
        height: Subject height in meters (enables metric scale output)
        mass: Subject mass in kg (for Pose2Sim biomechanics)
        output_dir: Output directory (default: data/output)

        # Pose estimation
        pose_model: Pose detector to use ('mediapipe' or 'rtmpose')
        visibility_min: Minimum visibility threshold (default 0.3)

        # Missing marker handling
        estimate_missing: Mirror occluded limbs from visible side
        force_complete: Estimate shoulder clusters and HJC
        hide_markers: List of markers to hide (set to NaN)

        # Augmentation
        augmentation_cycles: Number of averaging cycles (default 1)

        # Neural refinement (experimental, independent options)
        camera_pof: Enable POF 3D reconstruction from 2D keypoints
        joint_refinement: Enable neural joint constraint refinement
        pof_model_path: Path to POF model checkpoint

        # Output options
        compute_joint_angles: Compute ISB joint angles
        plot_joint_angles: Generate joint angle plots
    """

    video_path: Path
    height: float = 1.78
    mass: float = 75.0
    output_dir: Path = field(default_factory=lambda: Path("data/output"))

    # Pose estimation
    pose_model: str = "mediapipe"
    visibility_min: float = 0.3

    # Missing marker handling
    estimate_missing: bool = False
    force_complete: bool = False
    hide_markers: List[str] = field(default_factory=list)

    # Augmentation
    augmentation_cycles: int = 1

    # Neural refinement (experimental)
    camera_pof: bool = False
    joint_refinement: bool = False
    pof_model_path: Optional[Path] = None

    # Output options
    compute_joint_angles: bool = False
    plot_joint_angles: bool = False

    def __post_init__(self):
        """Validate and convert paths."""
        self.video_path = Path(self.video_path)
        self.output_dir = Path(self.output_dir)
        if self.pof_model_path:
            self.pof_model_path = Path(self.pof_model_path)

        # joint_refinement requires compute_joint_angles
        if self.joint_refinement:
            self.compute_joint_angles = True


@dataclass
class PipelineResult:
    """Result container for pipeline outputs.

    Attributes:
        success: Whether pipeline completed successfully
        video_name: Name of processed video (without extension)
        output_dir: Directory containing outputs

        # Generated files
        trc_initial: Path to initial TRC (22 markers)
        trc_final: Path to final TRC (64 markers after augmentation)
        raw_landmarks_csv: Path to raw landmark CSV

        # Joint angles (if computed)
        joint_angles_dir: Directory with joint angle CSV/PNG files
        joint_groups_computed: Number of joint groups computed

        # Metadata
        n_frames: Number of frames processed
        n_markers_final: Number of markers in final TRC

        # Errors (if any)
        error_message: Error message if pipeline failed
    """

    success: bool
    video_name: str
    output_dir: Path

    # Generated files
    trc_initial: Optional[Path] = None
    trc_final: Optional[Path] = None
    raw_landmarks_csv: Optional[Path] = None

    # Joint angles
    joint_angles_dir: Optional[Path] = None
    joint_groups_computed: int = 0

    # Metadata
    n_frames: int = 0
    n_markers_final: int = 0

    # Errors
    error_message: Optional[str] = None


def create_default_config(video_path: str, **kwargs) -> PipelineConfig:
    """Create pipeline config with sensible defaults.

    Args:
        video_path: Path to input video
        **kwargs: Override any config attributes

    Returns:
        PipelineConfig with defaults
    """
    return PipelineConfig(video_path=Path(video_path), **kwargs)


def create_recommended_config(video_path: str, height: float = 1.78) -> PipelineConfig:
    """Create pipeline config with recommended settings for stable results.

    Enables:
    - Missing marker estimation
    - 20-cycle augmentation averaging
    - Joint angle computation and plotting

    Note: Neural options (camera_pof, joint_refinement) are OFF by default.
    Enable them manually if desired.

    Args:
        video_path: Path to input video
        height: Subject height in meters

    Returns:
        PipelineConfig with recommended settings
    """
    return PipelineConfig(
        video_path=Path(video_path),
        height=height,
        estimate_missing=True,
        augmentation_cycles=20,
        compute_joint_angles=True,
        plot_joint_angles=True,
        visibility_min=0.1,
    )

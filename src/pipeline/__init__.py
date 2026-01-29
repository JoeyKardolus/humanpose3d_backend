"""Pipeline orchestration module.

Provides high-level pipeline interfaces and utilities:
- pipeline_interface: Configuration and result dataclasses
- refinement: Neural refinement functions (POF, joint constraints)
- cleanup: Output directory organization
"""

from .pipeline_interface import (
    PipelineConfig,
    PipelineResult,
    create_default_config,
    create_recommended_config,
)

from .refinement import (
    apply_neural_joint_refinement,
    apply_camera_pof_reconstruction,
)

from .cleanup import cleanup_output_directory

__all__ = [
    # Configuration
    "PipelineConfig",
    "PipelineResult",
    "create_default_config",
    "create_recommended_config",
    # Refinement
    "apply_neural_joint_refinement",
    "apply_camera_pof_reconstruction",
    # Cleanup
    "cleanup_output_directory",
]

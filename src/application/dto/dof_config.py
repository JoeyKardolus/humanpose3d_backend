"""DOF (Degrees of Freedom) configuration for joint angle filtering."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# Valid joint names that can be configured
VALID_JOINTS = frozenset([
    "pelvis",
    "trunk",
    "hip_R",
    "hip_L",
    "knee_R",
    "knee_L",
    "ankle_R",
    "ankle_L",
    "shoulder_R",
    "shoulder_L",
    "elbow_R",
    "elbow_L",
])

# Valid DOF types for each joint
VALID_DOF_TYPES = frozenset(["flex", "abd", "rot"])

# Joints that only have flexion (1 DOF)
SINGLE_DOF_JOINTS = frozenset(["elbow_R", "elbow_L"])


@dataclass(frozen=True)
class DofConfig:
    """Configuration specifying which DOF to include for each joint.

    Each joint maps to a frozenset of enabled DOF types ('flex', 'abd', 'rot').
    An empty set means the joint is excluded entirely from output.
    """

    joints: dict[str, frozenset[str]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        for joint_name, dof_set in self.joints.items():
            if joint_name not in VALID_JOINTS:
                raise ValueError(f"Invalid joint name: {joint_name}")
            invalid_dofs = dof_set - VALID_DOF_TYPES
            if invalid_dofs:
                raise ValueError(
                    f"Invalid DOF types for {joint_name}: {invalid_dofs}"
                )

    @classmethod
    def default(cls) -> DofConfig:
        """Create default config with all DOF enabled for all joints."""
        joints: dict[str, frozenset[str]] = {}
        for joint in VALID_JOINTS:
            if joint in SINGLE_DOF_JOINTS:
                joints[joint] = frozenset(["flex"])
            else:
                joints[joint] = frozenset(["flex", "abd", "rot"])
        return cls(joints=joints)

    @classmethod
    def from_json(cls, data: dict[str, Any] | None) -> DofConfig:
        """Parse DOF config from JSON request data.

        Expected format:
        {
            "pelvis": ["flex", "abd", "rot"],
            "hip_R": ["flex"],
            ...
        }

        Joints not specified in the input inherit from default config.
        """
        if not data:
            return cls.default()

        default = cls.default()
        joints: dict[str, frozenset[str]] = dict(default.joints)

        for joint_name, dof_list in data.items():
            if joint_name not in VALID_JOINTS:
                continue
            if not isinstance(dof_list, list):
                continue
            valid_dofs = [d for d in dof_list if d in VALID_DOF_TYPES]
            joints[joint_name] = frozenset(valid_dofs)

        return cls(joints=joints)

    def to_dict(self) -> dict[str, list[str]]:
        """Convert to JSON-serializable dict."""
        return {
            joint: sorted(dof_set) for joint, dof_set in self.joints.items()
        }

    def get_enabled_dof(self, joint_name: str) -> frozenset[str]:
        """Get enabled DOF for a joint, returns empty set if not configured."""
        return self.joints.get(joint_name, frozenset())

    def is_dof_enabled(self, joint_name: str, dof_type: str) -> bool:
        """Check if a specific DOF is enabled for a joint."""
        return dof_type in self.joints.get(joint_name, frozenset())

    def get_column_filter(self, joint_name: str) -> list[str]:
        """Get list of column suffixes to include for a joint.

        Returns list like ['flex_deg', 'abd_deg'] based on enabled DOF.
        """
        enabled = self.get_enabled_dof(joint_name)
        suffixes = []
        if "flex" in enabled:
            suffixes.append("flex_deg")
        if "abd" in enabled:
            suffixes.append("abd_deg")
        if "rot" in enabled:
            suffixes.append("rot_deg")
        return suffixes

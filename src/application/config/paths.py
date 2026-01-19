"""Path configuration helpers for the webapp domain."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AppPaths:
    """Resolved filesystem roots used by the webapp."""

    repo_root: Path
    output_root: Path
    upload_root: Path

    @classmethod
    def from_anchor(cls, anchor: Path) -> "AppPaths":
        """Resolve application roots relative to a module anchor."""
        repo_root = None
        for parent in anchor.resolve().parents:
            if parent.name == "src":
                repo_root = parent.parent
                break
        if repo_root is None:
            repo_root = anchor.resolve().parents[3]
        output_root = (repo_root / "data" / "output").resolve()
        upload_root = (repo_root / "data" / "input").resolve()
        return cls(
            repo_root=repo_root,
            output_root=output_root,
            upload_root=upload_root,
        )

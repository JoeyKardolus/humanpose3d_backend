"""DTO for normalized pipeline request data."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from django.core.files.uploadedfile import UploadedFile
from django.http import HttpRequest


@dataclass(frozen=True)
class PipelineRequestData:
    """Collects form and file data from a Django request."""

    form_data: Mapping[str, str]
    files: Mapping[str, UploadedFile]

    @classmethod
    def from_django_request(cls, request: HttpRequest) -> "PipelineRequestData":
        """Build a DTO from the incoming Django request."""
        return cls(form_data=request.POST, files=request.FILES)

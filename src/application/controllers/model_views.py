"""Django views for model management."""

from __future__ import annotations

from django.http import HttpRequest, JsonResponse
from django.views import View

from src.application.config.user_paths import UserPaths
from src.application.services.model_download_service import ModelDownloadService


class CheckModelsView(View):
    """Check if required model files exist."""

    def get(self, request: HttpRequest) -> JsonResponse:
        """Check model files existence."""
        user_paths = UserPaths.default()
        models_exist = user_paths.models_exist()

        return JsonResponse({
            "models_exist": models_exist,
            "models_path": str(user_paths.models),
        })


class DownloadModelsView(View):
    """Download model files from git repository."""

    def post(self, request: HttpRequest) -> JsonResponse:
        """Trigger model download."""
        user_paths = UserPaths.default()

        # Check if models already exist
        if user_paths.models_exist():
            return JsonResponse({
                "success": True,
                "message": "Models already exist",
            })

        # Ensure directories exist
        user_paths.ensure_directories()

        # Download models
        download_service = ModelDownloadService(user_paths.base)
        success, message = download_service.download_models()

        return JsonResponse({
            "success": success,
            "message": message,
        })

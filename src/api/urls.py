"""URL routing for the HumanPose3D API."""

from django.urls import path

from .views import (
    DeleteRunView,
    DownloadAllView,
    DownloadView,
    MediaView,
    ModelsDownloadView,
    ModelsDownloadProgressView,
    ModelsStatusView,
    PipelineProgressView,
    RunDetailView,
    RunListView,
    RunSyncView,
    StatisticsView,
    UploadMediaView,
)

urlpatterns = [
    path("runs/", RunListView.as_view(), name="api_run_list"),
    path("runs/sync/", RunSyncView.as_view(), name="api_run_sync"),
    path("models/status/", ModelsStatusView.as_view(), name="api_models_status"),
    path("models/download/", ModelsDownloadView.as_view(), name="api_models_download"),
    path(
        "models/download/<str:job_id>/progress/",
        ModelsDownloadProgressView.as_view(),
        name="api_models_download_progress",
    ),
    path("runs/<path:run_key>/", RunDetailView.as_view(), name="api_run_detail"),
    path(
        "runs/<path:run_key>/progress/",
        PipelineProgressView.as_view(),
        name="api_progress",
    ),
    path(
        "runs/<path:run_key>/statistics/",
        StatisticsView.as_view(),
        name="api_statistics",
    ),
    path(
        "runs/<path:run_key>/download-all/",
        DownloadAllView.as_view(),
        name="api_download_all",
    ),
    path(
        "runs/<path:run_key>/download/<path:file_path>/",
        DownloadView.as_view(),
        name="api_download",
    ),
    path(
        "runs/<path:run_key>/media/<path:file_path>/",
        MediaView.as_view(),
        name="api_media",
    ),
    path(
        "runs/<path:run_key>/upload/<path:file_path>/",
        UploadMediaView.as_view(),
        name="api_upload_media",
    ),
    path(
        "runs/<path:run_key>/delete/",
        DeleteRunView.as_view(),
        name="api_delete_run",
    ),
]

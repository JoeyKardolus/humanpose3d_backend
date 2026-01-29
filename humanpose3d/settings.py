"""Django settings for the HumanPose3D backend."""

from __future__ import annotations

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

SECRET_KEY = os.environ.get("DJANGO_SECRET_KEY", "replace-me-with-a-secret-key")
DEBUG = os.environ.get("DJANGO_DEBUG", "1") == "1"
os.environ.setdefault("MPLBACKEND", "Agg")

ALLOWED_HOSTS = ["*"]

# Trust ngrok and common dev origins for CSRF
CSRF_TRUSTED_ORIGINS = [
    "https://amari-off-collin.ngrok-free.dev",
    "https://*.ngrok-free.dev",
    "https://*.ngrok.io",
    "http://localhost:8000",
    "http://localhost:8001",
    "http://127.0.0.1:8000",
    "http://127.0.0.1:8001",
]

DJANGO_APPS = [
    "django.contrib.staticfiles",
]

LOCAL_APPS = [
    "src.application.apps.ApplicationConfig",
    "src.api.apps.ApiConfig",
    "src.cli.apps.CliConfig",
    "src.datastream.apps.DatastreamConfig",
    "src.depth_refinement.apps.DepthRefinementConfig",
    "src.joint_refinement.apps.JointRefinementConfig",
    "src.kinematics.apps.KinematicsConfig",
    "src.main_refinement.apps.MainRefinementConfig",
    "src.markeraugmentation.apps.MarkerAugmentationConfig",
    "src.mediastream.apps.MediastreamConfig",
    "src.pipeline.apps.PipelineConfig",
    "src.posedetector.apps.PosedetectorConfig",
    "src.postprocessing.apps.PostprocessingConfig",
    "src.visualizedata.apps.VisualizedataConfig",
]

INSTALLED_APPS = DJANGO_APPS + LOCAL_APPS

MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
]

ROOT_URLCONF = "humanpose3d.urls"

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [BASE_DIR / "src" / "application" / "templates"],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.debug",
                "django.template.context_processors.request",
            ],
        },
    }
]

WSGI_APPLICATION = "humanpose3d.wsgi.application"
ASGI_APPLICATION = "humanpose3d.asgi.application"

DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.dummy",
    }
}

LANGUAGE_CODE = "en-us"
TIME_ZONE = "UTC"
USE_I18N = True
USE_TZ = True

STATIC_URL = "static/"
STATICFILES_DIRS = [BASE_DIR / "src" / "application" / "static"]

DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"

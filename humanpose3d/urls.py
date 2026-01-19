"""Root URL configuration for the HumanPose3D Django project."""

from django.contrib import admin
from django.urls import include, path

urlpatterns = [
    path("admin/", admin.site.urls),
    path("api/", include("src.api.urls")),
    path("", include("src.application.urls")),
]

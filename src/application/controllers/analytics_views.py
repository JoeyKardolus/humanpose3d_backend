"""Django views for the analytics dashboard."""

from __future__ import annotations

import json
from pathlib import Path

from django.http import HttpRequest, HttpResponse, JsonResponse
from django.shortcuts import render
from django.views import View

from src.application.config.paths import AppPaths, StoragePaths
from src.application.repositories.analytics_repository import AnalyticsRepository
from src.application.services.analytics_service import AnalyticsService


# Module-level wiring
_STORAGE_PATHS = StoragePaths.load()
_ANALYTICS_REPO = AnalyticsRepository(_STORAGE_PATHS.logs_root)
_ANALYTICS_SERVICE = AnalyticsService(_ANALYTICS_REPO)


class AnalyticsDashboardView(View):
    """Analytics dashboard for viewing usage statistics."""

    def get(self, request: HttpRequest) -> HttpResponse:
        """Render the analytics dashboard."""
        stats = _ANALYTICS_SERVICE.get_dashboard_stats()
        quality_trend = _ANALYTICS_SERVICE.get_quality_trend(limit=50)
        settings_correlation = _ANALYTICS_SERVICE.get_settings_correlation()

        # Prepare chart data
        trend_labels = [item["timestamp"][:10] for item in quality_trend]
        trend_values = [item["plausibility_score"] or 0 for item in quality_trend]

        # Error breakdown for pie chart
        error_types = stats.get("error_types", {})
        error_labels = list(error_types.keys()) if error_types else ["No errors"]
        error_values = list(error_types.values()) if error_types else [0]

        context = {
            "stats": stats,
            "quality_trend": quality_trend,
            "settings_correlation": settings_correlation,
            "trend_labels_json": json.dumps(trend_labels),
            "trend_values_json": json.dumps(trend_values),
            "error_labels_json": json.dumps(error_labels),
            "error_values_json": json.dumps(error_values),
        }

        return render(request, "analytics.html", context)


class AnalyticsApiView(View):
    """JSON API for analytics data."""

    def get(self, request: HttpRequest) -> HttpResponse:
        """Return analytics data as JSON."""
        stats = _ANALYTICS_SERVICE.get_dashboard_stats()
        quality_trend = _ANALYTICS_SERVICE.get_quality_trend(limit=50)
        settings_correlation = _ANALYTICS_SERVICE.get_settings_correlation()

        return JsonResponse({
            "stats": stats,
            "quality_trend": quality_trend,
            "settings_correlation": settings_correlation,
        })


def get_analytics_service() -> AnalyticsService:
    """Get the module-level analytics service instance."""
    return _ANALYTICS_SERVICE


def get_analytics_repository() -> AnalyticsRepository:
    """Get the module-level analytics repository instance."""
    return _ANALYTICS_REPO

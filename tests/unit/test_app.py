"""Unit tests for main application (app.py).

Tests monitoring endpoints including health, performance, and logging.
"""
from __future__ import annotations

import os
from unittest.mock import AsyncMock, MagicMock, patch

# Disable Gradio analytics before importing app
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")

# Patch gradio.mount_gradio_app to avoid UI initialization
with patch("gradio.mount_gradio_app", side_effect=lambda app, *a, **k: app):
    import app as app_module


# ---------------------------------------------------------------------------
# Test Classes
# ---------------------------------------------------------------------------

class TestMonitoringHealthEndpoint:
    """Tests for monitoring health endpoint."""

    def test_returns_dashboard_status(self):
        """Health endpoint returns dashboard status."""
        dash = MagicMock()
        dash.api_status = AsyncMock(return_value={"status": "ok"})

        with patch("app.health_check.build_default_dashboard", return_value=dash):
            resp = app_module._client.get("/monitoring/health")

            assert resp.status_code == 200
            assert resp.json() == {"status": "ok"}
            dash.api_status.assert_called_once()


class TestMonitoringPerformanceEndpoint:
    """Tests for monitoring performance endpoint."""

    def test_returns_performance_stats(self):
        """Performance endpoint returns stats from monitor."""
        class Dummy:
            def __enter__(self):
                return self
            def __exit__(self, exc_type, exc, tb):
                return False
            stats = {"cpu": 1}

        with patch("app.performance.PerformanceMonitor", return_value=Dummy()):
            resp = app_module._client.get("/monitoring/performance")

            assert resp.status_code == 200
            assert resp.json() == {"cpu": 1}


class TestMonitoringLoggingEndpoint:
    """Tests for monitoring logging endpoint."""

    def test_returns_log_summary(self):
        """Logging endpoint returns summary and entries."""
        q = MagicMock()
        q.summary.return_value = {"duration": 1}
        q.logs = [1]

        with patch("app.log_utils.PerformanceLogQuery", return_value=q):
            resp = app_module._client.get("/monitoring/logging", params={"file": "f.json"})

            q.load_logs.assert_called_with("f.json")
            assert resp.status_code == 200
            assert resp.json() == {"summary": {"duration": 1}, "entries": [1]}

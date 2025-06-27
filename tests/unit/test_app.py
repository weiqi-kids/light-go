import pathlib
import sys
import os
from unittest.mock import AsyncMock, MagicMock, patch

# Add project root to path
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")

with patch("gradio.mount_gradio_app", side_effect=lambda app, *a, **k: app):
    import app


def test_monitoring_health_endpoint():
    dash = MagicMock()
    dash.api_status = AsyncMock(return_value={"status": "ok"})
    with patch("app.health_check.build_default_dashboard", return_value=dash):
        resp = app._client.get("/monitoring/health")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}
        dash.api_status.assert_called_once()


def test_monitoring_performance_endpoint():
    class Dummy:
        def __enter__(self):
            return self
        def __exit__(self, exc_type, exc, tb):
            return False
        stats = {"cpu": 1}
    with patch("app.performance.PerformanceMonitor", return_value=Dummy()):
        resp = app._client.get("/monitoring/performance")
        assert resp.status_code == 200
        assert resp.json() == {"cpu": 1}


def test_monitoring_logging_endpoint():
    q = MagicMock()
    q.summary.return_value = {"duration": 1}
    q.logs = [1]
    with patch("app.log_utils.PerformanceLogQuery", return_value=q):
        resp = app._client.get("/monitoring/logging", params={"file": "f.json"})
        q.load_logs.assert_called_with("f.json")
        assert resp.status_code == 200
        assert resp.json() == {"summary": {"duration": 1}, "entries": [1]}

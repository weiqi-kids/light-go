"""Unit tests for HealthCheckDashboard (monitoring/health_check.py).

Tests health check registration, execution, and status reporting
using real implementation.
"""
from __future__ import annotations

import asyncio

import pytest

from monitoring.health_check import CheckResult, HealthCheckDashboard


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def dashboard() -> HealthCheckDashboard:
    """Return a fresh HealthCheckDashboard instance."""
    return HealthCheckDashboard()


@pytest.fixture
def dashboard_with_api_check() -> HealthCheckDashboard:
    """Return a dashboard with an API check registered."""
    dash = HealthCheckDashboard()

    async def ok_check() -> CheckResult:
        return CheckResult("OK", 0.0)

    dash.register("api", ok_check)
    return dash


# ---------------------------------------------------------------------------
# Test Classes
# ---------------------------------------------------------------------------

class TestHealthCheckRegistration:
    """Tests for check registration and execution."""

    def test_register_and_run_single(self, dashboard: HealthCheckDashboard):
        """Register a check and run it successfully."""

        async def ok_check() -> CheckResult:
            return CheckResult("OK", 0.0)

        dashboard.register("svc", ok_check)
        result = asyncio.run(dashboard.run_check("svc"))

        assert result.status == "OK"
        assert "svc" in dashboard.results

    def test_run_all_multiple(self, dashboard: HealthCheckDashboard):
        """Run multiple registered checks."""

        async def warn_check() -> CheckResult:
            return CheckResult("WARN", 0.1, "slow")

        dashboard.register("a", lambda: CheckResult("OK", 0.0))
        dashboard.register("b", warn_check)

        results = asyncio.run(dashboard.run_all())

        assert results["a"].status == "OK"
        assert results["b"].status == "WARN"


class TestHealthCheckAPI:
    """Tests for API response methods."""

    def test_api_response(self, dashboard_with_api_check: HealthCheckDashboard):
        """api_status returns correct format."""
        data = asyncio.run(dashboard_with_api_check.api_status())

        assert data["api"]["status"] == "OK"

    def test_html_status_page(self, dashboard_with_api_check: HealthCheckDashboard):
        """html_status returns valid HTML response."""
        resp = asyncio.run(dashboard_with_api_check.html_status())

        assert resp.status_code == 200
        body = resp.body.decode()
        assert "Service Status" in body
        assert "api" in body


class TestHealthCheckAnomalies:
    """Tests for anomaly detection and recovery."""

    def test_anomaly_logging_and_recovery(self, dashboard: HealthCheckDashboard):
        """Anomalies are logged and recovery is tracked."""
        call_count = 0

        async def flapping_check() -> CheckResult:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return CheckResult("DOWN", 0.0, "err")
            return CheckResult("OK", 0.0)

        dashboard.register("svc", flapping_check)

        # First call: DOWN
        asyncio.run(dashboard.run_check("svc"))
        assert len(dashboard.history) == 1
        assert dashboard.history[0]["result"].status == "DOWN"

        # Second call: OK (recovery)
        asyncio.run(dashboard.run_check("svc"))
        assert len(dashboard.history) == 1
        assert dashboard.results["svc"].status == "OK"

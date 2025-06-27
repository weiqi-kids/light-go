import asyncio
import pathlib
import sys

import pytest

# Add project root to path
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from monitoring.health_check import CheckResult, HealthCheckDashboard


def test_register_and_run_single():
    dash = HealthCheckDashboard()

    async def ok_check() -> CheckResult:
        return CheckResult("OK", 0.0)

    dash.register("svc", ok_check)
    result = asyncio.run(dash.run_check("svc"))
    assert result.status == "OK"
    assert "svc" in dash.results


def test_run_all_multiple():
    dash = HealthCheckDashboard()

    async def warn_check() -> CheckResult:
        return CheckResult("WARN", 0.1, "slow")

    dash.register("a", lambda: CheckResult("OK", 0.0))
    dash.register("b", warn_check)
    results = asyncio.run(dash.run_all())
    assert results["a"].status == "OK"
    assert results["b"].status == "WARN"


def _build_dashboard_for_api() -> HealthCheckDashboard:
    dash = HealthCheckDashboard()

    async def ok_check() -> CheckResult:
        return CheckResult("OK", 0.0)

    dash.register("api", ok_check)
    return dash


def test_api_response():
    dash = _build_dashboard_for_api()
    data = asyncio.run(dash.api_status())
    assert data["api"]["status"] == "OK"


def test_html_status_page():
    dash = _build_dashboard_for_api()
    resp = asyncio.run(dash.html_status())
    assert resp.status_code == 200
    body = resp.body.decode()
    assert "Service Status" in body
    assert "api" in body


def test_anomaly_logging_and_recovery():
    dash = HealthCheckDashboard()
    call_count = 0

    async def flapping_check() -> CheckResult:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return CheckResult("DOWN", 0.0, "err")
        return CheckResult("OK", 0.0)

    dash.register("svc", flapping_check)

    asyncio.run(dash.run_check("svc"))
    assert len(dash.history) == 1
    assert dash.history[0]["result"].status == "DOWN"

    asyncio.run(dash.run_check("svc"))
    assert len(dash.history) == 1
    assert dash.results["svc"].status == "OK"

"""Health check dashboard for monitoring multiple services."""
from __future__ import annotations

import argparse
import asyncio
import logging
import sqlite3
import time
from dataclasses import asdict, dataclass
from typing import Any, Awaitable, Callable, Dict, List, Optional

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from jinja2 import Template

# Type alias for check callables
CheckCallable = Callable[[], Awaitable["CheckResult"] | "CheckResult"]


@dataclass
class CheckResult:
    """Result returned by a health check."""

    status: str
    latency: float
    error: Optional[str] = None


class HealthCheckDashboard:
    """Dashboard managing multiple service health checks."""

    def __init__(self) -> None:
        """Initialize empty check registry and FastAPI app."""
        self.checks: Dict[str, CheckCallable] = {}
        self.results: Dict[str, CheckResult] = {}
        self.history: List[Dict[str, Any]] = []
        self.app = FastAPI()
        self.app.get("/status")(self.api_status)
        self.app.get("/")(self.html_status)
        self._task: Optional[asyncio.Task[Any]] = None

    def register(self, name: str, check: CheckCallable) -> None:
        """Register a new check callable."""
        self.checks[name] = check

    async def run_check(self, name: str) -> CheckResult:
        """Run a single check by name."""
        check = self.checks[name]
        start = time.perf_counter()
        try:
            result = check()
            if asyncio.iscoroutine(result):
                result = await result
            if not isinstance(result, CheckResult):
                raise TypeError("Check must return CheckResult")
        except Exception as exc:  # pragma: no cover - unexpected errors
            result = CheckResult("DOWN", time.perf_counter() - start, str(exc))
        self.results[name] = result
        if result.status != "OK":
            self.history.append({"time": time.time(), "name": name, "result": result})
        return result

    async def run_all(self) -> Dict[str, CheckResult]:
        """Run all registered checks."""
        for name in list(self.checks):
            await self.run_check(name)
        return self.results

    async def api_status(self) -> Dict[str, Any]:
        """FastAPI endpoint returning JSON health status."""
        await self.run_all()
        return {name: asdict(res) for name, res in self.results.items()}

    async def html_status(self) -> HTMLResponse:
        """Return a simple HTML status page."""
        await self.run_all()
        template = Template(
            """
            <html>
            <head><title>Service Status</title></head>
            <body>
            <h1>Service Status</h1>
            <table border="1" cellpadding="5">
              <tr><th>Service</th><th>Status</th><th>Latency(ms)</th><th>Error</th></tr>
              {% for name, r in results.items() %}
              <tr>
                <td>{{name}}</td>
                <td>{{r.status}}</td>
                <td>{{ '%.2f' % (r.latency*1000) }}</td>
                <td>{{ r.error or '' }}</td>
              </tr>
              {% endfor %}
            </table>
            </body>
            </html>
            """
        )
        return HTMLResponse(template.render(results=self.results))

    def start_scheduler(self, interval: int = 60) -> None:
        """Start periodic checks in the background."""

        async def _loop() -> None:
            while True:
                await self.run_all()
                await asyncio.sleep(interval)

        if not self._task:
            self._task = asyncio.create_task(_loop())

    def stop_scheduler(self) -> None:
        """Cancel the periodic check task if running."""
        if self._task:
            self._task.cancel()
            self._task = None


# ---------------------------------------------------------------------------
# Example health checks
# ---------------------------------------------------------------------------

async def check_rest_api(url: str) -> CheckResult:
    """Ping a REST endpoint via HTTP GET."""
    import urllib.request

    start = time.perf_counter()
    try:
        with urllib.request.urlopen(url, timeout=5) as resp:
            status = "OK" if resp.status == 200 else "WARN"
            err = None if resp.status == 200 else f"HTTP {resp.status}"
    except Exception as exc:
        return CheckResult("DOWN", time.perf_counter() - start, str(exc))
    return CheckResult(status, time.perf_counter() - start, err)


async def check_websocket(url: str) -> CheckResult:
    """Open a WebSocket connection and close immediately."""
    import websockets

    start = time.perf_counter()
    try:
        async with websockets.connect(url) as ws:
            await ws.send("ping")
            await ws.recv()
        return CheckResult("OK", time.perf_counter() - start)
    except Exception as exc:
        return CheckResult("DOWN", time.perf_counter() - start, str(exc))


async def check_gpu() -> CheckResult:
    """Check if GPU (CUDA) is available."""
    import torch

    start = time.perf_counter()
    try:
        available = torch.cuda.is_available()
        status = "OK" if available else "WARN"
        err = None if available else "CUDA not available"
    except Exception as exc:  # pragma: no cover - unlikely
        return CheckResult("DOWN", time.perf_counter() - start, str(exc))
    return CheckResult(status, time.perf_counter() - start, err)


async def check_performance(threshold: float = 2.0) -> CheckResult:
    """Check system load average."""
    import os

    start = time.perf_counter()
    try:
        load, _, _ = os.getloadavg()
        status = "OK" if load < threshold else "WARN"
        err = None if load < threshold else f"load {load:.2f}"
    except Exception as exc:  # pragma: no cover - non POSIX
        return CheckResult("DOWN", time.perf_counter() - start, str(exc))
    return CheckResult(status, time.perf_counter() - start, err)


async def check_database(path: str = ":memory:") -> CheckResult:
    """Check simple sqlite3 query."""
    start = time.perf_counter()
    try:
        conn = sqlite3.connect(path)
        conn.execute("SELECT 1")
        conn.close()
    except Exception as exc:
        return CheckResult("DOWN", time.perf_counter() - start, str(exc))
    return CheckResult("OK", time.perf_counter() - start)


# ---------------------------------------------------------------------------
# CLI interface
# ---------------------------------------------------------------------------

def build_default_dashboard() -> HealthCheckDashboard:
    """Return a dashboard with example checks registered."""
    dash = HealthCheckDashboard()
    dash.register("rest_api", lambda: check_rest_api("http://localhost:8000"))
    dash.register("websocket", lambda: check_websocket("ws://localhost:8765"))
    dash.register("gpu", check_gpu)
    dash.register("performance", check_performance)
    dash.register("database", check_database)
    return dash


def main(argv: Optional[List[str]] = None) -> None:
    """Entry point for command line."""
    parser = argparse.ArgumentParser(description="Health check dashboard")
    parser.add_argument("--check", help="Run a specific check")
    parser.add_argument("--serve", action="store_true", help="Start web server")
    parser.add_argument("--interval", type=int, default=60, help="Scheduler interval")
    args = parser.parse_args(argv)

    dash = build_default_dashboard()

    if args.check:
        result = asyncio.run(dash.run_check(args.check))
        print(asdict(result))
        return

    if args.serve:
        import uvicorn  # pragma: no cover - manual run

        dash.start_scheduler(args.interval)
        uvicorn.run(dash.app, host="0.0.0.0", port=8000)
        return

    results = asyncio.run(dash.run_all())
    for name, res in results.items():
        print(name, asdict(res))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()

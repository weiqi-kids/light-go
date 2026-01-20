"""Performance tests for monitoring module (monitoring/performance.py).

Tests performance monitoring context manager and decorator.
Note: GPU monitoring requires mocking as GPUtil may not be available.
"""
from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Provide minimal psutil module for import if missing
if "psutil" not in sys.modules:
    psutil_mock = types.ModuleType("psutil")

    class _Proc:
        def cpu_times(self):
            return types.SimpleNamespace(user=0.0, system=0.0)

        def memory_info(self):
            return types.SimpleNamespace(rss=100)

    psutil_mock.Process = lambda pid=None: _Proc()
    sys.modules["psutil"] = psutil_mock

from monitoring.performance import PerformanceMonitor, monitor_performance


# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------

def _heavy_task() -> int:
    """Simulate a CPU-intensive task."""
    return sum(range(1000))


# ---------------------------------------------------------------------------
# Test Classes
# ---------------------------------------------------------------------------

class TestPerformanceMonitorContext:
    """Tests for PerformanceMonitor context manager."""

    def test_records_and_writes_output(self, tmp_path: Path):
        """Monitor records performance and writes to output file."""
        output_file = tmp_path / "perf.json"
        mock_gpu = MagicMock(id=0, load=0.5, memoryUsed=100, memoryTotal=200)

        with patch("monitoring.performance.GPUtil") as gpu_mod:
            gpu_mod.getGPUs.return_value = [mock_gpu]
            with PerformanceMonitor(output=str(output_file)) as monitor:
                _heavy_task()

        assert output_file.exists()
        data = json.loads(output_file.read_text())
        assert data["duration"] >= 0
        assert data["gpu_start"][0]["id"] == 0

    def test_exception_still_logs(self, tmp_path: Path):
        """Monitor logs even when exception occurs."""
        output_file = tmp_path / "error.json"

        with pytest.raises(ValueError):
            with PerformanceMonitor(output=str(output_file)):
                raise ValueError("boom")

        assert output_file.exists()


class TestPerformanceMonitorDecorator:
    """Tests for monitor_performance decorator."""

    def test_decorator_records_performance(self, tmp_path: Path):
        """Decorator records performance stats."""
        output_file = tmp_path / "decorator.json"
        mock_gpu = MagicMock(id=1, load=0.2, memoryUsed=50, memoryTotal=100)

        with patch("monitoring.performance.GPUtil") as gpu_mod:
            gpu_mod.getGPUs.return_value = [mock_gpu]

            @monitor_performance(output=str(output_file))
            def wrapped():
                return _heavy_task()

            result = wrapped()

        assert result == sum(range(1000))
        assert output_file.exists()
        stats = json.loads(output_file.read_text())
        assert stats["gpu_start"][0]["id"] == 1
        assert wrapped.last_performance["duration"] >= 0

import json
import pathlib
import sys
import types
from unittest.mock import MagicMock, patch

import pytest

# Provide minimal psutil module for import if missing
if 'psutil' not in sys.modules:
    psutil_mock = types.ModuleType('psutil')

    class _Proc:
        def cpu_times(self):
            return types.SimpleNamespace(user=0.0, system=0.0)

        def memory_info(self):
            return types.SimpleNamespace(rss=100)

    psutil_mock.Process = lambda pid=None: _Proc()
    sys.modules['psutil'] = psutil_mock

# Add project root to path
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from monitoring.performance import PerformanceMonitor, monitor_performance


def _heavy_task():
    return sum(range(1000))


def test_context_monitor_records_and_writes(tmp_path):
    out_file = tmp_path / "perf.json"
    mock_gpu = MagicMock(id=0, load=0.5, memoryUsed=100, memoryTotal=200)
    with patch("monitoring.performance.GPUtil") as gpu_mod:
        gpu_mod.getGPUs.return_value = [mock_gpu]
        with PerformanceMonitor(output=str(out_file)) as mon:
            _heavy_task()
    assert out_file.exists()
    data = json.loads(out_file.read_text())
    assert data["duration"] >= 0
    assert data["gpu_start"][0]["id"] == 0


def test_decorator_records(tmp_path):
    out_file = tmp_path / "decorator.json"
    mock_gpu = MagicMock(id=1, load=0.2, memoryUsed=50, memoryTotal=100)
    with patch("monitoring.performance.GPUtil") as gpu_mod:
        gpu_mod.getGPUs.return_value = [mock_gpu]

        @monitor_performance(output=str(out_file))
        def wrapped():
            return _heavy_task()

        result = wrapped()
    assert result == sum(range(1000))
    assert out_file.exists()
    stats = json.loads(out_file.read_text())
    assert stats["gpu_start"][0]["id"] == 1
    assert wrapped.last_performance["duration"] >= 0


def test_exception_still_logs(tmp_path):
    out_file = tmp_path / "error.json"
    with pytest.raises(ValueError):
        with PerformanceMonitor(output=str(out_file)):
            raise ValueError("boom")
    assert out_file.exists()

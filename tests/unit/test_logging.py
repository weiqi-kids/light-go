"""Unit tests for PerformanceLogQuery (monitoring/logging.py).

Tests log loading, querying, summarization, and anomaly detection
using real implementation.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import pytest

from monitoring.logging import PerformanceLogQuery, main


# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------

def _write_json_logs(path: Path) -> List[Dict]:
    """Write sample JSON log data and return it."""
    data = [
        {"task": "t1", "timestamp": "2023-01-01T00:00:00", "duration": 10, "cpu": 50, "gpu": 20, "ram": 1024},
        {"task": "t2", "timestamp": "2023-01-01T01:00:00", "duration": 5, "cpu": 60, "gpu": 30, "ram": 2048},
        {"task": "t1", "timestamp": "2023-01-01T02:00:00", "duration": 8, "cpu": 40, "gpu": 25, "ram": 1536},
    ]
    path.write_text(json.dumps(data))
    return data


def _write_csv_logs(path: Path) -> List[Dict]:
    """Write sample CSV log data and return expected dict format."""
    lines = [
        "task,timestamp,duration,cpu,gpu,ram",
        "t1,2023-01-01T00:00:00,10,50,20,1024",
        "t2,2023-01-01T01:00:00,5,60,30,2048",
    ]
    path.write_text("\n".join(lines))
    return [
        {"task": "t1", "timestamp": "2023-01-01T00:00:00", "duration": 10, "cpu": 50, "gpu": 20, "ram": 1024},
        {"task": "t2", "timestamp": "2023-01-01T01:00:00", "duration": 5, "cpu": 60, "gpu": 30, "ram": 2048},
    ]


# ---------------------------------------------------------------------------
# Test Classes
# ---------------------------------------------------------------------------

class TestLogLoading:
    """Tests for log file loading."""

    def test_load_json(self, tmp_path: Path):
        """Load logs from JSON file."""
        path = tmp_path / "perf.json"
        data = _write_json_logs(path)

        query = PerformanceLogQuery()
        query.load_logs(str(path))

        assert query.logs == data

    def test_load_csv(self, tmp_path: Path):
        """Load logs from CSV file."""
        path = tmp_path / "perf.csv"
        expected = _write_csv_logs(path)

        query = PerformanceLogQuery()
        query.load_logs(str(path))

        assert query.logs == expected


class TestLogQuery:
    """Tests for log querying and summarization."""

    def test_query_and_summary(self, tmp_path: Path):
        """Query logs and generate summary."""
        path = tmp_path / "perf.json"
        _write_json_logs(path)

        query = PerformanceLogQuery()
        query.load_logs(str(path))

        # Filter by task
        t1_logs = query.query(lambda r: r["task"] == "t1")
        assert len(t1_logs) == 2

        # Overall summary
        summary = query.summary()
        assert pytest.approx(summary["duration"], rel=1e-5) == (10 + 5 + 8) / 3

        # Grouped summary
        grouped = query.summary(by="task")
        assert pytest.approx(grouped["t1"]["duration"], rel=1e-5) == 9
        assert pytest.approx(grouped["t2"]["duration"], rel=1e-5) == 5


class TestLogComparison:
    """Tests for log comparison and anomaly detection."""

    def test_compare_and_anomalies(self, tmp_path: Path):
        """Compare logs and find anomalies."""
        path = tmp_path / "perf.json"
        data = _write_json_logs(path)

        query = PerformanceLogQuery()
        query.load_logs(str(path))

        # Compare two entries
        diff = query.compare(data[0], data[1])
        assert diff["duration"] == -5
        assert diff["cpu"] == 10

        # Find anomalies
        anomalies = query.find_anomalies({"duration": 9, "cpu": 55})
        assert len(anomalies) == 2


class TestLogCLI:
    """Tests for CLI interface."""

    def test_cli_summary(self, tmp_path: Path, capsys):
        """CLI outputs summary in JSON format."""
        path = tmp_path / "perf.json"
        _write_json_logs(path)

        argv = ["--file", str(path), "--summary", "--group", "task"]
        main(argv)

        out = capsys.readouterr().out
        result = json.loads(out)
        assert "t1" in result
        assert "duration" in result["t1"]

import json
import pathlib
import sys
from typing import List

import pytest

# Add project root
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from monitoring.logging import PerformanceLogQuery, main


def _write_json(path: pathlib.Path) -> List[dict]:
    data = [
        {"task": "t1", "timestamp": "2023-01-01T00:00:00", "duration": 10, "cpu": 50, "gpu": 20, "ram": 1024},
        {"task": "t2", "timestamp": "2023-01-01T01:00:00", "duration": 5, "cpu": 60, "gpu": 30, "ram": 2048},
        {"task": "t1", "timestamp": "2023-01-01T02:00:00", "duration": 8, "cpu": 40, "gpu": 25, "ram": 1536},
    ]
    path.write_text(json.dumps(data))
    return data


def _write_csv(path: pathlib.Path) -> List[dict]:
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


def test_load_json(tmp_path: pathlib.Path):
    path = tmp_path / "perf.json"
    data = _write_json(path)

    q = PerformanceLogQuery()
    q.load_logs(str(path))

    assert q.logs == data


def test_load_csv(tmp_path: pathlib.Path):
    path = tmp_path / "perf.csv"
    expected = _write_csv(path)

    q = PerformanceLogQuery()
    q.load_logs(str(path))

    assert q.logs == expected


def test_query_and_summary(tmp_path: pathlib.Path):
    path = tmp_path / "perf.json"
    _write_json(path)
    q = PerformanceLogQuery()
    q.load_logs(str(path))

    t1_logs = q.query(lambda r: r["task"] == "t1")
    assert len(t1_logs) == 2

    summary = q.summary()
    assert pytest.approx(summary["duration"], rel=1e-5) == (10 + 5 + 8) / 3

    grouped = q.summary(by="task")
    assert pytest.approx(grouped["t1"]["duration"], rel=1e-5) == 9
    assert pytest.approx(grouped["t2"]["duration"], rel=1e-5) == 5


def test_compare_and_anomalies(tmp_path: pathlib.Path):
    path = tmp_path / "perf.json"
    data = _write_json(path)
    q = PerformanceLogQuery()
    q.load_logs(str(path))

    diff = q.compare(data[0], data[1])
    assert diff["duration"] == -5
    assert diff["cpu"] == 10

    anomalies = q.find_anomalies({"duration": 9, "cpu": 55})
    assert len(anomalies) == 2


def test_cli_summary(tmp_path: pathlib.Path, capsys):
    path = tmp_path / "perf.json"
    _write_json(path)
    argv = ["--file", str(path), "--summary", "--group", "task"]
    main(argv)
    out = capsys.readouterr().out
    result = json.loads(out)
    assert "t1" in result
    assert "duration" in result["t1"]

"""Utilities for querying and comparing performance logs."""
from __future__ import annotations

import argparse
import csv
import json
from typing import Any, Callable, Dict, Iterable, List, Optional


class PerformanceLogQuery:
    """Load and analyze performance log files.

    The log files should be produced by ``performance.py`` and stored either as
    JSON or CSV.  Each record is represented as a dictionary containing at least
    a task name, timestamp and resource usage metrics.
    """

    def __init__(self) -> None:
        """Create an empty query object."""
        self.logs: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Loading and querying
    # ------------------------------------------------------------------
    def load_logs(self, filepath: str) -> None:
        """Load performance records from ``filepath``.

        The format is automatically detected from the file extension. JSON files
        must contain a list of objects, while CSV files will be parsed using
        ``csv.DictReader``.
        """

        if filepath.endswith(".json"):
            with open(filepath, "r", encoding="utf-8") as f:
                self.logs = json.load(f)
        elif filepath.endswith(".csv"):
            with open(filepath, newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                self.logs = [self._convert_types(row) for row in reader]
        else:
            raise ValueError("Unsupported log format: %s" % filepath)

    def query(self, filter_func: Callable[[Dict[str, Any]], bool]) -> List[Dict[str, Any]]:
        """Return all records matching ``filter_func``."""
        return [log for log in self.logs if filter_func(log)]

    def summary(self, by: Optional[str] = None) -> Dict[str, Any]:
        """Return aggregated statistics of the loaded logs.

        Parameters
        ----------
        by:
            Optional field name to group by.  When ``None`` the returned
            dictionary contains the global average for every numeric field.
        """
        if not self.logs:
            return {}

        numeric_fields = [k for k, v in self.logs[0].items() if isinstance(v, (int, float))]

        if by:
            groups: Dict[str, Dict[str, float]] = {}
            counts: Dict[str, int] = {}
            for entry in self.logs:
                key = str(entry.get(by))
                counts[key] = counts.get(key, 0) + 1
                agg = groups.setdefault(key, {f: 0.0 for f in numeric_fields})
                for field in numeric_fields:
                    agg[field] += float(entry.get(field, 0))
            for key, agg in groups.items():
                for field in numeric_fields:
                    agg[field] /= counts[key]
            return groups

        totals = {f: 0.0 for f in numeric_fields}
        for entry in self.logs:
            for field in numeric_fields:
                totals[field] += float(entry.get(field, 0))
        for field in numeric_fields:
            totals[field] /= len(self.logs)
        return totals

    def compare(self, log1: Dict[str, Any], log2: Dict[str, Any]) -> Dict[str, float]:
        """Return difference between ``log2`` and ``log1`` for numeric fields."""
        diff: Dict[str, float] = {}
        for key in log1:
            if key in log2 and isinstance(log1[key], (int, float)) and isinstance(log2[key], (int, float)):
                diff[key] = float(log2[key]) - float(log1[key])
        return diff

    def find_anomalies(self, thresholds: Dict[str, float]) -> List[Dict[str, Any]]:
        """Return log entries exceeding provided ``thresholds``."""
        results = []
        for entry in self.logs:
            for field, limit in thresholds.items():
                val = entry.get(field)
                if isinstance(val, (int, float)) and val > limit:
                    results.append(entry)
                    break
        return results

    # ------------------------------------------------------------------
    # Helper utilities
    # ------------------------------------------------------------------
    @staticmethod
    def _convert_types(row: Dict[str, str]) -> Dict[str, Any]:
        """Attempt to convert CSV values to numeric types."""
        out: Dict[str, Any] = {}
        for key, value in row.items():
            try:
                out[key] = int(value)
            except ValueError:
                try:
                    out[key] = float(value)
                except ValueError:
                    out[key] = value
        return out


def main(argv: Optional[Iterable[str]] = None) -> None:
    """Command line entry point for quick queries."""
    parser = argparse.ArgumentParser(description="Performance log query tool")
    parser.add_argument("--file", required=True, help="Path to log file")
    parser.add_argument("--summary", action="store_true", help="Show summary stats")
    parser.add_argument("--group", help="Group field for summary")
    parser.add_argument("--compare", nargs=2, type=int, metavar=("A", "B"), help="Compare two log indices")
    parser.add_argument("--task", help="Filter by task name")
    parser.add_argument("--anomaly", action="store_true", help="Detect anomalies")
    parser.add_argument("--threshold", nargs="*", default=[], help="Thresholds like field=value")

    args = parser.parse_args(list(argv) if argv is not None else None)

    q = PerformanceLogQuery()
    q.load_logs(args.file)

    logs = q.logs
    if args.task:
        logs = q.query(lambda r: r.get("task") == args.task)

    if args.compare:
        a, b = args.compare
        diff = q.compare(logs[a], logs[b])
        print(json.dumps(diff, indent=2))
        return

    if args.anomaly:
        thr = {}
        for item in args.threshold:
            if "=" in item:
                key, val = item.split("=", 1)
                try:
                    thr[key] = float(val)
                except ValueError:
                    continue
        print(json.dumps(q.find_anomalies(thr), indent=2))
        return

    if args.summary:
        print(json.dumps(q.summary(args.group), indent=2))
    else:
        print(json.dumps(logs, indent=2))


if __name__ == "__main__":  # pragma: no cover - manual execution
    main()

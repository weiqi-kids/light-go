"""Performance monitoring utilities.

This module provides the :class:`PerformanceMonitor` context manager and a
``@monitor_performance`` decorator for measuring CPU, memory and GPU usage of
code blocks or functions. Metrics can optionally be logged to JSON or CSV
files. The module can also be executed directly as a simple CLI which prints
example usage.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from typing import Any, Dict, List, Optional

import psutil

try:
    import GPUtil  # type: ignore
except Exception:  # pragma: no cover - library may not be installed
    GPUtil = None


class PerformanceMonitor:
    """Context manager for monitoring CPU, memory and GPU usage.

    Parameters
    ----------
    output : str, optional
        Path to an output file. If provided, metrics will be stored when the
        context exits. The format is determined by the file extension (``.json``
        or ``.csv``).
    """

    def __init__(self, output: Optional[str] = None) -> None:
        """Initialize the monitor; optional ``output`` path saves results."""
        self.output = output
        self.stats: Dict[str, Any] = {}
        self._process = psutil.Process(os.getpid())
        self._start_cpu = None
        self._start_mem = None
        self._start_gpu = None
        self._start_time = None

    # ------------------------------------------------------------------
    def _gpu_usage(self) -> Optional[List[Dict[str, Any]]]:
        """Return GPU usage information if GPUtil is available."""
        if GPUtil is None:
            return None
        try:
            gpus = GPUtil.getGPUs()  # type: ignore[attr-defined]
        except Exception:  # pragma: no cover - runtime errors are ignored
            return None
        usage = [
            {
                "id": gpu.id,
                "load": getattr(gpu, "load", 0),
                "memory_used": getattr(gpu, "memoryUsed", 0),
                "memory_total": getattr(gpu, "memoryTotal", 0),
            }
            for gpu in gpus
        ]
        return usage or None

    # ------------------------------------------------------------------
    def __enter__(self) -> "PerformanceMonitor":
        """Start recording performance metrics."""
        self._start_time = time.time()
        self._start_cpu = self._process.cpu_times()
        self._start_mem = self._process.memory_info().rss
        self._start_gpu = self._gpu_usage()
        return self

    # ------------------------------------------------------------------
    def __exit__(self, exc_type, exc, tb) -> bool:
        """Stop monitoring and optionally persist the results."""
        end_time = time.time()
        end_cpu = self._process.cpu_times()
        end_mem = self._process.memory_info().rss
        end_gpu = self._gpu_usage()

        cpu_start = (self._start_cpu.user + self._start_cpu.system) if self._start_cpu else 0
        cpu_end = end_cpu.user + end_cpu.system
        cpu_diff = cpu_end - cpu_start

        self.stats = {
            "start_time": self._start_time,
            "end_time": end_time,
            "duration": end_time - (self._start_time or end_time),
            "cpu_time_start": cpu_start,
            "cpu_time_end": cpu_end,
            "cpu_time_diff": cpu_diff,
            "memory_start": self._start_mem,
            "memory_end": end_mem,
            "memory_diff": end_mem - (self._start_mem or end_mem),
            "gpu_start": self._start_gpu,
            "gpu_end": end_gpu,
        }

        if self.output:
            self.log_performance(self.output)
        # Propagate any exception
        return False

    # ------------------------------------------------------------------
    def log_performance(self, output_file: Optional[str] = None) -> Dict[str, Any]:
        """Save the collected performance metrics.

        Parameters
        ----------
        output_file : str, optional
            Destination file. If omitted, ``self.output`` is used.

        Returns
        -------
        dict
            The metrics dictionary.
        """

        path = output_file or self.output
        if not path:
            return self.stats

        ext = os.path.splitext(path)[1].lower()
        if ext == ".json":
            with open(path, "w", encoding="utf-8") as f:
                json.dump(self.stats, f, indent=2)
        elif ext == ".csv":
            with open(path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["metric", "value"])
                for k, v in self.stats.items():
                    writer.writerow([k, v])
        else:
            raise ValueError(f"Unsupported output format: {ext}")
        return self.stats


# ----------------------------------------------------------------------

def monitor_performance(func=None, *, output: Optional[str] = None):
    """Decorator to monitor performance of a function."""

    def decorator(fn):
        def wrapper(*args, **kwargs):
            with PerformanceMonitor(output=output) as pm:
                result = fn(*args, **kwargs)
            wrapper.last_performance = pm.stats
            return result

        wrapper.last_performance = None  # type: ignore[attr-defined]
        wrapper.__name__ = getattr(fn, "__name__", "wrapped")
        wrapper.__doc__ = fn.__doc__
        return wrapper

    if callable(func):
        return decorator(func)
    return decorator


# ----------------------------------------------------------------------
# Example usage when run as a script

def _example_task():
    """Run a tiny workload used in the module self-test."""
    total = sum(range(10000))
    return total


@monitor_performance()
def _decorated_example():
    """Sleep briefly to demonstrate the decorator."""
    time.sleep(0.05)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Demo of PerformanceMonitor")
    parser.add_argument("--output", help="Path to output JSON or CSV file", default=None)
    args = parser.parse_args()

    print("Running example task using context manager...")
    with PerformanceMonitor(output=args.output) as mon:
        _example_task()
    print(json.dumps(mon.stats, indent=2))

    print("Running decorated example task...")
    _decorated_example()
    print(json.dumps(_decorated_example.last_performance, indent=2))

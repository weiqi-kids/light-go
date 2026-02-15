"""Worker health monitoring for distributed self-play.

This module monitors the health of distributed workers and handles
failure detection and recovery.
"""

from __future__ import annotations

import time
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class WorkerStatus(Enum):
    """Status of a worker."""
    UNKNOWN = "unknown"
    STARTING = "starting"
    RUNNING = "running"
    IDLE = "idle"
    DEAD = "dead"
    STOPPED = "stopped"


@dataclass
class WorkerStats:
    """Statistics for a single worker.

    Attributes
    ----------
    worker_id : int
        Worker identifier
    games_played : int
        Total games completed
    examples_generated : int
        Total training examples generated
    last_heartbeat : float
        Unix timestamp of last heartbeat
    model_version : int
        Current model version being used
    avg_game_length : float
        Average moves per game
    status : WorkerStatus
        Current worker status
    start_time : float
        When worker started
    error_count : int
        Number of errors encountered
    last_error : Optional[str]
        Last error message
    """

    worker_id: int
    games_played: int = 0
    examples_generated: int = 0
    last_heartbeat: float = 0.0
    model_version: int = 0
    avg_game_length: float = 0.0
    status: WorkerStatus = WorkerStatus.UNKNOWN
    start_time: float = field(default_factory=time.time)
    error_count: int = 0
    last_error: Optional[str] = None

    def update_from_heartbeat(self, data: Dict[str, Any]) -> None:
        """Update stats from heartbeat data.

        Parameters
        ----------
        data : Dict[str, Any]
            Heartbeat data from worker
        """
        self.last_heartbeat = time.time()
        self.games_played = data.get('games_played', self.games_played)
        self.examples_generated = data.get('examples_generated', self.examples_generated)
        self.model_version = data.get('model_version', self.model_version)
        self.avg_game_length = data.get('avg_game_length', self.avg_game_length)
        self.status = WorkerStatus.RUNNING

    def time_since_heartbeat(self) -> float:
        """Get time since last heartbeat.

        Returns
        -------
        float
            Seconds since last heartbeat
        """
        if self.last_heartbeat == 0:
            return float('inf')
        return time.time() - self.last_heartbeat


@dataclass
class AggregateStats:
    """Aggregate statistics across all workers.

    Attributes
    ----------
    total_workers : int
        Total number of workers
    active_workers : int
        Number of active workers
    dead_workers : int
        Number of dead workers
    total_games : int
        Total games across all workers
    total_examples : int
        Total examples across all workers
    games_per_second : float
        Aggregate game generation rate
    examples_per_second : float
        Aggregate example generation rate
    """

    total_workers: int = 0
    active_workers: int = 0
    dead_workers: int = 0
    total_games: int = 0
    total_examples: int = 0
    games_per_second: float = 0.0
    examples_per_second: float = 0.0


class HealthMonitor:
    """Monitors health of distributed workers.

    Features:
    - Heartbeat tracking
    - Dead worker detection
    - Performance metrics
    - Automatic restart triggers

    Parameters
    ----------
    heartbeat_timeout : float
        Seconds before worker considered dead (default 60)
    check_interval : float
        Seconds between health checks (default 10)
    """

    def __init__(
        self,
        heartbeat_timeout: float = 60.0,
        check_interval: float = 10.0
    ):
        self.heartbeat_timeout = heartbeat_timeout
        self.check_interval = check_interval

        self._workers: Dict[int, WorkerStats] = {}
        self._last_check: float = 0.0
        self._rate_tracker: Dict[int, Dict[str, Any]] = {}

    def register_worker(self, worker_id: int) -> None:
        """Register a new worker.

        Parameters
        ----------
        worker_id : int
            Worker identifier
        """
        self._workers[worker_id] = WorkerStats(
            worker_id=worker_id,
            status=WorkerStatus.STARTING
        )
        self._rate_tracker[worker_id] = {
            'last_games': 0,
            'last_examples': 0,
            'last_time': time.time()
        }
        logger.info(f"Registered worker {worker_id}")

    def unregister_worker(self, worker_id: int) -> None:
        """Unregister a worker.

        Parameters
        ----------
        worker_id : int
            Worker identifier
        """
        if worker_id in self._workers:
            self._workers[worker_id].status = WorkerStatus.STOPPED
            logger.info(f"Unregistered worker {worker_id}")

    def update_heartbeat(
        self,
        worker_id: int,
        stats: Dict[str, Any]
    ) -> None:
        """Update worker heartbeat and stats.

        Parameters
        ----------
        worker_id : int
            Worker identifier
        stats : Dict[str, Any]
            Worker statistics
        """
        if worker_id not in self._workers:
            self.register_worker(worker_id)

        worker = self._workers[worker_id]
        worker.update_from_heartbeat(stats)

        # Update rate tracking
        now = time.time()
        tracker = self._rate_tracker.get(worker_id, {})
        elapsed = now - tracker.get('last_time', now)

        if elapsed > 0:
            games_delta = worker.games_played - tracker.get('last_games', 0)
            examples_delta = worker.examples_generated - tracker.get('last_examples', 0)

            tracker['games_per_second'] = games_delta / elapsed
            tracker['examples_per_second'] = examples_delta / elapsed

        tracker['last_games'] = worker.games_played
        tracker['last_examples'] = worker.examples_generated
        tracker['last_time'] = now
        self._rate_tracker[worker_id] = tracker

    def report_error(self, worker_id: int, error: str) -> None:
        """Report an error from a worker.

        Parameters
        ----------
        worker_id : int
            Worker identifier
        error : str
            Error message
        """
        if worker_id in self._workers:
            worker = self._workers[worker_id]
            worker.error_count += 1
            worker.last_error = error
            logger.warning(f"Worker {worker_id} error: {error}")

    def check_workers(self) -> List[int]:
        """Check all workers and return dead worker IDs.

        Returns
        -------
        List[int]
            List of dead worker IDs
        """
        dead_workers = []
        now = time.time()

        for worker_id, stats in self._workers.items():
            if stats.status == WorkerStatus.STOPPED:
                continue

            time_since = stats.time_since_heartbeat()

            if time_since > self.heartbeat_timeout:
                if stats.status != WorkerStatus.DEAD:
                    stats.status = WorkerStatus.DEAD
                    dead_workers.append(worker_id)
                    logger.warning(
                        f"Worker {worker_id} marked dead "
                        f"(no heartbeat for {time_since:.1f}s)"
                    )

        self._last_check = now
        return dead_workers

    def get_worker_stats(self, worker_id: int) -> Optional[WorkerStats]:
        """Get stats for specific worker.

        Parameters
        ----------
        worker_id : int
            Worker identifier

        Returns
        -------
        Optional[WorkerStats]
            Worker stats or None
        """
        return self._workers.get(worker_id)

    def get_all_worker_stats(self) -> Dict[int, WorkerStats]:
        """Get stats for all workers.

        Returns
        -------
        Dict[int, WorkerStats]
            Worker ID -> stats
        """
        return dict(self._workers)

    def get_aggregate_stats(self) -> AggregateStats:
        """Get aggregate statistics across all workers.

        Returns
        -------
        AggregateStats
            Aggregate statistics
        """
        active = 0
        dead = 0
        total_games = 0
        total_examples = 0
        total_games_rate = 0.0
        total_examples_rate = 0.0

        for worker_id, stats in self._workers.items():
            if stats.status == WorkerStatus.RUNNING:
                active += 1
            elif stats.status == WorkerStatus.DEAD:
                dead += 1

            total_games += stats.games_played
            total_examples += stats.examples_generated

            tracker = self._rate_tracker.get(worker_id, {})
            total_games_rate += tracker.get('games_per_second', 0.0)
            total_examples_rate += tracker.get('examples_per_second', 0.0)

        return AggregateStats(
            total_workers=len(self._workers),
            active_workers=active,
            dead_workers=dead,
            total_games=total_games,
            total_examples=total_examples,
            games_per_second=total_games_rate,
            examples_per_second=total_examples_rate
        )

    def get_dead_workers(self) -> List[int]:
        """Get list of dead worker IDs.

        Returns
        -------
        List[int]
            Dead worker IDs
        """
        return [
            worker_id for worker_id, stats in self._workers.items()
            if stats.status == WorkerStatus.DEAD
        ]

    def get_active_workers(self) -> List[int]:
        """Get list of active worker IDs.

        Returns
        -------
        List[int]
            Active worker IDs
        """
        return [
            worker_id for worker_id, stats in self._workers.items()
            if stats.status == WorkerStatus.RUNNING
        ]

    def mark_worker_restarted(self, worker_id: int) -> None:
        """Mark a dead worker as restarting.

        Parameters
        ----------
        worker_id : int
            Worker identifier
        """
        if worker_id in self._workers:
            self._workers[worker_id].status = WorkerStatus.STARTING
            self._workers[worker_id].error_count = 0
            self._workers[worker_id].last_error = None
            logger.info(f"Worker {worker_id} marked for restart")

    def should_restart(self, worker_id: int, max_restarts: int = 3) -> bool:
        """Check if a dead worker should be restarted.

        Parameters
        ----------
        worker_id : int
            Worker identifier
        max_restarts : int
            Maximum restart attempts

        Returns
        -------
        bool
            True if worker should be restarted
        """
        if worker_id not in self._workers:
            return False

        stats = self._workers[worker_id]

        if stats.status != WorkerStatus.DEAD:
            return False

        if stats.error_count >= max_restarts:
            logger.warning(
                f"Worker {worker_id} exceeded max restarts ({max_restarts})"
            )
            return False

        return True

    def format_status_report(self) -> str:
        """Format a status report string.

        Returns
        -------
        str
            Formatted status report
        """
        agg = self.get_aggregate_stats()

        lines = [
            f"Workers: {agg.active_workers}/{agg.total_workers} active "
            f"({agg.dead_workers} dead)",
            f"Games: {agg.total_games} total ({agg.games_per_second:.2f}/s)",
            f"Examples: {agg.total_examples} total ({agg.examples_per_second:.2f}/s)",
            "",
            "Per-worker stats:"
        ]

        for worker_id, stats in sorted(self._workers.items()):
            status_icon = {
                WorkerStatus.RUNNING: "[OK]",
                WorkerStatus.DEAD: "[DEAD]",
                WorkerStatus.STARTING: "[...]",
                WorkerStatus.IDLE: "[IDLE]",
                WorkerStatus.STOPPED: "[STOP]",
                WorkerStatus.UNKNOWN: "[?]"
            }.get(stats.status, "[?]")

            lines.append(
                f"  {status_icon} Worker {worker_id}: "
                f"{stats.games_played} games, "
                f"{stats.examples_generated} examples, "
                f"model v{stats.model_version}"
            )

        return "\n".join(lines)

"""Distributed self-play coordinator.

This module implements the main coordinator process that manages
distributed self-play workers.
"""

from __future__ import annotations

import time
import logging
import yaml
from pathlib import Path
from typing import Dict, Any, List, Optional
import multiprocessing as mp

from core.distributed.worker import worker_main
from core.distributed.model_server import ModelManager, ModelNotification
from core.distributed.data_collector import DataCollector
from core.distributed.health_monitor import HealthMonitor, WorkerStatus
from core.self_play import TrainingExample

logger = logging.getLogger(__name__)


class SelfPlayCoordinator:
    """Coordinates distributed self-play across multiple workers.

    Responsibilities:
    - Spawn and manage worker processes
    - Distribute latest model to workers
    - Collect training data from workers
    - Handle worker failures and restarts

    Parameters
    ----------
    num_workers : int
        Number of worker processes (default 4)
    model_update_interval : int
        Games between model updates (default 100)
    config_path : str
        Path to configuration file
    config : Optional[Dict[str, Any]]
        Configuration dictionary (overrides file)
    """

    def __init__(
        self,
        num_workers: int = 4,
        model_update_interval: int = 100,
        config_path: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        self.num_workers = num_workers
        self.model_update_interval = model_update_interval

        # Load configuration
        self.config = self._load_config(config_path, config)

        # IPC queues
        self.model_queues: List[mp.Queue] = []
        self.data_queue: mp.Queue = mp.Queue(maxsize=1000)
        self.control_queues: List[mp.Queue] = []

        # Components
        self.model_manager = ModelManager(
            model_dir=self.config.get('model_dir', 'data/models/shared'),
            max_versions=self.config.get('max_versions', 5)
        )
        self.data_collector = DataCollector(
            data_queue=self.data_queue,
            buffer_size=self.config.get('buffer_size', 10000),
            persist_to_disk=self.config.get('persist_to_disk', True),
            persist_dir=self.config.get('persist_dir', 'data/selfplay_examples')
        )
        self.health_monitor = HealthMonitor(
            heartbeat_timeout=self.config.get('heartbeat_timeout', 60.0),
            check_interval=self.config.get('check_interval', 10.0)
        )

        # Workers
        self.workers: List[mp.Process] = []
        self.worker_configs: Dict[int, Dict[str, Any]] = {}

        # State
        self._running: bool = False
        self._total_games: int = 0
        self._games_since_update: int = 0

    def _load_config(
        self,
        config_path: Optional[str],
        config: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Load configuration from file and/or dict.

        Parameters
        ----------
        config_path : Optional[str]
            Path to YAML config file
        config : Optional[Dict[str, Any]]
            Configuration dictionary

        Returns
        -------
        Dict[str, Any]
            Merged configuration
        """
        result = {
            'mcts_simulations': 400,
            'board_size': 19,
            'komi': 7.5,
            'temperature_threshold': 30,
            'max_moves': 500,
            'batch_submit_size': 10,
            'heartbeat_interval': 30,
            'device': 'cpu',
            'model_dir': 'data/models/shared',
            'max_versions': 5,
            'buffer_size': 10000,
            'persist_to_disk': True,
            'persist_dir': 'data/selfplay_examples',
            'heartbeat_timeout': 60.0,
            'check_interval': 10.0,
            'auto_restart': True,
            'max_restart_attempts': 3
        }

        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                file_config = yaml.safe_load(f)
                if file_config:
                    # Extract distributed config section
                    if 'distributed' in file_config:
                        result.update(file_config['distributed'])
                    if 'resources' in file_config:
                        result.update(file_config['resources'])

        if config:
            result.update(config)

        return result

    def start(self) -> None:
        """Start coordinator and spawn workers."""
        if self._running:
            logger.warning("Coordinator already running")
            return

        logger.info(f"Starting coordinator with {self.num_workers} workers")
        self._running = True

        # Create queues for each worker
        for _ in range(self.num_workers):
            self.model_queues.append(mp.Queue(maxsize=10))
            self.control_queues.append(mp.Queue(maxsize=10))

        # Spawn workers
        for worker_id in range(self.num_workers):
            self._spawn_worker(worker_id)

        logger.info("All workers started")

    def stop(self, graceful: bool = True, timeout: float = 30.0) -> None:
        """Stop all workers and coordinator.

        Parameters
        ----------
        graceful : bool
            Wait for workers to finish current games
        timeout : float
            Maximum wait time for graceful shutdown
        """
        if not self._running:
            return

        logger.info("Stopping coordinator...")
        self._running = False

        # Send stop command to all workers
        for worker_id in range(len(self.control_queues)):
            try:
                self.control_queues[worker_id].put_nowait({'type': 'stop'})
            except Exception:
                pass

        if graceful:
            # Wait for workers to finish
            start_time = time.time()
            for worker in self.workers:
                remaining = timeout - (time.time() - start_time)
                if remaining > 0:
                    worker.join(timeout=remaining)

        # Force terminate remaining workers
        for worker in self.workers:
            if worker.is_alive():
                logger.warning(f"Force terminating worker {worker.pid}")
                worker.terminate()
                worker.join(timeout=5.0)

        # Collect any remaining data
        self.data_collector.collect(timeout=1.0)
        if self.data_collector.buffer_size_current() > 0:
            self.data_collector.persist_buffer()

        logger.info("Coordinator stopped")

    def _spawn_worker(self, worker_id: int) -> mp.Process:
        """Spawn a single worker process.

        Parameters
        ----------
        worker_id : int
            Worker identifier

        Returns
        -------
        mp.Process
            Worker process
        """
        worker_config = {
            'mcts_simulations': self.config.get('mcts_simulations', 400),
            'board_size': self.config.get('board_size', 19),
            'komi': self.config.get('komi', 7.5),
            'temperature_threshold': self.config.get('temperature_threshold', 30),
            'max_moves': self.config.get('max_moves', 500),
            'batch_submit_size': self.config.get('batch_submit_size', 10),
            'heartbeat_interval': self.config.get('heartbeat_interval', 30),
            'device': self.config.get('device', 'cpu')
        }

        self.worker_configs[worker_id] = worker_config

        process = mp.Process(
            target=worker_main,
            args=(
                worker_id,
                self.model_queues[worker_id],
                self.data_queue,
                self.control_queues[worker_id],
                worker_config
            ),
            daemon=True
        )
        process.start()

        self.workers.append(process)
        self.health_monitor.register_worker(worker_id)

        logger.info(f"Spawned worker {worker_id} (PID: {process.pid})")
        return process

    def update_model(
        self,
        model: 'torch.nn.Module',
        genome: Optional[Any] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Push new model to all workers.

        Parameters
        ----------
        model : torch.nn.Module
            New model
        genome : Optional[Any]
            Architecture genome
        metadata : Optional[Dict[str, Any]]
            Additional metadata
        """
        # Save model
        path = self.model_manager.save_model(model, genome, metadata)
        version = self.model_manager.get_model_version()

        # Notify all workers
        notification = {
            'type': 'model_update',
            'version': version,
            'path': path,
            'timestamp': time.time(),
            'metadata': metadata or {}
        }

        for worker_id, model_queue in enumerate(self.model_queues):
            try:
                model_queue.put_nowait(notification)
            except Exception as e:
                logger.warning(f"Failed to notify worker {worker_id}: {e}")

        logger.info(f"Distributed model v{version} to {len(self.model_queues)} workers")
        self._games_since_update = 0

    def collect_examples(
        self,
        min_examples: int,
        timeout: float = 60.0
    ) -> List[TrainingExample]:
        """Collect training examples from workers.

        Parameters
        ----------
        min_examples : int
            Minimum examples to collect
        timeout : float
            Maximum wait time

        Returns
        -------
        List[TrainingExample]
            Collected training examples
        """
        start_time = time.time()
        collected = 0

        while collected < min_examples:
            # Check timeout
            if time.time() - start_time > timeout:
                logger.warning(
                    f"Timeout collecting examples "
                    f"(got {collected}/{min_examples})"
                )
                break

            # Collect from queue
            new_collected = self._process_data_queue(timeout=1.0)
            collected += new_collected

            # Check worker health
            self._check_workers()

        examples = self.data_collector.get_all_examples(clear=True)
        return examples

    def _process_data_queue(self, timeout: float = 0.1) -> int:
        """Process items from data queue.

        Parameters
        ----------
        timeout : float
            Queue timeout

        Returns
        -------
        int
            Number of examples collected
        """
        collected = 0

        while True:
            try:
                item = self.data_queue.get(timeout=timeout)

                if item is None:
                    continue

                item_type = item.get('type', '')

                if item_type == 'heartbeat':
                    self._handle_heartbeat(item)

                elif item_type == 'examples':
                    examples = item.get('examples', [])
                    self.data_collector._buffer.extend(examples)
                    collected += len(examples)
                    self._total_games += len(item.get('game_indices', []))
                    self._games_since_update += len(item.get('game_indices', []))

                    # Update health monitor
                    worker_id = item.get('worker_id', -1)
                    self.health_monitor.update_heartbeat(worker_id, {
                        'games_played': self._total_games,
                        'examples_generated': collected,
                        'model_version': item.get('model_version', 0)
                    })

            except Exception:
                break

        return collected

    def _handle_heartbeat(self, heartbeat: Dict[str, Any]) -> None:
        """Handle worker heartbeat.

        Parameters
        ----------
        heartbeat : Dict[str, Any]
            Heartbeat data
        """
        worker_id = heartbeat.get('worker_id', -1)
        self.health_monitor.update_heartbeat(worker_id, heartbeat)

    def _check_workers(self) -> None:
        """Check worker health and restart dead workers."""
        dead_workers = self.health_monitor.check_workers()

        if not dead_workers:
            return

        if not self.config.get('auto_restart', True):
            return

        max_restarts = self.config.get('max_restart_attempts', 3)

        for worker_id in dead_workers:
            if self.health_monitor.should_restart(worker_id, max_restarts):
                self._restart_worker(worker_id)

    def _restart_worker(self, worker_id: int) -> None:
        """Restart a dead worker.

        Parameters
        ----------
        worker_id : int
            Worker to restart
        """
        logger.info(f"Restarting worker {worker_id}")

        # Terminate old process if still running
        if worker_id < len(self.workers):
            old_process = self.workers[worker_id]
            if old_process.is_alive():
                old_process.terminate()
                old_process.join(timeout=5.0)

        # Create new queues
        self.model_queues[worker_id] = mp.Queue(maxsize=10)
        self.control_queues[worker_id] = mp.Queue(maxsize=10)

        # Spawn new worker
        config = self.worker_configs.get(worker_id, self.config)
        process = mp.Process(
            target=worker_main,
            args=(
                worker_id,
                self.model_queues[worker_id],
                self.data_queue,
                self.control_queues[worker_id],
                config
            ),
            daemon=True
        )
        process.start()

        self.workers[worker_id] = process
        self.health_monitor.mark_worker_restarted(worker_id)

        # Send current model to new worker
        if self.model_manager.model_exists():
            notification = {
                'type': 'model_update',
                'version': self.model_manager.get_model_version(),
                'path': self.model_manager.get_current_model_path(),
                'timestamp': time.time()
            }
            try:
                self.model_queues[worker_id].put_nowait(notification)
            except Exception:
                pass

        logger.info(f"Worker {worker_id} restarted (PID: {process.pid})")

    def get_stats(self) -> Dict[str, Any]:
        """Get coordinator statistics.

        Returns
        -------
        Dict[str, Any]
            Statistics dictionary
        """
        agg_stats = self.health_monitor.get_aggregate_stats()

        return {
            'running': self._running,
            'num_workers': self.num_workers,
            'active_workers': agg_stats.active_workers,
            'dead_workers': agg_stats.dead_workers,
            'total_games': self._total_games,
            'total_examples': agg_stats.total_examples,
            'games_per_second': agg_stats.games_per_second,
            'examples_per_second': agg_stats.examples_per_second,
            'buffer_size': self.data_collector.buffer_size_current(),
            'model_version': self.model_manager.get_model_version(),
            'games_since_update': self._games_since_update
        }

    def pause_workers(self) -> None:
        """Pause all workers."""
        for control_queue in self.control_queues:
            try:
                control_queue.put_nowait({'type': 'pause'})
            except Exception:
                pass
        logger.info("All workers paused")

    def resume_workers(self) -> None:
        """Resume all workers."""
        for control_queue in self.control_queues:
            try:
                control_queue.put_nowait({'type': 'resume'})
            except Exception:
                pass
        logger.info("All workers resumed")

    def update_worker_config(self, config: Dict[str, Any]) -> None:
        """Update configuration for all workers.

        Parameters
        ----------
        config : Dict[str, Any]
            New configuration values
        """
        cmd = {'type': 'update_config', 'config': config}

        for control_queue in self.control_queues:
            try:
                control_queue.put_nowait(cmd)
            except Exception:
                pass

        logger.info(f"Updated worker config: {config}")

    def format_status(self) -> str:
        """Format status report.

        Returns
        -------
        str
            Formatted status string
        """
        stats = self.get_stats()
        health_report = self.health_monitor.format_status_report()

        return (
            f"=== Coordinator Status ===\n"
            f"Running: {stats['running']}\n"
            f"Model Version: {stats['model_version']}\n"
            f"Total Games: {stats['total_games']}\n"
            f"Buffer Size: {stats['buffer_size']}\n"
            f"\n{health_report}"
        )

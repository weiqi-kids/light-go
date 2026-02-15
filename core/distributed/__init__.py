"""Distributed self-play system for Light-Go.

This package implements distributed training with multiple self-play workers:
- SelfPlayCoordinator: Main coordinator process
- DistributedSelfPlayWorker: Worker process for self-play
- ModelManager: Model versioning and distribution
- DataCollector: Training data aggregation
- HealthMonitor: Worker health monitoring
"""

from core.distributed.coordinator import SelfPlayCoordinator
from core.distributed.worker import DistributedSelfPlayWorker, worker_main
from core.distributed.model_server import ModelManager
from core.distributed.data_collector import DataCollector
from core.distributed.health_monitor import HealthMonitor, WorkerStats

__all__ = [
    'SelfPlayCoordinator',
    'DistributedSelfPlayWorker',
    'worker_main',
    'ModelManager',
    'DataCollector',
    'HealthMonitor',
    'WorkerStats'
]

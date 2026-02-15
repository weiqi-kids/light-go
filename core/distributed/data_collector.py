"""Training data collection and aggregation for distributed self-play.

This module collects training examples from multiple workers and provides
them to the training loop.
"""

from __future__ import annotations

import time
import logging
import queue
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
import multiprocessing as mp
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ExampleBatch:
    """Batch of training examples from a worker.

    Attributes
    ----------
    worker_id : int
        ID of worker that generated examples
    examples : List[Any]
        List of TrainingExample objects
    game_indices : List[int]
        Game indices for these examples
    timestamp : float
        Time when batch was created
    model_version : int
        Model version used to generate these examples
    """

    worker_id: int
    examples: List[Any]
    game_indices: List[int]
    timestamp: float
    model_version: int


@dataclass
class CollectorStats:
    """Statistics for data collector.

    Attributes
    ----------
    total_collected : int
        Total examples collected
    examples_per_worker : Dict[int, int]
        Examples collected per worker
    games_per_worker : Dict[int, int]
        Games completed per worker
    collection_rate : float
        Examples per second
    """

    total_collected: int = 0
    examples_per_worker: Dict[int, int] = field(default_factory=dict)
    games_per_worker: Dict[int, int] = field(default_factory=dict)
    collection_rate: float = 0.0
    last_collection_time: float = field(default_factory=time.time)


class DataCollector:
    """Collects and aggregates training data from workers.

    Features:
    - Queue-based collection
    - Automatic batching
    - Optional disk persistence
    - Statistics tracking

    Parameters
    ----------
    data_queue : mp.Queue
        Queue for receiving data from workers
    buffer_size : int
        Maximum examples to buffer (default 10000)
    persist_to_disk : bool
        Whether to persist data to disk (default True)
    persist_dir : str
        Directory for persistence (default "data/selfplay_examples")
    """

    def __init__(
        self,
        data_queue: mp.Queue,
        buffer_size: int = 10000,
        persist_to_disk: bool = True,
        persist_dir: str = "data/selfplay_examples"
    ):
        self.data_queue = data_queue
        self.buffer_size = buffer_size
        self.persist_to_disk = persist_to_disk
        self.persist_dir = Path(persist_dir)

        if persist_to_disk:
            self.persist_dir.mkdir(parents=True, exist_ok=True)

        # Internal buffer
        self._buffer: List[Any] = []
        self._stats = CollectorStats()
        self._batch_counter = 0

    def collect(self, timeout: float = 0.1) -> int:
        """Collect available examples from queue.

        Non-blocking collection with timeout.

        Parameters
        ----------
        timeout : float
            Queue timeout in seconds

        Returns
        -------
        int
            Number of examples collected
        """
        collected = 0

        while True:
            try:
                batch = self.data_queue.get(timeout=timeout)

                if batch is None:
                    # Sentinel value, skip
                    continue

                if isinstance(batch, ExampleBatch):
                    self._process_batch(batch)
                    collected += len(batch.examples)
                elif isinstance(batch, dict):
                    # Handle dict format
                    self._process_dict_batch(batch)
                    collected += len(batch.get('examples', []))
                else:
                    # Assume it's a list of examples
                    self._buffer.extend(batch)
                    collected += len(batch)

            except queue.Empty:
                break

        # Update collection rate
        now = time.time()
        elapsed = now - self._stats.last_collection_time
        if elapsed > 0:
            self._stats.collection_rate = collected / elapsed
        self._stats.last_collection_time = now

        return collected

    def _process_batch(self, batch: ExampleBatch) -> None:
        """Process an ExampleBatch.

        Parameters
        ----------
        batch : ExampleBatch
            Batch to process
        """
        # Add examples to buffer
        self._buffer.extend(batch.examples)

        # Update stats
        worker_id = batch.worker_id
        self._stats.total_collected += len(batch.examples)
        self._stats.examples_per_worker[worker_id] = \
            self._stats.examples_per_worker.get(worker_id, 0) + len(batch.examples)
        self._stats.games_per_worker[worker_id] = \
            self._stats.games_per_worker.get(worker_id, 0) + len(batch.game_indices)

        # Trim buffer if too large
        if len(self._buffer) > self.buffer_size:
            self._buffer = self._buffer[-self.buffer_size:]

    def _process_dict_batch(self, batch: Dict[str, Any]) -> None:
        """Process a dictionary batch.

        Parameters
        ----------
        batch : Dict[str, Any]
            Batch dict with 'examples', 'worker_id', etc.
        """
        examples = batch.get('examples', [])
        worker_id = batch.get('worker_id', -1)

        self._buffer.extend(examples)

        self._stats.total_collected += len(examples)
        self._stats.examples_per_worker[worker_id] = \
            self._stats.examples_per_worker.get(worker_id, 0) + len(examples)

        if len(self._buffer) > self.buffer_size:
            self._buffer = self._buffer[-self.buffer_size:]

    def get_examples(
        self,
        count: int,
        remove: bool = True
    ) -> List[Any]:
        """Get examples from buffer.

        Parameters
        ----------
        count : int
            Number of examples to get
        remove : bool
            Whether to remove from buffer (default True)

        Returns
        -------
        List[Any]
            Training examples
        """
        count = min(count, len(self._buffer))

        if remove:
            examples = self._buffer[:count]
            self._buffer = self._buffer[count:]
        else:
            examples = self._buffer[:count]

        return examples

    def get_all_examples(self, clear: bool = True) -> List[Any]:
        """Get all buffered examples.

        Parameters
        ----------
        clear : bool
            Whether to clear buffer (default True)

        Returns
        -------
        List[Any]
            All training examples
        """
        examples = self._buffer.copy()

        if clear:
            self._buffer.clear()

        return examples

    def get_stats(self) -> CollectorStats:
        """Get collection statistics.

        Returns
        -------
        CollectorStats
            Current statistics
        """
        return self._stats

    def buffer_size_current(self) -> int:
        """Get current buffer size.

        Returns
        -------
        int
            Number of examples in buffer
        """
        return len(self._buffer)

    def persist_buffer(self) -> str:
        """Save current buffer to disk.

        Returns
        -------
        str
            Path to saved file
        """
        if not self._buffer:
            return ""

        self._batch_counter += 1
        timestamp = int(time.time())
        filename = f"examples_{timestamp}_{self._batch_counter}.npz"
        filepath = self.persist_dir / filename

        # Extract numpy arrays
        features_list = []
        policies_list = []
        values_list = []

        for ex in self._buffer:
            features_list.append(ex.features)
            policies_list.append(ex.policy)
            values_list.append(ex.value)

        np.savez_compressed(
            filepath,
            features=np.stack(features_list, axis=0),
            policies=np.stack(policies_list, axis=0),
            values=np.array(values_list, dtype=np.float32)
        )

        logger.info(f"Persisted {len(self._buffer)} examples to {filepath}")
        return str(filepath)

    def clear(self) -> None:
        """Clear the buffer."""
        self._buffer.clear()

    def reset_stats(self) -> None:
        """Reset collection statistics."""
        self._stats = CollectorStats()


def create_data_collector(
    data_queue: mp.Queue,
    config: Dict[str, Any]
) -> DataCollector:
    """Create data collector from configuration.

    Parameters
    ----------
    data_queue : mp.Queue
        Data queue
    config : Dict[str, Any]
        Configuration dictionary

    Returns
    -------
    DataCollector
        Configured collector
    """
    return DataCollector(
        data_queue=data_queue,
        buffer_size=config.get('buffer_size', 10000),
        persist_to_disk=config.get('persist_to_disk', True),
        persist_dir=config.get('persist_dir', 'data/selfplay_examples')
    )

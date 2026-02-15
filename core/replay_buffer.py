"""Experience replay buffer for reinforcement learning.

This module implements a persistent replay buffer that stores training examples
across generations, enabling more stable and sample-efficient learning.
"""

from __future__ import annotations

import os
import time
import logging
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from collections import deque

logger = logging.getLogger(__name__)


@dataclass
class BufferExample:
    """Training example with metadata for replay buffer.

    Attributes
    ----------
    features : np.ndarray
        Board features (2, board_size, board_size)
    policy : np.ndarray
        MCTS policy targets (board_size^2 + 1)
    value : float
        Game outcome from player's perspective
    generation : int
        Generation when this example was created
    timestamp : float
        Unix timestamp when example was added
    priority : float
        Priority weight for sampling
    """

    features: np.ndarray
    policy: np.ndarray
    value: float
    generation: int = 0
    timestamp: float = field(default_factory=time.time)
    priority: float = 1.0


class ReplayBuffer:
    """Persistent experience replay buffer for RL training.

    Key Features:
    - Circular buffer with configurable max size
    - Disk persistence (NPZ format)
    - Priority sampling support
    - Cross-generation persistence
    - Generation-based statistics

    Parameters
    ----------
    max_size : int
        Maximum number of examples to store (default 500,000)
    buffer_path : str
        Directory path for persistence (default "data/replay_buffer")
    priority_alpha : float
        Priority exponent for sampling (default 0.6)
    min_examples_to_sample : int
        Minimum examples required before sampling (default 1000)
    """

    def __init__(
        self,
        max_size: int = 500_000,
        buffer_path: str = "data/replay_buffer",
        priority_alpha: float = 0.6,
        min_examples_to_sample: int = 1000
    ):
        self.max_size = max_size
        self.buffer_path = Path(buffer_path)
        self.priority_alpha = priority_alpha
        self.min_examples_to_sample = min_examples_to_sample

        # Internal storage
        self._buffer: deque = deque(maxlen=max_size)
        self._priorities: List[float] = []
        self._generation_counts: Dict[int, int] = {}
        self._total_added: int = 0

        # Create buffer directory
        self.buffer_path.mkdir(parents=True, exist_ok=True)

    def __len__(self) -> int:
        """Return number of examples in buffer."""
        return len(self._buffer)

    @property
    def is_ready(self) -> bool:
        """Check if buffer has enough examples for sampling."""
        return len(self._buffer) >= self.min_examples_to_sample

    def add_example(
        self,
        features: np.ndarray,
        policy: np.ndarray,
        value: float,
        generation: int = 0,
        priority: float = 1.0
    ) -> None:
        """Add a single training example to the buffer.

        Parameters
        ----------
        features : np.ndarray
            Board features
        policy : np.ndarray
            MCTS policy targets
        value : float
            Game outcome
        generation : int
            Generation number
        priority : float
            Initial priority weight
        """
        example = BufferExample(
            features=features,
            policy=policy,
            value=value,
            generation=generation,
            priority=priority
        )

        # Handle overflow (deque handles this automatically with maxlen)
        if len(self._buffer) == self.max_size:
            # Track removed generation
            removed = self._buffer[0]
            old_gen = removed.generation
            if old_gen in self._generation_counts:
                self._generation_counts[old_gen] -= 1
                if self._generation_counts[old_gen] <= 0:
                    del self._generation_counts[old_gen]

        self._buffer.append(example)
        self._priorities.append(priority)

        # Maintain priorities list size
        if len(self._priorities) > self.max_size:
            self._priorities.pop(0)

        # Update generation counts
        self._generation_counts[generation] = self._generation_counts.get(generation, 0) + 1
        self._total_added += 1

    def add_examples(
        self,
        examples: List[Any],
        generation: int = 0
    ) -> int:
        """Add multiple training examples to the buffer.

        Parameters
        ----------
        examples : List[Any]
            List of TrainingExample objects (must have features, policy, value)
        generation : int
            Generation number for all examples

        Returns
        -------
        int
            Number of examples added
        """
        count = 0
        for ex in examples:
            self.add_example(
                features=ex.features,
                policy=ex.policy,
                value=ex.value,
                generation=generation
            )
            count += 1

        logger.info(f"Added {count} examples from generation {generation} to replay buffer")
        return count

    def sample_batch(
        self,
        batch_size: int,
        prioritized: bool = False
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Sample a batch of training examples.

        Parameters
        ----------
        batch_size : int
            Number of examples to sample
        prioritized : bool
            Whether to use priority-based sampling

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray]
            (features, policies, values) arrays
        """
        if len(self._buffer) == 0:
            raise ValueError("Cannot sample from empty buffer")

        batch_size = min(batch_size, len(self._buffer))

        if prioritized and self._priorities:
            # Priority-based sampling
            priorities = np.array(self._priorities[-len(self._buffer):])
            priorities = priorities ** self.priority_alpha
            probs = priorities / priorities.sum()
            indices = np.random.choice(len(self._buffer), size=batch_size, p=probs, replace=False)
        else:
            # Uniform sampling
            indices = np.random.choice(len(self._buffer), size=batch_size, replace=False)

        # Extract batch
        features_list = []
        policies_list = []
        values_list = []

        buffer_list = list(self._buffer)
        for idx in indices:
            ex = buffer_list[idx]
            features_list.append(ex.features)
            policies_list.append(ex.policy)
            values_list.append(ex.value)

        features = np.stack(features_list, axis=0)
        policies = np.stack(policies_list, axis=0)
        values = np.array(values_list, dtype=np.float32)

        return features, policies, values

    def sample_batch_as_dicts(
        self,
        batch_size: int,
        prioritized: bool = False
    ) -> List[Dict[str, Any]]:
        """Sample batch as list of dictionaries (for GoDataset compatibility).

        Parameters
        ----------
        batch_size : int
            Number of examples to sample
        prioritized : bool
            Whether to use priority-based sampling

        Returns
        -------
        List[Dict[str, Any]]
            List of {'position': features, 'policy': policy, 'value': value}
        """
        features, policies, values = self.sample_batch(batch_size, prioritized)

        return [
            {
                'position': features[i],
                'policy': policies[i],
                'value': values[i]
            }
            for i in range(len(values))
        ]

    def update_priorities(
        self,
        indices: List[int],
        priorities: List[float]
    ) -> None:
        """Update priorities for specific examples.

        Parameters
        ----------
        indices : List[int]
            Indices of examples to update
        priorities : List[float]
            New priority values
        """
        for idx, priority in zip(indices, priorities):
            if 0 <= idx < len(self._priorities):
                self._priorities[idx] = priority

    def save_to_disk(self, filename: Optional[str] = None) -> str:
        """Persist buffer to disk in NPZ format.

        Parameters
        ----------
        filename : Optional[str]
            Custom filename (default: auto-generated with timestamp)

        Returns
        -------
        str
            Path to saved file
        """
        if len(self._buffer) == 0:
            logger.warning("Cannot save empty buffer")
            return ""

        if filename is None:
            timestamp = int(time.time())
            filename = f"replay_buffer_{timestamp}.npz"

        filepath = self.buffer_path / filename

        # Extract all data
        features_list = []
        policies_list = []
        values_list = []
        generations_list = []
        timestamps_list = []
        priorities_list = []

        for ex in self._buffer:
            features_list.append(ex.features)
            policies_list.append(ex.policy)
            values_list.append(ex.value)
            generations_list.append(ex.generation)
            timestamps_list.append(ex.timestamp)
            priorities_list.append(ex.priority)

        # Save as compressed NPZ
        np.savez_compressed(
            filepath,
            features=np.stack(features_list, axis=0),
            policies=np.stack(policies_list, axis=0),
            values=np.array(values_list, dtype=np.float32),
            generations=np.array(generations_list, dtype=np.int32),
            timestamps=np.array(timestamps_list, dtype=np.float64),
            priorities=np.array(priorities_list, dtype=np.float32)
        )

        logger.info(f"Saved {len(self._buffer)} examples to {filepath}")
        return str(filepath)

    def load_from_disk(self, filepath: str) -> int:
        """Load buffer from disk.

        Parameters
        ----------
        filepath : str
            Path to NPZ file

        Returns
        -------
        int
            Number of examples loaded
        """
        if not os.path.exists(filepath):
            logger.warning(f"File not found: {filepath}")
            return 0

        data = np.load(filepath)

        features = data['features']
        policies = data['policies']
        values = data['values']
        generations = data.get('generations', np.zeros(len(values), dtype=np.int32))
        timestamps = data.get('timestamps', np.full(len(values), time.time()))
        priorities = data.get('priorities', np.ones(len(values), dtype=np.float32))

        count = 0
        for i in range(len(values)):
            example = BufferExample(
                features=features[i],
                policy=policies[i],
                value=float(values[i]),
                generation=int(generations[i]),
                timestamp=float(timestamps[i]),
                priority=float(priorities[i])
            )
            self._buffer.append(example)
            self._priorities.append(example.priority)
            self._generation_counts[example.generation] = \
                self._generation_counts.get(example.generation, 0) + 1
            count += 1

        logger.info(f"Loaded {count} examples from {filepath}")
        return count

    def load_latest(self) -> int:
        """Load the most recent buffer file from disk.

        Returns
        -------
        int
            Number of examples loaded
        """
        npz_files = list(self.buffer_path.glob("replay_buffer_*.npz"))
        if not npz_files:
            logger.info("No existing buffer files found")
            return 0

        # Sort by modification time, get latest
        latest = max(npz_files, key=lambda p: p.stat().st_mtime)
        return self.load_from_disk(str(latest))

    def clear(self) -> None:
        """Clear all examples from buffer."""
        self._buffer.clear()
        self._priorities.clear()
        self._generation_counts.clear()
        logger.info("Replay buffer cleared")

    def get_stats(self) -> Dict[str, Any]:
        """Get buffer statistics.

        Returns
        -------
        Dict[str, Any]
            Statistics dictionary
        """
        if len(self._buffer) == 0:
            return {
                "size": 0,
                "max_size": self.max_size,
                "utilization": 0.0,
                "total_added": self._total_added,
                "is_ready": False,
                "generations": {},
                "oldest_generation": None,
                "newest_generation": None
            }

        generations = sorted(self._generation_counts.keys())

        return {
            "size": len(self._buffer),
            "max_size": self.max_size,
            "utilization": len(self._buffer) / self.max_size,
            "total_added": self._total_added,
            "is_ready": self.is_ready,
            "generations": dict(self._generation_counts),
            "num_generations": len(generations),
            "oldest_generation": generations[0] if generations else None,
            "newest_generation": generations[-1] if generations else None
        }

    def get_generation_distribution(self) -> Dict[int, float]:
        """Get distribution of examples across generations.

        Returns
        -------
        Dict[int, float]
            Generation -> fraction of buffer
        """
        total = len(self._buffer)
        if total == 0:
            return {}

        return {
            gen: count / total
            for gen, count in self._generation_counts.items()
        }


# Convenience function for creating buffer with config
def create_replay_buffer(config: Dict[str, Any]) -> ReplayBuffer:
    """Create replay buffer from configuration dictionary.

    Parameters
    ----------
    config : Dict[str, Any]
        Configuration with keys: max_size, persistence_path, priority_alpha

    Returns
    -------
    ReplayBuffer
        Configured replay buffer
    """
    return ReplayBuffer(
        max_size=config.get('max_size', 500_000),
        buffer_path=config.get('persistence_path', 'data/replay_buffer'),
        priority_alpha=config.get('priority_alpha', 0.6),
        min_examples_to_sample=config.get('min_examples_to_sample', 1000)
    )

"""Automatic discovery and training allocation utilities for strategies."""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from input.sgf_to_input import parse_sgf
from core.liberty import count_liberties
from .strategy_manager import StrategyManager
from .sample_strategy import SampleGoStrategy


class AutoLearner:
    """Manage strategy discovery and dynamic training allocation.

    The class keeps track of individual strategy performance and decides which
    strategy should receive additional training for a given board position.  It
    provides utilities to create new strategies, assign training examples and
    update scores based on feedback from the :class:`StrategyManager` or other
    external evaluation processes.
    """

    def __init__(self, manager: StrategyManager) -> None:
        """Initialize with an existing :class:`StrategyManager`."""
        self.manager = manager
        # Map strategy name to average performance score
        self._scores: Dict[str, float] = {
            name: 0.0 for name in self.manager.list_strategies()
        }
        # Allocation weights used when selecting strategies for training
        self._allocation: Dict[str, float] = {
            name: 1.0 / len(self._scores) if self._scores else 0.0
            for name in self._scores
        }

    # ------------------------------------------------------------------
    # Utilities working with SGF data.  These remain for backward
    # compatibility with the simplified engine used in tests.
    # ------------------------------------------------------------------
    @staticmethod
    def _game_stats(matrix: List[List[int]]) -> Dict[str, float]:
        """Return simple statistics derived from a board matrix."""
        libs = count_liberties(matrix)
        black_libs = [abs(l) for _, _, l in libs if l > 0]
        white_libs = [abs(l) for _, _, l in libs if l < 0]
        return {
            "black_stones": len(black_libs),
            "white_stones": len(white_libs),
            "avg_liberties_black": sum(black_libs) / len(black_libs)
            if black_libs
            else 0.0,
            "avg_liberties_white": sum(white_libs) / len(white_libs)
            if white_libs
            else 0.0,
        }

    def train(self, data_path: str) -> Dict[str, Any]:
        """Aggregate statistics from all SGF files located in ``data_path``."""
        stats = {
            "games": 0,
            "total_black_stones": 0,
            "total_white_stones": 0,
            "avg_liberties_black": 0.0,
            "avg_liberties_white": 0.0,
        }
        for fname in os.listdir(data_path):
            if not fname.endswith(".sgf"):
                continue
            matrix, _, _ = parse_sgf(os.path.join(data_path, fname))
            gs = self._game_stats(matrix)
            stats["games"] += 1
            stats["total_black_stones"] += gs["black_stones"]
            stats["total_white_stones"] += gs["white_stones"]
            stats["avg_liberties_black"] += gs["avg_liberties_black"]
            stats["avg_liberties_white"] += gs["avg_liberties_white"]
        if stats["games"]:
            stats["avg_liberties_black"] /= stats["games"]
            stats["avg_liberties_white"] /= stats["games"]
        return stats

    def train_and_save(self, data_path: str) -> str:
        """Train from ``data_path`` and save the resulting strategy."""
        stats = self.train(data_path)
        name = self._next_strategy_name()
        strategy = SampleGoStrategy(name=name, stats=stats)
        self.manager.save_strategy(name, strategy)
        self._scores[name] = 0.0
        self._allocation[name] = self._default_allocation()
        self._normalize_allocation()
        return name

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def discover_strategy(
        self, data: Dict[str, Any], model: Optional[Dict[str, Any]] = None
    ) -> str:
        """Register ``model`` (or ``data``) as a brand new strategy."""

        name = self._next_strategy_name()
        strategy = SampleGoStrategy(name=name, stats=model or data)
        self.manager.save_strategy(name, strategy)
        self._scores[name] = 0.0
        self._allocation[name] = self._default_allocation()
        self._normalize_allocation()
        return name

    def assign_training(self, board_features: Dict[str, Any]) -> List[str]:
        """Return the list of strategies that should receive ``board_features``.

        The current implementation selects strategies based on the allocation
        weights computed from their performance scores.
        """
        if not self._allocation:
            return []
        threshold = 0.1
        selected = [
            name for name, weight in self._allocation.items() if weight >= threshold
        ]
        if not selected:
            # Fallback to the best scoring strategy
            if self._scores:
                best = max(self._scores, key=self._scores.get)
                selected = [best]
        return selected

    def receive_feedback(self, strategy_name: str, score: float) -> None:
        """Update internal statistics using performance ``score``."""
        if strategy_name not in self._scores:
            self._scores[strategy_name] = score
        else:
            old = self._scores[strategy_name]
            self._scores[strategy_name] = old * 0.9 + score * 0.1
        self._adjust_training_allocation()
        self._drop_weak_strategies()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _next_strategy_name(self) -> str:
        """Return a unique one-letter strategy name."""
        existing = self.manager.list_strategies()
        for idx in range(26):
            candidate = chr(ord("a") + idx)
            if candidate not in existing:
                return candidate
        return f"s{len(existing)}"

    def _default_allocation(self) -> float:
        """Return the default allocation weight for a new strategy."""
        return 1.0 / max(len(self._scores), 1)

    def _normalize_allocation(self) -> None:
        """Normalize allocation weights so they sum to one."""
        total = sum(self._allocation.values())
        if not total:
            return
        for name in self._allocation:
            self._allocation[name] /= total

    def _adjust_training_allocation(self) -> None:
        """Update allocation weights based on current scores."""
        total = sum(max(score, 0.0) for score in self._scores.values())
        if total == 0:
            val = self._default_allocation()
            for name in self._allocation:
                self._allocation[name] = val
            return
        for name, score in self._scores.items():
            self._allocation[name] = max(score, 0.0) / total

    def _drop_weak_strategies(self) -> None:
        """Remove poorly performing strategies from tracking."""
        if len(self._scores) <= 1:
            return
        threshold = -0.5
        to_drop = [name for name, score in self._scores.items() if score < threshold]
        for name in to_drop:
            self._scores.pop(name, None)
            self._allocation.pop(name, None)
            # Actual file removal is left to higher level tools
        if to_drop:
            self._normalize_allocation()


__all__ = [
    "AutoLearner",
]

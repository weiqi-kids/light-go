"""Simplified automatic learning utilities."""
from __future__ import annotations

import os
from typing import Dict

from input.sgf_to_input import parse_sgf
from core.liberty import count_liberties
from .strategy_manager import StrategyManager


class AutoLearner:
    """A very small learner that derives statistics from SGF files."""

    def __init__(self, manager: StrategyManager) -> None:
        self.manager = manager

    @staticmethod
    def _game_stats(board):
        libs = count_liberties(board)
        black_libs = [abs(l) for _, _, l in libs if l > 0]
        white_libs = [abs(l) for _, _, l in libs if l < 0]
        return {
            "black_stones": len(black_libs),
            "white_stones": len(white_libs),
            "avg_liberties_black": sum(black_libs) / len(black_libs) if black_libs else 0.0,
            "avg_liberties_white": sum(white_libs) / len(white_libs) if white_libs else 0.0,
        }

    def train(self, data_path: str) -> Dict:
        """Compute simple aggregated statistics from all SGF files."""
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
            board, _ = parse_sgf(os.path.join(data_path, fname))
            gs = self._game_stats(board)
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
        """Train from ``data_path`` and store a new strategy."""
        model_data = self.train(data_path)
        existing = self.manager.list_strategies()
        next_name = chr(ord("a") + len(existing))
        self.manager.save_strategy(next_name, model_data)
        return next_name

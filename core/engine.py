"""Main engine tying together strategy management and learning."""
from __future__ import annotations

from .strategy_manager import StrategyManager
from .auto_learner import AutoLearner


class Engine:
    """Facade for high level operations."""

    def __init__(self, model_dir: str) -> None:
        self.strategy_manager = StrategyManager(model_dir)
        self.auto_learner = AutoLearner(self.strategy_manager)

    def train(self, data_dir: str) -> str:
        """Train a new strategy from SGF files in ``data_dir``."""
        return self.auto_learner.train_and_save(data_dir)

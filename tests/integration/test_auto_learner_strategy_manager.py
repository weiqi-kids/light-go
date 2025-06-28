import tempfile

import pytest

from core.auto_learner import AutoLearner
from core.strategy_manager import StrategyManager


def test_auto_learner_strategy_manager_integration():
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = StrategyManager(tmpdir)
        learner = AutoLearner(manager)

        primary = learner.discover_strategy({"dummy": 1})
        secondary = learner.discover_strategy({"dummy": 2})

        board_features = {"board": []}
        selected_before = set(learner.assign_training(board_features))
        assert primary in selected_before
        alloc_before = dict(learner._allocation)

        learner.receive_feedback(primary, 1.0)

        selected_after = set(learner.assign_training(board_features))
        assert primary in selected_after
        assert secondary not in selected_after
        alloc_after = learner._allocation
        # Allocation for the winning strategy should increase
        assert alloc_after[primary] > alloc_before[primary]
        assert alloc_after[secondary] < alloc_before[secondary]

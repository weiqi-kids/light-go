import tempfile

import pytest

from core.auto_learner import AutoLearner
from core.strategy_manager import StrategyManager


def test_auto_learner_strategy_manager_integration():
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = StrategyManager(tmpdir)
        learner = AutoLearner(manager)

        strategy_name = learner.discover_strategy({'dummy': 1})
        assert strategy_name in manager.list_strategies()

        board_features = {'board': []}
        selected_before = learner.assign_training(board_features)
        assert strategy_name in selected_before
        alloc_before = learner._allocation[strategy_name]

        learner.receive_feedback(strategy_name, 1.0)

        selected_after = learner.assign_training(board_features)
        assert strategy_name in selected_after
        alloc_after = learner._allocation[strategy_name]
        assert selected_after == selected_before
        assert alloc_after != pytest.approx(alloc_before)

import sys
import pathlib
from unittest.mock import Mock

# Add project root to path
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from core.auto_learner import AutoLearner
from core.strategy_manager import StrategyManager


def _build_manager(existing=None):
    if existing is None:
        existing = []
    manager = Mock(spec=StrategyManager)
    manager.list_strategies.return_value = list(existing)
    return manager


def test_discover_strategy_registers_new_strategy(tmp_path):
    manager = _build_manager(["a"])
    learner = AutoLearner(manager)

    new_data = {"foo": 1}
    new_name = learner.discover_strategy(new_data)

    manager.save_strategy.assert_called_once_with(new_name, new_data)
    assert new_name in learner._scores
    assert new_name in learner._allocation
    assert abs(sum(learner._allocation.values()) - 1.0) < 1e-6


def test_assign_training_prefers_high_score():
    manager = _build_manager(["a", "b"])
    learner = AutoLearner(manager)

    learner.receive_feedback("a", 1.0)
    targets = learner.assign_training({})

    assert targets == ["a"]


def test_receive_feedback_adjusts_allocation():
    manager = _build_manager(["a", "b"])
    learner = AutoLearner(manager)

    initial = dict(learner._allocation)

    learner.receive_feedback("a", 1.0)

    assert learner._allocation["a"] > initial["a"]
    assert learner._allocation["b"] < initial["b"]


def test_drop_weak_strategy_after_negative_feedback():
    manager = _build_manager(["a", "b"])
    learner = AutoLearner(manager)

    for _ in range(7):
        learner.receive_feedback("b", -1.0)

    assert "b" not in learner._scores
    assert "b" not in learner._allocation
    assert list(learner._allocation.keys()) == ["a"]


"""Unit tests for AutoLearner (core/auto_learner.py).

Tests strategy discovery, training assignment, and feedback mechanisms
using real implementation.
"""
from __future__ import annotations

import pytest

from core.auto_learner import AutoLearner
from core.strategy_manager import StrategyManager


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def strategy_manager(tmp_path) -> StrategyManager:
    """Return a StrategyManager with a temporary directory."""
    return StrategyManager(str(tmp_path))


@pytest.fixture
def auto_learner(strategy_manager: StrategyManager) -> AutoLearner:
    """Return an AutoLearner with a real StrategyManager."""
    return AutoLearner(strategy_manager)


# ---------------------------------------------------------------------------
# Test Classes
# ---------------------------------------------------------------------------

class TestDiscoverStrategy:
    """Tests for discover_strategy() method."""

    def test_registers_new_strategy(
        self, auto_learner: AutoLearner, strategy_manager: StrategyManager
    ):
        """discover_strategy registers a new strategy and updates scores."""
        input_data = {"features": [1, 2, 3]}

        new_name = auto_learner.discover_strategy(input_data)

        assert new_name in strategy_manager.list_strategies()
        assert new_name in auto_learner._scores
        assert new_name in auto_learner._allocation
        assert abs(sum(auto_learner._allocation.values()) - 1.0) < 1e-6

    def test_multiple_discoveries(self, auto_learner: AutoLearner):
        """Multiple discoveries create multiple strategies."""
        name1 = auto_learner.discover_strategy({"data": 1})
        name2 = auto_learner.discover_strategy({"data": 2})

        assert name1 != name2
        assert name1 in auto_learner._allocation
        assert name2 in auto_learner._allocation


class TestAssignTraining:
    """Tests for assign_training() method."""

    def test_prefers_high_score_strategy(self, auto_learner: AutoLearner):
        """assign_training prefers strategies with higher scores."""
        auto_learner.discover_strategy({"a": 1})
        auto_learner.discover_strategy({"b": 2})

        strategies = list(auto_learner._allocation.keys())
        auto_learner.receive_feedback(strategies[0], 1.0)

        targets = auto_learner.assign_training({})

        assert strategies[0] in targets

    def test_returns_list(self, auto_learner: AutoLearner):
        """assign_training returns a list of strategy names."""
        auto_learner.discover_strategy({"data": 1})

        targets = auto_learner.assign_training({})

        assert isinstance(targets, list)
        assert len(targets) > 0


class TestReceiveFeedback:
    """Tests for receive_feedback() method."""

    def test_adjusts_allocation(self, auto_learner: AutoLearner):
        """receive_feedback adjusts allocation based on score."""
        auto_learner.discover_strategy({"a": 1})
        auto_learner.discover_strategy({"b": 2})

        strategies = list(auto_learner._allocation.keys())
        initial = dict(auto_learner._allocation)

        auto_learner.receive_feedback(strategies[0], 1.0)

        assert auto_learner._allocation[strategies[0]] > initial[strategies[0]]
        assert auto_learner._allocation[strategies[1]] < initial[strategies[1]]

    def test_drop_weak_strategy_after_negative_feedback(
        self, auto_learner: AutoLearner
    ):
        """Repeated negative feedback drops a strategy."""
        auto_learner.discover_strategy({"a": 1})
        auto_learner.discover_strategy({"b": 2})

        strategies = list(auto_learner._allocation.keys())
        weak_strategy = strategies[1]

        for _ in range(7):
            auto_learner.receive_feedback(weak_strategy, -1.0)

        assert weak_strategy not in auto_learner._scores
        assert weak_strategy not in auto_learner._allocation


class TestTrainAndSave:
    """Tests for train_and_save() method."""

    def test_creates_strategy_from_data_dir(
        self, auto_learner: AutoLearner, strategy_manager: StrategyManager, tmp_path
    ):
        """train_and_save creates a new strategy from data directory."""
        # Create a simple SGF file
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        sgf_file = data_dir / "game.sgf"
        sgf_file.write_text("(;FF[4]SZ[5];B[cc];W[dd])")

        strategy_name = auto_learner.train_and_save(str(data_dir))

        assert strategy_name is not None
        assert strategy_name in strategy_manager.list_strategies()

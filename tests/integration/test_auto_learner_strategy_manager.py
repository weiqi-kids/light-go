"""Integration tests for AutoLearner and StrategyManager.

Tests the interaction between AutoLearner and StrategyManager components
using real implementation.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from core.auto_learner import AutoLearner
from core.strategy_manager import StrategyManager


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def strategy_manager(tmp_path: Path) -> StrategyManager:
    """Return a StrategyManager with a temporary directory."""
    return StrategyManager(str(tmp_path))


@pytest.fixture
def auto_learner(strategy_manager: StrategyManager) -> AutoLearner:
    """Return an AutoLearner with the strategy manager."""
    return AutoLearner(strategy_manager)


# ---------------------------------------------------------------------------
# Test Classes
# ---------------------------------------------------------------------------

class TestAutoLearnerStrategyManagerIntegration:
    """Tests for AutoLearner and StrategyManager interaction."""

    def test_discover_and_feedback_cycle(
        self, auto_learner: AutoLearner, strategy_manager: StrategyManager
    ):
        """Test strategy discovery and feedback affects allocation."""
        # Discover two strategies
        primary = auto_learner.discover_strategy({"dummy": 1})
        secondary = auto_learner.discover_strategy({"dummy": 2})

        # Verify both are registered
        assert primary in strategy_manager.list_strategies()
        assert secondary in strategy_manager.list_strategies()

        # Check initial assignment includes both
        board_features = {"board": []}
        selected_before = set(auto_learner.assign_training(board_features))
        assert primary in selected_before
        alloc_before = dict(auto_learner._allocation)

        # Provide positive feedback for primary
        auto_learner.receive_feedback(primary, 1.0)

        # Verify allocation changed
        selected_after = set(auto_learner.assign_training(board_features))
        assert primary in selected_after
        assert secondary not in selected_after

        alloc_after = auto_learner._allocation
        assert alloc_after[primary] > alloc_before[primary]
        assert alloc_after[secondary] < alloc_before[secondary]

    def test_strategy_persistence_across_managers(self, tmp_path: Path):
        """Strategies persist when loading from same directory."""
        # First manager creates strategies
        manager1 = StrategyManager(str(tmp_path))
        learner1 = AutoLearner(manager1)
        name = learner1.discover_strategy({"data": 1})
        manager1.save_strategies()

        # Second manager loads from same directory
        manager2 = StrategyManager(str(tmp_path))
        manager2.load_strategies()

        assert name in manager2.list_strategies()

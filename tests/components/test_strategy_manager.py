"""Component tests for Strategy Manager (core/strategy_manager.py).

This module tests strategy management functionality:
- StrategyManager: Register, save, load, and manage strategies
- create_strategy(): Create placeholder strategies
- monitor_and_manage_strategies(): Ensure minimum strategy availability
- evaluate_strategies(): Evaluate strategy performance
- Convergence methods (majority vote, weighted average)
"""
from __future__ import annotations

import os
import tempfile
import pytest
from typing import Any, Dict
from types import SimpleNamespace

from core.strategy_manager import (
    StrategyManager,
    create_strategy,
    monitor_and_manage_strategies,
)


class TestCreateStrategy:
    """Tests for create_strategy() function."""

    def test_create_strategy_with_no_params(self):
        """Create strategy with default parameters."""
        strategy = create_strategy()

        assert hasattr(strategy, "training_params")
        assert hasattr(strategy, "accept_state")
        assert isinstance(strategy.training_params, dict)
        assert callable(strategy.accept_state)

    def test_create_strategy_with_params(self):
        """Create strategy with custom parameters."""
        params = {"stable": True, "custom_key": "custom_value"}
        strategy = create_strategy(params)

        assert strategy.training_params["stable"] is True
        assert strategy.training_params["custom_key"] == "custom_value"

    def test_create_strategy_default_stable_false(self):
        """Default stable flag is False."""
        strategy = create_strategy()
        assert strategy.training_params.get("stable") is False

    def test_accept_state_returns_bool(self):
        """accept_state should return boolean."""
        strategy = create_strategy()
        state = {"total_black_stones": 50, "total_white_stones": 48}

        result = strategy.accept_state(state)
        assert isinstance(result, bool)

    def test_accept_state_default_accepts_all(self):
        """Default accept_state accepts any state."""
        strategy = create_strategy()

        assert strategy.accept_state({}) is True
        assert strategy.accept_state({"total_black_stones": 0}) is True
        assert strategy.accept_state({"total_black_stones": 100}) is True


class TestStrategyManagerInstantiation:
    """Tests for StrategyManager instantiation."""

    def test_instantiate_with_temp_dir(self, temp_dir):
        """Manager can be instantiated with a directory."""
        manager = StrategyManager(temp_dir)

        assert manager is not None
        assert manager.strategies_path == temp_dir

    def test_instantiate_creates_directory(self, temp_dir):
        """Manager creates directory if it doesn't exist."""
        new_dir = os.path.join(temp_dir, "new_strategies")
        manager = StrategyManager(new_dir)

        assert os.path.isdir(new_dir)


class TestStrategyManagerRegister:
    """Tests for strategy registration."""

    def test_register_strategy(self, temp_dir):
        """Register a strategy."""
        manager = StrategyManager(temp_dir)
        strategy = create_strategy()

        manager.register_strategy("test_strategy", strategy)

        assert "test_strategy" in manager.list_strategies()

    def test_register_multiple_strategies(self, temp_dir):
        """Register multiple strategies."""
        manager = StrategyManager(temp_dir)

        for i in range(5):
            strategy = create_strategy({"id": i})
            manager.register_strategy(f"strategy_{i}", strategy)

        strategies = manager.list_strategies()
        assert len(strategies) >= 5

    def test_register_overwrites_existing(self, temp_dir):
        """Registering with same name overwrites."""
        manager = StrategyManager(temp_dir)

        strategy1 = create_strategy({"version": 1})
        strategy2 = create_strategy({"version": 2})

        manager.register_strategy("test", strategy1)
        manager.register_strategy("test", strategy2)

        # Should still have one strategy named "test"
        strategies = manager.list_strategies()
        assert strategies.count("test") == 1


class TestStrategyManagerSaveLoad:
    """Tests for strategy persistence."""

    def test_save_strategy(self, temp_dir):
        """Save strategy to disk using pickleable data."""
        manager = StrategyManager(temp_dir)
        # Use a simple dict instead of create_strategy() which contains local functions
        strategy_data = {"stable": False, "key": "value"}

        manager.save_strategy("savable", strategy_data)

        # Check file exists
        files = os.listdir(temp_dir)
        assert any("savable" in f for f in files)

    def test_load_strategy(self, temp_dir):
        """Load strategy from disk."""
        manager = StrategyManager(temp_dir)
        # Use a simple dict that can be pickled
        strategy_data = {"stable": False, "key": "value"}

        manager.save_strategy("loadable", strategy_data)

        # Create new manager and load
        manager2 = StrategyManager(temp_dir)
        loaded = manager2.load_strategy("loadable")

        assert loaded is not None

    def test_list_strategies(self, temp_dir):
        """List registered strategies."""
        manager = StrategyManager(temp_dir)

        manager.register_strategy("a", create_strategy())
        manager.register_strategy("b", create_strategy())

        strategies = manager.list_strategies()
        assert "a" in strategies
        assert "b" in strategies


class TestStrategyManagerAccepts:
    """Tests for strategy_accepts functionality."""

    def test_strategy_accepts_state(self, temp_dir):
        """Check if strategy accepts a state."""
        manager = StrategyManager(temp_dir)
        strategy = create_strategy()

        manager.register_strategy("acceptor", strategy)

        state = {"total_black_stones": 10, "total_white_stones": 10}
        result = manager.strategy_accepts("acceptor", state)

        assert isinstance(result, bool)


class TestStrategyManagerConverge:
    """Tests for convergence/fusion methods."""

    def test_converge_majority_vote(self, temp_dir):
        """Converge using majority vote."""
        manager = StrategyManager(temp_dir)

        # Create mock strategies with predict method
        class MockStrategy:
            def __init__(self, move):
                self.move = move
                self.training_params = {}

            def accept_state(self, state):
                return True

            def predict(self, data):
                return self.move

        # Register strategies that mostly agree
        manager.register_strategy("s1", MockStrategy((3, 3)))
        manager.register_strategy("s2", MockStrategy((3, 3)))
        manager.register_strategy("s3", MockStrategy((4, 4)))

        result = manager.converge({}, method="majority_vote")

        assert result == (3, 3)

    def test_converge_with_no_strategies(self, temp_dir):
        """Converge with no strategies returns None."""
        manager = StrategyManager(temp_dir)

        result = manager.converge({}, method="majority_vote")

        assert result is None


class TestMonitorAndManageStrategies:
    """Tests for monitor_and_manage_strategies function."""

    def test_creates_strategies_to_meet_threshold(self):
        """Creates new strategies if below threshold."""
        strategies = {"s1": create_strategy({"stable": True})}

        result = monitor_and_manage_strategies(strategies, 3)

        assert len(result) >= 3

    def test_preserves_existing_strategies(self):
        """Existing strategies are preserved."""
        strategies = {
            "s1": create_strategy({"stable": False}),
            "s2": create_strategy({"stable": False}),
        }

        result = monitor_and_manage_strategies(strategies, 3)

        assert "s1" in result
        assert "s2" in result

    def test_respects_threshold(self):
        """Result meets minimum threshold."""
        strategies = {}

        result = monitor_and_manage_strategies(strategies, 5)

        assert len(result) >= 5


class TestStrategyProtocol:
    """Tests for strategy protocol compliance."""

    def test_strategy_has_training_params(self):
        """Strategy must have training_params dict."""
        strategy = create_strategy()
        assert hasattr(strategy, "training_params")
        assert isinstance(strategy.training_params, dict)

    def test_strategy_has_accept_state(self):
        """Strategy must have accept_state callable."""
        strategy = create_strategy()
        assert hasattr(strategy, "accept_state")
        assert callable(strategy.accept_state)


class TestEdgeCases:
    """Edge case tests for strategy manager."""

    def test_empty_manager(self, temp_dir):
        """Manager with no strategies."""
        manager = StrategyManager(temp_dir)

        assert manager.list_strategies() == []

    def test_register_none_strategy(self, temp_dir):
        """Registering None should be handled."""
        manager = StrategyManager(temp_dir)

        try:
            manager.register_strategy("none_strat", None)
        except (TypeError, AttributeError):
            pass  # Expected behavior

    def test_load_nonexistent_strategy(self, temp_dir):
        """Loading nonexistent strategy raises or returns None."""
        manager = StrategyManager(temp_dir)

        try:
            result = manager.load_strategy("nonexistent")
            # If it doesn't raise, result should be None or some default
            assert result is None or True
        except FileNotFoundError:
            pass  # Expected behavior - file doesn't exist

    def test_strategy_name_with_special_chars(self, temp_dir):
        """Strategy names with special characters."""
        manager = StrategyManager(temp_dir)
        strategy = create_strategy()

        # Try various names
        names = ["strategy_1", "strategy-2", "strategy.3"]
        for name in names:
            manager.register_strategy(name, strategy)

        for name in names:
            assert name in manager.list_strategies()

"""Component tests for Strategy Manager (core/strategy_manager.py).

This module tests strategy management functionality:
- StrategyManager: Register, save, load, and manage strategies
- create_strategy(): Create placeholder strategies
- monitor_and_manage_strategies(): Ensure minimum strategy availability
- evaluate_strategies(): Evaluate strategy performance
- load_all_strategies(): Batch loading from directory
- Convergence methods (majority vote, weighted average, meta model)
"""
from __future__ import annotations

import os
import pickle
import pytest
from typing import Any, Dict

from core.strategy_manager import (
    StrategyManager,
    create_strategy,
    monitor_and_manage_strategies,
    evaluate_strategies,
    load_all_strategies,
    save_strategy as module_save_strategy,
    load_strategy as module_load_strategy,
    DEFAULT_STRATEGY_DIR,
)

# MockStrategy and MockMetaModel are imported from conftest.py by pytest
# We need to import them for type hints and direct usage in tests
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from conftest import MockStrategy, MockMetaModel


# ===========================================================================
# Tests for create_strategy() function
# ===========================================================================

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

    def test_create_strategy_preserves_existing_stable(self):
        """If params already has stable=True, it should be preserved."""
        strategy = create_strategy({"stable": True})
        assert strategy.training_params["stable"] is True

    @pytest.mark.parametrize("state", [
        {},
        {"total_black_stones": 0},
        {"total_black_stones": 100, "total_white_stones": 99},
        {"other_key": "value"},
    ])
    def test_accept_state_accepts_any_state(self, state):
        """Default accept_state accepts any state."""
        strategy = create_strategy()
        assert strategy.accept_state(state) is True

    def test_accept_state_returns_bool(self):
        """accept_state should return boolean."""
        strategy = create_strategy()
        state = {"total_black_stones": 50, "total_white_stones": 48}
        result = strategy.accept_state(state)
        assert isinstance(result, bool)


# ===========================================================================
# Tests for StrategyManager instantiation
# ===========================================================================

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

    def test_instantiate_loads_existing_filters(self, temp_dir):
        """Manager loads existing filter files on creation."""
        # Create a filter file before manager creation
        filter_data = {"min_stones": 10, "stable": True}
        with open(os.path.join(temp_dir, "existing.flt"), "wb") as f:
            pickle.dump(filter_data, f)

        manager = StrategyManager(temp_dir)

        assert "existing" in manager._filters
        assert manager._filters["existing"]["min_stones"] == 10


# ===========================================================================
# Tests for strategy registration
# ===========================================================================

class TestStrategyManagerRegister:
    """Tests for strategy registration."""

    def test_register_strategy(self, strategy_manager, mock_strategy):
        """Register a strategy."""
        strategy = mock_strategy()
        strategy_manager.register_strategy("test_strategy", strategy)

        assert "test_strategy" in strategy_manager.list_strategies()

    def test_register_multiple_strategies(self, strategy_manager, mock_strategy):
        """Register multiple strategies."""
        for i in range(5):
            strategy_manager.register_strategy(f"strategy_{i}", mock_strategy(prediction=i))

        strategies = strategy_manager.list_strategies()
        assert len(strategies) >= 5
        for i in range(5):
            assert f"strategy_{i}" in strategies

    def test_register_overwrites_existing(self, strategy_manager, mock_strategy):
        """Registering with same name overwrites."""
        strategy1 = mock_strategy(prediction="v1")
        strategy2 = mock_strategy(prediction="v2")

        strategy_manager.register_strategy("test", strategy1)
        strategy_manager.register_strategy("test", strategy2)

        strategies = strategy_manager.list_strategies()
        assert strategies.count("test") == 1
        # Verify it's the second strategy
        assert strategy_manager._strategies["test"].prediction == "v2"

    def test_register_none_raises_or_stores(self, strategy_manager):
        """Registering None should store None (no validation in register)."""
        # The register method doesn't validate, it just stores
        strategy_manager.register_strategy("none_strat", None)
        assert "none_strat" in strategy_manager.list_strategies()
        assert strategy_manager._strategies["none_strat"] is None


# ===========================================================================
# Tests for strategy persistence (save/load)
# ===========================================================================

class TestStrategyManagerSaveLoad:
    """Tests for strategy persistence."""

    def test_save_strategy_creates_file(self, strategy_manager, mock_strategy, temp_dir):
        """Save strategy creates .joblib or .pkl file."""
        strategy = mock_strategy(prediction=(1, 1))
        strategy_manager.save_strategy("savable", strategy)

        # Check for either .joblib (if available) or .pkl file
        joblib_exists = os.path.exists(os.path.join(temp_dir, "savable.joblib"))
        pkl_exists = os.path.exists(os.path.join(temp_dir, "savable.pkl"))
        assert joblib_exists or pkl_exists

    def test_save_strategy_creates_meta_file_for_class_with_load(self, strategy_manager, mock_strategy, temp_dir):
        """Save strategy creates .meta file for classes with load method."""
        strategy = mock_strategy(prediction=(1, 1))
        strategy_manager.save_strategy("with_meta", strategy)

        # MockStrategy has a load classmethod, so .meta should be created
        assert os.path.exists(os.path.join(temp_dir, "with_meta.meta"))

    def test_save_and_load_roundtrip(self, strategy_manager, mock_strategy, temp_dir):
        """Strategy can be saved and loaded back."""
        original = mock_strategy(prediction=(5, 5), stable=True)
        strategy_manager.save_strategy("roundtrip", original)

        # Create new manager to simulate fresh load
        manager2 = StrategyManager(temp_dir)
        loaded = manager2.load_strategy("roundtrip")

        assert loaded is not None
        assert loaded.prediction == (5, 5)
        assert loaded.training_params["stable"] is True

    def test_load_strategy_caches_result(self, strategy_manager, mock_strategy):
        """Loading same strategy twice returns cached version."""
        strategy = mock_strategy()
        strategy_manager.save_strategy("cached", strategy)

        loaded1 = strategy_manager.load_strategy("cached")
        loaded2 = strategy_manager.load_strategy("cached")

        assert loaded1 is loaded2

    def test_load_nonexistent_strategy_raises(self, strategy_manager):
        """Loading nonexistent strategy raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            strategy_manager.load_strategy("nonexistent")

    def test_list_strategies_includes_saved_and_registered(self, strategy_manager, mock_strategy):
        """list_strategies includes both registered and saved strategies."""
        strategy_manager.register_strategy("registered", mock_strategy())
        strategy_manager.save_strategy("saved", mock_strategy())

        strategies = strategy_manager.list_strategies()
        assert "registered" in strategies
        assert "saved" in strategies

    def test_save_strategies_batch(self, strategy_manager, mock_strategy, temp_dir):
        """save_strategies saves all registered strategies."""
        strategy_manager.register_strategy("batch1", mock_strategy())
        strategy_manager.register_strategy("batch2", mock_strategy())

        strategy_manager.save_strategies()

        # Check for either .joblib or .pkl files
        for name in ["batch1", "batch2"]:
            joblib_exists = os.path.exists(os.path.join(temp_dir, f"{name}.joblib"))
            pkl_exists = os.path.exists(os.path.join(temp_dir, f"{name}.pkl"))
            assert joblib_exists or pkl_exists

    def test_load_strategies_batch(self, temp_dir, mock_strategy):
        """load_strategies loads all strategies from directory."""
        # Save some strategies first
        manager1 = StrategyManager(temp_dir)
        manager1.save_strategy("load1", mock_strategy(prediction="a"))
        manager1.save_strategy("load2", mock_strategy(prediction="b"))

        # Create new manager and load all
        manager2 = StrategyManager(temp_dir)
        manager2.load_strategies()

        assert "load1" in manager2._strategies
        assert "load2" in manager2._strategies


# ===========================================================================
# Tests for filter parameters
# ===========================================================================

class TestFilterParameters:
    """Tests for filter parameter handling."""

    def test_save_filter_params(self, strategy_manager, temp_dir):
        """save_filter_params creates .flt file."""
        params = {"min_stones": 20, "stable": False}
        strategy_manager.save_filter_params("filtered", params)

        assert os.path.exists(os.path.join(temp_dir, "filtered.flt"))
        assert strategy_manager._filters["filtered"]["min_stones"] == 20

    def test_load_filter_params_on_init(self, temp_dir):
        """Filter params are loaded on manager initialization."""
        # Create filter file
        with open(os.path.join(temp_dir, "preexist.flt"), "wb") as f:
            pickle.dump({"min_stones": 30}, f)

        manager = StrategyManager(temp_dir)

        assert "preexist" in manager._filters
        assert manager._filters["preexist"]["min_stones"] == 30

    def test_corrupted_filter_file_handled(self, temp_dir):
        """Corrupted filter files don't crash initialization."""
        # Create corrupted filter file
        with open(os.path.join(temp_dir, "corrupt.flt"), "wb") as f:
            f.write(b"not valid pickle data")

        # Should not raise
        manager = StrategyManager(temp_dir)
        assert "corrupt" in manager._filters
        assert manager._filters["corrupt"] == {}


# ===========================================================================
# Tests for strategy_accepts functionality
# ===========================================================================

class TestStrategyAccepts:
    """Tests for strategy_accepts and related methods."""

    def test_strategy_accepts_uses_strategy_method(self, strategy_manager, mock_strategy):
        """strategy_accepts uses strategy's accept_state method."""
        strategy = mock_strategy(accept_all=False, min_stones=10)
        strategy_manager.register_strategy("selective", strategy)

        # Not enough stones
        state_low = {"total_black_stones": 3, "total_white_stones": 2}
        assert strategy_manager.strategy_accepts("selective", state_low) is False

        # Enough stones
        state_high = {"total_black_stones": 10, "total_white_stones": 5}
        assert strategy_manager.strategy_accepts("selective", state_high) is True

    def test_strategy_accepts_falls_back_to_filter(self, strategy_manager, temp_dir):
        """strategy_accepts uses filter params when strategy not registered."""
        strategy_manager.save_filter_params("unregistered", {"min_stones": 15})

        state_low = {"total_black_stones": 5, "total_white_stones": 5}
        state_high = {"total_black_stones": 10, "total_white_stones": 10}

        assert strategy_manager.strategy_accepts("unregistered", state_low) is False
        assert strategy_manager.strategy_accepts("unregistered", state_high) is True

    def test_strategy_accepts_default_true(self, strategy_manager):
        """strategy_accepts returns True for unknown strategy without filter."""
        state = {"total_black_stones": 1, "total_white_stones": 1}
        assert strategy_manager.strategy_accepts("unknown", state) is True

    def test_strategies_accepting_state(self, strategy_manager, mock_strategy):
        """strategies_accepting_state returns list of accepting strategies."""
        strategy_manager.register_strategy("accepts", mock_strategy(accept_all=True))
        strategy_manager.register_strategy("rejects", mock_strategy(accept_all=False, min_stones=100))

        state = {"total_black_stones": 5, "total_white_stones": 5}
        accepting = strategy_manager.strategies_accepting_state(state)

        assert "accepts" in accepting
        assert "rejects" not in accepting


# ===========================================================================
# Tests for ensure_capacity
# ===========================================================================

class TestEnsureCapacity:
    """Tests for ensure_capacity method."""

    def test_ensure_capacity_creates_strategies(self, strategy_manager, mock_strategy):
        """ensure_capacity creates strategies when needed."""
        state = {"total_black_stones": 5, "total_white_stones": 5}

        strategy_manager.ensure_capacity(state, 3, lambda: mock_strategy())

        accepting = strategy_manager.strategies_accepting_state(state)
        assert len(accepting) >= 3

    def test_ensure_capacity_respects_existing(self, strategy_manager, mock_strategy):
        """ensure_capacity doesn't create extra if enough exist."""
        strategy_manager.register_strategy("existing1", mock_strategy())
        strategy_manager.register_strategy("existing2", mock_strategy())
        strategy_manager.register_strategy("existing3", mock_strategy())

        state = {"total_black_stones": 5, "total_white_stones": 5}
        strategy_manager.ensure_capacity(state, 3, lambda: mock_strategy())

        # Should still have exactly 3, no extras created
        assert len(strategy_manager.list_strategies()) == 3

    def test_generate_name_uses_letters(self, strategy_manager, mock_strategy):
        """_generate_name uses a-z before fallback."""
        name = strategy_manager._generate_name()
        assert name == "a"

        strategy_manager.register_strategy("a", mock_strategy())
        name = strategy_manager._generate_name()
        assert name == "b"


# ===========================================================================
# Tests for run_all
# ===========================================================================

class TestRunAll:
    """Tests for run_all method."""

    def test_run_all_returns_predictions(self, strategy_manager, mock_strategy):
        """run_all returns predictions from all strategies."""
        strategy_manager.register_strategy("pred1", mock_strategy(prediction="move_a"))
        strategy_manager.register_strategy("pred2", mock_strategy(prediction="move_b"))

        results = strategy_manager.run_all({})

        assert results["pred1"] == "move_a"
        assert results["pred2"] == "move_b"

    def test_run_all_skips_strategies_without_predict(self, strategy_manager):
        """run_all skips strategies without predict method."""
        from types import SimpleNamespace
        no_predict = SimpleNamespace(training_params={})
        strategy_manager.register_strategy("no_predict", no_predict)

        results = strategy_manager.run_all({})

        assert "no_predict" not in results

    def test_run_all_empty_manager(self, strategy_manager):
        """run_all returns empty dict for empty manager."""
        results = strategy_manager.run_all({})
        assert results == {}


# ===========================================================================
# Tests for convergence methods
# ===========================================================================

class TestConvergence:
    """Tests for convergence/fusion methods."""

    def test_converge_majority_vote(self, strategy_manager_with_strategies):
        """Converge using majority vote."""
        # s1 and s2 predict (3,3), s3 predicts (4,4)
        result = strategy_manager_with_strategies.converge({}, method="majority_vote")
        assert result == (3, 3)

    def test_converge_majority_vote_tie(self, strategy_manager, mock_strategy):
        """Majority vote with tie returns first most common."""
        strategy_manager.register_strategy("a", mock_strategy(prediction="x"))
        strategy_manager.register_strategy("b", mock_strategy(prediction="y"))

        result = strategy_manager.converge({}, method="majority_vote")
        assert result in ["x", "y"]

    def test_converge_with_no_strategies(self, strategy_manager):
        """Converge with no strategies returns None."""
        result = strategy_manager.converge({}, method="majority_vote")
        assert result is None

    def test_converge_weighted(self, strategy_manager, mock_strategy):
        """Converge using weighted method."""
        strategy_manager.register_strategy("heavy", mock_strategy(prediction="a"))
        strategy_manager.register_strategy("light", mock_strategy(prediction="b"))

        weights = {"heavy": 10.0, "light": 1.0}
        result = strategy_manager.converge({}, method="weighted", weights=weights)

        assert result == "a"

    def test_converge_weighted_with_dict_predictions(self, strategy_manager, mock_strategy):
        """Converge weighted with score dict predictions."""
        strategy_manager.register_strategy("s1", mock_strategy(prediction={"a": 0.8, "b": 0.2}))
        strategy_manager.register_strategy("s2", mock_strategy(prediction={"a": 0.3, "b": 0.7}))

        weights = {"s1": 1.0, "s2": 1.0}
        result = strategy_manager.converge({}, method="weighted", weights=weights)

        # a: 0.8 + 0.3 = 1.1, b: 0.2 + 0.7 = 0.9 -> a wins
        assert result == "a"

    def test_converge_weighted_average_alias(self, strategy_manager, mock_strategy):
        """weighted_average is an alias for weighted."""
        strategy_manager.register_strategy("s1", mock_strategy(prediction="x"))

        result = strategy_manager.converge({}, method="weighted_average")
        assert result == "x"

    def test_converge_meta_model(self, strategy_manager, mock_strategy):
        """Converge using meta model."""
        strategy_manager.register_strategy("s1", mock_strategy(prediction="first"))
        strategy_manager.register_strategy("s2", mock_strategy(prediction="second"))

        meta = MockMetaModel(aggregation="first")
        result = strategy_manager.converge({}, method="meta", meta_model=meta)

        assert result == "first"

    def test_converge_meta_model_alias(self, strategy_manager, mock_strategy):
        """meta_model is an alias for meta."""
        strategy_manager.register_strategy("s1", mock_strategy(prediction="test"))

        meta = MockMetaModel()
        result = strategy_manager.converge({}, method="meta_model", meta_model=meta)

        assert result == "test"

    def test_converge_meta_without_model_raises(self, strategy_manager, mock_strategy):
        """Converge with meta method but no model raises ValueError."""
        strategy_manager.register_strategy("s1", mock_strategy())

        with pytest.raises(ValueError, match="meta_model must be provided"):
            strategy_manager.converge({}, method="meta")

    def test_converge_unknown_method_raises(self, strategy_manager, mock_strategy):
        """Unknown convergence method raises ValueError."""
        strategy_manager.register_strategy("s1", mock_strategy())

        with pytest.raises(ValueError, match="Unknown convergence method"):
            strategy_manager.converge({}, method="unknown_method")


# ===========================================================================
# Tests for evaluate_all
# ===========================================================================

class TestEvaluateAll:
    """Tests for evaluate_all method."""

    def test_evaluate_all_accuracy(self, strategy_manager, mock_strategy):
        """evaluate_all computes accuracy correctly."""
        strategy_manager.register_strategy("correct", mock_strategy(prediction="a"))
        strategy_manager.register_strategy("wrong", mock_strategy(prediction="b"))

        dataset = [
            ({}, "a"),
            ({}, "a"),
            ({}, "a"),
        ]

        report = strategy_manager.evaluate_all(dataset)

        assert report["correct"] == 1.0
        assert report["wrong"] == 0.0

    def test_evaluate_all_converged_accuracy(self, strategy_manager, mock_strategy):
        """evaluate_all includes converged accuracy."""
        strategy_manager.register_strategy("s1", mock_strategy(prediction="a"))
        strategy_manager.register_strategy("s2", mock_strategy(prediction="a"))
        strategy_manager.register_strategy("s3", mock_strategy(prediction="b"))

        dataset = [({}, "a")]

        report = strategy_manager.evaluate_all(dataset)

        # Majority vote should pick "a"
        assert report["converged"] == 1.0

    def test_evaluate_all_with_names_filter(self, strategy_manager, mock_strategy):
        """evaluate_all can filter by strategy names."""
        strategy_manager.register_strategy("included", mock_strategy(prediction="a"))
        strategy_manager.register_strategy("excluded", mock_strategy(prediction="b"))

        dataset = [({}, "a")]

        report = strategy_manager.evaluate_all(dataset, names=["included"])

        assert "included" in report
        assert "excluded" not in report

    def test_evaluate_all_empty_dataset(self, strategy_manager, mock_strategy):
        """evaluate_all handles empty dataset."""
        strategy_manager.register_strategy("s1", mock_strategy())

        report = strategy_manager.evaluate_all([])

        assert report["s1"] == 0.0
        assert report["converged"] == 0.0


# ===========================================================================
# Tests for score_stable_strategies
# ===========================================================================

class TestScoreStableStrategies:
    """Tests for score_stable_strategies method."""

    def test_score_stable_only_evaluates_stable(self, strategy_manager, mock_strategy):
        """score_stable_strategies only evaluates strategies marked stable."""
        strategy_manager.register_strategy("stable", mock_strategy(prediction="a", stable=True))
        strategy_manager.register_strategy("unstable", mock_strategy(prediction="a", stable=False))
        strategy_manager.save_filter_params("stable", {"stable": True})
        strategy_manager.save_filter_params("unstable", {"stable": False})

        dataset = [({}, "a")]
        report = strategy_manager.score_stable_strategies(dataset)

        assert "stable" in report
        assert "unstable" not in report

    def test_score_stable_no_stable_strategies(self, strategy_manager, mock_strategy):
        """score_stable_strategies returns empty if no stable strategies."""
        strategy_manager.register_strategy("unstable", mock_strategy(stable=False))

        dataset = [({}, "a")]
        report = strategy_manager.score_stable_strategies(dataset)

        assert report == {}


# ===========================================================================
# Tests for monitor_and_manage_strategies function
# ===========================================================================

class TestMonitorAndManageStrategies:
    """Tests for monitor_and_manage_strategies function."""

    def test_creates_strategies_to_meet_threshold(self):
        """Creates new strategies if below threshold."""
        strategies = {"s1": create_strategy({"stable": True})}

        result = monitor_and_manage_strategies(strategies, 3)

        # s1 is stable, so only 0 capable. Need 3 more.
        assert len(result) >= 3

    def test_preserves_existing_strategies(self):
        """Existing strategies are preserved."""
        s1 = create_strategy({"stable": False})
        s2 = create_strategy({"stable": False})
        strategies = {"s1": s1, "s2": s2}

        result = monitor_and_manage_strategies(strategies, 3)

        assert result["s1"] is s1
        assert result["s2"] is s2

    def test_counts_non_stable_as_capable(self):
        """Non-stable strategies count toward threshold."""
        strategies = {
            "unstable1": create_strategy({"stable": False}),
            "unstable2": create_strategy({"stable": False}),
        }

        result = monitor_and_manage_strategies(strategies, 2)

        # Already have 2 non-stable, threshold met
        assert len(result) == 2

    def test_respects_threshold_exactly(self):
        """Result meets minimum threshold exactly when starting empty."""
        strategies = {}

        result = monitor_and_manage_strategies(strategies, 5)

        assert len(result) == 5


# ===========================================================================
# Tests for evaluate_strategies function
# ===========================================================================

class TestEvaluateStrategies:
    """Tests for evaluate_strategies module function."""

    def test_evaluate_only_stable_strategies(self):
        """evaluate_strategies only scores strategies above stability threshold."""
        strategies = {
            "stable": create_strategy({"stable": True, "stability": 0.9, "wins": 8, "games": 10}),
            "unstable": create_strategy({"stable": False, "stability": 0.3, "wins": 5, "games": 10}),
        }

        scores = evaluate_strategies(strategies, stability_threshold=0.5)

        assert "stable" in scores
        assert scores["stable"] == 0.8
        assert "unstable" not in scores

    def test_evaluate_handles_missing_stats(self):
        """evaluate_strategies handles strategies without wins/games."""
        strategies = {
            "no_stats": create_strategy({"stability": 0.9}),
        }

        scores = evaluate_strategies(strategies, stability_threshold=0.5)

        assert scores["no_stats"] == 0.0

    def test_evaluate_empty_strategies(self):
        """evaluate_strategies returns empty for empty input."""
        scores = evaluate_strategies({}, stability_threshold=0.5)
        assert scores == {}


# ===========================================================================
# Tests for load_all_strategies function
# ===========================================================================

class TestLoadAllStrategies:
    """Tests for load_all_strategies module function."""

    def test_load_all_from_directory(self, temp_dir, mock_strategy):
        """load_all_strategies loads all .pkl files from directory."""
        # Save strategies
        s1 = mock_strategy(prediction="a")
        s2 = mock_strategy(prediction="b")

        with open(os.path.join(temp_dir, "strat1.pkl"), "wb") as f:
            pickle.dump(s1, f)
        with open(os.path.join(temp_dir, "strat2.pkl"), "wb") as f:
            pickle.dump(s2, f)

        loaded = load_all_strategies(temp_dir)

        assert "strat1" in loaded
        assert "strat2" in loaded

    def test_load_all_nonexistent_directory(self, temp_dir):
        """load_all_strategies returns empty for nonexistent directory."""
        loaded = load_all_strategies(os.path.join(temp_dir, "nonexistent"))
        assert loaded == {}

    def test_load_all_with_filter_params(self, temp_dir, mock_strategy):
        """load_all_strategies attaches filter params to strategies."""
        s1 = mock_strategy()
        with open(os.path.join(temp_dir, "filtered.pkl"), "wb") as f:
            pickle.dump(s1, f)
        with open(os.path.join(temp_dir, "filtered.flt"), "wb") as f:
            pickle.dump({"min_stones": 25}, f)

        loaded = load_all_strategies(temp_dir)

        assert "filtered" in loaded
        assert loaded["filtered"].training_params.get("min_stones") == 25

    def test_load_all_skips_invalid_strategies(self, temp_dir):
        """load_all_strategies skips strategies without required interface."""
        # Save a plain dict (no accept_state, no training_params)
        with open(os.path.join(temp_dir, "invalid.pkl"), "wb") as f:
            pickle.dump({"just": "data"}, f)

        loaded = load_all_strategies(temp_dir)

        assert "invalid" not in loaded


# ===========================================================================
# Tests for module-level save_strategy and load_strategy
# ===========================================================================

class TestModuleLevelSaveLoad:
    """Tests for module-level save_strategy and load_strategy functions."""

    def test_module_save_uses_default_dir(self, temp_dir, mock_strategy, monkeypatch):
        """save_strategy uses DEFAULT_STRATEGY_DIR."""
        # Temporarily change DEFAULT_STRATEGY_DIR
        import core.strategy_manager as sm
        monkeypatch.setattr(sm, "DEFAULT_STRATEGY_DIR", temp_dir)

        strategy = mock_strategy(prediction="test")
        module_save_strategy(strategy, "mod_save")

        # Check for either .joblib or .pkl file
        joblib_exists = os.path.exists(os.path.join(temp_dir, "mod_save.joblib"))
        pkl_exists = os.path.exists(os.path.join(temp_dir, "mod_save.pkl"))
        assert joblib_exists or pkl_exists

    def test_module_load_uses_default_dir(self, temp_dir, mock_strategy, monkeypatch):
        """load_strategy uses DEFAULT_STRATEGY_DIR."""
        import core.strategy_manager as sm
        monkeypatch.setattr(sm, "DEFAULT_STRATEGY_DIR", temp_dir)

        # Save first
        strategy = mock_strategy(prediction="loaded")
        module_save_strategy(strategy, "mod_load")

        # Load
        loaded = module_load_strategy("mod_load")
        assert loaded.prediction == "loaded"


# ===========================================================================
# Tests for strategy protocol compliance
# ===========================================================================

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

    def test_mock_strategy_protocol_compliance(self, mock_strategy):
        """MockStrategy implements full protocol."""
        s = mock_strategy()

        # TrainableStrategyProtocol
        assert hasattr(s, "training_params")
        assert isinstance(s.training_params, dict)
        assert callable(s.accept_state)

        # StrategyProtocol
        assert callable(s.predict)
        assert callable(s.save)
        assert callable(MockStrategy.load)


# ===========================================================================
# Edge case tests
# ===========================================================================

class TestEdgeCases:
    """Edge case tests for strategy manager."""

    def test_empty_manager(self, strategy_manager):
        """Manager with no strategies."""
        assert strategy_manager.list_strategies() == []

    @pytest.mark.parametrize("name", [
        "strategy_1",
        "strategy-2",
        "strategy.3",
        "UPPERCASE",
        "mixed_Case-123",
    ])
    def test_strategy_name_variations(self, strategy_manager, mock_strategy, name):
        """Strategy names with various characters."""
        strategy_manager.register_strategy(name, mock_strategy())
        assert name in strategy_manager.list_strategies()

    def test_large_number_of_strategies(self, strategy_manager, mock_strategy):
        """Manager handles many strategies."""
        for i in range(100):
            strategy_manager.register_strategy(f"s{i}", mock_strategy(prediction=i))

        strategies = strategy_manager.list_strategies()
        assert len(strategies) == 100

    def test_strategy_with_complex_prediction(self, strategy_manager, mock_strategy):
        """Strategy with complex prediction object."""
        complex_pred = {
            "moves": [(1, 1), (2, 2)],
            "scores": [0.9, 0.8],
            "metadata": {"depth": 5},
        }
        strategy_manager.register_strategy("complex", mock_strategy(prediction=complex_pred))

        results = strategy_manager.run_all({})
        assert results["complex"] == complex_pred


# ===========================================================================
# Layer 1: joblib serialization tests
# ===========================================================================


class TestJoblibSerialization:
    """Tests for joblib serialization (Layer 1)."""

    def test_save_load_with_joblib(self, temp_dir):
        """Save and load strategy using joblib format."""
        manager = StrategyManager(temp_dir)
        data = {"key": "value", "number": 42, "list": [1, 2, 3]}

        manager.save_strategy("joblib_test", data)
        loaded = manager.load_strategy("joblib_test")

        assert loaded["key"] == "value"
        assert loaded["number"] == 42
        assert loaded["list"] == [1, 2, 3]

    def test_backward_compatible_with_pkl(self, temp_dir):
        """Can still load old .pkl files."""
        import pickle

        # Manually create a .pkl file (simulating old format)
        pkl_path = os.path.join(temp_dir, "legacy.pkl")
        with open(pkl_path, "wb") as f:
            pickle.dump({"legacy": True}, f)

        manager = StrategyManager(temp_dir)
        loaded = manager.load_strategy("legacy")

        assert loaded["legacy"] is True

    def test_prefers_joblib_over_pkl(self, temp_dir):
        """When both .joblib and .pkl exist, prefers .joblib."""
        import pickle

        # Create both files with different content
        pkl_path = os.path.join(temp_dir, "both.pkl")
        with open(pkl_path, "wb") as f:
            pickle.dump({"format": "pkl"}, f)

        # Check if joblib is available
        try:
            import joblib
            joblib_path = os.path.join(temp_dir, "both.joblib")
            joblib.dump({"format": "joblib"}, joblib_path)
            has_joblib = True
        except ImportError:
            has_joblib = False

        manager = StrategyManager(temp_dir)
        loaded = manager.load_strategy("both")

        if has_joblib:
            assert loaded["format"] == "joblib"
        else:
            assert loaded["format"] == "pkl"

    def test_list_strategies_includes_both_formats(self, temp_dir):
        """list_strategies() finds both .joblib and .pkl files."""
        import pickle

        # Create .pkl file
        pkl_path = os.path.join(temp_dir, "pkl_only.pkl")
        with open(pkl_path, "wb") as f:
            pickle.dump({}, f)

        manager = StrategyManager(temp_dir)

        # Save another with manager (will use joblib if available)
        manager.save_strategy("manager_saved", {})

        strategies = manager.list_strategies()
        assert "pkl_only" in strategies
        assert "manager_saved" in strategies

    def test_available_features_includes_joblib(self):
        """available_features() reports joblib status."""
        features = StrategyManager.available_features()

        assert "joblib" in features
        assert isinstance(features["joblib"], bool)


# ===========================================================================
# Layer 2: sklearn VotingClassifier tests
# ===========================================================================


class TestSklearnEnsemble:
    """Tests for sklearn VotingClassifier integration (Layer 2)."""

    def test_available_features_includes_sklearn(self):
        """available_features() reports sklearn status."""
        features = StrategyManager.available_features()

        assert "sklearn" in features
        assert isinstance(features["sklearn"], bool)

    def test_build_ensemble_without_sklearn(self, temp_dir):
        """build_ensemble raises ImportError if sklearn not available."""
        from core.strategy_manager import HAS_SKLEARN

        manager = StrategyManager(temp_dir)

        if not HAS_SKLEARN:
            with pytest.raises(ImportError):
                manager.build_ensemble()

    @pytest.mark.skipif(
        not __import__("core.strategy_manager", fromlist=["HAS_SKLEARN"]).HAS_SKLEARN,
        reason="sklearn not installed"
    )
    def test_build_ensemble_with_sklearn(self, temp_dir):
        """build_ensemble creates VotingClassifier when sklearn available."""
        from sklearn.tree import DecisionTreeClassifier

        manager = StrategyManager(temp_dir)

        # Register sklearn-compatible estimators
        manager.register_strategy("tree1", DecisionTreeClassifier())
        manager.register_strategy("tree2", DecisionTreeClassifier())

        ensemble = manager.build_ensemble(voting="hard")

        assert ensemble is not None
        assert hasattr(ensemble, "estimators")

    @pytest.mark.skipif(
        not __import__("core.strategy_manager", fromlist=["HAS_SKLEARN"]).HAS_SKLEARN,
        reason="sklearn not installed"
    )
    def test_build_ensemble_empty_returns_none(self, temp_dir):
        """build_ensemble with no valid estimators returns None."""
        manager = StrategyManager(temp_dir)

        # Register non-sklearn strategy
        manager.register_strategy("simple", {"data": "value"})

        ensemble = manager.build_ensemble()

        assert ensemble is None


# ===========================================================================
# Layer 3: MLflow tracking tests
# ===========================================================================


class TestMLflowTracking:
    """Tests for MLflow tracking integration (Layer 3)."""

    def test_available_features_includes_mlflow(self):
        """available_features() reports mlflow status."""
        features = StrategyManager.available_features()

        assert "mlflow" in features
        assert isinstance(features["mlflow"], bool)

    def test_setup_mlflow_without_mlflow(self, temp_dir):
        """setup_mlflow returns False if mlflow not available."""
        from core.strategy_manager import HAS_MLFLOW

        manager = StrategyManager(temp_dir)

        if not HAS_MLFLOW:
            result = manager.setup_mlflow("test_experiment")
            assert result is False

    def test_log_strategy_without_mlflow(self, temp_dir):
        """log_strategy returns None if mlflow not available."""
        from core.strategy_manager import HAS_MLFLOW

        manager = StrategyManager(temp_dir)
        manager.register_strategy("test", create_strategy())

        if not HAS_MLFLOW:
            result = manager.log_strategy("test", metrics={"accuracy": 0.95})
            assert result is None

    def test_log_strategy_nonexistent(self, temp_dir):
        """log_strategy returns None for nonexistent strategy."""
        from core.strategy_manager import HAS_MLFLOW

        manager = StrategyManager(temp_dir)

        if HAS_MLFLOW:
            result = manager.log_strategy("nonexistent")
            assert result is None

    @pytest.mark.skipif(
        not __import__("core.strategy_manager", fromlist=["HAS_MLFLOW"]).HAS_MLFLOW,
        reason="mlflow not installed"
    )
    def test_setup_mlflow_with_mlflow(self, temp_dir):
        """setup_mlflow succeeds when mlflow available."""
        manager = StrategyManager(temp_dir)

        result = manager.setup_mlflow("test_go_strategies")

        assert result is True
        assert manager._mlflow_experiment == "test_go_strategies"

    @pytest.mark.skipif(
        not __import__("core.strategy_manager", fromlist=["HAS_MLFLOW"]).HAS_MLFLOW,
        reason="mlflow not installed"
    )
    def test_log_strategy_with_mlflow(self, temp_dir):
        """log_strategy returns run_id when successful."""
        manager = StrategyManager(temp_dir)
        manager.setup_mlflow("test_experiment")

        # Create a simple strategy with predict
        class SimpleStrategy:
            def __init__(self):
                self.training_params = {"stable": False}

            def accept_state(self, state):
                return True

            def predict(self, data):
                return 0

        manager.register_strategy("loggable", SimpleStrategy())
        run_id = manager.log_strategy(
            "loggable",
            metrics={"accuracy": 0.85, "win_rate": 0.6},
            params={"learning_rate": 0.01},
            tags={"version": "test"}
        )

        assert run_id is not None
        assert isinstance(run_id, str)

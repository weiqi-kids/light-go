"""Unit tests for StrategyManager (core/strategy_manager.py).

Tests strategy registration, persistence, convergence methods, and evaluation
using real implementation.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import pytest

from core.strategy_manager import StrategyManager


# ---------------------------------------------------------------------------
# Simple Strategy Class (real implementation for testing)
# ---------------------------------------------------------------------------

class SimpleStrategy:
    """Simple strategy implementation for testing."""

    def __init__(self, prediction: Any = None):
        self.prediction = prediction
        self.training_params: Dict[str, Any] = {"stable": False}

    def predict(self, input_data: Any) -> Any:
        """Return the configured prediction."""
        return self.prediction

    def accept_state(self, state: Dict[str, Any]) -> bool:
        """Accept any state."""
        return True

    def save(self, path: str) -> None:
        """Save strategy to path."""
        import pickle
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str) -> "SimpleStrategy":
        """Load strategy from path."""
        import pickle
        with open(path, "rb") as f:
            return pickle.load(f)


class SequenceStrategy:
    """Strategy that returns predictions from a sequence."""

    def __init__(self, predictions: list):
        self.predictions = predictions
        self.idx = 0
        self.training_params: Dict[str, Any] = {"stable": False}

    def predict(self, input_data: Any) -> Any:
        """Return next prediction in sequence."""
        result = self.predictions[self.idx % len(self.predictions)]
        self.idx += 1
        return result

    def accept_state(self, state: Dict[str, Any]) -> bool:
        """Accept any state."""
        return True

    def save(self, path: str) -> None:
        """Save strategy to path."""
        import pickle
        with open(path, "wb") as f:
            pickle.dump(self, f)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def strategy_manager(tmp_path: Path) -> StrategyManager:
    """Return a StrategyManager with a temporary directory."""
    return StrategyManager(str(tmp_path))


@pytest.fixture
def strategy_manager_with_strategies(tmp_path: Path) -> StrategyManager:
    """Return a StrategyManager with pre-registered strategies."""
    mgr = StrategyManager(str(tmp_path))
    mgr.register_strategy("s1", SimpleStrategy(prediction="X"))
    mgr.register_strategy("s2", SimpleStrategy(prediction="X"))
    mgr.register_strategy("s3", SimpleStrategy(prediction="Y"))
    return mgr


# ---------------------------------------------------------------------------
# Test Classes
# ---------------------------------------------------------------------------

class TestStrategyManagerRegistration:
    """Tests for strategy registration and persistence."""

    def test_register_and_persistence(
        self, strategy_manager: StrategyManager, tmp_path: Path
    ):
        """Register a strategy and verify persistence."""
        strategy = SimpleStrategy(prediction="A")
        strategy_manager.register_strategy("s1", strategy)

        strategy_manager.save_strategies()

        # Check for either .joblib (if available) or .pkl file
        assert (tmp_path / "s1.joblib").exists() or (tmp_path / "s1.pkl").exists()

        # Load in new manager
        mgr2 = StrategyManager(str(tmp_path))
        mgr2.load_strategies()
        assert "s1" in mgr2._strategies

    def test_list_strategies(
        self, strategy_manager_with_strategies: StrategyManager
    ):
        """list_strategies returns all registered strategy names."""
        names = strategy_manager_with_strategies.list_strategies()

        assert set(names) == {"s1", "s2", "s3"}


class TestStrategyManagerRunAll:
    """Tests for run_all() method."""

    def test_run_all_calls_predict(self, tmp_path: Path):
        """run_all returns predictions from all strategies."""
        mgr = StrategyManager(str(tmp_path))
        mgr.register_strategy("s1", SimpleStrategy(prediction="A"))
        mgr.register_strategy("s2", SimpleStrategy(prediction="B"))

        result = mgr.run_all("board_data")

        assert result == {"s1": "A", "s2": "B"}


class TestStrategyManagerConvergence:
    """Tests for convergence methods."""

    def test_converge_majority_vote(
        self, strategy_manager_with_strategies: StrategyManager
    ):
        """converge with majority_vote returns most common prediction."""
        result = strategy_manager_with_strategies.converge("data")

        assert result == "X"  # s1 and s2 predict "X", s3 predicts "Y"

    def test_converge_weighted(self, tmp_path: Path):
        """converge with weighted method uses weights."""
        mgr = StrategyManager(str(tmp_path))
        mgr.register_strategy("s1", SimpleStrategy(prediction={"A": 1, "B": 2}))
        mgr.register_strategy("s2", SimpleStrategy(prediction={"A": 2, "B": 1}))

        weights = {"s1": 1.0, "s2": 2.0}
        result = mgr.converge("data", method="weighted", weights=weights)

        assert result == "A"

    def test_converge_meta_model(
        self, strategy_manager_with_strategies: StrategyManager
    ):
        """converge with meta_model uses provided model."""

        class MetaModel:
            def predict(self, results: Dict[str, Any]) -> str:
                return "META"

        result = strategy_manager_with_strategies.converge(
            "data", method="meta_model", meta_model=MetaModel()
        )

        assert result == "META"


class TestStrategyManagerEvaluation:
    """Tests for evaluate_all() method."""

    def test_evaluate_all(self, tmp_path: Path):
        """evaluate_all computes accuracy for each strategy."""
        mgr = StrategyManager(str(tmp_path))

        mgr.register_strategy("s1", SequenceStrategy(["A", "A", "A"]))
        mgr.register_strategy("s2", SequenceStrategy(["A", "B", "A"]))
        mgr.register_strategy("s3", SequenceStrategy(["B", "B", "B"]))

        dataset = [(1, "A"), (2, "B"), (3, "A")]
        report = mgr.evaluate_all(dataset)

        assert pytest.approx(report["s1"], rel=1e-6) == 2 / 3
        assert report["s2"] == 1.0
        assert pytest.approx(report["s3"], rel=1e-6) == 1 / 3
        assert report["converged"] == 1.0

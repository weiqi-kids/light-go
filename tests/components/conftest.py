"""Shared fixtures for component tests.

This module provides component-specific fixtures. Common fixtures like
temp_dir, empty boards, and SGF content are inherited from tests/conftest.py.
"""
from __future__ import annotations

import pytest
from typing import Any, Dict, List

from core.strategy_manager import StrategyManager
from core.auto_learner import AutoLearner

Board = List[List[int]]


# ---------------------------------------------------------------------------
# Mock Classes for Testing
# ---------------------------------------------------------------------------

class MockStrategy:
    """Mock strategy for testing StrategyManager.

    Implements the TrainableStrategyProtocol interface with configurable
    predict behavior and state acceptance logic.
    """

    def __init__(
        self,
        prediction: Any = None,
        stable: bool = False,
        accept_all: bool = True,
        min_stones: int = 0,
    ):
        self.prediction = prediction
        self.training_params: Dict[str, Any] = {
            "stable": stable,
            "min_stones": min_stones,
        }
        self._accept_all = accept_all
        self._min_stones = min_stones

    def predict(self, input_data: Any) -> Any:
        """Return the configured prediction."""
        return self.prediction

    def accept_state(self, state: Dict[str, Any]) -> bool:
        """Check if this strategy should accept the given state."""
        if self._accept_all:
            return True
        stones = state.get("total_black_stones", 0) + state.get("total_white_stones", 0)
        return stones >= self._min_stones

    def save(self, path: str) -> None:
        """Save strategy to path."""
        import pickle
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str) -> "MockStrategy":
        """Load strategy from path."""
        import pickle
        with open(path, "rb") as f:
            return pickle.load(f)


class MockMetaModel:
    """Mock meta model for testing meta convergence."""

    def __init__(self, aggregation: str = "first"):
        self.aggregation = aggregation

    def predict(self, results: Dict[str, Any]) -> Any:
        """Aggregate results from multiple strategies."""
        if not results:
            return None
        values = list(results.values())
        if self.aggregation == "first":
            return values[0]
        elif self.aggregation == "last":
            return values[-1]
        return values[0]


# ---------------------------------------------------------------------------
# Strategy Manager Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_strategy():
    """Factory fixture to create MockStrategy instances.

    Usage:
        strategy = mock_strategy(prediction=(3, 3), stable=True)
    """
    def _create(
        prediction: Any = None,
        stable: bool = False,
        accept_all: bool = True,
        min_stones: int = 0,
    ) -> MockStrategy:
        return MockStrategy(
            prediction=prediction,
            stable=stable,
            accept_all=accept_all,
            min_stones=min_stones,
        )
    return _create


@pytest.fixture
def strategy_manager(temp_dir):
    """Provide a StrategyManager instance with a temporary directory."""
    return StrategyManager(temp_dir)


@pytest.fixture
def strategy_manager_with_strategies(strategy_manager, mock_strategy):
    """Provide a StrategyManager with pre-registered mock strategies."""
    strategy_manager.register_strategy("s1", mock_strategy(prediction=(3, 3)))
    strategy_manager.register_strategy("s2", mock_strategy(prediction=(3, 3)))
    strategy_manager.register_strategy("s3", mock_strategy(prediction=(4, 4)))
    return strategy_manager


@pytest.fixture
def auto_learner(strategy_manager) -> AutoLearner:
    """Return an AutoLearner with a StrategyManager."""
    return AutoLearner(strategy_manager)


# ---------------------------------------------------------------------------
# Board Pattern Fixtures (specific to component tests)
# ---------------------------------------------------------------------------

@pytest.fixture
def board_with_single_black_stone() -> Board:
    """Return a 5x5 board with a single black stone at center."""
    board = [[0] * 5 for _ in range(5)]
    board[2][2] = 1
    return board


@pytest.fixture
def board_with_cross_pattern() -> Board:
    """Return a 5x5 board with a cross pattern of black stones at center."""
    board = [[0] * 5 for _ in range(5)]
    board[1][2] = 1  # top
    board[2][1] = 1  # left
    board[2][2] = 1  # center
    board[2][3] = 1  # right
    board[3][2] = 1  # bottom
    return board


@pytest.fixture
def board_with_capture_scenario() -> Board:
    """Return a 5x5 board where white can be captured."""
    board = [[0] * 5 for _ in range(5)]
    # White stone at (1,1) surrounded by black on 3 sides
    board[0][1] = 1  # black above
    board[1][0] = 1  # black left
    board[2][1] = 1  # black below
    board[1][1] = -1  # white to be captured (needs black at (1,2))
    return board


@pytest.fixture
def board_with_ko_scenario() -> Board:
    """Return a board with a ko situation."""
    board = [[0] * 5 for _ in range(5)]
    # Classic ko pattern
    board[1][1] = 1   # black
    board[1][2] = -1  # white
    board[2][0] = 1   # black
    board[2][1] = -1  # white
    board[2][3] = -1  # white
    board[3][1] = 1   # black
    board[3][2] = -1  # white
    return board


# ---------------------------------------------------------------------------
# Engine Fixtures (Optimized for reuse)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="class")
def shared_engine(tmp_path_factory):
    """Class-scoped Engine instance for tests that don't modify state.

    This fixture creates a single Engine instance that is shared across
    all tests within a test class, reducing initialization overhead.
    """
    from core.engine import Engine
    tmp_dir = tmp_path_factory.mktemp("shared_engine")
    return Engine(str(tmp_dir))


@pytest.fixture(scope="module")
def trained_engine_factory(tmp_path_factory):
    """Module-scoped factory for pre-trained Engine.

    Returns a tuple of (engine, sgf_dir, output_dir, strategy_name) where
    the engine has already been trained with sample SGF data. This avoids
    repeated training across multiple test classes.
    """
    from core.engine import Engine

    # Create directories
    tmp_dir = tmp_path_factory.mktemp("trained_engine")
    sgf_dir = tmp_dir / "sgf_data"
    sgf_dir.mkdir()
    output_dir = tmp_dir / "output"
    output_dir.mkdir()

    # Create sample SGF file
    sgf_content = "(;GM[1]FF[4]SZ[9]KM[7.5];B[ee];W[gc])"
    sgf_path = sgf_dir / "test_game.sgf"
    sgf_path.write_text(sgf_content)

    # Create and train engine
    engine = Engine(str(tmp_dir))
    strategy_name = engine.train(str(sgf_dir), str(output_dir))

    return engine, str(sgf_dir), str(output_dir), strategy_name

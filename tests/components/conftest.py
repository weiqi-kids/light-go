"""Shared fixtures for component tests."""
from __future__ import annotations

import os
import sys
import tempfile
import pytest
from typing import Any, Dict, List, Tuple

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

Board = List[List[int]]


# ---------------------------------------------------------------------------
# Strategy Manager Fixtures
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


@pytest.fixture
def mock_strategy():
    """Factory fixture to create MockStrategy instances."""
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
    from core.strategy_manager import StrategyManager
    return StrategyManager(temp_dir)


@pytest.fixture
def strategy_manager_with_strategies(strategy_manager, mock_strategy):
    """Provide a StrategyManager with pre-registered mock strategies."""
    strategy_manager.register_strategy("s1", mock_strategy(prediction=(3, 3)))
    strategy_manager.register_strategy("s2", mock_strategy(prediction=(3, 3)))
    strategy_manager.register_strategy("s3", mock_strategy(prediction=(4, 4)))
    return strategy_manager


@pytest.fixture
def empty_board_5x5() -> Board:
    """Return a 5x5 empty board."""
    return [[0] * 5 for _ in range(5)]


@pytest.fixture
def empty_board_9x9() -> Board:
    """Return a 9x9 empty board."""
    return [[0] * 9 for _ in range(9)]


@pytest.fixture
def empty_board_19x19() -> Board:
    """Return a 19x19 empty board."""
    return [[0] * 19 for _ in range(19)]


@pytest.fixture
def board_with_single_black_stone() -> Board:
    """Return a 5x5 board with a single black stone at center."""
    board = [[0] * 5 for _ in range(5)]
    board[2][2] = 1
    return board


@pytest.fixture
def make_board():
    """Factory fixture to create boards with stones at specified positions.

    Usage:
        board = make_board(5, [(2, 2, 1), (0, 0, -1)])  # 5x5 with black at (2,2), white at (0,0)
    """
    def _make_board(size: int, stones: List[Tuple[int, int, int]] = None) -> Board:
        board = [[0] * size for _ in range(size)]
        if stones:
            for x, y, color in stones:
                board[y][x] = color
        return board
    return _make_board


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


@pytest.fixture
def simple_sgf_content() -> str:
    """Return a simple 9x9 SGF game string."""
    return "(;GM[1]FF[4]SZ[9]KM[7.5]RU[Chinese];B[ee];W[gc];B[cg])"


@pytest.fixture
def japanese_sgf_content() -> str:
    """Return an SGF with Japanese rules."""
    return "(;GM[1]FF[4]SZ[19]KM[6.5]RU[Japanese];B[pd];W[dp])"


@pytest.fixture
def sgf_with_handicap() -> str:
    """Return an SGF with handicap stones."""
    return "(;GM[1]FF[4]SZ[9]HA[2]KM[0.5]AB[gc][cg];W[ee])"


@pytest.fixture
def temp_dir():
    """Provide a temporary directory that is cleaned up after test."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def temp_sgf_dir(temp_dir):
    """Provide a temporary directory with a sample SGF file."""
    sgf_dir = os.path.join(temp_dir, "sgf_data")
    os.makedirs(sgf_dir, exist_ok=True)

    sgf_content = "(;GM[1]FF[4]SZ[9]KM[7.5];B[ee];W[gc])"
    sgf_path = os.path.join(sgf_dir, "test_game.sgf")
    with open(sgf_path, "w") as f:
        f.write(sgf_content)

    return sgf_dir

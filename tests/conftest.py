"""Root-level pytest configuration and shared fixtures.

This module provides:
- Automatic sys.path configuration for all tests
- Shared fixtures available to all test modules
- Common type definitions
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

import pytest

# ---------------------------------------------------------------------------
# Path Configuration (automatically applied to all tests)
# ---------------------------------------------------------------------------

# Add project root to sys.path so imports work from any test directory
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Type Definitions
# ---------------------------------------------------------------------------

Board = List[List[int]]
"""Type alias for Go board representation (0=empty, 1=black, -1=white)."""


# ---------------------------------------------------------------------------
# Temporary Directory Fixtures (unified naming)
# ---------------------------------------------------------------------------

@pytest.fixture
def temp_dir():
    """Provide a temporary directory as a string path.

    Alias for consistency with existing tests. Prefer tmp_path for new tests.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


# ---------------------------------------------------------------------------
# Board Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def make_board() -> Callable[[int, List[Tuple[int, int, int]]], Board]:
    """Factory fixture to create boards with stones at specified positions.

    Usage:
        board = make_board(5, [(2, 2, 1), (0, 0, -1)])  # 5x5 with black at (2,2), white at (0,0)

    Args:
        size: Board size (e.g., 5, 9, 19)
        stones: List of (x, y, color) tuples where color is 1 (black) or -1 (white)

    Returns:
        2D list representing the board
    """
    def _make_board(size: int, stones: List[Tuple[int, int, int]] = None) -> Board:
        board = [[0] * size for _ in range(size)]
        if stones:
            for x, y, color in stones:
                board[y][x] = color
        return board
    return _make_board


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


# ---------------------------------------------------------------------------
# SGF Fixtures
# ---------------------------------------------------------------------------

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
def temp_sgf_dir(temp_dir: str) -> str:
    """Provide a temporary directory with a sample SGF file."""
    sgf_dir = os.path.join(temp_dir, "sgf_data")
    os.makedirs(sgf_dir, exist_ok=True)

    sgf_content = "(;GM[1]FF[4]SZ[9]KM[7.5];B[ee];W[gc])"
    sgf_path = os.path.join(sgf_dir, "test_game.sgf")
    with open(sgf_path, "w") as f:
        f.write(sgf_content)

    return sgf_dir


# ---------------------------------------------------------------------------
# Pytest Configuration
# ---------------------------------------------------------------------------

def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )

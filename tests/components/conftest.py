"""Shared fixtures for component tests."""
from __future__ import annotations

import os
import sys
import tempfile
import pytest
from typing import List

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

Board = List[List[int]]


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

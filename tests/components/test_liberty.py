"""Component tests for Liberty Encoder (core/liberty.py).

This module tests the liberty counting functionality for Go stones:
- neighbors(): Get adjacent positions on the board
- group_and_liberties(): Find connected groups and their liberties
- count_liberties(): Count liberties for all stones on the board
"""
from __future__ import annotations

import pytest
from typing import List, Set, Tuple

from core.liberty import count_liberties, group_and_liberties, neighbors

Board = List[List[int]]


class TestNeighbors:
    """Tests for the neighbors() function."""

    def test_center_position_has_four_neighbors(self):
        """A center position should have exactly 4 neighbors."""
        result = list(neighbors(4, 4, 9))
        assert len(result) == 4
        assert set(result) == {(3, 4), (5, 4), (4, 3), (4, 5)}

    def test_corner_position_has_two_neighbors(self):
        """A corner position should have exactly 2 neighbors."""
        # Top-left corner
        result = list(neighbors(0, 0, 9))
        assert len(result) == 2
        assert set(result) == {(1, 0), (0, 1)}

        # Bottom-right corner
        result = list(neighbors(8, 8, 9))
        assert len(result) == 2
        assert set(result) == {(7, 8), (8, 7)}

    def test_edge_position_has_three_neighbors(self):
        """An edge position (not corner) should have exactly 3 neighbors."""
        # Top edge
        result = list(neighbors(4, 0, 9))
        assert len(result) == 3
        assert set(result) == {(3, 0), (5, 0), (4, 1)}

        # Left edge
        result = list(neighbors(0, 4, 9))
        assert len(result) == 3
        assert set(result) == {(1, 4), (0, 3), (0, 5)}

    def test_neighbors_respects_board_size(self):
        """Neighbors should respect different board sizes."""
        # 5x5 board
        result = list(neighbors(4, 4, 5))
        assert len(result) == 2  # Corner of 5x5
        assert set(result) == {(3, 4), (4, 3)}

        # 19x19 board center
        result = list(neighbors(9, 9, 19))
        assert len(result) == 4

    def test_neighbors_at_various_corners(self):
        """Test all four corners."""
        size = 9
        # Top-left (0, 0)
        assert set(neighbors(0, 0, size)) == {(1, 0), (0, 1)}
        # Top-right (8, 0)
        assert set(neighbors(8, 0, size)) == {(7, 0), (8, 1)}
        # Bottom-left (0, 8)
        assert set(neighbors(0, 8, size)) == {(1, 8), (0, 7)}
        # Bottom-right (8, 8)
        assert set(neighbors(8, 8, size)) == {(7, 8), (8, 7)}


class TestGroupAndLiberties:
    """Tests for the group_and_liberties() function."""

    def test_single_stone_center(self, board_with_single_black_stone):
        """Single stone in center should have 4 liberties."""
        group, liberties = group_and_liberties(board_with_single_black_stone, 2, 2)
        assert group == {(2, 2)}
        assert len(liberties) == 4
        assert liberties == {(1, 2), (3, 2), (2, 1), (2, 3)}

    def test_single_stone_corner(self):
        """Single stone in corner should have 2 liberties."""
        board = [[0] * 5 for _ in range(5)]
        board[0][0] = 1
        group, liberties = group_and_liberties(board, 0, 0)
        assert group == {(0, 0)}
        assert len(liberties) == 2
        assert liberties == {(1, 0), (0, 1)}

    def test_single_stone_edge(self):
        """Single stone on edge should have 3 liberties."""
        board = [[0] * 5 for _ in range(5)]
        board[0][2] = 1
        group, liberties = group_and_liberties(board, 2, 0)
        assert group == {(2, 0)}
        assert len(liberties) == 3

    def test_connected_group_cross_pattern(self, board_with_cross_pattern):
        """Cross pattern should have 5 stones and 8 liberties."""
        group, liberties = group_and_liberties(board_with_cross_pattern, 2, 2)
        assert len(group) == 5
        assert len(liberties) == 8

    def test_connected_group_horizontal_line(self):
        """Three stones in a horizontal line."""
        board = [[0] * 5 for _ in range(5)]
        board[2][1] = 1
        board[2][2] = 1
        board[2][3] = 1
        group, liberties = group_and_liberties(board, 2, 2)
        assert len(group) == 3
        # Line has 2 ends (2 libs each) + 6 sides = 8 liberties
        assert len(liberties) == 8

    def test_connected_group_vertical_line(self):
        """Three stones in a vertical line."""
        board = [[0] * 5 for _ in range(5)]
        board[1][2] = 1
        board[2][2] = 1
        board[3][2] = 1
        group, liberties = group_and_liberties(board, 2, 2)
        assert len(group) == 3
        assert len(liberties) == 8

    def test_group_with_one_liberty(self):
        """Group with only one liberty (atari situation)."""
        board = [[0] * 5 for _ in range(5)]
        # White stone surrounded by black, one liberty left
        board[1][1] = -1  # white
        board[0][1] = 1   # black above
        board[1][0] = 1   # black left
        board[2][1] = 1   # black below
        # (1, 2) is the only liberty for white

        group, liberties = group_and_liberties(board, 1, 1)
        assert group == {(1, 1)}
        assert len(liberties) == 1
        assert liberties == {(2, 1)}

    def test_group_with_no_liberties(self):
        """Completely surrounded stone has no liberties."""
        board = [[0] * 5 for _ in range(5)]
        board[1][1] = -1  # white
        board[0][1] = 1   # black above
        board[1][0] = 1   # black left
        board[2][1] = 1   # black below
        board[1][2] = 1   # black right

        group, liberties = group_and_liberties(board, 1, 1)
        assert group == {(1, 1)}
        assert len(liberties) == 0

    def test_separate_groups_not_connected(self):
        """Two separate stones are different groups."""
        board = [[0] * 5 for _ in range(5)]
        board[1][1] = 1
        board[3][3] = 1  # Not connected diagonally

        group1, _ = group_and_liberties(board, 1, 1)
        group2, _ = group_and_liberties(board, 3, 3)

        assert group1 == {(1, 1)}
        assert group2 == {(3, 3)}
        assert group1 != group2

    def test_diagonal_not_connected(self):
        """Diagonal stones are NOT connected in Go."""
        board = [[0] * 5 for _ in range(5)]
        board[1][1] = 1
        board[2][2] = 1  # Diagonal

        group, _ = group_and_liberties(board, 1, 1)
        assert len(group) == 1  # Only the single stone

    def test_large_connected_group(self):
        """Large L-shaped group."""
        board = [[0] * 9 for _ in range(9)]
        # Vertical line
        for y in range(5):
            board[y][2] = 1
        # Horizontal extension
        for x in range(2, 6):
            board[4][x] = 1

        group, liberties = group_and_liberties(board, 2, 2)
        assert len(group) == 8  # 5 vertical + 3 horizontal (center counts once)


class TestCountLiberties:
    """Tests for the count_liberties() function."""

    def test_empty_board_returns_empty(self, empty_board_5x5):
        """Empty board should return empty list."""
        result = count_liberties(empty_board_5x5)
        assert result == []

    def test_single_black_stone(self, board_with_single_black_stone):
        """Single black stone at center."""
        result = count_liberties(board_with_single_black_stone)
        assert len(result) == 1
        x, y, libs = result[0]
        assert libs == 4  # Positive for black
        assert (x, y) == (3, 3)  # 1-indexed in output

    def test_single_white_stone(self):
        """Single white stone returns negative liberty count."""
        board = [[0] * 5 for _ in range(5)]
        board[2][2] = -1  # white
        result = count_liberties(board)
        assert len(result) == 1
        x, y, libs = result[0]
        assert libs == -4  # Negative for white

    def test_multiple_separate_stones(self):
        """Multiple separate stones each get counted."""
        board = [[0] * 5 for _ in range(5)]
        board[0][0] = 1   # black corner
        board[4][4] = -1  # white corner

        result = count_liberties(board)
        assert len(result) == 2

        liberties = {(x, y, v) for x, y, v in result}
        # Black at corner has 2 liberties (positive)
        assert (1, 1, 2) in liberties
        # White at corner has 2 liberties (negative)
        assert (5, 5, -2) in liberties

    def test_connected_group_same_liberties(self, board_with_cross_pattern):
        """All stones in connected group report same liberty count."""
        result = count_liberties(board_with_cross_pattern)
        assert len(result) == 5

        # All should have 8 liberties (positive for black)
        for x, y, libs in result:
            assert libs == 8

    def test_adjacent_different_colors(self):
        """Adjacent stones of different colors."""
        board = [[0] * 5 for _ in range(5)]
        board[2][2] = 1   # black
        board[2][3] = -1  # white adjacent

        result = count_liberties(board)
        assert len(result) == 2

        # Find black and white liberties
        for x, y, libs in result:
            if libs > 0:  # black
                assert libs == 3  # 4 - 1 (blocked by white)
            else:  # white
                assert libs == -3  # -(4 - 1)

    def test_output_is_1_indexed(self):
        """Output coordinates should be 1-indexed."""
        board = [[0] * 5 for _ in range(5)]
        board[0][0] = 1  # Position (0,0) in 0-indexed

        result = count_liberties(board)
        x, y, _ = result[0]
        assert (x, y) == (1, 1)  # Should be 1-indexed


class TestEdgeCases:
    """Edge case tests for liberty functions."""

    def test_minimum_board_size(self):
        """Test with minimum 1x1 board."""
        board = [[1]]
        group, liberties = group_and_liberties(board, 0, 0)
        assert group == {(0, 0)}
        assert len(liberties) == 0  # No neighbors

        result = count_liberties(board)
        assert len(result) == 1
        assert result[0][2] == 0  # No liberties

    def test_2x2_board(self):
        """Test with 2x2 board."""
        board = [[1, 0], [0, -1]]
        # Black at (0,0), White at (1,1)

        b_group, b_libs = group_and_liberties(board, 0, 0)
        assert len(b_libs) == 2

        w_group, w_libs = group_and_liberties(board, 1, 1)
        assert len(w_libs) == 2

    def test_full_board_no_liberties(self):
        """Completely filled board - all groups have no liberties."""
        # Alternating pattern
        board = [[1 if (x + y) % 2 == 0 else -1 for x in range(3)] for y in range(3)]

        result = count_liberties(board)
        # All stones should have 0 liberties
        for x, y, libs in result:
            assert libs == 0 or libs == 0  # Both colors have 0

    def test_large_board_19x19(self, empty_board_19x19):
        """Test on standard 19x19 board."""
        empty_board_19x19[9][9] = 1  # Tengen

        result = count_liberties(empty_board_19x19)
        assert len(result) == 1
        x, y, libs = result[0]
        assert libs == 4  # Center still has 4 liberties
        assert (x, y) == (10, 10)  # 1-indexed

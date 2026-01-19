"""Component tests for Liberty Encoder (core/liberty.py).

This module tests the liberty counting functionality for Go stones:
- neighbors(): Get adjacent positions on the board
- group_and_liberties(): Find connected groups and their liberties
- count_liberties(): Count liberties for all stones on the board

Optimized with pytest.mark.parametrize for reduced code duplication.
"""
from __future__ import annotations

import pytest

from core.liberty import count_liberties, group_and_liberties, neighbors


class TestNeighbors:
    """Tests for the neighbors() function."""

    @pytest.mark.parametrize("x,y,size,expected", [
        # Center positions - 4 neighbors
        (4, 4, 9, {(3, 4), (5, 4), (4, 3), (4, 5)}),
        (9, 9, 19, {(8, 9), (10, 9), (9, 8), (9, 10)}),
        (2, 2, 5, {(1, 2), (3, 2), (2, 1), (2, 3)}),
    ], ids=["9x9_center", "19x19_center", "5x5_center"])
    def test_center_has_four_neighbors(self, x, y, size, expected):
        """Center positions should have exactly 4 neighbors."""
        assert set(neighbors(x, y, size)) == expected

    @pytest.mark.parametrize("x,y,size,expected", [
        # Corners - 2 neighbors
        (0, 0, 9, {(1, 0), (0, 1)}),      # top-left
        (8, 0, 9, {(7, 0), (8, 1)}),      # top-right
        (0, 8, 9, {(1, 8), (0, 7)}),      # bottom-left
        (8, 8, 9, {(7, 8), (8, 7)}),      # bottom-right
        (4, 4, 5, {(3, 4), (4, 3)}),      # 5x5 corner
    ], ids=["top_left", "top_right", "bottom_left", "bottom_right", "5x5_corner"])
    def test_corner_has_two_neighbors(self, x, y, size, expected):
        """Corner positions should have exactly 2 neighbors."""
        assert set(neighbors(x, y, size)) == expected

    @pytest.mark.parametrize("x,y,size,expected", [
        # Edges - 3 neighbors
        (4, 0, 9, {(3, 0), (5, 0), (4, 1)}),  # top edge
        (0, 4, 9, {(1, 4), (0, 3), (0, 5)}),  # left edge
        (8, 4, 9, {(7, 4), (8, 3), (8, 5)}),  # right edge
        (4, 8, 9, {(3, 8), (5, 8), (4, 7)}),  # bottom edge
    ], ids=["top", "left", "right", "bottom"])
    def test_edge_has_three_neighbors(self, x, y, size, expected):
        """Edge positions (non-corner) should have exactly 3 neighbors."""
        assert set(neighbors(x, y, size)) == expected


class TestGroupAndLiberties:
    """Tests for the group_and_liberties() function."""

    @pytest.mark.parametrize("x,y,expected_libs", [
        (2, 2, 4),  # center: 4 liberties
        (0, 0, 2),  # corner: 2 liberties
        (2, 0, 3),  # edge: 3 liberties
    ], ids=["center", "corner", "edge"])
    def test_single_stone_liberties(self, make_board, x, y, expected_libs):
        """Single stone liberty count depends on position."""
        board = make_board(5, [(x, y, 1)])
        group, liberties = group_and_liberties(board, x, y)
        assert group == {(x, y)}
        assert len(liberties) == expected_libs

    @pytest.mark.parametrize("stones,start,expected_group_size,expected_libs", [
        # Cross pattern: 5 stones, 8 liberties
        ([(2, 1, 1), (1, 2, 1), (2, 2, 1), (3, 2, 1), (2, 3, 1)], (2, 2), 5, 8),
        # Horizontal line: 3 stones, 8 liberties
        ([(1, 2, 1), (2, 2, 1), (3, 2, 1)], (2, 2), 3, 8),
        # Vertical line: 3 stones, 8 liberties
        ([(2, 1, 1), (2, 2, 1), (2, 3, 1)], (2, 2), 3, 8),
    ], ids=["cross", "horizontal", "vertical"])
    def test_connected_group_shapes(self, make_board, stones, start, expected_group_size, expected_libs):
        """Connected groups have shared liberties."""
        board = make_board(5, stones)
        group, liberties = group_and_liberties(board, *start)
        assert len(group) == expected_group_size
        assert len(liberties) == expected_libs

    @pytest.mark.parametrize("stones,target,expected_libs", [
        # Atari: white surrounded on 3 sides
        ([(1, 0, 1), (0, 1, 1), (1, 2, 1), (1, 1, -1)], (1, 1), 1),
        # Captured: white completely surrounded
        ([(1, 0, 1), (0, 1, 1), (1, 2, 1), (2, 1, 1), (1, 1, -1)], (1, 1), 0),
    ], ids=["atari", "captured"])
    def test_surrounded_stone(self, make_board, stones, target, expected_libs):
        """Surrounded stones have reduced or zero liberties."""
        board = make_board(5, stones)
        group, liberties = group_and_liberties(board, *target)
        assert len(liberties) == expected_libs

    def test_diagonal_stones_not_connected(self, make_board):
        """Diagonal stones are NOT connected in Go."""
        board = make_board(5, [(1, 1, 1), (2, 2, 1)])
        group, _ = group_and_liberties(board, 1, 1)
        assert len(group) == 1

    def test_separate_groups_independent(self, make_board):
        """Two separate stones form independent groups."""
        board = make_board(5, [(1, 1, 1), (3, 3, 1)])
        group1, _ = group_and_liberties(board, 1, 1)
        group2, _ = group_and_liberties(board, 3, 3)
        assert group1 == {(1, 1)}
        assert group2 == {(3, 3)}

    def test_large_l_shaped_group(self, make_board):
        """Large L-shaped group is fully connected."""
        # Vertical: (2,0) to (2,4), Horizontal: (2,4) to (5,4)
        stones = [(2, y, 1) for y in range(5)] + [(x, 4, 1) for x in range(3, 6)]
        board = make_board(9, stones)
        group, _ = group_and_liberties(board, 2, 2)
        assert len(group) == 8  # 5 + 3 (center shared)


class TestCountLiberties:
    """Tests for the count_liberties() function."""

    def test_empty_board_returns_empty(self, empty_board_5x5):
        """Empty board should return empty list."""
        assert count_liberties(empty_board_5x5) == []

    @pytest.mark.parametrize("color,expected_sign", [
        (1, 1),    # black: positive
        (-1, -1),  # white: negative
    ], ids=["black", "white"])
    def test_liberty_sign_by_color(self, make_board, color, expected_sign):
        """Black stones have positive liberties, white stones have negative."""
        board = make_board(5, [(2, 2, color)])
        result = count_liberties(board)
        assert len(result) == 1
        _, _, libs = result[0]
        assert libs == expected_sign * 4  # center has 4 liberties

    def test_output_is_1_indexed(self, make_board):
        """Output coordinates should be 1-indexed."""
        board = make_board(5, [(0, 0, 1)])
        result = count_liberties(board)
        x, y, _ = result[0]
        assert (x, y) == (1, 1)

    def test_connected_group_same_liberties(self, board_with_cross_pattern):
        """All stones in connected group report same liberty count."""
        result = count_liberties(board_with_cross_pattern)
        assert len(result) == 5
        for _, _, libs in result:
            assert libs == 8

    def test_adjacent_different_colors(self, make_board):
        """Adjacent stones of different colors block each other's liberties."""
        board = make_board(5, [(2, 2, 1), (3, 2, -1)])
        result = count_liberties(board)
        assert len(result) == 2
        for _, _, libs in result:
            assert abs(libs) == 3  # 4 - 1 blocked

    def test_multiple_separate_stones(self, make_board):
        """Multiple separate stones each get counted."""
        board = make_board(5, [(0, 0, 1), (4, 4, -1)])
        result = count_liberties(board)
        liberties = {(x, y, v) for x, y, v in result}
        assert (1, 1, 2) in liberties   # black corner
        assert (5, 5, -2) in liberties  # white corner


class TestEdgeCases:
    """Edge case tests for liberty functions."""

    @pytest.mark.parametrize("size,expected_libs", [
        (1, 0),  # 1x1: no neighbors
        (2, 2),  # 2x2: corner has 2
    ], ids=["1x1", "2x2"])
    def test_minimum_board_sizes(self, make_board, size, expected_libs):
        """Test with minimum board sizes."""
        board = make_board(size, [(0, 0, 1)])
        _, liberties = group_and_liberties(board, 0, 0)
        assert len(liberties) == expected_libs

    def test_full_board_no_liberties(self):
        """Completely filled board - all stones have no liberties."""
        board = [[1 if (x + y) % 2 == 0 else -1 for x in range(3)] for y in range(3)]
        result = count_liberties(board)
        for _, _, libs in result:
            assert libs == 0

    def test_large_board_19x19(self, make_board):
        """Test on standard 19x19 board (tengen position)."""
        board = make_board(19, [(9, 9, 1)])
        result = count_liberties(board)
        assert len(result) == 1
        x, y, libs = result[0]
        assert (x, y) == (10, 10)  # 1-indexed
        assert libs == 4

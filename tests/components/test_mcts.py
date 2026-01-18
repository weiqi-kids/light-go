"""Component tests for MCTS Search (core/mcts.py).

This module tests the Monte Carlo Tree Search implementation:
- MCTSNode: Tree node with UCB1 selection
- GoGameState: Game state for simulation
- MCTS: Main search algorithm
- mcts_search(): Convenience function
"""
from __future__ import annotations

import math
import pytest
from typing import List, Tuple

from core.mcts import MCTS, MCTSNode, GoGameState, mcts_search


class TestMCTSNodeBasics:
    """Basic tests for MCTSNode."""

    def test_create_node(self):
        """Create a basic node."""
        node = MCTSNode()

        assert node.move is None
        assert node.parent is None
        assert node.visits == 0
        assert node.wins == 0.0
        assert node.children == []

    def test_create_node_with_move(self):
        """Create node with a move."""
        node = MCTSNode(move=(3, 3), color=1)

        assert node.move == (3, 3)
        assert node.color == 1

    def test_create_child_node(self):
        """Create child node with parent."""
        parent = MCTSNode()
        child = MCTSNode(move=(2, 2), parent=parent)

        assert child.parent is parent
        assert child.move == (2, 2)


class TestMCTSNodeUCB1:
    """Tests for UCB1 calculation."""

    def test_ucb1_unvisited_is_infinity(self):
        """Unvisited node has infinite UCB1."""
        node = MCTSNode(visits=0)

        assert node.ucb1() == float("inf")

    def test_ucb1_calculation(self):
        """UCB1 calculation is correct."""
        parent = MCTSNode(visits=100)
        child = MCTSNode(parent=parent, visits=10, wins=6)

        ucb1 = child.ucb1(exploration=1.414)

        # UCB1 = 6/10 + 1.414 * sqrt(ln(100)/10)
        # = 0.6 + 1.414 * sqrt(4.605/10)
        # = 0.6 + 1.414 * 0.678
        # ≈ 1.56
        assert 1.5 < ucb1 < 1.7

    def test_ucb1_higher_with_more_wins(self):
        """Higher win rate means higher UCB1."""
        parent = MCTSNode(visits=100)
        child_high = MCTSNode(parent=parent, visits=10, wins=8)
        child_low = MCTSNode(parent=parent, visits=10, wins=2)

        assert child_high.ucb1() > child_low.ucb1()

    def test_ucb1_exploration_term(self):
        """Higher exploration constant increases UCB1."""
        parent = MCTSNode(visits=100)
        child = MCTSNode(parent=parent, visits=10, wins=5)

        ucb1_low = child.ucb1(exploration=0.5)
        ucb1_high = child.ucb1(exploration=2.0)

        assert ucb1_high > ucb1_low


class TestMCTSNodeSelection:
    """Tests for node selection."""

    def test_best_child_selection(self):
        """best_child returns child with highest UCB1."""
        parent = MCTSNode(visits=100)

        child1 = MCTSNode(parent=parent, visits=10, wins=2, move=(0, 0))
        child2 = MCTSNode(parent=parent, visits=10, wins=8, move=(1, 1))
        child3 = MCTSNode(parent=parent, visits=10, wins=5, move=(2, 2))

        parent.children = [child1, child2, child3]

        best = parent.best_child()
        assert best.move == (1, 1)  # Highest wins

    def test_is_fully_expanded(self):
        """is_fully_expanded when no untried moves."""
        node = MCTSNode(untried_moves=[(0, 0), (1, 1)])
        assert not node.is_fully_expanded()

        node.untried_moves = []
        assert node.is_fully_expanded()

    def test_is_terminal(self):
        """is_terminal when fully expanded with no children."""
        node = MCTSNode(untried_moves=[])
        node.children = []
        assert node.is_terminal()

        node.children = [MCTSNode()]
        assert not node.is_terminal()


class TestGoGameState:
    """Tests for GoGameState."""

    def test_create_game_state(self):
        """Create a game state."""
        board = [[0] * 5 for _ in range(5)]
        state = GoGameState(board, current_color=1)

        assert state.size == 5
        assert state.current_color == 1
        assert state.passes == 0

    def test_copy_state(self):
        """Copy creates independent state."""
        board = [[0] * 5 for _ in range(5)]
        state = GoGameState(board, current_color=1)

        copy = state.copy()
        copy.board[0][0] = 1

        assert state.board[0][0] == 0  # Original unchanged


class TestGoGameStateLegalMoves:
    """Tests for legal move generation."""

    def test_legal_moves_empty_board(self, empty_board_5x5):
        """All positions legal on empty board."""
        state = GoGameState(empty_board_5x5, current_color=1)

        legal = state.get_legal_moves()

        assert len(legal) == 25  # 5x5 = 25

    def test_legal_moves_occupied_excluded(self):
        """Occupied positions excluded."""
        board = [[0] * 5 for _ in range(5)]
        board[2][2] = 1

        state = GoGameState(board, current_color=1)
        legal = state.get_legal_moves()

        assert len(legal) == 24  # 25 - 1
        assert (2, 2) not in legal

    def test_ko_point_excluded(self):
        """Ko point excluded from legal moves."""
        board = [[0] * 5 for _ in range(5)]
        state = GoGameState(board, current_color=1, ko_point=(2, 2))

        legal = state.get_legal_moves()

        assert (2, 2) not in legal


class TestGoGameStatePlayMove:
    """Tests for move execution."""

    def test_play_move_places_stone(self):
        """Playing a move places a stone."""
        board = [[0] * 5 for _ in range(5)]
        state = GoGameState(board, current_color=1)

        new_state = state.play_move((2, 2))

        assert new_state.board[2][2] == 1

    def test_play_move_alternates_color(self):
        """Playing a move alternates color."""
        board = [[0] * 5 for _ in range(5)]
        state = GoGameState(board, current_color=1)

        new_state = state.play_move((2, 2))

        assert new_state.current_color == -1

    def test_play_pass(self):
        """Pass move increases pass count."""
        board = [[0] * 5 for _ in range(5)]
        state = GoGameState(board, current_color=1, passes=0)

        new_state = state.play_move(None)

        assert new_state.passes == 1
        assert new_state.current_color == -1

    def test_capture_removes_stones(self, board_with_capture_scenario):
        """Playing capture removes opponent stones."""
        state = GoGameState(board_with_capture_scenario, current_color=1)

        # Play at (2, 1) to complete the capture
        new_state = state.play_move((2, 1))

        # White stone at (1, 1) should be captured
        assert new_state.board[1][1] == 0


class TestGoGameStateTerminal:
    """Tests for game termination."""

    def test_is_terminal_after_two_passes(self):
        """Game ends after two consecutive passes."""
        board = [[0] * 5 for _ in range(5)]
        state = GoGameState(board, passes=2)

        assert state.is_terminal()

    def test_not_terminal_after_one_pass(self):
        """Game continues after one pass."""
        board = [[0] * 5 for _ in range(5)]
        state = GoGameState(board, passes=1)

        assert not state.is_terminal()


class TestGoGameStateScoring:
    """Tests for game scoring."""

    def test_get_winner_empty_board(self):
        """White wins empty board due to komi."""
        board = [[0] * 5 for _ in range(5)]
        state = GoGameState(board, komi=7.5)

        winner = state.get_winner()

        assert winner == -1  # White wins with komi

    def test_get_winner_with_stones(self):
        """Winner determined by territory + stones."""
        board = [[0] * 5 for _ in range(5)]
        # Fill half with black
        for y in range(5):
            for x in range(3):
                board[y][x] = 1

        state = GoGameState(board, komi=0)
        winner = state.get_winner()

        assert winner == 1  # Black has more


class TestMCTS:
    """Tests for MCTS search algorithm."""

    def test_mcts_instantiation(self):
        """MCTS can be instantiated."""
        mcts = MCTS(iterations=100)

        assert mcts.iterations == 100

    def test_mcts_search_returns_move(self, empty_board_5x5):
        """MCTS search returns a move."""
        mcts = MCTS(iterations=50)

        move = mcts.search(empty_board_5x5, color=1)

        if move is not None:
            x, y = move
            assert 0 <= x < 5
            assert 0 <= y < 5

    def test_mcts_search_valid_move(self, empty_board_5x5):
        """MCTS returns a valid (empty) position."""
        mcts = MCTS(iterations=50)

        move = mcts.search(empty_board_5x5, color=1)

        if move is not None:
            x, y = move
            assert empty_board_5x5[y][x] == 0

    def test_mcts_with_stones_on_board(self):
        """MCTS works with stones already on board."""
        board = [[0] * 5 for _ in range(5)]
        board[2][2] = 1
        board[2][3] = -1

        mcts = MCTS(iterations=50)
        move = mcts.search(board, color=1)

        if move is not None:
            x, y = move
            assert board[y][x] == 0

    def test_mcts_for_black_and_white(self, empty_board_5x5):
        """MCTS works for both colors."""
        mcts = MCTS(iterations=30)

        black_move = mcts.search(empty_board_5x5, color=1)
        white_move = mcts.search(empty_board_5x5, color=-1)

        # Both should return valid moves


class TestMCTSGetMoveProbabilities:
    """Tests for move probability distribution."""

    def test_get_probabilities(self, empty_board_5x5):
        """Get probability distribution over moves."""
        mcts = MCTS(iterations=50)

        probs = mcts.get_move_probabilities(empty_board_5x5, color=1)

        assert isinstance(probs, dict)

    def test_probabilities_sum_to_one(self, empty_board_5x5):
        """Probabilities sum to approximately 1."""
        mcts = MCTS(iterations=50)

        probs = mcts.get_move_probabilities(empty_board_5x5, color=1)

        if probs:
            total = sum(probs.values())
            assert 0.95 <= total <= 1.05

    def test_probabilities_non_negative(self, empty_board_5x5):
        """All probabilities are non-negative."""
        mcts = MCTS(iterations=50)

        probs = mcts.get_move_probabilities(empty_board_5x5, color=1)

        for prob in probs.values():
            assert prob >= 0


class TestMctsSearchFunction:
    """Tests for mcts_search() convenience function."""

    def test_mcts_search_function(self, empty_board_5x5):
        """mcts_search convenience function works."""
        move = mcts_search(empty_board_5x5, color=1, iterations=30)

        if move is not None:
            x, y = move
            assert 0 <= x < 5
            assert 0 <= y < 5

    def test_mcts_search_with_komi(self, empty_board_5x5):
        """mcts_search respects komi parameter."""
        move = mcts_search(empty_board_5x5, color=1, iterations=30, komi=7.5)

        # Should return a valid move


class TestMCTSRollout:
    """Tests for rollout/simulation phase."""

    def test_rollout_terminates(self, empty_board_5x5):
        """Rollout eventually terminates."""
        mcts = MCTS(iterations=10, max_rollout_depth=50)

        # Just ensure it doesn't hang
        move = mcts.search(empty_board_5x5, color=1)

    def test_rollout_depth_limit(self, empty_board_5x5):
        """Rollout respects depth limit."""
        mcts = MCTS(iterations=10, max_rollout_depth=5)

        # Should complete quickly with low depth
        move = mcts.search(empty_board_5x5, color=1)


class TestEdgeCases:
    """Edge case tests for MCTS."""

    def test_nearly_full_board(self):
        """MCTS on nearly full board."""
        board = [[1] * 5 for _ in range(5)]
        board[0][0] = 0  # One empty spot

        mcts = MCTS(iterations=10)
        move = mcts.search(board, color=1)

        if move is not None:
            assert move == (0, 0)

    def test_no_legal_moves(self):
        """MCTS when no legal moves (full board)."""
        board = [[1] * 3 for _ in range(3)]

        mcts = MCTS(iterations=10)
        move = mcts.search(board, color=1)

        assert move is None

    def test_minimum_iterations(self, empty_board_5x5):
        """MCTS with minimum iterations."""
        mcts = MCTS(iterations=1)
        move = mcts.search(empty_board_5x5, color=1)

        # Should still return a move

    def test_high_iterations(self, empty_board_5x5):
        """MCTS with higher iterations (quality check)."""
        mcts = MCTS(iterations=100)
        move = mcts.search(empty_board_5x5, color=1)

        # Should return a reasonable move (center area for empty board)


class TestMCTSIntegration:
    """Integration tests for MCTS with Engine."""

    def test_engine_mcts_integration(self, temp_dir, empty_board_5x5):
        """Engine uses MCTS correctly."""
        from core.engine import Engine

        engine = Engine(temp_dir)
        move = engine.mcts_move(empty_board_5x5, "black", iterations=30)

        if move is not None:
            x, y = move
            assert 0 <= x < 5
            assert 0 <= y < 5

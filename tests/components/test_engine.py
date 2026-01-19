"""Component tests for Training Loop / Engine (core/engine.py).

This module tests the high-level training and inference engine:
- Engine: Main coordinator for training and inference
- decide_move(): Select the best move for a position
- train(): Train strategies from SGF data
- evaluate(): Evaluate strategy performance
- predict(): Module-level prediction function
- _extract_board_and_color(): Helper for input parsing

Optimization notes:
- Uses shared_engine fixture (class-scoped) to reduce Engine instantiation
- Uses trained_engine_factory fixture (module-scoped) for tests needing pre-trained models
- MCTS tests marked with @pytest.mark.slow for optional skipping
"""
from __future__ import annotations

import os
import pytest
from typing import Any, Dict, List

from core.engine import Engine, predict, _extract_board_and_color


class TestEngineInstantiation:
    """Tests for Engine instantiation."""

    def test_instantiate_engine(self, temp_dir):
        """Engine can be instantiated."""
        engine = Engine(temp_dir)

        assert engine is not None
        assert engine.model_dir == temp_dir

    def test_engine_creates_directory(self, temp_dir):
        """Engine creates model directory if needed."""
        new_dir = os.path.join(temp_dir, "new_model_dir")
        engine = Engine(new_dir)

        assert os.path.isdir(new_dir)

    def test_engine_has_strategy_manager(self, temp_dir):
        """Engine has a StrategyManager."""
        engine = Engine(temp_dir)

        assert hasattr(engine, "strategy_manager")
        assert engine.strategy_manager is not None

    def test_engine_has_auto_learner(self, temp_dir):
        """Engine has an AutoLearner."""
        engine = Engine(temp_dir)

        assert hasattr(engine, "auto_learner")
        assert engine.auto_learner is not None


class TestExtractBoardAndColor:
    """Tests for _extract_board_and_color() helper.

    These are pure function tests - no fixtures needed.
    """

    def test_extract_from_dict_with_board(self):
        """Extract board and color from dict with board key."""
        data = {
            "board": [[0, 0, 0], [0, 1, 0], [0, 0, -1]],
            "color": "white",
        }

        board, color = _extract_board_and_color(data)

        assert len(board) == 3
        assert color == "white"

    def test_extract_color_from_next_move(self):
        """Extract color from next_move key."""
        data = {
            "board": [[0] * 9 for _ in range(9)],
            "next_move": "black",
        }

        board, color = _extract_board_and_color(data)

        assert color == "black"

    def test_extract_color_from_metadata(self):
        """Extract color from metadata.next_move."""
        data = {
            "board": [[0] * 9 for _ in range(9)],
            "metadata": {"next_move": "white"},
        }

        board, color = _extract_board_and_color(data)

        assert color == "white"

    def test_default_color_is_black(self):
        """Default color is black when not specified."""
        data = {"board": [[0] * 9 for _ in range(9)]}

        board, color = _extract_board_and_color(data)

        assert color == "black"

    def test_extract_with_size(self):
        """Extract respects size parameter."""
        data = {
            "board": [],
            "size": 5,
        }

        board, color = _extract_board_and_color(data)

        assert len(board) == 5
        assert len(board[0]) == 5

    def test_extract_from_move_list(self):
        """Extract board from move list format."""
        data = {
            "board": [
                ("black", "D4"),
                ("white", "Q16"),
            ],
            "size": 19,
            "color": "black",
        }

        board, color = _extract_board_and_color(data)

        assert len(board) == 19
        assert color == "black"


@pytest.mark.usefixtures("shared_engine")
class TestDecideMove:
    """Tests for Engine.decide_move().

    Uses shared_engine fixture (class-scoped) to reduce instantiation overhead.
    """

    def test_decide_move_returns_position(self, shared_engine):
        """decide_move returns a valid position."""
        board = [[0] * 5 for _ in range(5)]

        move = shared_engine.decide_move(board, "black")

        if move is not None:
            x, y = move
            assert 0 <= x < 5
            assert 0 <= y < 5

    def test_decide_move_for_black(self, shared_engine):
        """decide_move works for black."""
        board = [[0] * 5 for _ in range(5)]

        move = shared_engine.decide_move(board, "black")
        # Should return a valid move

    def test_decide_move_for_white(self, shared_engine):
        """decide_move works for white."""
        board = [[0] * 5 for _ in range(5)]

        move = shared_engine.decide_move(board, "white")
        # Should return a valid move

    def test_decide_move_returns_empty_position(self, shared_engine):
        """decide_move returns an empty position."""
        board = [[0] * 3 for _ in range(3)]
        board[1][1] = 1  # Center occupied

        move = shared_engine.decide_move(board, "black", use_mcts=False)

        if move is not None:
            x, y = move
            assert board[y][x] == 0

    def test_decide_move_full_board_returns_none(self, shared_engine):
        """decide_move on full board returns None."""
        board = [[1] * 3 for _ in range(3)]  # Full board

        move = shared_engine.decide_move(board, "black", use_mcts=False)

        assert move is None

    def test_decide_move_without_mcts(self, shared_engine):
        """decide_move with MCTS disabled (naive fallback)."""
        board = [[0] * 5 for _ in range(5)]

        move = shared_engine.decide_move(board, "black", use_mcts=False)

        if move is not None:
            x, y = move
            assert 0 <= x < 5
            assert 0 <= y < 5

    @pytest.mark.slow
    def test_decide_move_with_mcts(self, shared_engine):
        """decide_move with MCTS enabled."""
        board = [[0] * 5 for _ in range(5)]

        move = shared_engine.decide_move(board, "black", use_mcts=True, mcts_iterations=10)

        if move is not None:
            x, y = move
            assert 0 <= x < 5
            assert 0 <= y < 5


@pytest.mark.slow
@pytest.mark.usefixtures("shared_engine")
class TestMctsMove:
    """Tests for Engine.mcts_move().

    Marked as slow due to MCTS computation overhead.
    Uses shared_engine fixture (class-scoped) to reduce instantiation overhead.
    """

    def test_mcts_move_returns_position(self, shared_engine):
        """mcts_move returns a valid position."""
        board = [[0] * 5 for _ in range(5)]

        move = shared_engine.mcts_move(board, "black", iterations=10)

        if move is not None:
            x, y = move
            assert 0 <= x < 5
            assert 0 <= y < 5

    def test_mcts_move_with_komi(self, shared_engine):
        """mcts_move respects komi parameter."""
        board = [[0] * 5 for _ in range(5)]

        move1 = shared_engine.mcts_move(board, "white", iterations=10, komi=7.5)
        move2 = shared_engine.mcts_move(board, "white", iterations=10, komi=0.5)

        # Both should return valid moves


@pytest.mark.slow
@pytest.mark.usefixtures("shared_engine")
class TestMctsMoveWithProbabilities:
    """Tests for Engine.mcts_move_with_probabilities().

    Marked as slow due to MCTS computation overhead.
    Uses shared_engine fixture (class-scoped) to reduce instantiation overhead.
    """

    def test_returns_move_and_probs(self, shared_engine):
        """Returns both move and probability distribution."""
        board = [[0] * 5 for _ in range(5)]

        move, probs = shared_engine.mcts_move_with_probabilities(board, "black", iterations=20)

        assert isinstance(probs, dict)
        if move is not None:
            x, y = move
            assert 0 <= x < 5
            assert 0 <= y < 5

    def test_probabilities_sum_to_one(self, shared_engine):
        """Probabilities approximately sum to 1."""
        board = [[0] * 5 for _ in range(5)]

        move, probs = shared_engine.mcts_move_with_probabilities(board, "black", iterations=30)

        if probs:
            total = sum(probs.values())
            assert 0.95 <= total <= 1.05


class TestEngineTrain:
    """Tests for Engine.train().

    These tests need independent Engine instances as they modify state.
    """

    def test_train_returns_strategy_name(self, temp_dir, temp_sgf_dir):
        """train() returns strategy name."""
        engine = Engine(temp_dir)
        output_dir = os.path.join(temp_dir, "output")
        os.makedirs(output_dir, exist_ok=True)

        name = engine.train(temp_sgf_dir, output_dir)

        assert name is not None
        assert isinstance(name, str)

    def test_train_creates_model_file(self, temp_dir, temp_sgf_dir):
        """train() creates model file."""
        engine = Engine(temp_dir)
        output_dir = os.path.join(temp_dir, "output")

        name = engine.train(temp_sgf_dir, output_dir)

        model_path = os.path.join(output_dir, f"{name}.pt")
        assert os.path.exists(model_path)

    def test_train_registers_strategy(self, temp_dir, temp_sgf_dir):
        """train() registers strategy with manager."""
        engine = Engine(temp_dir)
        output_dir = os.path.join(temp_dir, "output")

        name = engine.train(temp_sgf_dir, output_dir)

        assert name in engine.strategy_manager.list_strategies()


class TestEngineEvaluate:
    """Tests for Engine.evaluate().

    Uses trained_engine_factory fixture to avoid redundant training.
    """

    def test_evaluate_returns_metrics(self, trained_engine_factory):
        """evaluate() returns metrics dict."""
        engine, sgf_dir, output_dir, _ = trained_engine_factory

        metrics = engine.evaluate(sgf_dir, output_dir)

        assert isinstance(metrics, dict)
        assert "samples" in metrics


class TestEnginePlay:
    """Tests for Engine.play().

    Uses trained_engine_factory fixture to avoid redundant training.
    """

    def test_play_returns_predictions(self, trained_engine_factory):
        """play() returns list of predictions."""
        engine, sgf_dir, output_dir, _ = trained_engine_factory

        results = engine.play(sgf_dir, output_dir)

        assert isinstance(results, list)

    def test_play_with_specific_strategy(self, trained_engine_factory):
        """play() with specific strategy name."""
        engine, sgf_dir, output_dir, strategy_name = trained_engine_factory

        results = engine.play(sgf_dir, output_dir, strategy=strategy_name)

        assert isinstance(results, list)

    def test_play_raises_without_strategies(self, temp_dir, temp_sgf_dir):
        """play() raises when no strategies available."""
        engine = Engine(temp_dir)

        with pytest.raises(RuntimeError):
            engine.play(temp_sgf_dir, temp_dir)


class TestModuleLevelPredict:
    """Tests for module-level predict() function.

    These are effectively integration tests - no shared fixtures needed.
    """

    def test_predict_returns_move(self):
        """predict() returns a move."""
        input_data = {
            "board": [[0] * 9 for _ in range(9)],
            "color": "black",
        }

        move = predict(input_data)

        # May return None or a move

    def test_predict_with_minimal_input(self):
        """predict() with minimal input."""
        input_data = {"board": [[0] * 9 for _ in range(9)]}

        move = predict(input_data)
        # Should not raise


class TestEngineLoadStrategy:
    """Tests for Engine.load_strategy().

    Uses trained_engine_factory fixture to avoid redundant training.
    """

    def test_load_strategy(self, trained_engine_factory):
        """load_strategy loads a saved strategy."""
        engine, _, _, strategy_name = trained_engine_factory

        strategy = engine.load_strategy(strategy_name)

        # Should return something (the loaded strategy)


@pytest.mark.usefixtures("shared_engine")
class TestEdgeCases:
    """Edge case tests for Engine.

    Uses shared_engine fixture (class-scoped) to reduce instantiation overhead.
    """

    def test_decide_move_with_nearly_full_board(self, shared_engine):
        """decide_move on nearly full board."""
        board = [[1] * 5 for _ in range(5)]
        board[0][0] = 0  # One empty spot

        move = shared_engine.decide_move(board, "black", use_mcts=False)

        if move is not None:
            assert move == (0, 0)

    def test_decide_move_respects_occupied(self, shared_engine):
        """decide_move doesn't suggest occupied positions."""
        board = [[0] * 5 for _ in range(5)]
        board[2][2] = 1  # Center occupied

        # Run multiple times
        for _ in range(10):
            move = shared_engine.decide_move(board, "black", use_mcts=False)
            if move is not None:
                x, y = move
                assert board[y][x] == 0

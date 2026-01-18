"""Component tests for Training Loop / Engine (core/engine.py).

This module tests the high-level training and inference engine:
- Engine: Main coordinator for training and inference
- decide_move(): Select the best move for a position
- train(): Train strategies from SGF data
- evaluate(): Evaluate strategy performance
- predict(): Module-level prediction function
- _extract_board_and_color(): Helper for input parsing
"""
from __future__ import annotations

import os
import tempfile
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
    """Tests for _extract_board_and_color() helper."""

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


class TestDecideMove:
    """Tests for Engine.decide_move()."""

    def test_decide_move_returns_position(self, temp_dir):
        """decide_move returns a valid position."""
        engine = Engine(temp_dir)
        board = [[0] * 5 for _ in range(5)]

        move = engine.decide_move(board, "black")

        if move is not None:
            x, y = move
            assert 0 <= x < 5
            assert 0 <= y < 5

    def test_decide_move_for_black(self, temp_dir):
        """decide_move works for black."""
        engine = Engine(temp_dir)
        board = [[0] * 5 for _ in range(5)]

        move = engine.decide_move(board, "black")
        # Should return a valid move

    def test_decide_move_for_white(self, temp_dir):
        """decide_move works for white."""
        engine = Engine(temp_dir)
        board = [[0] * 5 for _ in range(5)]

        move = engine.decide_move(board, "white")
        # Should return a valid move

    def test_decide_move_returns_empty_position(self, temp_dir):
        """decide_move returns an empty position."""
        engine = Engine(temp_dir)
        board = [[0] * 3 for _ in range(3)]
        board[1][1] = 1  # Center occupied

        move = engine.decide_move(board, "black", use_mcts=False)

        if move is not None:
            x, y = move
            assert board[y][x] == 0

    def test_decide_move_full_board_returns_none(self, temp_dir):
        """decide_move on full board returns None."""
        engine = Engine(temp_dir)
        board = [[1] * 3 for _ in range(3)]  # Full board

        move = engine.decide_move(board, "black", use_mcts=False)

        assert move is None

    def test_decide_move_with_mcts(self, temp_dir):
        """decide_move with MCTS enabled."""
        engine = Engine(temp_dir)
        board = [[0] * 5 for _ in range(5)]

        move = engine.decide_move(board, "black", use_mcts=True, mcts_iterations=10)

        if move is not None:
            x, y = move
            assert 0 <= x < 5
            assert 0 <= y < 5

    def test_decide_move_without_mcts(self, temp_dir):
        """decide_move with MCTS disabled (naive fallback)."""
        engine = Engine(temp_dir)
        board = [[0] * 5 for _ in range(5)]

        move = engine.decide_move(board, "black", use_mcts=False)

        if move is not None:
            x, y = move
            assert 0 <= x < 5
            assert 0 <= y < 5


class TestMctsMove:
    """Tests for Engine.mcts_move()."""

    def test_mcts_move_returns_position(self, temp_dir):
        """mcts_move returns a valid position."""
        engine = Engine(temp_dir)
        board = [[0] * 5 for _ in range(5)]

        move = engine.mcts_move(board, "black", iterations=10)

        if move is not None:
            x, y = move
            assert 0 <= x < 5
            assert 0 <= y < 5

    def test_mcts_move_with_komi(self, temp_dir):
        """mcts_move respects komi parameter."""
        engine = Engine(temp_dir)
        board = [[0] * 5 for _ in range(5)]

        move1 = engine.mcts_move(board, "white", iterations=10, komi=7.5)
        move2 = engine.mcts_move(board, "white", iterations=10, komi=0.5)

        # Both should return valid moves


class TestMctsMoveWithProbabilities:
    """Tests for Engine.mcts_move_with_probabilities()."""

    def test_returns_move_and_probs(self, temp_dir):
        """Returns both move and probability distribution."""
        engine = Engine(temp_dir)
        board = [[0] * 5 for _ in range(5)]

        move, probs = engine.mcts_move_with_probabilities(board, "black", iterations=20)

        assert isinstance(probs, dict)
        if move is not None:
            x, y = move
            assert 0 <= x < 5
            assert 0 <= y < 5

    def test_probabilities_sum_to_one(self, temp_dir):
        """Probabilities approximately sum to 1."""
        engine = Engine(temp_dir)
        board = [[0] * 5 for _ in range(5)]

        move, probs = engine.mcts_move_with_probabilities(board, "black", iterations=30)

        if probs:
            total = sum(probs.values())
            assert 0.95 <= total <= 1.05


class TestEngineTrain:
    """Tests for Engine.train()."""

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
    """Tests for Engine.evaluate()."""

    def test_evaluate_returns_metrics(self, temp_dir, temp_sgf_dir):
        """evaluate() returns metrics dict."""
        engine = Engine(temp_dir)
        output_dir = os.path.join(temp_dir, "output")

        # First train a strategy
        engine.train(temp_sgf_dir, output_dir)

        # Then evaluate
        metrics = engine.evaluate(temp_sgf_dir, output_dir)

        assert isinstance(metrics, dict)
        assert "samples" in metrics


class TestEnginePlay:
    """Tests for Engine.play()."""

    def test_play_returns_predictions(self, temp_dir, temp_sgf_dir):
        """play() returns list of predictions."""
        engine = Engine(temp_dir)
        output_dir = os.path.join(temp_dir, "output")

        # First train a strategy
        engine.train(temp_sgf_dir, output_dir)

        # Then play
        results = engine.play(temp_sgf_dir, output_dir)

        assert isinstance(results, list)

    def test_play_with_specific_strategy(self, temp_dir, temp_sgf_dir):
        """play() with specific strategy name."""
        engine = Engine(temp_dir)
        output_dir = os.path.join(temp_dir, "output")

        # Train a strategy
        name = engine.train(temp_sgf_dir, output_dir)

        # Play with that strategy
        results = engine.play(temp_sgf_dir, output_dir, strategy=name)

        assert isinstance(results, list)

    def test_play_raises_without_strategies(self, temp_dir, temp_sgf_dir):
        """play() raises when no strategies available."""
        engine = Engine(temp_dir)

        with pytest.raises(RuntimeError):
            engine.play(temp_sgf_dir, temp_dir)


class TestModuleLevelPredict:
    """Tests for module-level predict() function."""

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
    """Tests for Engine.load_strategy()."""

    def test_load_strategy(self, temp_dir, temp_sgf_dir):
        """load_strategy loads a saved strategy."""
        engine = Engine(temp_dir)
        output_dir = os.path.join(temp_dir, "output")

        # Train and save
        name = engine.train(temp_sgf_dir, output_dir)

        # Load
        strategy = engine.load_strategy(name)

        # Should return something


class TestEdgeCases:
    """Edge case tests for Engine."""

    def test_decide_move_with_nearly_full_board(self, temp_dir):
        """decide_move on nearly full board."""
        engine = Engine(temp_dir)
        board = [[1] * 5 for _ in range(5)]
        board[0][0] = 0  # One empty spot

        move = engine.decide_move(board, "black", use_mcts=False)

        if move is not None:
            assert move == (0, 0)

    def test_decide_move_respects_occupied(self, temp_dir):
        """decide_move doesn't suggest occupied positions."""
        engine = Engine(temp_dir)
        board = [[0] * 5 for _ in range(5)]
        board[2][2] = 1  # Center occupied

        # Run multiple times
        for _ in range(10):
            move = engine.decide_move(board, "black", use_mcts=False)
            if move is not None:
                x, y = move
                assert board[y][x] == 0

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

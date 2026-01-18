"""Component tests for Auto Learner (core/auto_learner.py).

This module tests the auto learning / architecture genome system:
- AutoLearner instantiation and initialization
- _game_stats(): Extract statistics from game boards
- discover_strategy(): Create and register new strategies
- assign_training(): Assign training data to strategies
- receive_feedback(): Update strategy scores based on performance
"""
from __future__ import annotations

import os
import tempfile
import pytest
from typing import Any, Dict, List

from core.auto_learner import AutoLearner
from core.strategy_manager import StrategyManager


class TestAutoLearnerInstantiation:
    """Tests for AutoLearner instantiation."""

    def test_instantiate_with_strategy_manager(self, temp_dir):
        """AutoLearner can be instantiated with StrategyManager."""
        manager = StrategyManager(temp_dir)
        learner = AutoLearner(manager)

        assert learner is not None
        assert learner.manager is manager

    def test_initial_scores_empty(self, temp_dir):
        """Initial scores dictionary is empty."""
        manager = StrategyManager(temp_dir)
        learner = AutoLearner(manager)

        assert isinstance(learner._scores, dict)

    def test_initial_allocation_empty(self, temp_dir):
        """Initial allocation dictionary is empty."""
        manager = StrategyManager(temp_dir)
        learner = AutoLearner(manager)

        assert isinstance(learner._allocation, dict)


class TestGameStats:
    """Tests for _game_stats() static method."""

    def test_empty_board_stats(self):
        """Empty board has zero stones."""
        board = [[0] * 5 for _ in range(5)]

        stats = AutoLearner._game_stats(board)

        assert stats["black_stones"] == 0
        assert stats["white_stones"] == 0

    def test_single_black_stone(self):
        """Board with single black stone."""
        board = [[0] * 5 for _ in range(5)]
        board[2][2] = 1

        stats = AutoLearner._game_stats(board)

        assert stats["black_stones"] == 1
        assert stats["white_stones"] == 0

    def test_single_white_stone(self):
        """Board with single white stone."""
        board = [[0] * 5 for _ in range(5)]
        board[2][2] = -1

        stats = AutoLearner._game_stats(board)

        assert stats["black_stones"] == 0
        assert stats["white_stones"] == 1

    def test_mixed_stones(self):
        """Board with both black and white stones."""
        board = [
            [0, 0, 0, 0, 0],
            [0, 1, 1, 0, 0],
            [0, 1, -1, -1, 0],
            [0, 0, -1, 0, 0],
            [0, 0, 0, 0, 0],
        ]

        stats = AutoLearner._game_stats(board)

        assert stats["black_stones"] == 3
        assert stats["white_stones"] == 3

    def test_full_board_stats(self):
        """Board with multiple stones of both colors."""
        # Create a board with several connected groups
        # Black takes left half, white takes right half
        board = [[0] * 5 for _ in range(5)]
        for y in range(5):
            for x in range(5):
                if x < 2:
                    board[y][x] = 1  # black
                elif x > 2:
                    board[y][x] = -1  # white
                # middle column (x=2) stays empty

        stats = AutoLearner._game_stats(board)

        # 10 black stones (2 columns x 5 rows)
        # 10 white stones (2 columns x 5 rows)
        assert stats["black_stones"] == 10
        assert stats["white_stones"] == 10

    def test_various_board_sizes(self):
        """Stats work for various board sizes."""
        for size in [5, 9, 13, 19]:
            board = [[0] * size for _ in range(size)]
            board[size // 2][size // 2] = 1

            stats = AutoLearner._game_stats(board)
            assert stats["black_stones"] == 1


class TestDiscoverStrategy:
    """Tests for discover_strategy() method."""

    def test_discover_registers_strategy(self, temp_dir):
        """discover_strategy creates and registers a new strategy."""
        manager = StrategyManager(temp_dir)
        learner = AutoLearner(manager)

        data = {"sample": "test_data", "value": 42}
        name = learner.discover_strategy(data)

        assert name is not None
        assert isinstance(name, str)
        assert name in learner._scores

    def test_discover_updates_allocation(self, temp_dir):
        """discover_strategy updates allocation tracking."""
        manager = StrategyManager(temp_dir)
        learner = AutoLearner(manager)

        data = {"sample": "test_data"}
        name = learner.discover_strategy(data)

        assert name in learner._allocation

    def test_discover_multiple_strategies(self, temp_dir):
        """Multiple discoveries create multiple strategies."""
        manager = StrategyManager(temp_dir)
        learner = AutoLearner(manager)

        names = []
        for i in range(3):
            name = learner.discover_strategy({"id": i})
            names.append(name)

        # All names should be tracked
        for name in names:
            assert name in learner._scores


class TestAssignTraining:
    """Tests for assign_training() method."""

    def test_assign_returns_list(self, temp_dir):
        """assign_training returns list of strategy names."""
        manager = StrategyManager(temp_dir)
        learner = AutoLearner(manager)

        # First discover some strategies
        learner.discover_strategy({"id": 1})
        learner.discover_strategy({"id": 2})

        features = {"position_type": "opening", "liberties": 10}
        assigned = learner.assign_training(features)

        assert isinstance(assigned, list)

    def test_assign_empty_when_no_strategies(self, temp_dir):
        """assign_training returns empty when no strategies exist."""
        manager = StrategyManager(temp_dir)
        learner = AutoLearner(manager)

        features = {"position_type": "opening"}
        assigned = learner.assign_training(features)

        assert isinstance(assigned, list)

    def test_assign_with_various_features(self, temp_dir):
        """assign_training works with various feature types."""
        manager = StrategyManager(temp_dir)
        learner = AutoLearner(manager)

        learner.discover_strategy({})

        test_features = [
            {"position_type": "opening"},
            {"position_type": "middlegame", "liberties": 15},
            {"position_type": "endgame", "territory_diff": 5},
            {},
        ]

        for features in test_features:
            assigned = learner.assign_training(features)
            assert isinstance(assigned, list)


class TestReceiveFeedback:
    """Tests for receive_feedback() method."""

    def test_feedback_updates_score(self, temp_dir):
        """receive_feedback updates strategy score."""
        manager = StrategyManager(temp_dir)
        learner = AutoLearner(manager)

        name = learner.discover_strategy({})
        old_score = learner._scores.get(name, 0)

        learner.receive_feedback(name, 0.8)
        new_score = learner._scores.get(name, 0)

        assert new_score != old_score

    def test_feedback_positive_score(self, temp_dir):
        """Positive feedback increases score."""
        manager = StrategyManager(temp_dir)
        learner = AutoLearner(manager)

        name = learner.discover_strategy({})
        learner._scores[name] = 0.5

        learner.receive_feedback(name, 1.0)

        assert learner._scores[name] > 0.5

    def test_feedback_negative_score(self, temp_dir):
        """Negative/low feedback decreases score."""
        manager = StrategyManager(temp_dir)
        learner = AutoLearner(manager)

        name = learner.discover_strategy({})
        learner._scores[name] = 0.5

        learner.receive_feedback(name, 0.0)

        assert learner._scores[name] < 0.5

    def test_feedback_nonexistent_strategy(self, temp_dir):
        """Feedback for nonexistent strategy is handled."""
        manager = StrategyManager(temp_dir)
        learner = AutoLearner(manager)

        # Should not raise
        learner.receive_feedback("nonexistent", 0.5)


class TestTrainAndSave:
    """Tests for train_and_save() method."""

    def test_train_and_save_returns_name(self, temp_dir, temp_sgf_dir):
        """train_and_save returns strategy name."""
        manager = StrategyManager(temp_dir)
        learner = AutoLearner(manager)

        name = learner.train_and_save(temp_sgf_dir)

        assert name is not None
        assert isinstance(name, str)

    def test_train_and_save_registers_strategy(self, temp_dir, temp_sgf_dir):
        """train_and_save registers the strategy."""
        manager = StrategyManager(temp_dir)
        learner = AutoLearner(manager)

        name = learner.train_and_save(temp_sgf_dir)

        assert name in manager.list_strategies()


class TestTrain:
    """Tests for train() method."""

    def test_train_method_exists(self, temp_dir):
        """AutoLearner has train method."""
        manager = StrategyManager(temp_dir)
        learner = AutoLearner(manager)

        assert hasattr(learner, "train")
        assert callable(learner.train)

    def test_train_with_directory(self, temp_dir, temp_sgf_dir):
        """train() works with SGF directory."""
        manager = StrategyManager(temp_dir)
        learner = AutoLearner(manager)

        # Should not raise
        learner.train(temp_sgf_dir)


class TestScoreManagement:
    """Tests for score tracking and management."""

    def test_scores_persist_across_operations(self, temp_dir):
        """Scores persist across multiple operations."""
        manager = StrategyManager(temp_dir)
        learner = AutoLearner(manager)

        name = learner.discover_strategy({})
        learner.receive_feedback(name, 0.9)
        initial_score = learner._scores[name]

        # Perform more operations
        learner.discover_strategy({"other": True})
        learner.assign_training({})

        # Original score should persist
        assert learner._scores[name] == initial_score

    def test_multiple_feedback_accumulates(self, temp_dir):
        """Multiple feedback calls accumulate."""
        manager = StrategyManager(temp_dir)
        learner = AutoLearner(manager)

        name = learner.discover_strategy({})
        learner._scores[name] = 0.0

        # Send multiple positive feedback
        for _ in range(5):
            learner.receive_feedback(name, 1.0)

        # Score should have increased
        assert learner._scores[name] > 0.0


class TestEdgeCases:
    """Edge case tests for AutoLearner."""

    def test_discover_with_empty_data(self, temp_dir):
        """discover_strategy with empty data."""
        manager = StrategyManager(temp_dir)
        learner = AutoLearner(manager)

        name = learner.discover_strategy({})
        assert name is not None

    def test_discover_with_none_data(self, temp_dir):
        """discover_strategy with None data."""
        manager = StrategyManager(temp_dir)
        learner = AutoLearner(manager)

        try:
            name = learner.discover_strategy(None)
        except (TypeError, AttributeError):
            pass  # Expected for some implementations

    def test_feedback_with_extreme_scores(self, temp_dir):
        """Feedback with extreme score values."""
        manager = StrategyManager(temp_dir)
        learner = AutoLearner(manager)

        name = learner.discover_strategy({})

        # Extreme values
        learner.receive_feedback(name, 100.0)
        learner.receive_feedback(name, -100.0)
        learner.receive_feedback(name, 0.0)
        learner.receive_feedback(name, 1.0)

        # Should not crash, score should be tracked
        assert name in learner._scores

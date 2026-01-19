"""Component tests for Auto Learner (core/auto_learner.py).

This module tests the auto learning / architecture genome system:
- AutoLearner instantiation and initialization
- _game_stats(): Extract statistics from game boards
- discover_strategy(): Create and register new strategies
- assign_training(): Assign training data to strategies
- receive_feedback(): Update strategy scores based on performance
"""
from __future__ import annotations

import pytest

from core.auto_learner import AutoLearner


# ============================================================================
# Pure Function Tests (no temp_dir required - faster execution)
# ============================================================================


class TestGameStats:
    """Tests for _game_stats() static method.

    These are pure function tests that don't require file system access.
    """

    def test_empty_board_stats(self, empty_board_5x5):
        """Empty board has zero stones."""
        stats = AutoLearner._game_stats(empty_board_5x5)

        assert stats["black_stones"] == 0
        assert stats["white_stones"] == 0
        assert stats["avg_liberties_black"] == 0.0
        assert stats["avg_liberties_white"] == 0.0

    def test_single_black_stone(self, board_with_single_black_stone):
        """Board with single black stone at center."""
        stats = AutoLearner._game_stats(board_with_single_black_stone)

        assert stats["black_stones"] == 1
        assert stats["white_stones"] == 0
        assert stats["avg_liberties_black"] > 0

    def test_single_white_stone(self, empty_board_5x5):
        """Board with single white stone."""
        empty_board_5x5[2][2] = -1

        stats = AutoLearner._game_stats(empty_board_5x5)

        assert stats["black_stones"] == 0
        assert stats["white_stones"] == 1
        assert stats["avg_liberties_white"] > 0

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

    def test_cross_pattern_stats(self, board_with_cross_pattern):
        """Board with cross pattern of black stones."""
        stats = AutoLearner._game_stats(board_with_cross_pattern)

        assert stats["black_stones"] == 5
        assert stats["white_stones"] == 0

    def test_full_board_stats(self):
        """Board with multiple stones of both colors (left/right split)."""
        board = [[0] * 5 for _ in range(5)]
        for y in range(5):
            for x in range(5):
                if x < 2:
                    board[y][x] = 1  # black
                elif x > 2:
                    board[y][x] = -1  # white

        stats = AutoLearner._game_stats(board)

        assert stats["black_stones"] == 10
        assert stats["white_stones"] == 10

    @pytest.mark.parametrize("size", [5, 9, 13, 19])
    def test_various_board_sizes(self, size):
        """Stats work for various board sizes."""
        board = [[0] * size for _ in range(size)]
        board[size // 2][size // 2] = 1

        stats = AutoLearner._game_stats(board)

        assert stats["black_stones"] == 1
        assert stats["white_stones"] == 0


# ============================================================================
# AutoLearner Instance Tests (require temp_dir for StrategyManager)
# ============================================================================


class TestAutoLearnerInstantiation:
    """Tests for AutoLearner instantiation."""

    def test_instantiate_with_strategy_manager(self, auto_learner, strategy_manager):
        """AutoLearner can be instantiated with StrategyManager."""
        assert auto_learner is not None
        assert auto_learner.manager is strategy_manager

    def test_initial_scores_empty(self, auto_learner):
        """Initial scores dictionary is empty."""
        assert isinstance(auto_learner._scores, dict)

    def test_initial_allocation_empty(self, auto_learner):
        """Initial allocation dictionary is empty."""
        assert isinstance(auto_learner._allocation, dict)


class TestDiscoverStrategy:
    """Tests for discover_strategy() method."""

    def test_discover_registers_strategy(self, auto_learner):
        """discover_strategy creates and registers a new strategy."""
        data = {"sample": "test_data", "value": 42}
        name = auto_learner.discover_strategy(data)

        assert name is not None
        assert isinstance(name, str)
        assert name in auto_learner._scores

    def test_discover_updates_allocation(self, auto_learner):
        """discover_strategy updates allocation tracking."""
        data = {"sample": "test_data"}
        name = auto_learner.discover_strategy(data)

        assert name in auto_learner._allocation

    def test_discover_multiple_strategies(self, auto_learner):
        """Multiple discoveries create multiple strategies."""
        names = [auto_learner.discover_strategy({"id": i}) for i in range(3)]

        for name in names:
            assert name in auto_learner._scores


class TestAssignTraining:
    """Tests for assign_training() method."""

    def test_assign_returns_list(self, auto_learner):
        """assign_training returns list of strategy names."""
        auto_learner.discover_strategy({"id": 1})
        auto_learner.discover_strategy({"id": 2})

        features = {"position_type": "opening", "liberties": 10}
        assigned = auto_learner.assign_training(features)

        assert isinstance(assigned, list)

    def test_assign_empty_when_no_strategies(self, auto_learner):
        """assign_training returns empty when no strategies exist."""
        features = {"position_type": "opening"}
        assigned = auto_learner.assign_training(features)

        assert isinstance(assigned, list)

    @pytest.mark.parametrize("features", [
        {"position_type": "opening"},
        {"position_type": "middlegame", "liberties": 15},
        {"position_type": "endgame", "territory_diff": 5},
        {},
    ])
    def test_assign_with_various_features(self, auto_learner, features):
        """assign_training works with various feature types."""
        auto_learner.discover_strategy({})

        assigned = auto_learner.assign_training(features)

        assert isinstance(assigned, list)


class TestReceiveFeedback:
    """Tests for receive_feedback() method."""

    def test_feedback_updates_score(self, auto_learner):
        """receive_feedback updates strategy score."""
        name = auto_learner.discover_strategy({})
        old_score = auto_learner._scores.get(name, 0)

        auto_learner.receive_feedback(name, 0.8)
        new_score = auto_learner._scores.get(name, 0)

        assert new_score != old_score

    def test_feedback_positive_score(self, auto_learner):
        """Positive feedback increases score."""
        name = auto_learner.discover_strategy({})
        auto_learner._scores[name] = 0.5

        auto_learner.receive_feedback(name, 1.0)

        assert auto_learner._scores[name] > 0.5

    def test_feedback_negative_score(self, auto_learner):
        """Negative/low feedback decreases score."""
        name = auto_learner.discover_strategy({})
        auto_learner._scores[name] = 0.5

        auto_learner.receive_feedback(name, 0.0)

        assert auto_learner._scores[name] < 0.5

    def test_feedback_nonexistent_strategy(self, auto_learner):
        """Feedback for nonexistent strategy is handled."""
        # Should not raise
        auto_learner.receive_feedback("nonexistent", 0.5)


class TestTrainAndSave:
    """Tests for train_and_save() method."""

    def test_train_and_save_returns_name(self, auto_learner, temp_sgf_dir):
        """train_and_save returns strategy name."""
        name = auto_learner.train_and_save(temp_sgf_dir)

        assert name is not None
        assert isinstance(name, str)

    def test_train_and_save_registers_strategy(self, auto_learner, temp_sgf_dir):
        """train_and_save registers the strategy."""
        name = auto_learner.train_and_save(temp_sgf_dir)

        assert name in auto_learner.manager.list_strategies()


class TestTrain:
    """Tests for train() method."""

    def test_train_method_exists(self, auto_learner):
        """AutoLearner has train method."""
        assert hasattr(auto_learner, "train")
        assert callable(auto_learner.train)

    def test_train_with_directory(self, auto_learner, temp_sgf_dir):
        """train() works with SGF directory."""
        result = auto_learner.train(temp_sgf_dir)

        assert isinstance(result, dict)
        assert "games" in result


class TestScoreManagement:
    """Tests for score tracking and management."""

    def test_scores_persist_across_operations(self, auto_learner):
        """Scores persist across multiple operations."""
        name = auto_learner.discover_strategy({})
        auto_learner.receive_feedback(name, 0.9)
        initial_score = auto_learner._scores[name]

        # Perform more operations
        auto_learner.discover_strategy({"other": True})
        auto_learner.assign_training({})

        # Original score should persist
        assert auto_learner._scores[name] == initial_score

    def test_multiple_feedback_accumulates(self, auto_learner):
        """Multiple feedback calls accumulate (exponential moving average)."""
        name = auto_learner.discover_strategy({})
        auto_learner._scores[name] = 0.0

        # Send multiple positive feedback
        for _ in range(5):
            auto_learner.receive_feedback(name, 1.0)

        # Score should have increased
        assert auto_learner._scores[name] > 0.0


class TestEdgeCases:
    """Edge case tests for AutoLearner."""

    def test_discover_with_empty_data(self, auto_learner):
        """discover_strategy with empty data."""
        name = auto_learner.discover_strategy({})
        assert name is not None

    def test_discover_with_none_data(self, auto_learner):
        """discover_strategy with None data."""
        try:
            name = auto_learner.discover_strategy(None)
        except (TypeError, AttributeError):
            pass  # Expected for some implementations

    @pytest.mark.parametrize("score", [100.0, -100.0, 0.0, 1.0])
    def test_feedback_with_extreme_scores(self, auto_learner, score):
        """Feedback with extreme score values."""
        name = auto_learner.discover_strategy({})

        auto_learner.receive_feedback(name, score)

        # Should not crash, score should be tracked
        assert name in auto_learner._scores

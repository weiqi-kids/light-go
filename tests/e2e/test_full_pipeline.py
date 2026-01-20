"""End-to-end tests for the full training pipeline.

Tests the complete workflow from SGF data to trained strategies
using real implementation.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from core.auto_learner import AutoLearner
from core.strategy_manager import StrategyManager


# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------

def _create_sgf_file(path: Path) -> None:
    """Create a simple SGF game file."""
    text = "(;FF[4]SZ[5];B[aa];W[bb])"
    path.write_text(text)


# ---------------------------------------------------------------------------
# Test Classes
# ---------------------------------------------------------------------------

class TestFullPipeline:
    """End-to-end tests for complete training workflow."""

    def test_train_discover_feedback_cycle(self, tmp_path: Path):
        """Test complete cycle: train, discover, feedback, allocation update."""
        # Setup directories
        strategy_dir = tmp_path / "strategies"
        strategy_dir.mkdir()
        data_dir = tmp_path / "data"
        data_dir.mkdir()

        # Create SGF data
        sgf_file = data_dir / "game.sgf"
        _create_sgf_file(sgf_file)

        # Initialize components
        manager = StrategyManager(str(strategy_dir))
        learner = AutoLearner(manager)

        # Train primary strategy
        primary = learner.train_and_save(str(data_dir))
        secondary = learner.discover_strategy({"dummy": 1})

        assert primary in manager.list_strategies()

        # Check initial assignment
        board_features = {"board": [[0] * 5 for _ in range(5)]}
        before = set(learner.assign_training(board_features))
        assert primary in before
        assert secondary in before
        alloc_before = dict(learner._allocation)

        # Provide positive feedback for primary
        learner.receive_feedback(primary, 1.0)

        # Verify allocation changed
        after = set(learner.assign_training(board_features))
        assert primary in after
        assert secondary not in after
        alloc_after = learner._allocation
        assert alloc_after[primary] > alloc_before[primary]
        assert alloc_after[secondary] < alloc_before[secondary]

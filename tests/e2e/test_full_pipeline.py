import tempfile
from pathlib import Path

from core.auto_learner import AutoLearner
from core.strategy_manager import StrategyManager


def create_sgf(path: Path):
    text = "(;FF[4]SZ[5];B[aa];W[bb])"
    path.write_text(text)


def test_full_pipeline():
    with tempfile.TemporaryDirectory() as strat_dir, tempfile.TemporaryDirectory() as data_dir:
        sgf_file = Path(data_dir) / "game.sgf"
        create_sgf(sgf_file)

        manager = StrategyManager(strat_dir)
        learner = AutoLearner(manager)

        primary = learner.train_and_save(data_dir)
        secondary = learner.discover_strategy({"dummy": 1})
        assert primary in manager.list_strategies()

        board_features = {'board': [[0]*5 for _ in range(5)]}
        before = set(learner.assign_training(board_features))
        assert primary in before
        assert secondary in before
        alloc_before = dict(learner._allocation)

        learner.receive_feedback(primary, 1.0)

        after = set(learner.assign_training(board_features))
        assert primary in after
        assert secondary not in after
        alloc_after = learner._allocation
        assert alloc_after[primary] > alloc_before[primary]
        assert alloc_after[secondary] < alloc_before[secondary]

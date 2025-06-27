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

        name = learner.train_and_save(data_dir)
        assert name in manager.list_strategies()

        board_features = {'board': [[0]*5 for _ in range(5)]}
        before = learner.assign_training(board_features)
        assert name in before
        alloc_before = learner._allocation[name]

        learner.receive_feedback(name, 1.0)

        after = learner.assign_training(board_features)
        alloc_after = learner._allocation[name]
        assert after == before
        assert alloc_after > alloc_before

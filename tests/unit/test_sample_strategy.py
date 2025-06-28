import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from core.sample_strategy import SampleGoStrategy


def test_predict_and_persistence(tmp_path):
    board = [
        [1, 0],
        [0, -1],
    ]
    strat = SampleGoStrategy(name="t")
    assert strat.predict(board) == (1, 0)

    path = tmp_path / "s.pkl"
    strat.save(str(path))
    loaded = SampleGoStrategy.load(str(path))

    assert isinstance(loaded, SampleGoStrategy)
    assert loaded.name == "t"
    assert loaded.stats == strat.stats
    assert loaded.predict(board) == (1, 0)

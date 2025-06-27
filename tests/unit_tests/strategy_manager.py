import pathlib
import pickle
import sys
from unittest.mock import Mock

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from core.strategy_manager import StrategyManager


def make_mock_strategy(name, predict_return):
    m = Mock()
    m.predict.return_value = predict_return

    def save(path):
        with open(path, "wb") as f:
            pickle.dump(predict_return, f)
    m.save.side_effect = save
    return m


def test_register_and_persistence(tmp_path):
    mgr = StrategyManager(str(tmp_path))
    s1 = make_mock_strategy("s1", "A")
    mgr.register_strategy("s1", s1)

    mgr.save_strategies()
    assert (tmp_path / "s1.pkl").exists()
    s1.save.assert_called_once()

    mgr2 = StrategyManager(str(tmp_path))
    mgr2.load_strategies()
    assert mgr2._strategies["s1"] == "A"


def test_run_all_calls_predict(tmp_path):
    mgr = StrategyManager(str(tmp_path))
    s1 = make_mock_strategy("s1", "A")
    s2 = make_mock_strategy("s2", "B")
    mgr.register_strategy("s1", s1)
    mgr.register_strategy("s2", s2)

    result = mgr.run_all("board")
    assert result == {"s1": "A", "s2": "B"}
    s1.predict.assert_called_with("board")
    s2.predict.assert_called_with("board")


def test_converge_methods(tmp_path):
    mgr = StrategyManager(str(tmp_path))
    s1 = make_mock_strategy("s1", "X")
    s2 = make_mock_strategy("s2", "X")
    s3 = make_mock_strategy("s3", "Y")
    mgr.register_strategy("s1", s1)
    mgr.register_strategy("s2", s2)
    mgr.register_strategy("s3", s3)

    assert mgr.converge("data") == "X"

    s1.predict.return_value = {"A": 1, "B": 2}
    s2.predict.return_value = {"A": 2, "B": 1}
    mgr.register_strategy("s1", s1)
    mgr.register_strategy("s2", s2)
    mgr.register_strategy("s3", s3)
    weights = {"s1": 1.0, "s2": 2.0}
    assert mgr.converge("data", method="weighted", weights=weights) == "A"

    meta = Mock()
    meta.predict.return_value = "META"
    assert mgr.converge("data", method="meta_model", meta_model=meta) == "META"
    meta.predict.assert_called()


def test_evaluate_all(tmp_path):
    mgr = StrategyManager(str(tmp_path))
    s1 = make_mock_strategy("s1", "A")
    s2 = make_mock_strategy("s2", "A")
    s3 = make_mock_strategy("s3", "B")

    s1.predict.side_effect = ["A", "A", "A"]
    s2.predict.side_effect = ["A", "B", "A"]
    s3.predict.side_effect = ["B", "B", "B"]

    mgr.register_strategy("s1", s1)
    mgr.register_strategy("s2", s2)
    mgr.register_strategy("s3", s3)

    dataset = [(1, "A"), (2, "B"), (3, "A")]
    report = mgr.evaluate_all(dataset)

    assert pytest.approx(report["s1"], rel=1e-6) == 2/3
    assert report["s2"] == 1.0
    assert pytest.approx(report["s3"], rel=1e-6) == 1/3
    assert report["converged"] == 1.0

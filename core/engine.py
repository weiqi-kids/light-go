"""High level pipeline engine for training and inference."""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from input import sgf_to_input
from core.strategy_manager import StrategyManager
from core.auto_learner import AutoLearner

try:  # hf_models may not be present in lightweight environments
    from hf_models.modeling_go_ai import GoAIModel
except Exception:  # pragma: no cover - fallback placeholder
    class GoAIModel:  # type: ignore
        """Fallback placeholder model used when hf_models is unavailable."""

        def train(self, data: List[Any]) -> None:  # noqa: D401 - simple stub
            """Pretend to train."""
            pass

        def predict(self, sample: Any) -> Any:
            """Return a dummy prediction."""
            return None

        def save_pretrained(self, path: str) -> None:
            with open(path, "wb") as f:
                f.write(b"model")

        @classmethod
        def from_pretrained(cls, path: str) -> "GoAIModel":
            return cls()


class Engine:
    """Coordinate the main data pipeline for Light-Go.

    The engine exposes ``train``, ``evaluate`` and ``play`` as the three main
    entry points used by :mod:`main`.  Each step relies on helper modules for
    data conversion, strategy management and learning.  The heavy lifting of the
    actual model is handled by :class:`~hf_models.modeling_go_ai.GoAIModel` but
    the implementation here keeps it lightweight so tests can run without the
    real model present.
    """

    def __init__(self, model_dir: str) -> None:
        self.model_dir = model_dir
        os.makedirs(self.model_dir, exist_ok=True)
        self.strategy_manager = StrategyManager(os.path.join(model_dir, "strategies"))
        self.auto_learner = AutoLearner(self.strategy_manager)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _convert_directory(path: str) -> List[Dict[str, Any]]:
        """Return a list of converted SGF data from ``path``."""
        results: List[Dict[str, Any]] = []
        for fname in os.listdir(path):
            if not fname.endswith(".sgf"):
                continue
            fpath = os.path.join(path, fname)
            results.append(sgf_to_input.convert(fpath))
        return results

    @staticmethod
    def _fuse_predictions(preds: Dict[str, List[Any]]) -> List[Any]:
        """Very simple fusion taking the first strategy's output."""
        if not preds:
            return []
        first = next(iter(preds))
        return preds[first]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def train(self, data_dir: str, output_dir: str, **settings: Any) -> str:
        """Train a new strategy from ``data_dir`` and save to ``output_dir``.

        Parameters
        ----------
        data_dir:
            Directory containing SGF files to use for training.
        output_dir:
            Directory where the resulting model checkpoint will be stored.
        settings:
            Optional keyword arguments for future extensions.

        Returns
        -------
        str
            The name of the newly created strategy.
        """
        converted = self._convert_directory(data_dir)
        strategy_name = self.auto_learner.train_and_save(data_dir)
        model = GoAIModel()
        model.train(converted)
        os.makedirs(output_dir, exist_ok=True)
        model_path = os.path.join(output_dir, f"{strategy_name}.pt")
        model.save_pretrained(model_path)
        return strategy_name

    def evaluate(self, data_dir: str, output_dir: str, **settings: Any) -> Dict[str, Any]:
        """Evaluate existing strategies on ``data_dir``.

        This performs a simple multi-strategy inference followed by a trivial
        fusion step.  The evaluation metrics are placeholder values suitable for
        unit tests.
        """
        converted = self._convert_directory(data_dir)
        strategies = self.strategy_manager.list_strategies()
        predictions: Dict[str, List[Any]] = {}
        for name in strategies:
            model_path = os.path.join(output_dir, f"{name}.pt")
            model = GoAIModel.from_pretrained(model_path)
            predictions[name] = [model.predict(sample) for sample in converted]
        fused = self._fuse_predictions(predictions)
        metrics = {"samples": len(fused), "strategies_tested": len(strategies)}
        # Feedback to the learner (placeholder)
        self.auto_learner.train(data_dir)
        return metrics

    def play(
        self,
        data_dir: str,
        output_dir: str,
        *,
        strategy: Optional[str] = None,
        **settings: Any,
    ) -> List[Any]:
        """Run inference using ``strategy`` or the latest one if ``None``."""
        strategies = self.strategy_manager.list_strategies()
        if not strategies:
            raise RuntimeError("No strategies available. Train one first.")
        strat_name = strategy or strategies[-1]
        model_path = os.path.join(output_dir, f"{strat_name}.pt")
        model = GoAIModel.from_pretrained(model_path)
        converted = self._convert_directory(data_dir)
        return [model.predict(sample) for sample in converted]


__all__ = ["Engine"]

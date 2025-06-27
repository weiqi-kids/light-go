"""Strategy management utilities.

This module provides a lightweight manager capable of handling multiple
strategy models.  Each strategy object is expected to implement a minimal
protocol with ``predict()``, ``save()`` and ``load()`` methods.  The manager
facilitates registration, persistent storage, batch inference and simple
fusion of predictions.
"""

from __future__ import annotations

import os
import pickle
from collections import Counter
from typing import Any, Dict, Iterable, List, Optional, Protocol


class StrategyProtocol(Protocol):
    """Required interface for strategy models."""

    def predict(self, input_data: Any) -> Any:
        """Return a prediction for ``input_data``."""

    def save(self, path: str) -> None:
        """Persist the model to ``path``."""

    @classmethod
    def load(cls, path: str) -> "StrategyProtocol":
        """Load a strategy from ``path``."""
        ...


class StrategyManager:
    """Manage registration and evaluation of strategies."""

    def __init__(self, strategies_path: str) -> None:
        """Create a manager storing strategies under ``strategies_path``."""
        self.strategies_path = strategies_path
        os.makedirs(self.strategies_path, exist_ok=True)
        self._strategies: Dict[str, StrategyProtocol | Any] = {}

    # ------------------------------------------------------------------
    # Utilities compatible with legacy code
    # ------------------------------------------------------------------
    def list_strategies(self) -> List[str]:
        """Return a sorted list of available strategy names."""
        names = set(self._strategies.keys())
        for fname in os.listdir(self.strategies_path):
            if fname.endswith(".pkl"):
                names.add(os.path.splitext(fname)[0])
        return sorted(names)

    def load_strategy(self, name: str) -> Any:
        """Load strategy ``name`` from disk and register it."""
        if name in self._strategies:
            return self._strategies[name]
        path = os.path.join(self.strategies_path, f"{name}.pkl")
        with open(path, "rb") as f:
            strategy = pickle.load(f)
        self._strategies[name] = strategy
        return strategy

    def save_strategy(self, name: str, data: Any) -> None:
        """Save ``data`` under ``name`` using pickle."""
        path = os.path.join(self.strategies_path, f"{name}.pkl")
        with open(path, "wb") as f:
            pickle.dump(data, f)
        self._strategies[name] = data

    # ------------------------------------------------------------------
    # New public API
    # ------------------------------------------------------------------
    def register_strategy(self, name: str, strategy: StrategyProtocol) -> None:
        """Register a strategy instance under ``name``."""
        self._strategies[name] = strategy

    def save_strategies(self) -> None:
        """Persist all registered strategies to ``strategies_path``."""
        for name, strategy in self._strategies.items():
            path = os.path.join(self.strategies_path, f"{name}.pkl")
            if hasattr(strategy, "save"):
                strategy.save(path)  # type: ignore[attr-defined]
            else:  # Fallback for plain objects
                with open(path, "wb") as f:
                    pickle.dump(strategy, f)

    def load_strategies(self) -> None:
        """Load all strategies found in ``strategies_path``."""
        for fname in os.listdir(self.strategies_path):
            if not fname.endswith(".pkl"):
                continue
            name = os.path.splitext(fname)[0]
            if name in self._strategies:
                continue
            path = os.path.join(self.strategies_path, fname)
            with open(path, "rb") as f:
                strategy = pickle.load(f)
            self._strategies[name] = strategy

    def run_all(self, input_data: Any) -> Dict[str, Any]:
        """Run ``predict`` on all registered strategies."""
        results: Dict[str, Any] = {}
        for name, strategy in self._strategies.items():
            if hasattr(strategy, "predict"):
                results[name] = strategy.predict(input_data)
        return results

    # ------------------------------------------------------------------
    # Convergence utilities
    # ------------------------------------------------------------------
    def _converge_from_results(
        self,
        results: Dict[str, Any],
        method: str = "majority_vote",
        weights: Optional[Dict[str, float]] = None,
        meta_model: Optional[StrategyProtocol] = None,
    ) -> Any:
        """Internal helper to merge predictions from ``results``."""
        if not results:
            return None

        if method == "majority_vote":
            counter = Counter(results.values())
            return counter.most_common(1)[0][0]

        if method in {"weighted", "weighted_average"}:
            score_sum: Dict[Any, float] = {}
            weights = weights or {name: 1.0 for name in results}
            for name, pred in results.items():
                weight = weights.get(name, 1.0)
                if isinstance(pred, dict):
                    for move, score in pred.items():
                        score_sum[move] = score_sum.get(move, 0.0) + score * weight
                else:
                    score_sum[pred] = score_sum.get(pred, 0.0) + weight
            return max(score_sum, key=score_sum.get)

        if method in {"meta", "meta_model"}:
            if meta_model is None:
                raise ValueError("meta_model must be provided for meta convergence")
            return meta_model.predict(results)

        raise ValueError(f"Unknown convergence method: {method}")

    def converge(
        self,
        input_data: Any,
        method: str = "majority_vote",
        weights: Optional[Dict[str, float]] = None,
        meta_model: Optional[StrategyProtocol] = None,
    ) -> Any:
        """Run all strategies on ``input_data`` and fuse their results."""
        results = self.run_all(input_data)
        return self._converge_from_results(results, method, weights, meta_model)

    # ------------------------------------------------------------------
    # Evaluation utilities
    # ------------------------------------------------------------------
    def evaluate_all(
        self,
        dataset: Iterable[tuple[Any, Any]],
        method: str = "majority_vote",
        weights: Optional[Dict[str, float]] = None,
        meta_model: Optional[StrategyProtocol] = None,
    ) -> Dict[str, float]:
        """Evaluate strategies on ``dataset``.

        ``dataset`` should yield ``(input_data, expected_move)`` tuples.
        The returned dictionary contains accuracy per strategy and for the
        fused prediction under the ``converged`` key.
        """
        totals: Dict[str, int] = {name: 0 for name in self._strategies}
        correct: Dict[str, int] = {name: 0 for name in self._strategies}
        conv_total = 0
        conv_correct = 0

        for input_data, expected in dataset:
            predictions = self.run_all(input_data)
            for name, pred in predictions.items():
                totals[name] += 1
                if pred == expected:
                    correct[name] += 1
            conv_pred = self._converge_from_results(
                predictions, method=method, weights=weights, meta_model=meta_model
            )
            conv_total += 1
            if conv_pred == expected:
                conv_correct += 1

        report = {
            name: (correct[name] / totals[name]) if totals[name] else 0.0
            for name in totals
        }
        report["converged"] = (conv_correct / conv_total) if conv_total else 0.0
        return report

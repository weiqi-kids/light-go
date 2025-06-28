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
from typing import Any, Dict, Iterable, List, Optional, Protocol, Type
import importlib


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
        # Mapping of strategy name to class information for loading
        self._strategy_classes: Dict[str, tuple[str, str]] = {}

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
        meta_path = os.path.join(self.strategies_path, f"{name}.meta")
        if os.path.exists(meta_path):
            with open(meta_path, "rb") as f:
                mod_name, cls_name = pickle.load(f)
            module = importlib.import_module(mod_name)
            cls: Type[StrategyProtocol] = getattr(module, cls_name)
            strategy = cls.load(path)
            self._strategy_classes[name] = (mod_name, cls_name)
        else:
            with open(path, "rb") as f:
                strategy = pickle.load(f)
        self._strategies[name] = strategy
        return strategy

    def save_strategy(self, name: str, data: Any) -> None:
        """Save ``data`` under ``name`` using pickle or custom ``save``."""
        path = os.path.join(self.strategies_path, f"{name}.pkl")
        meta_path = os.path.join(self.strategies_path, f"{name}.meta")
        if hasattr(data, "save"):
            data.save(path)  # type: ignore[attr-defined]
        else:
            with open(path, "wb") as f:
                pickle.dump(data, f)
        if hasattr(type(data), "load"):
            self._strategy_classes[name] = (
                data.__class__.__module__,
                data.__class__.__name__,
            )
            with open(meta_path, "wb") as f:
                pickle.dump(self._strategy_classes[name], f)
        elif os.path.exists(meta_path):
            os.remove(meta_path)
            self._strategy_classes.pop(name, None)
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
            self.save_strategy(name, strategy)

    def load_strategies(self) -> None:
        """Load all strategies found in ``strategies_path``."""
        for fname in os.listdir(self.strategies_path):
            if not fname.endswith(".pkl"):
                continue
            name = os.path.splitext(fname)[0]
            if name in self._strategies:
                continue
            self.load_strategy(name)

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


# ---------------------------------------------------------------------------
# Module level utility functions
# ---------------------------------------------------------------------------

DEFAULT_STRATEGY_DIR = os.path.join("data", "models", "strategies")


def load_all_strategies(directory: str) -> Dict[str, Dict[str, Any]]:
    """Return all strategy parameter dictionaries found in ``directory``.

    Parameters
    ----------
    directory:
        Path to the strategy directory containing ``*.pkl`` files.

    Returns
    -------
    Dict[str, Dict[str, Any]]
        Mapping of strategy names to the loaded dictionaries.  Each
        dictionary includes a callable under the ``accept`` key which takes
        a game state and returns ``True`` when the strategy should train on
        that state.
    """

    strategies: Dict[str, Dict[str, Any]] = {}
    os.makedirs(directory, exist_ok=True)
    for fname in os.listdir(directory):
        if not fname.endswith(".pkl"):
            continue
        name = os.path.splitext(fname)[0]
        path = os.path.join(directory, fname)
        with open(path, "rb") as f:
            params = pickle.load(f)

        if isinstance(params, dict):
            threshold = params.get("min_games", 0)

            def _accept(game_state: Dict[str, Any], th: int = threshold) -> bool:
                return game_state.get("games", 0) >= th

            params["accept"] = _accept
        strategies[name] = params
    return strategies


def monitor_and_manage_strategies(
    strategies: Dict[str, Dict[str, Any]], threshold: int
) -> Dict[str, Dict[str, Any]]:
    """Ensure at least ``threshold`` strategies can accept new game states.

    Parameters
    ----------
    strategies:
        Already loaded strategy dictionary.
    threshold:
        Minimum number of strategies that must be able to accept a new game
        state via their ``accept`` callable.

    Returns
    -------
    Dict[str, Dict[str, Any]]
        The potentially updated strategy mapping.  New empty strategies are
        created when necessary to satisfy ``threshold``.
    """

    def _is_capable(data: Dict[str, Any]) -> bool:
        accept = data.get("accept")
        return callable(accept)

    capable = [name for name, data in strategies.items() if _is_capable(data)]
    count = len(capable)

    while count < threshold:
        new_name = f"strategy_{len(strategies) + 1}"

        def _accept(_state: Dict[str, Any]) -> bool:
            return True

        strategies[new_name] = {"accept": _accept}
        capable.append(new_name)
        count += 1

    return strategies


def evaluate_strategies(
    strategies: Dict[str, Dict[str, Any]], stability_criteria: float
) -> Dict[str, float]:
    """Return evaluation scores for ``strategies`` based on stability.

    Parameters
    ----------
    strategies:
        Mapping of strategy names to their parameter dictionaries.
    stability_criteria:
        Numeric threshold used to decide whether the parameters of a strategy
        are considered stable.

    Returns
    -------
    Dict[str, float]
        A mapping from strategy name to an evaluation score.  Unstable
        strategies receive a score of ``0.0``.
    """

    scores: Dict[str, float] = {}
    for name, params in strategies.items():
        numeric = [v for v in params.values() if isinstance(v, (int, float))]
        if not numeric:
            scores[name] = 0.0
            continue
        stable = max(numeric) - min(numeric) <= stability_criteria
        scores[name] = sum(numeric) / len(numeric) if stable else 0.0
    return scores


def load_strategy(name: str) -> Dict[str, Any]:
    """Load ``name`` from :data:`DEFAULT_STRATEGY_DIR`.

    Parameters
    ----------
    name:
        Strategy identifier to load from disk.

    Returns
    -------
    Dict[str, Any]
        The loaded strategy object.
    """

    path = os.path.join(DEFAULT_STRATEGY_DIR, f"{name}.pkl")
    with open(path, "rb") as f:
        return pickle.load(f)


def save_strategy(strategy: Dict[str, Any], name: str) -> None:
    """Persist ``strategy`` under ``name`` inside :data:`DEFAULT_STRATEGY_DIR`.

    Parameters
    ----------
    strategy:
        Dictionary of strategy parameters to save.
    name:
        Name of the strategy file (without extension).
    """

    os.makedirs(DEFAULT_STRATEGY_DIR, exist_ok=True)
    path = os.path.join(DEFAULT_STRATEGY_DIR, f"{name}.pkl")
    with open(path, "wb") as f:
        pickle.dump(strategy, f)

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
import logging
from collections import Counter
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Protocol,
    Type,
)
import importlib
from types import SimpleNamespace

# ---------------------------------------------------------------------------
# Module level helpers and utilities
# ---------------------------------------------------------------------------

# Default folder used for persisted strategies.  The auto learner stores
# filter parameters and pickled models here by default.
DEFAULT_STRATEGY_DIR = os.path.join("data", "models", "strategies")

# module level logger
logger = logging.getLogger(__name__)


def create_strategy() -> Any:
    """Return a minimal placeholder strategy instance.

    The returned object exposes ``training_params`` and ``accept_state`` so
    that monitoring utilities can treat it like a fully fledged strategy until
    a real model is trained.
    """

    params: Dict[str, Any] = {"stable": False}

    def accept_state(_: Dict[str, Any]) -> bool:
        return True

    strat = SimpleNamespace(training_params=params, accept_state=accept_state)
    return strat


def load_all_strategies(directory: str) -> Dict[str, Any]:
    """Load every serialized strategy found in ``directory``.

    The function looks for ``.pkl`` files produced by :func:`save_strategy` and
    reconstructs each strategy object.  If a corresponding ``.flt`` file exists
    the filter parameters are loaded and attached to the strategy under the
    ``training_params`` attribute.  The associated ``accept_state`` callable is
    added both to the parameters dictionary and to the strategy instance so that
    callers can query it directly.

    Parameters
    ----------
    directory : str
        Path to the directory containing strategy files.  This typically points
        to ``data/models/strategies``.

    Returns
    -------
    Dict[str, Any]
        Mapping of strategy names to the loaded strategy objects.
    """

    loaded: Dict[str, Any] = {}
    if not os.path.isdir(directory):
        logger.warning("Strategy directory '%s' not found", directory)
        return loaded

    for fname in os.listdir(directory):
        if not fname.endswith(".pkl"):
            continue
        name = os.path.splitext(fname)[0]

        # --- load the core strategy object ---------------------------------
        strategy_path = os.path.join(directory, fname)
        meta_path = os.path.join(directory, f"{name}.meta")
        try:
            if os.path.exists(meta_path):
                with open(meta_path, "rb") as f:
                    mod_name, cls_name = pickle.load(f)
                module = importlib.import_module(mod_name)
                cls: Type[TrainableStrategyProtocol] = getattr(module, cls_name)
                strategy = cls.load(strategy_path)
            else:
                with open(strategy_path, "rb") as f:
                    strategy = pickle.load(f)
        except Exception as exc:
            logger.warning("Could not load strategy '%s': %s", name, exc)
            continue

        # --- load optional filter parameters -------------------------------
        filter_path = os.path.join(directory, f"{name}.flt")
        params: Dict[str, Any] = {}
        if os.path.exists(filter_path):
            with open(filter_path, "rb") as f:
                try:
                    params = pickle.load(f)
                except Exception as exc:
                    logger.warning(
                        "Failed to load filter parameters for '%s': %s",
                        name,
                        exc,
                    )
                    params = {}
        else:
            logger.debug("Filter file for '%s' not found", name)

        def accept_state(state: Dict[str, Any], params=params) -> bool:
            """Return ``True`` if ``state`` should be used for training."""

            min_stones = params.get("min_stones")
            if min_stones is not None:
                stones = state.get("total_black_stones", 0) + state.get(
                    "total_white_stones", 0
                )
                if stones < min_stones:
                    return False
            return True

        params["accept_state"] = accept_state

        # expose params and callable directly on the strategy object
        try:
            setattr(strategy, "training_params", params)
            setattr(strategy, "accept_state", accept_state)
        except Exception:
            pass

        loaded[name] = strategy

    return loaded


def monitor_and_manage_strategies(
    strategies: Dict[str, Any], acceptance_threshold: int
) -> Dict[str, Any]:
    """Ensure enough strategies remain receptive to new training data.

    Parameters
    ----------
    strategies : dict
        Mapping of strategy names to strategy objects.  Each object may expose a
        ``training_params`` attribute containing a ``stable`` flag.
    acceptance_threshold : int
        Minimum number of strategies that should be able to accept additional
        game states for training purposes.

    Returns
    -------
    Dict[str, Any]
        Possibly updated ``strategies`` dictionary.  New placeholder strategies
        created via :func:`create_strategy` are inserted if required so that at
        least ``acceptance_threshold`` entries can receive training data.
    """

    capable = []
    for name, strat in strategies.items():
        params = getattr(strat, "training_params", {})
        if not isinstance(params, dict) or not params.get("stable"):
            capable.append(name)
    
    count = len(capable)
    next_idx = 1
    while count < acceptance_threshold:
        new_name = f"strategy_{len(strategies) + next_idx}"
        strategies[new_name] = create_strategy()
        count += 1
        next_idx += 1

    return strategies


def evaluate_strategies(
    strategies: Dict[str, Any], stability_threshold: float
) -> Dict[str, float]:
    """Score strategies whose parameters have stabilized.

    Parameters
    ----------
    strategies : dict
        Mapping of strategy names to strategy objects.  Each object may expose a
        ``training_params`` dictionary containing ``wins``, ``games`` and a
        ``stability`` value.
    stability_threshold : float
        Minimum stability value required before a strategy is evaluated.

    Returns
    -------
    Dict[str, float]
        Dictionary mapping strategy names to their computed win ratios.
    """

    scores: Dict[str, float] = {}
    for name, strat in strategies.items():
        params = getattr(strat, "training_params", {})
        if not isinstance(params, dict):
            continue
        if params.get("stability", 0.0) < stability_threshold:
            continue
        wins = params.get("wins")
        games = params.get("games")
        if wins is not None and games:
            scores[name] = wins / games
        else:
            scores[name] = 0.0
    return scores


def load_strategy(name: str) -> Any:
    """Load a single strategy previously saved by :func:`save_strategy`.

    Parameters
    ----------
    name : str
        Name of the strategy file (without extension) located inside
        :data:`DEFAULT_STRATEGY_DIR`.

    Returns
    -------
    Any
        The reconstructed strategy object as produced by ``auto_learner``.
    """

    path = os.path.join(DEFAULT_STRATEGY_DIR, f"{name}.pkl")
    meta_path = os.path.join(DEFAULT_STRATEGY_DIR, f"{name}.meta")
    if os.path.exists(meta_path):
        with open(meta_path, "rb") as f:
            mod_name, cls_name = pickle.load(f)
        module = importlib.import_module(mod_name)
        cls: Type[TrainableStrategyProtocol] = getattr(module, cls_name)
        return cls.load(path)

    with open(path, "rb") as f:
        return pickle.load(f)


def save_strategy(strategy: Any, name: str) -> None:
    """Serialize ``strategy`` using the standard format produced by ``auto_learner``.

    Parameters
    ----------
    strategy : Any
        Strategy object to persist.  The object may implement a ``save`` method
        compatible with :class:`TrainableStrategyProtocol`.
    name : str
        Name to use for the generated ``.pkl`` and ``.meta`` files inside
        :data:`DEFAULT_STRATEGY_DIR`.
    """

    path = os.path.join(DEFAULT_STRATEGY_DIR, f"{name}.pkl")
    meta_path = os.path.join(DEFAULT_STRATEGY_DIR, f"{name}.meta")
    if hasattr(strategy, "save"):
        strategy.save(path)  # type: ignore[attr-defined]
    else:
        with open(path, "wb") as f:
            pickle.dump(strategy, f)
    if hasattr(type(strategy), "load"):
        data = (strategy.__class__.__module__, strategy.__class__.__name__)
        with open(meta_path, "wb") as f:
            pickle.dump(data, f)
    elif os.path.exists(meta_path):
        os.remove(meta_path)



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


class TrainableStrategyProtocol(StrategyProtocol, Protocol):
    """Strategy interface that exposes training metadata.

    Attributes
    ----------
    training_params : Dict[str, Any]
        Heuristics and metrics used during training.
    accept_state : Callable[[Dict[str, Any]], bool]
        Function determining whether a game state should be used for training.
    """

    training_params: Dict[str, Any]

    def accept_state(self, state: Dict[str, Any]) -> bool:
        ...


class StrategyManager:
    """Manage registration and evaluation of strategies."""

    def __init__(self, strategies_path: str) -> None:
        """Create a manager storing strategies under ``strategies_path``."""
        self.strategies_path = strategies_path
        os.makedirs(self.strategies_path, exist_ok=True)
        self._strategies: Dict[str, TrainableStrategyProtocol | Any] = {}
        # Mapping of strategy name to class information for loading
        self._strategy_classes: Dict[str, tuple[str, str]] = {}
        # Optional filter parameters controlling training data acceptance
        self._filters: Dict[str, Dict[str, Any]] = {}
        # Load any existing filter files on creation
        self.load_filter_params()

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

    # ------------------------------------------------------------------
    # Loading and saving of filter parameters
    # ------------------------------------------------------------------
    def _filter_path(self, name: str) -> str:
        """Return the file path storing filter parameters for ``name``."""

        return os.path.join(self.strategies_path, f"{name}.flt")

    def load_filter_params(self) -> None:
        """Load filter parameter files for all strategies.

        Each ``.flt`` file contains a dictionary of heuristics generated by the
        learning pipeline.  This method silently skips corrupted files so that a
        partially written filter does not break application startup.
        """

        for fname in os.listdir(self.strategies_path):
            if not fname.endswith(".flt"):
                continue
            name = os.path.splitext(fname)[0]
            with open(os.path.join(self.strategies_path, fname), "rb") as f:
                try:
                    self._filters[name] = pickle.load(f)
                except Exception:
                    self._filters[name] = {}

    def save_filter_params(self, name: str, params: Dict[str, Any]) -> None:
        """Persist ``params`` as filter configuration for ``name``."""

        path = self._filter_path(name)
        with open(path, "wb") as f:
            pickle.dump(params, f)
        self._filters[name] = params

    def strategy_accepts(self, name: str, state: Dict[str, Any]) -> bool:
        """Return ``True`` if strategy ``name`` should train on ``state``.

        The logic first checks for a ``should_use_state`` method on the strategy
        instance.  If absent, basic numeric thresholds stored in the filter
        parameters are applied instead.  Missing parameters default to accepting
        the state.
        """

        strat = self._strategies.get(name)
        if strat is not None and hasattr(strat, "should_use_state"):
            try:
                return bool(strat.should_use_state(state))  # type: ignore[attr-defined]
            except Exception:
                return True
        params = self._filters.get(name, {})
        if not params:
            return True
        min_stones = params.get("min_stones")
        if min_stones is not None:
            stones = state.get("total_black_stones", 0) + state.get("total_white_stones", 0)
            if stones < min_stones:
                return False
        return True

    def strategies_accepting_state(self, state: Dict[str, Any]) -> List[str]:
        """Return strategy names willing to accept ``state`` for training."""

        return [name for name in self.list_strategies() if self.strategy_accepts(name, state)]

    def ensure_capacity(
        self,
        state: Dict[str, Any],
        min_count: int,
        factory: Callable[[], TrainableStrategyProtocol],
    ) -> None:
        """Create new strategies via ``factory`` if too few accept ``state``."""

        available = self.strategies_accepting_state(state)
        while len(available) < min_count:
            new_name = self._generate_name()
            strategy = factory()
            self.register_strategy(new_name, strategy)
            self.save_strategy(new_name, strategy)
            available.append(new_name)

    def _generate_name(self) -> str:
        """Return a new strategy identifier not currently in use.

        Strategy files created by :class:`AutoLearner` use short alphabetical
        names.  When those are exhausted an ``s<index>`` fallback is returned.
        """

        existing = self.list_strategies()
        for idx in range(26):
            candidate = chr(ord("a") + idx)
            if candidate not in existing:
                return candidate
        return f"s{len(existing)}"

    def load_strategy(self, name: str) -> Any:
        """Load strategy ``name`` from disk and register it.

        This method is intentionally free of side effects other than loading
        the serialized object produced by :mod:`auto_learner`.  It simply
        reconstructs the strategy and stores it in the manager.
        """
        if name in self._strategies:
            return self._strategies[name]
        path = os.path.join(self.strategies_path, f"{name}.pkl")
        meta_path = os.path.join(self.strategies_path, f"{name}.meta")
        if os.path.exists(meta_path):
            with open(meta_path, "rb") as f:
                mod_name, cls_name = pickle.load(f)
            module = importlib.import_module(mod_name)
            cls: Type[TrainableStrategyProtocol] = getattr(module, cls_name)
            strategy = cls.load(path)
            self._strategy_classes[name] = (mod_name, cls_name)
        else:
            with open(path, "rb") as f:
                strategy = pickle.load(f)
        self._strategies[name] = strategy
        return strategy

    def save_strategy(self, name: str, data: Any) -> None:
        """Save ``data`` under ``name`` using pickle or custom ``save``.

        Only the contents provided by :mod:`auto_learner` should be persisted
        here.  No additional bookkeeping is performed beyond writing the files
        and updating internal caches.
        """
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
    def register_strategy(self, name: str, strategy: TrainableStrategyProtocol) -> None:
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
        *,
        names: Optional[List[str]] = None,
        method: str = "majority_vote",
        weights: Optional[Dict[str, float]] = None,
        meta_model: Optional[StrategyProtocol] = None,
    ) -> Dict[str, float]:
        """Evaluate strategies on ``dataset``.

        ``dataset`` should yield ``(input_data, expected_move)`` tuples.
        The returned dictionary contains accuracy per strategy and for the
        fused prediction under the ``converged`` key.
        """
        targets = names if names is not None else list(self._strategies.keys())
        totals: Dict[str, int] = {name: 0 for name in targets}
        correct: Dict[str, int] = {name: 0 for name in targets}
        conv_total = 0
        conv_correct = 0

        for input_data, expected in dataset:
            predictions = {
                name: self._strategies[name].predict(input_data)
                for name in targets
                if hasattr(self._strategies[name], "predict")
            }
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

    def score_stable_strategies(
        self,
        dataset: Iterable[tuple[Any, Any]],
    ) -> Dict[str, float]:
        """Evaluate only strategies whose filters are marked as stable."""

        stable = [name for name, params in self._filters.items() if params.get("stable")]
        if not stable:
            return {}
        return self.evaluate_all(dataset, names=stable)

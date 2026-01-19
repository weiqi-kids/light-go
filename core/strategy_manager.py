"""Strategy management utilities.

This module provides a lightweight manager capable of handling multiple
strategy models.  Each strategy object is expected to implement a minimal
protocol with ``predict()``, ``save()`` and ``load()`` methods.  The manager
facilitates registration, persistent storage, batch inference and simple
fusion of predictions.

Enhanced with:
- joblib serialization (faster for large NumPy arrays, backward compatible)
- sklearn VotingClassifier integration for ensemble methods
- Optional MLflow tracking for experiment management
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
    Union,
)
import importlib
from types import SimpleNamespace

# ---------------------------------------------------------------------------
# Optional dependencies - graceful degradation if not installed
# ---------------------------------------------------------------------------
try:
    import joblib
    HAS_JOBLIB = True
except ImportError:
    HAS_JOBLIB = False

try:
    from sklearn.ensemble import VotingClassifier
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

try:
    import mlflow
    import mlflow.sklearn
    HAS_MLFLOW = True
except ImportError:
    HAS_MLFLOW = False


# ---------------------------------------------------------------------------
# Module level helpers and utilities
# ---------------------------------------------------------------------------

# Default folder used for persisted strategies.  The auto learner stores
# filter parameters and pickled models here by default.
DEFAULT_STRATEGY_DIR = os.path.join("data", "models", "strategies")

# module level logger
logger = logging.getLogger(__name__)

# Supported file extensions for strategy files
JOBLIB_EXT = ".joblib"
PICKLE_EXT = ".pkl"


# ---------------------------------------------------------------------------
# Serialization utilities (Layer 1: joblib with pickle fallback)
# ---------------------------------------------------------------------------

def _serialize(obj: Any, path: str, compress: int = 3) -> None:
    """Serialize object using joblib (preferred) or pickle (fallback).

    Parameters
    ----------
    obj : Any
        Object to serialize.
    path : str
        Destination file path. Extension determines format:
        - .joblib: uses joblib with compression
        - .pkl: uses pickle
    compress : int, optional
        Compression level for joblib (0-9). Default is 3.
    """
    if HAS_JOBLIB and path.endswith(JOBLIB_EXT):
        joblib.dump(obj, path, compress=compress)
    else:
        # Fallback to pickle or explicit .pkl extension
        with open(path, "wb") as f:
            pickle.dump(obj, f)


def _deserialize(path: str) -> Any:
    """Deserialize object from joblib or pickle file.

    Parameters
    ----------
    path : str
        Source file path. Supports both .joblib and .pkl files.

    Returns
    -------
    Any
        Deserialized object.
    """
    if HAS_JOBLIB and path.endswith(JOBLIB_EXT):
        return joblib.load(path)
    else:
        with open(path, "rb") as f:
            return pickle.load(f)


def _find_strategy_file(directory: str, name: str) -> Optional[str]:
    """Find strategy file with either .joblib or .pkl extension.

    Parameters
    ----------
    directory : str
        Directory to search in.
    name : str
        Strategy name (without extension).

    Returns
    -------
    Optional[str]
        Full path to strategy file, or None if not found.
        Prefers .joblib over .pkl if both exist.
    """
    joblib_path = os.path.join(directory, f"{name}{JOBLIB_EXT}")
    pkl_path = os.path.join(directory, f"{name}{PICKLE_EXT}")

    if os.path.exists(joblib_path):
        return joblib_path
    if os.path.exists(pkl_path):
        return pkl_path
    return None


def _get_save_path(directory: str, name: str) -> str:
    """Get path for saving a strategy (prefers joblib if available).

    Parameters
    ----------
    directory : str
        Directory to save in.
    name : str
        Strategy name.

    Returns
    -------
    str
        Full path with appropriate extension.
    """
    ext = JOBLIB_EXT if HAS_JOBLIB else PICKLE_EXT
    return os.path.join(directory, f"{name}{ext}")


def create_strategy(params: Optional[Dict[str, Any]] = None) -> "TrainableStrategyProtocol":
    """Return a minimal placeholder strategy instance.

    Parameters
    ----------
    params : Optional[Dict[str, Any]], optional
        Optional dictionary used to pre-populate ``training_params``.  If
        omitted a default ``{"stable": False}`` mapping is created.

    Returns
    -------
    TrainableStrategyProtocol
        Object exposing ``training_params`` and ``accept_state`` so monitoring
        utilities can treat it as a fully fledged strategy until a real model is
        trained.

    Notes
    -----
    ``training_params`` must be a dictionary and ``accept_state`` must be a
    callable taking a ``state`` dictionary describing a board position.  The
    ``state`` is expected to contain ``total_black_stones`` and
    ``total_white_stones`` keys.
    """

    params = dict(params or {})
    params.setdefault("stable", False)

    def accept_state(_: Dict[str, Any]) -> bool:
        return True

    strat: TrainableStrategyProtocol = SimpleNamespace(
        training_params=params,
        accept_state=accept_state,
    )

    if not isinstance(strat.training_params, dict) or not callable(strat.accept_state):
        logger.warning("Generated strategy missing required interface")
    return strat


def load_all_strategies(directory: str) -> Dict[str, "TrainableStrategyProtocol"]:
    """Load every serialized strategy found in ``directory``.

    The function looks for ``.joblib`` and ``.pkl`` files produced by
    :func:`save_strategy` and reconstructs each strategy object.  If a
    corresponding ``.flt`` file exists the filter parameters are loaded and
    attached to the strategy under the ``training_params`` attribute.

    Parameters
    ----------
    directory : str
        Path to the directory containing strategy files.  This typically points
        to ``data/models/strategies``.

    Returns
    -------
    Dict[str, TrainableStrategyProtocol]
        Mapping of strategy names to the loaded strategy objects.  Only
        strategies exposing ``training_params`` (dict) and ``accept_state``
        (callable) are included.
    """

    loaded: Dict[str, TrainableStrategyProtocol] = {}
    if not os.path.isdir(directory):
        logger.warning("Strategy directory '%s' not found", directory)
        return loaded

    # Collect all strategy names (from both .joblib and .pkl files)
    strategy_names = set()
    for fname in os.listdir(directory):
        if fname.endswith(JOBLIB_EXT):
            strategy_names.add(fname[:-len(JOBLIB_EXT)])
        elif fname.endswith(PICKLE_EXT):
            strategy_names.add(fname[:-len(PICKLE_EXT)])

    for name in strategy_names:
        strategy_path = _find_strategy_file(directory, name)
        if strategy_path is None:
            continue

        meta_path = os.path.join(directory, f"{name}.meta")

        # --- load the core strategy object ---------------------------------
        try:
            if os.path.exists(meta_path):
                with open(meta_path, "rb") as f:
                    mod_name, cls_name = pickle.load(f)
                module = importlib.import_module(mod_name)
                cls: Type[TrainableStrategyProtocol] = getattr(module, cls_name)
                strategy = cls.load(strategy_path)
            else:
                strategy = _deserialize(strategy_path)
        except Exception as exc:
            logger.warning("Could not load strategy '%s': %s", name, exc)
            continue

        # --- load optional filter parameters -------------------------------
        filter_path = os.path.join(directory, f"{name}.flt")
        params: Dict[str, Any] = {}
        if os.path.exists(filter_path):
            try:
                params = _deserialize(filter_path)
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
        except Exception as exc:
            logger.warning("Failed to attach metadata to '%s': %s", name, exc)

        # ensure the loaded object conforms to the expected protocol
        if not isinstance(getattr(strategy, "training_params", None), dict):
            logger.warning("Strategy '%s' missing 'training_params'; skipping", name)
            continue
        if not callable(getattr(strategy, "accept_state", None)):
            logger.warning("Strategy '%s' missing 'accept_state'; skipping", name)
            continue

        loaded[name] = strategy

    return loaded


def monitor_and_manage_strategies(
    strategies: Dict[str, "TrainableStrategyProtocol"],
    acceptance_threshold: int,
) -> Dict[str, "TrainableStrategyProtocol"]:
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
    Dict[str, TrainableStrategyProtocol]
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
    strategies: Dict[str, "TrainableStrategyProtocol"],
    stability_threshold: float,
) -> Dict[str, float]:
    """Score strategies whose parameters have stabilized.

    Parameters
    ----------
    strategies : Dict[str, TrainableStrategyProtocol]
        Mapping of strategy names to strategy objects. Each object must expose a
        ``training_params`` dictionary containing ``wins``, ``games`` and a
        ``stability`` value as well as an ``accept_state`` callable.
    stability_threshold : float
        Minimum stability value required before a strategy is evaluated.

    Returns
    -------
    Dict[str, float]
        Dictionary mapping strategy names to their computed win ratios.  This
        function can easily be extended with alternative scoring logic by
        adjusting the computation inside the loop.
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


def load_strategy(name: str) -> "TrainableStrategyProtocol":
    """Load a single strategy previously saved by :func:`save_strategy`.

    Parameters
    ----------
    name : str
        Name of the strategy file (without extension) located inside
        :data:`DEFAULT_STRATEGY_DIR`.

    Returns
    -------
    TrainableStrategyProtocol
        The reconstructed strategy object as produced by ``auto_learner``.
    """

    strategy_path = _find_strategy_file(DEFAULT_STRATEGY_DIR, name)
    if strategy_path is None:
        raise FileNotFoundError(f"Strategy '{name}' not found in {DEFAULT_STRATEGY_DIR}")

    meta_path = os.path.join(DEFAULT_STRATEGY_DIR, f"{name}.meta")
    if os.path.exists(meta_path):
        with open(meta_path, "rb") as f:
            mod_name, cls_name = pickle.load(f)
        module = importlib.import_module(mod_name)
        cls: Type[TrainableStrategyProtocol] = getattr(module, cls_name)
        strategy = cls.load(strategy_path)
    else:
        strategy = _deserialize(strategy_path)

    if not isinstance(getattr(strategy, "training_params", None), dict) or not callable(
        getattr(strategy, "accept_state", None)
    ):
        logger.warning("Strategy '%s' missing required interface", name)
    return strategy


def save_strategy(strategy: Any, name: str) -> None:
    """Serialize ``strategy`` using the standard format produced by ``auto_learner``.

    Parameters
    ----------
    strategy : Any
        Strategy object to persist. The object should expose ``training_params``
        and ``accept_state`` so that it conforms to
        :class:`TrainableStrategyProtocol`. If ``save`` is implemented it will
        be used, otherwise joblib/pickle is used as a fallback.
    name : str
        Name to use for the generated strategy file and ``.meta`` files inside
        :data:`DEFAULT_STRATEGY_DIR`.
    """
    os.makedirs(DEFAULT_STRATEGY_DIR, exist_ok=True)

    path = _get_save_path(DEFAULT_STRATEGY_DIR, name)
    meta_path = os.path.join(DEFAULT_STRATEGY_DIR, f"{name}.meta")

    if hasattr(strategy, "save"):
        strategy.save(path)  # type: ignore[attr-defined]
    else:
        _serialize(strategy, path)

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
    """Manage registration and evaluation of strategies.

    Enhanced with:
    - joblib serialization for faster save/load of large models
    - sklearn VotingClassifier support for ensemble methods
    - Optional MLflow tracking for experiment management
    """

    def __init__(self, strategies_path: str) -> None:
        """Create a manager storing strategies under ``strategies_path``."""
        self.strategies_path = strategies_path
        os.makedirs(self.strategies_path, exist_ok=True)
        self._strategies: Dict[str, TrainableStrategyProtocol | Any] = {}
        # Mapping of strategy name to class information for loading
        self._strategy_classes: Dict[str, tuple[str, str]] = {}
        # Optional filter parameters controlling training data acceptance
        self._filters: Dict[str, Dict[str, Any]] = {}
        # sklearn VotingClassifier instance (Layer 2)
        self._ensemble: Optional[Any] = None
        # MLflow experiment tracking state (Layer 3)
        self._mlflow_experiment: Optional[str] = None
        # Load any existing filter files on creation
        self.load_filter_params()

    # ------------------------------------------------------------------
    # Utilities compatible with legacy code
    # ------------------------------------------------------------------
    def list_strategies(self) -> List[str]:
        """Return a sorted list of available strategy names."""
        names = set(self._strategies.keys())
        for fname in os.listdir(self.strategies_path):
            if fname.endswith(JOBLIB_EXT):
                names.add(fname[:-len(JOBLIB_EXT)])
            elif fname.endswith(PICKLE_EXT):
                names.add(fname[:-len(PICKLE_EXT)])
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
            flt_path = os.path.join(self.strategies_path, fname)
            try:
                self._filters[name] = _deserialize(flt_path)
            except Exception:
                self._filters[name] = {}

    def save_filter_params(self, name: str, params: Dict[str, Any]) -> None:
        """Persist ``params`` as filter configuration for ``name``."""

        path = self._filter_path(name)
        _serialize(params, path)
        self._filters[name] = params

    def strategy_accepts(self, name: str, state: Dict[str, Any]) -> bool:
        """Return ``True`` if strategy ``name`` should train on ``state``.

        ``state`` should contain at least ``total_black_stones`` and
        ``total_white_stones`` keys.  The method first tries the strategy's
        ``accept_state`` callable, falling back to ``should_use_state`` or simple
        heuristics stored in filter parameters.  All callbacks are wrapped in
        ``try``/``except`` blocks so that a misbehaving strategy cannot crash the
        manager.
        """

        strat = self._strategies.get(name)
        if strat is not None:
            callback = getattr(strat, "accept_state", None)
            if callable(callback):
                try:
                    return bool(callback(state))
                except Exception as exc:
                    logger.error("accept_state for '%s' failed: %s", name, exc)
            if hasattr(strat, "should_use_state"):
                try:
                    return bool(strat.should_use_state(state))  # type: ignore[attr-defined]
                except Exception as exc:
                    logger.error("should_use_state for '%s' failed: %s", name, exc)
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

        Supports both .joblib and .pkl files (prefers .joblib).
        """
        if name in self._strategies:
            return self._strategies[name]

        strategy_path = _find_strategy_file(self.strategies_path, name)
        if strategy_path is None:
            raise FileNotFoundError(
                f"Strategy '{name}' not found in {self.strategies_path}"
            )

        meta_path = os.path.join(self.strategies_path, f"{name}.meta")
        if os.path.exists(meta_path):
            with open(meta_path, "rb") as f:
                mod_name, cls_name = pickle.load(f)
            module = importlib.import_module(mod_name)
            cls: Type[TrainableStrategyProtocol] = getattr(module, cls_name)
            strategy = cls.load(strategy_path)
            self._strategy_classes[name] = (mod_name, cls_name)
        else:
            strategy = _deserialize(strategy_path)
        self._strategies[name] = strategy
        return strategy

    def save_strategy(self, name: str, data: Any) -> None:
        """Save ``data`` under ``name`` using joblib (preferred) or pickle.

        Only the contents provided by :mod:`auto_learner` should be persisted
        here.  No additional bookkeeping is performed beyond writing the files
        and updating internal caches.

        New saves use .joblib format if joblib is available.
        """
        path = _get_save_path(self.strategies_path, name)
        meta_path = os.path.join(self.strategies_path, f"{name}.meta")

        if hasattr(data, "save"):
            data.save(path)  # type: ignore[attr-defined]
        else:
            _serialize(data, path)

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
            if fname.endswith(JOBLIB_EXT):
                name = fname[:-len(JOBLIB_EXT)]
            elif fname.endswith(PICKLE_EXT):
                name = fname[:-len(PICKLE_EXT)]
            else:
                continue
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
    # Layer 2: sklearn VotingClassifier integration
    # ------------------------------------------------------------------
    def build_ensemble(
        self,
        names: Optional[List[str]] = None,
        voting: str = "soft",
        weights: Optional[List[float]] = None,
    ) -> Optional[Any]:
        """Build a sklearn VotingClassifier from registered strategies.

        Parameters
        ----------
        names : Optional[List[str]]
            Strategy names to include. If None, uses all strategies with predict.
        voting : str
            Voting method: 'hard' for majority vote, 'soft' for probability average.
        weights : Optional[List[float]]
            Weights for each strategy. If None, equal weights are used.

        Returns
        -------
        Optional[VotingClassifier]
            Configured VotingClassifier, or None if sklearn is not available.

        Raises
        ------
        ImportError
            If sklearn is not installed.

        Example
        -------
        >>> manager = StrategyManager("/path/to/strategies")
        >>> ensemble = manager.build_ensemble(voting="soft")
        >>> ensemble.fit(X_train, y_train)
        >>> predictions = ensemble.predict(X_test)
        """
        if not HAS_SKLEARN:
            raise ImportError(
                "sklearn is required for ensemble methods. "
                "Install with: pip install scikit-learn"
            )

        names = names or list(self._strategies.keys())
        estimators = [
            (name, self._strategies[name])
            for name in names
            if name in self._strategies and hasattr(self._strategies[name], "predict")
        ]

        if not estimators:
            logger.warning("No valid estimators found for ensemble")
            return None

        self._ensemble = VotingClassifier(
            estimators=estimators,
            voting=voting,
            weights=weights,
        )
        return self._ensemble

    def converge_sklearn(
        self,
        input_data: Any,
        voting: str = "soft",
        weights: Optional[List[float]] = None,
    ) -> Any:
        """Use sklearn VotingClassifier for prediction fusion.

        Parameters
        ----------
        input_data : Any
            Input data for prediction.
        voting : str
            Voting method: 'hard' or 'soft'.
        weights : Optional[List[float]]
            Strategy weights.

        Returns
        -------
        Any
            Fused prediction from the ensemble.

        Note
        ----
        This method requires strategies to be sklearn-compatible estimators.
        For non-sklearn models, use the regular ``converge()`` method.
        """
        if self._ensemble is None:
            self.build_ensemble(voting=voting, weights=weights)

        if self._ensemble is None:
            return self.converge(input_data)

        # VotingClassifier.predict expects 2D array
        import numpy as np
        if not isinstance(input_data, np.ndarray):
            input_data = np.array(input_data)
        if input_data.ndim == 1:
            input_data = input_data.reshape(1, -1)

        return self._ensemble.predict(input_data)[0]

    # ------------------------------------------------------------------
    # Convergence utilities (original implementation)
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
    # Layer 3: MLflow tracking integration
    # ------------------------------------------------------------------
    def setup_mlflow(
        self,
        experiment_name: str = "go_strategies",
        tracking_uri: Optional[str] = None,
    ) -> bool:
        """Initialize MLflow tracking for this manager.

        Parameters
        ----------
        experiment_name : str
            Name of the MLflow experiment.
        tracking_uri : Optional[str]
            MLflow tracking server URI. If None, uses local ./mlruns directory.

        Returns
        -------
        bool
            True if MLflow was successfully configured, False otherwise.

        Example
        -------
        >>> manager = StrategyManager("/path/to/strategies")
        >>> if manager.setup_mlflow("my_experiment"):
        ...     manager.log_strategy("strategy_a", metrics={"accuracy": 0.95})
        """
        if not HAS_MLFLOW:
            logger.warning(
                "MLflow is not installed. Install with: pip install mlflow"
            )
            return False

        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)

        mlflow.set_experiment(experiment_name)
        self._mlflow_experiment = experiment_name
        logger.info("MLflow tracking enabled for experiment: %s", experiment_name)
        return True

    def log_strategy(
        self,
        name: str,
        metrics: Optional[Dict[str, float]] = None,
        params: Optional[Dict[str, Any]] = None,
        tags: Optional[Dict[str, str]] = None,
        log_model: bool = True,
    ) -> Optional[str]:
        """Log a strategy to MLflow with metrics and parameters.

        Parameters
        ----------
        name : str
            Strategy name to log.
        metrics : Optional[Dict[str, float]]
            Metrics to record (e.g., {"accuracy": 0.95, "win_rate": 0.6}).
        params : Optional[Dict[str, Any]]
            Parameters to record.
        tags : Optional[Dict[str, str]]
            Tags for the run.
        log_model : bool
            Whether to log the model artifact.

        Returns
        -------
        Optional[str]
            MLflow run ID, or None if logging failed.

        Example
        -------
        >>> manager.log_strategy(
        ...     "strategy_a",
        ...     metrics={"accuracy": 0.95, "games_played": 1000},
        ...     params={"learning_rate": 0.01},
        ...     tags={"version": "v2"}
        ... )
        """
        if not HAS_MLFLOW:
            logger.warning("MLflow is not available")
            return None

        if name not in self._strategies:
            logger.warning("Strategy '%s' not found", name)
            return None

        strategy = self._strategies[name]

        try:
            with mlflow.start_run(run_name=f"strategy_{name}") as run:
                # Log parameters
                mlflow.log_param("strategy_name", name)
                if params:
                    mlflow.log_params(params)

                # Log training_params if available
                training_params = getattr(strategy, "training_params", {})
                if isinstance(training_params, dict):
                    # Filter out non-serializable values
                    safe_params = {
                        k: v for k, v in training_params.items()
                        if isinstance(v, (int, float, str, bool))
                    }
                    if safe_params:
                        mlflow.log_params({f"tp_{k}": v for k, v in safe_params.items()})

                # Log metrics
                if metrics:
                    mlflow.log_metrics(metrics)

                # Log tags
                if tags:
                    mlflow.set_tags(tags)

                # Log model artifact
                if log_model and hasattr(strategy, "predict"):
                    try:
                        mlflow.sklearn.log_model(
                            strategy,
                            artifact_path=f"model_{name}",
                            registered_model_name=f"go_strategy_{name}",
                        )
                    except Exception as model_exc:
                        # Fallback: save as generic artifact
                        logger.debug(
                            "Could not log as sklearn model: %s. Saving as artifact.",
                            model_exc
                        )
                        artifact_path = _get_save_path(self.strategies_path, f"_mlflow_{name}")
                        _serialize(strategy, artifact_path)
                        mlflow.log_artifact(artifact_path)

                logger.info("Logged strategy '%s' to MLflow (run_id=%s)", name, run.info.run_id)
                return run.info.run_id

        except Exception as exc:
            logger.error("Failed to log strategy '%s' to MLflow: %s", name, exc)
            return None

    def log_ensemble_evaluation(
        self,
        dataset: Iterable[tuple[Any, Any]],
        run_name: str = "ensemble_evaluation",
    ) -> Optional[str]:
        """Evaluate all strategies and log results to MLflow.

        Parameters
        ----------
        dataset : Iterable[tuple[Any, Any]]
            Evaluation dataset yielding (input, expected) tuples.
        run_name : str
            Name for the MLflow run.

        Returns
        -------
        Optional[str]
            MLflow run ID, or None if logging failed.
        """
        if not HAS_MLFLOW:
            logger.warning("MLflow is not available")
            return None

        try:
            results = self.evaluate_all(dataset)

            with mlflow.start_run(run_name=run_name) as run:
                mlflow.log_param("num_strategies", len(self._strategies))
                mlflow.log_param("strategy_names", ",".join(self._strategies.keys()))

                for name, accuracy in results.items():
                    mlflow.log_metric(f"accuracy_{name}", accuracy)

                logger.info(
                    "Logged ensemble evaluation to MLflow (run_id=%s)",
                    run.info.run_id
                )
                return run.info.run_id

        except Exception as exc:
            logger.error("Failed to log ensemble evaluation: %s", exc)
            return None

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

    # ------------------------------------------------------------------
    # Utility: Check available features
    # ------------------------------------------------------------------
    @staticmethod
    def available_features() -> Dict[str, bool]:
        """Return a dictionary of available optional features.

        Returns
        -------
        Dict[str, bool]
            Feature availability: joblib, sklearn, mlflow.
        """
        return {
            "joblib": HAS_JOBLIB,
            "sklearn": HAS_SKLEARN,
            "mlflow": HAS_MLFLOW,
        }

"""Entry point for Light-Go command line interface.

This file exposes a small CLI utility used throughout the simplified tests.  It
supports three modes:

``train``     - learn a new strategy from SGF files.
``evaluate``  - compute simple statistics for a dataset and optionally evaluate
                an existing strategy.
``play``      - decide a single move from an SGF board state.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from typing import Any, Dict

from core.engine import Engine

try:  # YAML is optional.  The small tests do not require it.
    import yaml
except Exception:  # pragma: no cover - YAML is not mandatory
    yaml = None


def _load_config(path: str | None) -> Dict[str, Any]:
    """Load optional YAML/JSON configuration file."""

    if not path:
        return {}
    try:
        with open(path, "r", encoding="utf-8") as fh:
            if path.endswith(".json"):
                return json.load(fh)
            if yaml is not None:
                return yaml.safe_load(fh)
    except FileNotFoundError:
        logging.warning("Config file %s not found", path)
    except Exception as exc:  # pragma: no cover - generic safety
        logging.warning("Failed to load config %s: %s", path, exc)
    return {}


def _run_train(engine: Engine, data: str, output_dir: str) -> str:
    """Run a training task and return the saved strategy name.

    Parameters
    ----------
    engine:
        Engine instance used for training.
    data:
        Directory containing SGF files for training.
    output_dir:
        Directory where the trained model checkpoint will be saved.
    """

    strategy = engine.train(data, output_dir)
    logging.info("Saved strategy %s to %s", strategy, engine.strategy_manager.strategies_path)
    return strategy


def _run_train_npz(engine: Engine, data: str, output_dir: str) -> str:
    """Run training from NPZ dataset."""
    strategy = engine.train_npz_directory(data, output_dir)
    logging.info("Saved strategy %s to %s", strategy, engine.strategy_manager.strategies_path)
    return strategy


def _run_evaluate(engine: Engine, data: str) -> Dict[str, Any]:
    """Run evaluation and return aggregated statistics."""

    stats = engine.auto_learner.train(data)
    logging.info("Evaluation finished on %s", data)
    return stats


def _run_play(engine: Engine, sgf_path: str) -> tuple[int, int] | None:
    """Decide a move from an SGF file."""

    from input.sgf_to_input import parse_sgf

    matrix, metadata, _ = parse_sgf(sgf_path)
    move = engine.decide_move(matrix, metadata["next_move"])
    return move


def detect_data_type(path: str) -> str:
    """Return the data type contained in ``path`` ('sgf' or 'npz')."""
    if not os.path.isdir(path):
        raise ValueError(f"{path} is not a directory")
    files = os.listdir(path)
    has_sgf = any(f.endswith('.sgf') for f in files)
    has_npz = any(f.endswith('.npz') for f in files)
    if has_sgf and has_npz:
        raise ValueError('Mixed SGF and NPZ files found')
    if has_sgf:
        return 'sgf'
    if has_npz:
        return 'npz'
    raise ValueError('No supported data files found')


def main() -> None:
    """Entry point for the ``light-go`` command line tool."""
    parser = argparse.ArgumentParser(description="Light-Go")
    parser.add_argument("--mode", choices=["train", "evaluate", "play"], required=True)
    parser.add_argument("--data", help="Input data directory or SGF file")
    parser.add_argument("--output", required=True, help="Output/working directory")
    parser.add_argument("--config", help="Optional configuration YAML/JSON")
    parser.add_argument("--strategy", help="Strategy name or fusion method")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    config = _load_config(args.config)
    logging.debug("Loaded config: %s", config)

    try:
        engine = Engine(args.output)
        if args.strategy:
            engine.load_strategy(args.strategy)

        if args.mode == "train":
            if not args.data:
                parser.error("--data is required for train mode")
            try:
                dtype = detect_data_type(args.data)
            except ValueError as exc:
                parser.error(str(exc))
            if dtype == 'sgf':
                name = _run_train(engine, args.data, args.output)
            else:
                name = _run_train_npz(engine, args.data, args.output)
            result_path = os.path.join(args.output, f"{name}.pkl")
            print(result_path)
        elif args.mode == "evaluate":
            if not args.data:
                parser.error("--data is required for evaluate mode")
            stats = _run_evaluate(engine, args.data)
            print(json.dumps(stats, indent=2))
        elif args.mode == "play":
            if not args.data:
                parser.error("--data is required for play mode")
            move = _run_play(engine, args.data)
            print(move)
    except Exception as exc:  # pragma: no cover - defensive
        logging.exception("Unhandled error: %s", exc)


if __name__ == "__main__":
    main()

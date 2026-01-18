#!/usr/bin/env python3
"""Component test runner for Light-Go.

This script runs the component tests for each of the 8 core components
either individually or all together.

Usage:
    python run_component_tests.py [component_name] [pytest_args...]

Components:
    1. liberty          - Liberty encoder (core/liberty.py)
    2. rules            - Go rules engine (input/sgf_to_input.py)
    3. neural_network   - Neural network model (core/engine.py:GoAIModel)
    4. strategy_manager - Strategy management (core/strategy_manager.py)
    5. auto_learner     - Auto learning system (core/auto_learner.py)
    6. engine           - Training loop and inference (core/engine.py)
    7. mcts             - MCTS search (core/mcts.py)
    8. self_play        - Self-play / GTP interface (api/gtp_interface.py)
    all                 - Run all component tests

Examples:
    python run_component_tests.py                    # Run all tests
    python run_component_tests.py liberty            # Run liberty tests only
    python run_component_tests.py mcts -v            # Run mcts tests with verbose
    python run_component_tests.py all --tb=short    # All tests, short traceback
"""
from __future__ import annotations

import subprocess
import sys
import os
from typing import List, Optional

# Component mapping to test files
COMPONENTS = {
    "1": "liberty",
    "2": "rules",
    "3": "neural_network",
    "4": "strategy_manager",
    "5": "auto_learner",
    "6": "engine",
    "7": "mcts",
    "8": "self_play",
    "liberty": "liberty",
    "rules": "rules",
    "neural_network": "neural_network",
    "strategy_manager": "strategy_manager",
    "auto_learner": "auto_learner",
    "engine": "engine",
    "mcts": "mcts",
    "self_play": "self_play",
    "all": "all",
}

# Test file paths
TEST_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tests", "components")

TEST_FILES = {
    "liberty": "test_liberty.py",
    "rules": "test_rules.py",
    "neural_network": "test_neural_network.py",
    "strategy_manager": "test_strategy_manager.py",
    "auto_learner": "test_auto_learner.py",
    "engine": "test_engine.py",
    "mcts": "test_mcts.py",
    "self_play": "test_self_play.py",
}


def print_header() -> None:
    """Print script header."""
    print("=" * 60)
    print(" Light-Go Component Tests")
    print("=" * 60)


def print_usage() -> None:
    """Print usage information."""
    print(__doc__)


def run_pytest(test_paths: List[str], extra_args: List[str]) -> int:
    """Run pytest with the given test paths and extra arguments.

    Parameters
    ----------
    test_paths : list[str]
        List of test file paths to run.
    extra_args : list[str]
        Additional pytest arguments.

    Returns
    -------
    int
        Pytest exit code.
    """
    cmd = [sys.executable, "-m", "pytest"] + test_paths + extra_args

    print(f"\nRunning: {' '.join(cmd)}\n")

    result = subprocess.run(cmd)
    return result.returncode


def run_component(component: str, extra_args: List[str]) -> int:
    """Run tests for a specific component.

    Parameters
    ----------
    component : str
        Component name or 'all'.
    extra_args : list[str]
        Additional pytest arguments.

    Returns
    -------
    int
        Exit code (0 = success).
    """
    if component == "all":
        # Run all component tests
        test_paths = [os.path.join(TEST_DIR, f) for f in TEST_FILES.values()]
        print(f"\nRunning all {len(test_paths)} component test files...")
    else:
        # Run specific component
        test_file = TEST_FILES.get(component)
        if not test_file:
            print(f"Unknown component: {component}")
            print_usage()
            return 1

        test_path = os.path.join(TEST_DIR, test_file)
        if not os.path.exists(test_path):
            print(f"Test file not found: {test_path}")
            return 1

        test_paths = [test_path]
        print(f"\nRunning tests for component: {component}")

    return run_pytest(test_paths, extra_args)


def main() -> int:
    """Main entry point.

    Returns
    -------
    int
        Exit code.
    """
    print_header()

    # Parse arguments
    args = sys.argv[1:]

    # Default to 'all' if no component specified
    if not args or args[0].startswith("-"):
        component = "all"
        extra_args = args
    else:
        component_arg = args[0].lower()
        component = COMPONENTS.get(component_arg)

        if component is None:
            print(f"Unknown component: {component_arg}")
            print_usage()
            return 1

        extra_args = args[1:]

    # Run the tests
    return run_component(component, extra_args)


if __name__ == "__main__":
    sys.exit(main())

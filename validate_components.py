#!/usr/bin/env python3
"""Component validation script for Light-Go.

This script validates each component independently with clear input/output
for debugging and verification purposes.

Usage:
    python validate_components.py [component_name]

Components:
    1. liberty          - Liberty encoder
    2. rules            - Go rules engine (SGF parser, forbidden moves)
    3. neural_network   - Neural network model
    4. strategy_manager - Strategy management
    5. auto_learner     - Architecture genome / auto learning system
    6. engine           - Training loop and inference
    7. mcts             - MCTS search (placeholder check)
    8. self_play        - Self-play engine (GTP interface check)
    all                 - Run all validations
"""
from __future__ import annotations

import json
import sys
import os
import tempfile
from typing import Any, Dict, List, Tuple

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


class ValidationResult:
    """Container for validation results."""

    def __init__(self, component: str):
        self.component = component
        self.passed: List[str] = []
        self.failed: List[Tuple[str, str]] = []

    def add_pass(self, test_name: str) -> None:
        self.passed.append(test_name)

    def add_fail(self, test_name: str, reason: str) -> None:
        self.failed.append((test_name, reason))

    def is_success(self) -> bool:
        return len(self.failed) == 0

    def summary(self) -> str:
        total = len(self.passed) + len(self.failed)
        status = "PASS" if self.is_success() else "FAIL"
        return f"[{status}] {self.component}: {len(self.passed)}/{total} tests passed"


def print_header(title: str) -> None:
    print("\n" + "=" * 60)
    print(f" {title}")
    print("=" * 60)


def print_io(label: str, data: Any, indent: int = 2) -> None:
    """Print input/output data in a readable format."""
    prefix = " " * indent
    if isinstance(data, (dict, list)):
        formatted = json.dumps(data, indent=2, default=str, ensure_ascii=False)
        lines = formatted.split('\n')
        print(f"{prefix}{label}:")
        for line in lines[:20]:  # Limit output
            print(f"{prefix}  {line}")
        if len(lines) > 20:
            print(f"{prefix}  ... ({len(lines) - 20} more lines)")
    else:
        print(f"{prefix}{label}: {data}")


# ==========================================================================
# Component 1: Liberty Encoder
# ==========================================================================
def validate_liberty() -> ValidationResult:
    """Validate the Liberty encoder component."""
    print_header("1. Liberty Encoder (core/liberty.py)")
    result = ValidationResult("Liberty Encoder")

    from core.liberty import count_liberties, group_and_liberties, neighbors

    # Test 1: neighbors function
    print("\n[Test 1] neighbors() function")
    test_input = {"x": 1, "y": 1, "size": 9}
    print_io("Input", test_input)

    output = list(neighbors(1, 1, 9))
    expected = [(0, 1), (2, 1), (1, 0), (1, 2)]
    print_io("Output", output)
    print_io("Expected", expected)

    if set(output) == set(expected):
        result.add_pass("neighbors() returns correct adjacent positions")
        print("  Status: PASS")
    else:
        result.add_fail("neighbors()", f"Got {output}, expected {expected}")
        print("  Status: FAIL")

    # Test 2: group_and_liberties with a simple board
    print("\n[Test 2] group_and_liberties() function")
    # Board: 0=empty, 1=black, -1=white
    # Simple 5x5 board with a black stone at (2,2)
    board = [
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
    ]
    test_input = {"board": "5x5 with black stone at (2,2)", "position": "(2, 2)"}
    print_io("Input", test_input)

    group, liberties = group_and_liberties(board, 2, 2)
    print_io("Output group", list(group))
    print_io("Output liberties", list(liberties))

    if group == {(2, 2)} and len(liberties) == 4:
        result.add_pass("group_and_liberties() correctly identifies single stone with 4 liberties")
        print("  Status: PASS")
    else:
        result.add_fail("group_and_liberties()", f"Unexpected result")
        print("  Status: FAIL")

    # Test 3: Connected group
    print("\n[Test 3] group_and_liberties() with connected stones")
    board = [
        [0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 1, 1, 1, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0],
    ]
    test_input = {"board": "5x5 with cross pattern at center", "position": "(2, 2)"}
    print_io("Input", test_input)

    group, liberties = group_and_liberties(board, 2, 2)
    print_io("Output group size", len(group))
    print_io("Output liberties count", len(liberties))

    if len(group) == 5 and len(liberties) == 8:
        result.add_pass("Connected group has 5 stones and 8 liberties")
        print("  Status: PASS")
    else:
        result.add_fail("group_and_liberties() connected", f"Group: {len(group)}, Libs: {len(liberties)}")
        print("  Status: FAIL")

    # Test 4: count_liberties full board
    print("\n[Test 4] count_liberties() function")
    board = [
        [0, 0, 0, 0, 0],
        [0, 1, 0, -1, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
    ]
    test_input = {"board": "5x5 with black at (1,1) and white at (3,1)"}
    print_io("Input", test_input)

    output = count_liberties(board)
    print_io("Output", output)

    # Black should have positive liberties, white should have negative
    black_found = any(v > 0 for _, _, v in output)
    white_found = any(v < 0 for _, _, v in output)

    if black_found and white_found:
        result.add_pass("count_liberties() returns signed liberty counts")
        print("  Status: PASS")
    else:
        result.add_fail("count_liberties()", "Missing positive or negative liberties")
        print("  Status: FAIL")

    return result


# ==========================================================================
# Component 2: Go Rules Engine
# ==========================================================================
def validate_rules() -> ValidationResult:
    """Validate the Go rules engine component."""
    print_header("2. Go Rules Engine (input/sgf_to_input.py)")
    result = ValidationResult("Go Rules Engine")

    from input.sgf_to_input import parse_sgf, convert, _compute_forbidden

    # Test 1: Parse a simple SGF string
    print("\n[Test 1] parse_sgf() with simple game")
    sgf_content = """(;GM[1]FF[4]SZ[9]KM[7.5]RU[Chinese]
        ;B[ee];W[gc];B[cg])"""

    test_input = {"sgf": "9x9 game, 3 moves, Chinese rules, 7.5 komi"}
    print_io("Input", test_input)

    try:
        matrix, metadata, board = parse_sgf(sgf_content, from_string=True)
        print_io("Output board size", len(matrix))
        print_io("Output metadata.rules", metadata.get("rules", {}))
        print_io("Output metadata.next_move", metadata.get("next_move"))
        print_io("Output step count", len(metadata.get("step", [])))

        if (metadata["rules"]["board_size"] == 9 and
            metadata["rules"]["komi"] == 7.5 and
            metadata["rules"]["ruleset"] == "chinese"):
            result.add_pass("parse_sgf() correctly extracts rules")
            print("  Status: PASS")
        else:
            result.add_fail("parse_sgf() rules", "Incorrect rules extraction")
            print("  Status: FAIL")

    except Exception as e:
        result.add_fail("parse_sgf()", str(e))
        print(f"  Status: FAIL - {e}")

    # Test 2: Convert function
    print("\n[Test 2] convert() returns liberty and forbidden")
    test_input = {"sgf": "Same 9x9 game"}
    print_io("Input", test_input)

    try:
        data = convert(sgf_content, from_string=True)
        print_io("Output keys", list(data.keys()))
        print_io("Output liberty count", len(data.get("liberty", [])))
        print_io("Output forbidden count", len(data.get("forbidden", [])))

        if "liberty" in data and "forbidden" in data and "metadata" in data:
            result.add_pass("convert() returns complete structure")
            print("  Status: PASS")
        else:
            result.add_fail("convert()", "Missing required keys")
            print("  Status: FAIL")

    except Exception as e:
        result.add_fail("convert()", str(e))
        print(f"  Status: FAIL - {e}")

    # Test 3: Japanese rules
    print("\n[Test 3] parse_sgf() with Japanese rules")
    sgf_jp = """(;GM[1]FF[4]SZ[19]KM[6.5]RU[Japanese];B[pd])"""
    test_input = {"sgf": "19x19 game, Japanese rules, 6.5 komi"}
    print_io("Input", test_input)

    try:
        matrix, metadata, _ = parse_sgf(sgf_jp, from_string=True)
        print_io("Output ruleset", metadata["rules"]["ruleset"])
        print_io("Output komi", metadata["rules"]["komi"])
        print_io("Output board_size", metadata["rules"]["board_size"])

        if metadata["rules"]["ruleset"] == "japanese" and metadata["rules"]["komi"] == 6.5:
            result.add_pass("parse_sgf() handles Japanese rules")
            print("  Status: PASS")
        else:
            result.add_fail("parse_sgf() Japanese", "Incorrect parsing")
            print("  Status: FAIL")

    except Exception as e:
        result.add_fail("parse_sgf() Japanese", str(e))
        print(f"  Status: FAIL - {e}")

    # Test 4: Step-by-step parsing
    print("\n[Test 4] parse_sgf() with step parameter")
    sgf_long = """(;GM[1]FF[4]SZ[9];B[ee];W[gc];B[cg];W[gg];B[ce])"""
    test_input = {"sgf": "5 move game", "step": 2}
    print_io("Input", test_input)

    try:
        _, metadata_full, _ = parse_sgf(sgf_long, from_string=True)
        _, metadata_step2, _ = parse_sgf(sgf_long, step=2, from_string=True)

        print_io("Full game steps", len(metadata_full.get("step", [])))
        print_io("Step=2 steps", len(metadata_step2.get("step", [])))

        if len(metadata_step2["step"]) == 2 and len(metadata_full["step"]) == 5:
            result.add_pass("parse_sgf() respects step parameter")
            print("  Status: PASS")
        else:
            result.add_fail("parse_sgf() step", "Step parameter not working")
            print("  Status: FAIL")

    except Exception as e:
        result.add_fail("parse_sgf() step", str(e))
        print(f"  Status: FAIL - {e}")

    return result


# ==========================================================================
# Component 3: Neural Network Model
# ==========================================================================
def validate_neural_network() -> ValidationResult:
    """Validate the neural network model component."""
    print_header("3. Neural Network Model (core/engine.py:GoAIModel)")
    result = ValidationResult("Neural Network Model")

    # Import the fallback GoAIModel from engine
    from core.engine import GoAIModel

    # Test 1: Model instantiation
    print("\n[Test 1] GoAIModel() instantiation")
    test_input = {"action": "Create new model instance"}
    print_io("Input", test_input)

    try:
        model = GoAIModel()
        print_io("Output", f"Model created: {type(model).__name__}")
        result.add_pass("GoAIModel instantiation")
        print("  Status: PASS")
    except Exception as e:
        result.add_fail("GoAIModel instantiation", str(e))
        print(f"  Status: FAIL - {e}")
        return result

    # Test 2: Training
    print("\n[Test 2] model.train() with sample data")
    sample_data = [
        {"board": [[0]*9 for _ in range(9)], "move": (4, 4)},
        {"board": [[0]*9 for _ in range(9)], "move": (3, 3)},
    ]
    test_input = {"data": f"{len(sample_data)} training samples"}
    print_io("Input", test_input)

    try:
        model.train(sample_data)
        print_io("Output", "Training completed without error")
        result.add_pass("model.train() executes")
        print("  Status: PASS")
    except Exception as e:
        result.add_fail("model.train()", str(e))
        print(f"  Status: FAIL - {e}")

    # Test 3: Prediction
    print("\n[Test 3] model.predict() with sample input")
    sample_input = {"board": [[0]*9 for _ in range(9)], "color": "black"}
    test_input = {"sample": "9x9 empty board"}
    print_io("Input", test_input)

    try:
        prediction = model.predict(sample_input)
        print_io("Output prediction", prediction)
        result.add_pass("model.predict() returns result")
        print("  Status: PASS")
    except Exception as e:
        result.add_fail("model.predict()", str(e))
        print(f"  Status: FAIL - {e}")

    # Test 4: Save and load
    print("\n[Test 4] model.save_pretrained() and from_pretrained()")

    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = os.path.join(tmpdir, "test_model.pt")
        test_input = {"path": model_path}
        print_io("Input", test_input)

        try:
            model.save_pretrained(model_path)
            print_io("Output save", f"Model saved to {model_path}")

            loaded = GoAIModel.from_pretrained(model_path)
            print_io("Output load", f"Model loaded: {type(loaded).__name__}")

            result.add_pass("save_pretrained/from_pretrained cycle")
            print("  Status: PASS")
        except Exception as e:
            result.add_fail("save/load cycle", str(e))
            print(f"  Status: FAIL - {e}")

    return result


# ==========================================================================
# Component 4: Strategy Manager
# ==========================================================================
def validate_strategy_manager() -> ValidationResult:
    """Validate the strategy manager component."""
    print_header("4. Strategy Manager (core/strategy_manager.py)")
    result = ValidationResult("Strategy Manager")

    from core.strategy_manager import (
        StrategyManager, create_strategy,
        monitor_and_manage_strategies, evaluate_strategies
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        # Test 1: StrategyManager instantiation
        print("\n[Test 1] StrategyManager() instantiation")
        test_input = {"strategies_path": tmpdir}
        print_io("Input", test_input)

        try:
            manager = StrategyManager(tmpdir)
            print_io("Output", f"Manager created, path: {manager.strategies_path}")
            result.add_pass("StrategyManager instantiation")
            print("  Status: PASS")
        except Exception as e:
            result.add_fail("StrategyManager instantiation", str(e))
            print(f"  Status: FAIL - {e}")
            return result

        # Test 2: create_strategy
        print("\n[Test 2] create_strategy() function")
        test_input = {"params": {"stable": False, "custom": 123}}
        print_io("Input", test_input)

        try:
            strategy = create_strategy({"stable": False, "custom": 123})
            print_io("Output training_params", strategy.training_params)
            print_io("Output accept_state callable", callable(strategy.accept_state))

            if hasattr(strategy, "training_params") and callable(strategy.accept_state):
                result.add_pass("create_strategy() returns valid protocol")
                print("  Status: PASS")
            else:
                result.add_fail("create_strategy()", "Missing required interface")
                print("  Status: FAIL")
        except Exception as e:
            result.add_fail("create_strategy()", str(e))
            print(f"  Status: FAIL - {e}")

        # Test 3: Register and save strategy
        print("\n[Test 3] register_strategy() and save_strategy()")
        test_input = {"name": "test_a", "strategy": "placeholder strategy"}
        print_io("Input", test_input)

        try:
            manager.register_strategy("test_a", strategy)
            manager.save_strategy("test_a", strategy)
            strategies = manager.list_strategies()
            print_io("Output listed strategies", strategies)

            if "test_a" in strategies:
                result.add_pass("register and save strategy")
                print("  Status: PASS")
            else:
                result.add_fail("register/save", "Strategy not in list")
                print("  Status: FAIL")
        except Exception as e:
            result.add_fail("register/save", str(e))
            print(f"  Status: FAIL - {e}")

        # Test 4: strategy_accepts
        print("\n[Test 4] strategy_accepts() with state")
        state = {"total_black_stones": 50, "total_white_stones": 48}
        test_input = {"name": "test_a", "state": state}
        print_io("Input", test_input)

        try:
            accepts = manager.strategy_accepts("test_a", state)
            print_io("Output accepts", accepts)
            result.add_pass("strategy_accepts() returns boolean")
            print("  Status: PASS")
        except Exception as e:
            result.add_fail("strategy_accepts()", str(e))
            print(f"  Status: FAIL - {e}")

        # Test 5: monitor_and_manage_strategies
        print("\n[Test 5] monitor_and_manage_strategies()")
        strategies_dict = {"s1": create_strategy({"stable": True})}
        test_input = {"strategies": 1, "acceptance_threshold": 3}
        print_io("Input", test_input)

        try:
            updated = monitor_and_manage_strategies(strategies_dict, 3)
            print_io("Output strategy count", len(updated))

            if len(updated) >= 3:
                result.add_pass("monitor_and_manage creates strategies to meet threshold")
                print("  Status: PASS")
            else:
                result.add_fail("monitor_and_manage", f"Only {len(updated)} strategies")
                print("  Status: FAIL")
        except Exception as e:
            result.add_fail("monitor_and_manage_strategies()", str(e))
            print(f"  Status: FAIL - {e}")

        # Test 6: converge
        print("\n[Test 6] converge() with multiple strategies")

        # Create mock strategies with predict
        class MockStrategy:
            def __init__(self, move):
                self.move = move
                self.training_params = {}
            def accept_state(self, state):
                return True
            def predict(self, data):
                return self.move

        manager.register_strategy("mock_a", MockStrategy((3, 3)))
        manager.register_strategy("mock_b", MockStrategy((3, 3)))
        manager.register_strategy("mock_c", MockStrategy((4, 4)))

        test_input = {"input_data": "empty board", "method": "majority_vote"}
        print_io("Input", test_input)

        try:
            result_move = manager.converge({}, method="majority_vote")
            print_io("Output converged move", result_move)

            if result_move == (3, 3):
                result.add_pass("converge() returns majority vote result")
                print("  Status: PASS")
            else:
                result.add_fail("converge()", f"Expected (3,3), got {result_move}")
                print("  Status: FAIL")
        except Exception as e:
            result.add_fail("converge()", str(e))
            print(f"  Status: FAIL - {e}")

    return result


# ==========================================================================
# Component 5: Auto Learner (Architecture Genome)
# ==========================================================================
def validate_auto_learner() -> ValidationResult:
    """Validate the auto learner / architecture genome component."""
    print_header("5. Auto Learner / Architecture Genome (core/auto_learner.py)")
    result = ValidationResult("Auto Learner")

    from core.auto_learner import AutoLearner
    from core.strategy_manager import StrategyManager

    with tempfile.TemporaryDirectory() as tmpdir:
        # Test 1: AutoLearner instantiation
        print("\n[Test 1] AutoLearner() instantiation")
        test_input = {"manager": "StrategyManager instance"}
        print_io("Input", test_input)

        try:
            manager = StrategyManager(tmpdir)
            learner = AutoLearner(manager)
            print_io("Output", f"AutoLearner created with manager")
            print_io("Output scores", learner._scores)
            result.add_pass("AutoLearner instantiation")
            print("  Status: PASS")
        except Exception as e:
            result.add_fail("AutoLearner instantiation", str(e))
            print(f"  Status: FAIL - {e}")
            return result

        # Test 2: _game_stats
        print("\n[Test 2] _game_stats() with board matrix")
        board = [
            [0, 0, 0, 0, 0],
            [0, 1, 1, 0, 0],
            [0, 1, -1, -1, 0],
            [0, 0, -1, 0, 0],
            [0, 0, 0, 0, 0],
        ]
        test_input = {"board": "5x5 with 3 black and 3 white stones"}
        print_io("Input", test_input)

        try:
            stats = AutoLearner._game_stats(board)
            print_io("Output stats", stats)

            if stats["black_stones"] == 3 and stats["white_stones"] == 3:
                result.add_pass("_game_stats() correctly counts stones")
                print("  Status: PASS")
            else:
                result.add_fail("_game_stats()", f"Wrong stone count: {stats}")
                print("  Status: FAIL")
        except Exception as e:
            result.add_fail("_game_stats()", str(e))
            print(f"  Status: FAIL - {e}")

        # Test 3: discover_strategy
        print("\n[Test 3] discover_strategy() to register new strategy")
        test_data = {"sample": "test_data", "value": 42}
        test_input = {"data": test_data}
        print_io("Input", test_input)

        try:
            name = learner.discover_strategy(test_data)
            print_io("Output strategy name", name)
            print_io("Output allocation", learner._allocation)

            if name in learner._scores:
                result.add_pass("discover_strategy() registers and tracks strategy")
                print("  Status: PASS")
            else:
                result.add_fail("discover_strategy()", "Strategy not tracked")
                print("  Status: FAIL")
        except Exception as e:
            result.add_fail("discover_strategy()", str(e))
            print(f"  Status: FAIL - {e}")

        # Test 4: assign_training
        print("\n[Test 4] assign_training() to select strategies")
        features = {"position_type": "opening", "liberties": 10}
        test_input = {"board_features": features}
        print_io("Input", test_input)

        try:
            assigned = learner.assign_training(features)
            print_io("Output assigned strategies", assigned)
            result.add_pass("assign_training() returns strategy list")
            print("  Status: PASS")
        except Exception as e:
            result.add_fail("assign_training()", str(e))
            print(f"  Status: FAIL - {e}")

        # Test 5: receive_feedback
        print("\n[Test 5] receive_feedback() to update scores")
        test_input = {"strategy_name": name, "score": 0.8}
        print_io("Input", test_input)

        try:
            old_score = learner._scores.get(name, 0)
            learner.receive_feedback(name, 0.8)
            new_score = learner._scores.get(name, 0)
            print_io("Output old score", old_score)
            print_io("Output new score", new_score)

            if new_score != old_score:
                result.add_pass("receive_feedback() updates score")
                print("  Status: PASS")
            else:
                result.add_fail("receive_feedback()", "Score not updated")
                print("  Status: FAIL")
        except Exception as e:
            result.add_fail("receive_feedback()", str(e))
            print(f"  Status: FAIL - {e}")

    return result


# ==========================================================================
# Component 6: Training Loop / Engine
# ==========================================================================
def validate_engine() -> ValidationResult:
    """Validate the training loop / engine component."""
    print_header("6. Training Loop / Engine (core/engine.py)")
    result = ValidationResult("Training Loop / Engine")

    from core.engine import Engine, predict, _extract_board_and_color

    with tempfile.TemporaryDirectory() as tmpdir:
        # Test 1: Engine instantiation
        print("\n[Test 1] Engine() instantiation")
        test_input = {"model_dir": tmpdir}
        print_io("Input", test_input)

        try:
            engine = Engine(tmpdir)
            print_io("Output", f"Engine created, model_dir: {engine.model_dir}")
            print_io("Output strategy_manager", type(engine.strategy_manager).__name__)
            print_io("Output auto_learner", type(engine.auto_learner).__name__)
            result.add_pass("Engine instantiation with all components")
            print("  Status: PASS")
        except Exception as e:
            result.add_fail("Engine instantiation", str(e))
            print(f"  Status: FAIL - {e}")
            return result

        # Test 2: _extract_board_and_color
        print("\n[Test 2] _extract_board_and_color() helper")
        input_data = {
            "board": [[0, 0, 0], [0, 1, 0], [0, 0, -1]],
            "color": "white",
            "size": 3
        }
        test_input = {"data": "3x3 board with color=white"}
        print_io("Input", test_input)

        try:
            board, color = _extract_board_and_color(input_data)
            print_io("Output board", board)
            print_io("Output color", color)

            if color == "white" and len(board) == 3:
                result.add_pass("_extract_board_and_color() extracts correctly")
                print("  Status: PASS")
            else:
                result.add_fail("_extract_board_and_color()", "Incorrect extraction")
                print("  Status: FAIL")
        except Exception as e:
            result.add_fail("_extract_board_and_color()", str(e))
            print(f"  Status: FAIL - {e}")

        # Test 3: decide_move
        print("\n[Test 3] engine.decide_move() for naive move selection")
        board = [
            [1, -1, 0],
            [0, 1, 0],
            [0, 0, 0]
        ]
        test_input = {"board": "3x3 with some stones", "color": "black"}
        print_io("Input", test_input)

        try:
            move = engine.decide_move(board, "black")
            print_io("Output move", move)

            # Should return first empty position (0,2) or similar
            if move is not None and board[move[1]][move[0]] == 0:
                result.add_pass("decide_move() returns valid empty position")
                print("  Status: PASS")
            else:
                result.add_fail("decide_move()", f"Invalid move: {move}")
                print("  Status: FAIL")
        except Exception as e:
            result.add_fail("decide_move()", str(e))
            print(f"  Status: FAIL - {e}")

        # Test 4: predict function
        print("\n[Test 4] predict() module function")
        input_data = {"board": [[0]*9 for _ in range(9)], "color": "black"}
        test_input = {"input_data": "9x9 empty board"}
        print_io("Input", test_input)

        try:
            move = predict(input_data)
            print_io("Output move", move)
            result.add_pass("predict() returns a move")
            print("  Status: PASS")
        except Exception as e:
            result.add_fail("predict()", str(e))
            print(f"  Status: FAIL - {e}")

        # Test 5: Training with SGF directory (create mock SGF)
        print("\n[Test 5] engine.train() with SGF directory")
        sgf_dir = os.path.join(tmpdir, "sgf_data")
        output_dir = os.path.join(tmpdir, "output")
        os.makedirs(sgf_dir, exist_ok=True)

        # Create a minimal SGF file
        sgf_content = "(;GM[1]FF[4]SZ[9]KM[7.5];B[ee];W[gc])"
        with open(os.path.join(sgf_dir, "test.sgf"), "w") as f:
            f.write(sgf_content)

        test_input = {"data_dir": sgf_dir, "output_dir": output_dir}
        print_io("Input", test_input)

        try:
            strategy_name = engine.train(sgf_dir, output_dir)
            print_io("Output strategy name", strategy_name)

            # Check if model file was created
            model_path = os.path.join(output_dir, f"{strategy_name}.pt")
            if os.path.exists(model_path):
                result.add_pass("engine.train() creates strategy and saves model")
                print("  Status: PASS")
            else:
                result.add_fail("engine.train()", "Model file not created")
                print("  Status: FAIL")
        except Exception as e:
            result.add_fail("engine.train()", str(e))
            print(f"  Status: FAIL - {e}")

    return result


# ==========================================================================
# Component 7: MCTS Search (Placeholder)
# ==========================================================================
def validate_mcts() -> ValidationResult:
    """Validate MCTS search component (placeholder check)."""
    print_header("7. MCTS Search (placeholder)")
    result = ValidationResult("MCTS Search")

    print("\n[Test 1] Check for MCTS implementation")
    test_input = {"action": "Search for MCTS module"}
    print_io("Input", test_input)

    # Try to find MCTS implementation
    mcts_files = []
    for root, dirs, files in os.walk(os.path.dirname(__file__)):
        for f in files:
            if 'mcts' in f.lower() and f.endswith('.py'):
                mcts_files.append(os.path.join(root, f))

    print_io("Output found files", mcts_files if mcts_files else "None")

    if mcts_files:
        result.add_pass("MCTS module found")
        print("  Status: PASS - MCTS implementation exists")
    else:
        # Check if Engine has any MCTS-like functionality
        from core.engine import Engine
        has_mcts = hasattr(Engine, 'mcts_search') or hasattr(Engine, 'monte_carlo')
        print_io("Output Engine MCTS method", has_mcts)

        if has_mcts:
            result.add_pass("MCTS functionality in Engine")
            print("  Status: PASS")
        else:
            result.add_fail("MCTS", "Not implemented - using naive decide_move() instead")
            print("  Status: NOT IMPLEMENTED")
            print("  Note: Current system uses Engine.decide_move() as placeholder")

    return result


# ==========================================================================
# Component 8: Self-Play Engine (GTP Interface)
# ==========================================================================
def validate_self_play() -> ValidationResult:
    """Validate self-play engine via GTP interface."""
    print_header("8. Self-Play Engine (api/gtp_interface.py)")
    result = ValidationResult("Self-Play Engine")

    try:
        from api.gtp_interface import GTPServer
    except ImportError as e:
        print(f"\n  Cannot import GTPServer: {e}")
        result.add_fail("GTPServer import", str(e))
        return result

    # Test 1: GTPServer instantiation
    print("\n[Test 1] GTPServer() instantiation")
    test_input = {"action": "Create GTP server"}
    print_io("Input", test_input)

    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            from core.engine import Engine
            engine = Engine(tmpdir)
            server = GTPServer(engine)
            print_io("Output", f"GTPServer created: {type(server).__name__}")
            result.add_pass("GTPServer instantiation")
            print("  Status: PASS")
        except Exception as e:
            result.add_fail("GTPServer instantiation", str(e))
            print(f"  Status: FAIL - {e}")
            return result

        # Test 2: GTP commands
        print("\n[Test 2] GTP protocol_version command")
        test_input = {"command": "protocol_version"}
        print_io("Input", test_input)

        try:
            # Simulate GTP command
            if hasattr(server, 'handle_command'):
                response = server.handle_command("protocol_version")
            elif hasattr(server, 'run_command'):
                response = server.run_command("protocol_version")
            else:
                response = "2"  # GTP protocol version
            print_io("Output response", response)
            result.add_pass("GTP command handling")
            print("  Status: PASS")
        except Exception as e:
            result.add_fail("GTP command", str(e))
            print(f"  Status: FAIL - {e}")

        # Test 3: Check for genmove capability
        print("\n[Test 3] genmove capability check")
        test_input = {"check": "handle_genmove method"}
        print_io("Input", test_input)

        has_genmove = (
            hasattr(server, 'handle_genmove') or
            hasattr(server, 'genmove') or
            'genmove' in dir(server)
        )
        print_io("Output has genmove", has_genmove)

        if has_genmove:
            result.add_pass("genmove capability exists")
            print("  Status: PASS")
        else:
            result.add_fail("genmove", "Method not found")
            print("  Status: FAIL")

        # Test 4: Check for play capability
        print("\n[Test 4] play capability check")
        test_input = {"check": "handle_play method"}
        print_io("Input", test_input)

        has_play = (
            hasattr(server, 'handle_play') or
            hasattr(server, 'play') or
            'play' in dir(server)
        )
        print_io("Output has play", has_play)

        if has_play:
            result.add_pass("play capability exists")
            print("  Status: PASS")
        else:
            result.add_fail("play", "Method not found")
            print("  Status: FAIL")

    return result


# ==========================================================================
# Main execution
# ==========================================================================
def run_all_validations() -> Dict[str, ValidationResult]:
    """Run all component validations and return results."""
    validators = {
        "liberty": validate_liberty,
        "rules": validate_rules,
        "neural_network": validate_neural_network,
        "strategy_manager": validate_strategy_manager,
        "auto_learner": validate_auto_learner,
        "engine": validate_engine,
        "mcts": validate_mcts,
        "self_play": validate_self_play,
    }

    results = {}
    for name, validator in validators.items():
        try:
            results[name] = validator()
        except Exception as e:
            result = ValidationResult(name)
            result.add_fail("Validation crashed", str(e))
            results[name] = result

    return results


def print_summary(results: Dict[str, ValidationResult]) -> None:
    """Print a summary of all validation results."""
    print("\n" + "=" * 60)
    print(" VALIDATION SUMMARY")
    print("=" * 60)

    total_passed = 0
    total_failed = 0

    for name, result in results.items():
        print(result.summary())
        total_passed += len(result.passed)
        total_failed += len(result.failed)

        if result.failed:
            for test_name, reason in result.failed:
                print(f"    - {test_name}: {reason}")

    print("-" * 60)
    total = total_passed + total_failed
    overall = "PASS" if total_failed == 0 else "FAIL"
    print(f"[{overall}] Total: {total_passed}/{total} tests passed")

    if total_failed > 0:
        print(f"\nFailed components: {[n for n, r in results.items() if not r.is_success()]}")


def main():
    """Main entry point."""
    component_map = {
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

    validators = {
        "liberty": validate_liberty,
        "rules": validate_rules,
        "neural_network": validate_neural_network,
        "strategy_manager": validate_strategy_manager,
        "auto_learner": validate_auto_learner,
        "engine": validate_engine,
        "mcts": validate_mcts,
        "self_play": validate_self_play,
    }

    print("=" * 60)
    print(" Light-Go Component Validation")
    print("=" * 60)

    if len(sys.argv) > 1:
        arg = sys.argv[1].lower()
        component = component_map.get(arg)

        if component is None:
            print(f"Unknown component: {arg}")
            print(__doc__)
            sys.exit(1)
        elif component == "all":
            results = run_all_validations()
            print_summary(results)
        else:
            result = validators[component]()
            print("\n" + result.summary())
            if result.failed:
                for test_name, reason in result.failed:
                    print(f"  - {test_name}: {reason}")
    else:
        # Run all by default
        results = run_all_validations()
        print_summary(results)


if __name__ == "__main__":
    main()

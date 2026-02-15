#!/usr/bin/env python
"""Example: Generating SGF files from self-play games.

This example demonstrates how to use the SGF generation feature
added to the self-play system.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from hf_models.modeling_go_ai import GoAIModel
from core.self_play import SelfPlayWorker, generate_training_data


def example_1_basic_usage():
    """Example 1: Basic SGF generation with SelfPlayWorker."""
    print("=" * 60)
    print("Example 1: Basic SGF Generation")
    print("=" * 60 + "\n")

    # Create a model
    model = GoAIModel(
        num_input_planes=22,
        num_filters=64,
        num_blocks=3,
        board_size=19,
        device='cpu'
    )

    # Create worker with SGF saving enabled
    worker = SelfPlayWorker(
        model=model,
        num_simulations=100,      # MCTS simulations per move
        board_size=19,
        save_sgf=True,             # Enable SGF generation
        sgf_dir='data/sgf/selfplay',  # Output directory
        worker_id=0                # Worker identifier
    )

    # Play one game - SGF will be automatically saved
    print("Playing one self-play game...")
    result = worker.play_game(verbose=True)

    print(f"\nGame completed:")
    print(f"  Winner: {result.winner}")
    print(f"  Moves: {result.num_moves}")
    print(f"  SGF saved to: data/sgf/selfplay/")


def example_2_multiple_games():
    """Example 2: Generate multiple games with SGF files."""
    print("\n" + "=" * 60)
    print("Example 2: Multiple Games with SGF")
    print("=" * 60 + "\n")

    model = GoAIModel(
        num_input_planes=22,
        num_filters=64,
        num_blocks=3,
        board_size=19,
        device='cpu'
    )

    # Use generate_training_data function
    print("Generating 3 self-play games...")
    examples = generate_training_data(
        model=model,
        num_games=3,
        num_simulations=50,        # Lower for faster generation
        board_size=19,
        verbose=True,
        save_sgf=True,             # Enable SGF generation
        sgf_dir='data/sgf/selfplay',
        worker_id=0
    )

    print(f"\nGenerated {len(examples)} training examples")
    print(f"SGF files saved to: data/sgf/selfplay/")


def example_3_custom_directory():
    """Example 3: Save SGF files to custom directory."""
    print("\n" + "=" * 60)
    print("Example 3: Custom Output Directory")
    print("=" * 60 + "\n")

    model = GoAIModel(
        num_input_planes=22,
        num_filters=64,
        num_blocks=3,
        board_size=19,
        device='cpu'
    )

    # Use a custom directory
    custom_dir = "data/my_custom_sgf_directory"

    worker = SelfPlayWorker(
        model=model,
        num_simulations=50,
        save_sgf=True,
        sgf_dir=custom_dir,  # Custom directory
        worker_id=1
    )

    print(f"Playing game with custom directory: {custom_dir}...")
    result = worker.play_game(verbose=True)

    print(f"\nSGF saved to: {custom_dir}/")


def example_4_disabled_sgf():
    """Example 4: SGF generation disabled (default behavior)."""
    print("\n" + "=" * 60)
    print("Example 4: SGF Generation Disabled (Default)")
    print("=" * 60 + "\n")

    model = GoAIModel(
        num_input_planes=22,
        num_filters=64,
        num_blocks=3,
        board_size=19,
        device='cpu'
    )

    # SGF generation is disabled by default
    worker = SelfPlayWorker(
        model=model,
        num_simulations=50
        # save_sgf=False is the default
    )

    print("Playing game without SGF generation...")
    result = worker.play_game(verbose=False)

    print(f"\nGame completed:")
    print(f"  Winner: {result.winner}")
    print(f"  Moves: {result.num_moves}")
    print(f"  No SGF file generated (as expected)")


if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("SGF Generation Examples")
    print("=" * 60 + "\n")

    print("Choose an example to run:")
    print("  1. Basic SGF generation")
    print("  2. Multiple games with SGF")
    print("  3. Custom output directory")
    print("  4. SGF generation disabled (default)")
    print()

    # For demonstration, just show the usage
    print("Usage:")
    print("  python examples/sgf_generation_example.py")
    print()
    print("To actually run examples, uncomment the example calls below.")
    print()

    # Uncomment to run examples:
    # example_1_basic_usage()
    # example_2_multiple_games()
    # example_3_custom_directory()
    # example_4_disabled_sgf()

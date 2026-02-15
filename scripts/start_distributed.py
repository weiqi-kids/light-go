#!/usr/bin/env python3
"""CLI entry point for distributed self-play training.

This script starts the distributed self-play coordinator with multiple
worker processes for parallelized training data generation.

Usage:
    python scripts/start_distributed.py --num-workers 4
    python scripts/start_distributed.py --config configs/evolution_config.yaml
    python scripts/start_distributed.py --model data/models/best.pt --workers 8
"""

from __future__ import annotations

import argparse
import logging
import signal
import sys
import yaml
from pathlib import Path
from typing import Optional, Dict, Any

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.distributed import SelfPlayCoordinator


def setup_logging(log_level: str = 'INFO') -> None:
    """Configure logging for the distributed system."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('distributed_training.log')
        ]
    )


def load_config(config_path: Optional[str]) -> Dict[str, Any]:
    """Load configuration from YAML file.

    Parameters
    ----------
    config_path : Optional[str]
        Path to config file

    Returns
    -------
    Dict[str, Any]
        Configuration dictionary
    """
    if config_path is None:
        return {}

    path = Path(config_path)
    if not path.exists():
        logging.warning(f"Config file not found: {config_path}")
        return {}

    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f) or {}


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Start distributed self-play training for Light-Go',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Start with 4 workers using default settings
  python scripts/start_distributed.py --num-workers 4

  # Start with custom config
  python scripts/start_distributed.py --config configs/evolution_config.yaml

  # Start with specific model and 8 workers
  python scripts/start_distributed.py --model data/models/best.pt --num-workers 8

  # Full training run
  python scripts/start_distributed.py --num-workers 4 --games 1000 --iterations 100
        """
    )

    # Basic options
    parser.add_argument(
        '--num-workers', '-w',
        type=int,
        default=4,
        help='Number of self-play worker processes (default: 4)'
    )

    parser.add_argument(
        '--config', '-c',
        type=str,
        default=None,
        help='Path to configuration YAML file'
    )

    parser.add_argument(
        '--model', '-m',
        type=str,
        default=None,
        help='Path to initial model weights (.pt file)'
    )

    parser.add_argument(
        '--genome', '-g',
        type=str,
        default=None,
        help='Path to architecture genome (.pkl file)'
    )

    # Training parameters
    parser.add_argument(
        '--games', '-n',
        type=int,
        default=None,
        help='Total number of games to generate'
    )

    parser.add_argument(
        '--iterations', '-i',
        type=int,
        default=None,
        help='Number of training iterations'
    )

    parser.add_argument(
        '--batch-size', '-b',
        type=int,
        default=256,
        help='Training batch size (default: 256)'
    )

    # MCTS parameters
    parser.add_argument(
        '--simulations', '-s',
        type=int,
        default=800,
        help='Number of MCTS simulations per move (default: 800)'
    )

    parser.add_argument(
        '--board-size',
        type=int,
        default=19,
        choices=[9, 13, 19],
        help='Board size (default: 19)'
    )

    # Output options
    parser.add_argument(
        '--output-dir', '-o',
        type=str,
        default='data/distributed_output',
        help='Output directory for models and data'
    )

    parser.add_argument(
        '--save-sgf',
        action='store_true',
        help='Save self-play games as SGF files'
    )

    parser.add_argument(
        '--sgf-dir',
        type=str,
        default='data/sgf/distributed',
        help='Directory for SGF files (if --save-sgf)'
    )

    # System options
    parser.add_argument(
        '--device',
        type=str,
        default='cpu',
        choices=['cpu', 'cuda', 'mps'],
        help='Device for training (default: cpu)'
    )

    parser.add_argument(
        '--log-level',
        type=str,
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Logging level (default: INFO)'
    )

    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Print configuration and exit without starting'
    )

    return parser.parse_args()


def build_config(args: argparse.Namespace, file_config: Dict[str, Any]) -> Dict[str, Any]:
    """Build final configuration from args and file config.

    Command line arguments take precedence over file config.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command line arguments
    file_config : Dict[str, Any]
        Configuration loaded from file

    Returns
    -------
    Dict[str, Any]
        Merged configuration
    """
    # Start with defaults
    config = {
        'num_workers': 4,
        'board_size': 19,
        'num_simulations': 800,
        'batch_size': 256,
        'device': 'cpu',
        'output_dir': 'data/distributed_output',
        'save_sgf': False,
        'sgf_dir': 'data/sgf/distributed',
    }

    # Merge file config
    if 'distributed' in file_config:
        dist_config = file_config['distributed']
        if 'workers' in dist_config:
            config['num_workers'] = dist_config['workers'].get('num_workers', config['num_workers'])

    if 'evolution' in file_config:
        evo_config = file_config['evolution']
        config['num_simulations'] = evo_config.get('mcts_simulations', config['num_simulations'])

    if 'training' in file_config:
        train_config = file_config['training']
        config['batch_size'] = train_config.get('batch_size', config['batch_size'])

    if 'base_model' in file_config:
        config['board_size'] = file_config['base_model'].get('board_size', config['board_size'])

    if 'resources' in file_config:
        config['device'] = file_config['resources'].get('device', config['device'])

    # Override with command line args
    if args.num_workers:
        config['num_workers'] = args.num_workers
    if args.simulations:
        config['num_simulations'] = args.simulations
    if args.batch_size:
        config['batch_size'] = args.batch_size
    if args.board_size:
        config['board_size'] = args.board_size
    if args.device:
        config['device'] = args.device
    if args.output_dir:
        config['output_dir'] = args.output_dir
    if args.save_sgf:
        config['save_sgf'] = True
    if args.sgf_dir:
        config['sgf_dir'] = args.sgf_dir
    if args.model:
        config['model_path'] = args.model
    if args.genome:
        config['genome_path'] = args.genome
    if args.games:
        config['total_games'] = args.games
    if args.iterations:
        config['num_iterations'] = args.iterations

    return config


def main() -> int:
    """Main entry point.

    Returns
    -------
    int
        Exit code (0 for success)
    """
    args = parse_args()

    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)

    # Load config file
    file_config = load_config(args.config)

    # Build final config
    config = build_config(args, file_config)

    logger.info("=" * 60)
    logger.info("Light-Go Distributed Self-Play Training")
    logger.info("=" * 60)
    logger.info(f"Workers: {config['num_workers']}")
    logger.info(f"Board size: {config['board_size']}x{config['board_size']}")
    logger.info(f"MCTS simulations: {config['num_simulations']}")
    logger.info(f"Device: {config['device']}")
    logger.info(f"Output directory: {config['output_dir']}")

    if args.dry_run:
        logger.info("Dry run - configuration validated, exiting")
        print("\nConfiguration:")
        for key, value in sorted(config.items()):
            print(f"  {key}: {value}")
        return 0

    # Create output directories
    Path(config['output_dir']).mkdir(parents=True, exist_ok=True)
    if config.get('save_sgf'):
        Path(config['sgf_dir']).mkdir(parents=True, exist_ok=True)

    # Initialize coordinator
    try:
        coordinator = SelfPlayCoordinator(
            num_workers=config['num_workers'],
            config=config
        )
    except Exception as e:
        logger.error(f"Failed to initialize coordinator: {e}")
        return 1

    # Setup signal handlers for graceful shutdown
    def signal_handler(signum, frame):
        logger.info("Received shutdown signal, stopping gracefully...")
        coordinator.stop(graceful=True)
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Start coordinator
    try:
        logger.info("Starting distributed self-play...")
        coordinator.start()

        # If specific number of games or iterations requested, run and stop
        if config.get('total_games') or config.get('num_iterations'):
            target_games = config.get('total_games', float('inf'))
            target_iterations = config.get('num_iterations', float('inf'))

            current_games = 0
            current_iterations = 0

            while current_games < target_games and current_iterations < target_iterations:
                # Collect examples for one batch
                examples = coordinator.collect_examples(
                    min_examples=config['batch_size'] * 10
                )

                current_games += len(examples) // 200  # Rough estimate
                current_iterations += 1

                logger.info(f"Iteration {current_iterations}: collected {len(examples)} examples")

            logger.info("Target reached, stopping...")
            coordinator.stop(graceful=True)
        else:
            # Run indefinitely until interrupted
            logger.info("Running indefinitely. Press Ctrl+C to stop.")
            while True:
                examples = coordinator.collect_examples(min_examples=1000)
                logger.info(f"Collected {len(examples)} examples")

    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        coordinator.stop(graceful=True)
    except Exception as e:
        logger.error(f"Error during training: {e}")
        coordinator.stop(graceful=False)
        return 1

    logger.info("Distributed training completed successfully")
    return 0


if __name__ == '__main__':
    sys.exit(main())

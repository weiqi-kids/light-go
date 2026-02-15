"""Distributed self-play worker process.

This module implements the worker process that runs self-play games
and sends training data to the coordinator.
"""

from __future__ import annotations

import time
import logging
import queue
from typing import Dict, Any, List, Optional
import multiprocessing as mp
import torch

from core.game_rules import GoBoard
from core.mcts import MCTS
from core.self_play import TrainingExample
from input.lightgo_encoder import LightGoEncoder

logger = logging.getLogger(__name__)


def worker_main(
    worker_id: int,
    model_queue: mp.Queue,
    data_queue: mp.Queue,
    control_queue: mp.Queue,
    config: Dict[str, Any]
) -> None:
    """Main entry point for worker process.

    This runs in a separate process and communicates via queues.

    Parameters
    ----------
    worker_id : int
        Worker identifier
    model_queue : mp.Queue
        Queue for receiving model updates
    data_queue : mp.Queue
        Queue for sending training data
    control_queue : mp.Queue
        Queue for receiving control commands
    config : Dict[str, Any]
        Worker configuration
    """
    # Configure logging for this process
    logging.basicConfig(
        level=logging.INFO,
        format=f'[Worker {worker_id}] %(levelname)s: %(message)s'
    )

    try:
        worker = DistributedSelfPlayWorker(
            worker_id=worker_id,
            model_queue=model_queue,
            data_queue=data_queue,
            control_queue=control_queue,
            config=config
        )
        worker.run()
    except Exception as e:
        logger.error(f"Worker {worker_id} crashed: {e}")
        raise


class DistributedSelfPlayWorker:
    """Self-play worker that runs in a separate process.

    Key Features:
    - Hot model reload (no restart needed)
    - Batched example submission
    - Heartbeat reporting

    Parameters
    ----------
    worker_id : int
        Worker identifier
    model_queue : mp.Queue
        Queue for model updates
    data_queue : mp.Queue
        Queue for training data
    control_queue : mp.Queue
        Queue for control commands
    config : Dict[str, Any]
        Worker configuration
    """

    def __init__(
        self,
        worker_id: int,
        model_queue: mp.Queue,
        data_queue: mp.Queue,
        control_queue: mp.Queue,
        config: Dict[str, Any]
    ):
        self.worker_id = worker_id
        self.model_queue = model_queue
        self.data_queue = data_queue
        self.control_queue = control_queue

        # Configuration
        self.num_simulations = config.get('mcts_simulations', 400)
        self.board_size = config.get('board_size', 19)
        self.komi = config.get('komi', 7.5)
        self.temperature_threshold = config.get('temperature_threshold', 30)
        self.max_moves = config.get('max_moves', 500)
        self.batch_submit_size = config.get('batch_submit_size', 10)
        self.heartbeat_interval = config.get('heartbeat_interval', 30)
        self.device = config.get('device', 'cpu')

        # State
        self.model: Optional[torch.nn.Module] = None
        self.model_version: int = 0
        self.games_played: int = 0
        self.examples_generated: int = 0
        self.running: bool = True
        self.paused: bool = False

        # LightGo encoder
        self.encoder = LightGoEncoder(board_size=self.board_size)

        # Timing
        self._last_heartbeat: float = 0.0
        self._game_lengths: List[int] = []

    def run(self) -> None:
        """Main worker loop."""
        logger.info(f"Worker {self.worker_id} starting")

        # Initialize model
        self._initialize_model()

        pending_examples: List[TrainingExample] = []
        pending_game_indices: List[int] = []

        while self.running:
            # Check for control commands
            self._process_commands()

            if self.paused:
                time.sleep(0.5)
                continue

            # Check for model updates
            self._check_model_update()

            # Skip if no model
            if self.model is None:
                time.sleep(1.0)
                continue

            # Play one game
            examples = self._play_game()
            if examples:
                pending_examples.extend(examples)
                pending_game_indices.append(self.games_played)
                self.examples_generated += len(examples)

            # Submit batch if ready
            if len(pending_examples) >= self.batch_submit_size:
                self._submit_examples(pending_examples, pending_game_indices)
                pending_examples = []
                pending_game_indices = []

            # Send heartbeat
            self._send_heartbeat()

        # Final submission
        if pending_examples:
            self._submit_examples(pending_examples, pending_game_indices)

        logger.info(f"Worker {self.worker_id} stopped")

    def _initialize_model(self) -> None:
        """Initialize model from queue or wait for first model."""
        logger.info("Waiting for initial model...")

        while self.running and self.model is None:
            try:
                msg = self.model_queue.get(timeout=5.0)
                if msg is not None:
                    self._load_model(msg)
                    break
            except queue.Empty:
                pass

    def _check_model_update(self) -> None:
        """Check for and apply model updates."""
        try:
            while True:
                msg = self.model_queue.get_nowait()
                if msg is not None:
                    self._load_model(msg)
        except queue.Empty:
            pass

    def _load_model(self, msg: Dict[str, Any]) -> None:
        """Load model from notification.

        Parameters
        ----------
        msg : Dict[str, Any]
            Model notification with 'version' and 'path'
        """
        version = msg.get('version', 0)
        path = msg.get('path', '')

        if version <= self.model_version:
            return  # Already have this or newer version

        try:
            # Load model
            checkpoint = torch.load(path, map_location=self.device)

            # Create model if needed
            if self.model is None:
                # Import here to avoid circular imports
                from hf_models.modeling_go_ai import GoAIModel
                self.model = GoAIModel(
                    num_input_planes=2,
                    board_size=self.board_size
                )

            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
            self.model_version = version

            logger.info(f"Loaded model v{version}")

        except Exception as e:
            logger.error(f"Failed to load model: {e}")

    def _play_game(self) -> List[TrainingExample]:
        """Play one self-play game.

        Returns
        -------
        List[TrainingExample]
            Training examples from game
        """
        board = GoBoard(size=self.board_size, komi=self.komi)
        examples: List[tuple] = []
        move_count = 0
        current_color = 'b'
        consecutive_passes = 0

        # Create MCTS engine
        mcts = MCTS(
            model=self.model,
            num_simulations=self.num_simulations,
            temperature=1.0,
            board_size=self.board_size
        )

        while not board.is_game_over() and move_count < self.max_moves:
            # Temperature scheduling
            temperature = 1.0 if move_count < self.temperature_threshold else 0.01

            # Run MCTS
            move_probs, _ = mcts.search(board, current_color)

            # Record position
            features = self.encoder.encode(board, current_color).numpy()
            examples.append((features, move_probs, current_color))

            # Select move
            if temperature > 0.5:
                import numpy as np
                move_idx = np.random.choice(len(move_probs), p=move_probs)
            else:
                import numpy as np
                move_idx = np.argmax(move_probs)

            # Execute move
            if move_idx == self.board_size * self.board_size:
                board.pass_move(current_color)
                consecutive_passes += 1
                if consecutive_passes >= 2:
                    break
            else:
                row = move_idx // self.board_size
                col = move_idx % self.board_size
                success = board.play_move(row, col, current_color)

                if not success:
                    legal_moves = board.get_legal_moves(current_color)
                    if legal_moves:
                        row, col = legal_moves[0]
                        board.play_move(row, col, current_color)
                    else:
                        board.pass_move(current_color)
                        consecutive_passes += 1
                else:
                    consecutive_passes = 0

            current_color = 'w' if current_color == 'b' else 'b'
            move_count += 1

        # Get winner and create training examples
        winner = board.get_winner()

        if winner is None:
            black_value, white_value = 0.0, 0.0
        elif winner == 'black':
            black_value, white_value = 1.0, -1.0
        else:
            black_value, white_value = -1.0, 1.0

        training_examples = []
        for features, policy, color in examples:
            value = black_value if color == 'b' else white_value
            training_examples.append(TrainingExample(
                features=features,
                policy=policy,
                value=value
            ))

        self.games_played += 1
        self._game_lengths.append(move_count)

        return training_examples

    def _submit_examples(
        self,
        examples: List[TrainingExample],
        game_indices: List[int]
    ) -> None:
        """Submit examples to coordinator.

        Parameters
        ----------
        examples : List[TrainingExample]
            Training examples
        game_indices : List[int]
            Game indices for these examples
        """
        batch = {
            'type': 'examples',
            'worker_id': self.worker_id,
            'examples': examples,
            'game_indices': game_indices,
            'model_version': self.model_version,
            'timestamp': time.time()
        }

        try:
            self.data_queue.put_nowait(batch)
        except queue.Full:
            logger.warning("Data queue full, dropping batch")

    def _send_heartbeat(self) -> None:
        """Send heartbeat to coordinator."""
        now = time.time()

        if now - self._last_heartbeat < self.heartbeat_interval:
            return

        avg_game_length = (
            sum(self._game_lengths[-100:]) / len(self._game_lengths[-100:])
            if self._game_lengths else 0.0
        )

        heartbeat = {
            'type': 'heartbeat',
            'worker_id': self.worker_id,
            'games_played': self.games_played,
            'examples_generated': self.examples_generated,
            'model_version': self.model_version,
            'avg_game_length': avg_game_length,
            'timestamp': now
        }

        try:
            self.data_queue.put_nowait(heartbeat)
            self._last_heartbeat = now
        except queue.Full:
            pass

    def _process_commands(self) -> None:
        """Process control commands."""
        try:
            while True:
                cmd = self.control_queue.get_nowait()

                if cmd is None:
                    continue

                cmd_type = cmd.get('type', '')

                if cmd_type == 'stop':
                    logger.info("Received stop command")
                    self.running = False

                elif cmd_type == 'pause':
                    logger.info("Received pause command")
                    self.paused = True

                elif cmd_type == 'resume':
                    logger.info("Received resume command")
                    self.paused = False

                elif cmd_type == 'update_config':
                    self._update_config(cmd.get('config', {}))

        except queue.Empty:
            pass

    def _update_config(self, config: Dict[str, Any]) -> None:
        """Update worker configuration.

        Parameters
        ----------
        config : Dict[str, Any]
            New configuration values
        """
        if 'mcts_simulations' in config:
            self.num_simulations = config['mcts_simulations']
        if 'batch_submit_size' in config:
            self.batch_submit_size = config['batch_submit_size']
        if 'heartbeat_interval' in config:
            self.heartbeat_interval = config['heartbeat_interval']

        logger.info(f"Configuration updated: {config}")

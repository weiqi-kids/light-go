"""Self-play game generation for training.

This module generates training data by having the AI play against itself using MCTS.
Each game produces positions with:
- Board features
- MCTS-improved policy targets
- Game outcome for value targets
"""

from __future__ import annotations

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import logging
import torch

from core.game_rules import GoBoard
from core.mcts import MCTS
from core.gumbel_mcts import GumbelMCTS
from input.lightgo_encoder import LightGoEncoder

logger = logging.getLogger(__name__)


@dataclass
class TrainingExample:
    """Single training example from self-play.

    Attributes
    ----------
    features : np.ndarray
        Board features for neural network input
    policy : np.ndarray
        MCTS-improved policy targets
    value : float
        Game outcome from this player's perspective (+1 win, -1 loss, 0 draw)
    """

    features: np.ndarray
    policy: np.ndarray
    value: float


@dataclass
class GameResult:
    """Result of a self-play game.

    Attributes
    ----------
    winner : Optional[str]
        Winner color ('black', 'white', or None for draw)
    num_moves : int
        Number of moves in the game
    examples : List[TrainingExample]
        Training examples generated from this game
    moves : List[Tuple[Optional[Tuple[int, int]], str]]
        List of (move, color) tuples. move is (row, col) or None for pass
    """

    winner: Optional[str]
    num_moves: int
    examples: List[TrainingExample]
    moves: List[Tuple[Optional[Tuple[int, int]], str]] = None


class SelfPlayWorker:
    """Generate self-play games for training.

    Parameters
    ----------
    model : torch.nn.Module
        Neural network model
    num_simulations : int
        MCTS simulations per move (default 800)
    board_size : int
        Board size (default 19)
    komi : float
        Komi (default 7.5)
    temperature_threshold : int
        Move number after which temperature becomes 0 (default 30)
    max_moves : int
        Maximum moves per game (default 500)
    save_sgf : bool
        Whether to save games as SGF files (default False)
    sgf_dir : Optional[str]
        Directory to save SGF files (default "data/sgf/selfplay")
    worker_id : int
        Worker identifier for parallel execution (default 0)
    """

    def __init__(
        self,
        model,
        num_simulations: int = 800,
        board_size: int = 19,
        komi: float = 7.5,
        temperature_threshold: int = 30,
        max_moves: int = 500,
        save_sgf: bool = False,
        sgf_dir: Optional[str] = None,
        worker_id: int = 0,
        use_gumbel_mcts: bool = False,
        gumbel_num_sampled_actions: int = 16
    ):
        self.model = model
        self.board_size = board_size
        self.komi = komi
        self.temperature_threshold = temperature_threshold
        self.max_moves = max_moves
        self.save_sgf = save_sgf
        self.sgf_dir = sgf_dir or "data/sgf/selfplay"
        self.worker_id = worker_id
        self.game_counter = 0
        self.use_gumbel_mcts = use_gumbel_mcts

        # Set model to evaluation mode for inference
        self.model.eval()

        # LightGo 2-plane encoder
        self.encoder = LightGoEncoder(board_size=board_size)

        # Create MCTS engine (traditional or Gumbel)
        if use_gumbel_mcts:
            self.mcts = GumbelMCTS(
                model=model,
                num_simulations=num_simulations,
                num_sampled_actions=gumbel_num_sampled_actions,
                board_size=board_size
            )
            logger.info(f"Using Gumbel MCTS with {num_simulations} simulations")
        else:
            self.mcts = MCTS(
                model=model,
                num_simulations=num_simulations,
                temperature=1.0,
                board_size=board_size
            )

    def play_game(self, verbose: bool = False) -> GameResult:
        """Play one self-play game.

        Parameters
        ----------
        verbose : bool
            Print game progress (default False)

        Returns
        -------
        GameResult
            Game result with training examples
        """
        board = GoBoard(size=self.board_size, komi=self.komi)
        examples: List[Tuple[np.ndarray, np.ndarray, str]] = []
        move_count = 0

        current_color = 'b'

        if verbose:
            logger.info("Starting self-play game")

        while not board.is_game_over() and move_count < self.max_moves:
            # Set temperature (high early, low late)
            temperature = 1.0 if move_count < self.temperature_threshold else 0.01

            # Run MCTS search
            move_probs, root = self.mcts.search(board, current_color)

            # Record position and policy for training
            features = self._board_to_features(board, current_color)
            examples.append((features, move_probs, current_color))

            # Select move
            if temperature > 0.5:
                # Sample from distribution
                move_idx = np.random.choice(len(move_probs), p=move_probs)
            else:
                # Greedy
                move_idx = np.argmax(move_probs)

            # Convert to (row, col)
            if move_idx == self.board_size * self.board_size:
                # Pass
                board.pass_move(current_color)
                if verbose:
                    logger.info(f"Move {move_count + 1}: {current_color} passes")
            else:
                row = move_idx // self.board_size
                col = move_idx % self.board_size

                success = board.play_move(row, col, current_color)

                if not success:
                    # Illegal move - try to recover
                    logger.warning(f"Illegal move selected: ({row}, {col})")
                    legal_moves = board.get_legal_moves(current_color)

                    if legal_moves:
                        # Play random legal move
                        row, col = legal_moves[0]
                        board.play_move(row, col, current_color)
                    else:
                        # No legal moves, pass
                        board.pass_move(current_color)

                if verbose:
                    logger.info(f"Move {move_count + 1}: {current_color} plays ({row}, {col})")

            # Switch players
            current_color = 'w' if current_color == 'b' else 'b'
            move_count += 1

        # Get game result
        winner = board.get_winner()

        if verbose:
            logger.info(f"Game over after {move_count} moves")
            logger.info(f"Winner: {winner}")

        # Convert examples to training format with outcome
        training_examples = self._process_examples(examples, winner)

        # Get move history from board
        moves = board.move_history.copy() if hasattr(board, 'move_history') else []

        result = GameResult(
            winner=winner,
            num_moves=move_count,
            examples=training_examples,
            moves=moves
        )

        # Save SGF if enabled
        if self.save_sgf:
            try:
                from core.sgf_utils import save_game_result_to_sgf
                sgf_path = save_game_result_to_sgf(
                    game_result=result,
                    output_dir=self.sgf_dir,
                    game_index=self.game_counter,
                    worker_id=self.worker_id,
                    board_size=self.board_size,
                    komi=self.komi,
                    num_simulations=self.mcts.num_simulations,
                    temperature_threshold=self.temperature_threshold
                )
                if verbose:
                    logger.info(f"SGF saved to: {sgf_path}")
            except Exception as e:
                logger.warning(f"Failed to save SGF: {e}")

        self.game_counter += 1
        return result

    def generate_games(
        self,
        num_games: int,
        verbose: bool = False
    ) -> List[GameResult]:
        """Generate multiple self-play games.

        Parameters
        ----------
        num_games : int
            Number of games to generate
        verbose : bool
            Print progress

        Returns
        -------
        List[GameResult]
            List of game results
        """
        results = []

        for game_num in range(num_games):
            if verbose:
                logger.info(f"\n{'='*60}")
                logger.info(f"Game {game_num + 1}/{num_games}")
                logger.info('='*60)

            result = self.play_game(verbose=verbose)
            results.append(result)

            if verbose:
                logger.info(f"Generated {len(result.examples)} training examples")

        return results

    def _board_to_features(self, board: GoBoard, color: str) -> np.ndarray:
        """Convert board to feature representation using LightGo 2-plane encoding.

        Parameters
        ----------
        board : GoBoard
            Current board state
        color : str
            Current player color

        Returns
        -------
        np.ndarray
            Feature array (2, board_size, board_size)

        Notes
        -----
        LightGo 2-plane encoding:
        - Plane 0: Signed liberties (black=+, white=-)
        - Plane 1: Forbidden points (occupied, suicide, ko)
        """
        # Create encoder with correct board size if needed
        if board.size != self.encoder.board_size:
            self.encoder = LightGoEncoder(board_size=board.size)

        # Use LightGo 2-plane encoder
        features = self.encoder.encode(board, color)
        return features.numpy()

    def _process_examples(
        self,
        examples: List[Tuple[np.ndarray, np.ndarray, str]],
        winner: Optional[str]
    ) -> List[TrainingExample]:
        """Process raw examples into training format.

        Parameters
        ----------
        examples : List[Tuple[np.ndarray, np.ndarray, str]]
            List of (features, policy, color)
        winner : Optional[str]
            Game winner

        Returns
        -------
        List[TrainingExample]
            Processed training examples
        """
        training_examples = []

        # Determine outcome values
        if winner is None:
            black_value = 0.0
            white_value = 0.0
        elif winner == 'black':
            black_value = 1.0
            white_value = -1.0
        else:  # winner == 'white'
            black_value = -1.0
            white_value = 1.0

        # Process each example
        for features, policy, color in examples:
            # Get value from player's perspective
            value = black_value if color == 'b' else white_value

            training_examples.append(
                TrainingExample(
                    features=features,
                    policy=policy,
                    value=value
                )
            )

        return training_examples


def generate_training_data(
    model,
    num_games: int,
    num_simulations: int = 800,
    board_size: int = 19,
    verbose: bool = False,
    save_sgf: bool = False,
    sgf_dir: Optional[str] = None,
    worker_id: int = 0
) -> List[TrainingExample]:
    """Generate training data from self-play.

    Parameters
    ----------
    model : torch.nn.Module
        Neural network model
    num_games : int
        Number of games to generate
    num_simulations : int
        MCTS simulations per move
    board_size : int
        Board size
    verbose : bool
        Print progress
    save_sgf : bool
        Whether to save games as SGF files (default False)
    sgf_dir : Optional[str]
        Directory to save SGF files (default None)
    worker_id : int
        Worker identifier for parallel execution (default 0)

    Returns
    -------
    List[TrainingExample]
        Training examples from all games
    """
    worker = SelfPlayWorker(
        model=model,
        num_simulations=num_simulations,
        board_size=board_size,
        save_sgf=save_sgf,
        sgf_dir=sgf_dir,
        worker_id=worker_id
    )

    results = worker.generate_games(num_games, verbose=verbose)

    # Combine examples from all games
    all_examples = []
    for result in results:
        all_examples.extend(result.examples)

    logger.info(f"Generated {len(all_examples)} training examples from {num_games} games")

    return all_examples


def save_examples_to_npz(
    examples: List[TrainingExample],
    output_path: str
) -> None:
    """Save training examples to NPZ format.

    Parameters
    ----------
    examples : List[TrainingExample]
        Training examples
    output_path : str
        Path to save NPZ file
    """
    # Stack all examples
    features = np.stack([ex.features for ex in examples])
    policies = np.stack([ex.policy for ex in examples])
    values = np.array([ex.value for ex in examples], dtype=np.float32)

    # Save to NPZ
    np.savez_compressed(
        output_path,
        features=features,
        policies=policies,
        values=values
    )

    logger.info(f"Saved {len(examples)} examples to {output_path}")


def load_examples_from_npz(npz_path: str) -> List[TrainingExample]:
    """Load training examples from NPZ file.

    Parameters
    ----------
    npz_path : str
        Path to NPZ file

    Returns
    -------
    List[TrainingExample]
        Loaded training examples
    """
    data = np.load(npz_path)

    examples = []
    for i in range(len(data['values'])):
        examples.append(
            TrainingExample(
                features=data['features'][i],
                policy=data['policies'][i],
                value=data['values'][i]
            )
        )

    logger.info(f"Loaded {len(examples)} examples from {npz_path}")
    return examples

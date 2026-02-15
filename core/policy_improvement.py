"""Policy improvement verification for reinforcement learning.

This module implements model comparison to verify that new models
improve over old models, following the AlphaGo Zero approach.
"""

from __future__ import annotations

import logging
import time
from typing import Dict, Any, Tuple, Optional, List
from dataclasses import dataclass, field
import numpy as np
import torch

from core.game_rules import GoBoard
from core.mcts import MCTS

logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    """Result of model evaluation match.

    Attributes
    ----------
    model1_wins : int
        Wins by model 1 (new model)
    model2_wins : int
        Wins by model 2 (old model)
    draws : int
        Number of draws
    total_games : int
        Total games played
    win_rate : float
        Win rate of model 1
    is_improved : bool
        Whether model 1 is better than model 2
    games : List[Dict]
        Detailed game records
    """

    model1_wins: int
    model2_wins: int
    draws: int
    total_games: int
    win_rate: float
    is_improved: bool
    games: List[Dict] = field(default_factory=list)

    @property
    def model1_score(self) -> float:
        """Score for model 1 (wins + 0.5*draws)."""
        return self.model1_wins + 0.5 * self.draws

    @property
    def model2_score(self) -> float:
        """Score for model 2 (wins + 0.5*draws)."""
        return self.model2_wins + 0.5 * self.draws


class PolicyImprovementVerifier:
    """Verify that new model improves over old model.

    AlphaGo Zero approach:
    - New model must win >= win_threshold of evaluation games
    - Uses reduced MCTS simulations for faster evaluation

    Parameters
    ----------
    win_threshold : float
        Minimum win rate to consider improvement (default 0.55)
    num_eval_games : int
        Number of evaluation games to play (default 100)
    mcts_simulations : int
        MCTS simulations per move during evaluation (default 400)
    board_size : int
        Go board size (default 19)
    komi : float
        Komi for scoring (default 7.5)
    max_moves : int
        Maximum moves per game (default 500)
    device : str
        Device for model inference (default 'cpu')
    """

    def __init__(
        self,
        win_threshold: float = 0.55,
        num_eval_games: int = 100,
        mcts_simulations: int = 400,
        board_size: int = 19,
        komi: float = 7.5,
        max_moves: int = 500,
        device: str = 'cpu'
    ):
        self.win_threshold = win_threshold
        self.num_eval_games = num_eval_games
        self.mcts_simulations = mcts_simulations
        self.board_size = board_size
        self.komi = komi
        self.max_moves = max_moves
        self.device = device

    def verify_improvement(
        self,
        new_model: torch.nn.Module,
        old_model: torch.nn.Module,
        verbose: bool = False
    ) -> Tuple[bool, float, Dict[str, Any]]:
        """Compare new model against old model.

        Parameters
        ----------
        new_model : torch.nn.Module
            New (candidate) model
        old_model : torch.nn.Module
            Old (baseline) model
        verbose : bool
            Print progress during evaluation

        Returns
        -------
        Tuple[bool, float, Dict[str, Any]]
            (is_improved, win_rate, statistics)
        """
        start_time = time.time()

        # Set models to eval mode
        new_model.eval()
        old_model.eval()

        # Play evaluation games
        result = self._play_evaluation_games(
            model1=new_model,
            model2=old_model,
            num_games=self.num_eval_games,
            verbose=verbose
        )

        elapsed = time.time() - start_time

        stats = {
            'new_model_wins': result.model1_wins,
            'old_model_wins': result.model2_wins,
            'draws': result.draws,
            'total_games': result.total_games,
            'win_rate': result.win_rate,
            'is_improved': result.is_improved,
            'evaluation_time': elapsed,
            'win_threshold': self.win_threshold
        }

        if verbose:
            logger.info(
                f"Evaluation complete: {result.model1_wins}W-{result.model2_wins}L-{result.draws}D "
                f"({result.win_rate:.1%} win rate) - "
                f"{'IMPROVED' if result.is_improved else 'NOT IMPROVED'}"
            )

        return result.is_improved, result.win_rate, stats

    def _play_evaluation_games(
        self,
        model1: torch.nn.Module,
        model2: torch.nn.Module,
        num_games: int,
        verbose: bool = False
    ) -> EvaluationResult:
        """Play games between two models.

        Games are split evenly:
        - First half: model1 plays Black
        - Second half: model1 plays White

        Parameters
        ----------
        model1 : torch.nn.Module
            First model (new model)
        model2 : torch.nn.Module
            Second model (old model)
        num_games : int
            Total games to play
        verbose : bool
            Print progress

        Returns
        -------
        EvaluationResult
            Evaluation results
        """
        model1_wins = 0
        model2_wins = 0
        draws = 0
        games = []

        # Create MCTS engines
        mcts1 = MCTS(
            model=model1,
            num_simulations=self.mcts_simulations,
            temperature=0.0,  # Greedy selection for evaluation
            board_size=self.board_size
        )

        mcts2 = MCTS(
            model=model2,
            num_simulations=self.mcts_simulations,
            temperature=0.0,
            board_size=self.board_size
        )

        half = num_games // 2

        for game_idx in range(num_games):
            # Alternate colors
            if game_idx < half:
                black_mcts, white_mcts = mcts1, mcts2
                model1_color = 'b'
            else:
                black_mcts, white_mcts = mcts2, mcts1
                model1_color = 'w'

            # Play game
            game_result = self._play_single_game(
                black_mcts=black_mcts,
                white_mcts=white_mcts
            )

            # Determine winner from model1's perspective
            winner = game_result['winner']
            if winner is None:
                draws += 1
                outcome = 'draw'
            elif (winner == 'black' and model1_color == 'b') or \
                 (winner == 'white' and model1_color == 'w'):
                model1_wins += 1
                outcome = 'win'
            else:
                model2_wins += 1
                outcome = 'loss'

            games.append({
                'game_idx': game_idx,
                'model1_color': model1_color,
                'winner': winner,
                'outcome': outcome,
                'num_moves': game_result['num_moves']
            })

            if verbose and (game_idx + 1) % 10 == 0:
                current_win_rate = model1_wins / (game_idx + 1)
                logger.info(
                    f"Game {game_idx + 1}/{num_games}: "
                    f"{model1_wins}W-{model2_wins}L-{draws}D "
                    f"({current_win_rate:.1%})"
                )

        total = model1_wins + model2_wins + draws
        win_rate = (model1_wins + 0.5 * draws) / total if total > 0 else 0.5

        return EvaluationResult(
            model1_wins=model1_wins,
            model2_wins=model2_wins,
            draws=draws,
            total_games=total,
            win_rate=win_rate,
            is_improved=win_rate >= self.win_threshold,
            games=games
        )

    def _play_single_game(
        self,
        black_mcts: MCTS,
        white_mcts: MCTS
    ) -> Dict[str, Any]:
        """Play a single game between two MCTS engines.

        Parameters
        ----------
        black_mcts : MCTS
            MCTS engine for Black
        white_mcts : MCTS
            MCTS engine for White

        Returns
        -------
        Dict[str, Any]
            Game result with winner and move count
        """
        board = GoBoard(size=self.board_size, komi=self.komi)
        move_count = 0
        current_color = 'b'
        consecutive_passes = 0

        while not board.is_game_over() and move_count < self.max_moves:
            # Select MCTS engine
            mcts = black_mcts if current_color == 'b' else white_mcts

            # Get move probabilities
            move_probs, _ = mcts.search(board, current_color)

            # Greedy move selection
            move_idx = np.argmax(move_probs)

            if move_idx == self.board_size * self.board_size:
                # Pass
                board.pass_move(current_color)
                consecutive_passes += 1
                if consecutive_passes >= 2:
                    break
            else:
                row = move_idx // self.board_size
                col = move_idx % self.board_size

                success = board.play_move(row, col, current_color)
                if not success:
                    # Illegal move - try legal moves or pass
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

        # Determine winner
        winner = board.get_winner()

        return {
            'winner': winner,
            'num_moves': move_count
        }

    def quick_verify(
        self,
        new_model: torch.nn.Module,
        old_model: torch.nn.Module,
        num_games: int = 20
    ) -> Tuple[bool, float]:
        """Quick verification with fewer games.

        Useful for early stopping if clearly better/worse.

        Parameters
        ----------
        new_model : torch.nn.Module
            New model
        old_model : torch.nn.Module
            Old model
        num_games : int
            Number of quick evaluation games

        Returns
        -------
        Tuple[bool, float]
            (is_likely_improved, win_rate)
        """
        is_improved, win_rate, _ = self.verify_improvement(
            new_model=new_model,
            old_model=old_model,
            verbose=False
        )

        # Use lower threshold for quick check
        quick_threshold = self.win_threshold - 0.05
        return win_rate >= quick_threshold, win_rate


class EloRating:
    """ELO rating system for model evaluation.

    Parameters
    ----------
    initial_rating : float
        Initial ELO rating for new models (default 1000)
    k_factor : float
        K-factor for rating updates (default 32)
    """

    def __init__(self, initial_rating: float = 1000, k_factor: float = 32):
        self.initial_rating = initial_rating
        self.k_factor = k_factor
        self._ratings: Dict[str, float] = {}

    def get_rating(self, model_id: str) -> float:
        """Get ELO rating for a model.

        Parameters
        ----------
        model_id : str
            Model identifier

        Returns
        -------
        float
            ELO rating (initial_rating if not yet rated)
        """
        return self._ratings.get(model_id, self.initial_rating)

    def set_rating(self, model_id: str, rating: float) -> None:
        """Set ELO rating for a model.

        Parameters
        ----------
        model_id : str
            Model identifier
        rating : float
            New ELO rating
        """
        self._ratings[model_id] = rating

    def expected_score(self, rating_a: float, rating_b: float) -> float:
        """Calculate expected score for player A vs player B.

        Parameters
        ----------
        rating_a : float
            Rating of player A
        rating_b : float
            Rating of player B

        Returns
        -------
        float
            Expected score (0-1)
        """
        return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400))

    def update_ratings(
        self,
        model_a: str,
        model_b: str,
        score_a: float
    ) -> Tuple[float, float]:
        """Update ratings after a match.

        Parameters
        ----------
        model_a : str
            First model ID
        model_b : str
            Second model ID
        score_a : float
            Score for model A (1=win, 0.5=draw, 0=loss)

        Returns
        -------
        Tuple[float, float]
            (new_rating_a, new_rating_b)
        """
        rating_a = self.get_rating(model_a)
        rating_b = self.get_rating(model_b)

        expected_a = self.expected_score(rating_a, rating_b)
        expected_b = 1.0 - expected_a

        score_b = 1.0 - score_a

        new_rating_a = rating_a + self.k_factor * (score_a - expected_a)
        new_rating_b = rating_b + self.k_factor * (score_b - expected_b)

        self._ratings[model_a] = new_rating_a
        self._ratings[model_b] = new_rating_b

        return new_rating_a, new_rating_b

    def update_from_result(
        self,
        model_a: str,
        model_b: str,
        result: EvaluationResult
    ) -> Tuple[float, float]:
        """Update ratings from evaluation result.

        Parameters
        ----------
        model_a : str
            Model 1 ID
        model_b : str
            Model 2 ID
        result : EvaluationResult
            Evaluation result

        Returns
        -------
        Tuple[float, float]
            (new_rating_a, new_rating_b)
        """
        score_a = result.model1_score / result.total_games
        return self.update_ratings(model_a, model_b, score_a)

    def get_all_ratings(self) -> Dict[str, float]:
        """Get all stored ratings.

        Returns
        -------
        Dict[str, float]
            Model ID -> ELO rating
        """
        return dict(self._ratings)


def compute_elo_from_win_rate(win_rate: float, base_elo: float = 1000) -> float:
    """Compute ELO rating from win rate against baseline.

    Parameters
    ----------
    win_rate : float
        Win rate against baseline (0-1)
    base_elo : float
        Baseline ELO rating

    Returns
    -------
    float
        Estimated ELO rating
    """
    if win_rate <= 0:
        return base_elo - 400
    if win_rate >= 1:
        return base_elo + 400

    # Inverse of expected score formula
    elo_diff = 400 * np.log10(win_rate / (1 - win_rate))
    return base_elo + elo_diff

"""Position complexity estimation for time allocation.

This module estimates the complexity of Go positions to determine
how much thinking time should be allocated.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any
import logging

logger = logging.getLogger(__name__)


@dataclass
class PositionComplexity:
    """Estimated complexity of a Go position.

    Attributes
    ----------
    overall_score : float
        Overall complexity (0.0 trivial to 1.0 very complex)
    tactical_tension : float
        Groups in danger, captures possible
    strategic_ambiguity : float
        Unclear best direction
    branching_factor : float
        Number of reasonable moves
    liberty_tension : float
        Liberties-related urgency
    ko_complexity : float
        Ko-related complexity
    game_phase_factor : float
        Opening/middle/endgame adjustment
    recommended_time_multiplier : float
        Suggested time allocation multiplier (0.5x to 2.0x)
    """

    overall_score: float
    tactical_tension: float
    strategic_ambiguity: float
    branching_factor: float
    liberty_tension: float
    ko_complexity: float
    game_phase_factor: float
    recommended_time_multiplier: float


class ComplexityEstimator:
    """Estimate position complexity for time allocation.

    Parameters
    ----------
    board_size : int
        Size of the Go board (default 19)
    weights : Optional[Dict[str, float]]
        Weights for complexity components
    """

    # Default weights for complexity components
    DEFAULT_WEIGHTS = {
        'tactical_tension': 0.25,
        'strategic_ambiguity': 0.20,
        'branching_factor': 0.15,
        'liberty_tension': 0.20,
        'ko_complexity': 0.10,
        'game_phase': 0.10
    }

    # Baseline complexity by game phase
    PHASE_BASELINES = {
        'opening': 0.3,
        'midgame': 0.6,
        'endgame': 0.4
    }

    def __init__(
        self,
        board_size: int = 19,
        weights: Optional[Dict[str, float]] = None
    ):
        self.board_size = board_size
        self.weights = weights or self.DEFAULT_WEIGHTS.copy()

    def estimate(
        self,
        board: 'GoBoard',
        color: str,
        policy_probs: Optional[np.ndarray] = None,
        value: Optional[float] = None
    ) -> PositionComplexity:
        """Estimate position complexity.

        Parameters
        ----------
        board : GoBoard
            Current board state
        color : str
            Color to move
        policy_probs : Optional[np.ndarray]
            Policy output from neural network (if available)
        value : Optional[float]
            Value output from neural network (if available)

        Returns
        -------
        PositionComplexity
            Complexity assessment
        """
        # Calculate individual components
        tactical = self._estimate_tactical_tension(board, color)
        strategic = self._estimate_strategic_ambiguity(policy_probs)
        branching = self._estimate_branching_factor(board, color, policy_probs)
        liberty = self._estimate_liberty_tension(board)
        ko = self._estimate_ko_complexity(board)
        phase = self._get_game_phase_factor(board)

        # Weighted combination
        overall = (
            self.weights['tactical_tension'] * tactical +
            self.weights['strategic_ambiguity'] * strategic +
            self.weights['branching_factor'] * branching +
            self.weights['liberty_tension'] * liberty +
            self.weights['ko_complexity'] * ko +
            self.weights['game_phase'] * phase
        )

        # Clamp to [0, 1]
        overall = max(0.0, min(1.0, overall))

        # Calculate time multiplier (0.5x to 2.0x)
        time_multiplier = 0.5 + 1.5 * overall

        return PositionComplexity(
            overall_score=overall,
            tactical_tension=tactical,
            strategic_ambiguity=strategic,
            branching_factor=branching,
            liberty_tension=liberty,
            ko_complexity=ko,
            game_phase_factor=phase,
            recommended_time_multiplier=time_multiplier
        )

    def _estimate_tactical_tension(
        self,
        board: 'GoBoard',
        color: str
    ) -> float:
        """Estimate tactical urgency.

        High tension when:
        - Groups are in atari
        - Large captures are possible
        - Multiple tactical threats exist

        Parameters
        ----------
        board : GoBoard
            Board state
        color : str
            Color to move

        Returns
        -------
        float
            Tactical tension (0.0-1.0)
        """
        tension = 0.0
        opponent = 'w' if color == 'b' else 'b'
        visited = set()

        try:
            for r in range(board.size):
                for c in range(board.size):
                    if (r, c) in visited:
                        continue

                    stone = board.get(r, c)
                    if stone is None:
                        continue

                    # Get group info
                    try:
                        group = board.get_group(r, c)
                        visited.update(group)
                        liberties = board.count_liberties(group)
                    except Exception:
                        continue

                    group_size = len(group)

                    if liberties == 1:
                        # Group in atari
                        size_factor = min(group_size / 10, 1.0)
                        if stone == color[0]:
                            # Our group in danger - high tension
                            tension += 0.3 * size_factor
                        else:
                            # Can capture opponent
                            tension += 0.2 * size_factor

                    elif liberties == 2:
                        # Group vulnerable
                        tension += 0.05 * min(group_size / 10, 1.0)

        except Exception as e:
            logger.debug(f"Error estimating tactical tension: {e}")
            tension = 0.3  # Default moderate tension

        return min(tension, 1.0)

    def _estimate_strategic_ambiguity(
        self,
        policy_probs: Optional[np.ndarray]
    ) -> float:
        """Estimate strategic ambiguity from policy entropy.

        Higher entropy = more ambiguous position = needs more thinking.

        Parameters
        ----------
        policy_probs : Optional[np.ndarray]
            Policy distribution from neural network

        Returns
        -------
        float
            Strategic ambiguity (0.0-1.0)
        """
        if policy_probs is None:
            return 0.5  # Default medium uncertainty

        try:
            probs = policy_probs.flatten()
            probs = probs[probs > 1e-10]  # Filter near-zero

            if len(probs) == 0:
                return 0.5

            # Calculate entropy
            entropy = -np.sum(probs * np.log(probs + 1e-10))

            # Normalize (max entropy for 361 moves is ~5.89)
            max_entropy = np.log(self.board_size * self.board_size)
            normalized = entropy / max_entropy

            return min(max(normalized, 0.0), 1.0)

        except Exception:
            return 0.5

    def _estimate_branching_factor(
        self,
        board: 'GoBoard',
        color: str,
        policy_probs: Optional[np.ndarray]
    ) -> float:
        """Estimate effective branching factor.

        More reasonable moves = higher branching = needs more thinking.

        Parameters
        ----------
        board : GoBoard
            Board state
        color : str
            Color to move
        policy_probs : Optional[np.ndarray]
            Policy distribution

        Returns
        -------
        float
            Branching factor score (0.0-1.0)
        """
        try:
            legal_moves = board.get_legal_moves(color)
            num_legal = len(legal_moves)

            if num_legal == 0:
                return 0.0

            if policy_probs is not None:
                # Count moves with significant probability
                significant_moves = np.sum(policy_probs > 0.01)
                effective_branching = significant_moves / max(num_legal, 1)
            else:
                # Estimate from board fullness
                total_positions = self.board_size * self.board_size
                effective_branching = num_legal / total_positions

            return min(effective_branching, 1.0)

        except Exception:
            return 0.3

    def _estimate_liberty_tension(self, board: 'GoBoard') -> float:
        """Estimate liberty-based tension on the board.

        More groups with low liberties = higher tension.

        Parameters
        ----------
        board : GoBoard
            Board state

        Returns
        -------
        float
            Liberty tension (0.0-1.0)
        """
        try:
            total_stones = 0
            low_liberty_stones = 0
            visited = set()

            for r in range(board.size):
                for c in range(board.size):
                    if (r, c) in visited:
                        continue

                    stone = board.get(r, c)
                    if stone is None:
                        continue

                    try:
                        group = board.get_group(r, c)
                        visited.update(group)
                        liberties = board.count_liberties(group)

                        total_stones += len(group)
                        if liberties <= 3:
                            low_liberty_stones += len(group)
                    except Exception:
                        total_stones += 1

            if total_stones == 0:
                return 0.0

            return low_liberty_stones / total_stones

        except Exception:
            return 0.2

    def _estimate_ko_complexity(self, board: 'GoBoard') -> float:
        """Estimate ko-related complexity.

        Parameters
        ----------
        board : GoBoard
            Board state

        Returns
        -------
        float
            Ko complexity (0.0-1.0)
        """
        try:
            if hasattr(board, 'ko_point') and board.ko_point is not None:
                return 0.8  # Active ko fight
            return 0.0
        except Exception:
            return 0.0

    def _get_game_phase_factor(self, board: 'GoBoard') -> float:
        """Get complexity baseline for current game phase.

        Parameters
        ----------
        board : GoBoard
            Board state

        Returns
        -------
        float
            Phase-based complexity baseline
        """
        try:
            # Estimate game phase from move count and board fullness
            move_count = len(board.move_history) if hasattr(board, 'move_history') else 0

            stone_count = 0
            for r in range(board.size):
                for c in range(board.size):
                    if board.get(r, c) is not None:
                        stone_count += 1

            # Determine phase
            if move_count < 50 or stone_count < 40:
                return self.PHASE_BASELINES['opening']
            elif move_count > 200 or stone_count > 150:
                return self.PHASE_BASELINES['endgame']
            else:
                return self.PHASE_BASELINES['midgame']

        except Exception:
            return self.PHASE_BASELINES['midgame']

    def get_game_phase(self, board: 'GoBoard') -> str:
        """Determine the current game phase.

        Parameters
        ----------
        board : GoBoard
            Board state

        Returns
        -------
        str
            Game phase: "opening", "midgame", or "endgame"
        """
        try:
            move_count = len(board.move_history) if hasattr(board, 'move_history') else 0

            stone_count = 0
            for r in range(board.size):
                for c in range(board.size):
                    if board.get(r, c) is not None:
                        stone_count += 1

            if move_count < 50 or stone_count < 40:
                return "opening"
            elif move_count > 200 or stone_count > 150:
                return "endgame"
            else:
                return "midgame"

        except Exception:
            return "midgame"

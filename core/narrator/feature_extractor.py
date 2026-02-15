"""Feature extraction for move explanations.

This module extracts explainable features from MCTS search trees
and neural network outputs.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any, Union
import logging

logger = logging.getLogger(__name__)


@dataclass
class ExplainableFeatures:
    """Features extracted for explanation generation.

    Attributes
    ----------
    chosen_move : Optional[Tuple[int, int]]
        The chosen move (row, col) or None for pass
    chosen_move_prior : float
        Neural network prior probability
    chosen_move_visits : int
        MCTS visit count
    chosen_move_value : float
        Q-value estimate
    confidence : float
        Confidence in the move (0.0-1.0)
    top_alternatives : List[Dict[str, Any]]
        Top alternative moves
    game_phase : str
        Current game phase
    move_number : int
        Current move number
    black_territory_estimate : float
        Estimated black territory
    white_territory_estimate : float
        Estimated white territory
    current_advantage : str
        Who is leading
    groups_in_atari : List[Tuple[int, int]]
        Groups with 1 liberty
    potential_captures : List[Tuple[int, int]]
        Possible capture moves
    ko_threat : Optional[Tuple[int, int]]
        Ko threat location
    joseki_match : Optional[str]
        Matched joseki pattern name
    tesuji_type : Optional[str]
        Detected tesuji type
    shape_quality : str
        Shape assessment
    """

    chosen_move: Optional[Tuple[int, int]]
    chosen_move_prior: float = 0.0
    chosen_move_visits: int = 0
    chosen_move_value: float = 0.0
    confidence: float = 0.0
    top_alternatives: List[Dict[str, Any]] = field(default_factory=list)
    game_phase: str = "midgame"
    move_number: int = 0
    black_territory_estimate: float = 0.0
    white_territory_estimate: float = 0.0
    current_advantage: str = "even"
    groups_in_atari: List[Tuple[int, int]] = field(default_factory=list)
    potential_captures: List[Tuple[int, int]] = field(default_factory=list)
    ko_threat: Optional[Tuple[int, int]] = None
    joseki_match: Optional[str] = None
    tesuji_type: Optional[str] = None
    shape_quality: str = "neutral"


class FeatureExtractor:
    """Extract explainable features from MCTS search.

    Parameters
    ----------
    board_size : int
        Size of the Go board (default 19)
    """

    def __init__(self, board_size: int = 19):
        self.board_size = board_size

    def extract(
        self,
        mcts_root: 'MCTSNode',
        board: 'GoBoard',
        color: str,
        move_probs: np.ndarray
    ) -> ExplainableFeatures:
        """Extract all explainable features.

        Parameters
        ----------
        mcts_root : MCTSNode
            Root of MCTS tree after search
        board : GoBoard
            Current board state
        color : str
            Color to move
        move_probs : np.ndarray
            Move probabilities from MCTS

        Returns
        -------
        ExplainableFeatures
            Extracted features
        """
        # Get top moves
        top_moves = self._extract_top_moves(mcts_root, move_probs, top_k=5)

        chosen = top_moves[0] if top_moves else None

        # Get move number
        try:
            move_number = len(board.move_history) if hasattr(board, 'move_history') else 0
        except Exception:
            move_number = 0

        # Determine game phase
        game_phase = self._detect_game_phase(board, move_number)

        # Extract tactical features
        tactical = self._extract_tactical_features(board, color)

        # Estimate territories (simplified)
        territory = self._estimate_territory(mcts_root, board)

        return ExplainableFeatures(
            chosen_move=chosen['move'] if chosen else None,
            chosen_move_prior=chosen['prior'] if chosen else 0.0,
            chosen_move_visits=chosen['visits'] if chosen else 0,
            chosen_move_value=chosen['value'] if chosen else 0.0,
            confidence=chosen['visits'] / max(mcts_root.visit_count, 1) if chosen else 0.0,
            top_alternatives=top_moves[1:] if len(top_moves) > 1 else [],
            game_phase=game_phase,
            move_number=move_number,
            black_territory_estimate=territory.get('black', 0.0),
            white_territory_estimate=territory.get('white', 0.0),
            current_advantage=territory.get('advantage', 'even'),
            groups_in_atari=tactical.get('atari_groups', []),
            potential_captures=tactical.get('captures', []),
            ko_threat=tactical.get('ko_threat'),
            joseki_match=None,  # Would need joseki database
            tesuji_type=None,   # Would need pattern matching
            shape_quality='neutral'
        )

    def _extract_top_moves(
        self,
        mcts_root: 'MCTSNode',
        move_probs: np.ndarray,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """Extract top moves with statistics.

        Parameters
        ----------
        mcts_root : MCTSNode
            MCTS root node
        move_probs : np.ndarray
            Move probabilities
        top_k : int
            Number of top moves to return

        Returns
        -------
        List[Dict[str, Any]]
            Top moves with stats
        """
        moves = []

        try:
            for move, child in mcts_root.children.items():
                if move == 'pass':
                    move_idx = self.board_size * self.board_size
                    move_coord = None
                else:
                    move_idx = move[0] * self.board_size + move[1]
                    move_coord = move

                prior = move_probs[move_idx] if move_idx < len(move_probs) else 0.0

                moves.append({
                    'move': move_coord,
                    'prior': float(prior),
                    'visits': child.visit_count,
                    'value': child.q_value
                })

        except Exception as e:
            logger.debug(f"Error extracting top moves: {e}")

        # Sort by visit count
        moves.sort(key=lambda x: x['visits'], reverse=True)
        return moves[:top_k]

    def _detect_game_phase(
        self,
        board: 'GoBoard',
        move_number: int
    ) -> str:
        """Detect current game phase.

        Parameters
        ----------
        board : GoBoard
            Board state
        move_number : int
            Current move number

        Returns
        -------
        str
            Game phase: "opening", "midgame", or "endgame"
        """
        try:
            stone_count = sum(
                1 for r in range(board.size) for c in range(board.size)
                if board.get(r, c) is not None
            )
        except Exception:
            stone_count = move_number

        if move_number < 50 or stone_count < 40:
            return "opening"
        elif move_number > 200 or stone_count > 150:
            return "endgame"
        else:
            return "midgame"

    def _extract_tactical_features(
        self,
        board: 'GoBoard',
        color: str
    ) -> Dict[str, Any]:
        """Extract tactical features.

        Parameters
        ----------
        board : GoBoard
            Board state
        color : str
            Color to move

        Returns
        -------
        Dict[str, Any]
            Tactical features
        """
        result = {
            'atari_groups': [],
            'captures': [],
            'ko_threat': None,
            'ladders': []
        }

        try:
            # Check for ko
            if hasattr(board, 'ko_point') and board.ko_point is not None:
                result['ko_threat'] = board.ko_point

            # Find groups in atari
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

                        if liberties == 1:
                            result['atari_groups'].append((r, c))
                            # The liberty point is a potential capture
                            liberty_point = self._find_liberty(board, group)
                            if liberty_point:
                                result['captures'].append(liberty_point)

                    except Exception:
                        continue

        except Exception as e:
            logger.debug(f"Error extracting tactical features: {e}")

        return result

    def _find_liberty(
        self,
        board: 'GoBoard',
        group: set
    ) -> Optional[Tuple[int, int]]:
        """Find a liberty of a group.

        Parameters
        ----------
        board : GoBoard
            Board state
        group : set
            Set of group stones

        Returns
        -------
        Optional[Tuple[int, int]]
            A liberty point, or None
        """
        try:
            for r, c in group:
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < board.size and 0 <= nc < board.size:
                        if board.get(nr, nc) is None:
                            return (nr, nc)
        except Exception:
            pass
        return None

    def _estimate_territory(
        self,
        mcts_root: 'MCTSNode',
        board: 'GoBoard'
    ) -> Dict[str, Any]:
        """Estimate territory control.

        Parameters
        ----------
        mcts_root : MCTSNode
            MCTS root
        board : GoBoard
            Board state

        Returns
        -------
        Dict[str, Any]
            Territory estimates
        """
        # Simplified territory estimation
        # In a real implementation, this would use more sophisticated methods

        try:
            black_stones = 0
            white_stones = 0

            for r in range(board.size):
                for c in range(board.size):
                    stone = board.get(r, c)
                    if stone == 'b':
                        black_stones += 1
                    elif stone == 'w':
                        white_stones += 1

            # Use value estimate if available
            if hasattr(mcts_root, 'total_value') and mcts_root.visit_count > 0:
                value = mcts_root.total_value / mcts_root.visit_count
                # Value is from black's perspective typically
                if value > 0.1:
                    advantage = 'black_leading'
                elif value < -0.1:
                    advantage = 'white_leading'
                else:
                    advantage = 'even'
            else:
                # Rough estimate from stone count
                diff = black_stones - white_stones
                if diff > 5:
                    advantage = 'black_leading'
                elif diff < -5:
                    advantage = 'white_leading'
                else:
                    advantage = 'even'

            # Very rough territory estimate
            total = board.size * board.size
            black_territory = black_stones * 2  # Rough estimate
            white_territory = white_stones * 2

            return {
                'black': min(black_territory, total / 2),
                'white': min(white_territory, total / 2),
                'advantage': advantage
            }

        except Exception:
            return {
                'black': 0.0,
                'white': 0.0,
                'advantage': 'even'
            }

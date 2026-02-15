"""Position analysis for the narrator system.

This module analyzes Go positions to provide context for explanations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any
import logging

logger = logging.getLogger(__name__)


@dataclass
class GroupInfo:
    """Information about a group of stones.

    Attributes
    ----------
    color : str
        Stone color ('b' or 'w')
    position : Tuple[int, int]
        A representative position
    size : int
        Number of stones
    liberties : int
        Number of liberties
    is_alive : bool
        Whether group is considered alive
    """

    color: str
    position: Tuple[int, int]
    size: int
    liberties: int
    is_alive: bool = True


@dataclass
class PositionContext:
    """Context information about the current position.

    Attributes
    ----------
    ko_active : bool
        Whether a ko is active
    ko_location : Optional[Tuple[int, int]]
        Location of ko
    recent_captures : Dict[str, int]
        Captured stones by color
    groups_at_risk : List[GroupInfo]
        Groups with few liberties
    weak_groups : Dict[str, List[GroupInfo]]
        Weak groups by color
    territory_control : Dict[str, float]
        Territory estimates
    critical_areas : List[Tuple[int, int]]
        Critical points
    game_phase : str
        Current game phase
    """

    ko_active: bool = False
    ko_location: Optional[Tuple[int, int]] = None
    recent_captures: Dict[str, int] = field(default_factory=dict)
    groups_at_risk: List[GroupInfo] = field(default_factory=list)
    weak_groups: Dict[str, List[GroupInfo]] = field(default_factory=dict)
    territory_control: Dict[str, float] = field(default_factory=dict)
    critical_areas: List[Tuple[int, int]] = field(default_factory=list)
    game_phase: str = "midgame"


class PositionAnalyzer:
    """Analyze Go positions for context generation.

    Parameters
    ----------
    board_size : int
        Size of the Go board (default 19)
    """

    def __init__(self, board_size: int = 19):
        self.board_size = board_size

    def analyze(
        self,
        board: 'GoBoard',
        color: str
    ) -> PositionContext:
        """Analyze board position for context.

        Parameters
        ----------
        board : GoBoard
            Current board state
        color : str
            Color to move

        Returns
        -------
        PositionContext
            Analyzed context
        """
        context = PositionContext()

        try:
            # Check for ko
            if hasattr(board, 'ko_point') and board.ko_point is not None:
                context.ko_active = True
                context.ko_location = board.ko_point

            # Get captures
            if hasattr(board, 'prisoners'):
                context.recent_captures = dict(board.prisoners)

            # Find weak groups
            weak_groups = self._find_weak_groups(board, color)
            context.groups_at_risk = [g for g in weak_groups if g.liberties <= 2]
            context.weak_groups = {
                'b': [g for g in weak_groups if g.color == 'b'],
                'w': [g for g in weak_groups if g.color == 'w']
            }

            # Estimate territory control
            context.territory_control = self._estimate_territory_control(board)

            # Find critical areas
            context.critical_areas = self._find_critical_points(board)

            # Determine game phase
            context.game_phase = self._determine_game_phase(board)

        except Exception as e:
            logger.debug(f"Error analyzing position: {e}")

        return context

    def _find_weak_groups(
        self,
        board: 'GoBoard',
        color: str
    ) -> List[GroupInfo]:
        """Find groups with few liberties.

        Parameters
        ----------
        board : GoBoard
            Board state
        color : str
            Color to move

        Returns
        -------
        List[GroupInfo]
            Weak groups
        """
        weak_groups = []
        visited = set()

        try:
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

                        if liberties <= 3:
                            weak_groups.append(GroupInfo(
                                color=stone,
                                position=(r, c),
                                size=len(group),
                                liberties=liberties,
                                is_alive=(liberties > 1)
                            ))

                    except Exception:
                        continue

        except Exception as e:
            logger.debug(f"Error finding weak groups: {e}")

        return weak_groups

    def _estimate_territory_control(
        self,
        board: 'GoBoard'
    ) -> Dict[str, float]:
        """Estimate territory control.

        Parameters
        ----------
        board : GoBoard
            Board state

        Returns
        -------
        Dict[str, float]
            Territory control: {black, white, neutral}
        """
        try:
            total = board.size * board.size
            black_influence = 0
            white_influence = 0

            for r in range(board.size):
                for c in range(board.size):
                    stone = board.get(r, c)
                    if stone == 'b':
                        black_influence += 3  # Stone + nearby influence
                    elif stone == 'w':
                        white_influence += 3

            black_control = min(black_influence / total, 0.5)
            white_control = min(white_influence / total, 0.5)
            neutral = 1.0 - black_control - white_control

            return {
                'black': black_control,
                'white': white_control,
                'neutral': neutral
            }

        except Exception:
            return {'black': 0.33, 'white': 0.33, 'neutral': 0.34}

    def _find_critical_points(
        self,
        board: 'GoBoard'
    ) -> List[Tuple[int, int]]:
        """Find critical points (big points).

        Parameters
        ----------
        board : GoBoard
            Board state

        Returns
        -------
        List[Tuple[int, int]]
            Critical points
        """
        critical = []

        try:
            size = board.size

            # Star points are often big points
            star_points = self._get_star_points(size)
            for point in star_points:
                if board.get(point[0], point[1]) is None:
                    critical.append(point)

            # Edges of influence zones
            # This is simplified - real implementation would be more sophisticated

        except Exception as e:
            logger.debug(f"Error finding critical points: {e}")

        return critical[:5]  # Return top 5

    def _get_star_points(self, size: int) -> List[Tuple[int, int]]:
        """Get star point locations.

        Parameters
        ----------
        size : int
            Board size

        Returns
        -------
        List[Tuple[int, int]]
            Star point coordinates
        """
        if size == 19:
            offset = 3
            points = [
                (offset, offset), (offset, 9), (offset, 15),
                (9, offset), (9, 9), (9, 15),
                (15, offset), (15, 9), (15, 15)
            ]
        elif size == 13:
            offset = 3
            points = [
                (offset, offset), (offset, 9),
                (6, 6),
                (9, offset), (9, 9)
            ]
        else:  # 9x9
            offset = 2
            points = [
                (offset, offset), (offset, 6),
                (4, 4),
                (6, offset), (6, 6)
            ]

        return points

    def _determine_game_phase(self, board: 'GoBoard') -> str:
        """Determine the current game phase.

        Parameters
        ----------
        board : GoBoard
            Board state

        Returns
        -------
        str
            Game phase
        """
        try:
            stone_count = sum(
                1 for r in range(board.size) for c in range(board.size)
                if board.get(r, c) is not None
            )

            move_count = len(board.move_history) if hasattr(board, 'move_history') else stone_count

            if move_count < 50 or stone_count < 40:
                return "opening"
            elif move_count > 200 or stone_count > 150:
                return "endgame"
            else:
                return "midgame"

        except Exception:
            return "midgame"

    def get_position_summary(
        self,
        board: 'GoBoard',
        color: str
    ) -> str:
        """Get a brief position summary.

        Parameters
        ----------
        board : GoBoard
            Board state
        color : str
            Color to move

        Returns
        -------
        str
            Position summary
        """
        context = self.analyze(board, color)

        parts = []

        # Game phase
        parts.append(f"Game phase: {context.game_phase}")

        # Ko situation
        if context.ko_active:
            parts.append("Ko is active")

        # Groups at risk
        if context.groups_at_risk:
            atari_count = sum(1 for g in context.groups_at_risk if g.liberties == 1)
            if atari_count > 0:
                parts.append(f"{atari_count} group(s) in atari")

        # Territory
        if context.territory_control:
            black = context.territory_control.get('black', 0)
            white = context.territory_control.get('white', 0)
            if black > white + 0.1:
                parts.append("Black has more influence")
            elif white > black + 0.1:
                parts.append("White has more influence")
            else:
                parts.append("Balanced position")

        return ". ".join(parts)

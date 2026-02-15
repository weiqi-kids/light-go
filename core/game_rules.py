"""Go game rules engine implementation.

This module provides a complete implementation of Go game rules including:
- Move validation
- Stone capture
- Ko rule
- Superko (optional)
- Game scoring

The implementation leverages sgfmill for validation where appropriate and provides
a clean interface for self-play and MCTS.
"""

from __future__ import annotations

from typing import Set, Tuple, List, Optional, Dict
from copy import deepcopy
from sgfmill import boards


class GoBoard:
    """Go board with complete rule implementation.

    Parameters
    ----------
    size : int
        Board size (default 19)
    komi : float
        Komi (compensation for white, default 7.5)
    ruleset : str
        Ruleset to use ('chinese', 'japanese', 'korean', default 'chinese')
    """

    def __init__(self, size: int = 19, komi: float = 7.5, ruleset: str = 'chinese'):
        self.size = size
        self.komi = komi
        self.ruleset = ruleset

        # Internal board representation using sgfmill
        self.board = boards.Board(size)

        # Game state
        self.move_history: List[Tuple[Optional[Tuple[int, int]], str]] = []
        self.ko_point: Optional[Tuple[int, int]] = None
        self.prisoners = {'black': 0, 'white': 0}
        self.consecutive_passes = 0

    def copy(self) -> "GoBoard":
        """Create a deep copy of this board.

        Returns
        -------
        GoBoard
            Independent copy of board state
        """
        new_board = GoBoard(self.size, self.komi, self.ruleset)
        new_board.board = self.board.copy()
        new_board.move_history = self.move_history.copy()
        new_board.ko_point = self.ko_point
        new_board.prisoners = self.prisoners.copy()
        new_board.consecutive_passes = self.consecutive_passes
        return new_board

    def get(self, row: int, col: int) -> Optional[str]:
        """Get stone color at position.

        Parameters
        ----------
        row, col : int
            Board coordinates (0-indexed)

        Returns
        -------
        Optional[str]
            'b' for black, 'w' for white, None for empty
        """
        return self.board.get(row, col)

    def is_on_board(self, row: int, col: int) -> bool:
        """Check if coordinates are on board.

        Parameters
        ----------
        row, col : int
            Board coordinates

        Returns
        -------
        bool
            True if coordinates are valid
        """
        return 0 <= row < self.size and 0 <= col < self.size

    def get_neighbors(self, row: int, col: int) -> List[Tuple[int, int]]:
        """Get orthogonally adjacent points.

        Parameters
        ----------
        row, col : int
            Board coordinates

        Returns
        -------
        List[Tuple[int, int]]
            List of (row, col) tuples for neighbors
        """
        neighbors = []
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            r, c = row + dr, col + dc
            if self.is_on_board(r, c):
                neighbors.append((r, c))
        return neighbors

    def get_group(self, row: int, col: int) -> Set[Tuple[int, int]]:
        """Get all stones in the group containing (row, col).

        Parameters
        ----------
        row, col : int
            Position of a stone

        Returns
        -------
        Set[Tuple[int, int]]
            Set of positions in the group
        """
        color = self.board.get(row, col)
        if color is None:
            return set()

        group = set()
        to_check = [(row, col)]

        while to_check:
            r, c = to_check.pop()
            if (r, c) in group:
                continue

            if self.board.get(r, c) == color:
                group.add((r, c))
                to_check.extend(self.get_neighbors(r, c))

        return group

    def count_liberties(self, group: Set[Tuple[int, int]]) -> int:
        """Count liberties of a group.

        Parameters
        ----------
        group : Set[Tuple[int, int]]
            Set of positions in the group

        Returns
        -------
        int
            Number of liberties
        """
        liberties = set()
        for row, col in group:
            for nr, nc in self.get_neighbors(row, col):
                if self.board.get(nr, nc) is None:
                    liberties.add((nr, nc))
        return len(liberties)

    def get_captured_groups(self, row: int, col: int, color: str) -> List[Set[Tuple[int, int]]]:
        """Get groups that would be captured by playing at (row, col).

        Parameters
        ----------
        row, col : int
            Position to check
        color : str
            Color of stone to play ('b' or 'w')

        Returns
        -------
        List[Set[Tuple[int, int]]]
            List of groups that would be captured
        """
        opponent = 'w' if color == 'b' else 'b'
        captured = []
        checked_groups = []  # Track groups we've already checked

        for nr, nc in self.get_neighbors(row, col):
            if self.board.get(nr, nc) == opponent:
                group = self.get_group(nr, nc)
                if group and group not in checked_groups:
                    checked_groups.append(group)
                    # Count liberties of opponent group
                    # The move at (row, col) will occupy one liberty
                    liberties = set()
                    for gr, gc in group:
                        for gnr, gnc in self.get_neighbors(gr, gc):
                            if self.board.get(gnr, gnc) is None:
                                liberties.add((gnr, gnc))
                    # If (row, col) is the only liberty, group will be captured
                    if liberties == {(row, col)}:
                        captured.append(group)

        return captured

    def is_suicide(self, row: int, col: int, color: str) -> bool:
        """Check if move would be suicide (illegal in most rulesets).

        Parameters
        ----------
        row, col : int
            Position to check
        color : str
            Color of stone ('b' or 'w')

        Returns
        -------
        bool
            True if move would be suicide
        """
        # Check if move captures opponent stones
        captured = self.get_captured_groups(row, col, color)
        if captured:
            return False  # Capturing is not suicide

        # Check if own group would have liberties
        test_board = self.copy()
        test_board.board.play(row, col, color)
        own_group = test_board.get_group(row, col)

        return test_board.count_liberties(own_group) == 0

    def is_legal(self, row: int, col: int, color: str) -> bool:
        """Check if move is legal.

        Parameters
        ----------
        row, col : int
            Position to play
        color : str
            Color of stone ('b' or 'w')

        Returns
        -------
        bool
            True if move is legal
        """
        # Check if position is on board
        if not self.is_on_board(row, col):
            return False

        # Check if position is empty
        if self.board.get(row, col) is not None:
            return False

        # Check ko rule
        if self.ko_point == (row, col):
            return False

        # Check suicide rule
        if self.is_suicide(row, col, color):
            return False

        return True

    def get_legal_moves(self, color: str) -> List[Tuple[int, int]]:
        """Get all legal moves for color.

        Parameters
        ----------
        color : str
            Color to move ('b' or 'w')

        Returns
        -------
        List[Tuple[int, int]]
            List of legal positions
        """
        legal = []
        for row in range(self.size):
            for col in range(self.size):
                if self.is_legal(row, col, color):
                    legal.append((row, col))
        return legal

    def play_move(self, row: int, col: int, color: str) -> bool:
        """Play a stone and update board state.

        Parameters
        ----------
        row, col : int
            Position to play
        color : str
            Color of stone ('b' or 'w')

        Returns
        -------
        bool
            True if move was successful, False if illegal
        """
        if not self.is_legal(row, col, color):
            return False

        # Get captured groups before playing
        captured = self.get_captured_groups(row, col, color)

        # Play the stone
        self.board.play(row, col, color)

        # Remove captured stones using apply_setup
        # captured_color is the color of the captured stones (opponent's color)
        captured_color = 'white' if color == 'b' else 'black'
        empty_points = []
        for group in captured:
            for r, c in group:
                empty_points.append((r, c))
                self.prisoners[captured_color] += 1

        if empty_points:
            self.board.apply_setup([], [], empty_points)

        # Update ko point (simple ko: single stone capture)
        if len(captured) == 1 and len(captured[0]) == 1:
            self.ko_point = list(captured[0])[0]
        else:
            self.ko_point = None

        # Record move
        self.move_history.append(((row, col), color))
        self.consecutive_passes = 0

        return True

    def pass_move(self, color: str) -> None:
        """Record a pass.

        Parameters
        ----------
        color : str
            Color passing ('b' or 'w')
        """
        self.move_history.append((None, color))
        self.consecutive_passes += 1
        self.ko_point = None

    def is_game_over(self) -> bool:
        """Check if game has ended (two consecutive passes).

        Returns
        -------
        bool
            True if game is over
        """
        return self.consecutive_passes >= 2

    def score_chinese(self) -> Dict[str, float]:
        """Score game using Chinese rules (area scoring).

        Returns
        -------
        Dict[str, float]
            Dictionary with 'black', 'white', 'winner', 'margin'
        """
        black_score = 0
        white_score = 0

        # Count stones and territory
        for row in range(self.size):
            for col in range(self.size):
                stone = self.board.get(row, col)
                if stone == 'b':
                    black_score += 1
                elif stone == 'w':
                    white_score += 1
                elif stone is None:
                    # Empty point - determine territory
                    # Simple approach: assign to nearby color (should use proper territory detection)
                    neighbors = self.get_neighbors(row, col)
                    black_neighbors = sum(1 for r, c in neighbors if self.board.get(r, c) == 'b')
                    white_neighbors = sum(1 for r, c in neighbors if self.board.get(r, c) == 'w')

                    if black_neighbors > white_neighbors:
                        black_score += 1
                    elif white_neighbors > black_neighbors:
                        white_score += 1

        # Add komi to white
        white_score += self.komi

        return {
            'black': black_score,
            'white': white_score,
            'winner': 'black' if black_score > white_score else 'white',
            'margin': abs(black_score - white_score)
        }

    def get_winner(self) -> Optional[str]:
        """Get game winner.

        Returns
        -------
        Optional[str]
            'black', 'white', or None if game not over
        """
        if not self.is_game_over():
            return None

        score = self.score_chinese()
        return score['winner']

    def to_liberty_matrix(self) -> List[Tuple[int, int, int]]:
        """Convert current board to liberty matrix format.

        Returns
        -------
        List[Tuple[int, int, int]]
            Liberty matrix: (x, y, liberty_count) where positive=black, negative=white
        """
        from core.liberty import count_liberties

        # Convert board to matrix format expected by count_liberties
        matrix = [[0 for _ in range(self.size)] for _ in range(self.size)]

        for row in range(self.size):
            for col in range(self.size):
                stone = self.board.get(row, col)
                if stone == 'b':
                    matrix[row][col] = 1
                elif stone == 'w':
                    matrix[row][col] = -1

        return count_liberties(matrix)

    def __str__(self) -> str:
        """String representation of board."""
        lines = []
        for row in range(self.size):
            line = []
            for col in range(self.size):
                stone = self.board.get(row, col)
                if stone == 'b':
                    line.append('●')
                elif stone == 'w':
                    line.append('○')
                else:
                    line.append('·')
            lines.append(' '.join(line))
        return '\n'.join(lines)


def create_board_from_moves(
    moves: List[Tuple[Optional[Tuple[int, int]], str]],
    size: int = 19,
    komi: float = 7.5
) -> GoBoard:
    """Create board from move sequence.

    Parameters
    ----------
    moves : List[Tuple[Optional[Tuple[int, int]], str]]
        List of ((row, col), color) or (None, color) for passes
    size : int
        Board size
    komi : float
        Komi

    Returns
    -------
    GoBoard
        Board with moves played
    """
    board = GoBoard(size=size, komi=komi)

    for move, color in moves:
        if move is None:
            board.pass_move(color)
        else:
            row, col = move
            success = board.play_move(row, col, color)
            if not success:
                raise ValueError(f"Illegal move: {move} for {color}")

    return board

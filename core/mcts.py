"""Monte Carlo Tree Search implementation for Go.

This module implements MCTS with UCB1 selection, providing a complete search
algorithm that can be used for move selection in Go games.

The algorithm follows the standard MCTS phases:
1. Selection - traverse tree using UCB1 until reaching unexpanded node
2. Expansion - add a new child node for an untried move
3. Simulation - play random moves until game ends (rollout)
4. Backpropagation - update statistics from leaf to root
"""
from __future__ import annotations

import math
import random
import copy
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Set, Dict, Any

from core.liberty import neighbors, group_and_liberties

Board = List[List[int]]
Move = Tuple[int, int]  # (x, y) coordinates


@dataclass
class MCTSNode:
    """A node in the MCTS search tree.

    Attributes
    ----------
    move : Move | None
        The move that led to this node (None for root).
    parent : MCTSNode | None
        Parent node in the tree.
    color : int
        The color that played to reach this state (1=black, -1=white).
    children : list[MCTSNode]
        Child nodes representing possible next moves.
    wins : float
        Number of wins (or partial wins) from simulations through this node.
    visits : int
        Number of times this node has been visited.
    untried_moves : list[Move]
        Legal moves that haven't been expanded yet.
    """

    move: Move | None = None
    parent: Optional["MCTSNode"] = None
    color: int = 1  # 1=black played to reach this, -1=white
    children: List["MCTSNode"] = field(default_factory=list)
    wins: float = 0.0
    visits: int = 0
    untried_moves: List[Move] = field(default_factory=list)

    def ucb1(self, exploration: float = 1.414) -> float:
        """Calculate UCB1 value for node selection.

        UCB1 = wins/visits + exploration * sqrt(ln(parent_visits) / visits)

        Parameters
        ----------
        exploration : float
            Exploration constant (default sqrt(2) ≈ 1.414).

        Returns
        -------
        float
            UCB1 value, or infinity if unvisited.
        """
        if self.visits == 0:
            return float("inf")
        parent_visits = self.parent.visits if self.parent else self.visits
        exploitation = self.wins / self.visits
        exploration_term = exploration * math.sqrt(math.log(parent_visits) / self.visits)
        return exploitation + exploration_term

    def best_child(self, exploration: float = 1.414) -> "MCTSNode":
        """Select child with highest UCB1 value."""
        return max(self.children, key=lambda c: c.ucb1(exploration))

    def is_fully_expanded(self) -> bool:
        """Check if all moves have been tried."""
        return len(self.untried_moves) == 0

    def is_terminal(self) -> bool:
        """Check if this is a terminal node (no children possible)."""
        return self.is_fully_expanded() and len(self.children) == 0


class GoGameState:
    """Represents a Go game state for MCTS simulation.

    This class handles the game rules needed for MCTS:
    - Legal move generation
    - Move execution with capture handling
    - Game termination detection
    - Simple scoring for rollout evaluation
    """

    def __init__(
        self,
        board: Board,
        current_color: int = 1,
        ko_point: Move | None = None,
        passes: int = 0,
        komi: float = 7.5,
    ):
        """Initialize game state.

        Parameters
        ----------
        board : Board
            Current board position (0=empty, 1=black, -1=white).
        current_color : int
            Color to play next (1=black, -1=white).
        ko_point : Move | None
            Position forbidden by ko rule.
        passes : int
            Consecutive passes (2 = game over).
        komi : float
            Compensation points for white.
        """
        self.board = [row[:] for row in board]  # Deep copy
        self.size = len(board)
        self.current_color = current_color
        self.ko_point = ko_point
        self.passes = passes
        self.komi = komi

    def copy(self) -> "GoGameState":
        """Create a deep copy of this state."""
        return GoGameState(
            board=self.board,
            current_color=self.current_color,
            ko_point=self.ko_point,
            passes=self.passes,
            komi=self.komi,
        )

    def get_legal_moves(self) -> List[Move]:
        """Return all legal moves for current player.

        A move is legal if:
        1. The position is empty
        2. It's not a ko recapture
        3. It doesn't result in self-capture (suicide)

        Returns
        -------
        list[Move]
            List of (x, y) coordinates for legal moves.
        """
        legal: List[Move] = []
        for y in range(self.size):
            for x in range(self.size):
                if self.board[y][x] != 0:
                    continue
                if self.ko_point == (x, y):
                    continue
                if self._is_legal_move(x, y):
                    legal.append((x, y))
        return legal

    def _is_legal_move(self, x: int, y: int) -> bool:
        """Check if placing a stone at (x, y) is legal."""
        # Temporarily place stone
        self.board[y][x] = self.current_color

        # Check if we capture any opponent stones
        captured_any = False
        opponent = -self.current_color
        for nx, ny in neighbors(x, y, self.size):
            if self.board[ny][nx] == opponent:
                _, libs = group_and_liberties(self.board, nx, ny)
                if len(libs) == 0:
                    captured_any = True
                    break

        # Check if our stone/group has liberties
        _, our_libs = group_and_liberties(self.board, x, y)
        has_liberty = len(our_libs) > 0

        # Remove temporary stone
        self.board[y][x] = 0

        # Legal if we capture something or have liberties
        return captured_any or has_liberty

    def play_move(self, move: Move | None) -> "GoGameState":
        """Execute a move and return new state.

        Parameters
        ----------
        move : Move | None
            (x, y) coordinates or None for pass.

        Returns
        -------
        GoGameState
            New state after the move.
        """
        new_state = self.copy()

        if move is None:
            # Pass
            new_state.passes += 1
            new_state.current_color = -self.current_color
            new_state.ko_point = None
            return new_state

        x, y = move
        new_state.passes = 0
        new_state.board[y][x] = self.current_color

        # Capture opponent stones
        captured_positions: List[Move] = []
        opponent = -self.current_color
        for nx, ny in neighbors(x, y, self.size):
            if new_state.board[ny][nx] == opponent:
                group, libs = group_and_liberties(new_state.board, nx, ny)
                if len(libs) == 0:
                    for gx, gy in group:
                        new_state.board[gy][gx] = 0
                        captured_positions.append((gx, gy))

        # Check for ko
        if len(captured_positions) == 1:
            # Check if this is a single stone capture that could be ko
            _, our_libs = group_and_liberties(new_state.board, x, y)
            if len(our_libs) == 1:
                new_state.ko_point = captured_positions[0]
            else:
                new_state.ko_point = None
        else:
            new_state.ko_point = None

        new_state.current_color = -self.current_color
        return new_state

    def is_terminal(self) -> bool:
        """Check if game is over (two consecutive passes)."""
        return self.passes >= 2

    def get_winner(self) -> int:
        """Determine winner using simple area scoring.

        Returns
        -------
        int
            1 if black wins, -1 if white wins, 0 for tie.
        """
        black_score = 0
        white_score = self.komi

        # Count stones and territory
        counted: Set[Move] = set()
        for y in range(self.size):
            for x in range(self.size):
                if (x, y) in counted:
                    continue
                val = self.board[y][x]
                if val == 1:
                    black_score += 1
                    counted.add((x, y))
                elif val == -1:
                    white_score += 1
                    counted.add((x, y))
                else:
                    # Empty - determine territory
                    territory, owner = self._flood_fill_territory(x, y)
                    for tx, ty in territory:
                        counted.add((tx, ty))
                    if owner == 1:
                        black_score += len(territory)
                    elif owner == -1:
                        white_score += len(territory)

        if black_score > white_score:
            return 1
        elif white_score > black_score:
            return -1
        return 0

    def _flood_fill_territory(self, x: int, y: int) -> Tuple[Set[Move], int]:
        """Find connected empty region and determine ownership.

        Returns
        -------
        tuple[set[Move], int]
            (territory positions, owner color or 0 if disputed)
        """
        territory: Set[Move] = set()
        border_colors: Set[int] = set()
        stack = [(x, y)]

        while stack:
            cx, cy = stack.pop()
            if (cx, cy) in territory:
                continue
            if self.board[cy][cx] != 0:
                border_colors.add(self.board[cy][cx])
                continue
            territory.add((cx, cy))
            for nx, ny in neighbors(cx, cy, self.size):
                if (nx, ny) not in territory:
                    stack.append((nx, ny))

        # Territory belongs to a color only if bordered by just that color
        if len(border_colors) == 1:
            return territory, border_colors.pop()
        return territory, 0


class MCTS:
    """Monte Carlo Tree Search for Go.

    This class implements the full MCTS algorithm with configurable
    parameters for search iterations and exploration.

    Example
    -------
    >>> board = [[0]*9 for _ in range(9)]
    >>> mcts = MCTS(iterations=1000)
    >>> best_move = mcts.search(board, color=1)
    >>> print(best_move)  # e.g., (4, 4)
    """

    def __init__(
        self,
        iterations: int = 1000,
        exploration: float = 1.414,
        max_rollout_depth: int = 200,
    ):
        """Initialize MCTS.

        Parameters
        ----------
        iterations : int
            Number of MCTS iterations per search.
        exploration : float
            UCB1 exploration constant.
        max_rollout_depth : int
            Maximum moves in a rollout before forcing termination.
        """
        self.iterations = iterations
        self.exploration = exploration
        self.max_rollout_depth = max_rollout_depth

    def search(
        self,
        board: Board,
        color: int = 1,
        komi: float = 7.5,
    ) -> Move | None:
        """Find the best move using MCTS.

        Parameters
        ----------
        board : Board
            Current board state.
        color : int
            Color to play (1=black, -1=white).
        komi : float
            Komi value for scoring.

        Returns
        -------
        Move | None
            Best move found, or None if no legal moves (should pass).
        """
        state = GoGameState(board, current_color=color, komi=komi)
        legal_moves = state.get_legal_moves()

        if not legal_moves:
            return None  # Should pass

        # Create root node
        root = MCTSNode(
            move=None,
            parent=None,
            color=-color,  # Parent "played" opposite color
            untried_moves=legal_moves[:],
        )

        for _ in range(self.iterations):
            node = root
            sim_state = state.copy()

            # 1. Selection - traverse tree using UCB1
            while node.is_fully_expanded() and node.children:
                node = node.best_child(self.exploration)
                if node.move is not None:
                    sim_state = sim_state.play_move(node.move)

            # 2. Expansion - add new child for untried move
            if node.untried_moves:
                move = random.choice(node.untried_moves)
                node.untried_moves.remove(move)
                sim_state = sim_state.play_move(move)

                child = MCTSNode(
                    move=move,
                    parent=node,
                    color=sim_state.current_color * -1,  # Color that just played
                    untried_moves=sim_state.get_legal_moves(),
                )
                node.children.append(child)
                node = child

            # 3. Simulation (Rollout) - play random moves
            winner = self._rollout(sim_state)

            # 4. Backpropagation - update statistics
            while node is not None:
                node.visits += 1
                # Win from perspective of the color that played to this node
                if winner == node.color:
                    node.wins += 1.0
                elif winner == 0:
                    node.wins += 0.5  # Draw
                node = node.parent

        # Return move with most visits (most robust choice)
        if not root.children:
            return legal_moves[0] if legal_moves else None

        best_child = max(root.children, key=lambda c: c.visits)
        return best_child.move

    def _rollout(self, state: GoGameState) -> int:
        """Perform random rollout from state.

        Parameters
        ----------
        state : GoGameState
            Starting state for rollout.

        Returns
        -------
        int
            Winner (1=black, -1=white, 0=tie).
        """
        sim = state.copy()
        depth = 0

        while not sim.is_terminal() and depth < self.max_rollout_depth:
            legal = sim.get_legal_moves()
            if not legal:
                # Pass
                sim = sim.play_move(None)
            else:
                # Random move selection with basic heuristics
                move = self._select_rollout_move(sim, legal)
                sim = sim.play_move(move)
            depth += 1

        return sim.get_winner()

    def _select_rollout_move(self, state: GoGameState, legal_moves: List[Move]) -> Move:
        """Select a move during rollout with light heuristics.

        This uses simple patterns to avoid obviously bad moves:
        - Prefer moves that don't fill own eyes
        - Prefer capturing moves

        Parameters
        ----------
        state : GoGameState
            Current state.
        legal_moves : list[Move]
            Available moves.

        Returns
        -------
        Move
            Selected move.
        """
        # Filter out eye-filling moves (very simple heuristic)
        good_moves = []
        for move in legal_moves:
            x, y = move
            if not self._is_eye(state.board, x, y, state.current_color):
                good_moves.append(move)

        if good_moves:
            return random.choice(good_moves)
        return random.choice(legal_moves)

    def _is_eye(self, board: Board, x: int, y: int, color: int) -> bool:
        """Check if position is an eye for the given color.

        A simple eye check: all neighbors are same color.
        """
        size = len(board)
        for nx, ny in neighbors(x, y, size):
            if board[ny][nx] != color:
                return False
        return True

    def get_move_probabilities(
        self,
        board: Board,
        color: int = 1,
        komi: float = 7.5,
    ) -> Dict[Move, float]:
        """Get visit distribution over moves (useful for training).

        Returns
        -------
        dict[Move, float]
            Mapping from move to visit probability.
        """
        state = GoGameState(board, current_color=color, komi=komi)
        legal_moves = state.get_legal_moves()

        if not legal_moves:
            return {}

        root = MCTSNode(
            move=None,
            parent=None,
            color=-color,
            untried_moves=legal_moves[:],
        )

        for _ in range(self.iterations):
            node = root
            sim_state = state.copy()

            while node.is_fully_expanded() and node.children:
                node = node.best_child(self.exploration)
                if node.move is not None:
                    sim_state = sim_state.play_move(node.move)

            if node.untried_moves:
                move = random.choice(node.untried_moves)
                node.untried_moves.remove(move)
                sim_state = sim_state.play_move(move)

                child = MCTSNode(
                    move=move,
                    parent=node,
                    color=sim_state.current_color * -1,
                    untried_moves=sim_state.get_legal_moves(),
                )
                node.children.append(child)
                node = child

            winner = self._rollout(sim_state)

            while node is not None:
                node.visits += 1
                if winner == node.color:
                    node.wins += 1.0
                elif winner == 0:
                    node.wins += 0.5
                node = node.parent

        # Calculate probabilities from visit counts
        total_visits = sum(c.visits for c in root.children)
        if total_visits == 0:
            return {m: 1.0 / len(legal_moves) for m in legal_moves}

        return {
            c.move: c.visits / total_visits
            for c in root.children
            if c.move is not None
        }


# Module-level convenience function
def mcts_search(
    board: Board,
    color: int = 1,
    iterations: int = 1000,
    komi: float = 7.5,
) -> Move | None:
    """Convenience function for MCTS search.

    Parameters
    ----------
    board : Board
        Current board state.
    color : int
        Color to play (1=black, -1=white).
    iterations : int
        Number of search iterations.
    komi : float
        Komi value.

    Returns
    -------
    Move | None
        Best move or None for pass.
    """
    mcts = MCTS(iterations=iterations)
    return mcts.search(board, color=color, komi=komi)


__all__ = [
    "MCTS",
    "MCTSNode",
    "GoGameState",
    "mcts_search",
    "Board",
    "Move",
]

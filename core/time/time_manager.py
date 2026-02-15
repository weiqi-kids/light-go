"""Time management for Go games.

This module provides the main TimeManager class that integrates
time control, complexity estimation, and allocation strategies.
"""

from __future__ import annotations

import time
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import numpy as np

from core.time.time_control import TimeControlSettings, TimeControlType, TimeState
from core.time.complexity_estimator import ComplexityEstimator, PositionComplexity
from core.time.allocation_strategy import (
    AllocationStrategy,
    ComplexityWeightedStrategy,
    create_strategy
)

logger = logging.getLogger(__name__)


@dataclass
class TimeAllocation:
    """Allocated time for a single move.

    Attributes
    ----------
    think_time_seconds : float
        Recommended thinking time
    num_simulations : int
        MCTS simulations to use
    is_time_pressure : bool
        Whether in time trouble
    complexity : PositionComplexity
        Position complexity estimate
    remaining_budget : float
        Remaining time budget
    """

    think_time_seconds: float
    num_simulations: int
    is_time_pressure: bool
    complexity: PositionComplexity
    remaining_budget: float


@dataclass
class MoveTimingRecord:
    """Record of time used for a move.

    Attributes
    ----------
    move_number : int
        Move number in game
    color : str
        Color that moved
    allocated_time : float
        Time that was allocated
    actual_time : float
        Time actually used
    allocated_simulations : int
        Simulations allocated
    complexity_score : float
        Position complexity score
    """

    move_number: int
    color: str
    allocated_time: float
    actual_time: float
    allocated_simulations: int
    complexity_score: float


class TimeManager:
    """Manage time allocation across a game.

    Parameters
    ----------
    settings : TimeControlSettings
        Time control configuration
    board_size : int
        Board size for complexity estimation (default 19)
    base_simulations : int
        Base MCTS simulation count (default 800)
    strategy : Optional[AllocationStrategy]
        Time allocation strategy (default ComplexityWeightedStrategy)
    """

    # Default simulation counts
    DEFAULT_SIMULATIONS = 800
    MIN_SIMULATIONS = 50
    MAX_SIMULATIONS = 3200

    # Time thresholds
    TIME_PRESSURE_THRESHOLD = 30.0  # Seconds remaining triggers time pressure
    CRITICAL_TIME_THRESHOLD = 10.0  # Very low time

    def __init__(
        self,
        settings: TimeControlSettings,
        board_size: int = 19,
        base_simulations: int = 800,
        strategy: Optional[AllocationStrategy] = None
    ):
        self.settings = settings
        self.board_size = board_size
        self.base_simulations = base_simulations

        # Time state per color
        self.time_states: Dict[str, TimeState] = {
            'b': TimeState(
                remaining_main_time=settings.main_time_seconds,
                remaining_byoyomi_periods=settings.byoyomi_periods
            ),
            'w': TimeState(
                remaining_main_time=settings.main_time_seconds,
                remaining_byoyomi_periods=settings.byoyomi_periods
            )
        }

        # Components
        self.complexity_estimator = ComplexityEstimator(board_size)
        self.strategy = strategy or ComplexityWeightedStrategy()

        # History
        self.timing_history: List[MoveTimingRecord] = []

        # Internal state
        self._move_start_time: Optional[float] = None
        self._current_allocation: Optional[TimeAllocation] = None
        self._move_count = 0

    def allocate_time(
        self,
        board: 'GoBoard',
        color: str,
        policy_probs: Optional[np.ndarray] = None,
        value: Optional[float] = None
    ) -> TimeAllocation:
        """Allocate time for the next move.

        Parameters
        ----------
        board : GoBoard
            Current board state
        color : str
            Color to move ('b' or 'w')
        policy_probs : Optional[np.ndarray]
            Policy output from neural network
        value : Optional[float]
            Value output from neural network

        Returns
        -------
        TimeAllocation
            Recommended time and simulation allocation
        """
        state = self.time_states[color]

        # Estimate position complexity
        complexity = self.complexity_estimator.estimate(
            board, color, policy_probs, value
        )

        # Calculate game progress
        game_progress = self._estimate_game_progress(board)

        # Calculate time pressure level
        time_pressure_level = self._calculate_time_pressure_level(state)

        # Calculate base time allocation
        if self.settings.control_type == TimeControlType.UNLIMITED:
            base_time = float('inf')
            simulations = self.base_simulations
        else:
            base_time = self._calculate_base_allocation(board, color, state)

            # Apply strategy multiplier
            multiplier = self.strategy.calculate_multiplier(
                complexity, game_progress, time_pressure_level
            )
            adjusted_time = base_time * multiplier

            # Check for time pressure
            is_time_pressure = state.is_time_pressure

            if is_time_pressure:
                # Reduce time in time pressure
                adjusted_time = min(adjusted_time, state.remaining_main_time * 0.1)
                adjusted_time = max(adjusted_time, 1.0)  # At least 1 second

            # Calculate simulations from time
            simulations = self._time_to_simulations(adjusted_time, is_time_pressure)

            base_time = adjusted_time

        is_time_pressure = state.is_time_pressure

        allocation = TimeAllocation(
            think_time_seconds=base_time,
            num_simulations=simulations,
            is_time_pressure=is_time_pressure,
            complexity=complexity,
            remaining_budget=state.remaining_main_time
        )

        self._current_allocation = allocation
        self._move_start_time = time.time()

        logger.debug(
            f"Time allocation for {color}: {base_time:.1f}s, "
            f"{simulations} sims, complexity={complexity.overall_score:.2f}"
        )

        return allocation

    def record_move_time(self, color: str, move_number: Optional[int] = None) -> None:
        """Record actual time used for a move.

        Parameters
        ----------
        color : str
            Color that moved
        move_number : Optional[int]
            Move number (default: auto-increment)
        """
        if self._move_start_time is None:
            return

        actual_time = time.time() - self._move_start_time
        state = self.time_states[color]

        # Update time state based on control type
        if self.settings.control_type != TimeControlType.UNLIMITED:
            self._update_time_state(state, color, actual_time)

        # Record history
        if move_number is None:
            self._move_count += 1
            move_number = self._move_count

        if self._current_allocation:
            record = MoveTimingRecord(
                move_number=move_number,
                color=color,
                allocated_time=self._current_allocation.think_time_seconds,
                actual_time=actual_time,
                allocated_simulations=self._current_allocation.num_simulations,
                complexity_score=self._current_allocation.complexity.overall_score
            )
            self.timing_history.append(record)

        self._move_start_time = None
        self._current_allocation = None

    def _update_time_state(
        self,
        state: TimeState,
        color: str,
        actual_time: float
    ) -> None:
        """Update time state after a move.

        Parameters
        ----------
        state : TimeState
            Time state to update
        color : str
            Color that moved
        actual_time : float
            Time used for the move
        """
        if state.in_byoyomi:
            # Byoyomi: check if exceeded period time
            if actual_time > self.settings.byoyomi_time:
                state.remaining_byoyomi_periods -= 1
            # Period resets after each move
        else:
            # Main time
            state.remaining_main_time -= actual_time

            # Add increment for Fischer
            if self.settings.control_type == TimeControlType.FISCHER:
                state.remaining_main_time += self.settings.increment_seconds

            # Check if entering byoyomi
            if state.remaining_main_time <= 0:
                if self.settings.byoyomi_periods > 0:
                    state.in_byoyomi = True
                    state.remaining_main_time = 0
                # For sudden death, time just goes negative

        # Handle Canadian overtime
        if self.settings.control_type == TimeControlType.CANADIAN:
            state.moves_in_period += 1
            if state.moves_in_period >= self.settings.canadian_stones:
                # Period completed, reset
                state.moves_in_period = 0
                if state.remaining_main_time <= 0:
                    state.remaining_main_time = self.settings.canadian_time

    def _calculate_base_allocation(
        self,
        board: 'GoBoard',
        color: str,
        state: TimeState
    ) -> float:
        """Calculate base time allocation.

        Parameters
        ----------
        board : GoBoard
            Board state
        color : str
            Color to move
        state : TimeState
            Current time state

        Returns
        -------
        float
            Base time allocation in seconds
        """
        # Estimate remaining moves
        remaining_moves = self._estimate_remaining_moves(board)

        if self.settings.control_type == TimeControlType.FISCHER:
            # Fischer: can use increment + portion of remaining time
            increment = self.settings.increment_seconds
            base = state.remaining_main_time / max(remaining_moves, 1)
            return base + increment * 0.8

        elif self.settings.control_type == TimeControlType.BYOYOMI:
            if state.in_byoyomi:
                # Use period time minus safety margin
                return self.settings.byoyomi_time * 0.9
            else:
                return state.remaining_main_time / max(remaining_moves, 1)

        elif self.settings.control_type == TimeControlType.CANADIAN:
            if state.in_byoyomi or state.remaining_main_time <= 0:
                # Overtime: divide period time by remaining stones
                stones_left = self.settings.canadian_stones - state.moves_in_period
                return self.settings.canadian_time / max(stones_left, 1)
            else:
                return state.remaining_main_time / max(remaining_moves, 1)

        elif self.settings.control_type == TimeControlType.SUDDEN_DEATH:
            return state.remaining_main_time / max(remaining_moves, 1)

        else:
            return 10.0  # Default fallback

    def _estimate_remaining_moves(self, board: 'GoBoard') -> int:
        """Estimate moves remaining in the game.

        Parameters
        ----------
        board : GoBoard
            Board state

        Returns
        -------
        int
            Estimated remaining moves
        """
        try:
            move_count = len(board.move_history) if hasattr(board, 'move_history') else 0
        except Exception:
            move_count = 0

        try:
            stone_count = sum(
                1 for r in range(board.size) for c in range(board.size)
                if board.get(r, c) is not None
            )
            board_fullness = stone_count / (board.size * board.size)
        except Exception:
            board_fullness = 0.5

        # Average game lengths by board size
        if self.board_size == 19:
            avg_game_length = 250
        elif self.board_size == 13:
            avg_game_length = 120
        else:
            avg_game_length = 60

        # Estimate remaining based on fullness
        estimated_remaining = max(
            int(avg_game_length * (1 - board_fullness)),
            10
        )

        return estimated_remaining

    def _estimate_game_progress(self, board: 'GoBoard') -> float:
        """Estimate game progress (0.0 to 1.0).

        Parameters
        ----------
        board : GoBoard
            Board state

        Returns
        -------
        float
            Game progress
        """
        try:
            move_count = len(board.move_history) if hasattr(board, 'move_history') else 0
        except Exception:
            move_count = 0

        if self.board_size == 19:
            avg_game_length = 250
        elif self.board_size == 13:
            avg_game_length = 120
        else:
            avg_game_length = 60

        return min(move_count / avg_game_length, 1.0)

    def _calculate_time_pressure_level(self, state: TimeState) -> float:
        """Calculate time pressure level (0.0 to 1.0).

        Parameters
        ----------
        state : TimeState
            Time state

        Returns
        -------
        float
            Time pressure level
        """
        if self.settings.control_type == TimeControlType.UNLIMITED:
            return 0.0

        if state.in_byoyomi:
            # Pressure based on remaining periods
            if state.remaining_byoyomi_periods <= 1:
                return 0.8
            elif state.remaining_byoyomi_periods <= 2:
                return 0.4
            return 0.1

        # Pressure based on remaining main time
        if state.remaining_main_time <= self.CRITICAL_TIME_THRESHOLD:
            return 1.0
        elif state.remaining_main_time <= self.TIME_PRESSURE_THRESHOLD:
            # Linear scale from 0.3 to 0.9
            ratio = 1.0 - (state.remaining_main_time - self.CRITICAL_TIME_THRESHOLD) / \
                    (self.TIME_PRESSURE_THRESHOLD - self.CRITICAL_TIME_THRESHOLD)
            return 0.3 + 0.6 * ratio
        else:
            return 0.0

    def _time_to_simulations(
        self,
        time_seconds: float,
        time_pressure: bool
    ) -> int:
        """Convert time budget to simulation count.

        Parameters
        ----------
        time_seconds : float
            Available time
        time_pressure : bool
            Whether in time pressure

        Returns
        -------
        int
            Number of MCTS simulations
        """
        if time_seconds == float('inf'):
            return self.base_simulations

        # Estimate: simulations per second (hardware dependent)
        simulations_per_second = 200

        target_sims = int(time_seconds * simulations_per_second)

        if time_pressure:
            target_sims = int(target_sims * 0.5)

        return max(self.MIN_SIMULATIONS, min(self.MAX_SIMULATIONS, target_sims))

    def get_remaining_time(self, color: str) -> float:
        """Get remaining time for a player.

        Parameters
        ----------
        color : str
            Player color

        Returns
        -------
        float
            Remaining time in seconds
        """
        return self.time_states[color].remaining_main_time

    def is_time_expired(self, color: str) -> bool:
        """Check if player has run out of time.

        Parameters
        ----------
        color : str
            Player color

        Returns
        -------
        bool
            True if time expired
        """
        if self.settings.control_type == TimeControlType.UNLIMITED:
            return False

        state = self.time_states[color]

        if state.in_byoyomi:
            return state.remaining_byoyomi_periods <= 0
        else:
            return (state.remaining_main_time <= 0 and
                    self.settings.byoyomi_periods == 0)

    def set_time(
        self,
        color: str,
        main_time: float,
        byoyomi_periods: int = 0
    ) -> None:
        """Manually set time for a player.

        Parameters
        ----------
        color : str
            Player color
        main_time : float
            Remaining main time
        byoyomi_periods : int
            Remaining byoyomi periods
        """
        self.time_states[color] = TimeState(
            remaining_main_time=main_time,
            remaining_byoyomi_periods=byoyomi_periods,
            in_byoyomi=(main_time <= 0 and byoyomi_periods > 0)
        )

    def set_strategy(self, strategy: AllocationStrategy) -> None:
        """Set time allocation strategy.

        Parameters
        ----------
        strategy : AllocationStrategy
            New strategy
        """
        self.strategy = strategy

    def get_timing_summary(self) -> Dict[str, Any]:
        """Get summary of timing history.

        Returns
        -------
        Dict[str, Any]
            Timing summary
        """
        if not self.timing_history:
            return {'total_moves': 0}

        total_allocated = sum(r.allocated_time for r in self.timing_history)
        total_actual = sum(r.actual_time for r in self.timing_history)
        avg_complexity = sum(r.complexity_score for r in self.timing_history) / len(self.timing_history)

        return {
            'total_moves': len(self.timing_history),
            'total_allocated_time': total_allocated,
            'total_actual_time': total_actual,
            'average_complexity': avg_complexity,
            'time_efficiency': total_actual / total_allocated if total_allocated > 0 else 1.0
        }

"""Time control mode definitions for Go games.

This module defines various time control systems used in Go:
- Sudden Death: Fixed total time
- Fischer: Time + increment per move
- Byoyomi: Main time + periods
- Canadian: Main time + X moves in Y seconds
"""

from __future__ import annotations

from enum import Enum
from dataclasses import dataclass
from typing import Optional


class TimeControlType(Enum):
    """Supported time control modes."""
    SUDDEN_DEATH = "sudden_death"    # Fixed total time, no increment
    FISCHER = "fischer"               # Time + increment per move
    BYOYOMI = "byoyomi"               # Main time + periods (Japanese)
    CANADIAN = "canadian"             # Main time + X moves in Y seconds
    UNLIMITED = "unlimited"           # No time limit


@dataclass
class TimeControlSettings:
    """Time control configuration.

    Attributes
    ----------
    control_type : TimeControlType
        Type of time control
    main_time_seconds : float
        Main time in seconds
    increment_seconds : float
        Fischer increment per move (default 0)
    byoyomi_periods : int
        Number of byoyomi periods (default 0)
    byoyomi_time : float
        Time per byoyomi period in seconds (default 0)
    canadian_stones : int
        Stones per Canadian period (default 0)
    canadian_time : float
        Time per Canadian period in seconds (default 0)
    """

    control_type: TimeControlType
    main_time_seconds: float
    increment_seconds: float = 0.0
    byoyomi_periods: int = 0
    byoyomi_time: float = 0.0
    canadian_stones: int = 0
    canadian_time: float = 0.0

    @classmethod
    def fischer(
        cls,
        main_time: float,
        increment: float
    ) -> 'TimeControlSettings':
        """Create Fischer time control.

        Parameters
        ----------
        main_time : float
            Initial time in seconds
        increment : float
            Increment per move in seconds

        Returns
        -------
        TimeControlSettings
            Fischer time control settings
        """
        return cls(
            control_type=TimeControlType.FISCHER,
            main_time_seconds=main_time,
            increment_seconds=increment
        )

    @classmethod
    def byoyomi(
        cls,
        main_time: float,
        periods: int,
        period_time: float
    ) -> 'TimeControlSettings':
        """Create Japanese byoyomi time control.

        Parameters
        ----------
        main_time : float
            Main time in seconds
        periods : int
            Number of byoyomi periods
        period_time : float
            Time per period in seconds

        Returns
        -------
        TimeControlSettings
            Byoyomi time control settings
        """
        return cls(
            control_type=TimeControlType.BYOYOMI,
            main_time_seconds=main_time,
            byoyomi_periods=periods,
            byoyomi_time=period_time
        )

    @classmethod
    def canadian(
        cls,
        main_time: float,
        stones: int,
        period_time: float
    ) -> 'TimeControlSettings':
        """Create Canadian byoyomi time control.

        Parameters
        ----------
        main_time : float
            Main time in seconds
        stones : int
            Stones per overtime period
        period_time : float
            Time per overtime period in seconds

        Returns
        -------
        TimeControlSettings
            Canadian time control settings
        """
        return cls(
            control_type=TimeControlType.CANADIAN,
            main_time_seconds=main_time,
            canadian_stones=stones,
            canadian_time=period_time
        )

    @classmethod
    def sudden_death(cls, total_time: float) -> 'TimeControlSettings':
        """Create sudden death time control.

        Parameters
        ----------
        total_time : float
            Total time in seconds

        Returns
        -------
        TimeControlSettings
            Sudden death settings
        """
        return cls(
            control_type=TimeControlType.SUDDEN_DEATH,
            main_time_seconds=total_time
        )

    @classmethod
    def unlimited(cls) -> 'TimeControlSettings':
        """Create unlimited time control.

        Returns
        -------
        TimeControlSettings
            Unlimited time settings
        """
        return cls(
            control_type=TimeControlType.UNLIMITED,
            main_time_seconds=float('inf')
        )

    def is_unlimited(self) -> bool:
        """Check if this is unlimited time control.

        Returns
        -------
        bool
            True if unlimited
        """
        return self.control_type == TimeControlType.UNLIMITED

    def total_time_estimate(self, expected_moves: int = 250) -> float:
        """Estimate total available time.

        Parameters
        ----------
        expected_moves : int
            Expected number of moves in game

        Returns
        -------
        float
            Estimated total time in seconds
        """
        if self.is_unlimited():
            return float('inf')

        total = self.main_time_seconds

        if self.control_type == TimeControlType.FISCHER:
            total += self.increment_seconds * expected_moves

        elif self.control_type == TimeControlType.BYOYOMI:
            total += self.byoyomi_periods * self.byoyomi_time

        elif self.control_type == TimeControlType.CANADIAN:
            overtime_periods = expected_moves / max(1, self.canadian_stones)
            total += overtime_periods * self.canadian_time

        return total


@dataclass
class TimeState:
    """Current time state for a player.

    Attributes
    ----------
    remaining_main_time : float
        Remaining main time in seconds
    remaining_byoyomi_periods : int
        Remaining byoyomi periods
    in_byoyomi : bool
        Whether currently in byoyomi
    moves_in_period : int
        Moves made in current Canadian period
    stones_in_period : int
        Stones required in current period
    """

    remaining_main_time: float
    remaining_byoyomi_periods: int = 0
    in_byoyomi: bool = False
    moves_in_period: int = 0
    stones_in_period: int = 0

    @property
    def is_time_pressure(self) -> bool:
        """Check if in time pressure (low time).

        Returns
        -------
        bool
            True if remaining time is low
        """
        if self.in_byoyomi:
            return self.remaining_byoyomi_periods <= 1
        return self.remaining_main_time < 30.0

    @property
    def is_critical(self) -> bool:
        """Check if in critical time situation.

        Returns
        -------
        bool
            True if time is very low
        """
        if self.in_byoyomi:
            return self.remaining_byoyomi_periods == 0
        return self.remaining_main_time < 10.0

    def copy(self) -> 'TimeState':
        """Create a copy of this time state.

        Returns
        -------
        TimeState
            Copy of time state
        """
        return TimeState(
            remaining_main_time=self.remaining_main_time,
            remaining_byoyomi_periods=self.remaining_byoyomi_periods,
            in_byoyomi=self.in_byoyomi,
            moves_in_period=self.moves_in_period,
            stones_in_period=self.stones_in_period
        )

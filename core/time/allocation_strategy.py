"""Time allocation strategies for Go games.

This module defines various strategies for allocating thinking time
based on position complexity, game phase, and time pressure.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
import logging

from core.time.complexity_estimator import PositionComplexity

logger = logging.getLogger(__name__)


class AllocationStrategy(ABC):
    """Base class for time allocation strategies."""

    @abstractmethod
    def calculate_multiplier(
        self,
        complexity: PositionComplexity,
        game_progress: float,
        time_pressure_level: float
    ) -> float:
        """Calculate time allocation multiplier.

        Parameters
        ----------
        complexity : PositionComplexity
            Position complexity assessment
        game_progress : float
            Game progress (0.0 to 1.0)
        time_pressure_level : float
            Time pressure (0.0 no pressure to 1.0 critical)

        Returns
        -------
        float
            Multiplier for base time (typically 0.5 to 2.0)
        """
        pass

    def get_name(self) -> str:
        """Get strategy name.

        Returns
        -------
        str
            Strategy name
        """
        return self.__class__.__name__


class UniformStrategy(AllocationStrategy):
    """Uniform time distribution (ignore complexity).

    Always returns multiplier of 1.0, distributing time evenly.
    """

    def calculate_multiplier(
        self,
        complexity: PositionComplexity,
        game_progress: float,
        time_pressure_level: float
    ) -> float:
        """Return uniform multiplier of 1.0."""
        return 1.0


class ComplexityWeightedStrategy(AllocationStrategy):
    """Allocate more time to complex positions.

    More complex positions get more thinking time.

    Parameters
    ----------
    sensitivity : float
        How much complexity affects allocation (0.0-2.0)
    """

    def __init__(self, sensitivity: float = 1.0):
        self.sensitivity = sensitivity

    def calculate_multiplier(
        self,
        complexity: PositionComplexity,
        game_progress: float,
        time_pressure_level: float
    ) -> float:
        """Calculate multiplier based on complexity.

        Higher complexity = higher multiplier.
        Reduces in time pressure.
        """
        # Base multiplier from complexity (0.5 to 1.5)
        base_multiplier = 0.5 + self.sensitivity * complexity.overall_score

        # Reduce in time pressure
        pressure_factor = 1.0 - 0.5 * time_pressure_level

        return base_multiplier * pressure_factor


class PhaseAwareStrategy(AllocationStrategy):
    """Adjust allocation based on game phase.

    Different phases of the game may warrant different time allocations.

    Parameters
    ----------
    opening_multiplier : float
        Multiplier for opening phase (default 0.8)
    midgame_multiplier : float
        Multiplier for midgame phase (default 1.2)
    endgame_multiplier : float
        Multiplier for endgame phase (default 0.7)
    """

    def __init__(
        self,
        opening_multiplier: float = 0.8,
        midgame_multiplier: float = 1.2,
        endgame_multiplier: float = 0.7
    ):
        self.phase_multipliers = {
            'opening': opening_multiplier,
            'midgame': midgame_multiplier,
            'endgame': endgame_multiplier
        }

    def calculate_multiplier(
        self,
        complexity: PositionComplexity,
        game_progress: float,
        time_pressure_level: float
    ) -> float:
        """Calculate multiplier based on game phase and complexity."""
        # Determine phase from progress
        if game_progress < 0.2:
            phase = 'opening'
        elif game_progress > 0.8:
            phase = 'endgame'
        else:
            phase = 'midgame'

        phase_mult = self.phase_multipliers[phase]

        # Also consider complexity (0.5 to 1.5)
        complexity_mult = 0.5 + 0.5 * complexity.overall_score

        # Reduce in time pressure
        pressure_factor = 1.0 - 0.3 * time_pressure_level

        return phase_mult * complexity_mult * pressure_factor


class CriticalMomentStrategy(AllocationStrategy):
    """Detect critical moments and allocate extra time.

    Positions above the critical threshold get significantly more time.

    Parameters
    ----------
    critical_threshold : float
        Complexity score above which position is critical (default 0.7)
    critical_boost : float
        Extra multiplier for critical positions (default 0.5)
    """

    def __init__(
        self,
        critical_threshold: float = 0.7,
        critical_boost: float = 0.5
    ):
        self.critical_threshold = critical_threshold
        self.critical_boost = critical_boost

    def calculate_multiplier(
        self,
        complexity: PositionComplexity,
        game_progress: float,
        time_pressure_level: float
    ) -> float:
        """Calculate multiplier with critical moment detection."""
        is_critical = complexity.overall_score > self.critical_threshold

        if is_critical and time_pressure_level < 0.5:
            # Critical position with time available: think more
            base = 1.0 + self.critical_boost
            extra = (complexity.overall_score - self.critical_threshold) * 0.5
            return base + extra

        elif is_critical:
            # Critical but time pressure: still think a bit more
            return 1.2

        else:
            # Normal position
            return 0.8 + 0.4 * complexity.overall_score


class AdaptiveStrategy(AllocationStrategy):
    """Adaptive strategy that combines multiple factors.

    Dynamically adjusts based on:
    - Position complexity
    - Game phase
    - Time pressure
    - Recent move patterns

    Parameters
    ----------
    base_sensitivity : float
        Base complexity sensitivity (default 1.0)
    """

    def __init__(self, base_sensitivity: float = 1.0):
        self.base_sensitivity = base_sensitivity
        self._recent_multipliers: list = []

    def calculate_multiplier(
        self,
        complexity: PositionComplexity,
        game_progress: float,
        time_pressure_level: float
    ) -> float:
        """Calculate adaptive multiplier."""
        # Phase-based adjustment
        if game_progress < 0.15:
            phase_factor = 0.85  # Opening: faster
        elif game_progress < 0.7:
            phase_factor = 1.1   # Midgame: more careful
        else:
            phase_factor = 0.8   # Endgame: faster

        # Complexity-based adjustment
        complexity_factor = 0.6 + 0.8 * complexity.overall_score

        # Tactical urgency boost
        tactical_boost = 0.0
        if complexity.tactical_tension > 0.5:
            tactical_boost = 0.2
        if complexity.ko_complexity > 0.5:
            tactical_boost += 0.15

        # Time pressure penalty
        pressure_penalty = 0.5 * time_pressure_level

        # Calculate base multiplier
        multiplier = (
            phase_factor *
            complexity_factor *
            (1.0 + tactical_boost) *
            (1.0 - pressure_penalty)
        )

        # Smoothing with recent history
        self._recent_multipliers.append(multiplier)
        if len(self._recent_multipliers) > 5:
            self._recent_multipliers.pop(0)

        # Slight smoothing to avoid erratic time usage
        if len(self._recent_multipliers) > 1:
            avg_recent = sum(self._recent_multipliers) / len(self._recent_multipliers)
            multiplier = 0.7 * multiplier + 0.3 * avg_recent

        # Clamp to reasonable range
        return max(0.3, min(2.5, multiplier))


def create_strategy(
    strategy_type: str,
    config: Optional[Dict[str, Any]] = None
) -> AllocationStrategy:
    """Create time allocation strategy from type name.

    Parameters
    ----------
    strategy_type : str
        Strategy type name
    config : Optional[Dict[str, Any]]
        Strategy configuration

    Returns
    -------
    AllocationStrategy
        Created strategy

    Raises
    ------
    ValueError
        If strategy type is unknown
    """
    config = config or {}

    strategy_type = strategy_type.lower().replace('-', '_')

    if strategy_type == 'uniform':
        return UniformStrategy()

    elif strategy_type in ('complexity_weighted', 'complexity'):
        return ComplexityWeightedStrategy(
            sensitivity=config.get('sensitivity', 1.0)
        )

    elif strategy_type in ('phase_aware', 'phase'):
        return PhaseAwareStrategy(
            opening_multiplier=config.get('opening_multiplier', 0.8),
            midgame_multiplier=config.get('midgame_multiplier', 1.2),
            endgame_multiplier=config.get('endgame_multiplier', 0.7)
        )

    elif strategy_type in ('critical_moment', 'critical'):
        return CriticalMomentStrategy(
            critical_threshold=config.get('critical_threshold', 0.7),
            critical_boost=config.get('critical_boost', 0.5)
        )

    elif strategy_type == 'adaptive':
        return AdaptiveStrategy(
            base_sensitivity=config.get('base_sensitivity', 1.0)
        )

    else:
        raise ValueError(f"Unknown strategy type: {strategy_type}")

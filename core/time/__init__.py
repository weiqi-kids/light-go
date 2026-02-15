"""Time allocation strategy system for Light-Go.

This package implements adaptive time management for Go games:
- TimeManager: Main time management class
- TimeControlSettings: Time control mode definitions
- ComplexityEstimator: Position complexity estimation
- AllocationStrategy: Time allocation strategies
"""

from core.time.time_control import (
    TimeControlType,
    TimeControlSettings,
    TimeState
)
from core.time.complexity_estimator import (
    PositionComplexity,
    ComplexityEstimator
)
from core.time.allocation_strategy import (
    AllocationStrategy,
    UniformStrategy,
    ComplexityWeightedStrategy,
    PhaseAwareStrategy,
    CriticalMomentStrategy,
    AdaptiveStrategy,
    create_strategy
)
from core.time.time_manager import (
    TimeAllocation,
    MoveTimingRecord,
    TimeManager
)

__all__ = [
    'TimeControlType',
    'TimeControlSettings',
    'TimeState',
    'PositionComplexity',
    'ComplexityEstimator',
    'AllocationStrategy',
    'UniformStrategy',
    'ComplexityWeightedStrategy',
    'PhaseAwareStrategy',
    'CriticalMomentStrategy',
    'AdaptiveStrategy',
    'create_strategy',
    'TimeAllocation',
    'MoveTimingRecord',
    'TimeManager'
]

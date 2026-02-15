"""Strategy narrator system for Light-Go.

This package provides natural language explanations for Go moves:
- StrategyNarrator: Main narrator class
- FeatureExtractor: Extract explainable features from MCTS
- TemplateEngine: Template-based text generation
- PositionAnalyzer: Position pattern analysis
"""

from core.narrator.styles import CommentaryStyle
from core.narrator.narrator import (
    MoveExplanation,
    StrategyNarrator,
    create_narrator
)
from core.narrator.feature_extractor import (
    ExplainableFeatures,
    FeatureExtractor
)
from core.narrator.position_analyzer import (
    PositionContext,
    PositionAnalyzer
)
from core.narrator.template_engine import TemplateEngine

__all__ = [
    'CommentaryStyle',
    'MoveExplanation',
    'StrategyNarrator',
    'create_narrator',
    'ExplainableFeatures',
    'FeatureExtractor',
    'PositionContext',
    'PositionAnalyzer',
    'TemplateEngine'
]

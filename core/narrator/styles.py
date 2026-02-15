"""Commentary style definitions for the narrator system.

This module defines different commentary styles for move explanations.
"""

from __future__ import annotations

from enum import Enum
from dataclasses import dataclass
from typing import Dict, Any


class CommentaryStyle(Enum):
    """Available commentary styles."""
    EDUCATIONAL = "educational"    # Detailed, teaches concepts
    PROFESSIONAL = "professional"  # Concise, technical terms
    CASUAL = "casual"              # Friendly, conversational
    MINIMAL = "minimal"            # Very brief


@dataclass
class StyleConfig:
    """Configuration for a commentary style.

    Attributes
    ----------
    name : str
        Style name
    description : str
        Style description
    verbosity : float
        Detail level (0.0 minimal to 1.0 very detailed)
    use_technical_terms : bool
        Whether to use Go terminology
    include_alternatives : bool
        Whether to mention alternative moves
    include_statistics : bool
        Whether to include numerical stats
    max_reasons : int
        Maximum number of reasons to give
    """

    name: str
    description: str
    verbosity: float
    use_technical_terms: bool
    include_alternatives: bool
    include_statistics: bool
    max_reasons: int


# Style configurations
STYLE_CONFIGS: Dict[CommentaryStyle, StyleConfig] = {
    CommentaryStyle.EDUCATIONAL: StyleConfig(
        name="educational",
        description="Detailed explanations that teach Go concepts",
        verbosity=1.0,
        use_technical_terms=True,
        include_alternatives=True,
        include_statistics=True,
        max_reasons=5
    ),
    CommentaryStyle.PROFESSIONAL: StyleConfig(
        name="professional",
        description="Concise analysis using proper terminology",
        verbosity=0.6,
        use_technical_terms=True,
        include_alternatives=True,
        include_statistics=False,
        max_reasons=3
    ),
    CommentaryStyle.CASUAL: StyleConfig(
        name="casual",
        description="Friendly, approachable commentary",
        verbosity=0.7,
        use_technical_terms=False,
        include_alternatives=False,
        include_statistics=False,
        max_reasons=2
    ),
    CommentaryStyle.MINIMAL: StyleConfig(
        name="minimal",
        description="Very brief, essential information only",
        verbosity=0.2,
        use_technical_terms=False,
        include_alternatives=False,
        include_statistics=False,
        max_reasons=1
    )
}


def get_style_config(style: CommentaryStyle) -> StyleConfig:
    """Get configuration for a style.

    Parameters
    ----------
    style : CommentaryStyle
        Commentary style

    Returns
    -------
    StyleConfig
        Style configuration
    """
    return STYLE_CONFIGS.get(style, STYLE_CONFIGS[CommentaryStyle.EDUCATIONAL])

"""Template engine for generating move explanations.

This module provides template-based text generation for explanations.
"""

from __future__ import annotations

import os
import json
import random
from pathlib import Path
from string import Template
from typing import Dict, Any, Optional, List
import logging

logger = logging.getLogger(__name__)


# Default templates when files are not available
DEFAULT_TEMPLATES = {
    'en': {
        'policy': {
            'high_confidence': "The move at $move is clearly the best choice.",
            'moderate_confidence': "$move appears to be the best move, though alternatives exist.",
            'low_confidence': "This is a difficult position. $move is chosen among several options.",
            'capture': "Playing at $move captures stones.",
            'defense': "$move defends a group that was in danger.",
            'extension': "$move extends influence.",
            'connection': "$move connects groups for safety.",
            'cut': "$move cuts opponent's stones apart."
        },
        'value': {
            'winning': "The position favors the current player.",
            'losing': "The position is difficult for the current player.",
            'even': "The position is roughly even.",
            'unclear': "The outcome is unclear."
        },
        'tactical': {
            'atari': "There is a group in atari that needs attention.",
            'ko': "A ko fight is in progress.",
            'ladder': "A ladder situation exists.",
            'snapback': "A snapback pattern is present."
        },
        'phase': {
            'opening': "In the opening, territorial frameworks are being established.",
            'midgame': "The middlegame involves complex fighting and territory disputes.",
            'endgame': "In the endgame, the focus is on small gains."
        }
    },
    'zh': {
        'policy': {
            'high_confidence': "$move 是明顯的最佳選擇。",
            'moderate_confidence': "$move 看起來是最好的一手，但也有其他選項。",
            'low_confidence': "這是一個困難的局面。在多個選項中選擇了 $move。",
            'capture': "在 $move 落子可以提子。",
            'defense': "$move 防守了一個危險的棋筋。",
            'extension': "$move 擴展了勢力範圍。",
            'connection': "$move 連接棋筋以確保安全。",
            'cut': "$move 切斷了對手的棋子。"
        },
        'value': {
            'winning': "當前局面有利。",
            'losing': "當前局面處於劣勢。",
            'even': "局面大致均衡。",
            'unclear': "勝負尚未分明。"
        },
        'tactical': {
            'atari': "有棋筋處於打吃狀態，需要注意。",
            'ko': "正在進行打劫。",
            'ladder': "存在征子局面。",
            'snapback': "存在倒扑局面。"
        },
        'phase': {
            'opening': "在佈局階段，正在建立領地框架。",
            'midgame': "中盤涉及複雜的戰鬥和領地爭奪。",
            'endgame': "在收官階段，重點是細小的得失。"
        }
    }
}


class TemplateEngine:
    """Template-based text generation for explanations.

    Parameters
    ----------
    template_dir : Optional[str]
        Directory containing template files
    language : str
        Language code (default "en")
    """

    def __init__(
        self,
        template_dir: Optional[str] = None,
        language: str = "en"
    ):
        self.language = language
        self.template_dir = Path(template_dir) if template_dir else None
        self.templates: Dict[str, Dict[str, str]] = {}

        self._load_templates()

    def _load_templates(self) -> None:
        """Load templates for current language."""
        # Start with defaults
        self.templates = DEFAULT_TEMPLATES.get(
            self.language,
            DEFAULT_TEMPLATES['en']
        ).copy()

        # Try to load from files
        if self.template_dir:
            lang_dir = self.template_dir / self.language
            if not lang_dir.is_dir():
                lang_dir = self.template_dir / "en"

            if lang_dir.is_dir():
                for filename in lang_dir.glob("*.json"):
                    category = filename.stem
                    try:
                        with open(filename, 'r', encoding='utf-8') as f:
                            self.templates[category] = json.load(f)
                    except Exception as e:
                        logger.warning(f"Failed to load {filename}: {e}")

    def render(
        self,
        category: str,
        template_name: str,
        context: Dict[str, Any]
    ) -> str:
        """Render a template with context.

        Parameters
        ----------
        category : str
            Template category (e.g., "policy", "value", "tactical")
        template_name : str
            Specific template name
        context : Dict[str, Any]
            Variables to substitute

        Returns
        -------
        str
            Rendered text
        """
        template_str = self.templates.get(category, {}).get(template_name, "")

        if not template_str:
            return ""

        try:
            return Template(template_str).safe_substitute(context)
        except Exception as e:
            logger.warning(f"Template rendering error: {e}")
            return ""

    def render_random(
        self,
        category: str,
        context: Dict[str, Any]
    ) -> str:
        """Render a random template from a category.

        Parameters
        ----------
        category : str
            Template category
        context : Dict[str, Any]
            Variables to substitute

        Returns
        -------
        str
            Rendered text
        """
        templates = self.templates.get(category, {})
        if not templates:
            return ""

        template_name = random.choice(list(templates.keys()))
        return self.render(category, template_name, context)

    def set_language(self, language: str) -> None:
        """Change language and reload templates.

        Parameters
        ----------
        language : str
            New language code
        """
        self.language = language
        self._load_templates()

    def get_available_templates(self) -> Dict[str, List[str]]:
        """Get all available templates.

        Returns
        -------
        Dict[str, List[str]]
            Category -> list of template names
        """
        return {
            category: list(templates.keys())
            for category, templates in self.templates.items()
        }

    def format_move(
        self,
        move: Optional[tuple],
        board_size: int = 19
    ) -> str:
        """Format a move coordinate as a string.

        Parameters
        ----------
        move : Optional[tuple]
            Move coordinate (row, col) or None for pass
        board_size : int
            Board size

        Returns
        -------
        str
            Formatted move string (e.g., "D4", "Q16", "PASS")
        """
        if move is None:
            return "PASS" if self.language == 'en' else "虛手"

        row, col = move

        # Convert to Go coordinates
        # Columns: A-T (skipping I)
        col_letters = "ABCDEFGHJKLMNOPQRST"
        if col < len(col_letters):
            col_str = col_letters[col]
        else:
            col_str = str(col + 1)

        # Rows: 1-19 (from bottom in standard notation)
        row_num = board_size - row

        return f"{col_str}{row_num}"

    def format_percentage(self, value: float) -> str:
        """Format a value as percentage.

        Parameters
        ----------
        value : float
            Value (0.0-1.0)

        Returns
        -------
        str
            Formatted percentage
        """
        return f"{value * 100:.1f}%"


def create_template_engine(
    template_dir: Optional[str] = None,
    language: str = "en"
) -> TemplateEngine:
    """Create a template engine.

    Parameters
    ----------
    template_dir : Optional[str]
        Template directory
    language : str
        Language code

    Returns
    -------
    TemplateEngine
        Configured template engine
    """
    return TemplateEngine(template_dir=template_dir, language=language)

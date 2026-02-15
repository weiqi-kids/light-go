"""Main narrator class for generating move explanations.

This module provides the StrategyNarrator class that generates
natural language explanations for Go moves.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any
import logging

from core.narrator.styles import CommentaryStyle, get_style_config, StyleConfig
from core.narrator.feature_extractor import FeatureExtractor, ExplainableFeatures
from core.narrator.position_analyzer import PositionAnalyzer, PositionContext
from core.narrator.template_engine import TemplateEngine

logger = logging.getLogger(__name__)


@dataclass
class MoveExplanation:
    """Explanation for a move decision.

    Attributes
    ----------
    move : Optional[Tuple[int, int]]
        The move (row, col) or None for pass
    move_name : str
        Formatted move name (e.g., "D4", "PASS")
    summary : str
        One-line summary
    policy_reasons : List[str]
        Why this move was chosen
    value_assessment : str
        Position evaluation
    alternative_moves : List[Dict[str, Any]]
        Top alternatives with explanations
    position_context : Dict[str, Any]
        Context information
    tactical_notes : List[str]
        Tactical observations
    confidence : float
        Confidence in the move (0.0-1.0)
    """

    move: Optional[Tuple[int, int]]
    move_name: str
    summary: str
    policy_reasons: List[str] = field(default_factory=list)
    value_assessment: str = ""
    alternative_moves: List[Dict[str, Any]] = field(default_factory=list)
    position_context: Dict[str, Any] = field(default_factory=dict)
    tactical_notes: List[str] = field(default_factory=list)
    confidence: float = 0.0


class StrategyNarrator:
    """Generate natural language explanations for Go moves.

    Parameters
    ----------
    style : CommentaryStyle
        Commentary style to use (default EDUCATIONAL)
    language : str
        Language code (default "en")
    template_dir : Optional[str]
        Directory for custom templates
    board_size : int
        Board size (default 19)
    """

    def __init__(
        self,
        style: CommentaryStyle = CommentaryStyle.EDUCATIONAL,
        language: str = "en",
        template_dir: Optional[str] = None,
        board_size: int = 19
    ):
        self.style = style
        self.language = language
        self.board_size = board_size

        # Components
        self.template_engine = TemplateEngine(
            template_dir=template_dir,
            language=language
        )
        self.feature_extractor = FeatureExtractor(board_size=board_size)
        self.position_analyzer = PositionAnalyzer(board_size=board_size)

        # Style configuration
        self.style_config = get_style_config(style)

    def explain_move(
        self,
        mcts_root: 'MCTSNode',
        chosen_move: Optional[Tuple[int, int]],
        board: 'GoBoard',
        color: str,
        move_probs: np.ndarray
    ) -> MoveExplanation:
        """Generate explanation for a move.

        Parameters
        ----------
        mcts_root : MCTSNode
            Root node of MCTS search tree
        chosen_move : Optional[Tuple[int, int]]
            The chosen move (None for pass)
        board : GoBoard
            Current board state
        color : str
            Color that moved ('b' or 'w')
        move_probs : np.ndarray
            Move probability distribution from MCTS

        Returns
        -------
        MoveExplanation
            Structured explanation
        """
        # Extract features
        features = self.feature_extractor.extract(
            mcts_root=mcts_root,
            board=board,
            color=color,
            move_probs=move_probs
        )

        # Analyze position
        context = self.position_analyzer.analyze(board, color)

        # Generate explanation
        return self._generate_explanation(
            features=features,
            context=context,
            chosen_move=chosen_move
        )

    def _generate_explanation(
        self,
        features: ExplainableFeatures,
        context: PositionContext,
        chosen_move: Optional[Tuple[int, int]]
    ) -> MoveExplanation:
        """Generate explanation from extracted features.

        Parameters
        ----------
        features : ExplainableFeatures
            Extracted features
        context : PositionContext
            Position context
        chosen_move : Optional[Tuple[int, int]]
            The move

        Returns
        -------
        MoveExplanation
            Generated explanation
        """
        # Format move name
        move_name = self.template_engine.format_move(
            chosen_move,
            self.board_size
        )

        # Generate summary
        summary = self._generate_summary(features, context, move_name)

        # Generate policy reasons
        policy_reasons = self._generate_policy_reasons(features, context)

        # Generate value assessment
        value_assessment = self._generate_value_assessment(features)

        # Generate alternative moves explanation
        alternative_moves = self._generate_alternatives(features)

        # Generate tactical notes
        tactical_notes = self._generate_tactical_notes(features, context)

        # Build position context dict
        position_context = {
            'game_phase': context.game_phase,
            'ko_active': context.ko_active,
            'groups_at_risk': len(context.groups_at_risk),
            'territory_control': context.territory_control
        }

        return MoveExplanation(
            move=chosen_move,
            move_name=move_name,
            summary=summary,
            policy_reasons=policy_reasons[:self.style_config.max_reasons],
            value_assessment=value_assessment,
            alternative_moves=alternative_moves,
            position_context=position_context,
            tactical_notes=tactical_notes,
            confidence=features.confidence
        )

    def _generate_summary(
        self,
        features: ExplainableFeatures,
        context: PositionContext,
        move_name: str
    ) -> str:
        """Generate one-line summary.

        Parameters
        ----------
        features : ExplainableFeatures
            Features
        context : PositionContext
            Context
        move_name : str
            Formatted move name

        Returns
        -------
        str
            Summary text
        """
        template_context = {'move': move_name}

        # Choose template based on confidence
        if features.confidence > 0.6:
            return self.template_engine.render(
                'policy', 'high_confidence', template_context
            )
        elif features.confidence > 0.3:
            return self.template_engine.render(
                'policy', 'moderate_confidence', template_context
            )
        else:
            return self.template_engine.render(
                'policy', 'low_confidence', template_context
            )

    def _generate_policy_reasons(
        self,
        features: ExplainableFeatures,
        context: PositionContext
    ) -> List[str]:
        """Generate reasons for the move choice.

        Parameters
        ----------
        features : ExplainableFeatures
            Features
        context : PositionContext
            Context

        Returns
        -------
        List[str]
            Reasons
        """
        reasons = []

        # Visit count reason
        if features.chosen_move_visits > 100:
            if self.language == 'zh':
                reasons.append(f"MCTS 搜索了 {features.chosen_move_visits} 次，顯示這是可靠的選擇")
            else:
                reasons.append(f"MCTS explored {features.chosen_move_visits} variations, indicating reliability")

        # Prior probability reason
        if features.chosen_move_prior > 0.1:
            pct = self.template_engine.format_percentage(features.chosen_move_prior)
            if self.language == 'zh':
                reasons.append(f"神經網路給予 {pct} 的初始概率")
            else:
                reasons.append(f"Neural network assigned {pct} prior probability")

        # Value assessment
        if features.chosen_move_value > 0.3:
            if self.language == 'zh':
                reasons.append("這手棋提升了獲勝機會")
            else:
                reasons.append("This move improves winning chances")
        elif features.chosen_move_value < -0.3:
            if self.language == 'zh':
                reasons.append("這是防守性的必要之舉")
            else:
                reasons.append("This is a necessary defensive move")

        # Tactical reasons
        if context.ko_active:
            if self.language == 'zh':
                reasons.append("正在處理打劫局面")
            else:
                reasons.append("Addressing the ko situation")

        if features.groups_in_atari:
            if self.language == 'zh':
                reasons.append("有棋筋處於打吃狀態需要處理")
            else:
                reasons.append("Responding to groups in atari")

        # Phase-specific reason
        phase_reason = self.template_engine.render(
            'phase', features.game_phase, {}
        )
        if phase_reason:
            reasons.append(phase_reason)

        return reasons

    def _generate_value_assessment(
        self,
        features: ExplainableFeatures
    ) -> str:
        """Generate value/position assessment.

        Parameters
        ----------
        features : ExplainableFeatures
            Features

        Returns
        -------
        str
            Value assessment
        """
        if features.current_advantage == 'black_leading':
            return self.template_engine.render('value', 'winning', {})
        elif features.current_advantage == 'white_leading':
            return self.template_engine.render('value', 'losing', {})
        else:
            return self.template_engine.render('value', 'even', {})

    def _generate_alternatives(
        self,
        features: ExplainableFeatures
    ) -> List[Dict[str, Any]]:
        """Generate alternative moves explanation.

        Parameters
        ----------
        features : ExplainableFeatures
            Features

        Returns
        -------
        List[Dict[str, Any]]
            Alternatives with explanations
        """
        if not self.style_config.include_alternatives:
            return []

        alternatives = []

        for alt in features.top_alternatives[:3]:
            move_name = self.template_engine.format_move(
                alt['move'],
                self.board_size
            )

            alt_info = {
                'move': alt['move'],
                'move_name': move_name,
                'visits': alt['visits'],
                'value': alt['value']
            }

            if self.style_config.include_statistics:
                alt_info['prior'] = alt['prior']

            alternatives.append(alt_info)

        return alternatives

    def _generate_tactical_notes(
        self,
        features: ExplainableFeatures,
        context: PositionContext
    ) -> List[str]:
        """Generate tactical observations.

        Parameters
        ----------
        features : ExplainableFeatures
            Features
        context : PositionContext
            Context

        Returns
        -------
        List[str]
            Tactical notes
        """
        notes = []

        if context.ko_active:
            notes.append(self.template_engine.render('tactical', 'ko', {}))

        if features.groups_in_atari:
            notes.append(self.template_engine.render('tactical', 'atari', {}))

        return notes

    def explain_position(
        self,
        board: 'GoBoard',
        color: str
    ) -> str:
        """Generate position assessment without move selection.

        Parameters
        ----------
        board : GoBoard
            Board state
        color : str
            Color to move

        Returns
        -------
        str
            Position assessment
        """
        return self.position_analyzer.get_position_summary(board, color)

    def set_style(self, style: CommentaryStyle) -> None:
        """Change commentary style.

        Parameters
        ----------
        style : CommentaryStyle
            New style
        """
        self.style = style
        self.style_config = get_style_config(style)

    def set_language(self, language: str) -> None:
        """Change output language.

        Parameters
        ----------
        language : str
            New language code
        """
        self.language = language
        self.template_engine.set_language(language)


def create_narrator(
    style: str = "educational",
    language: str = "en",
    board_size: int = 19
) -> StrategyNarrator:
    """Create a narrator with specified settings.

    Parameters
    ----------
    style : str
        Style name
    language : str
        Language code
    board_size : int
        Board size

    Returns
    -------
    StrategyNarrator
        Configured narrator
    """
    style_enum = CommentaryStyle[style.upper()]
    return StrategyNarrator(
        style=style_enum,
        language=language,
        board_size=board_size
    )

"""Hugging Face compatible Go AI models with architecture evolution support.

This module provides neural network models for Go/Weiqi with evolvable architectures.
"""

from .modeling_go_ai import GoAIModel
from .board_encoder import BoardEncoder

__all__ = ["GoAIModel", "BoardEncoder"]

"""Position evaluator：盤面 → (policy, logits, value) 的唯一接縫。

所有搜尋（MCTS／PUCT、GumbelMCTS；見 core.search）都只透過 :class:`PositionEvaluator`
取得網路評估，編碼、批次、裝置、softmax 都藏在這裡。

慣例（所有實作都必須遵守）：

- ``policy`` / ``logits``：長度 ``N*N + 1``，索引 ``row * N + col``，pass 在最後
- ``value``：**輪走方（color）視角**，範圍 [-1, 1]，+1 表示輪走方必勝

實作：

- :class:`NetworkEvaluator` —— 包一個 ``model(features) -> (logits, value)`` 的網路
- :class:`MuZeroEvaluator` —— MuZero 的 representation + prediction
- :class:`UniformEvaluator` —— 均勻 policy、固定 value（測試／無模型時使用）
- :class:`MaterialEvaluator` —— 均勻 policy、value 依子數差（不需模型的基準）
"""
from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch

from core.game_rules import GoBoard
from input.lightgo_encoder import board_to_arrays, encode_board

Move = Union[Tuple[int, int], str]


@dataclass(frozen=True)
class Evaluation:
    """單一盤面的評估結果。

    Attributes
    ----------
    policy : np.ndarray
        (N*N + 1,) 機率，pass 在最後
    logits : np.ndarray
        (N*N + 1,) 未正規化的 logits（Gumbel 採樣用）
    value : float
        輪走方視角的價值，[-1, 1]
    """

    policy: np.ndarray
    logits: np.ndarray
    value: float

    def policy_dict(self) -> Dict[Move, float]:
        """轉成 ``{(row, col): p, 'pass': p}``。"""
        num_points = len(self.policy) - 1
        size = math.isqrt(num_points)
        probs = self.policy.tolist()
        result: Dict[Move, float] = {
            (i // size, i % size): probs[i] for i in range(num_points)
        }
        result['pass'] = probs[num_points]
        return result


class PositionEvaluator(ABC):
    """盤面評估介面。"""

    @abstractmethod
    def evaluate(self, boards: Sequence[GoBoard], colors: Sequence[str]) -> List[Evaluation]:
        """批次評估；``colors[i]`` 是 ``boards[i]`` 的輪走方。"""

    def evaluate_one(self, board: GoBoard, color: str) -> Evaluation:
        """評估單一盤面。"""
        return self.evaluate([board], [color])[0]


class NetworkEvaluator(PositionEvaluator):
    """包裝 ``model(features) -> (policy_logits, value)`` 的神經網路。

    Parameters
    ----------
    model : torch.nn.Module
        輸入 LightGo 2-plane 特徵 (B, 2, N, N)，輸出 logits (B, N*N+1) 與 value (B, 1)
    device : Optional[torch.device | str]
        推論裝置；預設取模型參數所在裝置
    """

    def __init__(self, model: torch.nn.Module, device: Optional[Union[torch.device, str]] = None):
        self.model = model
        if device is None:
            try:
                device = next(model.parameters()).device
            except StopIteration:
                device = 'cpu'
        self.device = torch.device(device)

    def _forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.model(features)

    def evaluate(self, boards: Sequence[GoBoard], colors: Sequence[str]) -> List[Evaluation]:
        if not boards:
            return []
        if len({board.size for board in boards}) != 1:
            raise ValueError("同一批次的棋盤大小必須相同")
        features = np.stack([encode_board(board, color) for board, color in zip(boards, colors)])
        features_tensor = torch.from_numpy(features).to(self.device)

        with torch.no_grad():
            logits, values = self._forward(features_tensor)
            probs = torch.softmax(logits, dim=1)

        logits_np = logits.float().cpu().numpy()
        probs_np = probs.float().cpu().numpy()
        values_np = values.float().reshape(len(boards), -1)[:, 0].cpu().numpy()
        return [
            Evaluation(policy=probs_np[i], logits=logits_np[i], value=float(values_np[i]))
            for i in range(len(boards))
        ]


class MuZeroEvaluator(NetworkEvaluator):
    """MuZero 網路：representation → prediction（不展開 dynamics）。"""

    def __init__(self, networks: torch.nn.Module, device: Optional[Union[torch.device, str]] = None):
        super().__init__(networks, device)
        self.networks = networks

    def _forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        hidden_state = self.networks.representation(features)
        return self.networks.prediction(hidden_state)


class UniformEvaluator(PositionEvaluator):
    """均勻 policy、固定 value 的評估器（測試或無模型時使用）。"""

    def __init__(self, value: float = 0.0):
        self.value = float(value)

    def evaluate(self, boards: Sequence[GoBoard], colors: Sequence[str]) -> List[Evaluation]:
        results = []
        for board in boards:
            num_actions = board.size * board.size + 1
            results.append(Evaluation(
                policy=np.full(num_actions, 1.0 / num_actions, dtype=np.float32),
                logits=np.zeros(num_actions, dtype=np.float32),
                value=self.value,
            ))
        return results


class MaterialEvaluator(UniformEvaluator):
    """均勻 policy、value = tanh((輪走方子數 − 對手子數) / scale)。

    不需模型、可解釋的基準評估器；搜尋的「會不會提子」行為測試用它。
    """

    def __init__(self, scale: float = 2.0):
        super().__init__(0.0)
        self.scale = float(scale)

    def evaluate(self, boards: Sequence[GoBoard], colors: Sequence[str]) -> List[Evaluation]:
        results = super().evaluate(boards, colors)
        for i, (board, color) in enumerate(zip(boards, colors)):
            black, white = board_to_arrays(board)
            own, opp = (black, white) if color == 'b' else (white, black)
            value = math.tanh((int(own.sum()) - int(opp.sum())) / self.scale)
            results[i] = Evaluation(policy=results[i].policy, logits=results[i].logits, value=value)
        return results


def as_evaluator(
    model_or_evaluator: Union[PositionEvaluator, torch.nn.Module, None],
    evaluator: Optional[PositionEvaluator] = None,
    device: Optional[Union[torch.device, str]] = None,
) -> PositionEvaluator:
    """搜尋類別建構子的向後相容入口：``model`` 自動包成 :class:`NetworkEvaluator`。"""
    if evaluator is not None:
        return evaluator
    if isinstance(model_or_evaluator, PositionEvaluator):
        return model_or_evaluator
    if model_or_evaluator is None:
        raise ValueError("必須提供 model 或 evaluator")
    return NetworkEvaluator(model_or_evaluator, device)

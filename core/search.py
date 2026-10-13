"""統一的搜尋介面（issue #55）。

所有對局／自我對弈／GTP 只依賴 :class:`Search` 這個介面，不 import 具體類別：

- ``search(board, color) -> (policy, root)``：``policy`` 長度 ``N*N + 1``、總和 1，
  索引 ``row * N + col``，pass 在最後
- ``select_move(board, color, temperature=None) -> Optional[(row, col)]``：
  ``None`` 表示 pass；``temperature=None`` 用實例預設值，``0`` 為取最大
- ``advance_tree(move)`` / ``clear_tree()``：樹重用。重用前一律以
  :func:`position_key` 驗證盤面，不符就丟掉舊樹重新搜尋

兩個實作（adapter）：

- ``'puct'`` —— :class:`core.mcts.MCTS`（AlphaZero PUCT；含批次葉節點評估與評估快取）
- ``'gumbel'`` —— :class:`core.gumbel_mcts.GumbelMCTS`（Gumbel-Top-k + Sequential Halving）

用 :func:`create_search` 依設定建立。
"""
from __future__ import annotations

from typing import Any, Hashable, Optional, Protocol, Tuple, runtime_checkable

import numpy as np

from core.evaluator import PositionEvaluator, as_evaluator
from core.game_rules import GoBoard

Move = Optional[Tuple[int, int]]

#: :func:`create_search` 支援的搜尋種類
SEARCH_KINDS: Tuple[str, ...] = ('puct', 'gumbel')


@runtime_checkable
class Search(Protocol):
    """棋盤搜尋介面。"""

    num_simulations: int

    def search(self, board: GoBoard, color: str) -> Tuple[np.ndarray, Any]:
        """搜尋 ``board``（``color`` 輪走），回傳 (N*N+1 機率, 根節點)。"""

    def select_move(
        self, board: GoBoard, color: str, temperature: Optional[float] = None
    ) -> Move:
        """搜尋並選一手；``None`` 表示 pass。"""

    def advance_tree(self, move: Move) -> bool:
        """把快取的根推進到 ``move`` 的子節點；不在樹裡則清空並回傳 False。"""

    def clear_tree(self) -> None:
        """丟掉快取的搜尋樹。"""


def position_key(board: GoBoard, color: str) -> Hashable:
    """盤面＋輪走方的識別鍵（樹重用驗證與評估快取共用）。

    包含棋盤大小、所有棋子、劫點、連續 pass 數與輪走方。
    """
    stones = tuple(tuple(row) for row in board.board.board)
    return (board.size, stones, board.ko_point, board.consecutive_passes, color)


def find_matching_node(cached_root: Any, board: GoBoard, color: str, max_depth: int = 2) -> Any:
    """在快取樹（根、子、孫…到 ``max_depth``）中找局面等於 (board, color) 的節點。

    節點需有 ``board`` / ``to_play`` / ``children`` / ``visit_count``；只走訪過的節點。
    找不到回傳 ``None``。
    """
    if cached_root is None:
        return None
    target = position_key(board, color)
    frontier = [cached_root]
    for _ in range(max_depth + 1):
        for node in frontier:
            if position_key(node.board, node.to_play) == target:
                return node
        frontier = [
            child
            for node in frontier
            for child in node.children.values()
            if child.visit_count > 0
        ]
    return None


def policy_to_move(policy: np.ndarray, size: int, temperature: float) -> Move:
    """依 ``temperature`` 從 ``policy`` 選一手（0 = argmax）；pass 回傳 ``None``。"""
    if temperature == 0:
        move_idx = int(np.argmax(policy))
    else:
        weights = np.power(policy.astype(np.float64), 1.0 / temperature)
        total = weights.sum()
        if total <= 0:
            move_idx = int(np.argmax(policy))
        else:
            move_idx = int(np.random.choice(len(policy), p=weights / total))
    if move_idx == size * size:
        return None
    return divmod(move_idx, size)


def create_search(
    kind: str = 'puct',
    evaluator: Optional[PositionEvaluator] = None,
    *,
    model=None,
    **config: Any,
) -> Search:
    """依 ``kind`` 建立搜尋。

    Parameters
    ----------
    kind : str
        ``'puct'`` 或 ``'gumbel'``
    evaluator : PositionEvaluator, optional
        盤面評估器
    model : torch.nn.Module, optional
        沒給 ``evaluator`` 時，包成 :class:`~core.evaluator.NetworkEvaluator`
    **config
        轉給具體類別的設定（``num_simulations``、``board_size``、``temperature`` …）
    """
    resolved = as_evaluator(model, evaluator, config.pop('device', None))
    if kind == 'puct':
        from core.mcts import MCTS
        return MCTS(model=model, evaluator=resolved, **config)
    if kind == 'gumbel':
        from core.gumbel_mcts import GumbelMCTS
        return GumbelMCTS(model=model, evaluator=resolved, **config)
    raise ValueError(f"未知的搜尋種類: {kind!r}（可用：{', '.join(SEARCH_KINDS)}）")


__all__ = [
    'Search', 'SEARCH_KINDS', 'create_search', 'find_matching_node', 'position_key', 'policy_to_move',
]

"""已併入 :class:`core.mcts.MCTS`（issue #55），這裡只保留向後相容的別名。

批次葉節點評估、虛擬損失、動態批次、評估快取現在都是 ``MCTS`` 的選項：

    create_search('puct', evaluator, batch_size=8, dynamic_batch=True, eval_cache_size=100_000)

差異：``select_move`` 依 :class:`core.search.Search` 介面，pass 回傳 ``None``
（舊版會改回傳第一個合法點）。
"""
from __future__ import annotations

import warnings
from typing import Optional, Tuple

import numpy as np

from core.evaluator import PositionEvaluator
from core.game_rules import GoBoard
from core.mcts import MCTS, MCTSNode

#: 舊節點類別名稱
MCTSNodeOptimized = MCTSNode


class MCTSOptimized(MCTS):
    """Deprecated：等同 ``MCTS(batch_size=8, dynamic_batch=True, eval_cache_size=100000, reuse_tree=False)``。"""

    def __init__(
        self,
        model=None,
        num_simulations: int = 800,
        temperature: float = 1.0,
        batch_size: int = 8,
        use_virtual_loss: bool = True,
        dirichlet_alpha: float = 0.03,
        dirichlet_epsilon: float = 0.25,
        dynamic_batch: bool = True,
        min_batch_size: int = 4,
        max_batch_size: int = 32,
        evaluator: Optional[PositionEvaluator] = None,
        reuse_tree: bool = False,
        eval_cache_size: int = 100000,
        **kwargs,
    ):
        warnings.warn(
            "MCTSOptimized 已併入 core.mcts.MCTS；請改用 core.search.create_search('puct', ...)",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(
            model=model,
            num_simulations=num_simulations,
            temperature=temperature,
            dirichlet_alpha=dirichlet_alpha,
            dirichlet_epsilon=dirichlet_epsilon,
            reuse_tree=reuse_tree,
            evaluator=evaluator,
            batch_size=batch_size,
            use_virtual_loss=use_virtual_loss,
            dynamic_batch=dynamic_batch,
            min_batch_size=min_batch_size,
            max_batch_size=max_batch_size,
            eval_cache_size=eval_cache_size,
            **kwargs,
        )

    def search(
        self, board: GoBoard, color: str, reuse_tree: Optional[bool] = None
    ) -> Tuple[np.ndarray, MCTSNode]:
        """同 :meth:`MCTS.search`；``reuse_tree`` 參數保留舊簽名（給值時覆寫實例設定）。"""
        if reuse_tree is None:
            return super().search(board, color)
        saved, self.reuse_tree = self.reuse_tree, reuse_tree
        try:
            return super().search(board, color)
        finally:
            self.reuse_tree = saved


__all__ = ["MCTSOptimized", "MCTSNodeOptimized"]

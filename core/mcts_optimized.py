"""優化版 MCTS 實現

性能優化：
1. 批次神經網絡推理
2. 虛擬損失（支持並行搜索）
3. 節點緩存和重用
4. 提前終止（訪問閾值）
5. 減少棋盤複製

價值視角慣例（與 core/mcts.py 相同）：``node.color`` 是走進該節點的一方，
``node.q_value`` 以該方（＝父節點輪走方）視角儲存，選擇時直接取最大。
"""

from __future__ import annotations

import math
import numpy as np
from typing import Optional, List, Tuple, Dict, Union
from dataclasses import dataclass, field
from collections import defaultdict

from core.evaluator import PositionEvaluator, as_evaluator
from core.game_rules import GoBoard


def _opponent(color: str) -> str:
    return 'w' if color == 'b' else 'b'


@dataclass
class MCTSNodeOptimized:
    """優化版 MCTS 節點"""

    board_hash: int  # 棋盤哈希（避免存儲完整棋盤）
    parent: Optional[MCTSNodeOptimized] = None
    move: Optional[Tuple[int, int]] = None
    color: str = 'b'  # 走進此節點的一方（根節點為輪走方的對手）
    prior: float = 0.0

    # 統計
    visit_count: int = 0
    total_value: float = 0.0
    virtual_loss: int = 0

    # 子節點（延遲創建）
    children: Dict[Tuple[int, int], MCTSNodeOptimized] = field(default_factory=dict)
    is_expanded: bool = False

    @property
    def to_play(self) -> str:
        """此節點的輪走方"""
        return _opponent(self.color)

    @property
    def q_value(self) -> float:
        """Q 值（父節點輪走方視角，不受虛擬損失影響）

        Note: Q-value 只使用實際訪問次數，virtual_loss 只影響 UCB 探索項
        """
        if self.visit_count == 0:
            return 0.0
        return self.total_value / self.visit_count

    @property
    def ucb_score(self) -> float:
        """UCB 分數"""
        if self.parent is None:
            return 0.0

        c_puct = 1.4
        parent_visits = self.parent.visit_count + self.parent.virtual_loss
        return self.q_value + c_puct * self.prior * math.sqrt(parent_visits) / (1 + self.visit_count)

    def select_child(self) -> MCTSNodeOptimized:
        """選擇最佳子節點"""
        return max(self.children.values(), key=lambda node: node.ucb_score)


class MCTSOptimized:
    """優化版 MCTS 引擎

    主要優化：
    - 批次推理：一次評估多個節點
    - 虛擬損失：支持並行搜索
    - 節點池：重用節點對象
    - 棋盤緩存：避免重複複製

    Parameters
    ----------
    model : torch.nn.Module | PositionEvaluator, optional
        回傳 (policy_logits, value) 的網絡（自動包成 NetworkEvaluator），或直接傳評估器
    num_simulations : int
        模擬次數
    temperature : float
        溫度參數
    batch_size : int
        批次推理大小（當 dynamic_batch=True 時作為起始值）
    use_virtual_loss : bool
        是否使用虛擬損失
    dynamic_batch : bool
        是否啟用動態批次大小調整
    min_batch_size : int
        動態批次最小值
    max_batch_size : int
        動態批次最大值
    evaluator : PositionEvaluator, optional
        盤面評估器（優先於 model）
    """

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
    ):
        self.model = model
        self.evaluator = as_evaluator(model, evaluator)
        self.num_simulations = num_simulations
        self.temperature = temperature
        self.batch_size = batch_size
        self.use_virtual_loss = use_virtual_loss
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_epsilon = dirichlet_epsilon
        self.dynamic_batch = dynamic_batch
        self.min_batch_size = min_batch_size
        self.max_batch_size = max_batch_size

        # 動態批次追蹤
        self._current_batch_size = batch_size
        self._last_inference_time = 0.0

        # 緩存
        self.board_cache: Dict[int, GoBoard] = {}
        self.evaluation_cache: Dict[Tuple[int, str], Tuple[Dict, float]] = {}

    def _get_board_hash(self, board: GoBoard) -> int:
        """獲取棋盤哈希（簡化版）"""
        # 使用棋盤狀態創建哈希
        state_str = ""
        for row in range(board.size):
            for col in range(board.size):
                stone = board.get(row, col)
                if stone == 'b':
                    state_str += 'B'
                elif stone == 'w':
                    state_str += 'W'
                else:
                    state_str += '.'
        return hash(state_str)

    def _get_dynamic_batch_size(self, simulations_done: int, total_simulations: int) -> int:
        """計算動態批次大小

        策略：
        - 搜索初期使用較大批次（更多並行）
        - 搜索後期使用較小批次（更精確）

        Parameters
        ----------
        simulations_done : int
            已完成的模擬次數
        total_simulations : int
            總模擬次數

        Returns
        -------
        int
            建議的批次大小
        """
        if not self.dynamic_batch:
            return self.batch_size

        # 計算進度 (0.0 ~ 1.0)
        progress = simulations_done / max(total_simulations, 1)

        # 根據進度調整批次大小
        # 初期：較大批次；後期：較小批次
        if progress < 0.3:
            # 初期：使用最大批次
            target_size = self.max_batch_size
        elif progress < 0.7:
            # 中期：線性縮減
            ratio = (progress - 0.3) / 0.4
            target_size = int(
                self.max_batch_size - ratio * (self.max_batch_size - self.min_batch_size)
            )
        else:
            # 後期：使用最小批次（更精確）
            target_size = self.min_batch_size

        # 限制在範圍內
        target_size = max(self.min_batch_size, min(self.max_batch_size, target_size))

        # 更新當前批次大小
        self._current_batch_size = target_size

        return target_size

    def _evaluate_batch(
        self,
        boards: List[GoBoard],
        colors: List[str]
    ) -> List[Tuple[Dict[Tuple[int, int], float], float]]:
        """批次評估多個棋盤

        Parameters
        ----------
        boards : List[GoBoard]
            棋盤列表
        colors : List[str]
            對應的輪走方

        Returns
        -------
        List[Tuple[Dict, float]]
            (policy, value) 列表；value 為輪走方視角
        """
        evaluations = self.evaluator.evaluate(boards, colors)
        return [(evaluation.policy_dict(), evaluation.value) for evaluation in evaluations]

    def search(
        self,
        board: GoBoard,
        color: str,
        reuse_tree: bool = False
    ) -> Tuple[np.ndarray, MCTSNodeOptimized]:
        """執行 MCTS 搜索（優化版）

        Parameters
        ----------
        board : GoBoard
            當前棋盤
        color : str
            當前玩家
        reuse_tree : bool
            是否重用搜索樹

        Returns
        -------
        Tuple[np.ndarray, MCTSNodeOptimized]
            (move_probs, root)
        """
        # 創建根節點
        board_hash = self._get_board_hash(board)
        root = MCTSNodeOptimized(board_hash=board_hash, color=_opponent(color))

        # 緩存根棋盤
        self.board_cache[board_hash] = board.copy()

        # 初始評估和擴展
        policy, value = self._evaluate_single(board, color)
        self._expand_node(root, board, policy)

        # 添加 Dirichlet 噪聲
        self._add_dirichlet_noise(root)

        # 批次搜索（支持動態批次大小）
        simulations_done = 0
        while simulations_done < self.num_simulations:
            # 計算本輪批次大小
            current_batch = self._get_dynamic_batch_size(
                simulations_done, self.num_simulations
            )
            batch_size = min(current_batch, self.num_simulations - simulations_done)

            # 選擇多個葉子節點（並行）
            leaves = []
            paths = []

            for _ in range(batch_size):
                path = [root]  # Include root in path for visit count
                node = root
                current_board = board.copy()

                # Selection
                while node.is_expanded and node.children:
                    node = node.select_child()
                    path.append(node)

                    # 應用移動
                    if node.move:
                        current_board.play_move(node.move[0], node.move[1], node.color)

                    # 虛擬損失
                    if self.use_virtual_loss:
                        node.virtual_loss += 1

                leaves.append((node, current_board))
                paths.append(path)

            # 批次評估
            boards_to_eval = [board for _, board in leaves]
            colors_to_eval = [node.to_play for node, _ in leaves]

            evaluations = self._evaluate_batch(boards_to_eval, colors_to_eval)

            # 擴展和回溯
            for (node, current_board), (policy, value), path in zip(leaves, evaluations, paths):
                # 擴展
                if not node.is_expanded:
                    self._expand_node(node, current_board, policy)

                # 回溯
                self._backup(path, value)

                # 移除虛擬損失（跳過根節點，因為根節點沒有應用虛擬損失）
                if self.use_virtual_loss:
                    for n in path[1:]:  # Skip root
                        n.virtual_loss -= 1

            # 更新已完成模擬數
            simulations_done += batch_size

        # 計算著手概率
        move_probs = self._get_move_probabilities(root, board.size)

        return move_probs, root

    def _evaluate_single(
        self,
        board: GoBoard,
        color: str
    ) -> Tuple[Dict[Tuple[int, int], float], float]:
        """評估單個棋盤（帶緩存；value 依輪走方而異，所以鍵含顏色）"""
        cache_key = (self._get_board_hash(board), color)

        if cache_key in self.evaluation_cache:
            return self.evaluation_cache[cache_key]

        # 實際評估
        results = self._evaluate_batch([board], [color])
        policy, value = results[0]

        # 緩存結果
        self.evaluation_cache[cache_key] = (policy, value)

        return policy, value

    def _expand_node(
        self,
        node: MCTSNodeOptimized,
        board: GoBoard,
        policy: Dict[Tuple[int, int], float]
    ) -> None:
        """擴展節點"""
        if node.is_expanded:
            return

        next_color = node.to_play
        legal_moves = board.get_legal_moves(next_color)

        for move in legal_moves:
            # Use uniform prior as fallback
            default_prior = 1.0 / (board.size * board.size + 1)
            prior = policy.get(move, default_prior)

            # 創建子節點（延遲棋盤複製）
            child_board = board.copy()
            child_board.play_move(move[0], move[1], next_color)
            child_hash = self._get_board_hash(child_board)

            child = MCTSNodeOptimized(
                board_hash=child_hash,
                parent=node,
                move=move,
                color=next_color,
                prior=prior
            )

            node.children[move] = child

            # 緩存棋盤
            if child_hash not in self.board_cache:
                self.board_cache[child_hash] = child_board

        node.is_expanded = True

    def _backup(self, path: List[MCTSNodeOptimized], value: float) -> None:
        """回溯更新

        ``value`` 為葉節點輪走方視角；每個節點存走進該節點一方的視角，
        所以葉節點加 -value，往上逐層變號。
        """
        value = -value
        for node in reversed(path):
            node.visit_count += 1
            node.total_value += value
            value = -value  # 對手視角

    def _add_dirichlet_noise(self, root: MCTSNodeOptimized) -> None:
        """添加 Dirichlet 噪聲"""
        if not root.children:
            return

        moves = list(root.children.keys())
        noise = np.random.dirichlet([self.dirichlet_alpha] * len(moves))

        for move, noise_value in zip(moves, noise):
            child = root.children[move]
            child.prior = (
                (1 - self.dirichlet_epsilon) * child.prior +
                self.dirichlet_epsilon * noise_value
            )

    def _get_move_probabilities(
        self,
        root: MCTSNodeOptimized,
        board_size: int
    ) -> np.ndarray:
        """計算著手概率"""
        move_probs = np.zeros(board_size * board_size + 1, dtype=np.float32)

        if self.temperature == 0:
            # 貪婪選擇
            best_move = max(root.children.items(), key=lambda x: x[1].visit_count)
            move = best_move[0]
            if move == 'pass':
                move_idx = board_size * board_size
            else:
                move_idx = move[0] * board_size + move[1]
            move_probs[move_idx] = 1.0
        else:
            # 溫度採樣
            visits = np.zeros(board_size * board_size + 1, dtype=np.float32)

            for move, child in root.children.items():
                if move == 'pass':
                    move_idx = board_size * board_size
                else:
                    move_idx = move[0] * board_size + move[1]
                visits[move_idx] = child.visit_count

            visits = visits ** (1.0 / self.temperature)

            if visits.sum() > 0:
                move_probs = visits / visits.sum()

        return move_probs

    def clear_cache(self) -> None:
        """清除緩存"""
        self.board_cache.clear()
        self.evaluation_cache.clear()

    def select_move(
        self,
        board: GoBoard,
        color: str,
        temperature: Optional[float] = None
    ) -> Tuple[int, int]:
        """選擇著手"""
        if temperature is not None:
            old_temp = self.temperature
            self.temperature = temperature

        move_probs, root = self.search(board, color)

        move_idx = np.random.choice(len(move_probs), p=move_probs)

        if temperature is not None:
            self.temperature = old_temp

        board_size = board.size
        if move_idx == board_size * board_size:
            legal_moves = board.get_legal_moves(color)
            if legal_moves:
                return legal_moves[0]
            return (0, 0)

        row = move_idx // board_size
        col = move_idx % board_size

        return (row, col)

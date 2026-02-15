"""Gumbel MCTS 實作

使用 Gumbel-Top-k 採樣和 Sequential Halving 的改進版 MCTS。
相比傳統 MCTS 需要 800+ 次模擬，Gumbel MCTS 只需 2-16 次即可達到相似品質。

參考論文：
- "Policy improvement by planning with Gumbel" (Danihelka et al., 2022)

核心改進：
1. 根節點使用 Gumbel-Top-k 採樣選擇候選動作
2. Sequential Halving 逐步淘汰較弱候選
3. 非根節點使用確定性選擇（完成已開始的動作）
4. 最小化簡單遺憾（Simple Regret）而非累積遺憾
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict, Union

import numpy as np
import torch

from core.game_rules import GoBoard
from core.sequential_halving import sequential_halving
from input.lightgo_encoder import LightGoEncoder


@dataclass
class GumbelMCTSNode:
    """Gumbel MCTS 節點

    Attributes
    ----------
    board : GoBoard
        當前棋盤狀態
    parent : Optional[GumbelMCTSNode]
        父節點
    move : Optional[Tuple[int, int]]
        到達此節點的動作
    color : str
        剛下完棋的顏色 ('b' or 'w')
    prior : float
        神經網路先驗機率
    gumbel_value : float
        Gumbel 雜訊值（僅根節點使用）
    is_root : bool
        是否為根節點
    """

    board: GoBoard
    parent: Optional['GumbelMCTSNode'] = None
    move: Optional[Tuple[int, int]] = None
    color: str = 'b'
    prior: float = 0.0
    gumbel_value: float = 0.0
    is_root: bool = False

    def __post_init__(self):
        """初始化可變屬性"""
        self.children: Dict[Union[Tuple[int, int], str], GumbelMCTSNode] = {}
        self.visit_count: int = 0
        self.total_value: float = 0.0
        self.value_estimate: float = 0.0  # 神經網路價值估計

    @property
    def q_value(self) -> float:
        """平均動作價值 (Q)"""
        if self.visit_count == 0:
            return 0.0
        return self.total_value / self.visit_count

    def is_leaf(self) -> bool:
        """是否為葉節點"""
        return len(self.children) == 0

    def select_child_deterministic(self) -> 'GumbelMCTSNode':
        """確定性選擇子節點（非根節點使用）

        選擇訪問次數最多的子節點，完成已承諾的搜尋路徑。
        """
        if not self.children:
            raise ValueError("無子節點可選擇")

        return max(self.children.values(), key=lambda n: n.visit_count)

    def expand(
        self,
        policy: Dict[Union[Tuple[int, int], str], float],
        value: float
    ) -> None:
        """擴展節點

        Parameters
        ----------
        policy : Dict
            策略機率分布
        value : float
            價值估計
        """
        self.value_estimate = value

        # 下一個玩家
        next_color = 'w' if self.color == 'b' else 'b'

        # 取得合法動作
        legal_moves = self.board.get_legal_moves(next_color)

        for move in legal_moves:
            if move in self.children:
                continue

            # 建立子棋盤
            child_board = self.board.copy()
            child_board.play_move(move[0], move[1], next_color)

            # 取得先驗機率
            default_prior = 1.0 / (self.board.size ** 2 + 1)
            prior = policy.get(move, default_prior)

            child = GumbelMCTSNode(
                board=child_board,
                parent=self,
                move=move,
                color=next_color,
                prior=prior,
                is_root=False
            )

            self.children[move] = child

    def update(self, value: float) -> None:
        """更新節點統計"""
        self.visit_count += 1
        self.total_value += value


class GumbelMCTS:
    """Gumbel MCTS 搜尋引擎

    使用 Gumbel-Top-k 採樣和 Sequential Halving 的改進版 MCTS。

    Parameters
    ----------
    model : torch.nn.Module
        神經網路模型（策略 + 價值）
    num_simulations : int
        每次搜尋的模擬次數（預設 16，可低至 2）
    num_sampled_actions : int
        Gumbel-Top-k 的 k 值（預設 16）
    c_visit : float
        未訪問懲罰係數（預設 50.0）
    c_scale : float
        Q 值縮放係數（預設 1.0）
    board_size : int
        棋盤大小
    """

    def __init__(
        self,
        model,
        num_simulations: int = 16,
        num_sampled_actions: int = 16,
        c_visit: float = 50.0,
        c_scale: float = 1.0,
        board_size: int = 19
    ):
        self.model = model
        self.num_simulations = num_simulations
        self.num_sampled_actions = num_sampled_actions
        self.c_visit = c_visit
        self.c_scale = c_scale
        self.board_size = board_size

        # 取得模型設備
        self.device = next(model.parameters()).device

        # LightGo 2-plane encoder
        self.encoder = LightGoEncoder(board_size=board_size)

    def search(
        self,
        board: GoBoard,
        color: str
    ) -> Tuple[np.ndarray, GumbelMCTSNode]:
        """執行 Gumbel MCTS 搜尋

        Parameters
        ----------
        board : GoBoard
            當前棋盤狀態
        color : str
            當前玩家顏色

        Returns
        -------
        Tuple[np.ndarray, GumbelMCTSNode]
            - 動作機率分布 (board_size^2 + 1)
            - 搜尋樹根節點
        """
        # 建立根節點
        root = GumbelMCTSNode(
            board=board.copy(),
            color=color,
            is_root=True
        )

        # Step 1: 取得神經網路策略和價值
        policy_logits, policy_dict, value = self._evaluate(board, color)

        # 儲存價值估計
        root.value_estimate = value

        # Step 2: Gumbel-Top-k 採樣候選動作
        candidates = self._gumbel_top_k(policy_logits, board, color)

        if not candidates:
            # 無合法動作，回傳 pass
            move_probs = np.zeros(self.board_size ** 2 + 1, dtype=np.float32)
            move_probs[-1] = 1.0  # pass
            return move_probs, root

        # 擴展根節點
        root.expand(policy_dict, value)

        # Step 3: Sequential Halving 選擇最佳動作
        def evaluate_action(action_idx: int) -> float:
            """評估單個動作"""
            move = self._idx_to_move(action_idx)

            if move not in root.children:
                return root.value_estimate

            child = root.children[move]

            # 執行一次模擬
            self._simulate(child)

            # 回傳已完成 Q 值
            return self._completed_q_value(root, move)

        # 執行 Sequential Halving
        best_action_idx = sequential_halving(
            evaluate_fn=evaluate_action,
            candidates=candidates,
            total_budget=self.num_simulations
        )

        # 建立動作機率分布
        move_probs = self._build_policy(root, best_action_idx)

        return move_probs, root

    def _evaluate(
        self,
        board: GoBoard,
        color: str
    ) -> Tuple[np.ndarray, Dict[Union[Tuple[int, int], str], float], float]:
        """評估局面

        Returns
        -------
        Tuple[np.ndarray, Dict, float]
            - 策略 logits (board_size^2 + 1)
            - 策略字典 {move: probability}
            - 價值估計
        """
        # 編碼棋盤
        if board.size != self.encoder.board_size:
            self.encoder = LightGoEncoder(board_size=board.size)

        features = self.encoder.encode(board, color)
        features_tensor = features.unsqueeze(0).to(self.device)

        # 神經網路推理
        with torch.no_grad():
            policy_logits_tensor, value_tensor = self.model(features_tensor)
            policy_logits = policy_logits_tensor[0].cpu().numpy()
            policy_probs = torch.softmax(policy_logits_tensor, dim=1)[0].cpu().numpy()

        # 建立策略字典
        policy_dict: Dict[Union[Tuple[int, int], str], float] = {}
        board_size = board.size

        for i in range(board_size * board_size):
            row = i // board_size
            col = i % board_size
            policy_dict[(row, col)] = float(policy_probs[i])

        policy_dict['pass'] = float(policy_probs[board_size * board_size])

        value = float(value_tensor[0].item())

        return policy_logits, policy_dict, value

    def _gumbel_top_k(
        self,
        policy_logits: np.ndarray,
        board: GoBoard,
        color: str
    ) -> List[int]:
        """Gumbel-Top-k 採樣

        Parameters
        ----------
        policy_logits : np.ndarray
            策略 logits
        board : GoBoard
            棋盤狀態
        color : str
            當前玩家顏色

        Returns
        -------
        List[int]
            候選動作索引列表
        """
        # 採樣 Gumbel(0, 1) 雜訊
        gumbel_noise = self._sample_gumbel(policy_logits.shape)

        # 計算 Gumbel 分數
        gumbel_scores = policy_logits + gumbel_noise

        # 取得合法動作
        legal_moves = board.get_legal_moves(color)

        # 建立合法動作索引列表
        legal_indices = []
        board_size = board.size

        for move in legal_moves:
            if isinstance(move, tuple):
                idx = move[0] * board_size + move[1]
                legal_indices.append(idx)

        # 加入 pass（如果允許）
        legal_indices.append(board_size * board_size)

        if not legal_indices:
            return []

        # 篩選合法動作的 Gumbel 分數
        legal_scores = [(idx, gumbel_scores[idx]) for idx in legal_indices]
        legal_scores.sort(key=lambda x: x[1], reverse=True)

        # 取 Top-k
        k = min(self.num_sampled_actions, len(legal_scores))
        top_k = [idx for idx, _ in legal_scores[:k]]

        return top_k

    def _sample_gumbel(self, shape) -> np.ndarray:
        """採樣 Gumbel(0, 1) 分布"""
        u = np.random.uniform(1e-20, 1.0, shape)
        return -np.log(-np.log(u))

    def _simulate(self, node: GumbelMCTSNode) -> float:
        """執行一次模擬

        Parameters
        ----------
        node : GumbelMCTSNode
            起始節點

        Returns
        -------
        float
            模擬結果價值
        """
        # 如果是葉節點，評估並擴展
        if node.is_leaf():
            _, policy_dict, value = self._evaluate(node.board, node.color)
            node.expand(policy_dict, value)
            self._backup(node, value)
            return value

        # 非葉節點：確定性選擇
        if node.children:
            child = node.select_child_deterministic()
            value = self._simulate(child)
            return value

        # 無子節點，使用價值估計
        return node.value_estimate

    def _backup(self, node: GumbelMCTSNode, value: float) -> None:
        """回傳價值

        Parameters
        ----------
        node : GumbelMCTSNode
            葉節點
        value : float
            價值估計
        """
        current = node

        while current is not None:
            current.update(value)
            value = -value  # 對手視角
            current = current.parent

    def _completed_q_value(
        self,
        node: GumbelMCTSNode,
        move: Union[Tuple[int, int], str]
    ) -> float:
        """計算「已完成」Q 值

        包含未訪問懲罰，用於 Sequential Halving 的動作比較。

        Parameters
        ----------
        node : GumbelMCTSNode
            父節點
        move : Union[Tuple[int, int], str]
            動作

        Returns
        -------
        float
            已完成 Q 值
        """
        child = node.children.get(move)

        if child is None or child.visit_count == 0:
            # 未訪問：使用價值估計
            return node.value_estimate

        # 已訪問：Q 值 + 未訪問懲罰
        if node.visit_count > 0 and child.visit_count > 0:
            penalty = self.c_visit * math.sqrt(
                math.log(node.visit_count + 1) / child.visit_count
            )
        else:
            penalty = 0.0

        return child.q_value * self.c_scale - penalty

    def _idx_to_move(self, idx: int) -> Union[Tuple[int, int], str]:
        """將索引轉換為動作"""
        board_size = self.board_size
        if idx == board_size * board_size:
            return 'pass'
        row = idx // board_size
        col = idx % board_size
        return (row, col)

    def _move_to_idx(self, move: Union[Tuple[int, int], str]) -> int:
        """將動作轉換為索引"""
        if move == 'pass':
            return self.board_size * self.board_size
        return move[0] * self.board_size + move[1]

    def _build_policy(
        self,
        root: GumbelMCTSNode,
        best_action_idx: int
    ) -> np.ndarray:
        """建立最終策略機率分布

        Parameters
        ----------
        root : GumbelMCTSNode
            根節點
        best_action_idx : int
            最佳動作索引

        Returns
        -------
        np.ndarray
            動作機率分布
        """
        move_probs = np.zeros(self.board_size ** 2 + 1, dtype=np.float32)

        # 根據訪問次數建立分布（或直接選擇最佳）
        total_visits = sum(
            child.visit_count for child in root.children.values()
        )

        if total_visits == 0:
            # 沒有訪問，直接選擇最佳
            move_probs[best_action_idx] = 1.0
        else:
            # 按訪問次數分配機率
            for move, child in root.children.items():
                idx = self._move_to_idx(move)
                move_probs[idx] = child.visit_count / total_visits

        return move_probs

    def select_move(
        self,
        board: GoBoard,
        color: str,
        temperature: float = 0.0
    ) -> Optional[Tuple[int, int]]:
        """選擇動作

        Parameters
        ----------
        board : GoBoard
            棋盤狀態
        color : str
            當前玩家
        temperature : float
            溫度參數（0 = 確定性選擇）

        Returns
        -------
        Optional[Tuple[int, int]]
            選擇的動作，None 表示 pass
        """
        move_probs, _ = self.search(board, color)

        if temperature == 0:
            # 確定性選擇
            move_idx = np.argmax(move_probs)
        else:
            # 溫度採樣
            adjusted_probs = move_probs ** (1.0 / temperature)
            adjusted_probs /= adjusted_probs.sum()
            move_idx = np.random.choice(len(move_probs), p=adjusted_probs)

        # 轉換為動作
        board_size = board.size
        if move_idx == board_size * board_size:
            return None  # pass

        row = move_idx // board_size
        col = move_idx % board_size

        return (row, col)

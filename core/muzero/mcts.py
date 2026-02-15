"""MuZero MCTS 實作

在隱藏狀態空間執行 MCTS 搜尋。
與傳統 MCTS 不同，MuZero MCTS 使用學習的動態模型來模擬狀態轉移，
而不是使用真實的遊戲規則。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict, Union

import numpy as np
import torch

from core.muzero.networks import MuZeroNetworks


@dataclass
class MuZeroNode:
    """MuZero MCTS 節點

    在隱藏狀態空間中的搜尋節點。

    Attributes
    ----------
    hidden_state : Optional[torch.Tensor]
        隱藏狀態表示
    reward : float
        到達此節點的即時獎勵
    prior : np.ndarray
        子節點的先驗機率
    parent : Optional[MuZeroNode]
        父節點
    action : Optional[int]
        從父節點到達此節點的動作
    """

    hidden_state: Optional[torch.Tensor] = None
    reward: float = 0.0
    prior: Optional[np.ndarray] = None
    parent: Optional['MuZeroNode'] = None
    action: Optional[int] = None

    def __post_init__(self):
        """初始化可變屬性"""
        self.children: Dict[int, MuZeroNode] = {}
        self.visit_count: int = 0
        self.total_value: float = 0.0
        self.value_sum: float = 0.0

    @property
    def q_value(self) -> float:
        """平均動作價值"""
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

    def is_expanded(self) -> bool:
        """是否已擴展"""
        return len(self.children) > 0

    def expand(self, prior: np.ndarray) -> None:
        """擴展節點

        Parameters
        ----------
        prior : np.ndarray
            子節點的先驗機率
        """
        self.prior = prior

        for action in range(len(prior)):
            if prior[action] > 0:
                self.children[action] = MuZeroNode(
                    parent=self,
                    action=action
                )

    def select_child(self, c_puct: float = 1.4) -> Tuple[int, 'MuZeroNode']:
        """使用 UCB 選擇子節點

        Parameters
        ----------
        c_puct : float
            探索常數

        Returns
        -------
        Tuple[int, MuZeroNode]
            選中的動作和子節點
        """
        # 使用 max(visit_count, 1) 避免第一次選擇時所有 UCB 為 0 的問題
        parent_visits = max(self.visit_count, 1)

        # 計算所有子節點的 UCB 分數
        scores = []
        actions = []
        children = []

        for action, child in self.children.items():
            prior = self.prior[action] if self.prior is not None else 1.0 / len(self.children)
            u_value = c_puct * prior * math.sqrt(parent_visits) / (1 + child.visit_count)
            score = child.q_value + u_value
            scores.append(score)
            actions.append(action)
            children.append(child)

        # 找出最高分並添加小量隨機雜訊打破平手
        scores = np.array(scores)
        # 添加微小雜訊避免完全相同分數時總是選擇第一個
        noise = np.random.uniform(0, 1e-8, len(scores))
        scores_with_noise = scores + noise

        best_idx = np.argmax(scores_with_noise)
        return actions[best_idx], children[best_idx]

    def update(self, value: float) -> None:
        """更新節點統計"""
        self.visit_count += 1
        self.value_sum += value


class MuZeroMCTS:
    """MuZero MCTS 搜尋引擎

    在隱藏狀態空間執行 MCTS 搜尋。

    Parameters
    ----------
    networks : MuZeroNetworks
        MuZero 網路
    num_simulations : int
        每次搜尋的模擬次數
    discount : float
        折扣因子
    c_puct : float
        UCB 探索常數
    dirichlet_alpha : float
        Dirichlet 雜訊 alpha
    dirichlet_epsilon : float
        Dirichlet 雜訊比例
    use_gumbel : bool
        是否使用 Gumbel MCTS（預設 False）
    """

    def __init__(
        self,
        networks: MuZeroNetworks,
        num_simulations: int = 50,
        discount: float = 0.997,
        c_puct: float = 1.4,
        dirichlet_alpha: float = 0.3,
        dirichlet_epsilon: float = 0.25,
        use_gumbel: bool = False
    ):
        self.networks = networks
        self.num_simulations = num_simulations
        self.discount = discount
        self.c_puct = c_puct
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_epsilon = dirichlet_epsilon
        self.use_gumbel = use_gumbel

        # 獲取設備
        self.device = next(networks.parameters()).device

    def search(
        self,
        observation: torch.Tensor,
        add_exploration_noise: bool = True
    ) -> Tuple[np.ndarray, MuZeroNode, float]:
        """執行 MCTS 搜尋

        Parameters
        ----------
        observation : torch.Tensor
            棋盤觀察 (1, num_input_planes, board_size, board_size)
        add_exploration_noise : bool
            是否在根節點添加探索雜訊

        Returns
        -------
        Tuple[np.ndarray, MuZeroNode, float]
            - 動作機率分布
            - 根節點
            - 價值估計
        """
        # 確保在正確的設備上
        observation = observation.to(self.device)

        # 初始推理
        with torch.no_grad():
            hidden_state, policy_logits, value = self.networks.initial_inference(observation)

        # 轉換為機率
        policy = torch.softmax(policy_logits, dim=1)[0].cpu().numpy()
        root_value = value[0].item()

        # 建立根節點
        root = MuZeroNode(hidden_state=hidden_state)

        # 添加 Dirichlet 雜訊
        if add_exploration_noise:
            noise = np.random.dirichlet([self.dirichlet_alpha] * len(policy))
            policy = (1 - self.dirichlet_epsilon) * policy + self.dirichlet_epsilon * noise

        # 擴展根節點
        root.expand(policy)

        # 執行模擬
        for _ in range(self.num_simulations):
            node = root
            search_path = [node]

            # Selection: 選擇到葉節點
            while node.is_expanded() and node.hidden_state is not None:
                action, child = node.select_child(self.c_puct)
                search_path.append(child)
                node = child

            # Expansion: 如果節點有父節點且尚未有隱藏狀態
            parent = node.parent
            if parent is not None and parent.hidden_state is not None and node.action is not None:
                with torch.no_grad():
                    action_tensor = torch.tensor([node.action], device=self.device)
                    next_hidden, reward, policy_logits, value = self.networks.recurrent_inference(
                        parent.hidden_state, action_tensor
                    )

                node.hidden_state = next_hidden
                node.reward = reward[0].item()

                policy = torch.softmax(policy_logits, dim=1)[0].cpu().numpy()
                node.expand(policy)

                value_estimate = value[0].item()
            else:
                value_estimate = 0.0

            # Backup: 回傳價值
            self._backup(search_path, value_estimate)

        # 計算動作機率
        move_probs = self._get_policy(root)

        return move_probs, root, root_value

    def _backup(self, search_path: List[MuZeroNode], value: float) -> None:
        """回傳價值

        Parameters
        ----------
        search_path : List[MuZeroNode]
            搜尋路徑
        value : float
            葉節點價值
        """
        for node in reversed(search_path):
            node.update(value)
            # 折扣 + 獎勵
            value = node.reward + self.discount * value

    def _get_policy(self, root: MuZeroNode, temperature: float = 1.0) -> np.ndarray:
        """從訪問次數計算策略

        Parameters
        ----------
        root : MuZeroNode
            根節點
        temperature : float
            溫度參數

        Returns
        -------
        np.ndarray
            動作機率分布
        """
        num_actions = self.networks.num_actions
        visits = np.zeros(num_actions, dtype=np.float32)

        for action, child in root.children.items():
            visits[action] = child.visit_count

        if temperature == 0:
            # 確定性選擇
            best_action = np.argmax(visits)
            probs = np.zeros(num_actions, dtype=np.float32)
            probs[best_action] = 1.0
            return probs

        # 溫度採樣
        visits = visits ** (1.0 / temperature)
        total = visits.sum()

        if total > 0:
            return visits / total
        else:
            return np.ones(num_actions, dtype=np.float32) / num_actions

    def select_action(
        self,
        observation: torch.Tensor,
        temperature: float = 1.0,
        add_exploration_noise: bool = True
    ) -> Tuple[int, np.ndarray, float]:
        """選擇動作

        Parameters
        ----------
        observation : torch.Tensor
            棋盤觀察
        temperature : float
            溫度參數
        add_exploration_noise : bool
            是否添加探索雜訊

        Returns
        -------
        Tuple[int, np.ndarray, float]
            - 選擇的動作
            - 策略分布
            - 價值估計
        """
        policy, root, value = self.search(observation, add_exploration_noise)

        if temperature == 0:
            action = np.argmax(policy)
        else:
            action = np.random.choice(len(policy), p=policy)

        return action, policy, value

    def get_action_value(self, root: MuZeroNode, action: int) -> float:
        """獲取特定動作的 Q 值

        Parameters
        ----------
        root : MuZeroNode
            根節點
        action : int
            動作

        Returns
        -------
        float
            Q 值
        """
        if action in root.children:
            return root.children[action].q_value
        return 0.0

"""MuZero 軌跡重放緩衝區

儲存完整的遊戲軌跡，用於 K 步展開訓練。
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import numpy as np


@dataclass
class Trajectory:
    """單場遊戲軌跡

    Attributes
    ----------
    observations : List[np.ndarray]
        每一步的觀察
    actions : List[int]
        每一步的動作
    rewards : List[float]
        每一步的獎勵
    policies : List[np.ndarray]
        每一步 MCTS 產生的策略
    values : List[float]
        每一步的價值估計
    game_result : float
        遊戲結果 (+1 勝, -1 負, 0 平)
    """

    observations: List[np.ndarray] = field(default_factory=list)
    actions: List[int] = field(default_factory=list)
    rewards: List[float] = field(default_factory=list)
    policies: List[np.ndarray] = field(default_factory=list)
    values: List[float] = field(default_factory=list)
    game_result: float = 0.0

    def __len__(self) -> int:
        return len(self.observations)

    def add_step(
        self,
        observation: np.ndarray,
        action: int,
        reward: float,
        policy: np.ndarray,
        value: float
    ) -> None:
        """添加一步"""
        self.observations.append(observation)
        self.actions.append(action)
        self.rewards.append(reward)
        self.policies.append(policy)
        self.values.append(value)

    def compute_target_values(
        self,
        td_steps: int = 10,
        discount: float = 0.997
    ) -> List[float]:
        """計算 TD(n) 目標價值

        Parameters
        ----------
        td_steps : int
            TD 步數
        discount : float
            折扣因子

        Returns
        -------
        List[float]
            每一步的目標價值
        """
        n = len(self)
        target_values = []

        for i in range(n):
            # Bootstrap 值
            bootstrap_idx = min(i + td_steps, n - 1)

            if bootstrap_idx < n - 1:
                # 使用 MCTS 價值估計
                bootstrap_value = self.values[bootstrap_idx]
            else:
                # 使用遊戲結果
                bootstrap_value = self.game_result

            # 累積折扣獎勵
            value = bootstrap_value * (discount ** td_steps)

            for j in range(min(td_steps, n - i)):
                reward = self.rewards[i + j] if i + j < len(self.rewards) else 0.0
                value += reward * (discount ** j)

            target_values.append(value)

        return target_values


class MuZeroReplayBuffer:
    """MuZero 重放緩衝區

    Parameters
    ----------
    max_size : int
        最大軌跡數量
    td_steps : int
        TD 步數
    discount : float
        折扣因子
    """

    def __init__(
        self,
        max_size: int = 10000,
        td_steps: int = 10,
        discount: float = 0.997
    ):
        self.max_size = max_size
        self.td_steps = td_steps
        self.discount = discount
        self.buffer: List[Trajectory] = []
        self.total_samples = 0

    def add_trajectory(self, trajectory: Trajectory) -> None:
        """添加軌跡

        Parameters
        ----------
        trajectory : Trajectory
            遊戲軌跡
        """
        if len(self.buffer) >= self.max_size:
            # 移除最舊的軌跡
            removed = self.buffer.pop(0)
            self.total_samples -= len(removed)

        self.buffer.append(trajectory)
        self.total_samples += len(trajectory)

    def sample_batch(
        self,
        batch_size: int,
        unroll_steps: int = 5
    ) -> Dict[str, np.ndarray]:
        """採樣訓練批次

        Parameters
        ----------
        batch_size : int
            批次大小
        unroll_steps : int
            展開步數

        Returns
        -------
        Dict[str, np.ndarray]
            訓練批次，包含：
            - observations: (batch, C, H, W)
            - actions: (batch, unroll_steps)
            - target_values: (batch, unroll_steps + 1)
            - target_rewards: (batch, unroll_steps)
            - target_policies: (batch, unroll_steps + 1, num_actions)
        """
        batch_observations = []
        batch_actions = []
        batch_target_values = []
        batch_target_rewards = []
        batch_target_policies = []

        for _ in range(batch_size):
            # 隨機選擇軌跡和起始位置
            trajectory = random.choice(self.buffer)
            max_start = max(0, len(trajectory) - unroll_steps - 1)
            start_idx = random.randint(0, max_start) if max_start > 0 else 0

            # 計算目標價值
            target_values = trajectory.compute_target_values(
                td_steps=self.td_steps,
                discount=self.discount
            )

            # 提取樣本
            observation = trajectory.observations[start_idx]

            actions = []
            values = []
            rewards = []
            policies = []

            # 初始策略和價值
            values.append(target_values[start_idx])
            policies.append(trajectory.policies[start_idx])

            # K 步展開
            for k in range(unroll_steps):
                idx = start_idx + k
                if idx < len(trajectory) - 1:
                    actions.append(trajectory.actions[idx])
                    rewards.append(trajectory.rewards[idx])
                    values.append(target_values[idx + 1])
                    policies.append(trajectory.policies[idx + 1])
                else:
                    # 超出軌跡長度，填充
                    actions.append(0)
                    rewards.append(0.0)
                    values.append(0.0)
                    policies.append(trajectory.policies[-1])

            batch_observations.append(observation)
            batch_actions.append(actions)
            batch_target_values.append(values)
            batch_target_rewards.append(rewards)
            batch_target_policies.append(policies)

        return {
            'observations': np.array(batch_observations),
            'actions': np.array(batch_actions),
            'target_values': np.array(batch_target_values),
            'target_rewards': np.array(batch_target_rewards),
            'target_policies': np.array(batch_target_policies)
        }

    def __len__(self) -> int:
        return len(self.buffer)

    @property
    def num_samples(self) -> int:
        """總樣本數（所有軌跡的步數總和）"""
        return self.total_samples

    def clear(self) -> None:
        """清空緩衝區"""
        self.buffer.clear()
        self.total_samples = 0

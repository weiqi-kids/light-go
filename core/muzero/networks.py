"""MuZero 三大網路

實作 MuZero 的核心網路架構：
- h (representation): 觀察 → 隱藏狀態
- g (dynamics): (隱藏狀態, 動作) → (下一隱藏狀態, 即時獎勵)
- f (prediction): 隱藏狀態 → (策略, 價值)
"""

from __future__ import annotations

from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    """殘差塊"""

    def __init__(self, num_filters: int):
        super().__init__()
        self.conv1 = nn.Conv2d(num_filters, num_filters, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_filters)
        self.conv2 = nn.Conv2d(num_filters, num_filters, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(num_filters)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return F.relu(out)


class RepresentationNetwork(nn.Module):
    """表徵網路 h: 觀察 → 隱藏狀態

    將棋盤觀察編碼為固定大小的隱藏狀態表示。
    """

    def __init__(
        self,
        num_input_planes: int,
        hidden_size: int,
        num_blocks: int,
        board_size: int
    ):
        super().__init__()
        self.board_size = board_size

        # 初始卷積
        self.conv_in = nn.Conv2d(num_input_planes, hidden_size, 3, padding=1, bias=False)
        self.bn_in = nn.BatchNorm2d(hidden_size)

        # 殘差塊
        self.res_blocks = nn.ModuleList([
            ResBlock(hidden_size) for _ in range(num_blocks)
        ])

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        observation : torch.Tensor
            棋盤觀察 (batch, num_input_planes, board_size, board_size)

        Returns
        -------
        torch.Tensor
            隱藏狀態 (batch, hidden_size, board_size, board_size)
        """
        x = F.relu(self.bn_in(self.conv_in(observation)))

        for block in self.res_blocks:
            x = block(x)

        return x


class DynamicsNetwork(nn.Module):
    """動態網路 g: (隱藏狀態, 動作) → (下一隱藏狀態, 即時獎勵)

    模擬環境的狀態轉移。
    """

    def __init__(
        self,
        hidden_size: int,
        num_blocks: int,
        board_size: int,
        num_actions: int
    ):
        super().__init__()
        self.board_size = board_size
        self.num_actions = num_actions
        self.hidden_size = hidden_size

        # 動作編碼層：將 one-hot 動作轉換為空間特徵
        # 輸入：hidden_size + num_actions（action plane）
        self.conv_in = nn.Conv2d(
            hidden_size + num_actions,
            hidden_size,
            3,
            padding=1,
            bias=False
        )
        self.bn_in = nn.BatchNorm2d(hidden_size)

        # 殘差塊
        self.res_blocks = nn.ModuleList([
            ResBlock(hidden_size) for _ in range(num_blocks)
        ])

        # 獎勵頭
        self.reward_conv = nn.Conv2d(hidden_size, 1, 1, bias=False)
        self.reward_bn = nn.BatchNorm2d(1)
        self.reward_fc = nn.Linear(board_size * board_size, 256)
        self.reward_out = nn.Linear(256, 1)

    def forward(
        self,
        hidden_state: torch.Tensor,
        action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        hidden_state : torch.Tensor
            當前隱藏狀態 (batch, hidden_size, board_size, board_size)
        action : torch.Tensor
            動作索引 (batch,) - 會被編碼為空間平面

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            - 下一隱藏狀態 (batch, hidden_size, board_size, board_size)
            - 即時獎勵 (batch, 1)
        """
        batch_size = hidden_state.shape[0]

        # 將動作編碼為空間平面
        action_plane = self._encode_action(action, batch_size, hidden_state.device)

        # 合併隱藏狀態和動作
        x = torch.cat([hidden_state, action_plane], dim=1)

        # 動態網路主體
        x = F.relu(self.bn_in(self.conv_in(x)))

        for block in self.res_blocks:
            x = block(x)

        next_hidden = x

        # 獎勵預測
        reward = F.relu(self.reward_bn(self.reward_conv(x)))
        reward = reward.view(batch_size, -1)
        reward = F.relu(self.reward_fc(reward))
        reward = torch.tanh(self.reward_out(reward))

        return next_hidden, reward

    def _encode_action(
        self,
        action: torch.Tensor,
        batch_size: int,
        device: torch.device
    ) -> torch.Tensor:
        """將動作編碼為空間平面

        每個動作對應一個平面，平面上對應位置為 1。
        """
        # 建立 one-hot 動作編碼
        action_onehot = F.one_hot(action.long(), num_classes=self.num_actions)
        action_onehot = action_onehot.float()

        # 擴展為空間維度 (batch, num_actions, board_size, board_size)
        action_plane = action_onehot.view(batch_size, self.num_actions, 1, 1)
        action_plane = action_plane.expand(-1, -1, self.board_size, self.board_size)

        return action_plane


class PredictionNetwork(nn.Module):
    """預測網路 f: 隱藏狀態 → (策略, 價值)

    從隱藏狀態預測策略和價值。
    """

    def __init__(
        self,
        hidden_size: int,
        board_size: int,
        num_actions: int
    ):
        super().__init__()
        self.board_size = board_size
        self.num_actions = num_actions

        # 策略頭
        self.policy_conv = nn.Conv2d(hidden_size, 32, 1, bias=False)
        self.policy_bn = nn.BatchNorm2d(32)
        self.policy_fc = nn.Linear(32 * board_size * board_size, num_actions)

        # 價值頭
        self.value_conv = nn.Conv2d(hidden_size, 1, 1, bias=False)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(board_size * board_size, 256)
        self.value_fc2 = nn.Linear(256, 1)

    def forward(self, hidden_state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        hidden_state : torch.Tensor
            隱藏狀態 (batch, hidden_size, board_size, board_size)

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            - 策略 logits (batch, num_actions)
            - 價值 (batch, 1)
        """
        batch_size = hidden_state.shape[0]

        # 策略
        policy = F.relu(self.policy_bn(self.policy_conv(hidden_state)))
        policy = policy.view(batch_size, -1)
        policy = self.policy_fc(policy)

        # 價值
        value = F.relu(self.value_bn(self.value_conv(hidden_state)))
        value = value.view(batch_size, -1)
        value = F.relu(self.value_fc1(value))
        value = torch.tanh(self.value_fc2(value))

        return policy, value


class MuZeroNetworks(nn.Module):
    """MuZero 完整網路

    整合三個子網路：
    - representation (h): 觀察 → 隱藏狀態
    - dynamics (g): (隱藏狀態, 動作) → (下一隱藏狀態, 獎勵)
    - prediction (f): 隱藏狀態 → (策略, 價值)
    """

    def __init__(
        self,
        board_size: int = 19,
        num_input_planes: int = 2,
        hidden_size: int = 256,
        num_blocks: int = 16
    ):
        super().__init__()
        self.board_size = board_size
        self.num_input_planes = num_input_planes
        self.hidden_size = hidden_size
        self.num_actions = board_size * board_size + 1  # +1 for pass

        # 三大網路
        self.representation = RepresentationNetwork(
            num_input_planes=num_input_planes,
            hidden_size=hidden_size,
            num_blocks=num_blocks // 2,
            board_size=board_size
        )

        self.dynamics = DynamicsNetwork(
            hidden_size=hidden_size,
            num_blocks=num_blocks // 4,
            board_size=board_size,
            num_actions=self.num_actions
        )

        self.prediction = PredictionNetwork(
            hidden_size=hidden_size,
            board_size=board_size,
            num_actions=self.num_actions
        )

    def initial_inference(
        self,
        observation: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """初始推理：觀察 → (隱藏狀態, 策略, 價值)

        Parameters
        ----------
        observation : torch.Tensor
            棋盤觀察 (batch, num_input_planes, board_size, board_size)

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            - 隱藏狀態 (batch, hidden_size, board_size, board_size)
            - 策略 logits (batch, num_actions)
            - 價值 (batch, 1)
        """
        hidden_state = self.representation(observation)
        policy_logits, value = self.prediction(hidden_state)
        return hidden_state, policy_logits, value

    def recurrent_inference(
        self,
        hidden_state: torch.Tensor,
        action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """遞迴推理：(隱藏狀態, 動作) → (下一隱藏狀態, 獎勵, 策略, 價值)

        Parameters
        ----------
        hidden_state : torch.Tensor
            當前隱藏狀態
        action : torch.Tensor
            動作索引 (batch,)

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
            - 下一隱藏狀態
            - 即時獎勵 (batch, 1)
            - 策略 logits (batch, num_actions)
            - 價值 (batch, 1)
        """
        next_hidden, reward = self.dynamics(hidden_state, action)
        policy_logits, value = self.prediction(next_hidden)
        return next_hidden, reward, policy_logits, value

    def forward(self, observation: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """標準前向傳遞（相容 AlphaZero 風格介面）

        Parameters
        ----------
        observation : torch.Tensor
            棋盤觀察

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            - 策略 logits
            - 價值
        """
        _, policy_logits, value = self.initial_inference(observation)
        return policy_logits, value

    def get_param_count(self) -> int:
        """回傳參數總數"""
        return sum(p.numel() for p in self.parameters())

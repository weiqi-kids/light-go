"""MuZero 訓練器

實作 K 步展開訓練，同時訓練三個網路（h, g, f）。
支援：
- 監督預訓練（僅 policy + value）
- 混合精度訓練（fp16）
- Gradient accumulation
- 學習率調度（WarmupCosine, Step, Cosine 等）
"""

from __future__ import annotations

from typing import Dict, Optional, List, Iterator, Any
import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from core.muzero.networks import MuZeroNetworks
from core.muzero.replay_buffer import MuZeroReplayBuffer
from core.learning_scheduler import (
    GoTrainingScheduler,
    WarmupCosineScheduler,
    StepScheduler,
    CosineScheduler,
    create_scheduler
)

logger = logging.getLogger(__name__)


class SupervisedDataset(Dataset):
    """監督學習資料集

    用於 Phase 1 監督預訓練，從 NPZ 檔案載入資料。
    """

    def __init__(self, features: np.ndarray, policies: np.ndarray, values: np.ndarray):
        """
        Parameters
        ----------
        features : np.ndarray
            輸入特徵 (N, C, H, W)
        policies : np.ndarray
            目標策略 (N, num_actions)
        values : np.ndarray
            目標價值 (N,)
        """
        self.features = features.astype(np.float32)
        self.policies = policies.astype(np.float32)
        self.values = values.astype(np.float32)

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            'features': torch.tensor(self.features[idx]),
            'policy': torch.tensor(self.policies[idx]),
            'value': torch.tensor(self.values[idx])
        }


class MuZeroTrainer:
    """MuZero 訓練器

    使用 K 步展開訓練 MuZero 網路。

    Parameters
    ----------
    networks : MuZeroNetworks
        MuZero 網路
    replay_buffer : MuZeroReplayBuffer
        重放緩衝區
    unroll_steps : int
        展開步數
    learning_rate : float
        學習率
    weight_decay : float
        權重衰減
    value_loss_weight : float
        價值損失權重
    reward_loss_weight : float
        獎勵損失權重
    device : str
        運算設備
    scheduler_config : Optional[Dict]
        學習率調度器配置，例如：
        {'type': 'warmup_cosine', 'warmup_steps': 1000, 'total_steps': 100000}
    """

    def __init__(
        self,
        networks: MuZeroNetworks,
        replay_buffer: Optional[MuZeroReplayBuffer] = None,
        unroll_steps: int = 5,
        learning_rate: float = 0.001,
        weight_decay: float = 1e-4,
        value_loss_weight: float = 1.0,
        reward_loss_weight: float = 1.0,
        device: str = 'cpu',
        mixed_precision: bool = False,
        gradient_accumulation_steps: int = 1,
        scheduler_config: Optional[Dict[str, Any]] = None
    ):
        self.networks = networks
        self.replay_buffer = replay_buffer if replay_buffer is not None else MuZeroReplayBuffer()
        self.unroll_steps = unroll_steps
        self.value_loss_weight = value_loss_weight
        self.reward_loss_weight = reward_loss_weight
        self.device = device
        self.mixed_precision = mixed_precision
        self.gradient_accumulation_steps = gradient_accumulation_steps

        # 優化器
        self.optimizer = torch.optim.Adam(
            networks.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )

        # 學習率調度器
        self.scheduler: Optional[GoTrainingScheduler] = None
        if scheduler_config is not None:
            # 確保 initial_lr 設定正確
            if 'initial_lr' not in scheduler_config:
                scheduler_config['initial_lr'] = learning_rate
            self.scheduler = create_scheduler(self.optimizer, scheduler_config)
            logger.info(f"學習率調度器已啟用: {scheduler_config.get('type', 'step')}")

        # 混合精度支援
        self.scaler = None
        if mixed_precision:
            # MPS 使用 autocast，CUDA 使用 GradScaler
            if device == 'cuda':
                self.scaler = torch.cuda.amp.GradScaler()
            logger.info("混合精度訓練已啟用")

        # 將網路移至設備
        self.networks.to(device)

        # 訓練統計
        self.global_step = 0

    def train_step(self, batch: Dict[str, np.ndarray]) -> Dict[str, float]:
        """執行單步訓練

        Parameters
        ----------
        batch : Dict[str, np.ndarray]
            訓練批次，包含：
            - observations: (batch, C, H, W)
            - actions: (batch, K)
            - target_values: (batch, K+1)
            - target_rewards: (batch, K)
            - target_policies: (batch, K+1, num_actions)

        Returns
        -------
        Dict[str, float]
            損失字典
        """
        self.networks.train()

        # 轉換為張量
        observations = torch.tensor(
            batch['observations'], dtype=torch.float32, device=self.device
        )
        actions = torch.tensor(
            batch['actions'], dtype=torch.long, device=self.device
        )
        target_values = torch.tensor(
            batch['target_values'], dtype=torch.float32, device=self.device
        )
        target_rewards = torch.tensor(
            batch['target_rewards'], dtype=torch.float32, device=self.device
        )
        target_policies = torch.tensor(
            batch['target_policies'], dtype=torch.float32, device=self.device
        )

        # 初始推理
        hidden_state, policy_logits, value = self.networks.initial_inference(observations)

        # 計算初始損失
        policy_loss = self._policy_loss(policy_logits, target_policies[:, 0])
        value_loss = self._value_loss(value.squeeze(-1), target_values[:, 0])
        reward_loss = torch.tensor(0.0, device=self.device)

        # K 步展開
        for k in range(self.unroll_steps):
            # 遞迴推理
            hidden_state, reward, policy_logits, value = self.networks.recurrent_inference(
                hidden_state, actions[:, k]
            )

            # 累積損失
            policy_loss += self._policy_loss(policy_logits, target_policies[:, k + 1])
            value_loss += self._value_loss(value.squeeze(-1), target_values[:, k + 1])
            reward_loss += self._value_loss(reward.squeeze(-1), target_rewards[:, k])

        # 總損失
        total_loss = (
            policy_loss +
            self.value_loss_weight * value_loss +
            self.reward_loss_weight * reward_loss
        )

        # 反向傳播
        self.optimizer.zero_grad()
        total_loss.backward()

        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(self.networks.parameters(), max_norm=1.0)

        self.optimizer.step()

        return {
            'total_loss': total_loss.item(),
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'reward_loss': reward_loss.item()
        }

    def _policy_loss(
        self,
        policy_logits: torch.Tensor,
        target_policy: torch.Tensor
    ) -> torch.Tensor:
        """計算策略損失（交叉熵）

        Parameters
        ----------
        policy_logits : torch.Tensor
            預測的策略 logits (batch, num_actions)
        target_policy : torch.Tensor
            目標策略分布 (batch, num_actions)

        Returns
        -------
        torch.Tensor
            策略損失
        """
        # 使用 KL 散度或交叉熵
        log_probs = F.log_softmax(policy_logits, dim=-1)
        loss = -torch.sum(target_policy * log_probs, dim=-1)
        return loss.mean()

    def _value_loss(
        self,
        predicted: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """計算價值/獎勵損失（MSE）

        Parameters
        ----------
        predicted : torch.Tensor
            預測值 (batch,)
        target : torch.Tensor
            目標值 (batch,)

        Returns
        -------
        torch.Tensor
            損失
        """
        return F.mse_loss(predicted, target)

    def train_epoch(
        self,
        batch_size: int = 32,
        num_batches: int = 100
    ) -> Dict[str, float]:
        """訓練一個 epoch

        Parameters
        ----------
        batch_size : int
            批次大小
        num_batches : int
            批次數量

        Returns
        -------
        Dict[str, float]
            平均損失
        """
        if len(self.replay_buffer) == 0:
            logger.warning("重放緩衝區為空，跳過訓練")
            return {
                'total_loss': 0.0,
                'policy_loss': 0.0,
                'value_loss': 0.0,
                'reward_loss': 0.0
            }

        total_losses = []
        policy_losses = []
        value_losses = []
        reward_losses = []

        for _ in range(num_batches):
            batch = self.replay_buffer.sample_batch(
                batch_size=batch_size,
                unroll_steps=self.unroll_steps
            )
            losses = self.train_step(batch)

            total_losses.append(losses['total_loss'])
            policy_losses.append(losses['policy_loss'])
            value_losses.append(losses['value_loss'])
            reward_losses.append(losses['reward_loss'])

        return {
            'total_loss': np.mean(total_losses),
            'policy_loss': np.mean(policy_losses),
            'value_loss': np.mean(value_losses),
            'reward_loss': np.mean(reward_losses)
        }

    def evaluate(self, batch: Dict[str, np.ndarray]) -> Dict[str, float]:
        """評估模式

        Parameters
        ----------
        batch : Dict[str, np.ndarray]
            評估批次

        Returns
        -------
        Dict[str, float]
            損失字典
        """
        self.networks.eval()

        with torch.no_grad():
            # 轉換為張量
            observations = torch.tensor(
                batch['observations'], dtype=torch.float32, device=self.device
            )
            actions = torch.tensor(
                batch['actions'], dtype=torch.long, device=self.device
            )
            target_values = torch.tensor(
                batch['target_values'], dtype=torch.float32, device=self.device
            )
            target_rewards = torch.tensor(
                batch['target_rewards'], dtype=torch.float32, device=self.device
            )
            target_policies = torch.tensor(
                batch['target_policies'], dtype=torch.float32, device=self.device
            )

            # 初始推理
            hidden_state, policy_logits, value = self.networks.initial_inference(observations)

            # 計算初始損失
            policy_loss = self._policy_loss(policy_logits, target_policies[:, 0])
            value_loss = self._value_loss(value.squeeze(-1), target_values[:, 0])
            reward_loss = torch.tensor(0.0, device=self.device)

            # K 步展開
            for k in range(self.unroll_steps):
                hidden_state, reward, policy_logits, value = self.networks.recurrent_inference(
                    hidden_state, actions[:, k]
                )
                policy_loss += self._policy_loss(policy_logits, target_policies[:, k + 1])
                value_loss += self._value_loss(value.squeeze(-1), target_values[:, k + 1])
                reward_loss += self._value_loss(reward.squeeze(-1), target_rewards[:, k])

            total_loss = (
                policy_loss +
                self.value_loss_weight * value_loss +
                self.reward_loss_weight * reward_loss
            )

        return {
            'total_loss': total_loss.item(),
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'reward_loss': reward_loss.item()
        }

    def save_checkpoint(self, path: str, batch_size: Optional[int] = None) -> None:
        """儲存檢查點

        Parameters
        ----------
        path : str
            儲存路徑
        batch_size : Optional[int]
            訓練時使用的 batch size（用於恢復驗證）
        """
        from datetime import datetime

        # 獲取當前學習率
        current_lr = self.optimizer.param_groups[0]['lr']

        checkpoint = {
            'networks_state': self.networks.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'unroll_steps': self.unroll_steps,
            'value_loss_weight': self.value_loss_weight,
            'reward_loss_weight': self.reward_loss_weight,
            'global_step': self.global_step,
            'epoch': getattr(self, 'current_epoch', 0),
            # 學習率調度器狀態
            'scheduler_state': self.scheduler.state_dict() if self.scheduler else None,
            # 新增元數據
            'metadata': {
                'learning_rate': current_lr,
                'batch_size': batch_size,
                'gradient_accumulation_steps': self.gradient_accumulation_steps,
                'device': str(self.device),
                'timestamp': datetime.now().isoformat(),
                'mixed_precision': self.mixed_precision,
            }
        }
        torch.save(checkpoint, path)
        logger.info(f"檢查點已儲存至 {path} (epoch: {checkpoint['epoch']}, lr: {current_lr:.6f})")

    def load_checkpoint(self, path: str, expected_batch_size: Optional[int] = None) -> None:
        """載入檢查點

        Parameters
        ----------
        path : str
            檢查點路徑
        expected_batch_size : Optional[int]
            預期的 batch size（用於驗證）

        Raises
        ------
        FileNotFoundError
            如果檔案不存在
        KeyError
            如果檢查點缺少必要的鍵
        """
        import os

        if not os.path.exists(path):
            raise FileNotFoundError(f"檢查點不存在: {path}")

        checkpoint = torch.load(path, map_location=self.device, weights_only=False)

        # 驗證必要的鍵
        required_keys = ['networks_state', 'optimizer_state']
        for key in required_keys:
            if key not in checkpoint:
                raise KeyError(f"檢查點缺少必要的鍵: {key}")

        # 載入狀態
        self.networks.load_state_dict(checkpoint['networks_state'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        self.unroll_steps = checkpoint.get('unroll_steps', self.unroll_steps)
        self.value_loss_weight = checkpoint.get('value_loss_weight', self.value_loss_weight)
        self.reward_loss_weight = checkpoint.get('reward_loss_weight', self.reward_loss_weight)
        self.global_step = checkpoint.get('global_step', 0)
        self.current_epoch = checkpoint.get('epoch', 0)

        # 載入學習率調度器狀態
        scheduler_state = checkpoint.get('scheduler_state')
        if scheduler_state is not None and self.scheduler is not None:
            self.scheduler.load_state_dict(scheduler_state)
            logger.info(f"學習率調度器已恢復 (step: {self.scheduler.current_step}, lr: {self.scheduler.get_lr():.6f})")

        # 驗證元數據（如果存在）
        metadata = checkpoint.get('metadata', {})
        if metadata:
            saved_lr = metadata.get('learning_rate')
            saved_batch_size = metadata.get('batch_size')
            saved_timestamp = metadata.get('timestamp', 'unknown')

            current_lr = self.optimizer.param_groups[0]['lr']
            if saved_lr is not None and abs(saved_lr - current_lr) > 1e-8:
                logger.warning(
                    f"學習率不匹配: 檢查點={saved_lr:.6f}, 當前={current_lr:.6f}"
                )

            if expected_batch_size is not None and saved_batch_size is not None:
                if saved_batch_size != expected_batch_size:
                    logger.warning(
                        f"Batch size 不匹配: 檢查點={saved_batch_size}, 當前={expected_batch_size}"
                    )

            logger.info(f"檢查點元數據: timestamp={saved_timestamp}")
        else:
            logger.warning("檢查點缺少元數據，這是舊版本的檢查點")

        logger.info(f"檢查點已從 {path} 載入 (epoch: {self.current_epoch}, global_step: {self.global_step})")

    # ========== Phase 1: 監督預訓練 ==========

    def supervised_train_step(
        self,
        features: torch.Tensor,
        target_policies: torch.Tensor,
        target_values: torch.Tensor
    ) -> Dict[str, float]:
        """監督學習訓練步驟（僅 policy + value，不訓練 dynamics）

        Parameters
        ----------
        features : torch.Tensor
            輸入特徵 (batch, C, H, W)
        target_policies : torch.Tensor
            目標策略 (batch, num_actions)
        target_values : torch.Tensor
            目標價值 (batch,)

        Returns
        -------
        Dict[str, float]
            損失字典
        """
        self.networks.train()

        features = features.to(self.device)
        target_policies = target_policies.to(self.device)
        target_values = target_values.to(self.device)

        # 決定 autocast dtype
        if self.mixed_precision:
            if self.device == 'mps':
                autocast_dtype = torch.float16
            else:
                autocast_dtype = torch.float16
        else:
            autocast_dtype = None

        # 前向傳播（可選混合精度）
        if self.mixed_precision:
            with torch.autocast(device_type=self.device if self.device != 'mps' else 'cpu',
                                dtype=autocast_dtype, enabled=self.device != 'mps'):
                # 使用 initial_inference 獲取 policy 和 value
                _, policy_logits, value = self.networks.initial_inference(features)

                # 計算損失
                policy_loss = self._policy_loss(policy_logits, target_policies)
                value_loss = self._value_loss(value.squeeze(-1), target_values)
                total_loss = policy_loss + self.value_loss_weight * value_loss
        else:
            # 使用 initial_inference 獲取 policy 和 value
            _, policy_logits, value = self.networks.initial_inference(features)

            # 計算損失
            policy_loss = self._policy_loss(policy_logits, target_policies)
            value_loss = self._value_loss(value.squeeze(-1), target_values)
            total_loss = policy_loss + self.value_loss_weight * value_loss

        # 縮放損失用於 gradient accumulation
        scaled_loss = total_loss / self.gradient_accumulation_steps

        # 反向傳播
        if self.scaler is not None:
            self.scaler.scale(scaled_loss).backward()
        else:
            scaled_loss.backward()

        return {
            'total_loss': total_loss.item(),
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item()
        }

    def supervised_optimizer_step(self) -> None:
        """執行優化器更新（用於 gradient accumulation）"""
        # 梯度裁剪
        if self.scaler is not None:
            self.scaler.unscale_(self.optimizer)

        torch.nn.utils.clip_grad_norm_(self.networks.parameters(), max_norm=1.0)

        # 優化器步驟
        if self.scaler is not None:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()

        # 學習率調度器步驟
        if self.scheduler is not None:
            self.scheduler.step()

        self.optimizer.zero_grad()
        self.global_step += 1

    def get_lr(self) -> float:
        """獲取當前學習率"""
        if self.scheduler is not None:
            return self.scheduler.get_lr()
        return self.optimizer.param_groups[0]['lr']

    def supervised_train_epoch(
        self,
        dataloader: DataLoader,
        epoch: int = 0
    ) -> Dict[str, float]:
        """監督學習訓練一個 epoch

        Parameters
        ----------
        dataloader : DataLoader
            訓練資料載入器
        epoch : int
            當前 epoch 編號

        Returns
        -------
        Dict[str, float]
            平均損失
        """
        self.networks.train()
        self.optimizer.zero_grad()

        total_losses = []
        policy_losses = []
        value_losses = []

        accumulation_counter = 0

        for batch_idx, batch in enumerate(dataloader):
            features = batch['features']
            policies = batch['policy']
            values = batch['value']

            losses = self.supervised_train_step(features, policies, values)

            total_losses.append(losses['total_loss'])
            policy_losses.append(losses['policy_loss'])
            value_losses.append(losses['value_loss'])

            accumulation_counter += 1

            # Gradient accumulation
            if accumulation_counter >= self.gradient_accumulation_steps:
                self.supervised_optimizer_step()
                accumulation_counter = 0

            # 進度日誌
            if (batch_idx + 1) % 100 == 0:
                avg_loss = np.mean(total_losses[-100:])
                logger.info(
                    f"Epoch {epoch}, Batch {batch_idx + 1}/{len(dataloader)}, "
                    f"Loss: {avg_loss:.4f}"
                )

        # 處理剩餘梯度
        if accumulation_counter > 0:
            self.supervised_optimizer_step()

        return {
            'total_loss': np.mean(total_losses),
            'policy_loss': np.mean(policy_losses),
            'value_loss': np.mean(value_losses)
        }

    def supervised_evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        """監督學習評估

        Parameters
        ----------
        dataloader : DataLoader
            驗證資料載入器

        Returns
        -------
        Dict[str, float]
            平均損失
        """
        self.networks.eval()

        total_losses = []
        policy_losses = []
        value_losses = []

        with torch.no_grad():
            for batch in dataloader:
                features = batch['features'].to(self.device)
                policies = batch['policy'].to(self.device)
                values = batch['value'].to(self.device)

                _, policy_logits, value = self.networks.initial_inference(features)

                policy_loss = self._policy_loss(policy_logits, policies)
                value_loss = self._value_loss(value.squeeze(-1), values)
                total_loss = policy_loss + self.value_loss_weight * value_loss

                total_losses.append(total_loss.item())
                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())

        return {
            'total_loss': np.mean(total_losses),
            'policy_loss': np.mean(policy_losses),
            'value_loss': np.mean(value_losses)
        }

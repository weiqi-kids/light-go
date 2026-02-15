#!/usr/bin/env python3
"""MuZero 訓練腳本

Phase 1: 監督預訓練（使用 KataGo NPZ 資料）
Phase 2: Gumbel MCTS 自我對弈強化（只需 2-16 次模擬）

使用方式:
    # 完整訓練（Phase 1 + Phase 2）
    python scripts/train_muzero.py

    # 僅 Phase 1
    python scripts/train_muzero.py --phase 1

    # 僅 Phase 2（從檢查點繼續）
    python scripts/train_muzero.py --phase 2 --resume data/models/muzero/phase1_best.pt

    # 自訂 epochs
    python scripts/train_muzero.py --epochs 100
"""

from __future__ import annotations

import argparse
import logging
import sys
import os
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# 添加專案根目錄
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.muzero.networks import MuZeroNetworks
from core.muzero.trainer import MuZeroTrainer, SupervisedDataset
from core.muzero.replay_buffer import MuZeroReplayBuffer, Trajectory
from core.game_rules import GoBoard
from core.gumbel_mcts import GumbelMCTS

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ========== 配置 ==========

DEFAULT_CONFIG = {
    # 模型參數 (b28c512)
    'board_size': 19,
    'num_input_planes': 2,
    'hidden_size': 512,
    'num_blocks': 28,

    # Phase 1: 監督預訓練
    'supervised_epochs': 50,
    'batch_size': 32,
    'gradient_accumulation': 4,  # 等效 batch_size = 128
    'learning_rate': 0.01,       # 初始學習率（scheduler 會調整）
    'weight_decay': 0.0001,
    'loading_mode': 'batch',     # 'batch' | 'streaming' | 'preload'
    'batch_num_files': 12000,    # 每批載入的檔案數（~47 樣本/檔案 × 12000 ≈ 56 萬樣本）
    'batch_max_samples': 500_000,  # 每批最大樣本數（~2GB 記憶體）
    'num_workers': 0,            # macOS 上避免多進程問題

    # 學習率調度器（AlphaGo Zero 風格：warmup + cosine decay）
    'scheduler': {
        'type': 'warmup_cosine',
        'initial_lr': 0.01,
        'min_lr': 0.0001,
        'warmup_steps': 1000,        # 前 1000 步 warmup
        'total_steps': 500_000,      # 50 epochs × ~10000 batches
    },

    # Phase 2: Gumbel MCTS 自我對弈（只需 2-16 次模擬）
    'selfplay_iterations': 50,
    'games_per_iteration': 100,
    'num_simulations': 16,       # Gumbel MCTS 只需 16 次（相當於傳統 800+）
    'num_sampled_actions': 16,   # Gumbel-Top-k 的 k 值
    'temperature_threshold': 30,

    # MuZero 參數
    'td_steps': 10,
    'unroll_steps': 5,
    'discount': 0.997,
    'replay_buffer_size': 10000,

    # 路徑
    'data_dir': 'data/lightgo/training_data/katago_v2',
    'output_dir': 'data/models/muzero',
    'sgf_dir': 'data/sgf/muzero_selfplay',
}


# ========== MuZero 到 GumbelMCTS 適配器 ==========

class MuZeroModelAdapter(torch.nn.Module):
    """將 MuZeroNetworks 適配為 GumbelMCTS 可用的接口

    GumbelMCTS 期望模型的 forward 返回 (policy_logits, value)。
    這個適配器將 MuZeroNetworks 的 representation + prediction 網路組合起來。
    """

    def __init__(self, networks: MuZeroNetworks):
        super().__init__()
        self.networks = networks

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        x : torch.Tensor
            棋盤特徵 (batch, 2, 19, 19)

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            (policy_logits, value)
        """
        # 使用 representation network 編碼
        hidden_state = self.networks.representation(x)
        # 使用 prediction network 得到策略和價值
        policy_logits, value = self.networks.prediction(hidden_state)
        return policy_logits, value


# ========== 資料載入 ==========

class MuZeroNPZDataset(Dataset):
    """從 LightGo NPZ 檔案載入訓練資料（用於 MuZero 監督預訓練）"""

    def __init__(self, data_dir: str, max_files: Optional[int] = None):
        self.features: List[np.ndarray] = []
        self.policies: List[np.ndarray] = []
        self.values: List[float] = []
        self._load_npz_files(data_dir, max_files)

    def _load_npz_files(self, data_dir: str, max_files: Optional[int]) -> None:
        data_path = Path(data_dir)
        npz_files = sorted(data_path.glob("*.npz"))

        if max_files:
            npz_files = npz_files[:max_files]

        logger.info(f"載入 {len(npz_files)} 個 NPZ 檔案...")

        for i, npz_file in enumerate(npz_files):
            try:
                data = np.load(npz_file)
                inputs = data['inputs']  # (N, 2, 19, 19)
                policies = data['policy_targets']  # (N, 362)
                values = data['value_targets']  # (N, 1)

                for j in range(len(inputs)):
                    value = values[j].flatten()[0] if values[j].size > 1 else float(values[j])
                    self.features.append(inputs[j].astype(np.float32))
                    self.policies.append(policies[j].astype(np.float32))
                    self.values.append(np.float32(value))

                if (i + 1) % 500 == 0:
                    logger.info(f"  已載入 {i + 1}/{len(npz_files)} 檔案, "
                               f"{len(self.features)} 個樣本")

            except Exception as e:
                logger.debug(f"載入 {npz_file} 失敗: {e}")

        logger.info(f"總共載入 {len(self.features)} 個訓練樣本")

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            'features': torch.from_numpy(self.features[idx]),
            'policy': torch.from_numpy(self.policies[idx]),
            'value': torch.tensor(self.values[idx], dtype=torch.float32)
        }


class StreamingNPZDataset(Dataset):
    """流式載入 NPZ 檔案，不預載全部資料到記憶體

    適用於大量訓練資料（如 600 萬+ 樣本）的場景。
    使用 LRU 快取來減少重複讀取同一檔案的開銷。
    """

    def __init__(self, data_dir: str, cache_size: int = 100):
        """
        Args:
            data_dir: NPZ 檔案目錄
            cache_size: LRU 快取大小（檔案數）
        """
        self.data_dir = Path(data_dir)
        self.npz_files: List[Path] = []
        self.index_map: List[Tuple[int, int]] = []  # (file_idx, sample_idx)
        self._file_cache: Dict[int, Dict[str, np.ndarray]] = {}
        self._cache_order: List[int] = []
        self._cache_size = cache_size

        self._build_index()

    def _build_index(self) -> None:
        """建立索引表，只讀取每個檔案的樣本數"""
        self.npz_files = sorted(self.data_dir.glob("*.npz"))
        logger.info(f"掃描 {len(self.npz_files)} 個 NPZ 檔案...")

        self.index_map = []  # 重置索引
        total_samples = 0
        for file_idx, npz_file in enumerate(self.npz_files):
            try:
                # 只讀取檔案頭來取得樣本數
                with np.load(npz_file) as data:
                    n_samples = len(data['inputs'])

                for sample_idx in range(n_samples):
                    self.index_map.append((file_idx, sample_idx))

                total_samples += n_samples

                if (file_idx + 1) % 5000 == 0:
                    logger.info(f"  已掃描 {file_idx + 1}/{len(self.npz_files)} 檔案, "
                               f"{total_samples} 個樣本")
            except Exception as e:
                logger.debug(f"掃描 {npz_file} 失敗: {e}")

        logger.info(f"總共 {total_samples} 個訓練樣本（流式載入模式）")

    def rescan(self) -> int:
        """重新掃描目錄，納入新檔案

        Returns:
            新增的樣本數
        """
        old_count = len(self.index_map)
        self._file_cache.clear()  # 清除快取
        self._cache_order.clear()
        self._build_index()
        new_count = len(self.index_map)
        added = new_count - old_count
        if added > 0:
            logger.info(f"新增 {added} 個樣本（總計 {new_count}）")
        return added

    def _get_file_data(self, file_idx: int) -> Dict[str, np.ndarray]:
        """取得檔案資料（帶 LRU 快取）"""
        if file_idx in self._file_cache:
            # 移到最近使用
            self._cache_order.remove(file_idx)
            self._cache_order.append(file_idx)
            return self._file_cache[file_idx]

        # 載入檔案
        data = dict(np.load(self.npz_files[file_idx]))

        # 加入快取
        self._file_cache[file_idx] = data
        self._cache_order.append(file_idx)

        # 清理超出快取大小的舊檔案
        while len(self._cache_order) > self._cache_size:
            old_idx = self._cache_order.pop(0)
            del self._file_cache[old_idx]

        return data

    def __len__(self) -> int:
        return len(self.index_map)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        file_idx, sample_idx = self.index_map[idx]
        data = self._get_file_data(file_idx)

        features = data['inputs'][sample_idx].astype(np.float32)
        policy = data['policy_targets'][sample_idx].astype(np.float32)
        value = data['value_targets'][sample_idx].flatten()[0]
        if hasattr(value, '__len__'):
            value = float(value)

        return {
            'features': torch.from_numpy(features),
            'policy': torch.from_numpy(policy),
            'value': torch.tensor(value, dtype=torch.float32)
        }


class BatchLoadingDataset(Dataset):
    """批次隨機載入資料集

    每個 epoch 隨機選取一批 NPZ 檔案載入記憶體，
    解決流式載入的 I/O 瓶頸問題，同時控制記憶體用量。

    特點：
    - 無需預處理，直接使用現有 NPZ 檔案
    - 記憶體用量可控（取決於 max_samples）
    - 每個 epoch 可以看到不同的資料子集
    """

    def __init__(
        self,
        data_dir: str,
        num_files: int = 2000,
        max_samples: int = 200_000
    ):
        """
        Args:
            data_dir: NPZ 檔案目錄
            num_files: 每批隨機選取的檔案數
            max_samples: 每批最大樣本數
        """
        self.data_dir = Path(data_dir)
        self.all_files = sorted(self.data_dir.glob("*.npz"))
        self.num_files = num_files
        self.max_samples = max_samples

        if len(self.all_files) == 0:
            raise ValueError(f"在 {data_dir} 找不到任何 NPZ 檔案")

        logger.info(f"找到 {len(self.all_files)} 個 NPZ 檔案")

        # 初始載入
        self.features: np.ndarray = np.array([])
        self.policies: np.ndarray = np.array([])
        self.values: np.ndarray = np.array([])
        self.reload()

    def reload(self) -> int:
        """重新隨機載入一批資料

        Returns:
            載入的樣本數
        """
        import random

        # 隨機選取檔案
        n_files = min(self.num_files, len(self.all_files))
        selected_files = random.sample(self.all_files, n_files)

        features_list: List[np.ndarray] = []
        policies_list: List[np.ndarray] = []
        values_list: List[np.ndarray] = []
        total_samples = 0

        for f in selected_files:
            try:
                data = np.load(f)
                n = len(data['inputs'])

                features_list.append(data['inputs'])
                policies_list.append(data['policy_targets'])
                values_list.append(data['value_targets'])
                total_samples += n

                if total_samples >= self.max_samples:
                    break
            except Exception as e:
                logger.debug(f"載入 {f} 失敗: {e}")
                continue

        if len(features_list) == 0:
            raise ValueError("無法載入任何資料")

        # 合併並截斷
        self.features = np.concatenate(features_list)[:self.max_samples].astype(np.float32)
        self.policies = np.concatenate(policies_list)[:self.max_samples].astype(np.float32)
        self.values = np.concatenate(values_list)[:self.max_samples].astype(np.float32)

        logger.info(f"已載入 {len(self.features)} 個樣本（從 {len(selected_files)} 個檔案）")
        return len(self.features)

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        value = self.values[idx]
        if hasattr(value, '__len__'):
            value = float(value.flatten()[0])
        else:
            value = float(value)

        return {
            'features': torch.from_numpy(self.features[idx]),
            'policy': torch.from_numpy(self.policies[idx]),
            'value': torch.tensor(value, dtype=torch.float32)
        }


# ========== 工具函數 ==========

def get_device() -> str:
    """自動選擇最佳運算裝置"""
    if torch.cuda.is_available():
        return 'cuda'
    elif torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'


def create_networks(config: Dict, device: str) -> MuZeroNetworks:
    """建立 MuZero 網路"""
    networks = MuZeroNetworks(
        board_size=config['board_size'],
        num_input_planes=config['num_input_planes'],
        hidden_size=config['hidden_size'],
        num_blocks=config['num_blocks']
    )
    networks.to(device)

    param_count = networks.get_param_count()
    logger.info(f"MuZero 網路參數量: {param_count:,}")

    return networks


# ========== Phase 1: 監督預訓練 ==========

def train_phase1(
    config: Dict,
    device: str,
    networks: Optional[MuZeroNetworks] = None,
    resume_path: Optional[str] = None
) -> Tuple[MuZeroNetworks, str]:
    """Phase 1: 監督預訓練

    Parameters
    ----------
    config : Dict
        訓練配置
    device : str
        運算設備
    networks : Optional[MuZeroNetworks]
        已存在的網路（可選）
    resume_path : Optional[str]
        恢復訓練的檢查點路徑

    Returns
    -------
    Tuple[MuZeroNetworks, str]
        訓練後的網路和檢查點路徑
    """
    logger.info("=" * 60)
    logger.info("Phase 1: MuZero 監督預訓練")
    logger.info("=" * 60)

    # 建立或使用現有網路
    if networks is None:
        networks = create_networks(config, device)

    # 載入資料（根據配置選擇載入模式）
    loading_mode = config.get('loading_mode', 'batch')
    num_workers = config.get('num_workers', 0)

    # 向後相容：舊的 'streaming' 配置
    if config.get('streaming', False) and loading_mode != 'batch':
        loading_mode = 'streaming'

    if loading_mode == 'batch':
        logger.info("使用批次隨機載入模式（推薦）")
        dataset = BatchLoadingDataset(
            config['data_dir'],
            num_files=config.get('batch_num_files', 2000),
            max_samples=config.get('batch_max_samples', 200_000)
        )
    elif loading_mode == 'streaming':
        logger.info("使用流式載入模式（較慢）")
        dataset = StreamingNPZDataset(config['data_dir'])
    else:
        logger.info("使用預載模式（需大量記憶體）")
        dataset = MuZeroNPZDataset(config['data_dir'])

    if len(dataset) == 0:
        raise ValueError("沒有找到訓練資料！")

    train_size = len(dataset)
    val_loader = None

    # batch 模式可以啟用 shuffle（資料都在記憶體中）
    use_shuffle = (loading_mode == 'batch')

    train_loader = DataLoader(
        dataset,
        batch_size=config['batch_size'],
        shuffle=use_shuffle,
        num_workers=num_workers,
        pin_memory=(device == 'cuda')
    )

    logger.info(f"訓練樣本: {train_size}（{loading_mode} 模式）")
    logger.info(f"Batch size: {config['batch_size']}, "
                f"Gradient accumulation: {config['gradient_accumulation']}")
    logger.info(f"等效 batch size: {config['batch_size'] * config['gradient_accumulation']}")

    # 建立訓練器（含學習率調度器）
    scheduler_config = config.get('scheduler', None)
    trainer = MuZeroTrainer(
        networks=networks,
        learning_rate=config['learning_rate'],
        weight_decay=config['weight_decay'],
        device=device,
        mixed_precision=False,  # MPS 不完全支援混合精度
        gradient_accumulation_steps=config['gradient_accumulation'],
        scheduler_config=scheduler_config
    )

    # 恢復訓練
    start_epoch = 0
    if resume_path and Path(resume_path).exists():
        trainer.load_checkpoint(resume_path)
        start_epoch = getattr(trainer, 'current_epoch', 0) + 1  # 從下一個 epoch 開始
        logger.info(f"從 {resume_path} 恢復訓練，從 epoch {start_epoch} 開始")

    # 建立輸出目錄
    output_dir = Path(config['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)

    # 訓練
    best_val_loss = float('inf')
    history = {'train_loss': [], 'val_loss': []}

    for epoch in range(start_epoch, config['supervised_epochs']):
        # batch 模式：每個 epoch 重新隨機載入一批資料
        if loading_mode == 'batch' and epoch > 0:
            dataset.reload()
            train_loader = DataLoader(
                dataset,
                batch_size=config['batch_size'],
                shuffle=True,
                num_workers=num_workers,
                pin_memory=(device == 'cuda')
            )

        # streaming 模式：重新掃描納入新檔案
        elif loading_mode == 'streaming' and hasattr(dataset, 'rescan') and epoch > 0:
            added = dataset.rescan()
            if added > 0:
                train_loader = DataLoader(
                    dataset,
                    batch_size=config['batch_size'],
                    shuffle=True,
                    num_workers=num_workers,
                    pin_memory=(device == 'cuda')
                )
                logger.info(f"資料已更新：訓練樣本 {len(dataset)}")

        # 更新當前 epoch（用於 checkpoint）
        trainer.current_epoch = epoch

        # 訓練一個 epoch
        train_losses = trainer.supervised_train_epoch(train_loader, epoch)
        history['train_loss'].append(train_losses['total_loss'])

        # 驗證（流式模式跳過）
        if val_loader is not None:
            val_losses = trainer.supervised_evaluate(val_loader)
            history['val_loss'].append(val_losses['total_loss'])
            val_loss_str = f", 驗證損失: {val_losses['total_loss']:.4f}"
        else:
            val_losses = {'total_loss': train_losses['total_loss']}  # 用訓練損失代替
            val_loss_str = ""

        # 獲取當前學習率
        current_lr = trainer.get_lr()
        logger.info(
            f"Epoch {epoch + 1}/{config['supervised_epochs']} - "
            f"訓練損失: {train_losses['total_loss']:.4f} "
            f"(policy: {train_losses['policy_loss']:.4f}, value: {train_losses['value_loss']:.4f})"
            f"{val_loss_str}, lr: {current_lr:.6f}"
        )

        # 儲存最佳模型（用訓練損失判斷）
        if train_losses['total_loss'] < best_val_loss:
            best_val_loss = train_losses['total_loss']
            best_path = output_dir / "phase1_best.pt"
            trainer.save_checkpoint(str(best_path))
            logger.info(f"  新最佳模型已儲存 (train_loss: {best_val_loss:.4f})")

    # 儲存最終模型
    final_path = output_dir / "phase1_final.pt"
    trainer.save_checkpoint(str(final_path))

    logger.info("=" * 60)
    logger.info("Phase 1 完成！")
    logger.info(f"最終訓練損失: {history['train_loss'][-1]:.4f}")
    logger.info(f"最佳驗證損失: {best_val_loss:.4f}")
    logger.info(f"模型已儲存至: {final_path}")
    logger.info("=" * 60)

    return networks, str(output_dir / "phase1_best.pt")


# ========== Phase 2: MuZero 自我對弈 ==========

def encode_board(board: GoBoard) -> np.ndarray:
    """將棋盤編碼為 2-plane 特徵

    Plane 0: 己方棋子的氣數（正值）/ 對方棋子的氣數（負值）
    Plane 1: 禁著點（1 = 不可落子）
    """
    size = board.size
    features = np.zeros((2, size, size), dtype=np.float32)

    # 計算各點的氣數
    for i in range(size):
        for j in range(size):
            stone = board.get(i, j)  # 'b', 'w', or None
            if stone is not None:  # 有棋子
                group = board.get_group(i, j)
                liberties = board.count_liberties(group)
                # 黑棋正值，白棋負值
                features[0, i, j] = liberties if stone == 'b' else -liberties
                # 標記已佔用的點
                features[1, i, j] = 1.0

    return features


def save_game_to_sgf(trajectory: Trajectory, board_size: int, result: float, filepath: str) -> None:
    """儲存棋局為 SGF 檔案"""
    from sgfmill import sgf

    game = sgf.Sgf_game(size=board_size)
    root = game.get_root()
    root.set("KM", 7.5)
    root.set("PB", "MuZero-Black")
    root.set("PW", "MuZero-White")

    if result > 0:
        root.set("RE", "B+")
    elif result < 0:
        root.set("RE", "W+")
    else:
        root.set("RE", "0")

    node = root
    for i, action in enumerate(trajectory.actions):
        color = 'b' if i % 2 == 0 else 'w'
        node = node.new_child()
        if action == board_size * board_size:  # pass
            node.set_move(color, None)
        else:
            row, col = action // board_size, action % board_size
            node.set_move(color, (row, col))

    with open(filepath, 'wb') as f:
        f.write(game.serialise())


def play_gumbel_game(
    gumbel_mcts: GumbelMCTS,
    board_size: int = 19,
    temperature_threshold: int = 30,
    max_moves: int = 400
) -> Trajectory:
    """使用 Gumbel MCTS 進行一局自我對弈

    Parameters
    ----------
    gumbel_mcts : GumbelMCTS
        Gumbel MCTS 搜尋引擎
    board_size : int
        棋盤大小
    temperature_threshold : int
        溫度閾值（前 N 手使用高溫度）
    max_moves : int
        最大步數

    Returns
    -------
    Trajectory
        遊戲軌跡
    """
    board = GoBoard(size=board_size)
    trajectory = Trajectory()
    current_player = 'b'  # 黑棋先手

    consecutive_passes = 0

    for move_num in range(max_moves):
        # 編碼當前盤面
        obs = encode_board(board)

        # 決定溫度
        temperature = 1.0 if move_num < temperature_threshold else 0.0

        # Gumbel MCTS 搜尋
        move_probs, root = gumbel_mcts.search(board, current_player)

        # 選擇動作
        if temperature == 0:
            action = int(np.argmax(move_probs))
        else:
            # 溫度採樣
            adjusted_probs = move_probs ** (1.0 / temperature)
            adjusted_probs /= adjusted_probs.sum()
            action = int(np.random.choice(len(move_probs), p=adjusted_probs))

        # 取得價值估計
        value = root.value_estimate if root else 0.0

        # 添加到軌跡
        trajectory.add_step(
            observation=obs,
            action=action,
            reward=0.0,  # 中間獎勵為 0
            policy=move_probs,
            value=value
        )

        # 執行動作
        if action == board_size * board_size:  # Pass
            consecutive_passes += 1
            if consecutive_passes >= 2:
                break
        else:
            row, col = action // board_size, action % board_size

            # 嘗試落子
            if board.is_legal(row, col, current_player):
                board.play_move(row, col, current_player)
                consecutive_passes = 0
            else:
                # 非法動作，視為 pass
                consecutive_passes += 1
                if consecutive_passes >= 2:
                    break

        # 切換玩家
        current_player = 'w' if current_player == 'b' else 'b'

    # 計算遊戲結果
    score_result = board.score_chinese()
    if score_result['winner'] == 'black':
        trajectory.game_result = 1.0
    elif score_result['winner'] == 'white':
        trajectory.game_result = -1.0
    else:
        trajectory.game_result = 0.0

    return trajectory


def train_phase2(
    config: Dict,
    device: str,
    networks: MuZeroNetworks,
    resume_path: Optional[str] = None
) -> MuZeroNetworks:
    """Phase 2: Gumbel MCTS 自我對弈訓練

    使用 Gumbel MCTS 進行自我對弈，只需 2-16 次模擬即可達到
    傳統 MCTS 800+ 次模擬的效果。

    Parameters
    ----------
    config : Dict
        訓練配置
    device : str
        運算設備
    networks : MuZeroNetworks
        預訓練的網路
    resume_path : Optional[str]
        恢復訓練的檢查點路徑

    Returns
    -------
    MuZeroNetworks
        訓練後的網路
    """
    logger.info("=" * 60)
    logger.info("Phase 2: Gumbel MCTS 自我對弈訓練")
    logger.info(f"  模擬次數: {config['num_simulations']} (相當於傳統 800+)")
    logger.info("=" * 60)

    # 建立 Replay Buffer
    replay_buffer = MuZeroReplayBuffer(
        max_size=config['replay_buffer_size'],
        td_steps=config['td_steps'],
        discount=config['discount']
    )

    # 建立訓練器
    trainer = MuZeroTrainer(
        networks=networks,
        replay_buffer=replay_buffer,
        unroll_steps=config['unroll_steps'],
        learning_rate=config['learning_rate'] * 0.1,  # 較小學習率
        device=device,
        gradient_accumulation_steps=config['gradient_accumulation']
    )

    # 建立 Gumbel MCTS（使用適配器）
    adapter_model = MuZeroModelAdapter(networks)
    gumbel_mcts = GumbelMCTS(
        model=adapter_model,
        board_size=config['board_size'],
        num_simulations=config['num_simulations'],
        num_sampled_actions=config.get('num_sampled_actions', 16),
        c_visit=50.0,
        c_scale=1.0,
        device=device
    )

    # 恢復訓練
    start_iteration = 0
    if resume_path and Path(resume_path).exists():
        trainer.load_checkpoint(resume_path)
        start_iteration = getattr(trainer, 'current_epoch', 0) + 1  # 從下一個迭代開始
        logger.info(f"從 {resume_path} 恢復訓練，從迭代 {start_iteration} 開始")

    # 建立輸出目錄
    output_dir = Path(config['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)

    sgf_dir = Path(config['sgf_dir'])
    sgf_dir.mkdir(parents=True, exist_ok=True)

    # 追蹤最佳 loss
    best_loss = float('inf')

    # 自我對弈訓練循環
    for iteration in range(start_iteration, config['selfplay_iterations']):
        # 更新當前迭代（用於 checkpoint）
        trainer.current_epoch = iteration

        logger.info(f"\n--- 迭代 {iteration + 1}/{config['selfplay_iterations']} ---")

        # 自我對弈生成軌跡
        logger.info(f"生成 {config['games_per_iteration']} 局自我對弈...")
        networks.eval()

        for game_idx in range(config['games_per_iteration']):
            logger.info(f"  開始第 {game_idx + 1} 局...")
            trajectory = play_gumbel_game(
                gumbel_mcts=gumbel_mcts,
                board_size=config['board_size'],
                temperature_threshold=config['temperature_threshold']
            )
            replay_buffer.add_trajectory(trajectory)

            # 儲存 SGF
            sgf_path = sgf_dir / f"iter{iteration+1}_game{game_idx+1}.sgf"
            save_game_to_sgf(trajectory, config['board_size'], trajectory.game_result, str(sgf_path))

            logger.info(f"  完成第 {game_idx + 1}/{config['games_per_iteration']} 局 (步數: {len(trajectory)}, 結果: {trajectory.game_result})")

        logger.info(f"Replay buffer 大小: {len(replay_buffer)} 軌跡, "
                    f"{replay_buffer.num_samples} 樣本")
        logger.info(f"Trainer replay buffer: {len(trainer.replay_buffer)} 軌跡 "
                    f"(same object: {replay_buffer is trainer.replay_buffer})")

        # 訓練
        if len(replay_buffer) >= 10:  # 確保有足夠資料
            logger.info("訓練中...")
            losses = trainer.train_epoch(
                batch_size=config['batch_size'],
                num_batches=100
            )

            logger.info(
                f"損失 - total: {losses['total_loss']:.4f}, "
                f"policy: {losses['policy_loss']:.4f}, "
                f"value: {losses['value_loss']:.4f}, "
                f"reward: {losses['reward_loss']:.4f}"
            )

            # 儲存最佳模型（loss 改善時）
            if losses['total_loss'] < best_loss:
                best_loss = losses['total_loss']
                best_path = output_dir / "phase2_best.pt"
                trainer.save_checkpoint(str(best_path))
                logger.info(f"  新最佳模型已儲存 (loss: {best_loss:.4f})")

    # 儲存最終模型
    final_path = output_dir / "phase2_final.pt"
    trainer.save_checkpoint(str(final_path))

    logger.info("=" * 60)
    logger.info("Phase 2 完成！")
    logger.info(f"模型已儲存至: {final_path}")
    logger.info("=" * 60)

    return networks


# ========== 主程式 ==========

def main():
    parser = argparse.ArgumentParser(description='MuZero 訓練')

    # 訓練階段
    parser.add_argument('--phase', type=int, default=0,
                       help='訓練階段 (0=全部, 1=僅Phase1, 2=僅Phase2)')
    parser.add_argument('--resume', type=str, default=None,
                       help='恢復訓練的檢查點路徑')

    # 資料路徑
    parser.add_argument('--data-dir', type=str,
                       default=DEFAULT_CONFIG['data_dir'],
                       help='訓練資料目錄')
    parser.add_argument('--output-dir', type=str,
                       default=DEFAULT_CONFIG['output_dir'],
                       help='模型輸出目錄')

    # 訓練參數
    parser.add_argument('--epochs', type=int, default=None,
                       help='Phase 1 訓練週期數')
    parser.add_argument('--iterations', type=int, default=None,
                       help='Phase 2 自我對弈迭代數')
    parser.add_argument('--batch-size', type=int, default=None,
                       help='批次大小')
    parser.add_argument('--learning-rate', type=float, default=None,
                       help='學習率')

    args = parser.parse_args()

    # 合併配置
    config = DEFAULT_CONFIG.copy()
    config['data_dir'] = args.data_dir
    config['output_dir'] = args.output_dir

    if args.epochs:
        config['supervised_epochs'] = args.epochs
    if args.iterations:
        config['selfplay_iterations'] = args.iterations
    if args.batch_size:
        config['batch_size'] = args.batch_size
    if args.learning_rate:
        config['learning_rate'] = args.learning_rate

    # 選擇設備
    device = get_device()

    logger.info("=" * 60)
    logger.info("MuZero 訓練開始")
    logger.info("=" * 60)
    logger.info(f"設備: {device}")
    logger.info(f"模型: hidden_size={config['hidden_size']}, num_blocks={config['num_blocks']}")
    logger.info(f"資料目錄: {config['data_dir']}")
    logger.info(f"輸出目錄: {config['output_dir']}")

    # 執行訓練
    networks = None
    checkpoint_path = args.resume

    if args.phase == 0 or args.phase == 1:
        # Phase 1: 監督預訓練
        networks, checkpoint_path = train_phase1(
            config, device, networks, checkpoint_path
        )

    if args.phase == 0 or args.phase == 2:
        # Phase 2: 自我對弈
        if networks is None:
            networks = create_networks(config, device)
            if checkpoint_path and Path(checkpoint_path).exists():
                # 載入 Phase 1 的權重
                checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
                networks.load_state_dict(checkpoint['networks_state'])
                logger.info(f"已載入 Phase 1 模型: {checkpoint_path}")

        train_phase2(config, device, networks, None)

    logger.info("\n訓練完成！")


if __name__ == '__main__':
    main()

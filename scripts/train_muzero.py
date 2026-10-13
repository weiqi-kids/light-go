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
from torch.utils.data import DataLoader

# 添加專案根目錄
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.muzero.networks import MuZeroNetworks
from core.muzero.trainer import MuZeroTrainer
from core.muzero.replay_buffer import MuZeroReplayBuffer, Trajectory
from core.game_rules import GoBoard
from core.gumbel_mcts import GumbelMCTS
from core.sample_store import open_sample_dataset
from input.lightgo_encoder import encode_board

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


# ========== 工具函數 ==========

def get_device() -> str:
    """自動選擇最佳運算裝置

    注意：MPS (Metal) 在 AMD GPU 上不穩定，預設只啟用 CUDA。
    如需使用 MPS，請明確指定 --device mps。
    """
    if torch.cuda.is_available():
        return 'cuda'
    # MPS 在 AMD GPU 上容易出錯，預設不自動啟用
    # 如需使用，請用 --device mps 明確指定
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

    mode_desc = {'batch': '批次隨機載入模式（推薦）',
                 'streaming': '流式載入模式（較慢）',
                 'memory': '預載模式（需大量記憶體）'}
    if loading_mode not in mode_desc:
        loading_mode = 'memory'  # 其他值一律視為預載（沿用舊行為）
    logger.info(f"使用{mode_desc[loading_mode]}")
    dataset = open_sample_dataset(
        config['data_dir'],
        loading_mode,
        num_files=config.get('batch_num_files', 2000),
        max_samples=config.get('batch_max_samples', 200_000)
    )

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
        # 每個 epoch 更新資料：batch 模式重新隨機載入，streaming 模式納入新檔案
        if epoch > 0 and dataset.refresh() > 0:
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
        # 編碼當前盤面（輪走方視角，與推論端同一編碼）
        obs = encode_board(board, current_player)

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
                       help='恢復訓練的檢查點路徑（預設自動偵測）')
    parser.add_argument('--from-scratch', action='store_true',
                       help='強制從頭訓練，忽略現有 checkpoint')

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
    parser.add_argument('--device', type=str, default=None,
                       choices=['cpu', 'cuda', 'mps'],
                       help='訓練設備 (預設自動偵測，MPS 不穩定時建議用 cpu)')

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
    if args.device:
        device = args.device
        logger.info(f"使用指定設備: {device}")
    else:
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

    # 自動偵測現有 checkpoint（除非指定 --from-scratch）
    if not args.from_scratch and checkpoint_path is None:
        output_dir = Path(config['output_dir'])
        if args.phase in (0, 1):
            auto_ckpt = output_dir / "phase1_best.pt"
        else:
            auto_ckpt = output_dir / "phase2_best.pt"

        if auto_ckpt.exists():
            checkpoint_path = str(auto_ckpt)
            logger.info(f"自動偵測到 checkpoint: {checkpoint_path}")
            # 顯示 checkpoint 資訊
            ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            logger.info(f"  - Epoch: {ckpt.get('epoch', 'N/A')}")
            logger.info(f"  - Global step: {ckpt.get('global_step', 'N/A')}")
            logger.info("  （如要從頭訓練，請加 --from-scratch）")

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

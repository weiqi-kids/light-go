#!/usr/bin/env python3
"""從 SGF 棋譜訓練 Light-Go 模型

這個腳本提供簡單的方式從 SGF 棋譜檔案訓練圍棋 AI 模型。

使用方式:
    # 基本使用 - 從 SGF 資料夾訓練
    python scripts/train_from_sgf.py --sgf-dir data/sgf/games --epochs 10

    # 指定輸出目錄
    python scripts/train_from_sgf.py --sgf-dir data/sgf/games --output-dir data/models/my_model

    # 調整模型大小
    python scripts/train_from_sgf.py --sgf-dir data/sgf/games --num-blocks 10 --num-filters 128

    # 使用小棋盤
    python scripts/train_from_sgf.py --sgf-dir data/sgf/games --board-size 9
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from datetime import datetime

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# 添加專案根目錄到路徑
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from input.sgf_to_input import parse_sgf
from input.lightgo_encoder import LightGoEncoder
from core.game_rules import GoBoard
from core.architecture_genome import ArchitectureGenome
from core.trainer import ModelTrainer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SGFDataset(Dataset):
    """從 SGF 檔案建立的訓練資料集"""

    def __init__(
        self,
        sgf_dir: str,
        board_size: int = 19,
        max_files: Optional[int] = None
    ):
        """
        Parameters
        ----------
        sgf_dir : str
            包含 SGF 檔案的資料夾路徑
        board_size : int
            棋盤大小 (9, 13, 或 19)
        max_files : Optional[int]
            最多讀取的檔案數量 (None 表示全部)
        """
        self.board_size = board_size
        self.encoder = LightGoEncoder(board_size=board_size)
        self.samples: List[Dict] = []

        self._load_sgf_files(sgf_dir, max_files)

    def _load_sgf_files(self, sgf_dir: str, max_files: Optional[int]) -> None:
        """載入所有 SGF 檔案並轉換為訓練樣本"""
        sgf_path = Path(sgf_dir)

        if not sgf_path.exists():
            raise FileNotFoundError(f"找不到資料夾: {sgf_dir}")

        sgf_files = list(sgf_path.glob("**/*.sgf"))
        if max_files:
            sgf_files = sgf_files[:max_files]

        logger.info(f"找到 {len(sgf_files)} 個 SGF 檔案")

        for i, sgf_file in enumerate(sgf_files):
            try:
                self._process_sgf_file(sgf_file)
                if (i + 1) % 100 == 0:
                    logger.info(f"已處理 {i + 1}/{len(sgf_files)} 個檔案, "
                               f"目前 {len(self.samples)} 個訓練樣本")
            except Exception as e:
                logger.debug(f"處理 {sgf_file} 失敗: {e}")
                continue

        logger.info(f"總共載入 {len(self.samples)} 個訓練樣本")

    def _process_sgf_file(self, sgf_file: Path) -> None:
        """處理單個 SGF 檔案，提取所有移動作為訓練樣本"""
        try:
            # 解析完整棋譜
            matrix, metadata, sgf_board = parse_sgf(str(sgf_file))

            if metadata is None or 'step' not in metadata:
                return

            moves = metadata['step']
            if not moves:
                return

            # 檢查棋盤大小
            if matrix.shape[0] != self.board_size:
                return

            # 決定遊戲結果
            game_result = self._get_game_result(metadata)

            # 建立棋盤並逐步執行每一手
            board = GoBoard(size=self.board_size)

            for move_idx, (color, pos) in enumerate(moves):
                if pos is None:  # Pass
                    continue

                row, col = pos
                current_player = 'b' if color == 'black' else 'w'

                # 建立訓練樣本
                # 1. 編碼當前局面
                features = self.encoder.encode(board, current_player)

                # 2. 建立策略目標 (one-hot for the actual move)
                policy = np.zeros(self.board_size * self.board_size + 1, dtype=np.float32)
                move_idx_flat = row * self.board_size + col
                policy[move_idx_flat] = 1.0

                # 3. 價值目標 (從當前玩家視角)
                if game_result is not None:
                    if current_player == 'b':
                        value = game_result  # +1 黑勝, -1 白勝
                    else:
                        value = -game_result
                else:
                    value = 0.0  # 無結果時用 0

                self.samples.append({
                    'features': features.numpy(),
                    'policy': policy,
                    'value': np.float32(value)
                })

                # 執行移動
                if board.is_valid_move(row, col, current_player):
                    board.play(row, col, current_player)

        except Exception as e:
            logger.debug(f"處理棋譜時出錯: {e}")

    def _get_game_result(self, metadata: Dict) -> Optional[float]:
        """從元資料中提取遊戲結果"""
        # 嘗試從不同欄位獲取結果
        result = metadata.get('result') or metadata.get('RE')

        if result is None:
            return None

        result = str(result).upper()

        if 'B+' in result:
            return 1.0  # 黑勝
        elif 'W+' in result:
            return -1.0  # 白勝
        else:
            return None

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        sample = self.samples[idx]
        return (
            torch.from_numpy(sample['features']),
            torch.from_numpy(sample['policy']),
            torch.tensor(sample['value'])
        )


def create_model(
    board_size: int,
    num_blocks: int,
    num_filters: int,
    device: str
) -> Tuple[torch.nn.Module, ArchitectureGenome]:
    """建立神經網路模型"""
    genome = ArchitectureGenome.create_baseline(
        board_size=board_size,
        num_blocks=num_blocks,
        base_filters=num_filters
    )
    model = genome.to_pytorch_model(device=device)
    return model, genome


def train(
    sgf_dir: str,
    output_dir: str,
    board_size: int = 19,
    num_blocks: int = 10,
    num_filters: int = 128,
    epochs: int = 10,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    max_files: Optional[int] = None,
    device: Optional[str] = None
) -> Dict:
    """執行訓練

    Parameters
    ----------
    sgf_dir : str
        SGF 檔案資料夾
    output_dir : str
        模型輸出資料夾
    board_size : int
        棋盤大小
    num_blocks : int
        殘差塊數量
    num_filters : int
        卷積濾波器數量
    epochs : int
        訓練週期數
    batch_size : int
        批次大小
    learning_rate : float
        學習率
    max_files : Optional[int]
        最多讀取的 SGF 檔案數
    device : Optional[str]
        運算裝置 ('cuda', 'mps', 'cpu')

    Returns
    -------
    Dict
        訓練結果資訊
    """
    # 自動選擇裝置
    if device is None:
        if torch.cuda.is_available():
            device = 'cuda'
        elif torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'

    logger.info(f"使用裝置: {device}")
    logger.info(f"棋盤大小: {board_size}x{board_size}")
    logger.info(f"模型架構: {num_blocks} blocks, {num_filters} filters")

    # 載入資料
    logger.info(f"從 {sgf_dir} 載入 SGF 檔案...")
    dataset = SGFDataset(
        sgf_dir=sgf_dir,
        board_size=board_size,
        max_files=max_files
    )

    if len(dataset) == 0:
        raise ValueError("沒有找到有效的訓練樣本！請確認 SGF 檔案路徑正確。")

    # 分割訓練/驗證集
    val_size = max(1, int(len(dataset) * 0.1))
    train_size = len(dataset) - val_size

    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    logger.info(f"訓練樣本: {train_size}, 驗證樣本: {val_size}")

    # 建立模型
    model, genome = create_model(board_size, num_blocks, num_filters, device)
    logger.info(f"模型參數量: {sum(p.numel() for p in model.parameters()):,}")

    # 建立訓練器
    trainer = ModelTrainer(
        model=model,
        device=device,
        learning_rate=learning_rate
    )

    # 訓練
    logger.info(f"開始訓練 {epochs} 個週期...")
    history = {'train_loss': [], 'val_loss': []}

    for epoch in range(epochs):
        # 訓練
        train_metrics = trainer.train_epoch(train_loader, verbose=False)
        history['train_loss'].append(train_metrics['total_loss'])

        # 驗證
        val_metrics = trainer.validate(val_loader)
        history['val_loss'].append(val_metrics['total_loss'])

        logger.info(
            f"Epoch {epoch + 1}/{epochs} - "
            f"訓練損失: {train_metrics['total_loss']:.4f} "
            f"(策略: {train_metrics['policy_loss']:.4f}, "
            f"價值: {train_metrics['value_loss']:.4f}) - "
            f"驗證損失: {val_metrics['total_loss']:.4f}"
        )

    # 儲存模型
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = f"lightgo_{board_size}x{board_size}_{timestamp}"

    # 儲存模型權重
    model_path = output_path / f"{model_name}.pt"
    torch.save(model.state_dict(), model_path)
    logger.info(f"模型已儲存至: {model_path}")

    # 儲存基因組
    genome_path = output_path / f"{model_name}.pkl"
    genome.save(str(genome_path))
    logger.info(f"基因組已儲存至: {genome_path}")

    return {
        'model_path': str(model_path),
        'genome_path': str(genome_path),
        'train_samples': train_size,
        'val_samples': val_size,
        'final_train_loss': history['train_loss'][-1],
        'final_val_loss': history['val_loss'][-1],
        'history': history
    }


def main():
    parser = argparse.ArgumentParser(
        description='從 SGF 棋譜訓練 Light-Go 模型',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
範例:
  # 基本訓練
  python scripts/train_from_sgf.py --sgf-dir data/sgf/games

  # 使用 9x9 小棋盤快速測試
  python scripts/train_from_sgf.py --sgf-dir data/sgf/games --board-size 9 --epochs 5

  # 訓練較大的模型
  python scripts/train_from_sgf.py --sgf-dir data/sgf/games --num-blocks 20 --num-filters 256
        """
    )

    parser.add_argument(
        '--sgf-dir', '-d',
        type=str,
        required=True,
        help='SGF 棋譜檔案資料夾路徑'
    )
    parser.add_argument(
        '--output-dir', '-o',
        type=str,
        default='data/models/sgf_trained',
        help='模型輸出資料夾 (預設: data/models/sgf_trained)'
    )
    parser.add_argument(
        '--board-size', '-s',
        type=int,
        default=19,
        choices=[9, 13, 19],
        help='棋盤大小 (預設: 19)'
    )
    parser.add_argument(
        '--num-blocks', '-b',
        type=int,
        default=10,
        help='殘差塊數量 (預設: 10)'
    )
    parser.add_argument(
        '--num-filters', '-f',
        type=int,
        default=128,
        help='卷積濾波器數量 (預設: 128)'
    )
    parser.add_argument(
        '--epochs', '-e',
        type=int,
        default=10,
        help='訓練週期數 (預設: 10)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='批次大小 (預設: 32)'
    )
    parser.add_argument(
        '--learning-rate', '-lr',
        type=float,
        default=0.001,
        help='學習率 (預設: 0.001)'
    )
    parser.add_argument(
        '--max-files',
        type=int,
        default=None,
        help='最多讀取的 SGF 檔案數 (預設: 全部)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        choices=['cuda', 'mps', 'cpu'],
        help='運算裝置 (預設: 自動選擇)'
    )

    args = parser.parse_args()

    try:
        result = train(
            sgf_dir=args.sgf_dir,
            output_dir=args.output_dir,
            board_size=args.board_size,
            num_blocks=args.num_blocks,
            num_filters=args.num_filters,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            max_files=args.max_files,
            device=args.device
        )

        print("\n" + "=" * 50)
        print("訓練完成！")
        print("=" * 50)
        print(f"模型路徑: {result['model_path']}")
        print(f"基因組路徑: {result['genome_path']}")
        print(f"訓練樣本: {result['train_samples']}")
        print(f"最終訓練損失: {result['final_train_loss']:.4f}")
        print(f"最終驗證損失: {result['final_val_loss']:.4f}")

    except Exception as e:
        logger.error(f"訓練失敗: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()

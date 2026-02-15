"""使用 KataGo 訓練數據訓練模型

這個腳本展示如何使用 KataGo 的 .npz 訓練數據來訓練 Light-Go 模型。

使用方法：
    # 基本用法
    python examples/train_from_katago.py

    # 自定義參數
    python examples/train_from_katago.py \
        --data-dir data/sgf/2026-01-07npzs/kata1-b28c512nbt-s12192929536-d5655876072 \
        --epochs 10 \
        --batch-size 64 \
        --num-blocks 10

前置需求：
    1. 解壓 KataGo 數據：cd data/sgf && tar -xzf 2026-01-07npzs.tgz
    2. 確保有 GPU（強烈推薦）：nvidia-smi
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Any
import glob

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# 添加專案根目錄到路徑
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from hf_models.modeling_go_ai import GoAIModel
from core.architecture_genome import ArchitectureGenome
from core.trainer import ModelTrainer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def print_header(title: str):
    """打印格式化標題"""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def find_npz_files(data_dir: str, max_files: int = None) -> List[str]:
    """查找 .npz 文件

    Parameters
    ----------
    data_dir : str
        數據目錄
    max_files : int, optional
        最多載入多少個文件（用於快速測試）

    Returns
    -------
    List[str]
        .npz 文件路徑列表
    """
    pattern = os.path.join(data_dir, "**/*.npz")
    files = glob.glob(pattern, recursive=True)

    if max_files:
        files = files[:max_files]

    return files


def load_npz_batch_simple(npz_files: List[str], batch_size: int = 32):
    """簡化版 NPZ 載入器（不使用 KataGo 複雜的解碼）

    直接載入未壓縮的數據進行訓練

    Parameters
    ----------
    npz_files : List[str]
        .npz 文件列表
    batch_size : int
        批次大小

    Yields
    ------
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        (positions, policies, values)
    """
    for npz_file in npz_files:
        try:
            data = np.load(npz_file)

            # 解碼壓縮的輸入（簡化版本）
            binary_packed = data['binaryInputNCHWPacked']
            global_input = data['globalInputNC']
            policy_targets = data['policyTargetsNCMove'].astype(np.float32)
            value_targets = data['valueTargetsNCHW'].astype(np.float32)

            # 解包位壓縮數據
            binary_input = np.unpackbits(binary_packed, axis=2)
            binary_input = binary_input[:, :, :19*19]
            binary_input = binary_input.reshape(
                binary_input.shape[0], binary_input.shape[1], 19, 19
            ).astype(np.float32)

            # 提取 policy（使用訪問計數）
            visit_counts = policy_targets[:, 0, :]
            total_visits = visit_counts.sum(axis=1, keepdims=True)
            total_visits = np.maximum(total_visits, 1.0)
            policy = visit_counts / total_visits

            # 提取 value（簡化）
            value = value_targets[:, 0, :, :].mean(axis=(1, 2))
            value = np.clip(value, -1.0, 1.0)

            # 批次化
            num_samples = binary_input.shape[0]
            for i in range(0, num_samples, batch_size):
                end = min(i + batch_size, num_samples)

                batch_pos = torch.from_numpy(binary_input[i:end])
                batch_pol = torch.from_numpy(policy[i:end])
                batch_val = torch.from_numpy(value[i:end])

                yield batch_pos, batch_pol, batch_val

        except Exception as e:
            logger.warning(f"跳過 {os.path.basename(npz_file)}: {e}")
            continue


def train_from_katago(
    data_dir: str,
    epochs: int = 10,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    num_blocks: int = 10,
    num_filters: int = 128,
    max_files: int = None,
    save_dir: str = "data/models/from_katago"
):
    """從 KataGo 數據訓練模型

    Parameters
    ----------
    data_dir : str
        KataGo .npz 數據目錄
    epochs : int
        訓練輪數
    batch_size : int
        批次大小
    learning_rate : float
        學習率
    num_blocks : int
        殘差模塊數量
    num_filters : int
        卷積層 filter 數量
    max_files : int, optional
        最多使用多少個文件（用於測試）
    save_dir : str
        模型保存目錄
    """
    print_header("使用 KataGo 數據訓練 Light-Go 模型")

    # 檢查數據目錄
    if not os.path.exists(data_dir):
        logger.error(f"數據目錄不存在：{data_dir}")
        logger.info("\n請先解壓數據：")
        logger.info("  cd data/sgf && tar -xzf 2026-01-07npzs.tgz")
        return

    # 查找 NPZ 文件
    print_header("步驟 1：載入數據")
    npz_files = find_npz_files(data_dir, max_files)

    if not npz_files:
        logger.error(f"在 {data_dir} 中沒有找到 .npz 文件")
        return

    logger.info(f"找到 {len(npz_files)} 個 .npz 文件")
    if max_files:
        logger.info(f"（限制：只使用前 {max_files} 個文件用於測試）")

    # 顯示前幾個文件
    for i, f in enumerate(npz_files[:3]):
        logger.info(f"  [{i+1}] {os.path.basename(f)}")
    if len(npz_files) > 3:
        logger.info(f"  ... 還有 {len(npz_files) - 3} 個文件")

    # 創建模型
    print_header("步驟 2：創建模型")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"使用裝置：{device}")

    if device == 'cpu':
        logger.warning("⚠️  使用 CPU 訓練會非常慢！強烈建議使用 GPU。")

    # 創建架構基因組
    genome = ArchitectureGenome(
        num_blocks=num_blocks,
        base_filters=num_filters,
        filters_per_block=[num_filters] * num_blocks,
        kernel_sizes=[3] * num_blocks,
        block_types=['residual'] * num_blocks,
        use_se_blocks=False,
        dropout_rate=0.0,
        num_input_planes=22,  # KataGo 格式
        board_size=19
    )

    logger.info(f"\n架構基因組：{genome}")
    logger.info(f"參數量：{genome.estimate_parameters():,}")

    # 構建模型
    model = genome.to_pytorch_model(device=device)

    # 創建優化器
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)

    # 訓練循環
    print_header("步驟 3：訓練模型")

    logger.info(f"\n訓練參數：")
    logger.info(f"  Epochs: {epochs}")
    logger.info(f"  Batch size: {batch_size}")
    logger.info(f"  Learning rate: {learning_rate}")
    logger.info(f"  數據文件: {len(npz_files)}")

    total_batches = 0
    epoch_losses = []

    for epoch in range(epochs):
        logger.info(f"\n{'='*60}")
        logger.info(f"Epoch {epoch + 1}/{epochs}")
        logger.info('='*60)

        model.train()
        epoch_loss = 0.0
        epoch_policy_loss = 0.0
        epoch_value_loss = 0.0
        batch_count = 0

        # 載入並訓練
        for positions, policies, values in load_npz_batch_simple(npz_files, batch_size):
            positions = positions.to(device)
            policies = policies.to(device)
            values = values.to(device)

            # 前向傳播
            policy_logits, value_pred = model(positions)

            # 計算損失
            policy_loss = nn.functional.cross_entropy(
                policy_logits,
                policies
            )
            # 確保形狀匹配：value_pred 是 (batch, 1)，values 是 (batch,)
            value_loss = nn.functional.mse_loss(
                value_pred.squeeze(-1),  # (batch, 1) -> (batch,)
                values.view(-1)  # 確保是 (batch,)
            )
            total_loss = policy_loss + value_loss

            # 反向傳播
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            # 記錄
            epoch_loss += total_loss.item()
            epoch_policy_loss += policy_loss.item()
            epoch_value_loss += value_loss.item()
            batch_count += 1
            total_batches += 1

            # 定期輸出
            if batch_count % 10 == 0:
                avg_loss = epoch_loss / batch_count
                logger.info(
                    f"  Batch {batch_count}: "
                    f"Loss={avg_loss:.4f} "
                    f"(Policy={epoch_policy_loss/batch_count:.4f}, "
                    f"Value={epoch_value_loss/batch_count:.4f})"
                )

        # Epoch 總結
        avg_epoch_loss = epoch_loss / max(batch_count, 1)
        avg_policy = epoch_policy_loss / max(batch_count, 1)
        avg_value = epoch_value_loss / max(batch_count, 1)

        epoch_losses.append(avg_epoch_loss)

        logger.info(f"\nEpoch {epoch + 1} 總結：")
        logger.info(f"  平均 Loss: {avg_epoch_loss:.4f}")
        logger.info(f"  Policy Loss: {avg_policy:.4f}")
        logger.info(f"  Value Loss: {avg_value:.4f}")
        logger.info(f"  批次數: {batch_count}")

    # 訓練總結
    print_header("步驟 4：訓練總結")

    logger.info(f"\n訓練完成！")
    logger.info(f"  總 Epochs: {epochs}")
    logger.info(f"  總批次: {total_batches}")
    logger.info(f"  初始 Loss: {epoch_losses[0]:.4f}")
    logger.info(f"  最終 Loss: {epoch_losses[-1]:.4f}")
    logger.info(f"  改進: {epoch_losses[0] - epoch_losses[-1]:.4f}")

    # 保存模型
    print_header("步驟 5：保存模型")

    os.makedirs(save_dir, exist_ok=True)

    model_path = os.path.join(save_dir, "model.pt")
    genome_path = os.path.join(save_dir, "genome.pkl")

    # 保存模型權重
    torch.save(model.state_dict(), model_path)
    logger.info(f"✅ 模型已保存：{model_path}")

    # 保存基因組
    import pickle
    with open(genome_path, 'wb') as f:
        pickle.dump(genome, f)
    logger.info(f"✅ 基因組已保存：{genome_path}")

    print_header("完成！")

    logger.info(f"""
🎉 訓練成功完成！

📁 保存的文件：
   • {model_path}
   • {genome_path}

🚀 下一步：
   1. 評估模型性能
   2. 實現 MCTS 讓模型自己下棋
   3. 開始架構演化（Phase 3）

💡 使用模型：
   >>> from hf_models.modeling_go_ai import GoAIModel
   >>> model = GoAIModel(num_input_planes=22)
   >>> model.load_state_dict(torch.load('{model_path}'))
    """)


def main():
    """主函數"""
    parser = argparse.ArgumentParser(
        description='使用 KataGo 訓練數據訓練 Light-Go 模型',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        '--data-dir',
        type=str,
        default='data/sgf/2026-01-07npzs/kata1-b28c512nbt-s12192929536-d5655876072',
        help='KataGo .npz 數據目錄'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=10,
        help='訓練輪數（默認：10）'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='批次大小（默認：32）'
    )
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=0.001,
        help='學習率（默認：0.001）'
    )
    parser.add_argument(
        '--num-blocks',
        type=int,
        default=10,
        help='殘差模塊數量（默認：10）'
    )
    parser.add_argument(
        '--num-filters',
        type=int,
        default=128,
        help='卷積層 filter 數量（默認：128）'
    )
    parser.add_argument(
        '--max-files',
        type=int,
        default=None,
        help='最多使用多少個文件（用於快速測試）'
    )
    parser.add_argument(
        '--save-dir',
        type=str,
        default='data/models/from_katago',
        help='模型保存目錄'
    )

    args = parser.parse_args()

    train_from_katago(
        data_dir=args.data_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_blocks=args.num_blocks,
        num_filters=args.num_filters,
        max_files=args.max_files,
        save_dir=args.save_dir
    )


if __name__ == "__main__":
    main()

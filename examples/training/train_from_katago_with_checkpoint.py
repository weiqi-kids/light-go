"""使用 KataGo 訓練數據訓練模型（帶檢查點支持）

這個腳本展示如何使用 KataGo 的 .npz 訓練數據來訓練 Light-Go 模型。
**增強功能**：支持訓練中斷後的檢查點恢復。

使用方法：
    # 基本用法（新訓練）
    python examples/train_from_katago_with_checkpoint.py

    # 從檢查點恢復訓練
    python examples/train_from_katago_with_checkpoint.py --resume

    # 從特定檢查點恢復
    python examples/train_from_katago_with_checkpoint.py \
        --resume \
        --checkpoint data/models/from_katago/checkpoint_epoch_5.pt

    # 自定義參數
    python examples/train_from_katago_with_checkpoint.py \
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


def save_checkpoint(
    epoch: int,
    model: nn.Module,
    optimizer: optim.Optimizer,
    genome: ArchitectureGenome,
    avg_epoch_loss: float,
    epoch_losses: List[float],
    save_dir: str,
    args: Any
):
    """保存訓練檢查點

    Parameters
    ----------
    epoch : int
        當前 epoch 數
    model : nn.Module
        模型
    optimizer : optim.Optimizer
        優化器
    genome : ArchitectureGenome
        架構基因組
    avg_epoch_loss : float
        當前 epoch 平均損失
    epoch_losses : List[float]
        所有 epoch 的損失歷史
    save_dir : str
        保存目錄
    args : Any
        命令行參數
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': avg_epoch_loss,
        'epoch_losses': epoch_losses,
        'genome': genome,
        'args': {
            'num_blocks': args.num_blocks,
            'num_filters': args.num_filters,
            'learning_rate': args.learning_rate,
            'batch_size': args.batch_size
        }
    }

    # 保存當前 epoch 檢查點
    checkpoint_path = os.path.join(save_dir, f'checkpoint_epoch_{epoch}.pt')
    torch.save(checkpoint, checkpoint_path)
    logger.info(f"💾 檢查點已保存：{checkpoint_path}")

    # 保存最新檢查點（覆蓋）
    latest_path = os.path.join(save_dir, 'checkpoint_latest.pt')
    torch.save(checkpoint, latest_path)
    logger.info(f"💾 最新檢查點已更新：{latest_path}")


def load_checkpoint(checkpoint_path: str, device: str):
    """載入訓練檢查點

    Parameters
    ----------
    checkpoint_path : str
        檢查點文件路徑
    device : str
        設備（'cpu' 或 'cuda'）

    Returns
    -------
    Dict
        檢查點字典
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"檢查點文件不存在：{checkpoint_path}")

    logger.info(f"📥 載入檢查點：{checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    logger.info(f"✅ 檢查點已載入：")
    logger.info(f"   Epoch: {checkpoint['epoch'] + 1}")
    logger.info(f"   Loss: {checkpoint['loss']:.4f}")
    logger.info(f"   訓練歷史: {len(checkpoint['epoch_losses'])} epochs")

    return checkpoint


def train_from_katago(
    data_dir: str,
    epochs: int = 10,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    num_blocks: int = 10,
    num_filters: int = 128,
    max_files: int = None,
    save_dir: str = "data/models/from_katago",
    resume: bool = False,
    checkpoint_path: str = None
):
    """從 KataGo 數據訓練模型（帶檢查點支持）

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
    resume : bool
        是否從檢查點恢復
    checkpoint_path : str, optional
        檢查點文件路徑（默認使用 checkpoint_latest.pt）
    """
    print_header("使用 KataGo 數據訓練 Light-Go 模型（帶檢查點）")

    # 檢查數據目錄
    if not os.path.exists(data_dir):
        logger.error(f"數據目錄不存在：{data_dir}")
        logger.info("\n請先解壓數據：")
        logger.info("  cd data/sgf && tar -xzf 2026-01-07npzs.tgz")
        return

    # 創建保存目錄
    os.makedirs(save_dir, exist_ok=True)

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
    print_header("步驟 2：創建/載入模型")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"使用裝置：{device}")

    if device == 'cpu':
        logger.warning("⚠️  使用 CPU 訓練會非常慢！強烈建議使用 GPU。")

    # 初始化變量
    start_epoch = 0
    epoch_losses = []

    # 嘗試從檢查點恢復
    if resume:
        if checkpoint_path is None:
            checkpoint_path = os.path.join(save_dir, 'checkpoint_latest.pt')

        try:
            checkpoint = load_checkpoint(checkpoint_path, device)

            # 從檢查點恢復架構基因組
            genome = checkpoint['genome']
            logger.info(f"從檢查點恢復架構基因組")

            # 構建模型
            model = genome.to_pytorch_model(device=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            logger.info(f"✅ 模型權重已恢復")

            # 創建優化器並恢復狀態
            optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            logger.info(f"✅ 優化器狀態已恢復")

            # 恢復訓練進度
            start_epoch = checkpoint['epoch'] + 1
            epoch_losses = checkpoint['epoch_losses']

            logger.info(f"\n🔄 從 Epoch {start_epoch + 1} 繼續訓練")
            logger.info(f"   之前訓練了 {len(epoch_losses)} 個 epochs")
            logger.info(f"   最後 loss: {checkpoint['loss']:.4f}")

        except FileNotFoundError as e:
            logger.error(f"❌ 檢查點文件不存在：{checkpoint_path}")
            logger.error(f"   請確認路徑或使用 --checkpoint 指定檢查點")
            return
        except Exception as e:
            logger.error(f"❌ 載入檢查點失敗：{e}")
            logger.error(f"   將開始新的訓練")
            resume = False

    # 如果不是恢復模式，創建新模型
    if not resume:
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
    logger.info(f"  總 Epochs: {epochs}")
    logger.info(f"  開始 Epoch: {start_epoch + 1}")
    logger.info(f"  剩餘 Epochs: {epochs - start_epoch}")
    logger.info(f"  Batch size: {batch_size}")
    logger.info(f"  Learning rate: {learning_rate}")
    logger.info(f"  數據文件: {len(npz_files)}")

    # 為了兼容性，創建 args 對象
    class Args:
        pass
    args = Args()
    args.num_blocks = num_blocks
    args.num_filters = num_filters
    args.learning_rate = learning_rate
    args.batch_size = batch_size

    total_batches = 0

    for epoch in range(start_epoch, epochs):
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

        # 保存檢查點（每個 epoch 結束後）
        save_checkpoint(
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            genome=genome,
            avg_epoch_loss=avg_epoch_loss,
            epoch_losses=epoch_losses,
            save_dir=save_dir,
            args=args
        )

    # 訓練總結
    print_header("步驟 4：訓練總結")

    logger.info(f"\n訓練完成！")
    logger.info(f"  總 Epochs: {epochs}")
    logger.info(f"  總批次: {total_batches}")
    logger.info(f"  初始 Loss: {epoch_losses[0]:.4f}")
    logger.info(f"  最終 Loss: {epoch_losses[-1]:.4f}")
    logger.info(f"  改進: {epoch_losses[0] - epoch_losses[-1]:.4f}")
    logger.info(f"  改進率: {(1 - epoch_losses[-1] / epoch_losses[0]) * 100:.1f}%")

    # 保存最終模型
    print_header("步驟 5：保存最終模型")

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
   • 最終模型: {model_path}
   • 基因組: {genome_path}
   • 檢查點: {save_dir}/checkpoint_epoch_*.pt
   • 最新檢查點: {save_dir}/checkpoint_latest.pt

🚀 下一步：
   1. 評估模型性能
   2. 實現 MCTS 讓模型自己下棋
   3. 開始架構演化（Phase 3）

💡 使用模型：
   >>> from hf_models.modeling_go_ai import GoAIModel
   >>> model = GoAIModel(num_input_planes=22)
   >>> model.load_state_dict(torch.load('{model_path}'))

💡 恢復訓練：
   python examples/train_from_katago_with_checkpoint.py --resume
    """)


def main():
    """主函數"""
    parser = argparse.ArgumentParser(
        description='使用 KataGo 訓練數據訓練 Light-Go 模型（帶檢查點支持）',
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
    parser.add_argument(
        '--resume',
        action='store_true',
        help='從檢查點恢復訓練'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        default=None,
        help='檢查點文件路徑（默認：使用 checkpoint_latest.pt）'
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
        save_dir=args.save_dir,
        resume=args.resume,
        checkpoint_path=args.checkpoint
    )


if __name__ == "__main__":
    main()

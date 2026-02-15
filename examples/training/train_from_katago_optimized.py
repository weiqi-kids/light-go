"""優化版 KataGo 訓練腳本

性能優化：
1. PyTorch DataLoader（多線程數據加載）
2. 混合精度訓練（AMP）- 如果支持
3. 梯度累積
4. 更高效的數據預處理
5. 批次大小自適應
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import List
import glob
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np

# 添加專案根目錄到路徑
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from hf_models.modeling_go_ai import GoAIModel
from core.architecture_genome import ArchitectureGenome

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class KataGoDataset(Dataset):
    """優化的 KataGo 數據集

    使用內存映射和延遲加載提高效率
    """

    def __init__(self, npz_files: List[str], max_samples_per_file: int = 40):
        """初始化數據集

        Parameters
        ----------
        npz_files : List[str]
            NPZ 文件列表
        max_samples_per_file : int
            每個文件的最大樣本數
        """
        self.npz_files = npz_files
        self.max_samples_per_file = max_samples_per_file

        # 計算總樣本數（粗略估計）
        self.samples_per_file = []
        self.cumulative_samples = [0]

        logger.info(f"初始化數據集，掃描 {len(npz_files)} 個文件...")

        for i, npz_file in enumerate(npz_files):
            try:
                # 快速讀取文件大小（不載入數據）
                data = np.load(npz_file, mmap_mode='r')
                num_samples = min(data['binaryInputNCHWPacked'].shape[0], max_samples_per_file)
                self.samples_per_file.append(num_samples)
                self.cumulative_samples.append(self.cumulative_samples[-1] + num_samples)
            except Exception as e:
                logger.warning(f"跳過文件 {os.path.basename(npz_file)}: {e}")
                self.samples_per_file.append(0)
                self.cumulative_samples.append(self.cumulative_samples[-1])

        self.total_samples = self.cumulative_samples[-1]
        logger.info(f"✅ 數據集初始化完成：{self.total_samples:,} 個樣本")

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        """獲取單個樣本

        Parameters
        ----------
        idx : int
            樣本索引

        Returns
        -------
        tuple
            (features, policy, value)
        """
        # 找到對應的文件
        file_idx = 0
        for i, cum_samples in enumerate(self.cumulative_samples[1:]):
            if idx < cum_samples:
                file_idx = i
                break

        # 計算文件內索引
        local_idx = idx - self.cumulative_samples[file_idx]

        # 載入數據
        try:
            data = np.load(self.npz_files[file_idx])

            # 解碼輸入
            binary_packed = data['binaryInputNCHWPacked'][local_idx]
            binary_input = np.unpackbits(binary_packed, axis=1)
            binary_input = binary_input[:, :19*19]
            binary_input = binary_input.reshape(binary_input.shape[0], 19, 19).astype(np.float32)

            # 提取 policy
            policy_targets = data['policyTargetsNCMove'][local_idx].astype(np.float32)
            visit_counts = policy_targets[0, :]
            total_visits = visit_counts.sum()
            policy = visit_counts / max(total_visits, 1.0)

            # 提取 value
            value_targets = data['valueTargetsNCHW'][local_idx].astype(np.float32)
            value = value_targets[0, :, :].mean()
            value = np.clip(value, -1.0, 1.0)

            return (
                torch.from_numpy(binary_input),
                torch.from_numpy(policy),
                torch.tensor(value, dtype=torch.float32)
            )

        except Exception as e:
            logger.warning(f"讀取樣本 {idx} 失敗: {e}")
            # 返回零數據（備用）
            return (
                torch.zeros(22, 19, 19, dtype=torch.float32),
                torch.zeros(362, dtype=torch.float32),
                torch.tensor(0.0, dtype=torch.float32)
            )


def find_npz_files(data_dir: str, max_files: int = None) -> List[str]:
    """查找 NPZ 文件"""
    pattern = os.path.join(data_dir, "**/*.npz")
    files = glob.glob(pattern, recursive=True)

    if max_files:
        files = files[:max_files]

    return files


def train_optimized(
    data_dir: str,
    epochs: int = 10,
    batch_size: int = 64,
    learning_rate: float = 0.001,
    num_blocks: int = 10,
    num_filters: int = 128,
    max_files: int = None,
    save_dir: str = "data/models/from_katago_optimized",
    num_workers: int = 4,
    use_amp: bool = True,
    gradient_accumulation_steps: int = 1
):
    """優化版訓練

    Parameters
    ----------
    data_dir : str
        數據目錄
    epochs : int
        訓練輪數
    batch_size : int
        批次大小
    learning_rate : float
        學習率
    num_blocks : int
        殘差塊數量
    num_filters : int
        濾波器數量
    max_files : int
        最大文件數
    save_dir : str
        保存目錄
    num_workers : int
        數據載入線程數
    use_amp : bool
        是否使用混合精度訓練
    gradient_accumulation_steps : int
        梯度累積步數
    """
    logger.info("="*70)
    logger.info("  優化版 KataGo 訓練")
    logger.info("="*70)

    # 檢查設備
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"\n使用裝置：{device}")

    if device == 'cpu':
        logger.warning("⚠️  使用 CPU 訓練會很慢！")
        use_amp = False  # CPU 不支持 AMP

    # 查找文件
    logger.info(f"\n步驟 1：載入數據")
    npz_files = find_npz_files(data_dir, max_files)

    if not npz_files:
        logger.error(f"在 {data_dir} 中沒有找到 NPZ 文件")
        return

    logger.info(f"找到 {len(npz_files)} 個文件")

    # 創建數據集和 DataLoader
    logger.info(f"\n創建 DataLoader（{num_workers} 個工作線程）...")
    dataset = KataGoDataset(npz_files)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device == 'cuda'),
        prefetch_factor=2 if num_workers > 0 else None
    )

    logger.info(f"✅ DataLoader 已創建")
    logger.info(f"   總樣本：{len(dataset):,}")
    logger.info(f"   批次數：{len(dataloader):,}")

    # 創建模型
    logger.info(f"\n步驟 2：創建模型")

    genome = ArchitectureGenome(
        num_blocks=num_blocks,
        base_filters=num_filters,
        filters_per_block=[num_filters] * num_blocks,
        kernel_sizes=[3] * num_blocks,
        block_types=['residual'] * num_blocks,
        use_se_blocks=False,
        dropout_rate=0.0,
        num_input_planes=22,
        board_size=19
    )

    logger.info(f"架構：{genome}")
    logger.info(f"參數量：{genome.estimate_parameters():,}")

    model = genome.to_pytorch_model(device=device)

    # 創建優化器
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)

    # 學習率調度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2, verbose=True
    )

    # 混合精度訓練
    scaler = None
    if use_amp and device == 'cuda':
        scaler = torch.cuda.amp.GradScaler()
        logger.info("✅ 啟用混合精度訓練（AMP）")

    # 訓練循環
    logger.info(f"\n步驟 3：開始訓練")
    logger.info(f"  Epochs: {epochs}")
    logger.info(f"  Batch size: {batch_size}")
    logger.info(f"  Learning rate: {learning_rate}")
    logger.info(f"  Gradient accumulation: {gradient_accumulation_steps}")
    logger.info(f"  Data workers: {num_workers}")

    best_loss = float('inf')
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

        epoch_start = time.time()

        # 訓練
        for batch_idx, (positions, policies, values) in enumerate(dataloader):
            positions = positions.to(device)
            policies = policies.to(device)
            values = values.to(device)

            # 混合精度訓練
            if use_amp and scaler:
                with torch.cuda.amp.autocast():
                    policy_logits, value_pred = model(positions)

                    policy_loss = nn.functional.cross_entropy(policy_logits, policies)
                    value_loss = nn.functional.mse_loss(
                        value_pred.squeeze(-1),
                        values.view(-1)
                    )
                    total_loss = policy_loss + value_loss

                # 梯度累積
                total_loss = total_loss / gradient_accumulation_steps

                scaler.scale(total_loss).backward()

                if (batch_idx + 1) % gradient_accumulation_steps == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()

            else:
                # 標準訓練
                policy_logits, value_pred = model(positions)

                policy_loss = nn.functional.cross_entropy(policy_logits, policies)
                value_loss = nn.functional.mse_loss(
                    value_pred.squeeze(-1),
                    values.view(-1)
                )
                total_loss = policy_loss + value_loss

                # 梯度累積
                total_loss = total_loss / gradient_accumulation_steps
                total_loss.backward()

                if (batch_idx + 1) % gradient_accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()

            # 記錄
            epoch_loss += total_loss.item() * gradient_accumulation_steps
            epoch_policy_loss += policy_loss.item()
            epoch_value_loss += value_loss.item()
            batch_count += 1

            # 定期輸出
            if (batch_idx + 1) % 100 == 0:
                avg_loss = epoch_loss / batch_count
                elapsed = time.time() - epoch_start
                batches_per_sec = batch_count / elapsed

                logger.info(
                    f"  Batch {batch_idx + 1}/{len(dataloader)}: "
                    f"Loss={avg_loss:.4f} "
                    f"(Policy={epoch_policy_loss/batch_count:.4f}, "
                    f"Value={epoch_value_loss/batch_count:.4f}) "
                    f"[{batches_per_sec:.2f} batch/s]"
                )

        # Epoch 總結
        epoch_time = time.time() - epoch_start
        avg_epoch_loss = epoch_loss / max(batch_count, 1)
        avg_policy = epoch_policy_loss / max(batch_count, 1)
        avg_value = epoch_value_loss / max(batch_count, 1)

        epoch_losses.append(avg_epoch_loss)

        logger.info(f"\nEpoch {epoch + 1} 總結：")
        logger.info(f"  平均 Loss: {avg_epoch_loss:.4f}")
        logger.info(f"  Policy Loss: {avg_policy:.4f}")
        logger.info(f"  Value Loss: {avg_value:.4f}")
        logger.info(f"  時間：{epoch_time:.1f} 秒")
        logger.info(f"  吞吐量：{batch_count/epoch_time:.2f} batch/s")

        # 學習率調整
        scheduler.step(avg_epoch_loss)

        # 保存最佳模型
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            os.makedirs(save_dir, exist_ok=True)
            model_path = os.path.join(save_dir, "best_model.pt")
            torch.save(model.state_dict(), model_path)
            logger.info(f"  ✅ 保存最佳模型（Loss={best_loss:.4f}）")

    # 訓練完成
    logger.info(f"\n{'='*60}")
    logger.info("  訓練完成！")
    logger.info('='*60)
    logger.info(f"  最佳 Loss: {best_loss:.4f}")
    logger.info(f"  最終 Loss: {epoch_losses[-1]:.4f}")
    logger.info(f"  改進: {epoch_losses[0] - epoch_losses[-1]:.4f}")

    # 保存最終模型
    os.makedirs(save_dir, exist_ok=True)
    final_model_path = os.path.join(save_dir, "final_model.pt")
    torch.save(model.state_dict(), final_model_path)

    import pickle
    genome_path = os.path.join(save_dir, "genome.pkl")
    with open(genome_path, 'wb') as f:
        pickle.dump(genome, f)

    logger.info(f"\n✅ 模型已保存至：{save_dir}")


def main():
    """主函數"""
    parser = argparse.ArgumentParser(description='優化版 KataGo 訓練')

    parser.add_argument('--data-dir', type=str,
                       default='data/sgf/2026-01-07npzs/kata1-b28c512nbt-s12192929536-d5655876072')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--learning-rate', type=float, default=0.001)
    parser.add_argument('--num-blocks', type=int, default=10)
    parser.add_argument('--num-filters', type=int, default=128)
    parser.add_argument('--max-files', type=int, default=None)
    parser.add_argument('--save-dir', type=str, default='data/models/from_katago_optimized')
    parser.add_argument('--num-workers', type=int, default=4, help='數據載入線程數')
    parser.add_argument('--no-amp', action='store_true', help='禁用混合精度訓練')
    parser.add_argument('--gradient-accumulation', type=int, default=1,
                       help='梯度累積步數（模擬更大 batch）')

    args = parser.parse_args()

    train_optimized(
        data_dir=args.data_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_blocks=args.num_blocks,
        num_filters=args.num_filters,
        max_files=args.max_files,
        save_dir=args.save_dir,
        num_workers=args.num_workers,
        use_amp=not args.no_amp,
        gradient_accumulation_steps=args.gradient_accumulation
    )


if __name__ == "__main__":
    main()

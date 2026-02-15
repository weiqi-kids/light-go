"""訓練 LightGo 基礎模型（2 Planes: Signed liberties + Forbidden points）

這是分階段訓練的第一階段，訓練輕量級的 2-plane 模型。
模型將學習基本的圍棋戰術：氣數判斷、吃子、做活等。

訓練完成後，權重可以遷移到 4-plane 模型（添加 Ladder 特徵）。
"""

import os
import sys
import logging
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path

# 添加專案根目錄到路徑
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from hf_models.modeling_go_ai import GoAIModel
from input.katago_to_input import process_katago_file
from input.sgf_to_input import parse_sgf

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def convert_lightgo_to_tensor(liberty_list, forbidden_list, board_size=19):
    """將 LightGo 格式轉換為 2-plane tensor

    Args:
        liberty_list: List[(x, y, signed_liberties)]
            黑子：liberties > 0
            白子：liberties < 0
            空點：不在列表中（視為 0）
        forbidden_list: List[(x, y)]
        board_size: 棋盤大小

    Returns:
        tensor: (2, board_size, board_size)
            Plane 0: Signed liberties
            Plane 1: Forbidden points
    """
    planes = np.zeros((2, board_size, board_size), dtype=np.float32)

    # Plane 0: Signed liberties
    for x, y, lib_count in liberty_list:
        # 注意：liberty_list 使用 1-indexed，需要轉換
        if 1 <= x <= board_size and 1 <= y <= board_size:
            planes[0, y-1, x-1] = lib_count  # 直接使用 signed value

    # Plane 1: Forbidden points
    for x, y in forbidden_list:
        # 注意：forbidden_list 可能使用 0-indexed 或 1-indexed，需要檢查
        # 這裡假設是 0-indexed（從 katago_to_input.py）
        if 0 <= x < board_size and 0 <= y < board_size:
            planes[1, y, x] = 1.0

    return torch.from_numpy(planes)


def load_katago_jsonl_data(data_dir, max_files=None):
    """從 KataGo JSONL 文件載入 LightGo 2-plane 訓練數據

    Args:
        data_dir: 包含 .jsonl 文件的目錄
        max_files: 最大文件數（用於測試）

    Returns:
        List of (position_tensor, policy_target, value_target)
    """
    training_data = []
    jsonl_files = list(Path(data_dir).glob("*.jsonl"))

    if max_files:
        jsonl_files = jsonl_files[:max_files]

    logger.info(f"找到 {len(jsonl_files)} 個 JSONL 文件")

    for jsonl_file in jsonl_files:
        logger.info(f"處理 {jsonl_file.name}...")

        try:
            processed_moves = process_katago_file(str(jsonl_file))

            for move_data in processed_moves:
                liberty_list = move_data['liberty']
                forbidden_list = move_data['forbidden']
                metadata = move_data['metadata']

                # 轉換為 2-plane tensor
                position = convert_lightgo_to_tensor(liberty_list, forbidden_list)

                # TODO: 從 metadata 提取 policy 和 value targets
                # 這裡暫時用 placeholder
                board_size = metadata.get('rules', {}).get('board_size', 19)
                policy = np.zeros(board_size * board_size + 1, dtype=np.float32)
                policy[0] = 1.0  # Placeholder

                value = 0.0  # Placeholder

                training_data.append({
                    'position': position,
                    'policy': torch.from_numpy(policy),
                    'value': torch.tensor(value, dtype=torch.float32)
                })

        except Exception as e:
            logger.warning(f"跳過 {jsonl_file.name}: {e}")
            continue

    logger.info(f"總共載入 {len(training_data)} 個訓練樣本")
    return training_data


def load_sgf_data(sgf_dir, max_files=None):
    """從 SGF 文件載入 LightGo 2-plane 訓練數據

    Args:
        sgf_dir: 包含 .sgf 文件的目錄
        max_files: 最大文件數

    Returns:
        List of training samples
    """
    from core.liberty import count_liberties

    training_data = []
    sgf_files = list(Path(sgf_dir).glob("*.sgf"))

    if max_files:
        sgf_files = sgf_files[:max_files]

    logger.info(f"找到 {len(sgf_files)} 個 SGF 文件")

    for sgf_file in sgf_files:
        logger.info(f"處理 {sgf_file.name}...")

        try:
            matrix, metadata, board = parse_sgf(str(sgf_file))

            # 計算 liberty
            liberty_list = count_liberties(matrix)

            # TODO: 計算 forbidden points
            forbidden_list = []

            # 轉換為 2-plane tensor
            board_size = len(matrix)
            position = convert_lightgo_to_tensor(liberty_list, forbidden_list, board_size)

            # Placeholder targets
            policy = np.zeros(board_size * board_size + 1, dtype=np.float32)
            policy[0] = 1.0
            value = 0.0

            training_data.append({
                'position': position,
                'policy': torch.from_numpy(policy),
                'value': torch.tensor(value, dtype=torch.float32)
            })

        except Exception as e:
            logger.warning(f"跳過 {sgf_file.name}: {e}")
            continue

    logger.info(f"總共載入 {len(training_data)} 個訓練樣本")
    return training_data


def train_2plane_model(
    data_dir,
    data_type='katago_jsonl',  # 'katago_jsonl' or 'sgf'
    epochs=10,
    batch_size=64,
    learning_rate=0.001,
    num_blocks=10,
    num_filters=128,
    output_dir='data/models/lightgo_2planes',
    device=None
):
    """訓練 LightGo 2-plane 基礎模型"""

    # 設置設備
    if device is None:
        if torch.cuda.is_available():
            device = 'cuda'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'

    logger.info(f"使用設備: {device}")

    # 創建模型（2 input planes）
    model = GoAIModel(
        num_input_planes=2,  # ← LightGo 基礎設計！
        num_filters=num_filters,
        num_blocks=num_blocks,
        board_size=19
    ).to(device)

    logger.info(f"模型架構: {num_blocks} blocks, {num_filters} filters, 2 input planes")

    # 載入數據
    logger.info(f"從 {data_dir} 載入數據...")
    if data_type == 'katago_jsonl':
        training_data = load_katago_jsonl_data(data_dir)
    elif data_type == 'sgf':
        training_data = load_sgf_data(data_dir)
    else:
        raise ValueError(f"不支援的數據類型: {data_type}")

    if not training_data:
        logger.error("沒有訓練數據！")
        return

    # 優化器
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 訓練循環
    model.train()

    for epoch in range(epochs):
        total_loss = 0.0
        total_policy_loss = 0.0
        total_value_loss = 0.0
        num_batches = 0

        # 簡單的批次訓練（可以改用 DataLoader）
        for i in range(0, len(training_data), batch_size):
            batch_data = training_data[i:i+batch_size]

            # 組裝批次
            positions = torch.stack([d['position'] for d in batch_data]).to(device)
            policies = torch.stack([d['policy'] for d in batch_data]).to(device)
            values = torch.stack([d['value'] for d in batch_data]).to(device)

            # 前向傳播
            policy_logits, value_pred = model(positions)

            # 計算損失
            policy_loss = -(policies * torch.log_softmax(policy_logits, dim=1)).sum(dim=1).mean()
            value_loss = nn.functional.mse_loss(value_pred.squeeze(), values)
            loss = policy_loss + value_loss

            # 反向傳播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 統計
            total_loss += loss.item()
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            num_batches += 1

            if (num_batches % 50) == 0:
                logger.info(
                    f"Epoch {epoch+1}/{epochs}, Batch {num_batches}, "
                    f"Loss: {loss.item():.4f} "
                    f"(Policy: {policy_loss.item():.4f}, Value: {value_loss.item():.4f})"
                )

        # Epoch 統計
        avg_loss = total_loss / num_batches
        avg_policy_loss = total_policy_loss / num_batches
        avg_value_loss = total_value_loss / num_batches

        logger.info(
            f"Epoch {epoch+1}/{epochs} 完成 - "
            f"Avg Loss: {avg_loss:.4f} "
            f"(Policy: {avg_policy_loss:.4f}, Value: {avg_value_loss:.4f})"
        )

    # 保存模型
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, 'model_2planes.pt')

    torch.save({
        'model_state_dict': model.state_dict(),
        'model_config': {
            'num_input_planes': 2,
            'num_filters': num_filters,
            'num_blocks': num_blocks,
            'board_size': 19
        },
        'training_info': {
            'epochs': epochs,
            'final_loss': avg_loss
        }
    }, model_path)

    logger.info(f"模型已保存到 {model_path}")
    logger.info("✅ LightGo 2-plane 基礎模型訓練完成！")
    logger.info("💡 下一步：使用 train_lightgo_4planes.py 進行強化訓練")


def main():
    parser = argparse.ArgumentParser(description='訓練 LightGo 2-plane 基礎模型')
    parser.add_argument('--data-dir', type=str, required=True,
                        help='訓練數據目錄（JSONL 或 SGF）')
    parser.add_argument('--data-type', type=str, default='katago_jsonl',
                        choices=['katago_jsonl', 'sgf'],
                        help='數據類型')
    parser.add_argument('--epochs', type=int, default=10,
                        help='訓練輪數')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='批次大小')
    parser.add_argument('--learning-rate', type=float, default=0.001,
                        help='學習率')
    parser.add_argument('--num-blocks', type=int, default=10,
                        help='殘差塊數量')
    parser.add_argument('--num-filters', type=int, default=128,
                        help='濾波器數量')
    parser.add_argument('--output-dir', type=str, default='data/models/lightgo_2planes',
                        help='輸出目錄')
    parser.add_argument('--device', type=str, default=None,
                        choices=['cuda', 'mps', 'cpu'],
                        help='訓練設備')

    args = parser.parse_args()

    train_2plane_model(
        data_dir=args.data_dir,
        data_type=args.data_type,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_blocks=args.num_blocks,
        num_filters=args.num_filters,
        output_dir=args.output_dir,
        device=args.device
    )


if __name__ == '__main__':
    main()

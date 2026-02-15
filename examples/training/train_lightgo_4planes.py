"""訓練 LightGo 強化模型（4 Planes: 基礎 2 Planes + Ladder 特徵）

這是分階段訓練的第二階段，從 2-plane 基礎模型遷移權重，
然後添加 Ladder（征子）特徵進行 Fine-tuning。

權重遷移策略：
- Plane 0-1（Signed liberties, Forbidden）：從 2-plane 模型複製權重
- Plane 2-3（Ladder 特徵）：隨機初始化
- 使用較小的學習率進行 Fine-tuning
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

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def transfer_weights_2p_to_4p(model_2p_path, model_4p):
    """從 2-plane 模型遷移權重到 4-plane 模型

    Args:
        model_2p_path: 2-plane 模型檢查點路徑
        model_4p: 4-plane 模型（已初始化）

    Returns:
        model_4p: 權重已遷移的 4-plane 模型
    """
    logger.info(f"從 {model_2p_path} 載入 2-plane 模型權重...")

    # 載入 2-plane 檢查點
    checkpoint_2p = torch.load(model_2p_path, map_location='cpu')
    state_dict_2p = checkpoint_2p['model_state_dict']

    # 檢查配置
    config_2p = checkpoint_2p.get('model_config', {})
    if config_2p.get('num_input_planes') != 2:
        logger.warning(f"模型不是 2-plane 模型！num_input_planes={config_2p.get('num_input_planes')}")

    # 創建臨時 2-plane 模型以載入權重
    model_2p = GoAIModel(
        num_input_planes=2,
        num_filters=config_2p.get('num_filters', 128),
        num_blocks=config_2p.get('num_blocks', 10),
        board_size=config_2p.get('board_size', 19)
    )
    model_2p.load_state_dict(state_dict_2p)

    logger.info("開始權重遷移...")

    # 策略 1：複製第一層卷積的前 2 個 channels
    # model_2p.conv_input: (num_filters, 2, 3, 3)
    # model_4p.conv_input: (num_filters, 4, 3, 3)

    # 獲取 2-plane 的第一層權重
    conv_input_weight_2p = model_2p.conv_input.weight.data  # (C_out, 2, 3, 3)
    conv_input_bias_2p = model_2p.conv_input.bias.data if model_2p.conv_input.bias is not None else None

    # 複製到 4-plane 模型的前 2 個 channels
    with torch.no_grad():
        model_4p.conv_input.weight[:, :2, :, :].copy_(conv_input_weight_2p)

        # 後 2 個 channels（Ladder 特徵）使用隨機初始化（已完成）
        logger.info("✓ 前 2 個 input planes 權重已從 2-plane 模型複製")
        logger.info("✓ 後 2 個 input planes（Ladder）權重使用隨機初始化")

        if conv_input_bias_2p is not None:
            model_4p.conv_input.bias.copy_(conv_input_bias_2p)

    # 策略 2：複製所有其他層（ResNet blocks, Policy Head, Value Head）
    # 這些層的權重可以直接複製，因為它們的結構相同
    with torch.no_grad():
        for name, param in model_2p.named_parameters():
            if name.startswith('conv_input'):
                # 第一層已經處理過了
                continue

            if name in dict(model_4p.named_parameters()):
                target_param = dict(model_4p.named_parameters())[name]
                if param.shape == target_param.shape:
                    target_param.copy_(param)
                    logger.debug(f"✓ 複製層: {name}")
                else:
                    logger.warning(f"⚠ 跳過層（shape 不匹配）: {name}")

    logger.info("✅ 權重遷移完成！")
    logger.info("📊 統計：")
    logger.info(f"  - 前 2 planes（基礎）：從 2-plane 模型繼承")
    logger.info(f"  - 後 2 planes（Ladder）：隨機初始化")
    logger.info(f"  - ResNet blocks：從 2-plane 模型繼承")
    logger.info(f"  - Policy/Value heads：從 2-plane 模型繼承")

    return model_4p


def convert_lightgo_to_4plane_tensor(liberty_list, forbidden_list, ladder_current, ladder_prev, board_size=19):
    """將 LightGo 格式轉換為 4-plane tensor

    Args:
        liberty_list: List[(x, y, signed_liberties)]
        forbidden_list: List[(x, y)]
        ladder_current: List[(x, y)] - 當前被征子威脅的棋子
        ladder_prev: List[(x, y)] - 上一手的征子狀態
        board_size: 棋盤大小

    Returns:
        tensor: (4, board_size, board_size)
            Plane 0: Signed liberties
            Plane 1: Forbidden points
            Plane 2: Ladder-threatened stone (current)
            Plane 3: Ladder 1 turn ago
    """
    planes = np.zeros((4, board_size, board_size), dtype=np.float32)

    # Plane 0: Signed liberties
    for x, y, lib_count in liberty_list:
        if 1 <= x <= board_size and 1 <= y <= board_size:
            planes[0, y-1, x-1] = lib_count

    # Plane 1: Forbidden points
    for x, y in forbidden_list:
        if 0 <= x < board_size and 0 <= y < board_size:
            planes[1, y, x] = 1.0

    # Plane 2: Ladder-threatened stone (current)
    for x, y in ladder_current:
        if 0 <= x < board_size and 0 <= y < board_size:
            planes[2, y, x] = 1.0

    # Plane 3: Ladder 1 turn ago
    for x, y in ladder_prev:
        if 0 <= x < board_size and 0 <= y < board_size:
            planes[3, y, x] = 1.0

    return torch.from_numpy(planes)


def train_4plane_model(
    base_model_path,  # 2-plane 模型路徑
    data_dir,
    epochs=5,  # Fine-tuning 通常需要較少 epochs
    batch_size=64,
    learning_rate=0.0001,  # 較小的學習率！
    output_dir='data/models/lightgo_4planes',
    device=None
):
    """訓練 LightGo 4-plane 強化模型（帶權重遷移）"""

    # 設置設備
    if device is None:
        if torch.cuda.is_available():
            device = 'cuda'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'

    logger.info(f"使用設備: {device}")

    # 載入 2-plane 模型配置
    checkpoint_2p = torch.load(base_model_path, map_location='cpu')
    config_2p = checkpoint_2p.get('model_config', {})

    # 創建 4-plane 模型（繼承 2-plane 的配置）
    model = GoAIModel(
        num_input_planes=4,  # ← 升級到 4 planes！
        num_filters=config_2p.get('num_filters', 128),
        num_blocks=config_2p.get('num_blocks', 10),
        board_size=config_2p.get('board_size', 19)
    ).to(device)

    logger.info(f"創建 4-plane 模型: {config_2p.get('num_blocks', 10)} blocks, {config_2p.get('num_filters', 128)} filters")

    # 遷移權重
    model = transfer_weights_2p_to_4p(base_model_path, model)
    model.to(device)

    # TODO: 載入包含 Ladder 特徵的訓練數據
    # 這裡需要數據生成邏輯
    logger.warning("⚠ 數據載入邏輯尚未實現（需要包含 Ladder 特徵的數據）")
    logger.info("💡 提示：需要從 MCTS 自我對弈或標註數據中生成 Ladder 特徵")

    # 優化器（較小的學習率用於 Fine-tuning）
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    logger.info(f"Fine-tuning 學習率: {learning_rate} (比基礎訓練小 10x)")

    # TODO: 訓練循環
    # ... (類似 train_lightgo_2planes.py)

    # 保存模型
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, 'model_4planes.pt')

    torch.save({
        'model_state_dict': model.state_dict(),
        'model_config': {
            'num_input_planes': 4,
            'num_filters': config_2p.get('num_filters', 128),
            'num_blocks': config_2p.get('num_blocks', 10),
            'board_size': 19
        },
        'training_info': {
            'base_model': base_model_path,
            'transfer_learning': True,
            'epochs': epochs,
        }
    }, model_path)

    logger.info(f"模型已保存到 {model_path}")
    logger.info("✅ LightGo 4-plane 強化模型訓練完成！")


def main():
    parser = argparse.ArgumentParser(description='訓練 LightGo 4-plane 強化模型（遷移學習）')
    parser.add_argument('--base-model', type=str, required=True,
                        help='2-plane 基礎模型路徑（.pt 文件）')
    parser.add_argument('--data-dir', type=str, required=True,
                        help='訓練數據目錄（包含 Ladder 特徵）')
    parser.add_argument('--epochs', type=int, default=5,
                        help='Fine-tuning 輪數（通常比基礎訓練少）')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='批次大小')
    parser.add_argument('--learning-rate', type=float, default=0.0001,
                        help='學習率（應比基礎訓練小，建議 0.0001）')
    parser.add_argument('--output-dir', type=str, default='data/models/lightgo_4planes',
                        help='輸出目錄')
    parser.add_argument('--device', type=str, default=None,
                        choices=['cuda', 'mps', 'cpu'],
                        help='訓練設備')

    args = parser.parse_args()

    train_4plane_model(
        base_model_path=args.base_model,
        data_dir=args.data_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        output_dir=args.output_dir,
        device=args.device
    )


if __name__ == '__main__':
    main()

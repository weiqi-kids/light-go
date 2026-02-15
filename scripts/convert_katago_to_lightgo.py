#!/usr/bin/env python3
"""將 KataGo 訓練數據轉換為 LightGo 2-Plane 格式

從 KataGo 22-plane NPZ 文件轉換為 LightGo 2-plane NPZ 文件
"""

import os
import sys
import numpy as np
import argparse
import logging
from pathlib import Path
from tqdm import tqdm

# 添加專案根目錄
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from input.lightgo_encoder import convert_katago_to_lightgo

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def convert_single_file(input_path: str, output_path: str):
    """轉換單個 NPZ 文件"""
    # 載入 KataGo 數據
    data = np.load(input_path)

    # KataGo NPZ 格式
    # - 'binaryInputNCHWPacked': 棋盤狀態 (22 planes)
    # - 'globalInputNC': 全局特徵
    # - 'policyTargetsNCMove': 策略標籤
    # - 'globalTargetsNC': 價值標籤
    # - 'scoreDistrN': 分數分布

    inputs = data['binaryInputNCHWPacked']  # (N, 22, packed)
    policy_targets_raw = data['policyTargetsNCMove']  # (N, 2, 362) 或 (N, 362)
    global_targets = data['globalTargetsNC']  # (N, 64)

    # 處理 policy_targets 格式
    # KataGo 的 policy 可能是 (N, 2, 362) 格式（黑/白視角）
    # 我們取第一個視角（黑棋），並歸一化為機率分布
    if policy_targets_raw.ndim == 3:
        policy_targets = policy_targets_raw[:, 0, :]  # 取黑棋視角 (N, 362)
    else:
        policy_targets = policy_targets_raw

    # 歸一化為機率分布
    policy_sum = policy_targets.sum(axis=1, keepdims=True)
    policy_sum = np.maximum(policy_sum, 1e-6)  # 避免除以零
    policy_targets = policy_targets / policy_sum  # 歸一化

    num_samples = inputs.shape[0]
    logger.info(f"轉換 {num_samples} 個樣本...")

    # 轉換為 LightGo 格式
    lightgo_inputs = []
    for i in tqdm(range(num_samples), desc="Converting"):
        katago_planes = inputs[i]  # (22, 19, 19)
        lightgo_planes = convert_katago_to_lightgo(katago_planes, board_size=19)
        lightgo_inputs.append(lightgo_planes.numpy())

    lightgo_inputs = np.array(lightgo_inputs)  # (N, 2, 19, 19)

    # 保存
    np.savez_compressed(
        output_path,
        inputs=lightgo_inputs,  # (N, 2, 19, 19)
        policy_targets=policy_targets,  # (N, 362)
        value_targets=global_targets[:, 0:1],  # (N, 1) - 只取價值
    )

    logger.info(f"✅ 已保存: {output_path}")
    logger.info(f"   Input shape: {lightgo_inputs.shape} (KataGo: {inputs.shape})")
    logger.info(f"   Size reduction: {inputs.nbytes / lightgo_inputs.nbytes:.1f}x")


def convert_directory(input_dir: str, output_dir: str, max_files: int = None):
    """轉換整個目錄"""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 找到所有 NPZ 文件（遞迴搜尋）
    npz_files = list(input_path.glob("**/*.npz"))

    if max_files:
        npz_files = npz_files[:max_files]

    logger.info(f"找到 {len(npz_files)} 個 NPZ 文件")
    logger.info(f"輸出目錄: {output_path}")

    for npz_file in npz_files:
        output_file = output_path / npz_file.name
        if output_file.exists():
            logger.info(f"⏭️  跳過（已存在）: {npz_file.name}")
            continue

        try:
            convert_single_file(str(npz_file), str(output_file))
        except Exception as e:
            logger.error(f"❌ 轉換失敗: {npz_file.name} - {e}")


def main():
    parser = argparse.ArgumentParser(description="將 KataGo 數據轉換為 LightGo 格式")
    parser.add_argument(
        '--input-dir',
        type=str,
        default='data/katago/training_data/npz/2026-01-07_batch1',
        help='KataGo NPZ 文件目錄'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/lightgo/training_data/katago_converted/2026-01-07_batch1',
        help='LightGo NPZ 輸出目錄'
    )
    parser.add_argument(
        '--max-files',
        type=int,
        default=None,
        help='最多轉換文件數（測試用）'
    )

    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("KataGo → LightGo 數據轉換")
    logger.info("=" * 60)
    logger.info(f"輸入: {args.input_dir}")
    logger.info(f"輸出: {args.output_dir}")

    convert_directory(args.input_dir, args.output_dir, args.max_files)

    logger.info("\n" + "=" * 60)
    logger.info("✅ 轉換完成！")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()

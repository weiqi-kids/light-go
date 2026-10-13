#!/usr/bin/env python3
"""KataGo NPZ 批量轉換腳本

將 KataGo 訓練資料轉換成 LightGo 格式（輪走方視角的 2-plane 特徵，
由 input.lightgo_encoder.encode_planes 產生，與推論端完全一致）。

KataGo 資料佈局（cpp/neuralnet/nninputs.cpp fillRowV7、cpp/dataio/trainingwrite.h）：
- binaryInputNCHWPacked plane 0 = on-board、plane 1 = 輪走方棋子、plane 2 = 對手棋子、
  plane 6 = 輪走方的 ko-ban 點（含 superko）
- globalTargetsNC[:, 0/1/2] = 輪走方的 win/loss/noresult 機率

使用方式:
    python scripts/convert_katago_npz.py --input data/katago_raw/extracted --output data/lightgo/training_data/katago
"""

import argparse
import logging
import os
import sys
import multiprocessing as mp
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

# 添加專案根目錄（直接以 python scripts/... 執行時需要）
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from core.sample_store import write_samples  # noqa: E402
from input.lightgo_encoder import encode_planes  # noqa: E402

# 設定 logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# KataGo spatial input plane indices（V7）
KATAGO_PLANE_OWN = 1
KATAGO_PLANE_OPP = 2
KATAGO_PLANE_KO_BAN = 6


def unpack_katago_binary(packed: np.ndarray, board_size: int = 19) -> np.ndarray:
    """Unpack KataGo packed binary format.

    Args:
        packed: (num_planes, packed_size) uint8 array
        board_size: board size (default 19)

    Returns:
        unpacked: (num_planes, board_size, board_size) int8 array
    """
    num_planes = packed.shape[0]
    total_points = board_size * board_size

    # Unpack bits
    unpacked = np.unpackbits(packed, axis=1)[:, :total_points]

    return unpacked.reshape(num_planes, board_size, board_size).astype(np.int8)


def katago_sample_to_lightgo(
    packed: np.ndarray,
    policy_target: np.ndarray,
    global_target: np.ndarray,
    board_size: int = 19,
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """將單筆 KataGo 樣本轉成 LightGo (features, policy, value)。

    Args:
        packed: (22, packed_size) binaryInputNCHWPacked 的一列
        policy_target: (2, 362) policyTargetsNCMove 的一列
        global_target: (64,) globalTargetsNC 的一列
        board_size: 棋盤大小

    Returns:
        features: (2, N, N) float32，輪走方視角
        policy: (N*N+1,) float32，總和為 1（pass = 最後一格）
        value: (1,) float32，輪走方視角 [-1, 1] = P(win) - P(loss)
        若 policy 全部落在禁著點（遮罩後總和為 0），回傳 None，呼叫端應略過此樣本。
    """
    planes = unpack_katago_binary(packed, board_size)
    own = planes[KATAGO_PLANE_OWN].astype(bool)
    opp = planes[KATAGO_PLANE_OPP].astype(bool)
    # plane 6 = 輪走方的 ko-ban；實測 3071 筆樣本中 121/128 個點為「空點且四鄰皆對手子」
    # 的簡單劫型，其餘為 superko 禁點，兩者對輪走方都是禁著。
    ko_ban = planes[KATAGO_PLANE_KO_BAN].astype(bool)
    features = encode_planes(own, opp, ko_ban)

    # 禁著點的自殺判定採 LightGo 規則（禁止自殺）。KataGo 部分樣本的規則允許
    # 多子自殺（globalInputNC[:, 8] = 1），其 policy 可能落在這些點上
    #（實測 3071 筆中 19 筆、總質量 ≈ 0.07%）→ 把禁著點的機率清零後重新正規化。
    num_actions = board_size * board_size + 1
    policy = policy_target[0, :num_actions].astype(np.float32)
    policy[:-1][features[1].ravel() > 0] = 0.0
    policy_sum = policy.sum()
    if policy_sum <= 0:
        return None
    policy = policy / policy_sum

    value = np.array([global_target[0] - global_target[1]], dtype=np.float32)
    return features, policy, value


def convert_single_npz(npz_path: str, output_dir: str) -> Tuple[int, int]:
    """轉換單個 NPZ 檔案

    Args:
        npz_path: 輸入 NPZ 路徑
        output_dir: 輸出目錄

    Returns:
        (成功樣本數, 失敗樣本數)
    """
    try:
        data = np.load(npz_path)

        packed = data['binaryInputNCHWPacked']  # (N, 22, 46)
        policy_targets = data['policyTargetsNCMove']  # (N, 2, 362)
        global_targets = data['globalTargetsNC']  # (N, 64)

        num_samples = packed.shape[0]

        features_list = []
        policies_list = []
        values_list = []

        skipped = 0
        for i in range(num_samples):
            sample = katago_sample_to_lightgo(
                packed[i], policy_targets[i], global_targets[i]
            )
            if sample is None:
                skipped += 1
                continue
            features, policy, value = sample
            features_list.append(features)
            policies_list.append(policy)
            values_list.append(value)

        if not features_list:
            return 0, skipped

        # 輸出檔案名
        output_name = Path(npz_path).stem + '_lightgo.npz'
        output_path = os.path.join(output_dir, output_name)

        # 儲存（使用訓練腳本期望的 key 名稱）
        write_samples(
            output_path,
            inputs=np.array(features_list),         # (N, 2, 19, 19)
            policy_targets=np.array(policies_list), # (N, 362)
            value_targets=np.array(values_list)     # (N, 1)
        )

        return num_samples - skipped, skipped

    except Exception as e:
        logger.warning(f"轉換失敗 {npz_path}: {e}")
        return 0, 1


def convert_worker(args: Tuple[str, str]) -> Tuple[int, int]:
    """Worker function for multiprocessing"""
    return convert_single_npz(args[0], args[1])


def main():
    parser = argparse.ArgumentParser(description='將 KataGo NPZ 轉換為 LightGo 格式')
    parser.add_argument('--input', '-i', required=True, help='KataGo NPZ 目錄')
    parser.add_argument('--output', '-o', required=True, help='輸出目錄')
    parser.add_argument('--workers', '-w', type=int, default=4, help='並行處理數')
    parser.add_argument('--limit', '-l', type=int, default=None, help='限制處理檔案數（測試用）')
    args = parser.parse_args()

    # 建立輸出目錄
    os.makedirs(args.output, exist_ok=True)

    # 收集所有 NPZ 檔案
    npz_files = list(Path(args.input).rglob('*.npz'))
    if args.limit:
        npz_files = npz_files[:args.limit]

    logger.info(f"找到 {len(npz_files)} 個 NPZ 檔案")

    if len(npz_files) == 0:
        logger.error("沒有找到 NPZ 檔案！")
        return

    # 準備任務
    tasks = [(str(f), args.output) for f in npz_files]

    # 並行處理
    total_success = 0
    total_failed = 0

    logger.info(f"使用 {args.workers} 個 worker 進行轉換...")

    with mp.Pool(args.workers) as pool:
        for i, (success, failed) in enumerate(pool.imap_unordered(convert_worker, tasks)):
            total_success += success
            total_failed += failed

            if (i + 1) % 1000 == 0:
                logger.info(f"進度: {i+1}/{len(npz_files)}, 樣本: {total_success}, 失敗檔案: {total_failed}")

    logger.info("=" * 60)
    logger.info(f"轉換完成！")
    logger.info(f"總樣本數: {total_success}")
    logger.info(f"失敗檔案: {total_failed}")
    logger.info(f"輸出目錄: {args.output}")


if __name__ == '__main__':
    main()

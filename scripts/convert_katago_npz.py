#!/usr/bin/env python3
"""KataGo NPZ 批量轉換腳本

將 KataGo 訓練資料轉換成 LightGo 格式。

使用方式:
    python scripts/convert_katago_npz.py --input data/katago_raw/extracted --output data/lightgo/training_data/katago
"""

import argparse
import logging
import os
import sys
import multiprocessing as mp
from pathlib import Path
from typing import Tuple, Optional

import numpy as np

# 設定 logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


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


def compute_liberties_fast(black_stones: np.ndarray, white_stones: np.ndarray) -> np.ndarray:
    """快速計算帶符號氣數矩陣

    使用 Union-Find 進行連通分量分析（純 numpy 實現）。

    Args:
        black_stones: (19, 19) 黑子位置
        white_stones: (19, 19) 白子位置

    Returns:
        liberties: (19, 19) 帶符號氣數，黑正白負
    """
    board_size = black_stones.shape[0]
    liberties = np.zeros((board_size, board_size), dtype=np.float32)

    # 空點遮罩
    empty = ~(black_stones.astype(bool) | white_stones.astype(bool))

    def find_groups_and_liberties(stones: np.ndarray, empty_mask: np.ndarray) -> dict:
        """找出所有棋串和對應的氣數"""
        if not stones.any():
            return {}

        # Union-Find
        parent = {}
        rank = {}

        def find(x):
            if x not in parent:
                parent[x] = x
                rank[x] = 0
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]

        def union(x, y):
            rx, ry = find(x), find(y)
            if rx == ry:
                return
            if rank[rx] < rank[ry]:
                rx, ry = ry, rx
            parent[ry] = rx
            if rank[rx] == rank[ry]:
                rank[rx] += 1

        # 找出所有棋子位置
        stone_positions = np.argwhere(stones)

        # 建立位置集合用於快速查找
        stone_set = set(map(tuple, stone_positions))

        # Union 相鄰棋子
        for r, c in stone_positions:
            pos = (r, c)
            find(pos)
            for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                nr, nc = r + dr, c + dc
                if (nr, nc) in stone_set:
                    union(pos, (nr, nc))

        # 統計每個棋串的氣
        group_liberties = {}
        for r, c in stone_positions:
            root = find((r, c))
            if root not in group_liberties:
                group_liberties[root] = set()

            # 檢查四個方向的空點
            for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < board_size and 0 <= nc < board_size:
                    if empty_mask[nr, nc]:
                        group_liberties[root].add((nr, nc))

        # 轉換為氣數
        return {root: len(libs) for root, libs in group_liberties.items()}

    # 處理黑子
    black_group_libs = find_groups_and_liberties(black_stones, empty)

    # Union-Find for liberty assignment
    if black_stones.any():
        parent = {}

        def find(x):
            if x not in parent:
                parent[x] = x
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]

        def union(x, y):
            rx, ry = find(x), find(y)
            if rx != ry:
                parent[ry] = rx

        stone_positions = np.argwhere(black_stones)
        stone_set = set(map(tuple, stone_positions))

        for r, c in stone_positions:
            pos = (r, c)
            find(pos)
            for dr, dc in [(0, 1), (1, 0)]:
                nr, nc = r + dr, c + dc
                if (nr, nc) in stone_set:
                    union(pos, (nr, nc))

        for r, c in stone_positions:
            root = find((r, c))
            libs = black_group_libs.get(root, 0)
            liberties[r, c] = min(libs, 127)

    # 處理白子
    white_group_libs = find_groups_and_liberties(white_stones, empty)

    if white_stones.any():
        parent = {}

        def find(x):
            if x not in parent:
                parent[x] = x
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]

        def union(x, y):
            rx, ry = find(x), find(y)
            if rx != ry:
                parent[ry] = rx

        stone_positions = np.argwhere(white_stones)
        stone_set = set(map(tuple, stone_positions))

        for r, c in stone_positions:
            pos = (r, c)
            find(pos)
            for dr, dc in [(0, 1), (1, 0)]:
                nr, nc = r + dr, c + dc
                if (nr, nc) in stone_set:
                    union(pos, (nr, nc))

        for r, c in stone_positions:
            root = find((r, c))
            libs = white_group_libs.get(root, 0)
            liberties[r, c] = -min(libs, 127)

    return liberties


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

        # 準備輸出陣列
        features_list = []
        policies_list = []
        values_list = []

        for i in range(num_samples):
            # Unpack binary
            planes = unpack_katago_binary(packed[i])  # (22, 19, 19)

            # 取出黑白子位置 (KataGo: plane 1 = 當前玩家, plane 2 = 對手)
            # 但實際上我們需要原始黑白子
            # KataGo 的 plane 0 是空點，plane 1 是當前玩家棋子，plane 2 是對手棋子
            # 我們需要知道當前是誰在下
            # 簡化：假設 global_targets 第一個值是輪次 (0=黑, 1=白)

            # 暫時使用 plane 1, 2 作為黑白子
            current_player_stones = planes[1].astype(bool)
            opponent_stones = planes[2].astype(bool)

            # 從 globalInputNC 判斷當前玩家（通常第一個值）
            # KataGo 格式中 global 輸入第 0 個通道是 "is white to play"
            is_white_to_play = data['globalInputNC'][i, 0] > 0.5

            if is_white_to_play:
                white_stones = current_player_stones
                black_stones = opponent_stones
            else:
                black_stones = current_player_stones
                white_stones = opponent_stones

            # 計算帶符號氣數
            signed_liberties = compute_liberties_fast(
                black_stones.astype(np.int8),
                white_stones.astype(np.int8)
            )

            # 禁止點 = 已佔位置（自殺點和劫需要更複雜的計算，暫時省略）
            forbidden = (black_stones | white_stones).astype(np.float32)

            # 組合成 2-plane 特徵
            features = np.stack([
                signed_liberties / 127.0,  # 歸一化到 [-1, 1]
                forbidden
            ], axis=0).astype(np.float32)  # (2, 19, 19)

            # Policy target (取第一個 head, 包含 pass=index 361)
            policy = policy_targets[i, 0, :362].astype(np.float32)
            policy_sum = policy.sum()
            if policy_sum > 0:
                policy = policy / policy_sum  # 歸一化
            else:
                policy = np.ones(362, dtype=np.float32) / 362  # 均勻分佈

            # Value target (從 global targets 提取勝率)
            # KataGo globalTargetsNC 格式：index 0 是 result (-1 到 1)
            value = np.array([global_targets[i, 0]], dtype=np.float32)

            features_list.append(features)
            policies_list.append(policy)
            values_list.append(value)

        # 輸出檔案名
        output_name = Path(npz_path).stem + '_lightgo.npz'
        output_path = os.path.join(output_dir, output_name)

        # 儲存（使用訓練腳本期望的 key 名稱）
        np.savez_compressed(
            output_path,
            inputs=np.array(features_list),         # (N, 2, 19, 19)
            policy_targets=np.array(policies_list), # (N, 362)
            value_targets=np.array(values_list)     # (N, 1)
        )

        return num_samples, 0

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

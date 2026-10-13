#!/usr/bin/env python3
"""LightGo 2-Plane 輸入編碼器（唯一的盤面編碼實作）

以「輪走方（side to move）視角」將盤面編碼為 2 個平面：
- Plane 0: Signed Liberties —— 輪走方棋子 +氣數、對手棋子 −氣數、空點 0，
  上限 127，再 ÷127 → [-1, 1]
- Plane 1: Forbidden Points —— 輪走方不可落子的點：已佔據點 + 自殺點 + 劫禁點

所有入口（MCTS 推論、KataGo 資料轉換、Phase 2 自我對弈紀錄）都必須經過
:func:`encode_planes`，確保訓練與推論看到的特徵一致。

實作為純 numpy 向量化（連通分量用最小標籤傳播 + pointer jumping），
適合批量轉換數百萬筆 KataGo 樣本。
"""
from __future__ import annotations

from typing import List, Optional, Tuple, Union

import numpy as np
import torch

from core.game_rules import GoBoard

#: 氣數上限（Int8 範圍），同時是 Plane 0 的歸一化分母
MAX_LIBERTIES = 127

KoSpec = Union[None, Tuple[int, int], np.ndarray]


def _shift(arr: np.ndarray, dr: int, dc: int, fill) -> np.ndarray:
    """回傳 out[r, c] = arr[r + dr, c + dc]，超出棋盤處填 ``fill``。"""
    n_rows, n_cols = arr.shape
    out = np.full_like(arr, fill)
    rs_dst = slice(max(0, -dr), n_rows - max(0, dr))
    cs_dst = slice(max(0, -dc), n_cols - max(0, dc))
    rs_src = slice(max(0, dr), n_rows - max(0, -dr))
    cs_src = slice(max(0, dc), n_cols - max(0, -dc))
    out[rs_dst, cs_dst] = arr[rs_src, cs_src]
    return out


_DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def group_liberties(stones_a: np.ndarray, stones_b: np.ndarray) -> np.ndarray:
    """計算每個棋子所屬棋串的氣數（未截斷、無符號）。

    Args:
        stones_a: (N, N) bool，一方棋子位置（顏色可互換，只需兩方不同）
        stones_b: (N, N) bool，另一方棋子位置

    Returns:
        (N, N) int64：棋子位置為其棋串的氣數，空點為 0
    """
    size = stones_a.shape[0]
    total = size * size
    color = stones_a.astype(np.int8) + 2 * stones_b.astype(np.int8)  # 0 空, 1 a, 2 b
    stones = color > 0
    if not stones.any():
        return np.zeros((size, size), dtype=np.int64)

    # 同色相鄰遮罩（每個方向一個）
    same = [(_shift(color, dr, dc, 0) == color) & stones for dr, dc in _DIRS]

    # 連通分量：最小標籤傳播 + pointer jumping
    sentinel = total
    labels = np.where(stones, np.arange(total).reshape(size, size), sentinel)
    while True:
        new = labels
        for (dr, dc), mask in zip(_DIRS, same):
            new = np.where(mask, np.minimum(new, _shift(labels, dr, dc, sentinel)), new)
        flat = np.append(new.ravel(), sentinel)
        new = flat[new]  # pointer jumping：標籤一定指向同棋串的棋子
        if np.array_equal(new, labels):
            break
        labels = new

    # 每個空點對每個相鄰棋串只算一口氣
    empty = ~stones
    nb_labels = [np.where(empty, _shift(labels, dr, dc, sentinel), sentinel) for dr, dc in _DIRS]
    counted = []
    for i, lab in enumerate(nb_labels):
        keep = lab != sentinel
        for prev in nb_labels[:i]:
            keep &= lab != prev
        counted.append(lab[keep])
    lib_count = np.bincount(np.concatenate(counted), minlength=total + 1)
    lib_count[sentinel] = 0
    return lib_count[labels]


def encode_planes(
    own: np.ndarray,
    opp: np.ndarray,
    ko_point: KoSpec = None,
) -> np.ndarray:
    """將盤面以「輪走方視角」編碼為 LightGo 2-plane 特徵。

    Args:
        own: (N, N) 輪到落子那一方（side to move）的棋子（bool 或 0/1）
        opp: (N, N) 對手的棋子（bool 或 0/1）
        ko_point: 輪走方的劫禁點 ``(row, col)``；或 (N, N) bool 遮罩
            （例如 KataGo plane 6 ko-ban，可能含多點）；無則 None

    Returns:
        (2, N, N) float32：
            [0] 帶符號氣數 ÷127（輪走方棋子 +、對手棋子 −）
            [1] 輪走方的禁著點（佔據 + 自殺 + 劫）
    """
    own = np.asarray(own).astype(bool)
    opp = np.asarray(opp).astype(bool)
    if own.shape != opp.shape or own.ndim != 2 or own.shape[0] != own.shape[1]:
        raise ValueError(f"own/opp 必須是相同的 (N, N)，收到 {own.shape} / {opp.shape}")
    if (own & opp).any():
        raise ValueError("同一點不能同時有雙方棋子")

    libs = group_liberties(own, opp)
    occupied = own | opp
    empty = ~occupied

    planes = np.zeros((2,) + own.shape, dtype=np.float32)
    clipped = np.minimum(libs, MAX_LIBERTIES).astype(np.float32)
    planes[0] = (np.where(own, clipped, 0.0) - np.where(opp, clipped, 0.0)) / float(MAX_LIBERTIES)

    # 自殺點：無空鄰點、無可延氣的己方棋串（氣 > 1）、無可提的對方棋串（氣 == 1）
    has_escape = np.zeros_like(empty)
    for dr, dc in _DIRS:
        nb_empty = _shift(empty, dr, dc, False)
        nb_own = _shift(own, dr, dc, False)
        nb_opp = _shift(opp, dr, dc, False)
        nb_libs = _shift(libs, dr, dc, 0)
        has_escape |= nb_empty | (nb_own & (nb_libs > 1)) | (nb_opp & (nb_libs == 1))
    suicide = empty & ~has_escape

    forbidden = occupied | suicide
    if ko_point is not None:
        if isinstance(ko_point, np.ndarray):
            forbidden |= ko_point.astype(bool)
        else:
            forbidden[ko_point[0], ko_point[1]] = True
    planes[1] = forbidden
    return planes


def board_to_arrays(board: GoBoard) -> Tuple[np.ndarray, np.ndarray]:
    """從 GoBoard 取出 (black, white) bool 陣列。"""
    size = board.size
    black = np.zeros((size, size), dtype=bool)
    white = np.zeros((size, size), dtype=bool)
    for row in range(size):
        for col in range(size):
            stone = board.get(row, col)
            if stone == 'b':
                black[row, col] = True
            elif stone == 'w':
                white[row, col] = True
    return black, white


def encode_board(board: GoBoard, color: str) -> np.ndarray:
    """GoBoard 便利介面：以 ``color``（輪走方）視角編碼，含 board.ko_point。

    Returns:
        (2, N, N) float32 numpy 特徵，語意同 :func:`encode_planes`
    """
    if color not in ('b', 'w'):
        raise ValueError(f"color 必須是 'b' 或 'w'，收到 {color!r}")
    black, white = board_to_arrays(board)
    own, opp = (black, white) if color == 'b' else (white, black)
    return encode_planes(own, opp, getattr(board, 'ko_point', None))


class LightGoEncoder:
    """LightGo 2-Plane 編碼器（MCTS / 自我對弈使用的 torch 介面）。

    薄包裝：實際編碼委派給 :func:`encode_planes`。
    """

    def __init__(self, board_size: int = 19):
        """
        Parameters
        ----------
        board_size : int
            棋盤大小（預設 19）
        """
        self.board_size = board_size

    def encode(self, board: GoBoard, current_player: str, normalize: bool = True) -> torch.Tensor:
        """將棋盤編碼為 2-plane tensor

        Args:
            board: GoBoard 物件
            current_player: 'b' or 'w'（輪走方；Plane 0 以其棋子為正）
            normalize: Plane 0 是否 ÷127（預設 True；False 回傳原始氣數）

        Returns:
            tensor: (2, board_size, board_size)
        """
        planes = encode_board(board, current_player)
        if not normalize:
            planes[0] = np.rint(planes[0] * MAX_LIBERTIES)
        return torch.from_numpy(planes)

    def batch_encode(self, boards: List[GoBoard], players: List[str]) -> torch.Tensor:
        """批次編碼

        Args:
            boards: 棋盤列表
            players: 對應的當前玩家 ['b', 'w', ...]

        Returns:
            tensor: (batch_size, 2, board_size, board_size)
        """
        return torch.stack([self.encode(board, player) for board, player in zip(boards, players)])


if __name__ == '__main__':
    demo = GoBoard(size=19)
    demo.play_move(3, 3, 'b')
    demo.play_move(3, 4, 'w')
    demo.play_move(4, 3, 'b')
    encoded = LightGoEncoder(board_size=19).encode(demo, current_player='b')
    print(f"Encoded shape: {encoded.shape}")
    print(encoded[0, 2:6, 2:6] * MAX_LIBERTIES)
    print(encoded[1, 2:6, 2:6])

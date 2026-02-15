#!/usr/bin/env python3
"""LightGo 2-Plane 輸入編碼器

將棋盤狀態編碼為 2 個平面：
- Plane 0: Signed Liberties (帶符號的氣數)
- Plane 1: Forbidden Points (非法著手點)
"""

import numpy as np
import torch
from typing import List, Tuple, Set
from core.game_rules import GoBoard


class LightGoEncoder:
    """LightGo 2-Plane 編碼器

    使用 BFS 計算每個棋串的氣數，對於圍棋的稀疏棋盤效率較高（~14ms/board）。
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
            current_player: 'b' or 'w'
            normalize: 是否歸一化到 [-1, 1] 範圍（建議開啟）

        Returns:
            tensor: (2, board_size, board_size)
                Plane 0: Signed liberties (歸一化後 [-1, 1])
                Plane 1: Forbidden points (Binary [0, 1])
        """
        planes = np.zeros((2, self.board_size, self.board_size), dtype=np.float32)

        # Plane 0: Signed Liberties（使用 BFS，對稀疏棋盤效率高）
        self._encode_signed_liberties(board, planes[0])

        # Plane 1: Forbidden Points
        self._encode_forbidden_points(board, current_player, planes[1])

        # 歸一化 Plane 0 到 [-1, 1] 範圍
        if normalize:
            planes[0] = planes[0] / 127.0  # -127~+127 → -1~+1

        return torch.from_numpy(planes)

    def _encode_signed_liberties(self, board: GoBoard, plane: np.ndarray):
        """編碼 Plane 0: 帶符號的氣數

        - 黑子：正值（+氣數）
        - 白子：負值（-氣數）
        - 空點：0

        氣數範圍限制在 Int8: [-127, +127]

        優化：使用 group cache 避免重複計算同一棋串的氣數
        """
        visited = set()  # 已處理過的位置

        for row in range(self.board_size):
            for col in range(self.board_size):
                if (row, col) in visited:
                    continue

                stone = board.get(row, col)

                if stone is None:  # 空點
                    plane[row, col] = 0
                else:
                    # 計算整個棋串的氣數，並獲取棋串所有成員
                    group, liberties_count = self._count_group_liberties_with_members(
                        board, row, col
                    )
                    # 限制在 Int8 範圍
                    liberties_count = min(liberties_count, 127)

                    # 設定棋串所有成員的值（避免重複計算）
                    sign = 1 if stone == 'b' else -1
                    for r, c in group:
                        plane[r, c] = sign * liberties_count
                        visited.add((r, c))

    def _count_group_liberties(self, board: GoBoard, row: int, col: int) -> int:
        """計算棋串的氣數"""
        _, liberties_count = self._count_group_liberties_with_members(board, row, col)
        return liberties_count

    def _count_group_liberties_with_members(
        self, board: GoBoard, row: int, col: int
    ) -> Tuple[Set[Tuple[int, int]], int]:
        """計算棋串的氣數並返回棋串成員

        Returns:
            (group, liberties_count): 棋串成員集合與氣數
        """
        stone_color = board.get(row, col)
        if stone_color is None:
            return set(), 0

        # BFS 找到整個棋串
        group = set()
        liberties = set()
        stack = [(row, col)]
        visited = set()

        while stack:
            r, c = stack.pop()
            if (r, c) in visited:
                continue
            visited.add((r, c))

            if not (0 <= r < self.board_size and 0 <= c < self.board_size):
                continue

            current = board.get(r, c)

            if current == stone_color:
                group.add((r, c))
                # 檢查四個方向
                for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                    nr, nc = r + dr, c + dc
                    if (nr, nc) not in visited:
                        stack.append((nr, nc))
            elif current is None:
                liberties.add((r, c))

        return group, len(liberties)

    def _encode_forbidden_points(self, board: GoBoard, current_player: str, plane: np.ndarray):
        """編碼 Plane 1: 非法著手點

        包含：
        1. 所有黑子位置
        2. 所有白子位置
        3. 自殺點（落子後立刻被提）
        4. 打劫點（如果存在）
        """
        # 1. 所有占據的點
        for row in range(self.board_size):
            for col in range(self.board_size):
                if board.get(row, col) is not None:
                    plane[row, col] = 1

        # 2. 自殺點
        suicide_points = self._find_suicide_points(board, current_player)
        for row, col in suicide_points:
            plane[row, col] = 1

        # 3. 打劫點
        if hasattr(board, 'ko_point') and board.ko_point is not None:
            ko_row, ko_col = board.ko_point
            plane[ko_row, ko_col] = 1

    def _find_suicide_points(self, board: GoBoard, player: str) -> Set[Tuple[int, int]]:
        """找出所有自殺點

        自殺點：落子後該棋串立刻無氣，且不能提掉對方棋子
        """
        suicide_points = set()

        for row in range(self.board_size):
            for col in range(self.board_size):
                if board.get(row, col) is not None:
                    continue

                # 嘗試落子
                if self._is_suicide(board, row, col, player):
                    suicide_points.add((row, col))

        return suicide_points

    def _is_suicide(self, board: GoBoard, row: int, col: int, player: str) -> bool:
        """檢查落子是否為自殺

        自殺條件：
        1. 落子後該棋串無氣
        2. 且沒有提掉對方棋子
        """
        # 使用 GoBoard 的 is_legal 方法更準確
        # 但如果沒有，我們簡化檢查
        # 檢查是否能提掉對方棋子
        opponent = 'w' if player == 'b' else 'b'

        # 檢查四個方向是否能提子
        for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            nr, nc = row + dr, col + dc
            if (0 <= nr < self.board_size and 0 <= nc < self.board_size
                    and board.get(nr, nc) == opponent):
                # 如果相鄰對方棋子只有一氣（當前空點），落子會提掉它
                if self._count_group_liberties(board, nr, nc) == 1:
                    return False  # 不是自殺，因為能提子

        # 檢查落子後自己是否有氣
        # 簡化：檢查四個方向是否有空點或友軍有氣
        for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            nr, nc = row + dr, col + dc
            if 0 <= nr < self.board_size and 0 <= nc < self.board_size:
                stone = board.get(nr, nc)
                if stone is None:
                    return False  # 有氣
                elif stone == player:
                    if self._count_group_liberties(board, nr, nc) > 1:
                        return False  # 友軍有多餘氣

        # 沒有氣且不能提子 = 自殺
        return True

    def batch_encode(self, boards: List[GoBoard], players: List[str]) -> torch.Tensor:
        """批次編碼

        Args:
            boards: 棋盤列表
            players: 對應的當前玩家 ['b', 'w', ...]

        Returns:
            tensor: (batch_size, 2, board_size, board_size)
        """
        batch = []
        for board, player in zip(boards, players):
            encoded = self.encode(board, player)
            batch.append(encoded)
        return torch.stack(batch)


def unpack_katago_binary(packed: np.ndarray, board_size: int = 19) -> np.ndarray:
    """Unpack KataGo packed binary format to full planes.

    KataGo stores board data as packed bits: (num_planes, ceil(board_size^2 / 8))
    This function unpacks to: (num_planes, board_size, board_size)

    Args:
        packed: (num_planes, packed_size) uint8 array
        board_size: size of the board (default 19)

    Returns:
        unpacked: (num_planes, board_size, board_size) float32 array
    """
    num_planes = packed.shape[0]
    total_points = board_size * board_size

    # Unpack bits
    unpacked = np.unpackbits(packed, axis=1)[:, :total_points]

    # Reshape to (num_planes, board_size, board_size)
    return unpacked.reshape(num_planes, board_size, board_size).astype(np.float32)


def convert_katago_to_lightgo(katago_planes: np.ndarray, board_size: int = 19) -> torch.Tensor:
    """從 KataGo 22-plane 轉換為 LightGo 2-plane

    KataGo format:
        Plane 0-1: Current black/white stones
        Plane 2-17: History (8 moves × 2 colors)
        Plane 18-21: Ladder features, Ko, etc.

    LightGo format:
        Plane 0: Signed liberties
        Plane 1: Forbidden points

    Args:
        katago_planes: (22, 19, 19) or (22, packed_size) numpy array
                       If packed_size != 19*19, will unpack first

    Returns:
        tensor: (2, 19, 19)
    """
    # 如果是 packed 格式，先 unpack
    if katago_planes.ndim == 2 or katago_planes.shape[-1] != board_size:
        katago_planes = unpack_katago_binary(katago_planes, board_size)

    # 從 KataGo 重建棋盤狀態
    board = GoBoard(size=board_size)

    # KataGo Plane 0 = 黑子, Plane 1 = 白子
    black_stones = katago_planes[0]
    white_stones = katago_planes[1]

    # 使用 apply_setup 批量設置棋盤（避免逐個 play_move 的副作用）
    black_points = []
    white_points = []

    for row in range(board_size):
        for col in range(board_size):
            if black_stones[row, col] > 0.5:
                black_points.append((row, col))
            elif white_stones[row, col] > 0.5:
                white_points.append((row, col))

    # 使用 sgfmill 的 apply_setup 設置棋盤
    board.board.apply_setup(black_points, white_points, [])

    # 檢測當前玩家（從 history 推斷，或使用額外信息）
    # 簡化：假設黑先
    current_player = 'b'

    # 使用 LightGo encoder
    encoder = LightGoEncoder(board_size=board_size)
    return encoder.encode(board, current_player)


if __name__ == '__main__':
    # 測試編碼器
    board = GoBoard(size=19)

    # 放置一些棋子進行測試
    board.play_move(3, 3, 'b')  # D4
    board.play_move(3, 4, 'w')  # D5
    board.play_move(4, 3, 'b')  # E4

    encoder = LightGoEncoder(board_size=19)
    encoded = encoder.encode(board, current_player='b')

    print("LightGo 2-Plane Encoding Test")
    print("=" * 60)
    print(f"Encoded shape: {encoded.shape}")
    print(f"\nPlane 0 (Signed Liberties) - Center region:")
    print(encoded[0, 2:6, 2:6])
    print(f"\nPlane 1 (Forbidden Points) - Center region:")
    print(encoded[1, 2:6, 2:6])
    print("\n✅ Encoding test complete!")

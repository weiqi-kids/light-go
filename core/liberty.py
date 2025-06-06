"""Utilities for counting liberties on a Go board."""
from __future__ import annotations

from typing import List, Set, Tuple

Board = List[List[int]]


def neighbors(x: int, y: int, size: int):
    """Yield the coordinates adjacent to ``(x, y)`` on a ``size`` x ``size`` board."""
    if x > 0:
        yield x - 1, y
    if x < size - 1:
        yield x + 1, y
    if y > 0:
        yield x, y - 1
    if y < size - 1:
        yield x, y + 1


def group_and_liberties(board: Board, x: int, y: int) -> Tuple[Set[Tuple[int, int]], Set[Tuple[int, int]]]:
    """Return the connected group at ``(x, y)`` and its liberties."""
    color = board[y][x]
    size = len(board)
    group = {(x, y)}
    liberties: Set[Tuple[int, int]] = set()
    stack = [(x, y)]
    while stack:
        cx, cy = stack.pop()
        for nx, ny in neighbors(cx, cy, size):
            val = board[ny][nx]
            if val == 0:
                liberties.add((nx, ny))
            elif val == color and (nx, ny) not in group:
                group.add((nx, ny))
                stack.append((nx, ny))
    return group, liberties


def count_liberties(board: Board) -> List[Tuple[int, int, int]]:
    """Return a list of ``(x, y, liberties)`` for all stones on the board."""
    size = len(board)
    visited: Set[Tuple[int, int]] = set()
    result: List[Tuple[int, int, int]] = []
    for y in range(size):
        for x in range(size):
            color = board[y][x]
            if color == 0 or (x, y) in visited:
                continue
            group, libs = group_and_liberties(board, x, y)
            for gx, gy in group:
                visited.add((gx, gy))
                val = len(libs) if color == 1 else -len(libs)
                result.append((gx + 1, gy + 1, val))
    return result


__all__ = ["count_liberties", "group_and_liberties", "neighbors", "Board"]

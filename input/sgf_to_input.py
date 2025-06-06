"""Convert SGF files to the internal liberty/forbidden/metadata format."""
from __future__ import annotations

import re
from typing import Dict, List, Tuple

from core.show_board import render_board

Board = List[List[int]]


def _coord_to_xy(coord: str) -> Tuple[int, int]:
    """Convert SGF coordinate like 'aa' to (x, y) 0-based."""
    if coord == "" or coord is None:
        return -1, -1  # pass move
    x = ord(coord[0]) - ord("a")
    y = ord(coord[1]) - ord("a")
    return x, y


def _neighbors(x: int, y: int, size: int):
    if x > 0:
        yield x - 1, y
    if x < size - 1:
        yield x + 1, y
    if y > 0:
        yield x, y - 1
    if y < size - 1:
        yield x, y + 1


def _group_and_liberties(board: Board, x: int, y: int) -> Tuple[set[Tuple[int, int]], set[Tuple[int, int]]]:
    color = board[y][x]
    size = len(board)
    group = {(x, y)}
    liberties: set[Tuple[int, int]] = set()
    stack = [(x, y)]
    while stack:
        cx, cy = stack.pop()
        for nx, ny in _neighbors(cx, cy, size):
            val = board[ny][nx]
            if val == 0:
                liberties.add((nx, ny))
            elif val == color and (nx, ny) not in group:
                group.add((nx, ny))
                stack.append((nx, ny))
    return group, liberties


def _place_stone(board: Board, x: int, y: int, color: int) -> int:
    """Place a stone and return number of opponent captures."""
    size = len(board)
    board[y][x] = color
    captured = 0
    for nx, ny in _neighbors(x, y, size):
        if board[ny][nx] == -color:
            g, libs = _group_and_liberties(board, nx, ny)
            if not libs:
                for gx, gy in g:
                    board[gy][gx] = 0
                captured += len(g)
    g, libs = _group_and_liberties(board, x, y)
    if not libs:
        for gx, gy in g:
            board[gy][gx] = 0
        # suicide stones count for opponent
        captured -= len(g)
    return captured


def parse_sgf(path: str) -> Tuple[Board, Dict]:
    text = open(path, "r", encoding="utf-8").read()
    size_match = re.search(r"SZ\[(\d+)\]", text)
    size = int(size_match.group(1)) if size_match else 19
    komi_match = re.search(r"KM\[([^\]]+)\]", text)
    komi = float(komi_match.group(1)) if komi_match else 0.0
    rules_match = re.search(r"RU\[([^\]]+)\]", text)
    rules = rules_match.group(1) if rules_match else "chinese"

    board = [[0 for _ in range(size)] for _ in range(size)]
    capture_black = 0
    capture_white = 0
    moves = re.findall(r";([BW])\[([^\]]*)\]", text)
    next_color = "black"
    for color_char, coord in moves:
        x, y = _coord_to_xy(coord)
        if x == -1:
            # pass
            next_color = "white" if color_char == "B" else "black"
            continue
        color = 1 if color_char == "B" else -1
        captured = _place_stone(board, x, y, color)
        if color == 1:
            capture_black += captured
            next_color = "white"
        else:
            capture_white += captured
            next_color = "black"
    metadata = {
        "rules": {
            "ruleset": rules,
            "komi": komi,
            "board_size": size,
            "handicap": 0,
        },
        "capture": {"black": capture_black, "white": capture_white},
        "next_move": next_color,
    }
    return board, metadata


def convert(path: str) -> Dict:
    board, metadata = parse_sgf(path)
    render_board(board)
    size = len(board)
    liberty: List[Tuple[int, int, int]] = []
    visited: set[Tuple[int, int]] = set()
    for y in range(size):
        for x in range(size):
            color = board[y][x]
            if color == 0:
                continue
            if (x, y) in visited:
                continue
            group, libs = _group_and_liberties(board, x, y)
            for gx, gy in group:
                visited.add((gx, gy))
                val = len(libs) if color == 1 else -len(libs)
                liberty.append((gx + 1, gy + 1, val))
    return {"liberty": liberty, "forbidden": [], "metadata": metadata}


__all__ = ["convert", "parse_sgf"]


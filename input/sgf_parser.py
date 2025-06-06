"""Simple SGF parser used for extracting basic game information and board matrix."""
from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple

Board = List[List[int]]


def parse_sgf(sgf_file_path: str) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Parse an SGF file and return game info and list of moves."""
    with open(sgf_file_path, "r", encoding="utf-8") as f:
        sgf_content = f.read().strip()

    game_info: Dict[str, Any] = {}

    size_match = re.search(r"SZ\[(\d+)\]", sgf_content)
    game_info["board_size"] = int(size_match.group(1)) if size_match else 19

    pb_match = re.search(r"PB\[([^\]]+)\]", sgf_content)
    game_info["black_player"] = pb_match.group(1) if pb_match else "Unknown"

    pw_match = re.search(r"PW\[([^\]]+)\]", sgf_content)
    game_info["white_player"] = pw_match.group(1) if pw_match else "Unknown"

    re_match = re.search(r"RE\[([^\]]+)\]", sgf_content)
    game_info["result"] = re_match.group(1) if re_match else "Unknown"

    moves: List[Dict[str, Any]] = []
    move_pattern = r"([BW])\[([a-z]{2})\]"
    for color, pos in re.findall(move_pattern, sgf_content):
        if pos:
            x = ord(pos[0]) - ord("a")
            y = ord(pos[1]) - ord("a")
            moves.append({"color": color, "x": x, "y": y, "sgf_pos": pos})

    return game_info, moves


def sgf_to_input_matrix(sgf_file_path: str) -> Tuple[Board, Dict[str, Any], List[Dict[str, Any]]]:
    """Convert SGF file to a board matrix format."""
    game_info, moves = parse_sgf(sgf_file_path)
    size = game_info["board_size"]
    board: Board = [[0 for _ in range(size)] for _ in range(size)]

    for move in moves:
        x, y = move["x"], move["y"]
        board[y][x] = 1 if move["color"] == "B" else -1

    return board, game_info, moves


def print_board(board: Board) -> None:
    """Print a board using unicode symbols."""
    symbols = {0: "\u00b7", 1: "\u25CF", -1: "\u25CB"}
    for row in board:
        print(" ".join(symbols[cell] for cell in row))


__all__ = ["parse_sgf", "sgf_to_input_matrix", "print_board"]

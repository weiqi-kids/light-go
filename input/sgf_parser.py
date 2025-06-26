"""Simple SGF parser used for extracting basic game information and board matrix."""
from __future__ import annotations

from typing import Any, Dict, List, Tuple

from sgfmill import sgf, sgf_properties

Board = List[List[int]]


def parse_sgf(sgf_file_path: str) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Parse an SGF file and return game info and list of moves."""
    with open(sgf_file_path, "rb") as f:
        sgf_bytes = f.read()

    sgf_game = sgf.Sgf_game.from_bytes(sgf_bytes)
    root = sgf_game.get_root()

    size = sgf_game.get_size()
    game_info: Dict[str, Any] = {
        "board_size": size,
        "black_player": root.get("PB") if root.has_property("PB") else "Unknown",
        "white_player": root.get("PW") if root.has_property("PW") else "Unknown",
        "result": root.get("RE") if root.has_property("RE") else "Unknown",
    }

    moves: List[Dict[str, Any]] = []
    for node in sgf_game.get_main_sequence()[1:]:
        color, move = node.get_move()
        if color is None:
            continue
        color = color.upper()
        if move is not None:
            row, col = move
            sgf_pos = (
                sgf_properties.serialise_go_point(move, size).decode()
            )
            y = size - 1 - row
            x = col
            moves.append({"color": color, "x": x, "y": y, "sgf_pos": sgf_pos})
        else:
            moves.append({"color": color, "x": None, "y": None, "sgf_pos": ""})

    return game_info, moves


def sgf_to_input_matrix(sgf_file_path: str) -> Tuple[Board, Dict[str, Any], List[Dict[str, Any]]]:
    """Convert SGF file to a board matrix format."""
    game_info, moves = parse_sgf(sgf_file_path)
    size = game_info["board_size"]
    board: Board = [[0 for _ in range(size)] for _ in range(size)]

    for move in moves:
        x, y = move["x"], move["y"]
        if x is None or y is None:
            continue
        board[y][x] = 1 if move["color"] == "B" else -1

    return board, game_info, moves


def print_board(board: Board) -> None:
    """Print a board using unicode symbols."""
    symbols = {0: "\u00b7", 1: "\u25CF", -1: "\u25CB"}
    for row in board:
        print(" ".join(symbols[cell] for cell in row))


__all__ = ["parse_sgf", "sgf_to_input_matrix", "print_board"]

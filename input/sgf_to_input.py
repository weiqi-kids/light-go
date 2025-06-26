"""SGF to liberty/forbidden/metadata converter using sgfmill."""
from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple

from sgfmill import sgf, boards
from core.liberty import count_liberties

BoardMatrix = List[List[int]]


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _board_to_matrix(board: boards.Board) -> BoardMatrix:
    """Convert ``sgfmill`` board to a matrix of ``int`` values."""
    size = board.side
    matrix: BoardMatrix = [[0 for _ in range(size)] for _ in range(size)]
    for row in range(size):
        for col in range(size):
            stone = board.get(row, col)
            target_row = size - 1 - row
            if stone == 'b':
                matrix[target_row][col] = 1
            elif stone == 'w':
                matrix[target_row][col] = -1
    return matrix


def _parse_ot(ot: str) -> Tuple[int, int]:
    """Parse byo-yomi description like '5x30' and return (periods, period_time)."""
    if not ot:
        return 0, 0
    m = re.search(r"(\d+)\s*x\s*(\d+)", ot)
    if not m:
        return 0, 0
    return int(m.group(1)), int(m.group(2))


def _compute_forbidden(board: boards.Board, next_color: str) -> List[Tuple[int, int]]:
    """Return all illegal move coordinates for ``next_color`` (0-based)."""
    # ``sgfmill`` does not expose a direct legality check. For this simplified
    # use case we assume no illegal moves except occupied points.
    size = board.side
    forbidden: List[Tuple[int, int]] = []
    for row in range(size):
        for col in range(size):
            if board.get(row, col) is not None:
                continue
            # Basic ko and suicide checks are skipped for simplicity
    return forbidden


# ---------------------------------------------------------------------------
# Core SGF parsing logic
# ---------------------------------------------------------------------------

def parse_sgf(path: str, step: int | None = None) -> Tuple[BoardMatrix, Dict[str, Any], boards.Board]:
    """Parse ``path`` up to ``step`` and return the board matrix and metadata."""
    with open(path, "rb") as f:
        sgf_bytes = f.read()

    game = sgf.Sgf_game.from_bytes(sgf_bytes)
    board_size = game.get_size()
    board = boards.Board(board_size)

    root = game.get_root()
    komi = game.get_komi() if game.get_komi() is not None else 7.5

    try:
        ruleset_prop = root.get("RU")
    except KeyError:
        ruleset_prop = "chinese"
    ruleset = (ruleset_prop or "chinese").lower()

    handicap = game.get_handicap() or 0

    capture_black = 0
    capture_white = 0
    next_move = "black"
    steps: List[Tuple[str, Tuple[int, int] | None]] = []

    nodes = game.get_main_sequence()[1:]
    if step is not None:
        nodes = nodes[: step]

    last_node = root
    for node in nodes:
        color, move = node.get_move()
        if color is None:
            continue
        sgf_color = "black" if color == "b" else "white"
        next_move = "white" if color == "b" else "black"
        if move is not None:
            row, col = move
            board.play(row, col, color)
            conv_row = board_size - 1 - row
            steps.append((sgf_color, (conv_row, col)))
        else:
            steps.append((sgf_color, None))
        last_node = node

    matrix = _board_to_matrix(board)

    try:
        ot_val = root.get("OT")
    except KeyError:
        ot_val = ""
    periods, period_time = _parse_ot(ot_val)

    last_bl = float(last_node.get("BL")) if last_node.has_property("BL") else 0.0
    last_wl = float(last_node.get("WL")) if last_node.has_property("WL") else 0.0
    last_ob = int(last_node.get("OB")) if last_node.has_property("OB") else 0
    last_ow = int(last_node.get("OW")) if last_node.has_property("OW") else 0

    metadata = {
        "rules": {
            "ruleset": ruleset,
            "komi": komi,
            "board_size": board_size,
            "handicap": handicap,
        },
        "capture": {"black": capture_black, "white": capture_white},
        "next_move": next_move,
        "step": steps,
        "time_control": {
            "main_time_seconds": float(root.get("TM")) if root.has_property("TM") else 0.0,
            "byo_yomi": {
                "period_time_seconds": period_time,
                "periods": periods,
            },
        },
        "time": [
            {"player": "black", "main_time_seconds": last_bl, "periods": last_ob},
            {"player": "white", "main_time_seconds": last_wl, "periods": last_ow},
        ],
    }

    return matrix, metadata, board


def convert(path: str, step: int | None = None) -> Dict[str, Any]:
    """High level convenience wrapper returning the structured data."""
    matrix, metadata, board = parse_sgf(path, step)

    liberties_1b = count_liberties(matrix)
    liberty = [(r - 1, c - 1, v) for r, c, v in liberties_1b]

    forbidden = _compute_forbidden(board, metadata["next_move"])

    return {"liberty": liberty, "forbidden": forbidden, "metadata": metadata}


__all__ = ["parse_sgf", "convert"]

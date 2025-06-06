"""Utility functions to render a Go board in a human readable form."""
from typing import List

Board = List[List[int]]  # 0 empty, 1 black, -1 white

SYMBOLS = {0: '.', 1: 'X', -1: 'O'}


def board_to_string(board: Board) -> str:
    """Return a string representation of the board."""
    size = len(board)
    lines = []
    header = '   ' + ' '.join(f"{i:2d}" for i in range(1, size + 1))
    lines.append(header)
    for y in range(size):
        row = [SYMBOLS[board[y][x]] for x in range(size)]
        lines.append(f"{y+1:2d} " + ' '.join(row))
    return '\n'.join(lines)


def render_board(board: Board) -> None:
    """Print the board to stdout."""
    print(board_to_string(board))


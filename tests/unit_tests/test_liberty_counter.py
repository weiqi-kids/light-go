import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from input import sgf_to_input
from core import liberty


def test_count_liberties(tmp_path: pathlib.Path):
    sgf_content = "(;FF[4]SZ[5];B[aa];W[bb];B[cc];W[dd])"
    sgf_file = tmp_path / "game.sgf"
    sgf_file.write_text(sgf_content)

    board, _ = sgf_to_input.parse_sgf(str(sgf_file))
    result = liberty.count_liberties(board)

    assert (1, 1, 2) in result
    assert len(result) == 4

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from input import sgf_to_input


def test_basic_sgf(tmp_path: pathlib.Path):
    sgf_content = "(;FF[4]SZ[5];B[aa];W[bb];B[cc];W[dd])"
    sgf_file = tmp_path / "game.sgf"
    sgf_file.write_text(sgf_content)

    result = sgf_to_input.convert(str(sgf_file))

    assert set(result.keys()) == {"liberty", "forbidden", "metadata"}
    metadata = result["metadata"]
    assert metadata["rules"]["board_size"] == 5
    assert metadata["next_move"] == "black"

    liberty = result["liberty"]
    assert (1, 1, 2) in liberty  # corner stone liberties
    assert len(liberty) == 4


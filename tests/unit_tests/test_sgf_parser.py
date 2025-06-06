import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from input import sgf_parser


def test_parser_moves(tmp_path: pathlib.Path):
    sgf_content = "(;FF[4]SZ[5]PB[Black]PW[White];B[aa];W[bb];B[cc])"
    sgf_file = tmp_path / "game.sgf"
    sgf_file.write_text(sgf_content)

    game_info, moves = sgf_parser.parse_sgf(str(sgf_file))

    assert game_info["board_size"] == 5
    assert game_info["black_player"] == "Black"
    assert game_info["white_player"] == "White"
    assert len(moves) == 3
    assert moves[0]["color"] == "B"
    assert moves[0]["x"] == 0 and moves[0]["y"] == 0


def test_sgf_to_input_matrix(tmp_path: pathlib.Path):
    sgf_content = "(;FF[4]SZ[5];B[aa];W[bb])"
    sgf_file = tmp_path / "g2.sgf"
    sgf_file.write_text(sgf_content)

    board, game_info, moves = sgf_parser.sgf_to_input_matrix(str(sgf_file))

    assert len(board) == 5
    assert board[0][0] == 1
    assert board[1][1] == -1
    assert len(moves) == 2
    assert game_info["board_size"] == 5

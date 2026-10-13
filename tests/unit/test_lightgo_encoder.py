"""Unit tests for the LightGo 2-plane position encoding (input/lightgo_encoder.py).

驗證唯一的編碼模組，以及三個入口產生完全相同的平面：
- LightGoEncoder.encode（MCTS / 自我對弈推論）
- scripts.train_muzero 的 Phase 2 軌跡紀錄（encode_board）
- scripts.convert_katago_npz 的 KataGo 樣本轉換（katago_sample_to_lightgo）

語意：輪走方視角。Plane 0 = 帶符號氣數 ÷127（輪走方 +、對手 −），
Plane 1 = 輪走方禁著點（佔據 + 自殺 + 劫）。
"""
from __future__ import annotations

import random
from typing import Callable, Dict, List, Tuple

import numpy as np
import pytest

from core.game_rules import GoBoard
from input.lightgo_encoder import (
    MAX_LIBERTIES,
    LightGoEncoder,
    board_to_arrays,
    encode_board,
    encode_planes,
)
from scripts import convert_katago_npz, train_muzero


Move = Tuple[str, int, int]

# 具名局面：落子序列（皆為合法著手），最後一項為輪走方
POSITIONS: Dict[str, Tuple[List[Move], str]] = {
    # 黑提白兩子（(0,0),(0,1)），提子後輪白
    "capture": (
        [("b", 1, 0), ("w", 0, 0), ("b", 1, 1), ("w", 0, 1), ("b", 0, 2)],
        "w",
    ),
    # (0,0) 對白是自殺點、對黑不是
    "suicide": (
        [("b", 0, 1), ("w", 5, 5), ("b", 1, 0)],
        "w",
    ),
    # 黑在 (1,2) 提白 (1,1) 形成劫，ko_point = (1,1)，輪白
    "ko": (
        [
            ("b", 0, 1), ("w", 0, 2), ("b", 1, 0), ("w", 1, 3),
            ("b", 2, 1), ("w", 2, 2), ("b", 10, 10), ("w", 1, 1),
            ("b", 1, 2),
        ],
        "w",
    ),
    # 一般局面，輪白
    "opening": (
        [("b", 3, 3), ("w", 3, 4), ("b", 4, 3), ("w", 15, 15), ("b", 4, 4)],
        "w",
    ),
}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def make_position() -> Callable[[str], Tuple[GoBoard, str]]:
    """Factory fixture: build a named position on a 19x19 GoBoard."""
    def _create(name: str) -> Tuple[GoBoard, str]:
        moves, color = POSITIONS[name]
        board = GoBoard(size=19)
        for move_color, row, col in moves:
            assert board.play_move(row, col, move_color), (name, move_color, row, col)
        return board, color
    return _create


def _to_katago_packed(board: GoBoard, color: str) -> np.ndarray:
    """Pack a GoBoard as a KataGo V7 binaryInputNCHWPacked row (22, 46)."""
    black, white = board_to_arrays(board)
    own, opp = (black, white) if color == "b" else (white, black)
    planes = np.zeros((22, 19 * 19), dtype=np.uint8)
    planes[0] = 1
    planes[1] = own.ravel()
    planes[2] = opp.ravel()
    if board.ko_point is not None:
        planes[6, board.ko_point[0] * 19 + board.ko_point[1]] = 1
    return np.packbits(planes, axis=1)


def _katago_features(board: GoBoard, color: str) -> np.ndarray:
    """Encode a GoBoard position through the KataGo conversion path."""
    policy_target = np.zeros((2, 362), dtype=np.float32)
    policy_target[0, 361] = 1.0  # pass：永遠合法，不會被遮罩剔除
    global_target = np.zeros(64, dtype=np.float32)
    features, _, _ = convert_katago_npz.katago_sample_to_lightgo(
        _to_katago_packed(board, color), policy_target, global_target
    )
    return features


def _reference_planes(board: GoBoard, color: str) -> np.ndarray:
    """Ground truth computed from GoBoard's own rule implementation."""
    size = board.size
    planes = np.zeros((2, size, size), dtype=np.float32)
    for row in range(size):
        for col in range(size):
            stone = board.get(row, col)
            if stone is not None:
                libs = min(board.count_liberties(board.get_group(row, col)), MAX_LIBERTIES)
                planes[0, row, col] = (libs if stone == color else -libs) / MAX_LIBERTIES
            planes[1, row, col] = 0.0 if board.is_legal(row, col, color) else 1.0
    return planes


def _random_position(seed: int, num_moves: int) -> Tuple[GoBoard, str]:
    """Play a random legal game on 19x19 and return (board, side to move)."""
    rng = random.Random(seed)
    board = GoBoard(size=19)
    color = "b"
    for _ in range(num_moves):
        empties = [(r, c) for r in range(19) for c in range(19) if board.get(r, c) is None]
        rng.shuffle(empties)
        if not any(board.play_move(row, col, color) for row, col in empties[:20]):
            board.pass_move(color)
        color = "w" if color == "b" else "b"
    return board, color


# ---------------------------------------------------------------------------
# Test Classes
# ---------------------------------------------------------------------------

class TestEncodePlanesSemantics:
    """Canonical semantics of encode_planes / encode_board."""

    def test_normalization_divides_by_127(self):
        """Plane 0 holds liberties / 127 with side-to-move stones positive."""
        own = np.zeros((19, 19), dtype=bool)
        opp = np.zeros((19, 19), dtype=bool)
        own[9, 9] = True
        opp[0, 0] = True

        planes = encode_planes(own, opp)

        assert planes.dtype == np.float32
        assert planes.shape == (2, 19, 19)
        assert planes[0, 9, 9] == pytest.approx(4 / 127)
        assert planes[0, 0, 0] == pytest.approx(-2 / 127)
        assert np.abs(planes[0]).max() <= 1.0

    @pytest.mark.parametrize("name", list(POSITIONS), ids=list(POSITIONS))
    def test_side_to_move_flips_plane0(
        self, make_position: Callable[[str], Tuple[GoBoard, str]], name: str
    ):
        """Encoding for 'b' vs 'w' negates plane 0."""
        board, _ = make_position(name)

        as_black = encode_board(board, "b")
        as_white = encode_board(board, "w")

        np.testing.assert_array_equal(as_black[0], -as_white[0])

    def test_capture_removes_stones_and_updates_liberties(
        self, make_position: Callable[[str], Tuple[GoBoard, str]]
    ):
        """Captured points are empty, capturing group's liberties are updated."""
        board, color = make_position("capture")

        planes = encode_board(board, color)

        assert planes[0, 0, 0] == 0.0 and planes[0, 0, 1] == 0.0
        assert planes[1, 0, 0] == 0.0 and planes[1, 0, 1] == 0.0
        # 黑串 (1,0),(1,1) 氣：(0,0),(0,1),(2,0),(2,1),(1,2) = 5；輪白 → 負
        assert planes[0, 1, 0] == pytest.approx(-5 / 127)

    def test_suicide_point_forbidden_for_side_to_move_only(
        self, make_position: Callable[[str], Tuple[GoBoard, str]]
    ):
        """A suicide point is forbidden for the side to move, not for the opponent."""
        board, _ = make_position("suicide")

        assert encode_board(board, "w")[1, 0, 0] == 1.0
        assert encode_board(board, "b")[1, 0, 0] == 0.0

    def test_ko_point_forbidden(self, make_position: Callable[[str], Tuple[GoBoard, str]]):
        """The ko point recorded on the board is forbidden."""
        board, color = make_position("ko")
        assert board.ko_point == (1, 1)

        planes = encode_board(board, color)

        assert planes[1, 1, 1] == 1.0
        assert planes[0, 1, 1] == 0.0

    def test_overlapping_stones_rejected(self):
        """Own and opponent stones on the same point raise ValueError."""
        stones = np.zeros((9, 9), dtype=bool)
        stones[4, 4] = True

        with pytest.raises(ValueError):
            encode_planes(stones, stones)

    @pytest.mark.parametrize(
        "seed,num_moves",
        [(0, 60), (1, 150), (2, 300), (3, 450)],
        ids=["seed0_60", "seed1_150", "seed2_300", "seed3_450"],
    )
    def test_matches_goboard_rules(self, seed: int, num_moves: int):
        """Planes equal liberties and legality computed by GoBoard itself."""
        board, color = _random_position(seed, num_moves)

        np.testing.assert_array_equal(encode_board(board, color), _reference_planes(board, color))


class TestEncodingEntryPointsAgree:
    """All three entry points produce identical planes."""

    def test_train_muzero_uses_shared_encoder(self):
        """Phase 2 recording uses the shared module's encode_board."""
        assert train_muzero.encode_board is encode_board

    @pytest.mark.parametrize("name", list(POSITIONS), ids=list(POSITIONS))
    @pytest.mark.parametrize("color", ["b", "w"], ids=["black_to_move", "white_to_move"])
    def test_named_positions(
        self,
        make_position: Callable[[str], Tuple[GoBoard, str]],
        name: str,
        color: str,
    ):
        """LightGoEncoder, train_muzero and KataGo conversion agree."""
        board, _ = make_position(name)

        mcts_planes = LightGoEncoder(board_size=19).encode(board, color).numpy()
        phase2_planes = train_muzero.encode_board(board, color)
        katago_planes = _katago_features(board, color)

        np.testing.assert_array_equal(mcts_planes, phase2_planes)
        np.testing.assert_array_equal(mcts_planes, katago_planes)

    @pytest.mark.parametrize("seed", [10, 11, 12], ids=["seed10", "seed11", "seed12"])
    def test_random_positions(self, seed: int):
        """Entry points agree on random mid-game positions."""
        board, color = _random_position(seed, 200)

        mcts_planes = LightGoEncoder(board_size=19).encode(board, color).numpy()
        katago_planes = _katago_features(board, color)

        np.testing.assert_array_equal(mcts_planes, train_muzero.encode_board(board, color))
        np.testing.assert_array_equal(mcts_planes, katago_planes)


class TestKatagoSampleConversion:
    """Targets produced by scripts.convert_katago_npz.katago_sample_to_lightgo."""

    @pytest.mark.parametrize(
        "win,loss,expected",
        [(0.8, 0.15, 0.65), (0.1, 0.9, -0.8), (0.5, 0.5, 0.0)],
        ids=["winning", "losing", "even"],
    )
    def test_value_is_win_minus_loss(
        self,
        make_position: Callable[[str], Tuple[GoBoard, str]],
        win: float,
        loss: float,
        expected: float,
    ):
        """Value target = P(win) - P(loss) from the side-to-move perspective."""
        board, color = make_position("opening")
        policy_target = np.zeros((2, 362), dtype=np.float32)
        policy_target[0, 361] = 3.0
        global_target = np.zeros(64, dtype=np.float32)
        global_target[0], global_target[1] = win, loss

        _, policy, value = convert_katago_npz.katago_sample_to_lightgo(
            _to_katago_packed(board, color), policy_target, global_target
        )

        assert value.shape == (1,)
        assert value[0] == pytest.approx(expected, abs=1e-6)
        assert policy.shape == (362,)
        assert policy[361] == pytest.approx(1.0)

    def test_white_to_move_uses_own_plane_as_positive(
        self, make_position: Callable[[str], Tuple[GoBoard, str]]
    ):
        """With white to move, KataGo plane 1 (white stones) maps to positive liberties."""
        board, _ = make_position("opening")

        planes = _katago_features(board, "w")

        assert planes[0, 3, 4] > 0  # 白子
        assert planes[0, 3, 3] < 0  # 黑子

    def test_policy_on_forbidden_points_is_masked_and_renormalized(
        self, make_position: Callable[[str], Tuple[GoBoard, str]]
    ):
        """Policy mass on forbidden points (e.g. multi-stone suicide) is dropped."""
        board, _ = make_position("suicide")
        policy_target = np.zeros((2, 362), dtype=np.float32)
        policy_target[0, 0] = 1.0   # (0,0)：白方的自殺點
        policy_target[0, 361] = 3.0  # pass
        global_target = np.zeros(64, dtype=np.float32)

        features, policy, _ = convert_katago_npz.katago_sample_to_lightgo(
            _to_katago_packed(board, "w"), policy_target, global_target
        )

        assert features[1, 0, 0] == 1.0
        assert policy[0] == 0.0
        assert policy[361] == pytest.approx(1.0)
        assert policy.sum() == pytest.approx(1.0)

    def test_sample_dropped_when_all_policy_mass_is_forbidden(
        self, make_position: Callable[[str], Tuple[GoBoard, str]]
    ):
        """A sample whose whole policy lies on forbidden points returns None."""
        board, _ = make_position("suicide")
        policy_target = np.zeros((2, 362), dtype=np.float32)
        policy_target[0, 0] = 1.0
        global_target = np.zeros(64, dtype=np.float32)

        sample = convert_katago_npz.katago_sample_to_lightgo(
            _to_katago_packed(board, "w"), policy_target, global_target
        )

        assert sample is None

"""統一搜尋介面（core.search.Search）的契約測試（issue #55）。

同一組測試跑在兩個實作上（ids=["puct", "gumbel"]）：
合法著手或 pass、policy 形狀與正規化、advance_tree 後的樹重用、
不同局面時拒絕重用、MaterialEvaluator 下會選提子。
"""
from __future__ import annotations

from typing import Any, Callable, List, Tuple

import numpy as np
import pytest

from core.evaluator import MaterialEvaluator, UniformEvaluator
from core.game_rules import GoBoard
from core.mcts import MCTS
from core.search import SEARCH_KINDS, Search, create_search, position_key


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SIZE = 5

#: 白子 (2,2) 只剩 (2,3) 一口氣；黑下 (2,3) 提子
CAPTURE_SETUP: List[Tuple[int, int, str]] = [
    (2, 2, 'w'), (1, 2, 'b'), (3, 2, 'b'), (2, 1, 'b'),
]
CAPTURE_MOVE = (2, 3)

#: 每種搜尋的小型設定（快、且 gumbel 會看過所有合法著手）
SEARCH_CONFIGS = {
    'puct': {'num_simulations': 60, 'dirichlet_epsilon': 0.0},
    'gumbel': {'num_simulations': 32, 'num_sampled_actions': SIZE * SIZE + 1},
}

KINDS = pytest.mark.parametrize("kind", list(SEARCH_KINDS), ids=list(SEARCH_KINDS))


def _setup_board(stones: List[Tuple[int, int, str]], size: int = SIZE) -> GoBoard:
    board = GoBoard(size=size)
    for row, col, color in stones:
        assert board.play_move(row, col, color), (row, col, color)
    return board


def _most_visited_move(root: Any) -> Tuple[int, int]:
    return max(root.children.items(), key=lambda item: item[1].visit_count)[0]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def make_search() -> Callable[..., Search]:
    """Factory：``make_search(kind, evaluator=None, **overrides)``，預設開啟樹重用。"""
    def _create(kind: str, evaluator: Any = None, **overrides: Any) -> Search:
        config = {**SEARCH_CONFIGS[kind], 'board_size': SIZE, 'reuse_tree': True, **overrides}
        return create_search(kind, evaluator or UniformEvaluator(), **config)
    return _create


@pytest.fixture
def empty_board_5x5() -> GoBoard:
    """空的 5x5 GoBoard。"""
    return GoBoard(size=SIZE)


# ---------------------------------------------------------------------------
# Test Classes
# ---------------------------------------------------------------------------

class TestSearchInterfaceContract:
    """兩個實作都滿足 Search 介面的基本契約。"""

    @KINDS
    def test_is_search(self, make_search: Callable[..., Search], kind: str):
        """create_search 回傳的物件符合 Search Protocol。"""
        assert isinstance(make_search(kind), Search)

    @KINDS
    @pytest.mark.parametrize("color", ['b', 'w'], ids=["black", "white"])
    def test_policy_shape_and_normalization(
        self, make_search: Callable[..., Search], empty_board_5x5: GoBoard, kind: str, color: str
    ):
        """policy 長度 N*N+1、非負、總和為 1。"""
        policy, root = make_search(kind).search(empty_board_5x5, color)

        assert policy.shape == (SIZE * SIZE + 1,)
        assert np.all(policy >= 0)
        assert policy.sum() == pytest.approx(1.0, abs=1e-5)
        assert root.to_play == color

    @KINDS
    @pytest.mark.parametrize("temperature", [0.0, 1.0], ids=["greedy", "sampled"])
    def test_select_move_is_legal(
        self, make_search: Callable[..., Search], kind: str, temperature: float
    ):
        """select_move 回傳合法著手或 None（pass）。"""
        board = _setup_board(CAPTURE_SETUP)

        move = make_search(kind).select_move(board, 'w', temperature=temperature)

        assert move is None or board.is_legal(move[0], move[1], 'w')

    @KINDS
    def test_no_legal_moves_returns_pass(self, make_search: Callable[..., Search], kind: str):
        """沒有合法著手（只剩白棋兩眼）時回傳 pass。"""
        board = GoBoard(size=3)
        white = [(r, c) for r in range(3) for c in range(3) if (r, c) not in [(0, 0), (2, 2)]]
        board.board.apply_setup([], white, [])

        search = make_search(kind, board_size=3)
        policy, _ = search.search(board, 'b')

        assert policy[-1] == pytest.approx(1.0)
        assert search.select_move(board, 'b', temperature=0) is None


class TestSearchTreeReuse:
    """advance_tree 後重用子樹；局面不同時一律重建。"""

    @KINDS
    def test_reuse_after_advance(self, make_search: Callable[..., Search], empty_board_5x5: GoBoard, kind: str):
        """advance_tree 到已搜過的子節點，下一次 search 從該子樹繼續。"""
        search = make_search(kind)
        _, root = search.search(empty_board_5x5, 'b')
        move = _most_visited_move(root)
        child = root.children[move]
        visits_before = child.visit_count

        assert search.advance_tree(move) is True
        empty_board_5x5.play_move(move[0], move[1], 'b')
        _, new_root = search.search(empty_board_5x5, 'w')

        assert new_root is child
        assert new_root.parent is None
        assert new_root.visit_count > visits_before

    @KINDS
    def test_reuse_rejected_for_different_position(
        self, make_search: Callable[..., Search], empty_board_5x5: GoBoard, kind: str
    ):
        """同一方輪走但局面不同（如互動對弈沒呼叫 advance_tree）→ 新樹。"""
        search = make_search(kind)
        _, old_root = search.search(empty_board_5x5, 'b')
        other = _setup_board([(0, 0, 'b'), (4, 4, 'w')])

        _, new_root = search.search(other, 'b')

        assert new_root is not old_root
        assert position_key(new_root.board, new_root.to_play) == position_key(other, 'b')
        assert all(other.is_legal(m[0], m[1], 'b') for m in new_root.children)

    @KINDS
    def test_advance_then_different_position_rebuilds(
        self, make_search: Callable[..., Search], empty_board_5x5: GoBoard, kind: str
    ):
        """advance_tree 成功但實際局面不是那一手 → 驗證失敗、重建。"""
        search = make_search(kind)
        _, root = search.search(empty_board_5x5, 'b')
        move = _most_visited_move(root)
        child = root.children[move]
        search.advance_tree(move)
        other_move = next(m for m in root.children if m != move)
        empty_board_5x5.play_move(other_move[0], other_move[1], 'b')

        _, new_root = search.search(empty_board_5x5, 'w')

        assert new_root is not child
        assert position_key(new_root.board, 'w') == position_key(empty_board_5x5, 'w')

    @KINDS
    def test_advance_unknown_move_clears_tree(
        self, make_search: Callable[..., Search], empty_board_5x5: GoBoard, kind: str
    ):
        """advance_tree 到樹外的動作回傳 False；clear_tree 後不重用。"""
        search = make_search(kind)
        _, root = search.search(empty_board_5x5, 'b')

        assert search.advance_tree((SIZE, SIZE)) is False
        search.clear_tree()
        _, new_root = search.search(empty_board_5x5, 'b')
        assert new_root is not root


class TestSearchMaterialPreference:
    """MaterialEvaluator 下，兩種搜尋都選提子。"""

    @KINDS
    def test_picks_capture(self, make_search: Callable[..., Search], kind: str):
        """黑在 (2,3) 提白子。"""
        np.random.seed(0)
        board = _setup_board(CAPTURE_SETUP)
        search = make_search(kind, MaterialEvaluator(scale=2.0), num_simulations=120, reuse_tree=False)

        assert search.select_move(board, 'b', temperature=0) == CAPTURE_MOVE


class TestCreateSearch:
    """create_search 工廠與 MCTSOptimized 相容別名。"""

    def test_unknown_kind_raises(self):
        """未知種類丟 ValueError。"""
        with pytest.raises(ValueError):
            create_search('alphabeta', UniformEvaluator())

    def test_mcts_optimized_alias(self, empty_board_5x5: GoBoard):
        """MCTSOptimized 是帶批次＋快取設定的 MCTS，且會發出 DeprecationWarning。"""
        from core.mcts_optimized import MCTSOptimized

        with pytest.warns(DeprecationWarning):
            search = MCTSOptimized(UniformEvaluator(), num_simulations=20, batch_size=4)
        _, root = search.search(empty_board_5x5, 'b')

        assert isinstance(search, MCTS)
        assert root.visit_count == 20
        assert len(search.evaluation_cache) > 0

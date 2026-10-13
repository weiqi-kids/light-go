"""搜尋價值視角（value perspective）行為測試（issue #60）。

驗證四種搜尋（MCTS、MCTSOptimized、GumbelMCTS、MuZeroMCTS）都以
「輪走方視角」最大化價值：給一個能提掉大塊的局面，搜尋必須選提子那手；
並驗證 backup 的正負號隨輪次交替、MuZero 軌跡的價值目標轉成每步輪走方視角。
"""
from __future__ import annotations

from typing import List, Tuple, Union

import numpy as np
import pytest
import torch
import torch.nn as nn

from core.evaluator import MaterialEvaluator, PositionEvaluator
from core.game_rules import GoBoard
from core.gumbel_mcts import GumbelMCTS
from core.mcts import MCTS, MCTSNode
from core.mcts_optimized import MCTSOptimized
from core.muzero.mcts import MuZeroMCTS, MuZeroNode
from core.muzero.replay_buffer import Trajectory


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SIZE = 7

#: 黑白各有一塊叫吃中的棋，且雙方的急所不同：
#: Row 0: W W W W . B .     白四子只剩 (0,4) 一口氣，白自己下 (0,4) 是自殺
#: Row 1: B B B B B . .     → 黑在 (0,4) 提四子是黑的最佳著
#: Row 5: . . . . . . W
#: Row 6: . . . . . . B     黑一子只剩 (6,5) 一口氣 → 白的最佳著是 (6,5) 提一子
#: 若搜尋替「對手」挑最佳著，就會選到 (6,5) 或其他點，而不是 (0,4)。
BIG_GROUP_SETUP: List[Tuple[int, int, str]] = (
    [(0, c, 'w') for c in range(4)]
    + [(1, c, 'b') for c in range(5)]
    + [(0, 5, 'b'), (5, 6, 'w'), (6, 6, 'b')]
)
CAPTURE_BIG = (0, 4)
CAPTURE_SMALL = (6, 5)


def _swap(color: str) -> str:
    return 'w' if color == 'b' else 'b'


def _build_board(swap_colors: bool) -> GoBoard:
    """建立 BIG_GROUP_SETUP 局面；``swap_colors`` 時黑白互換。"""
    board = GoBoard(size=SIZE)
    for row, col, color in BIG_GROUP_SETUP:
        stone = _swap(color) if swap_colors else color
        assert board.play_move(row, col, stone), (row, col, stone)
    return board


class MaterialNet(nn.Module):
    """以 LightGo 特徵算子數差的真實小網路：均勻 policy、value = 輪走方子數優勢。

    Plane 0 中輪走方棋子為正、對手為負，所以
    value = tanh((own - opp) / 2)，正是「輪走方視角」的價值。
    """

    def __init__(self, size: int = SIZE):
        super().__init__()
        self.num_actions = size * size + 1
        self.anchor = nn.Parameter(torch.zeros(1))

    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        plane = features[:, 0].flatten(1)
        diff = (plane > 0).sum(dim=1).float() - (plane < 0).sum(dim=1).float()
        value = torch.tanh(diff / 2.0).unsqueeze(1) + 0.0 * self.anchor
        logits = torch.zeros(features.shape[0], self.num_actions) + 0.0 * self.anchor
        return logits, value


MaterialSource = Union[MaterialNet, PositionEvaluator]


class TargetActionNetworks(nn.Module):
    """MuZero 隱藏狀態網路：根節點玩家第一手下 ``target`` 就贏，否則輸。

    隱藏狀態 = [第一手動作, 深度]；value 一律回傳「該節點輪走方視角」：
    深度 1（對手輪走）時，第一手是 target → −1、否則 +1；更深的節點價值未知（0）。
    """

    def __init__(self, target: int, num_actions: int = 10):
        super().__init__()
        self.target = target
        self.num_actions = num_actions
        self.anchor = nn.Parameter(torch.zeros(1))

    def _uniform_logits(self) -> torch.Tensor:
        return torch.zeros(1, self.num_actions)

    def initial_inference(self, observation: torch.Tensor):
        hidden = torch.tensor([[-1.0, 0.0]])
        return hidden, self._uniform_logits(), torch.zeros(1, 1)

    def recurrent_inference(self, hidden_state: torch.Tensor, action: torch.Tensor):
        first, depth = hidden_state[0].tolist()
        first = float(action.item()) if depth == 0 else first
        depth += 1
        root_player_value = 1.0 if int(first) == self.target else -1.0
        to_play_value = -root_player_value if int(depth) == 1 else 0.0
        next_hidden = torch.tensor([[first, depth]])
        return next_hidden, torch.zeros(1, 1), self._uniform_logits(), torch.tensor([[to_play_value]])


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(params=["network_model", "material_evaluator"])
def material_net(request: pytest.FixtureRequest) -> MaterialSource:
    """子數評估：舊介面（傳 model，經 NetworkEvaluator）或直接傳評估器。"""
    if request.param == "network_model":
        return MaterialNet().eval()
    return MaterialEvaluator(scale=2.0)


# ---------------------------------------------------------------------------
# Test Classes
# ---------------------------------------------------------------------------

COLOR_CASES = pytest.mark.parametrize(
    "color,swap_colors",
    [('b', False), ('w', True)],
    ids=["black_to_move", "white_to_move"],
)


def _argmax_move(move_probs: np.ndarray) -> Tuple[int, int]:
    idx = int(np.argmax(move_probs))
    return divmod(idx, SIZE)


class TestSearchPicksCapture:
    """每種搜尋都要替輪走方選提大塊的那手。"""

    @COLOR_CASES
    def test_mcts_picks_capture(self, material_net: MaterialSource, color: str, swap_colors: bool):
        """MCTS 選提四子（而不是對手的最佳著）。"""
        np.random.seed(0)
        board = _build_board(swap_colors)
        mcts = MCTS(material_net, num_simulations=300, temperature=0,
                    dirichlet_epsilon=0.0, board_size=SIZE, reuse_tree=False)

        move_probs, _ = mcts.search(board, color)

        assert _argmax_move(move_probs) == CAPTURE_BIG

    @COLOR_CASES
    def test_mcts_optimized_picks_capture(self, material_net: MaterialSource, color: str, swap_colors: bool):
        """MCTSOptimized 選提四子。"""
        np.random.seed(0)
        board = _build_board(swap_colors)
        mcts = MCTSOptimized(material_net, num_simulations=300, temperature=0,
                             dirichlet_epsilon=0.0, batch_size=4, dynamic_batch=False)

        move_probs, _ = mcts.search(board, color)

        assert _argmax_move(move_probs) == CAPTURE_BIG

    @COLOR_CASES
    def test_gumbel_picks_capture(self, material_net: MaterialSource, color: str, swap_colors: bool):
        """GumbelMCTS 選提四子。"""
        np.random.seed(0)
        board = _build_board(swap_colors)
        mcts = GumbelMCTS(material_net, num_simulations=200, num_sampled_actions=64,
                          board_size=SIZE)

        move = mcts.select_move(board, color, temperature=0.0)

        assert move == CAPTURE_BIG

    @pytest.mark.parametrize("target", [3, 7], ids=["target_3", "target_7"])
    def test_muzero_picks_winning_action(self, target: int):
        """MuZeroMCTS 選讓根節點玩家獲勝的動作（兩人交替 backup）。"""
        np.random.seed(0)
        networks = TargetActionNetworks(target=target)
        mcts = MuZeroMCTS(networks, num_simulations=200, discount=1.0)

        policy, _, _ = mcts.search(torch.zeros(1, 2, 3, 3), add_exploration_noise=False)

        assert int(np.argmax(policy)) == target


class TestBackupAlternation:
    """Backup 的值必須逐層變號，且子節點存的是父節點輪走方視角。"""

    def test_mcts_backup_signs(self):
        """MCTS：葉節點評估 v（葉輪走方視角）→ 葉存 −v、父存 +v、祖父存 −v。"""
        board = GoBoard(size=SIZE)
        root = MCTSNode(board=board, color='w')
        child = MCTSNode(board=board, parent=root, color='b')
        leaf = MCTSNode(board=board, parent=child, color='w')
        mcts = MCTS(evaluator=MaterialEvaluator(), num_simulations=1, board_size=SIZE)

        mcts._backup(leaf, 0.6)

        assert [leaf.total_value, child.total_value, root.total_value] == pytest.approx([-0.6, 0.6, -0.6])

    def test_muzero_backup_signs(self):
        """MuZeroMCTS：葉存 −v、父存 +v、根存 −v（discount=1、reward=0）。"""
        mcts = MuZeroMCTS(TargetActionNetworks(target=0), num_simulations=1, discount=1.0)
        root = MuZeroNode()
        child = MuZeroNode(parent=root, action=0)
        leaf = MuZeroNode(parent=child, action=1)

        mcts._backup([root, child, leaf], 0.6)

        assert [leaf.q_value, child.q_value, root.q_value] == pytest.approx([-0.6, 0.6, -0.6])


class TestTrajectoryValuePerspective:
    """Trajectory 的價值目標要轉成每一步輪走方視角。"""

    @pytest.mark.parametrize("game_result", [1.0, -1.0], ids=["first_player_wins", "first_player_loses"])
    def test_game_result_alternates(self, game_result: float):
        """game_result（第 0 步輪走方視角）→ 第 i 步目標 = game_result × (−1)^i。"""
        trajectory = Trajectory()
        for step in range(4):
            trajectory.add_step(np.zeros((2, 3, 3)), action=step, reward=0.0,
                                policy=np.ones(10) / 10, value=0.0)
        trajectory.game_result = game_result

        targets = trajectory.compute_target_values(td_steps=10, discount=1.0)

        assert targets == pytest.approx([game_result, -game_result, game_result, -game_result])

    def test_bootstrap_value_flips_with_odd_td_steps(self):
        """bootstrap 值取自 td_steps 步之後（輪走方相反時要變號）。"""
        trajectory = Trajectory()
        for step in range(6):
            trajectory.add_step(np.zeros((2, 3, 3)), action=step, reward=0.0,
                                policy=np.ones(10) / 10, value=0.5)

        targets = trajectory.compute_target_values(td_steps=1, discount=1.0)

        assert targets[0] == pytest.approx(-0.5)

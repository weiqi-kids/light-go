"""Position evaluator 接縫測試（issue #54）。

驗證 core/evaluator.py 的各實作都遵守同一個介面慣例：
policy/logits 長度 N²+1（pass 在最後）、policy 為機率分布、value 為輪走方視角。
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from core.evaluator import (
    Evaluation,
    MaterialEvaluator,
    MuZeroEvaluator,
    NetworkEvaluator,
    PositionEvaluator,
    UniformEvaluator,
    as_evaluator,
)
from core.game_rules import GoBoard
from core.muzero.networks import MuZeroNetworks
from hf_models.modeling_go_ai import GoAIModel
from input.lightgo_encoder import encode_board


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def board_with_stones() -> GoBoard:
    """9x9 board: black 3 stones, white 1 stone."""
    board = GoBoard(size=9)
    for row, col, color in [(2, 2, 'b'), (2, 3, 'b'), (6, 6, 'b'), (4, 4, 'w')]:
        board.play_move(row, col, color)
    return board


@pytest.fixture(scope="module")
def go_ai_model() -> GoAIModel:
    """Small 9x9 GoAIModel in eval mode."""
    torch.manual_seed(0)
    return GoAIModel(num_input_planes=2, num_filters=16, num_blocks=1, board_size=9).eval()


@pytest.fixture(scope="module")
def muzero_networks() -> MuZeroNetworks:
    """Small 9x9 MuZeroNetworks in eval mode."""
    torch.manual_seed(0)
    return MuZeroNetworks(board_size=9, num_input_planes=2, hidden_size=16, num_blocks=4).eval()


# ---------------------------------------------------------------------------
# Test Classes
# ---------------------------------------------------------------------------

class TestEvaluatorContract:
    """所有評估器輸出相同形狀與語意。"""

    @pytest.mark.parametrize(
        "kind",
        ["network", "muzero", "uniform", "material"],
        ids=["network", "muzero", "uniform", "material"],
    )
    def test_output_shape_and_normalization(
        self,
        kind: str,
        go_ai_model: GoAIModel,
        muzero_networks: MuZeroNetworks,
        board_with_stones: GoBoard,
    ):
        """policy/logits 長度 82、policy 加總為 1、value 在 [-1, 1]。"""
        evaluator: PositionEvaluator = {
            "network": lambda: NetworkEvaluator(go_ai_model),
            "muzero": lambda: MuZeroEvaluator(muzero_networks),
            "uniform": lambda: UniformEvaluator(),
            "material": lambda: MaterialEvaluator(),
        }[kind]()

        results = evaluator.evaluate([board_with_stones, GoBoard(size=9)], ['b', 'w'])

        assert len(results) == 2
        for evaluation in results:
            assert isinstance(evaluation, Evaluation)
            assert evaluation.policy.shape == (82,)
            assert evaluation.logits.shape == (82,)
            assert evaluation.policy.sum() == pytest.approx(1.0, abs=1e-5)
            assert -1.0 <= evaluation.value <= 1.0

    def test_empty_batch(self, go_ai_model: GoAIModel):
        """空批次回傳空列表。"""
        assert NetworkEvaluator(go_ai_model).evaluate([], []) == []


class TestNetworkEvaluator:
    """NetworkEvaluator 擁有編碼、批次、裝置與 softmax。"""

    def test_matches_direct_model_call(self, go_ai_model: GoAIModel, board_with_stones: GoBoard):
        """結果與手動 encode → model → softmax 一致。"""
        features = torch.from_numpy(encode_board(board_with_stones, 'w')).unsqueeze(0)
        with torch.no_grad():
            logits, value = go_ai_model(features)

        evaluation = NetworkEvaluator(go_ai_model).evaluate_one(board_with_stones, 'w')

        np.testing.assert_allclose(evaluation.logits, logits[0].numpy(), rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(evaluation.policy, torch.softmax(logits, dim=1)[0].numpy(), rtol=1e-5, atol=1e-6)
        assert evaluation.value == pytest.approx(value.item(), abs=1e-6)

    def test_batch_equals_single(self, go_ai_model: GoAIModel, board_with_stones: GoBoard):
        """批次評估與逐一評估結果相同。"""
        evaluator = NetworkEvaluator(go_ai_model)
        boards = [board_with_stones, GoBoard(size=9)]
        colors = ['b', 'w']

        batch = evaluator.evaluate(boards, colors)
        singles = [evaluator.evaluate_one(board, color) for board, color in zip(boards, colors)]

        for got, expected in zip(batch, singles):
            np.testing.assert_allclose(got.policy, expected.policy, rtol=1e-4, atol=1e-6)
            assert got.value == pytest.approx(expected.value, abs=1e-5)

    def test_mixed_board_sizes_rejected(self, go_ai_model: GoAIModel):
        """同一批次不同大小的棋盤要明確報錯。"""
        with pytest.raises(ValueError):
            NetworkEvaluator(go_ai_model).evaluate([GoBoard(size=9), GoBoard(size=13)], ['b', 'b'])


class TestMuZeroEvaluator:
    """MuZeroEvaluator = representation + prediction。"""

    def test_matches_initial_inference(self, muzero_networks: MuZeroNetworks, board_with_stones: GoBoard):
        """結果與 networks.initial_inference 一致。"""
        features = torch.from_numpy(encode_board(board_with_stones, 'b')).unsqueeze(0)
        with torch.no_grad():
            _, logits, value = muzero_networks.initial_inference(features)

        evaluation = MuZeroEvaluator(muzero_networks).evaluate_one(board_with_stones, 'b')

        np.testing.assert_allclose(evaluation.logits, logits[0].numpy(), rtol=1e-5, atol=1e-6)
        assert evaluation.value == pytest.approx(value.item(), abs=1e-6)


class TestSimpleEvaluators:
    """UniformEvaluator / MaterialEvaluator。"""

    def test_uniform_policy_and_value(self):
        """均勻 policy、value 為指定常數。"""
        evaluation = UniformEvaluator(value=0.25).evaluate_one(GoBoard(size=5), 'b')

        np.testing.assert_allclose(evaluation.policy, np.full(26, 1 / 26))
        assert evaluation.value == 0.25

    @pytest.mark.parametrize("color,sign", [('b', 1.0), ('w', -1.0)], ids=["black", "white"])
    def test_material_value_side_to_move(self, board_with_stones: GoBoard, color: str, sign: float):
        """子數優勢以輪走方視角計算：黑多兩子 → 黑 +、白 −。"""
        value = MaterialEvaluator(scale=2.0).evaluate_one(board_with_stones, color).value

        assert value == pytest.approx(sign * np.tanh(1.0))

    def test_policy_dict(self):
        """policy_dict 以 (row, col) 與 'pass' 為鍵。"""
        policy = UniformEvaluator().evaluate_one(GoBoard(size=3), 'b').policy_dict()

        assert len(policy) == 10
        assert policy[(2, 2)] == pytest.approx(0.1)
        assert policy['pass'] == pytest.approx(0.1)


class TestAsEvaluator:
    """搜尋建構子的向後相容入口。"""

    def test_wraps_model(self, go_ai_model: GoAIModel):
        """傳 model → NetworkEvaluator。"""
        assert isinstance(as_evaluator(go_ai_model), NetworkEvaluator)

    def test_passes_evaluator_through(self):
        """傳評估器（位置參數或 evaluator=）→ 原樣回傳，evaluator= 優先。"""
        uniform = UniformEvaluator()
        material = MaterialEvaluator()

        assert as_evaluator(uniform) is uniform
        assert as_evaluator(uniform, material) is material

    def test_requires_something(self):
        """兩者都沒給 → ValueError。"""
        with pytest.raises(ValueError):
            as_evaluator(None)

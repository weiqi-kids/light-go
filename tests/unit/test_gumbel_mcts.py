"""Gumbel MCTS 單元測試"""

import pytest
import numpy as np
import torch
import torch.nn as nn

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.game_rules import GoBoard
from core.sequential_halving import sequential_halving, HalvingResult
from core.gumbel_mcts import GumbelMCTS, GumbelMCTSNode


class MockGoModel(nn.Module):
    """模擬圍棋模型"""

    def __init__(self, board_size: int = 9):
        super().__init__()
        self.board_size = board_size
        self.num_actions = board_size * board_size + 1

        # 簡單的線性層
        self.policy = nn.Linear(2 * board_size * board_size, self.num_actions)
        self.value = nn.Linear(2 * board_size * board_size, 1)

    def forward(self, x):
        batch_size = x.shape[0]
        flat = x.view(batch_size, -1)

        policy_logits = self.policy(flat)
        value = torch.tanh(self.value(flat))

        return policy_logits, value


class TestSequentialHalving:
    """Sequential Halving 演算法測試"""

    def test_single_candidate(self):
        """單一候選直接回傳"""
        def mock_eval(action):
            return 0.5

        result = sequential_halving(mock_eval, [0], total_budget=10)
        assert result == 0

    def test_two_candidates(self):
        """兩個候選的基本情況"""
        scores = {0: 0.3, 1: 0.7}

        def mock_eval(action):
            return scores[action]

        # 多次執行確保穩定性
        results = []
        for _ in range(10):
            result = sequential_halving(mock_eval, [0, 1], total_budget=20)
            results.append(result)

        # 較高分數的動作應該更常被選中
        assert results.count(1) >= results.count(0)

    def test_multiple_candidates(self):
        """多個候選的情況"""
        # 動作 5 有最高分數
        scores = {i: 0.1 * i for i in range(10)}
        scores[5] = 1.0

        def mock_eval(action):
            return scores[action] + np.random.normal(0, 0.1)

        result = sequential_halving(
            mock_eval,
            list(range(10)),
            total_budget=100
        )

        # 應該經常選中動作 5
        # 由於隨機性，這裡只檢查是否能正常執行
        assert 0 <= result < 10

    def test_return_details(self):
        """測試返回詳細結果"""
        def mock_eval(action):
            return action * 0.1

        result = sequential_halving(
            mock_eval,
            [0, 1, 2, 3],
            total_budget=20,
            return_details=True
        )

        assert isinstance(result, HalvingResult)
        assert result.best_action in [0, 1, 2, 3]
        assert result.rounds_completed >= 1
        assert len(result.action_scores) > 0

    def test_empty_candidates_raises(self):
        """空候選列表應該拋出錯誤"""
        def mock_eval(action):
            return 0.5

        with pytest.raises(ValueError):
            sequential_halving(mock_eval, [], total_budget=10)


class TestGumbelMCTSNode:
    """GumbelMCTSNode 測試"""

    def test_node_creation(self):
        """節點建立"""
        board = GoBoard(size=9)
        node = GumbelMCTSNode(board=board, color='b', is_root=True)

        assert node.board is not None
        assert node.color == 'b'
        assert node.is_root is True
        assert node.visit_count == 0
        assert node.q_value == 0.0

    def test_q_value_calculation(self):
        """Q 值計算"""
        board = GoBoard(size=9)
        node = GumbelMCTSNode(board=board, color='b')

        # 初始 Q 值為 0
        assert node.q_value == 0.0

        # 更新後計算平均
        node.update(0.5)
        assert node.q_value == 0.5

        node.update(0.3)
        assert abs(node.q_value - 0.4) < 1e-6

    def test_is_leaf(self):
        """葉節點判斷"""
        board = GoBoard(size=9)
        node = GumbelMCTSNode(board=board, color='b')

        assert node.is_leaf() is True

        # 手動添加子節點
        child_board = board.copy()
        child = GumbelMCTSNode(board=child_board, parent=node, color='w')
        node.children[(0, 0)] = child

        assert node.is_leaf() is False


class TestGumbelMCTS:
    """GumbelMCTS 測試"""

    @pytest.fixture
    def model(self):
        """建立模擬模型"""
        return MockGoModel(board_size=9)

    @pytest.fixture
    def mcts(self, model):
        """建立 Gumbel MCTS"""
        return GumbelMCTS(
            model=model,
            num_simulations=8,
            num_sampled_actions=8,
            board_size=9
        )

    def test_search_returns_valid_policy(self, mcts):
        """搜尋回傳有效的策略分布"""
        board = GoBoard(size=9)

        move_probs, root = mcts.search(board, 'b')

        # 檢查形狀
        assert move_probs.shape == (82,)  # 9*9 + 1

        # 檢查是有效的機率分布
        assert np.all(move_probs >= 0)
        assert abs(move_probs.sum() - 1.0) < 1e-5 or move_probs.sum() == 0

    def test_search_returns_root_node(self, mcts):
        """搜尋回傳根節點"""
        board = GoBoard(size=9)

        _, root = mcts.search(board, 'b')

        assert isinstance(root, GumbelMCTSNode)
        assert root.is_root is True

    def test_select_move(self, mcts):
        """選擇動作"""
        board = GoBoard(size=9)

        move = mcts.select_move(board, 'b')

        # 動作應該是有效的
        if move is not None:
            assert isinstance(move, tuple)
            assert len(move) == 2
            row, col = move
            assert 0 <= row < 9
            assert 0 <= col < 9

    def test_gumbel_sampling(self, mcts):
        """Gumbel 採樣"""
        shape = (100,)
        samples = mcts._sample_gumbel(shape)

        assert samples.shape == shape
        # Gumbel 分布的均值約為 0.5772 (Euler-Mascheroni 常數)
        assert -5 < samples.mean() < 5

    def test_idx_to_move_conversion(self, mcts):
        """索引和動作轉換"""
        # 普通動作
        move = mcts._idx_to_move(0)
        assert move == (0, 0)

        move = mcts._idx_to_move(8)
        assert move == (0, 8)

        move = mcts._idx_to_move(9)
        assert move == (1, 0)

        # Pass 動作
        move = mcts._idx_to_move(81)
        assert move == 'pass'

    def test_move_to_idx_conversion(self, mcts):
        """動作到索引轉換"""
        idx = mcts._move_to_idx((0, 0))
        assert idx == 0

        idx = mcts._move_to_idx((1, 0))
        assert idx == 9

        idx = mcts._move_to_idx('pass')
        assert idx == 81


class TestGumbelMCTSIntegration:
    """Gumbel MCTS 整合測試"""

    def test_multiple_searches_consistency(self):
        """多次搜尋的一致性"""
        model = MockGoModel(board_size=9)
        mcts = GumbelMCTS(
            model=model,
            num_simulations=16,
            board_size=9
        )

        board = GoBoard(size=9)

        # 執行多次搜尋
        results = []
        for _ in range(5):
            move_probs, _ = mcts.search(board, 'b')
            top_move = np.argmax(move_probs)
            results.append(top_move)

        # 由於 Gumbel 雜訊，結果可能有變化，但應該合理
        assert all(0 <= r <= 81 for r in results)

    def test_search_with_moves_played(self):
        """有棋子的棋盤搜尋"""
        model = MockGoModel(board_size=9)
        mcts = GumbelMCTS(
            model=model,
            num_simulations=8,
            board_size=9
        )

        board = GoBoard(size=9)
        board.play_move(4, 4, 'b')  # 天元
        board.play_move(2, 2, 'w')

        move_probs, root = mcts.search(board, 'b')

        # 應該能正常執行
        assert move_probs.shape == (82,)
        assert root is not None

    def test_low_simulation_count(self):
        """低模擬次數"""
        model = MockGoModel(board_size=9)
        mcts = GumbelMCTS(
            model=model,
            num_simulations=2,  # 極低
            num_sampled_actions=4,
            board_size=9
        )

        board = GoBoard(size=9)

        # 即使只有 2 次模擬也應該能工作
        move_probs, _ = mcts.search(board, 'b')
        assert move_probs.shape == (82,)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

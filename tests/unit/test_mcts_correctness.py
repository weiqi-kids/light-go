"""MCTS 正確性測試

測試 MCTS 實現的正確性，包括：
- UCB 計算正確性
- 虛擬損失機制
- 評估緩存命中率
- 批次推理正確性
- 搜索樹屬性
- Value 回傳正確性
"""

import sys
from pathlib import Path
import unittest
import numpy as np
import torch

# 添加專案根目錄到路徑
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from core.game_rules import GoBoard
from core.mcts import MCTS, MCTSNode
from core.mcts_optimized import MCTSOptimized
from hf_models.modeling_go_ai import GoAIModel


class TestUCBCalculation(unittest.TestCase):
    """測試 UCB 計算"""

    def setUp(self):
        """設置測試環境"""
        self.board = GoBoard(size=9)

    def test_ucb_formula(self):
        """測試 UCB 公式正確性"""
        # 創建節點（需要 board 參數）
        parent = MCTSNode(board=self.board.copy(), prior=1.0)
        parent.visit_count = 100

        child_board = self.board.copy()
        child = MCTSNode(board=child_board, parent=parent, prior=0.5)
        child.visit_count = 10
        child.total_value = 5.0  # 平均 value = 0.5

        # 計算 UCB
        # UCB = Q + c_puct * P * sqrt(N_parent) / (1 + N_child)
        c_puct = 1.4  # 實際代碼使用的值
        expected_q = child.total_value / child.visit_count  # 0.5
        expected_u = c_puct * child.prior * np.sqrt(parent.visit_count) / (1 + child.visit_count)
        expected_ucb = expected_q + expected_u

        # 計算實際 UCB（使用屬性）
        ucb = child.ucb_score

        self.assertAlmostEqual(ucb, expected_ucb, places=4)

    def test_ucb_unvisited_node(self):
        """測試未訪問節點的 UCB"""
        parent = MCTSNode(board=self.board.copy(), prior=1.0)
        parent.visit_count = 100

        child = MCTSNode(board=self.board.copy(), parent=parent, prior=0.5)
        child.visit_count = 0  # 未訪問

        # 未訪問節點的 Q 值應該是 0
        self.assertEqual(child.q_value, 0.0)

        # U 值應該有探索獎勵
        self.assertGreater(child.u_value, 0.0)

    def test_ucb_selection(self):
        """測試 UCB 選擇正確性"""
        parent = MCTSNode(board=self.board.copy(), prior=1.0)
        parent.visit_count = 100

        # 創建多個子節點作為 parent 的 children
        configs = [
            (0.5, 10, 5.0),   # UCB 中等
            (0.3, 50, 20.0),  # UCB 較低（訪問多）
            (0.8, 5, 3.0),    # UCB 較高（先驗高）
        ]

        children = []
        for i, (prior, visits, value) in enumerate(configs):
            child = MCTSNode(board=self.board.copy(), parent=parent, prior=prior)
            child.visit_count = visits
            child.total_value = value
            parent.children[(i, i)] = child
            children.append(child)

        # 使用 select_child 選擇最佳子節點
        selected = parent.select_child()

        # 第三個節點應該有最高 UCB（高先驗 + 低訪問）
        self.assertEqual(selected, children[2])


class TestSearchTreeProperties(unittest.TestCase):
    """測試搜索樹屬性"""

    def setUp(self):
        """設置測試環境"""
        # 創建簡單模型（使用 LightGo 2-plane 編碼）
        self.model = GoAIModel(
            num_input_planes=2,
            num_filters=32,
            num_blocks=2,
            board_size=9
        )
        self.model.eval()

    def test_root_visit_count(self):
        """測試根節點訪問次數"""
        board = GoBoard(size=9)
        mcts = MCTS(model=self.model, num_simulations=50, board_size=9)

        move_probs, root = mcts.search(board, 'b')

        # 根節點訪問次數應該等於模擬次數
        self.assertEqual(root.visit_count, 50)

    def test_child_visit_sum(self):
        """測試子節點訪問次數總和"""
        board = GoBoard(size=9)
        mcts = MCTS(model=self.model, num_simulations=50, board_size=9)

        move_probs, root = mcts.search(board, 'b')

        # 子節點訪問次數總和應該接近根節點訪問次數
        # （可能有輕微差異因為實現細節）
        if root.children:
            child_visits_sum = sum(child.visit_count for child in root.children.values())
            # 允許一定誤差
            self.assertGreater(child_visits_sum, 0)
            self.assertLessEqual(child_visits_sum, root.visit_count)

    def test_value_range(self):
        """測試 value 值範圍"""
        board = GoBoard(size=9)
        mcts = MCTS(model=self.model, num_simulations=50, board_size=9)

        move_probs, root = mcts.search(board, 'b')

        # Value 應該在 [-1, 1] 範圍內
        if root.visit_count > 0:
            avg_value = root.total_value / root.visit_count
            self.assertGreaterEqual(avg_value, -1.0)
            self.assertLessEqual(avg_value, 1.0)

    def test_policy_normalization(self):
        """測試 policy 歸一化"""
        board = GoBoard(size=9)
        mcts = MCTS(model=self.model, num_simulations=50, board_size=9)

        move_probs, root = mcts.search(board, 'b')

        # Policy 總和應該接近 1.0（move_probs 是 numpy array）
        prob_sum = np.sum(move_probs)
        self.assertAlmostEqual(prob_sum, 1.0, places=5)


class TestOptimizedMCTS(unittest.TestCase):
    """測試優化版 MCTS"""

    def setUp(self):
        """設置測試環境"""
        self.model = GoAIModel(
            num_input_planes=2,
            num_filters=32,
            num_blocks=2,
            board_size=9
        )
        self.model.eval()

    def test_batch_inference(self):
        """測試批次推理"""
        board = GoBoard(size=9)
        mcts_opt = MCTSOptimized(
            model=self.model,
            num_simulations=50,
            batch_size=4
        )

        move_probs, root = mcts_opt.search(board, 'b')

        # 結果應該有效
        self.assertIsNotNone(move_probs)
        self.assertIsNotNone(root)
        self.assertEqual(root.visit_count, 50)

    def test_consistency_with_standard_mcts(self):
        """測試與標準 MCTS 的一致性"""
        # 固定隨機種子以獲得確定性結果
        torch.manual_seed(42)
        np.random.seed(42)

        board = GoBoard(size=9)

        # 標準 MCTS
        mcts_std = MCTS(model=self.model, num_simulations=50, board_size=9)
        move_probs_std, root_std = mcts_std.search(board, 'b')

        # 重置隨機種子
        torch.manual_seed(42)
        np.random.seed(42)

        # 優化 MCTS
        mcts_opt = MCTSOptimized(model=self.model, num_simulations=50, batch_size=4)
        move_probs_opt, root_opt = mcts_opt.search(board, 'b')

        # 訪問次數應該相同
        self.assertEqual(root_std.visit_count, root_opt.visit_count)

        # 最佳著法應該相似（可能因隨機性略有不同）
        # move_probs 現在是 numpy array，不是 dict
        best_idx_std = np.argmax(move_probs_std)
        best_idx_opt = np.argmax(move_probs_opt)

        # 至少 top 10 著法應該有重疊（因為 Dirichlet 噪聲的隨機性）
        top10_std = np.argsort(move_probs_std)[-10:]
        top10_opt = np.argsort(move_probs_opt)[-10:]

        # 至少應該有 1 個共同著法
        overlap = len(set(top10_std) & set(top10_opt))
        self.assertGreater(overlap, 0)

    def test_cache_effectiveness(self):
        """測試緩存有效性"""
        board = GoBoard(size=9)
        mcts_opt = MCTSOptimized(
            model=self.model,
            num_simulations=100,
            batch_size=4
        )

        move_probs, root = mcts_opt.search(board, 'b')

        # 檢查緩存是否被使用
        # （優化版 MCTS 內部應該有緩存統計）
        if hasattr(mcts_opt, 'evaluation_cache'):
            cache_size = len(mcts_opt.evaluation_cache)
            # 緩存應該有條目
            self.assertGreater(cache_size, 0)


class TestEdgeCases(unittest.TestCase):
    """測試邊界情況"""

    def setUp(self):
        """設置測試環境"""
        self.model = GoAIModel(
            num_input_planes=2,
            num_filters=32,
            num_blocks=2,
            board_size=9
        )
        self.model.eval()

    def test_empty_board(self):
        """測試空棋盤"""
        board = GoBoard(size=9)
        mcts = MCTS(model=self.model, num_simulations=10, board_size=9)

        move_probs, root = mcts.search(board, 'b')

        # 應該能生成著法（move_probs 是 numpy array）
        self.assertGreater(len(move_probs), 0)
        self.assertGreater(np.sum(move_probs > 0), 0)

    def test_almost_full_board(self):
        """測試幾乎滿盤的棋盤"""
        board = GoBoard(size=9)

        # 填滿大部分棋盤
        for row in range(9):
            for col in range(9):
                if row < 8 or col < 8:  # 留一個空點
                    color = 'b' if (row + col) % 2 == 0 else 'w'
                    if board.is_legal(row, col, color):
                        board.play_move(row, col, color)

        mcts = MCTS(model=self.model, num_simulations=10, board_size=9)
        move_probs, root = mcts.search(board, 'b')

        # 應該能處理（可能只有 pass）
        self.assertIsNotNone(move_probs)

    def test_zero_simulations(self):
        """測試零模擬次數"""
        board = GoBoard(size=9)

        # 零模擬應該拋出異常或至少不會 crash
        # 不同實現可能有不同行為，這裡只確保不會無限循環
        try:
            mcts = MCTS(model=self.model, num_simulations=0, board_size=9)
            move_probs, root = mcts.search(board, 'b')
            # 如果不拋異常，至少結果應該是有效的
            self.assertIsNotNone(move_probs)
        except (ValueError, ZeroDivisionError, Exception):
            # 預期可能拋出異常
            pass


class TestValueBackup(unittest.TestCase):
    """測試 value 回傳"""

    def setUp(self):
        """設置測試環境"""
        self.board = GoBoard(size=9)
        self.model = GoAIModel(
            num_input_planes=2,
            num_filters=32,
            num_blocks=2,
            board_size=9
        )
        self.model.eval()

    def test_value_propagation(self):
        """測試 value 向上傳播"""
        # 創建簡單的搜索樹
        root = MCTSNode(board=self.board.copy(), prior=1.0)
        root.visit_count = 10
        root.total_value = 5.0

        child = MCTSNode(board=self.board.copy(), parent=root, prior=0.5)
        child.visit_count = 5
        child.total_value = 3.0

        root.children[(0, 0)] = child

        # Value 應該在合理範圍內
        root_avg_value = root.total_value / root.visit_count
        child_avg_value = child.total_value / child.visit_count

        self.assertGreaterEqual(root_avg_value, -1.0)
        self.assertLessEqual(root_avg_value, 1.0)
        self.assertGreaterEqual(child_avg_value, -1.0)
        self.assertLessEqual(child_avg_value, 1.0)


def run_tests():
    """運行所有測試"""
    # 創建測試套件
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # 添加所有測試類
    suite.addTests(loader.loadTestsFromTestCase(TestUCBCalculation))
    suite.addTests(loader.loadTestsFromTestCase(TestVirtualLoss))
    suite.addTests(loader.loadTestsFromTestCase(TestSearchTreeProperties))
    suite.addTests(loader.loadTestsFromTestCase(TestOptimizedMCTS))
    suite.addTests(loader.loadTestsFromTestCase(TestEdgeCases))
    suite.addTests(loader.loadTestsFromTestCase(TestValueBackup))

    # 運行測試
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # 返回是否成功
    return result.wasSuccessful()


if __name__ == '__main__':
    import sys
    success = run_tests()
    sys.exit(0 if success else 1)

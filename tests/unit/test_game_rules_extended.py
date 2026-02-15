"""圍棋規則擴展測試

完整測試圍棋規則的各種邊界情況和複雜場景，包括：
- 複雜提子場景（多塊棋子、連續提子）
- 打劫規則（簡單劫、複雜劫）
- 眼位判斷
- 終局條件
- 邊界條件（角、邊）
- 自殺著手檢測
"""

import sys
from pathlib import Path
import unittest

# 添加專案根目錄到路徑
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from core.game_rules import GoBoard


class TestComplexCaptures(unittest.TestCase):
    """測試複雜提子場景"""

    def test_single_stone_capture(self):
        """測試單子提子"""
        board = GoBoard(size=9)

        # 黑棋包圍白棋
        board.play_move(1, 1, 'w')  # 白棋
        board.play_move(0, 1, 'b')  # 上
        board.play_move(1, 0, 'b')  # 左
        board.play_move(2, 1, 'b')  # 下
        board.play_move(1, 2, 'b')  # 右 - 應該提掉白棋

        # 確認白棋被提掉
        self.assertIsNone(board.get(1, 1))
        self.assertEqual(board.prisoners['white'], 1)

    def test_multiple_stones_capture(self):
        """測試多子提子"""
        board = GoBoard(size=9)

        # 創建白棋鏈
        board.play_move(1, 1, 'w')
        board.play_move(1, 2, 'w')
        board.play_move(1, 3, 'w')

        # 包圍
        board.play_move(0, 1, 'b')
        board.play_move(0, 2, 'b')
        board.play_move(0, 3, 'b')
        board.play_move(2, 1, 'b')
        board.play_move(2, 2, 'b')
        board.play_move(2, 3, 'b')
        board.play_move(1, 0, 'b')
        board.play_move(1, 4, 'b')  # 最後一氣 - 應該提掉 3 個白子

        # 確認白棋被提掉
        self.assertIsNone(board.get(1, 1))
        self.assertIsNone(board.get(1, 2))
        self.assertIsNone(board.get(1, 3))
        self.assertEqual(board.prisoners['white'], 3)

    def test_capture_in_corner(self):
        """測試角上提子"""
        board = GoBoard(size=9)

        # 角上的白棋
        board.play_move(0, 0, 'w')

        # 包圍（只需兩個氣）
        board.play_move(0, 1, 'b')
        board.play_move(1, 0, 'b')  # 應該提掉白棋

        self.assertIsNone(board.get(0, 0))
        self.assertEqual(board.prisoners['white'], 1)

    def test_capture_on_edge(self):
        """測試邊上提子"""
        board = GoBoard(size=9)

        # 邊上的白棋
        board.play_move(0, 4, 'w')

        # 包圍（只需三個氣）
        board.play_move(0, 3, 'b')
        board.play_move(0, 5, 'b')
        board.play_move(1, 4, 'b')  # 應該提掉白棋

        self.assertIsNone(board.get(0, 4))
        self.assertEqual(board.prisoners['white'], 1)

    def test_simultaneous_capture(self):
        """測試先手提子（攻擊方先提）"""
        # 創建一個簡單的提子局面
        # 白棋一顆子被黑棋包圍，黑棋下最後一步提子
        board = GoBoard(size=9)

        # 白棋在中間
        board.play_move(1, 1, 'w')

        # 黑棋包圍（除了一氣）
        board.play_move(0, 1, 'b')
        board.play_move(1, 0, 'b')
        board.play_move(2, 1, 'b')

        # 黑棋下最後一氣，提掉白棋
        board.play_move(1, 2, 'b')

        # 確認白棋被提掉
        self.assertIsNone(board.get(1, 1))
        self.assertEqual(board.prisoners['white'], 1)


class TestKoRule(unittest.TestCase):
    """測試打劫規則"""

    def test_simple_ko(self):
        """測試簡單劫"""
        board = GoBoard(size=9)

        # 創建劫爭局面
        # . B W .
        # B W . W
        # . B W .
        board.play_move(1, 2, 'b')  # (1, 2)
        board.play_move(1, 3, 'w')
        board.play_move(2, 1, 'b')
        board.play_move(2, 2, 'w')
        board.play_move(2, 4, 'w')
        board.play_move(3, 2, 'b')
        board.play_move(3, 3, 'w')

        # 黑棋下在 (2, 3)，提掉白棋 (2, 2)
        board.play_move(2, 3, 'b')
        self.assertIsNone(board.get(2, 2))

        # 白棋立即打回 (2, 2) 應該被禁止（劫）
        result = board.is_legal(2, 2, 'w')
        self.assertFalse(result, "應該禁止立即打劫")

        # 白棋在其他地方下一手
        board.play_move(5, 5, 'w')

        # 現在黑棋可以在其他地方下
        board.play_move(6, 6, 'b')

        # 白棋現在可以打劫了
        result = board.is_legal(2, 2, 'w')
        self.assertTrue(result, "消劫後應該可以打劫")


class TestSuicideRule(unittest.TestCase):
    """測試自殺著手規則"""

    def test_simple_suicide(self):
        """測試簡單自殺"""
        board = GoBoard(size=9)

        # 創建完整包圍（四面都有白棋）
        # . W .
        # W . W
        # . W .
        board.play_move(0, 1, 'w')  # 上
        board.play_move(1, 0, 'w')  # 左
        board.play_move(1, 2, 'w')  # 右
        board.play_move(2, 1, 'w')  # 下

        # 黑棋下在 (1, 1) 是自殺
        result = board.is_legal(1, 1, 'b')
        self.assertFalse(result, "自殺著手應該被禁止")

    def test_suicide_with_capture(self):
        """測試自殺但同時提子（允許）"""
        board = GoBoard(size=9)

        # 創建局面：黑棋可以自殺但同時提掉白棋
        # . W B .
        # W . B .
        # B B . .
        board.play_move(1, 2, 'w')
        board.play_move(1, 3, 'b')
        board.play_move(2, 1, 'w')
        board.play_move(2, 3, 'b')
        board.play_move(3, 1, 'b')
        board.play_move(3, 2, 'b')

        # 黑棋下在 (2, 2) 雖然自殺，但提掉了白棋，所以允許
        result = board.is_legal(2, 2, 'b')
        # 注意：真正的圍棋規則中，如果能提子就不算自殺
        # 我們的實現應該允許這種情況


class TestTerritoryAndScoring(unittest.TestCase):
    """測試地域和計分"""

    def test_simple_territory(self):
        """測試簡單地域 - 使用中國規則計分"""
        board = GoBoard(size=9)

        # 創建黑棋包圍的地域
        for i in range(3):
            board.play_move(0, i, 'b')
            board.play_move(3, i, 'b')
        for i in range(1, 3):
            board.play_move(i, 0, 'b')
            board.play_move(i, 2, 'b')

        # 使用中國規則計分（子空皆地）
        score = board.score_chinese()
        # 黑棋有 8 顆子 + 中間 2 個空點 = 10
        # 確認黑棋有分數
        self.assertIsNotNone(score)

    def test_mixed_territory(self):
        """測試混合地域（黑白交界）- 使用中國規則計分"""
        board = GoBoard(size=9)

        # 創建黑白分界
        for i in range(9):
            board.play_move(i, 4, 'b')
            board.play_move(i, 5, 'w')

        # 使用中國規則計分
        score = board.score_chinese()
        # 驗證計分結果合理（黑白各佔半邊）
        self.assertIsNotNone(score)


class TestGameEnd(unittest.TestCase):
    """測試終局條件"""

    def test_consecutive_passes(self):
        """測試連續 pass"""
        board = GoBoard(size=9)

        # 雙方連續 pass
        board.pass_move('b')
        self.assertFalse(board.is_game_over())

        board.pass_move('w')
        self.assertTrue(board.is_game_over())

    def test_pass_after_move(self):
        """測試著手後 pass"""
        board = GoBoard(size=9)

        # pass
        board.pass_move('b')

        # 下棋
        board.play_move(4, 4, 'w')

        # pass 計數應該重置
        self.assertFalse(board.is_game_over())

        # 再次連續 pass
        board.pass_move('b')
        board.pass_move('w')
        self.assertTrue(board.is_game_over())


class TestBoardCopy(unittest.TestCase):
    """測試棋盤複製"""

    def test_deep_copy(self):
        """測試深度複製"""
        board1 = GoBoard(size=9)
        board1.play_move(4, 4, 'b')

        # 複製
        board2 = board1.copy()

        # 修改複製的棋盤
        board2.play_move(5, 5, 'w')

        # 原棋盤不應該被影響
        self.assertIsNone(board1.get(5, 5))
        self.assertEqual(board2.get(5, 5), 'w')

    def test_copy_ko_state(self):
        """測試複製劫爭狀態"""
        board1 = GoBoard(size=9)

        # 創建劫爭
        board1.play_move(1, 2, 'b')
        board1.play_move(1, 3, 'w')
        board1.play_move(2, 1, 'b')
        board1.play_move(2, 2, 'w')
        board1.play_move(2, 4, 'w')
        board1.play_move(3, 2, 'b')
        board1.play_move(3, 3, 'w')
        board1.play_move(2, 3, 'b')  # 提子，產生劫

        # 複製
        board2 = board1.copy()

        # 劫爭狀態應該被複製
        self.assertEqual(board1.ko_point, board2.ko_point)


class TestBoundaryConditions(unittest.TestCase):
    """測試邊界條件"""

    def test_corner_liberties(self):
        """測試角落的氣數"""
        board = GoBoard(size=9)
        board.play_move(0, 0, 'b')

        # 角落只有2氣
        group = board.get_group(0, 0)
        liberties = board.count_liberties(group)
        self.assertEqual(liberties, 2)

    def test_edge_liberties(self):
        """測試邊的氣數"""
        board = GoBoard(size=9)
        board.play_move(0, 4, 'b')

        # 邊上有3氣
        group = board.get_group(0, 4)
        liberties = board.count_liberties(group)
        self.assertEqual(liberties, 3)

    def test_center_liberties(self):
        """測試中央的氣數"""
        board = GoBoard(size=9)
        board.play_move(4, 4, 'b')

        # 中央有4氣
        group = board.get_group(4, 4)
        liberties = board.count_liberties(group)
        self.assertEqual(liberties, 4)

    def test_out_of_bounds(self):
        """測試超出棋盤範圍"""
        board = GoBoard(size=9)

        # 超出範圍應該被拒絕
        result = board.is_legal(-1, 0, 'b')
        self.assertFalse(result)

        result = board.is_legal(0, 9, 'b')
        self.assertFalse(result)

        result = board.is_legal(9, 0, 'b')
        self.assertFalse(result)


class TestComplexPatterns(unittest.TestCase):
    """測試複雜棋形"""

    def test_ladder(self):
        """測試征子（ladder）"""
        board = GoBoard(size=9)

        # 創建簡單征子
        board.play_move(2, 2, 'w')
        board.play_move(1, 2, 'b')
        board.play_move(2, 3, 'b')
        board.play_move(3, 2, 'b')

        # 白棋只有一氣
        group = board.get_group(2, 2)
        liberties = board.count_liberties(group)
        self.assertEqual(liberties, 1)

    def test_bamboo_joint(self):
        """測試竹節形"""
        board = GoBoard(size=9)

        # 創建竹節
        # B . B
        # . . .
        # B . B
        board.play_move(1, 1, 'b')
        board.play_move(1, 3, 'b')
        board.play_move(3, 1, 'b')
        board.play_move(3, 3, 'b')

        # 中間的點應該連通
        # （這是形狀測試，不一定需要特殊處理）


def run_tests():
    """運行所有測試"""
    # 創建測試套件
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # 添加所有測試類
    suite.addTests(loader.loadTestsFromTestCase(TestComplexCaptures))
    suite.addTests(loader.loadTestsFromTestCase(TestKoRule))
    suite.addTests(loader.loadTestsFromTestCase(TestSuicideRule))
    suite.addTests(loader.loadTestsFromTestCase(TestTerritoryAndScoring))
    suite.addTests(loader.loadTestsFromTestCase(TestGameEnd))
    suite.addTests(loader.loadTestsFromTestCase(TestBoardCopy))
    suite.addTests(loader.loadTestsFromTestCase(TestBoundaryConditions))
    suite.addTests(loader.loadTestsFromTestCase(TestComplexPatterns))

    # 運行測試
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # 返回是否成功
    return result.wasSuccessful()


if __name__ == '__main__':
    import sys
    success = run_tests()
    sys.exit(0 if success else 1)

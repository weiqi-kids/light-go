"""快速自我對弈測試

使用極少的 MCTS 模擬次數進行快速測試，驗證系統功能。
適合在開發過程中快速驗證代碼正確性。
"""

import sys
from pathlib import Path
import torch
import logging
import time

# 添加專案根目錄到路徑
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from hf_models.modeling_go_ai import GoAIModel
from core.self_play import SelfPlayWorker
from core.game_rules import GoBoard
from core.mcts import MCTS

# 設置日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def quick_test_mcts():
    """快速測試 MCTS 搜索"""
    print("\n" + "="*70)
    print("  測試 1：MCTS 搜索")
    print("="*70)

    # 創建簡單模型
    model = GoAIModel(
        num_input_planes=22,
        num_filters=64,  # 減少 filters 以加快速度
        num_blocks=3,     # 減少 blocks
        board_size=19
    )
    model.eval()

    # 創建測試棋盤
    board = GoBoard(size=19)

    # 創建 MCTS（極少模擬次數）
    mcts = MCTS(
        model=model,
        num_simulations=10,  # 只做 10 次模擬（快速測試）
        temperature=1.0
    )

    print("\n🔍 執行 MCTS 搜索...")
    start_time = time.time()

    try:
        move_probs, root = mcts.search(board, 'b')
        elapsed = time.time() - start_time

        print(f"✅ MCTS 搜索成功")
        print(f"   時間：{elapsed:.2f} 秒")
        print(f"   根節點訪問次數：{root.visit_count}")
        print(f"   子節點數：{len(root.children)}")

        # 顯示 top 5 著手
        top_moves_idx = move_probs.argsort()[-5:][::-1]
        print(f"\n   Top 5 著手：")
        for i, idx in enumerate(top_moves_idx):
            if idx < 19 * 19:
                row = idx // 19
                col = idx % 19
                prob = move_probs[idx]
                print(f"     {i+1}. ({row}, {col}) - {prob*100:.2f}%")

        return True

    except Exception as e:
        print(f"❌ MCTS 搜索失敗：{e}")
        import traceback
        traceback.print_exc()
        return False


def quick_test_self_play():
    """快速測試自我對弈"""
    print("\n" + "="*70)
    print("  測試 2：自我對弈")
    print("="*70)

    # 創建簡單模型
    model = GoAIModel(
        num_input_planes=22,
        num_filters=64,
        num_blocks=3,
        board_size=19
    )
    model.eval()

    # 創建自我對弈工作器（極快速設置）
    worker = SelfPlayWorker(
        model=model,
        num_simulations=5,    # 極少模擬次數
        board_size=19,
        temperature_threshold=3,  # 很快就變貪婪
        max_moves=20          # 限制最大步數
    )

    print("\n🎮 開始快速自我對弈...")
    print("   (MCTS 模擬次數: 5, 最大步數: 20)")
    start_time = time.time()

    try:
        result = worker.play_game(verbose=False)
        elapsed = time.time() - start_time

        print(f"✅ 對弈完成")
        print(f"   時間：{elapsed:.2f} 秒")
        print(f"   勝者：{result.winner or '平局'}")
        print(f"   步數：{result.num_moves}")
        print(f"   訓練樣本：{len(result.examples)}")

        if result.examples:
            example = result.examples[0]
            print(f"\n📊 訓練樣本格式：")
            print(f"   Features: {example.features.shape}")
            print(f"   Policy: {example.policy.shape}")
            print(f"   Value: {example.value:.3f}")

        return True

    except Exception as e:
        print(f"❌ 自我對弈失敗：{e}")
        import traceback
        traceback.print_exc()
        return False


def quick_test_game_rules():
    """快速測試圍棋規則"""
    print("\n" + "="*70)
    print("  測試 3：圍棋規則")
    print("="*70)

    try:
        # 創建棋盤
        board = GoBoard(size=19)
        print("\n✅ 棋盤創建成功")

        # 測試下子
        success = board.play_move(3, 3, 'b')
        print(f"✅ 下子測試：{'成功' if success else '失敗'}")

        # 測試非法著手
        illegal = board.play_move(3, 3, 'w')  # 同一位置
        print(f"✅ 非法著手檢測：{'正確' if not illegal else '錯誤'}")

        # 測試合法著手列表
        legal_moves = board.get_legal_moves('w')
        print(f"✅ 合法著手數：{len(legal_moves)}")

        # 測試 pass
        board.pass_move('w')
        board.pass_move('b')
        print(f"✅ Pass 測試：遊戲{'已' if board.is_game_over() else '未'}結束")

        # 測試棋盤複製
        board2 = board.copy()
        print(f"✅ 棋盤複製：{'成功' if board2 is not board else '失敗'}")

        return True

    except Exception as e:
        print(f"❌ 規則測試失敗：{e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """執行所有快速測試"""
    print("\n" + "="*70)
    print("  Light-Go 快速功能測試")
    print("="*70)
    print("\n這些測試使用極簡配置以快速驗證功能正確性。")
    print("不需要訓練好的模型。\n")

    results = {}

    # 測試 1：圍棋規則
    results['rules'] = quick_test_game_rules()

    # 測試 2：MCTS
    results['mcts'] = quick_test_mcts()

    # 測試 3：自我對弈
    results['self_play'] = quick_test_self_play()

    # 總結
    print("\n" + "="*70)
    print("  測試總結")
    print("="*70)

    all_passed = all(results.values())

    for test_name, passed in results.items():
        status = "✅ 通過" if passed else "❌ 失敗"
        print(f"  {test_name:15s}: {status}")

    print("\n" + "="*70)
    if all_passed:
        print("  🎉 所有測試通過！Phase 2 功能正常運作。")
    else:
        print("  ⚠️  部分測試失敗，需要檢查。")
    print("="*70)

    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

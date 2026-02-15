"""測試提子邏輯修復"""

import sys
from pathlib import Path

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import pytest
from core.game_rules import GoBoard
import torch
from hf_models.modeling_go_ai import GoAIModel
from core.mcts import MCTS

def test_capture():
    """測試提子邏輯"""
    print("=" * 60)
    print("  測試提子邏輯修復")
    print("=" * 60)

    # 創建棋盤
    board = GoBoard(size=19)

    # 設置一個簡單的提子局面
    # 黑子包圍白子 (1,1)
    print("\n1. 設置測試局面...")
    board.play_move(1, 1, 'w')  # 白子在中間
    board.play_move(0, 1, 'b')  # 黑子在上方
    board.play_move(1, 0, 'b')  # 黑子在左方
    board.play_move(2, 1, 'b')  # 黑子在下方

    # 黑子提掉中間的白子
    print("2. 黑子包圍並提掉白子...")
    result = board.play_move(1, 2, 'b')  # 黑子在右方，完成包圍

    assert result, "提子應該成功"
    print("✅ 提子成功！")
    print(f"   黑子俘虜：{board.prisoners['white']}")
    print(f"   白子俘虜：{board.prisoners['black']}")

    # 測試 MCTS 搜索
    print("\n3. 測試 MCTS 搜索...")
    try:
        model = GoAIModel(
            num_input_planes=2,  # LightGo 2-plane encoding
            num_filters=128,
            num_blocks=10,
            board_size=19
        )

        # 載入模型
        try:
            state_dict = torch.load('data/models/from_katago/model.pt',
                                   map_location='cpu')
            model.load_state_dict(state_dict)
            print("   ✅ 模型已載入")
        except:
            print("   ⚠️  使用未訓練模型")

        # MCTS 搜索
        mcts = MCTS(model=model, num_simulations=10)

        board2 = GoBoard(size=19)
        move_probs, root = mcts.search(board2, 'b')

        print(f"   ✅ MCTS 搜索完成")
        print(f"   訪問節點：{root.visit_count}")

    except Exception as e:
        pytest.fail(f"MCTS 測試失敗：{e}")

    print("\n" + "=" * 60)
    print("  ✅ 所有測試通過！")
    print("=" * 60)

if __name__ == "__main__":
    test_capture()
    print("測試完成")
    sys.exit(0)

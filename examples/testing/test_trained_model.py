"""測試訓練好的模型

驗證模型能否正確加載並進行預測
"""

import sys
from pathlib import Path
import torch
import numpy as np

# 添加專案根目錄到路徑
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from hf_models.modeling_go_ai import GoAIModel

def test_model():
    """測試模型加載和預測"""
    print("\n" + "="*70)
    print("  測試訓練好的模型")
    print("="*70)

    # 載入模型（使用訓練時的參數）
    print("\n📥 載入模型...")
    model = GoAIModel(
        num_input_planes=22,
        num_filters=128,
        num_blocks=10,  # 訓練時使用 10 個 blocks
        board_size=19
    )
    state_dict = torch.load('data/models/from_katago/model.pt', map_location='cpu')
    model.load_state_dict(state_dict)
    model.eval()
    print(f"✅ 模型已載入")
    print(f"   參數: {sum(p.numel() for p in model.parameters()):,}")

    # 創建測試輸入（隨機棋盤）
    print("\n🎲 創建測試棋盤...")
    batch_size = 1
    test_input = torch.randn(batch_size, 22, 19, 19)

    # 進行預測
    print("\n🔮 進行預測...")
    with torch.no_grad():
        policy_logits, value = model(test_input)

    # 分析預測結果
    print("\n📊 預測結果：")
    print(f"   Policy logits shape: {policy_logits.shape}")
    print(f"   Policy logits range: [{policy_logits.min():.3f}, {policy_logits.max():.3f}]")

    # 轉換為概率
    policy_probs = torch.softmax(policy_logits, dim=1)
    top_5_moves = torch.topk(policy_probs[0], 5)

    print(f"\n   Top 5 predicted moves:")
    for i, (prob, move_idx) in enumerate(zip(top_5_moves.values, top_5_moves.indices)):
        row = move_idx.item() // 19
        col = move_idx.item() % 19
        if move_idx.item() == 361:
            print(f"     {i+1}. Pass - {prob.item()*100:.2f}%")
        else:
            print(f"     {i+1}. ({row}, {col}) - {prob.item()*100:.2f}%")

    print(f"\n   Value prediction: {value.item():.3f}")
    print(f"   (勝率預測: {(value.item() + 1) / 2 * 100:.1f}%)")

    # 測試批次預測
    print("\n🔄 測試批次預測...")
    batch_input = torch.randn(4, 22, 19, 19)
    with torch.no_grad():
        batch_policy, batch_value = model(batch_input)
    print(f"✅ 批次預測成功")
    print(f"   Batch size: {batch_input.shape[0]}")
    print(f"   Policy shape: {batch_policy.shape}")
    print(f"   Value shape: {batch_value.shape}")

    print("\n" + "="*70)
    print("  ✅ 模型測試完成！")
    print("="*70)

    return True

if __name__ == "__main__":
    test_model()

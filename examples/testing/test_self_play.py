"""測試自我對弈系統

驗證 MCTS 和自我對弈是否能正常工作
"""

import sys
from pathlib import Path
import torch
import logging

# 添加專案根目錄到路徑
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from hf_models.modeling_go_ai import GoAIModel
from core.self_play import SelfPlayWorker, generate_training_data

# 設置日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_self_play():
    """測試自我對弈"""
    print("\n" + "="*70)
    print("  測試自我對弈系統")
    print("="*70)

    # 載入訓練好的模型
    print("\n📥 載入模型...")
    model = GoAIModel(
        num_input_planes=22,
        num_filters=128,
        num_blocks=10,
        board_size=19
    )

    try:
        state_dict = torch.load('data/models/from_katago/model.pt', map_location='cpu')
        model.load_state_dict(state_dict)
        model.eval()
        print("✅ 模型已載入")
    except FileNotFoundError:
        print("⚠️  找不到訓練好的模型，使用未訓練的模型測試")

    # 創建自我對弈工作器
    print("\n🎮 創建自我對弈工作器...")
    worker = SelfPlayWorker(
        model=model,
        num_simulations=100,  # 減少模擬次數以加快測試
        board_size=19,
        temperature_threshold=10,
        max_moves=100  # 限制最大步數
    )
    print("✅ 工作器已創建")
    print(f"   MCTS 模擬次數: 100")
    print(f"   最大步數: 100")

    # 進行一局測試對弈
    print("\n🔄 開始自我對弈...")
    print("   (這可能需要幾分鐘...)")

    result = worker.play_game(verbose=True)

    # 顯示結果
    print("\n" + "="*70)
    print("  對弈結果")
    print("="*70)
    print(f"   勝者: {result.winner or '平局'}")
    print(f"   步數: {result.num_moves}")
    print(f"   生成訓練樣本: {len(result.examples)}")

    # 檢查訓練樣本
    if result.examples:
        example = result.examples[0]
        print(f"\n📊 訓練樣本格式：")
        print(f"   Features shape: {example.features.shape}")
        print(f"   Policy shape: {example.policy.shape}")
        print(f"   Value: {example.value:.3f}")

    print("\n" + "="*70)
    print("  ✅ 測試完成！")
    print("="*70)

    return True


if __name__ == "__main__":
    test_self_play()

"""快速性能基準測試"""

import sys
from pathlib import Path
import time
import torch
import numpy as np

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from hf_models.modeling_go_ai import GoAIModel
from core.game_rules import GoBoard
from core.mcts import MCTS
from core.mcts_optimized import MCTSOptimized

def print_header(title: str):
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)

def benchmark_inference(model):
    """測試推理速度"""
    print_header("推理速度測試")

    device = next(model.parameters()).device
    model.eval()

    for batch_size in [1, 4, 8]:
        test_input = torch.randn(batch_size, 22, 19, 19).to(device)

        # 預熱
        with torch.no_grad():
            for _ in range(5):
                model(test_input)

        # 測量
        times = []
        with torch.no_grad():
            for _ in range(50):
                start = time.time()
                model(test_input)
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                times.append(time.time() - start)

        avg_time = np.mean(times) * 1000  # ms
        throughput = batch_size / (avg_time / 1000)

        print(f"\nBatch {batch_size:2d}: {avg_time:.2f} ms/batch, "
              f"{throughput:.1f} samples/sec")

def benchmark_mcts_simple(model):
    """簡化的 MCTS 測試"""
    print_header("MCTS 搜索速度測試")

    board = GoBoard(size=19)
    sims = 50

    # 原始 MCTS
    print(f"\n原始 MCTS ({sims} 次模擬):")
    mcts = MCTS(model=model, num_simulations=sims)

    start = time.time()
    move_probs, root = mcts.search(board, 'b')
    elapsed_original = time.time() - start

    print(f"  時間：{elapsed_original:.2f} 秒")
    print(f"  訪問：{root.visit_count} 節點")

    # 優化 MCTS
    print(f"\n優化 MCTS ({sims} 次模擬，批次=4):")
    mcts_opt = MCTSOptimized(
        model=model,
        num_simulations=sims,
        batch_size=4
    )

    start = time.time()
    move_probs_opt, root_opt = mcts_opt.search(board, 'b')
    elapsed_optimized = time.time() - start

    print(f"  時間：{elapsed_optimized:.2f} 秒")
    print(f"  訪問：{root_opt.visit_count} 節點")

    # 比較
    speedup = elapsed_original / elapsed_optimized
    print(f"\n⚡ 加速比：{speedup:.2f}x")
    print(f"  時間節省：{(1 - 1/speedup) * 100:.1f}%")

def main():
    print_header("Light-Go 快速性能測試")

    # 載入模型
    print("\n載入模型...")
    model = GoAIModel(
        num_input_planes=22,
        num_filters=128,
        num_blocks=10,
        board_size=19
    )

    try:
        state_dict = torch.load('data/models/from_katago/model.pt',
                               map_location='cpu')
        model.load_state_dict(state_dict)
        print("✅ 模型已載入")
    except Exception as e:
        print(f"⚠️  使用未訓練模型：{e}")

    model.eval()

    # 1. 推理速度測試
    benchmark_inference(model)

    # 2. MCTS 測試
    benchmark_mcts_simple(model)

    print("\n" + "=" * 60)
    print("  ✅ 測試完成！")
    print("=" * 60)

if __name__ == "__main__":
    main()

"""性能基準測試

比較優化前後的性能差異：
1. MCTS 搜索速度
2. 訓練吞吐量
3. 內存使用
4. 自我對弈速度
"""

import sys
from pathlib import Path
import time
import argparse
import torch
import numpy as np
import psutil
import os

# 添加專案根目錄到路徑
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from hf_models.modeling_go_ai import GoAIModel
from core.game_rules import GoBoard
from core.mcts import MCTS
from core.mcts_optimized import MCTSOptimized


def print_header(title: str):
    """打印標題"""
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70)


def get_memory_usage():
    """獲取當前內存使用（MB）"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024


def benchmark_mcts(model, num_simulations: int = 100):
    """基準測試 MCTS"""
    print_header("基準測試：MCTS 搜索速度")

    board = GoBoard(size=19)

    # 測試原始版本
    print("\n📊 測試原始 MCTS...")
    mcts_original = MCTS(model=model, num_simulations=num_simulations)

    mem_before = get_memory_usage()
    start = time.time()

    move_probs, root = mcts_original.search(board, 'b')

    elapsed_original = time.time() - start
    mem_after = get_memory_usage()
    mem_used_original = mem_after - mem_before

    print(f"   時間：{elapsed_original:.2f} 秒")
    print(f"   訪問節點：{root.visit_count}")
    print(f"   內存增加：{mem_used_original:.1f} MB")

    # 測試優化版本
    print("\n📊 測試優化 MCTS...")
    mcts_optimized = MCTSOptimized(
        model=model,
        num_simulations=num_simulations,
        batch_size=8
    )

    mem_before = get_memory_usage()
    start = time.time()

    move_probs_opt, root_opt = mcts_optimized.search(board, 'b')

    elapsed_optimized = time.time() - start
    mem_after = get_memory_usage()
    mem_used_optimized = mem_after - mem_before

    print(f"   時間：{elapsed_optimized:.2f} 秒")
    print(f"   訪問節點：{root_opt.visit_count}")
    print(f"   內存增加：{mem_used_optimized:.1f} MB")

    # 比較
    print(f"\n⚡ 性能提升：")
    speedup = elapsed_original / elapsed_optimized
    print(f"   速度提升：{speedup:.2f}x")
    print(f"   時間節省：{(1 - 1/speedup) * 100:.1f}%")

    mem_saving = (mem_used_original - mem_used_optimized) / mem_used_original * 100
    if mem_saving > 0:
        print(f"   內存節省：{mem_saving:.1f}%")
    else:
        print(f"   內存增加：{-mem_saving:.1f}%")

    return {
        'original_time': elapsed_original,
        'optimized_time': elapsed_optimized,
        'speedup': speedup,
        'original_memory': mem_used_original,
        'optimized_memory': mem_used_optimized
    }


def benchmark_inference(model, batch_sizes: list = [1, 4, 8, 16, 32]):
    """基準測試推理速度"""
    print_header("基準測試：推理速度（不同批次大小）")

    device = next(model.parameters()).device
    model.eval()

    results = {}

    for batch_size in batch_sizes:
        # 準備輸入
        test_input = torch.randn(batch_size, 22, 19, 19).to(device)

        # 預熱
        with torch.no_grad():
            for _ in range(10):
                model(test_input)

        # 測量
        times = []
        with torch.no_grad():
            for _ in range(100):
                start = time.time()
                model(test_input)
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                times.append(time.time() - start)

        avg_time = np.mean(times) * 1000  # ms
        throughput = batch_size / (avg_time / 1000)

        results[batch_size] = {
            'time_ms': avg_time,
            'throughput': throughput
        }

        print(f"\n   Batch {batch_size:2d}: "
              f"{avg_time:.2f} ms/batch, "
              f"{throughput:.1f} samples/sec")

    # 找出最優批次大小
    best_batch = max(results.items(), key=lambda x: x[1]['throughput'])
    print(f"\n✅ 最優批次大小：{best_batch[0]} "
          f"({best_batch[1]['throughput']:.1f} samples/sec)")

    return results


def benchmark_self_play_game(model, mcts_sims: int = 50):
    """基準測試單局自我對弈"""
    print_header(f"基準測試：自我對弈（{mcts_sims} 次模擬）")

    # 原始 MCTS
    print("\n📊 使用原始 MCTS...")
    mcts = MCTS(model=model, num_simulations=mcts_sims)

    board = GoBoard(size=19)
    move_count = 0
    max_moves = 50

    start = time.time()

    while not board.is_game_over() and move_count < max_moves:
        color = 'b' if move_count % 2 == 0 else 'w'
        move = mcts.select_move(board, color, temperature=0.5)
        board.play_move(move[0], move[1], color)
        move_count += 1

    elapsed_original = time.time() - start

    print(f"   {move_count} 步")
    print(f"   總時間：{elapsed_original:.1f} 秒")
    print(f"   平均每步：{elapsed_original/move_count:.2f} 秒")

    # 優化 MCTS
    print("\n📊 使用優化 MCTS...")
    mcts_opt = MCTSOptimized(
        model=model,
        num_simulations=mcts_sims,
        batch_size=4
    )

    board = GoBoard(size=19)
    move_count = 0

    start = time.time()

    while not board.is_game_over() and move_count < max_moves:
        color = 'b' if move_count % 2 == 0 else 'w'
        move = mcts_opt.select_move(board, color, temperature=0.5)
        board.play_move(move[0], move[1], color)
        move_count += 1

    elapsed_optimized = time.time() - start

    print(f"   {move_count} 步")
    print(f"   總時間：{elapsed_optimized:.1f} 秒")
    print(f"   平均每步：{elapsed_optimized/move_count:.2f} 秒")

    # 比較
    speedup = elapsed_original / elapsed_optimized
    print(f"\n⚡ 自我對弈加速：{speedup:.2f}x")

    return {
        'original_time': elapsed_original,
        'optimized_time': elapsed_optimized,
        'speedup': speedup
    }


def main():
    """主函數"""
    parser = argparse.ArgumentParser(description='性能基準測試')

    parser.add_argument('--model-path', type=str,
                       default='data/models/from_katago/model.pt')
    parser.add_argument('--num-blocks', type=int, default=10)
    parser.add_argument('--num-filters', type=int, default=128)
    parser.add_argument('--mcts-sims', type=int, default=100,
                       help='MCTS 模擬次數')

    args = parser.parse_args()

    print_header("Light-Go 性能基準測試")

    # 載入模型
    print(f"\n📥 載入模型：{args.model_path}")

    model = GoAIModel(
        num_input_planes=22,
        num_filters=args.num_filters,
        num_blocks=args.num_blocks,
        board_size=19
    )

    try:
        state_dict = torch.load(args.model_path, map_location='cpu')
        model.load_state_dict(state_dict)
        model.eval()
        print(f"✅ 模型已載入")
    except Exception as e:
        print(f"⚠️  載入失敗，使用未訓練模型：{e}")

    # 執行基準測試
    results = {}

    # 1. MCTS 搜索
    results['mcts'] = benchmark_mcts(model, num_simulations=args.mcts_sims)

    # 2. 推理速度
    results['inference'] = benchmark_inference(model)

    # 3. 自我對弈
    results['self_play'] = benchmark_self_play_game(model, mcts_sims=50)

    # 總結
    print_header("性能總結")

    print(f"\n🎯 MCTS 搜索：")
    print(f"   加速：{results['mcts']['speedup']:.2f}x")
    print(f"   原始：{results['mcts']['original_time']:.2f}s")
    print(f"   優化：{results['mcts']['optimized_time']:.2f}s")

    print(f"\n🎯 自我對弈：")
    print(f"   加速：{results['self_play']['speedup']:.2f}x")
    print(f"   原始：{results['self_play']['original_time']:.1f}s")
    print(f"   優化：{results['self_play']['optimized_time']:.1f}s")

    print(f"\n🎯 推理優化建議：")
    best_batch = max(results['inference'].items(),
                    key=lambda x: x[1]['throughput'])
    print(f"   建議批次大小：{best_batch[0]}")
    print(f"   最大吞吐量：{best_batch[1]['throughput']:.1f} samples/sec")

    print("\n" + "="*70)


if __name__ == "__main__":
    main()

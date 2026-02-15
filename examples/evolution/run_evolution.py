"""運行架構演化

示範如何使用演化流水線進行神經架構搜索
"""

import sys
from pathlib import Path
import argparse
import logging

# 添加專案根目錄到路徑
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.evolution_pipeline import EvolutionPipeline

# 設置日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """主函數"""
    parser = argparse.ArgumentParser(
        description='運行架構演化',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        '--population-size',
        type=int,
        default=5,
        help='種群大小（默認：5）'
    )
    parser.add_argument(
        '--generations',
        type=int,
        default=3,
        help='演化世代數（默認：3）'
    )
    parser.add_argument(
        '--games-per-gen',
        type=int,
        default=10,
        help='每世代自我對弈局數（默認：10）'
    )
    parser.add_argument(
        '--num-simulations',
        type=int,
        default=100,
        help='MCTS 模擬次數（默認：100）'
    )
    parser.add_argument(
        '--save-dir',
        type=str,
        default='data/evolution',
        help='保存目錄（默認：data/evolution）'
    )

    args = parser.parse_args()

    print("\n" + "="*70)
    print("  Light-Go 架構演化")
    print("="*70)
    print(f"\n配置：")
    print(f"  種群大小：{args.population_size}")
    print(f"  演化世代：{args.generations}")
    print(f"  每世代對弈：{args.games_per_gen} 局")
    print(f"  MCTS 模擬：{args.num_simulations}")
    print(f"  保存目錄：{args.save_dir}")
    print("\n⚠️  注意：完整演化需要大量計算資源和時間")
    print("    建議先使用小參數測試（如上述默認值）\n")

    # 創建演化流水線
    pipeline = EvolutionPipeline(
        population_size=args.population_size,
        num_simulations=args.num_simulations,
        save_dir=args.save_dir
    )

    # 初始化
    pipeline.initialize()

    # 運行演化
    pipeline.run_evolution_cycle(
        num_generations=args.generations,
        games_per_generation=args.games_per_gen,
        device='cpu'  # 可以改為 'cuda'
    )

    print("\n" + "="*70)
    print("  演化完成！")
    print("="*70)
    print(f"\n結果已保存至：{args.save_dir}")


if __name__ == "__main__":
    main()

"""MCTS 質量分析工具

分析 MCTS 搜索樹的質量，包括：
- 搜索樹訪問分佈
- Value 估計準確性
- Policy 引導效果
- 常見開局覆蓋率
- 搜索深度統計

使用方法：
    # 基本分析
    python examples/test_mcts_quality.py \
        --model-path data/models/from_katago/model.pt

    # 詳細分析（更多模擬次數）
    python examples/test_mcts_quality.py \
        --model-path data/models/from_katago/model.pt \
        --mcts-simulations 800 \
        --num-positions 20

    # 使用未訓練模型測試
    python examples/test_mcts_quality.py --no-model
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any
import time

import torch
import numpy as np
import matplotlib.pyplot as plt

# 添加專案根目錄到路徑
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from hf_models.modeling_go_ai import GoAIModel
from core.architecture_genome import ArchitectureGenome
from core.game_rules import GoBoard
from core.mcts import MCTS
from core.mcts_optimized import MCTSOptimized

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def print_header(title: str):
    """打印格式化標題"""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def analyze_search_tree(root, depth=0, max_depth=5) -> Dict[str, Any]:
    """分析搜索樹結構

    Parameters
    ----------
    root : MCTSNode
        根節點
    depth : int
        當前深度
    max_depth : int
        最大分析深度

    Returns
    -------
    Dict[str, Any]
        分析結果
    """
    stats = {
        'total_nodes': 0,
        'total_visits': 0,
        'max_depth': 0,
        'avg_branching_factor': 0,
        'visit_distribution': [],
        'depth_distribution': {}
    }

    def traverse(node, current_depth):
        if current_depth > max_depth:
            return

        stats['total_nodes'] += 1
        stats['total_visits'] += node.visit_count
        stats['max_depth'] = max(stats['max_depth'], current_depth)

        if current_depth not in stats['depth_distribution']:
            stats['depth_distribution'][current_depth] = 0
        stats['depth_distribution'][current_depth] += 1

        if node.children:
            visits = [child.visit_count for child in node.children.values()]
            stats['visit_distribution'].extend(visits)

            for child in node.children.values():
                traverse(child, current_depth + 1)

    traverse(root, depth)

    # 計算平均分支因子
    if stats['total_nodes'] > 1:
        stats['avg_branching_factor'] = len(stats['visit_distribution']) / (stats['total_nodes'] - 1)

    return stats


def test_search_quality(
    model: GoAIModel,
    num_positions: int = 10,
    mcts_simulations: int = 400,
    use_optimized: bool = True
) -> Dict[str, Any]:
    """測試 MCTS 搜索質量

    Parameters
    ----------
    model : GoAIModel
        被測試的模型
    num_positions : int
        測試局面數量
    mcts_simulations : int
        每個局面的模擬次數
    use_optimized : bool
        是否使用優化版 MCTS

    Returns
    -------
    Dict[str, Any]
        測試結果
    """
    print_header("MCTS 搜索質量測試")

    logger.info(f"測試局面數：{num_positions}")
    logger.info(f"MCTS 模擬次數：{mcts_simulations}")

    # 創建 MCTS
    if use_optimized:
        mcts = MCTSOptimized(model=model, num_simulations=mcts_simulations, batch_size=4)
        logger.info("使用優化版 MCTS")
    else:
        mcts = MCTS(model=model, num_simulations=mcts_simulations)
        logger.info("使用標準 MCTS")

    results = {
        'search_times': [],
        'tree_stats': [],
        'top_moves_concentration': [],
        'value_estimates': [],
        'policy_entropy': []
    }

    for pos_idx in range(num_positions):
        board = GoBoard(size=19)
        current_player = 'b'

        # 隨機下幾手（0-20手）
        num_random_moves = np.random.randint(0, 21)
        for _ in range(num_random_moves):
            legal_moves = []
            for row in range(19):
                for col in range(19):
                    if board.is_legal_move(row, col, current_player):
                        legal_moves.append((row, col))

            if legal_moves:
                row, col = legal_moves[np.random.randint(len(legal_moves))]
                board.play_move(row, col, current_player)
                current_player = 'w' if current_player == 'b' else 'b'

        logger.info(f"\n測試局面 {pos_idx + 1}/{num_positions}（已下 {num_random_moves} 手）")

        # 執行 MCTS 搜索
        start_time = time.time()
        move_probs, root = mcts.search(board, current_player)
        search_time = time.time() - start_time

        results['search_times'].append(search_time)

        # 分析搜索樹
        tree_stats = analyze_search_tree(root)
        results['tree_stats'].append(tree_stats)

        logger.info(f"  搜索時間：{search_time:.2f} 秒")
        logger.info(f"  總節點數：{tree_stats['total_nodes']}")
        logger.info(f"  最大深度：{tree_stats['max_depth']}")
        logger.info(f"  平均分支因子：{tree_stats['avg_branching_factor']:.2f}")

        # 分析著法分佈
        sorted_moves = sorted(move_probs.items(), key=lambda x: x[1], reverse=True)
        top_5_probs = [prob for _, prob in sorted_moves[:5]]
        top_5_concentration = sum(top_5_probs)

        results['top_moves_concentration'].append(top_5_concentration)

        logger.info(f"  Top 5 著法集中度：{top_5_concentration*100:.1f}%")

        # Value 估計
        value_estimate = root.value_sum / root.visit_count if root.visit_count > 0 else 0
        results['value_estimates'].append(value_estimate)

        logger.info(f"  Value 估計：{value_estimate:.3f}")

        # Policy 熵（衡量策略的不確定性）
        probs = np.array([p for _, p in move_probs.items()])
        probs = probs[probs > 0]  # 移除零概率
        if len(probs) > 0:
            entropy = -np.sum(probs * np.log(probs + 1e-8))
        else:
            entropy = 0

        results['policy_entropy'].append(entropy)

        logger.info(f"  Policy 熵：{entropy:.3f}")

    return results


def test_opening_coverage(
    model: GoAIModel,
    mcts_simulations: int = 400,
    use_optimized: bool = True
) -> Dict[str, Any]:
    """測試常見開局覆蓋率

    Parameters
    ----------
    model : GoAIModel
        被測試的模型
    mcts_simulations : int
        MCTS 模擬次數
    use_optimized : bool
        是否使用優化版 MCTS

    Returns
    -------
    Dict[str, Any]
        測試結果
    """
    print_header("常見開局覆蓋率測試")

    # 常見開局位置（19x19 棋盤）
    common_openings = {
        '小目': [(3, 3), (15, 15), (3, 15), (15, 3)],
        '星位': [(3, 9), (15, 9), (9, 3), (9, 15)],
        '高目': [(3, 4), (15, 14)],
        '天元': [(9, 9)]
    }

    # 創建 MCTS
    if use_optimized:
        mcts = MCTSOptimized(model=model, num_simulations=mcts_simulations, batch_size=4)
    else:
        mcts = MCTS(model=model, num_simulations=mcts_simulations)

    results = {}

    board = GoBoard(size=19)
    move_probs, root = mcts.search(board, 'b')

    # 檢查各類開局的概率
    for opening_name, positions in common_openings.items():
        opening_probs = []
        for pos in positions:
            if pos in move_probs:
                opening_probs.append(move_probs[pos])
            else:
                opening_probs.append(0.0)

        total_prob = sum(opening_probs)
        results[opening_name] = {
            'total_probability': total_prob,
            'positions': positions,
            'individual_probs': opening_probs
        }

        logger.info(f"\n{opening_name}：")
        logger.info(f"  總概率：{total_prob*100:.2f}%")
        for pos, prob in zip(positions, opening_probs):
            logger.info(f"  {pos}: {prob*100:.2f}%")

    return results


def generate_report(
    search_quality_results: Dict[str, Any],
    opening_coverage_results: Dict[str, Any],
    output_path: str = None
) -> str:
    """生成分析報告

    Parameters
    ----------
    search_quality_results : Dict[str, Any]
        搜索質量結果
    opening_coverage_results : Dict[str, Any]
        開局覆蓋率結果
    output_path : str, optional
        報告保存路徑

    Returns
    -------
    str
        報告內容
    """
    print_header("MCTS 質量分析報告")

    # 計算統計量
    avg_search_time = np.mean(search_quality_results['search_times'])
    avg_nodes = np.mean([s['total_nodes'] for s in search_quality_results['tree_stats']])
    avg_depth = np.mean([s['max_depth'] for s in search_quality_results['tree_stats']])
    avg_branching = np.mean([s['avg_branching_factor'] for s in search_quality_results['tree_stats']])
    avg_concentration = np.mean(search_quality_results['top_moves_concentration'])
    avg_entropy = np.mean(search_quality_results['policy_entropy'])

    report = f"""
# MCTS 質量分析報告

**生成時間**：{time.strftime('%Y-%m-%d %H:%M:%S')}

---

## 1. 搜索效率

- **平均搜索時間**：{avg_search_time:.2f} 秒
- **平均節點數**：{avg_nodes:.0f}
- **平均最大深度**：{avg_depth:.1f}
- **平均分支因子**：{avg_branching:.2f}

**評估**：{'✅ 優秀' if avg_search_time < 10 else '⚠️ 需優化'}

---

## 2. 著法質量

- **Top 5 集中度**：{avg_concentration*100:.1f}%
  - 理想範圍：60-80%
  - {'✅ 良好' if 0.6 <= avg_concentration <= 0.8 else '⚠️ 可能過於集中或分散'}

- **Policy 熵**：{avg_entropy:.3f}
  - 較低熵表示策略確定性高
  - 較高熵表示策略探索性強

---

## 3. 開局覆蓋率

"""

    for opening_name, data in opening_coverage_results.items():
        prob = data['total_probability']
        report += f"- **{opening_name}**：{prob*100:.2f}%\n"

    report += f"""
---

## 4. 詳細統計

### 搜索時間分佈
- 最小：{min(search_quality_results['search_times']):.2f} 秒
- 最大：{max(search_quality_results['search_times']):.2f} 秒
- 標準差：{np.std(search_quality_results['search_times']):.2f} 秒

### Value 估計範圍
- 最小：{min(search_quality_results['value_estimates']):.3f}
- 最大：{max(search_quality_results['value_estimates']):.3f}
- 平均：{np.mean(search_quality_results['value_estimates']):.3f}

---

## 5. 總結

"""

    # 總體評估
    issues = []
    if avg_search_time > 10:
        issues.append("搜索時間較長，考慮減少模擬次數或優化")
    if avg_concentration < 0.6:
        issues.append("著法過於分散，策略可能不夠明確")
    if avg_concentration > 0.8:
        issues.append("著法過於集中，可能缺乏探索")

    if not issues:
        report += "✅ **MCTS 質量良好**：所有指標都在正常範圍內\n"
    else:
        report += "⚠️ **發現以下問題**：\n"
        for issue in issues:
            report += f"- {issue}\n"

    report += f"\n**生成時間**：{time.strftime('%Y-%m-%d %H:%M:%S')}\n"

    print(report)

    # 保存報告
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report)
        logger.info(f"✅ 報告已保存：{output_path}")

    return report


def load_model(
    model_path: str = None,
    num_blocks: int = 10,
    num_filters: int = 128,
    device: str = 'cpu'
) -> GoAIModel:
    """載入模型"""
    genome = ArchitectureGenome(
        num_blocks=num_blocks,
        base_filters=num_filters,
        filters_per_block=[num_filters] * num_blocks,
        kernel_sizes=[3] * num_blocks,
        block_types=['residual'] * num_blocks,
        use_se_blocks=False,
        dropout_rate=0.0,
        num_input_planes=22,
        board_size=19
    )

    model = genome.to_pytorch_model(device=device)

    if model_path and os.path.exists(model_path):
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        logger.info(f"✅ 模型已載入：{model_path}")
    else:
        if model_path:
            logger.warning(f"⚠️  模型文件不存在：{model_path}")
        logger.info("ℹ️  使用未訓練模型")

    model.eval()
    return model


def main():
    """主函數"""
    parser = argparse.ArgumentParser(
        description='MCTS 質量分析工具',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        '--model-path',
        type=str,
        default='data/models/from_katago/model.pt',
        help='模型文件路徑'
    )
    parser.add_argument(
        '--no-model',
        action='store_true',
        help='使用未訓練模型'
    )
    parser.add_argument(
        '--num-blocks',
        type=int,
        default=10,
        help='殘差塊數量（默認：10）'
    )
    parser.add_argument(
        '--num-filters',
        type=int,
        default=128,
        help='濾波器數量（默認：128）'
    )
    parser.add_argument(
        '--num-positions',
        type=int,
        default=10,
        help='測試局面數量（默認：10）'
    )
    parser.add_argument(
        '--mcts-simulations',
        type=int,
        default=400,
        help='MCTS 模擬次數（默認：400）'
    )
    parser.add_argument(
        '--use-standard-mcts',
        action='store_true',
        help='使用標準 MCTS（默認：優化版）'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='data/mcts_quality_reports/report.md',
        help='報告保存路徑'
    )

    args = parser.parse_args()

    print_header("MCTS 質量分析")

    # 載入模型
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"使用裝置：{device}")

    model_path = None if args.no_model else args.model_path
    model = load_model(
        model_path=model_path,
        num_blocks=args.num_blocks,
        num_filters=args.num_filters,
        device=device
    )

    # 1. 搜索質量測試
    search_quality_results = test_search_quality(
        model=model,
        num_positions=args.num_positions,
        mcts_simulations=args.mcts_simulations,
        use_optimized=not args.use_standard_mcts
    )

    # 2. 開局覆蓋率測試
    opening_coverage_results = test_opening_coverage(
        model=model,
        mcts_simulations=args.mcts_simulations,
        use_optimized=not args.use_standard_mcts
    )

    # 3. 生成報告
    report = generate_report(
        search_quality_results=search_quality_results,
        opening_coverage_results=opening_coverage_results,
        output_path=args.output
    )

    print_header("完成！")


if __name__ == "__main__":
    main()

"""演化結果分析工具

分析演化系統的結果，包括：
- ELO 評級追蹤與可視化
- 架構多樣性分析
- 性能提升趨勢
- 最佳策略識別
- 收斂性分析

使用方法：
    # 分析演化結果
    python examples/analyze_evolution.py \
        --evolution-dir data/evolution_results/run_001

    # 生成可視化圖表
    python examples/analyze_evolution.py \
        --evolution-dir data/evolution_results/run_001 \
        --generate-plots

    # 比較多次運行
    python examples/analyze_evolution.py \
        --evolution-dirs data/evolution_results/run_001 data/evolution_results/run_002
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
import time

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 使用非交互式後端

# 添加專案根目錄到路徑
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

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


def load_evolution_data(evolution_dir: str) -> Dict[str, Any]:
    """載入演化數據

    Parameters
    ----------
    evolution_dir : str
        演化結果目錄

    Returns
    -------
    Dict[str, Any]
        演化數據
    """
    logger.info(f"載入演化數據：{evolution_dir}")

    data = {
        'generations': [],
        'population_history': [],
        'best_elo_history': [],
        'avg_elo_history': [],
        'diversity_history': [],
        'config': None
    }

    # 載入配置
    config_path = os.path.join(evolution_dir, 'config.json')
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            data['config'] = json.load(f)

    # 載入每代的結果
    generation_files = sorted(Path(evolution_dir).glob('generation_*.json'))

    for gen_file in generation_files:
        with open(gen_file, 'r') as f:
            gen_data = json.load(f)
            data['generations'].append(gen_data)

        # 提取關鍵指標
        generation_num = gen_data.get('generation', 0)
        population = gen_data.get('population', [])

        if population:
            elos = [ind.get('elo', 1000) for ind in population]
            data['best_elo_history'].append(max(elos))
            data['avg_elo_history'].append(np.mean(elos))

            # 計算多樣性（架構參數的標準差）
            param_counts = [ind.get('param_count', 0) for ind in population]
            diversity = np.std(param_counts) if len(param_counts) > 1 else 0
            data['diversity_history'].append(diversity)

        data['population_history'].append(population)

    logger.info(f"✅ 載入了 {len(data['generations'])} 代數據")

    return data


def analyze_elo_progression(data: Dict[str, Any]) -> Dict[str, Any]:
    """分析 ELO 評級進展

    Parameters
    ----------
    data : Dict[str, Any]
        演化數據

    Returns
    -------
    Dict[str, Any]
        分析結果
    """
    print_header("ELO 評級進展分析")

    if not data['best_elo_history']:
        logger.warning("無 ELO 數據")
        return {}

    best_elos = np.array(data['best_elo_history'])
    avg_elos = np.array(data['avg_elo_history'])

    results = {
        'initial_best_elo': best_elos[0] if len(best_elos) > 0 else 0,
        'final_best_elo': best_elos[-1] if len(best_elos) > 0 else 0,
        'elo_improvement': best_elos[-1] - best_elos[0] if len(best_elos) > 0 else 0,
        'max_elo': np.max(best_elos) if len(best_elos) > 0 else 0,
        'avg_elo_final': avg_elos[-1] if len(avg_elos) > 0 else 0,
        'convergence_generation': None
    }

    # 檢測收斂點（ELO 不再顯著提升）
    if len(best_elos) > 10:
        improvement_threshold = 10  # ELO 改善閾值
        for i in range(10, len(best_elos)):
            recent_improvement = best_elos[i] - best_elos[i-10]
            if recent_improvement < improvement_threshold:
                results['convergence_generation'] = i
                break

    logger.info(f"初始最佳 ELO：{results['initial_best_elo']:.1f}")
    logger.info(f"最終最佳 ELO：{results['final_best_elo']:.1f}")
    logger.info(f"ELO 提升：{results['elo_improvement']:.1f}")
    logger.info(f"歷史最高 ELO：{results['max_elo']:.1f}")

    if results['convergence_generation']:
        logger.info(f"收斂代數：第 {results['convergence_generation']} 代")
    else:
        logger.info("尚未收斂")

    return results


def analyze_diversity(data: Dict[str, Any]) -> Dict[str, Any]:
    """分析架構多樣性

    Parameters
    ----------
    data : Dict[str, Any]
        演化數據

    Returns
    -------
    Dict[str, Any]
        分析結果
    """
    print_header("架構多樣性分析")

    if not data['diversity_history']:
        logger.warning("無多樣性數據")
        return {}

    diversity = np.array(data['diversity_history'])

    results = {
        'initial_diversity': diversity[0] if len(diversity) > 0 else 0,
        'final_diversity': diversity[-1] if len(diversity) > 0 else 0,
        'avg_diversity': np.mean(diversity) if len(diversity) > 0 else 0,
        'min_diversity': np.min(diversity) if len(diversity) > 0 else 0,
        'max_diversity': np.max(diversity) if len(diversity) > 0 else 0
    }

    logger.info(f"初始多樣性：{results['initial_diversity']:.1f}")
    logger.info(f"最終多樣性：{results['final_diversity']:.1f}")
    logger.info(f"平均多樣性：{results['avg_diversity']:.1f}")
    logger.info(f"多樣性範圍：{results['min_diversity']:.1f} - {results['max_diversity']:.1f}")

    # 評估多樣性維護
    if results['final_diversity'] > results['avg_diversity'] * 0.5:
        logger.info("✅ 多樣性維護良好")
    else:
        logger.warning("⚠️  多樣性下降過快，種群可能過早收斂")

    return results


def analyze_best_architectures(data: Dict[str, Any], top_k: int = 5) -> List[Dict[str, Any]]:
    """分析最佳架構

    Parameters
    ----------
    data : Dict[str, Any]
        演化數據
    top_k : int
        返回前 K 個最佳架構

    Returns
    -------
    List[Dict[str, Any]]
        最佳架構列表
    """
    print_header(f"Top {top_k} 最佳架構")

    all_individuals = []
    for gen_idx, population in enumerate(data['population_history']):
        for ind in population:
            ind['generation'] = gen_idx
            all_individuals.append(ind)

    # 按 ELO 排序
    all_individuals.sort(key=lambda x: x.get('elo', 0), reverse=True)
    top_architectures = all_individuals[:top_k]

    for i, arch in enumerate(top_architectures):
        logger.info(f"\n第 {i+1} 名：")
        logger.info(f"  ELO：{arch.get('elo', 0):.1f}")
        logger.info(f"  世代：{arch.get('generation', 0)}")
        logger.info(f"  參數量：{arch.get('param_count', 0):,}")
        logger.info(f"  架構：{arch.get('genome', {})}")

    return top_architectures


def generate_plots(
    data: Dict[str, Any],
    elo_results: Dict[str, Any],
    diversity_results: Dict[str, Any],
    output_dir: str
) -> List[str]:
    """生成可視化圖表

    Parameters
    ----------
    data : Dict[str, Any]
        演化數據
    elo_results : Dict[str, Any]
        ELO 分析結果
    diversity_results : Dict[str, Any]
        多樣性分析結果
    output_dir : str
        輸出目錄

    Returns
    -------
    List[str]
        生成的圖表文件路徑
    """
    print_header("生成可視化圖表")

    os.makedirs(output_dir, exist_ok=True)
    plot_files = []

    generations = list(range(len(data['best_elo_history'])))

    # 1. ELO 進展圖
    plt.figure(figsize=(12, 6))
    plt.plot(generations, data['best_elo_history'], 'b-', linewidth=2, label='最佳 ELO')
    plt.plot(generations, data['avg_elo_history'], 'g--', linewidth=2, label='平均 ELO')
    plt.xlabel('代數', fontsize=12)
    plt.ylabel('ELO 評級', fontsize=12)
    plt.title('ELO 評級進展', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)

    elo_plot_path = os.path.join(output_dir, 'elo_progression.png')
    plt.savefig(elo_plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    plot_files.append(elo_plot_path)
    logger.info(f"✅ ELO 進展圖：{elo_plot_path}")

    # 2. 多樣性進展圖
    if data['diversity_history']:
        plt.figure(figsize=(12, 6))
        plt.plot(generations, data['diversity_history'], 'r-', linewidth=2)
        plt.xlabel('代數', fontsize=12)
        plt.ylabel('架構多樣性（參數量標準差）', fontsize=12)
        plt.title('架構多樣性進展', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)

        diversity_plot_path = os.path.join(output_dir, 'diversity_progression.png')
        plt.savefig(diversity_plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        plot_files.append(diversity_plot_path)
        logger.info(f"✅ 多樣性進展圖：{diversity_plot_path}")

    # 3. ELO vs 參數量散點圖（最後一代）
    if data['population_history']:
        last_population = data['population_history'][-1]
        param_counts = [ind.get('param_count', 0) for ind in last_population]
        elos = [ind.get('elo', 0) for ind in last_population]

        plt.figure(figsize=(10, 6))
        plt.scatter(param_counts, elos, alpha=0.6, s=100)
        plt.xlabel('參數量', fontsize=12)
        plt.ylabel('ELO 評級', fontsize=12)
        plt.title('最終種群：ELO vs 參數量', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)

        scatter_plot_path = os.path.join(output_dir, 'elo_vs_params.png')
        plt.savefig(scatter_plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        plot_files.append(scatter_plot_path)
        logger.info(f"✅ 散點圖：{scatter_plot_path}")

    return plot_files


def generate_report(
    data: Dict[str, Any],
    elo_results: Dict[str, Any],
    diversity_results: Dict[str, Any],
    best_architectures: List[Dict[str, Any]],
    plot_files: List[str],
    output_path: str
) -> str:
    """生成分析報告

    Parameters
    ----------
    data : Dict[str, Any]
        演化數據
    elo_results : Dict[str, Any]
        ELO 分析結果
    diversity_results : Dict[str, Any]
        多樣性分析結果
    best_architectures : List[Dict[str, Any]]
        最佳架構列表
    plot_files : List[str]
        圖表文件路徑
    output_path : str
        報告保存路徑

    Returns
    -------
    str
        報告內容
    """
    print_header("生成演化分析報告")

    num_generations = len(data['generations'])

    report = f"""
# 演化系統分析報告

**生成時間**：{time.strftime('%Y-%m-%d %H:%M:%S')}
**演化代數**：{num_generations}

---

## 1. ELO 評級進展

- **初始最佳 ELO**：{elo_results.get('initial_best_elo', 0):.1f}
- **最終最佳 ELO**：{elo_results.get('final_best_elo', 0):.1f}
- **ELO 提升**：{elo_results.get('elo_improvement', 0):.1f}
- **歷史最高 ELO**：{elo_results.get('max_elo', 0):.1f}
- **最終平均 ELO**：{elo_results.get('avg_elo_final', 0):.1f}

"""

    if elo_results.get('convergence_generation'):
        report += f"- **收斂代數**：第 {elo_results['convergence_generation']} 代\n"
    else:
        report += f"- **收斂狀態**：尚未收斂\n"

    report += f"""
**評估**：
"""

    if elo_results.get('elo_improvement', 0) > 100:
        report += "- ✅ ELO 顯著提升（> 100 分）\n"
    elif elo_results.get('elo_improvement', 0) > 50:
        report += "- ✅ ELO 中等提升（50-100 分）\n"
    else:
        report += "- ⚠️ ELO 提升有限（< 50 分）\n"

    report += f"""
---

## 2. 架構多樣性

- **初始多樣性**：{diversity_results.get('initial_diversity', 0):.1f}
- **最終多樣性**：{diversity_results.get('final_diversity', 0):.1f}
- **平均多樣性**：{diversity_results.get('avg_diversity', 0):.1f}
- **多樣性範圍**：{diversity_results.get('min_diversity', 0):.1f} - {diversity_results.get('max_diversity', 0):.1f}

**評估**：
"""

    final_div = diversity_results.get('final_diversity', 0)
    avg_div = diversity_results.get('avg_diversity', 1)

    if final_div > avg_div * 0.5:
        report += "- ✅ 多樣性維護良好\n"
    else:
        report += "- ⚠️ 多樣性下降過快，可能過早收斂\n"

    report += f"""
---

## 3. Top {len(best_architectures)} 最佳架構

"""

    for i, arch in enumerate(best_architectures):
        report += f"""
### 第 {i+1} 名

- **ELO**：{arch.get('elo', 0):.1f}
- **世代**：{arch.get('generation', 0)}
- **參數量**：{arch.get('param_count', 0):,}
- **架構配置**：
  - 殘差塊數：{arch.get('genome', {}).get('num_blocks', 'N/A')}
  - 濾波器數：{arch.get('genome', {}).get('base_filters', 'N/A')}
"""

    report += """
---

## 4. 可視化圖表

"""

    for plot_file in plot_files:
        plot_name = os.path.basename(plot_file)
        report += f"- [{plot_name}]({plot_file})\n"

    report += f"""
---

## 5. 總結與建議

"""

    # 總體評估
    issues = []
    recommendations = []

    if elo_results.get('elo_improvement', 0) < 50:
        issues.append("ELO 提升有限")
        recommendations.append("考慮增加演化代數或調整突變率")

    if diversity_results.get('final_diversity', 0) < diversity_results.get('avg_diversity', 1) * 0.3:
        issues.append("多樣性過度下降")
        recommendations.append("增加突變率或引入多樣性獎勵機制")

    if elo_results.get('convergence_generation'):
        gen = elo_results['convergence_generation']
        total = num_generations
        if gen < total * 0.5:
            issues.append("過早收斂")
            recommendations.append("增加種群大小或提高突變率")

    if not issues:
        report += "✅ **演化系統運作良好**：\n"
        report += "- ELO 評級穩定提升\n"
        report += "- 多樣性維護良好\n"
        report += "- 收斂速度適中\n"
    else:
        report += "⚠️ **發現以下問題**：\n"
        for issue in issues:
            report += f"- {issue}\n"

        report += "\n**改進建議**：\n"
        for rec in recommendations:
            report += f"- {rec}\n"

    report += f"\n---\n\n**生成時間**：{time.strftime('%Y-%m-%d %H:%M:%S')}\n"

    print(report)

    # 保存報告
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report)
        logger.info(f"✅ 報告已保存：{output_path}")

    return report


def main():
    """主函數"""
    parser = argparse.ArgumentParser(
        description='演化結果分析工具',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        '--evolution-dir',
        type=str,
        required=True,
        help='演化結果目錄'
    )
    parser.add_argument(
        '--generate-plots',
        action='store_true',
        help='生成可視化圖表'
    )
    parser.add_argument(
        '--top-k',
        type=int,
        default=5,
        help='分析前 K 個最佳架構（默認：5）'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='報告保存路徑'
    )

    args = parser.parse_args()

    print_header("演化結果分析")

    # 載入數據
    data = load_evolution_data(args.evolution_dir)

    if not data['generations']:
        logger.error("❌ 無法載入演化數據")
        return

    # 分析 ELO 進展
    elo_results = analyze_elo_progression(data)

    # 分析多樣性
    diversity_results = analyze_diversity(data)

    # 分析最佳架構
    best_architectures = analyze_best_architectures(data, top_k=args.top_k)

    # 生成圖表
    plot_files = []
    if args.generate_plots:
        plot_dir = os.path.join(args.evolution_dir, 'plots')
        plot_files = generate_plots(data, elo_results, diversity_results, plot_dir)

    # 生成報告
    if args.output is None:
        output_path = os.path.join(args.evolution_dir, 'analysis_report.md')
    else:
        output_path = args.output

    report = generate_report(
        data=data,
        elo_results=elo_results,
        diversity_results=diversity_results,
        best_architectures=best_architectures,
        plot_files=plot_files,
        output_path=output_path
    )

    print_header("完成！")


if __name__ == "__main__":
    main()

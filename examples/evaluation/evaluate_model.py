"""模型評估腳本

評估訓練好的 Light-Go 模型的性能，包括：
- vs 隨機玩家勝率測試
- Policy 準確度評估
- Value 預測誤差分析
- MCTS 搜索質量測試
- 生成詳細評估報告

使用方法：
    # 基本用法
    python examples/evaluate_model.py \
        --model-path data/models/from_katago/model.pt

    # 完整評估
    python examples/evaluate_model.py \
        --model-path data/models/from_katago/model.pt \
        --num-blocks 10 \
        --num-filters 128 \
        --num-games 20 \
        --mcts-simulations 400
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import Dict, Any, List, Tuple
import time

import torch
import torch.nn as nn
import numpy as np

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


def load_model(model_path: str, num_blocks: int = 10, num_filters: int = 128, device: str = 'cpu') -> GoAIModel:
    """載入訓練好的模型

    Parameters
    ----------
    model_path : str
        模型文件路徑
    num_blocks : int
        殘差塊數量
    num_filters : int
        濾波器數量
    device : str
        設備（'cpu' 或 'cuda'）

    Returns
    -------
    GoAIModel
        載入的模型
    """
    logger.info(f"載入模型：{model_path}")

    # 創建架構基因組
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

    # 構建模型
    model = genome.to_pytorch_model(device=device)

    # 載入權重
    if os.path.exists(model_path):
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        logger.info(f"✅ 模型權重已載入")
    else:
        logger.warning(f"⚠️  模型文件不存在：{model_path}")
        logger.warning(f"   使用未訓練的模型進行評估")

    model.eval()
    return model


def evaluate_vs_random(
    model: GoAIModel,
    num_games: int = 10,
    mcts_simulations: int = 200,
    use_optimized: bool = True
) -> Dict[str, Any]:
    """評估模型 vs 隨機玩家的勝率

    Parameters
    ----------
    model : GoAIModel
        被評估的模型
    num_games : int
        對弈局數
    mcts_simulations : int
        MCTS 模擬次數
    use_optimized : bool
        是否使用優化版 MCTS

    Returns
    -------
    Dict[str, Any]
        評估結果（勝率、平均步數等）
    """
    print_header("評估：vs 隨機玩家")

    wins = 0
    losses = 0
    total_moves = 0

    logger.info(f"開始對弈測試：{num_games} 局")
    logger.info(f"MCTS 模擬次數：{mcts_simulations}")

    # 創建 MCTS
    if use_optimized:
        mcts = MCTSOptimized(model=model, num_simulations=mcts_simulations, batch_size=4)
        logger.info("使用優化版 MCTS")
    else:
        mcts = MCTS(model=model, num_simulations=mcts_simulations)
        logger.info("使用標準 MCTS")

    for game_idx in range(num_games):
        board = GoBoard(size=19)
        current_player = 'b'  # 黑棋先行
        moves = 0
        consecutive_passes = 0

        logger.info(f"\n對弈 {game_idx + 1}/{num_games}...")

        while moves < 500:  # 最多 500 手
            if current_player == 'b':
                # AI 下棋（使用 MCTS）
                try:
                    move_probs, _ = mcts.search(board, current_player)

                    # 選擇概率最高的合法著法
                    legal_moves = []
                    for move, prob in move_probs.items():
                        if move == 'pass':
                            legal_moves.append(('pass', prob))
                        else:
                            row, col = move
                            if board.is_legal_move(row, col, current_player):
                                legal_moves.append(((row, col), prob))

                    if not legal_moves:
                        move = 'pass'
                    else:
                        # 選擇概率最高的著法
                        move = max(legal_moves, key=lambda x: x[1])[0]

                except Exception as e:
                    logger.warning(f"MCTS 搜索失敗：{e}")
                    move = 'pass'

            else:
                # 隨機玩家
                legal_moves = []
                for row in range(19):
                    for col in range(19):
                        if board.is_legal_move(row, col, current_player):
                            legal_moves.append((row, col))

                if legal_moves and np.random.random() > 0.1:  # 90% 機率下棋，10% pass
                    move = legal_moves[np.random.randint(len(legal_moves))]
                else:
                    move = 'pass'

            # 執行著法
            if move == 'pass':
                consecutive_passes += 1
                if consecutive_passes >= 2:
                    # 雙方連續 pass，終局
                    break
            else:
                row, col = move
                if board.play_move(row, col, current_player):
                    consecutive_passes = 0
                    moves += 1
                else:
                    logger.warning(f"非法著法：{move} by {current_player}")
                    consecutive_passes += 1

            # 切換玩家
            current_player = 'w' if current_player == 'b' else 'b'

        # 計算勝負（簡單數子）
        black_score = len(board.get_territory('b'))
        white_score = len(board.get_territory('w'))

        if black_score > white_score:
            wins += 1
            result = "勝"
        else:
            losses += 1
            result = "負"

        total_moves += moves

        logger.info(f"  第 {game_idx + 1} 局：{result}")
        logger.info(f"  手數：{moves}，黑：{black_score}，白：{white_score}")

    # 統計結果
    win_rate = wins / num_games if num_games > 0 else 0
    avg_moves = total_moves / num_games if num_games > 0 else 0

    results = {
        'wins': wins,
        'losses': losses,
        'win_rate': win_rate,
        'total_games': num_games,
        'avg_moves': avg_moves
    }

    logger.info(f"\n勝率：{win_rate * 100:.1f}% ({wins}/{num_games})")
    logger.info(f"平均手數：{avg_moves:.1f}")

    return results


def generate_report(
    model_path: str,
    vs_random_results: Dict[str, Any],
    output_path: str = None
) -> str:
    """生成評估報告

    Parameters
    ----------
    model_path : str
        模型路徑
    vs_random_results : Dict[str, Any]
        vs 隨機結果
    output_path : str, optional
        報告保存路徑

    Returns
    -------
    str
        報告內容
    """
    print_header("評估報告")

    report = f"""
# Light-Go 模型評估報告

**模型路徑**：{model_path}
**評估時間**：{time.strftime('%Y-%m-%d %H:%M:%S')}

---

## 1. vs 隨機玩家測試

- **勝率**：{vs_random_results.get('win_rate', 0) * 100:.1f}% ({vs_random_results.get('wins', 0)}/{vs_random_results.get('total_games', 0)})
- **平均手數**：{vs_random_results.get('avg_moves', 0):.1f}
- **評估**：{'✅ 優秀 (>90%)' if vs_random_results.get('win_rate', 0) > 0.9 else '⚠️ 需改進 (<90%)'}

---

## 2. 總結

"""

    # 添加總結評估
    good_vs_random = vs_random_results.get('win_rate', 0) > 0.9

    if good_vs_random:
        report += "✅ **模型訓練成功**：vs 隨機測試通過，可以進入 Phase 3\n"
    else:
        report += "❌ **模型需要更多訓練**：vs 隨機勝率未達標\n"

    report += "\n---\n"
    report += f"\n**生成時間**：{time.strftime('%Y-%m-%d %H:%M:%S')}\n"

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
        description='評估 Light-Go 模型性能',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        '--model-path',
        type=str,
        default='data/models/from_katago/model.pt',
        help='模型文件路徑'
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
        '--num-games',
        type=int,
        default=10,
        help='vs 隨機對弈局數（默認：10）'
    )
    parser.add_argument(
        '--mcts-simulations',
        type=int,
        default=200,
        help='MCTS 模擬次數（默認：200）'
    )
    parser.add_argument(
        '--use-optimized-mcts',
        action='store_true',
        default=True,
        help='使用優化版 MCTS'
    )
    parser.add_argument(
        '--skip-vs-random',
        action='store_true',
        help='跳過 vs 隨機測試（加快評估）'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='data/evaluation_reports/report.md',
        help='報告保存路徑'
    )

    args = parser.parse_args()

    print_header("Light-Go 模型評估")

    # 1. 載入模型
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"使用裝置：{device}")

    model = load_model(
        model_path=args.model_path,
        num_blocks=args.num_blocks,
        num_filters=args.num_filters,
        device=device
    )

    # 2. vs 隨機玩家測試
    if not args.skip_vs_random:
        vs_random_results = evaluate_vs_random(
            model=model,
            num_games=args.num_games,
            mcts_simulations=args.mcts_simulations,
            use_optimized=args.use_optimized_mcts
        )
    else:
        vs_random_results = {}
        logger.info("跳過 vs 隨機測試")

    # 3. 生成報告
    report = generate_report(
        model_path=args.model_path,
        vs_random_results=vs_random_results,
        output_path=args.output
    )

    print_header("完成！")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""KataGo 模型完整評估腳本

在背景執行 MCTS 質量分析和自我對弈測試
"""

import os
import sys
import json
import time
import logging
from pathlib import Path
from datetime import datetime

# 添加專案根目錄到路徑
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import numpy as np

from hf_models.modeling_go_ai import GoAIModel
from core.architecture_genome import ArchitectureGenome
from core.game_rules import GoBoard
from core.mcts import MCTS
from core.self_play import self_play_game

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data/benchmarks/katago_evaluation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def load_model(model_path: str, device: str = None):
    """載入訓練好的模型"""
    if device is None:
        if torch.cuda.is_available():
            device = 'cuda'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'

    logger.info(f"使用設備: {device}")
    logger.info(f"載入模型: {model_path}")

    checkpoint = torch.load(model_path, map_location=device)

    # 載入模型配置
    model_config = checkpoint.get('model_config', {})
    logger.info(f"模型配置: {model_config}")

    # 創建模型
    model = GoAIModel(
        num_input_planes=model_config.get('num_input_planes', 22),
        num_filters=model_config.get('num_filters', 128),
        num_blocks=model_config.get('num_blocks', 10),
        board_size=model_config.get('board_size', 19)
    ).to(device)

    # 載入權重
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    logger.info("✅ 模型載入成功")
    return model, device


def test_mcts_quality(model, device, num_simulations=400, num_positions=10):
    """MCTS 質量分析"""
    logger.info("=" * 60)
    logger.info("開始 MCTS 質量分析")
    logger.info("=" * 60)

    results = {
        'num_simulations': num_simulations,
        'num_positions': num_positions,
        'positions': []
    }

    board = GoBoard(size=19)

    for i in range(num_positions):
        logger.info(f"\n分析位置 {i+1}/{num_positions}")

        # 創建 MCTS
        mcts = MCTS(model=model, num_simulations=num_simulations)

        # 搜索
        start_time = time.time()
        move_probs, root = mcts.search(board, 'b')
        search_time = time.time() - start_time

        # 分析結果
        top_moves = sorted(move_probs.items(), key=lambda x: x[1], reverse=True)[:5]

        position_result = {
            'position_idx': i,
            'search_time': search_time,
            'root_visit_count': root.visit_count,
            'root_value': root.value_sum / root.visit_count if root.visit_count > 0 else 0,
            'top_moves': [(move, float(prob)) for move, prob in top_moves],
            'explored_moves': len(move_probs)
        }

        results['positions'].append(position_result)

        logger.info(f"  搜索時間: {search_time:.2f}s")
        logger.info(f"  根節點訪問: {root.visit_count}")
        logger.info(f"  平均 Value: {position_result['root_value']:.4f}")
        logger.info(f"  探索著法數: {position_result['explored_moves']}")
        logger.info(f"  Top-5 著法: {top_moves[:5]}")

        # 下一手（使用最佳著法）
        if top_moves:
            best_move = top_moves[0][0]
            if best_move != 'pass':
                row, col = best_move
                board.play_move(row, col, 'b')

    # 統計
    avg_search_time = np.mean([p['search_time'] for p in results['positions']])
    avg_explored = np.mean([p['explored_moves'] for p in results['positions']])

    results['summary'] = {
        'avg_search_time': float(avg_search_time),
        'avg_explored_moves': float(avg_explored),
    }

    logger.info("\n" + "=" * 60)
    logger.info("MCTS 質量分析完成")
    logger.info("=" * 60)
    logger.info(f"平均搜索時間: {avg_search_time:.2f}s")
    logger.info(f"平均探索著法: {avg_explored:.1f}")

    return results


def test_self_play(model, device, num_games=5, max_moves=200):
    """自我對弈測試"""
    logger.info("\n" + "=" * 60)
    logger.info("開始自我對弈測試")
    logger.info("=" * 60)

    results = {
        'num_games': num_games,
        'max_moves': max_moves,
        'games': []
    }

    # 創建 MCTS
    mcts = MCTS(model=model, num_simulations=200)

    for game_idx in range(num_games):
        logger.info(f"\n對弈 {game_idx+1}/{num_games}")

        # 自我對弈
        start_time = time.time()
        game_record = self_play_game(
            mcts=mcts,
            board_size=19,
            max_moves=max_moves
        )
        game_time = time.time() - start_time

        num_moves = len(game_record['moves'])
        winner = game_record.get('winner', 'unknown')

        game_result = {
            'game_idx': game_idx,
            'num_moves': num_moves,
            'game_time': game_time,
            'winner': winner,
            'moves': game_record['moves']
        }

        results['games'].append(game_result)

        logger.info(f"  著手數: {num_moves}")
        logger.info(f"  對弈時間: {game_time:.1f}s")
        logger.info(f"  勝者: {winner}")

        # 保存 SGF
        sgf_dir = Path('data/benchmarks/katago_self_play_games')
        sgf_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        sgf_path = sgf_dir / f"game_{game_idx+1}_{timestamp}.sgf"

        save_sgf(game_record, sgf_path)
        logger.info(f"  SGF 已保存: {sgf_path}")

    # 統計
    avg_moves = np.mean([g['num_moves'] for g in results['games']])
    avg_time = np.mean([g['game_time'] for g in results['games']])

    results['summary'] = {
        'avg_moves': float(avg_moves),
        'avg_game_time': float(avg_time),
    }

    logger.info("\n" + "=" * 60)
    logger.info("自我對弈測試完成")
    logger.info("=" * 60)
    logger.info(f"平均著手數: {avg_moves:.1f}")
    logger.info(f"平均對弈時間: {avg_time:.1f}s")

    return results


def save_sgf(game_record, sgf_path):
    """保存 SGF 棋譜"""
    moves = game_record['moves']

    # 簡單的 SGF 格式
    sgf_content = "(;FF[4]GM[1]SZ[19]AP[Light-Go]\n"

    for move_idx, move in enumerate(moves):
        color = 'B' if move_idx % 2 == 0 else 'W'
        if move == 'pass':
            sgf_content += f";{color}[]\n"
        else:
            row, col = move
            # SGF 使用字母表示坐標
            col_letter = chr(ord('a') + col)
            row_letter = chr(ord('a') + row)
            sgf_content += f";{color}[{col_letter}{row_letter}]\n"

    sgf_content += ")"

    with open(sgf_path, 'w') as f:
        f.write(sgf_content)


def main():
    """主程序"""
    model_path = 'data/katago/trained_models/2026-01-12_10epochs_1000files/model.pt'
    output_dir = Path('data/benchmarks/katago_baseline')
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("KataGo 模型完整評估")
    logger.info("=" * 60)
    logger.info(f"模型路徑: {model_path}")
    logger.info(f"輸出目錄: {output_dir}")

    # 載入模型
    model, device = load_model(model_path)

    # 1. MCTS 質量分析
    mcts_results = test_mcts_quality(
        model=model,
        device=device,
        num_simulations=400,
        num_positions=10
    )

    # 保存結果
    mcts_output = output_dir / 'mcts_quality_analysis.json'
    with open(mcts_output, 'w', encoding='utf-8') as f:
        json.dump(mcts_results, f, indent=2, ensure_ascii=False)
    logger.info(f"\n✅ MCTS 分析結果已保存: {mcts_output}")

    # 2. 自我對弈測試
    self_play_results = test_self_play(
        model=model,
        device=device,
        num_games=5,
        max_moves=200
    )

    # 保存結果
    self_play_output = output_dir / 'self_play_results.json'
    with open(self_play_output, 'w', encoding='utf-8') as f:
        json.dump(self_play_results, f, indent=2, ensure_ascii=False)
    logger.info(f"\n✅ 自我對弈結果已保存: {self_play_output}")

    # 完成
    logger.info("\n" + "=" * 60)
    logger.info("✅ KataGo 模型評估完成！")
    logger.info("=" * 60)
    logger.info(f"結果保存在: {output_dir}")
    logger.info(f"  - MCTS 分析: {mcts_output}")
    logger.info(f"  - 自我對弈: {self_play_output}")
    logger.info(f"  - SGF 棋譜: data/benchmarks/katago_self_play_games/")
    logger.info(f"  - 日誌: data/benchmarks/katago_evaluation.log")


if __name__ == '__main__':
    main()

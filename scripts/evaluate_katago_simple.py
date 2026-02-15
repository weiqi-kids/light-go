#!/usr/bin/env python3
"""KataGo 模型簡化評估腳本

執行 MCTS 質量分析和自我對弈測試
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
from core.game_rules import GoBoard
from core.mcts import MCTS
from core.self_play import SelfPlayWorker

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_model(model_path: str):
    """載入訓練好的模型"""
    device = 'cuda' if torch.cuda.is_available() else ('mps' if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() else 'cpu')

    logger.info(f"使用設備: {device}")
    logger.info(f"載入模型: {model_path}")

    checkpoint = torch.load(model_path, map_location=device)

    # 從 checkpoint 推斷模型配置
    if 'conv_input.weight' in checkpoint:
        # checkpoint 是直接的 state_dict
        conv_weight = checkpoint['conv_input.weight']
        num_filters = conv_weight.shape[0]
        num_input_planes = conv_weight.shape[1]

        # 計算 residual blocks 數量
        num_blocks = sum(1 for key in checkpoint.keys()
                        if 'residual_blocks' in key and 'conv1.weight' in key)

        logger.info(f"推斷配置: planes={num_input_planes}, filters={num_filters}, blocks={num_blocks}")

        model = GoAIModel(
            num_input_planes=num_input_planes,
            num_filters=num_filters,
            num_blocks=num_blocks,
            board_size=19
        ).to(device)

        model.load_state_dict(checkpoint)
    else:
        # checkpoint 包含配置字典
        model_config = checkpoint.get('model_config', {})
        model = GoAIModel(
            num_input_planes=model_config.get('num_input_planes', 22),
            num_filters=model_config.get('num_filters', 128),
            num_blocks=model_config.get('num_blocks', 10),
            board_size=model_config.get('board_size', 19)
        ).to(device)
        model.load_state_dict(checkpoint['model_state_dict'])

    model.eval()

    logger.info("✅ 模型載入成功")
    return model, device


def test_mcts_quality(model, num_simulations=200, num_positions=5):
    """MCTS 質量分析（簡化版）"""
    logger.info("=" * 60)
    logger.info("開始 MCTS 質量分析")
    logger.info("=" * 60)

    results = {
        'num_simulations': num_simulations,
        'num_positions': num_positions,
        'positions': []
    }

    board = GoBoard(size=19)
    mcts = MCTS(model=model, num_simulations=num_simulations)

    for i in range(num_positions):
        logger.info(f"\n分析位置 {i+1}/{num_positions}")

        start_time = time.time()
        move_probs, root = mcts.search(board, 'b')
        search_time = time.time() - start_time

        # move_probs 是 numpy array (board_size^2 + 1,)
        board_size = board.size

        # 轉換為 (move, prob) 列表
        move_prob_list = []
        for idx in range(board_size * board_size):
            if move_probs[idx] > 0:
                row = idx // board_size
                col = idx % board_size
                move_prob_list.append(((row, col), float(move_probs[idx])))

        # Pass move
        if move_probs[-1] > 0:
            move_prob_list.append(('pass', float(move_probs[-1])))

        # 排序取前 5
        top_moves = sorted(move_prob_list, key=lambda x: x[1], reverse=True)[:5]

        position_result = {
            'position_idx': i,
            'search_time': search_time,
            'root_visit_count': root.visit_count,
            'root_value': root.total_value / root.visit_count if root.visit_count > 0 else 0,
            'top_moves': [(str(move), prob) for move, prob in top_moves],
            'explored_moves': len([p for p in move_probs if p > 0])
        }

        results['positions'].append(position_result)

        logger.info(f"  搜索時間: {search_time:.2f}s")
        logger.info(f"  根節點訪問: {root.visit_count}")
        logger.info(f"  平均 Value: {position_result['root_value']:.4f}")
        logger.info(f"  探索著法數: {position_result['explored_moves']}")

        # 下一手
        if top_moves and top_moves[0][0] != 'pass':
            move = top_moves[0][0]
            if isinstance(move, tuple):
                row, col = move
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


def test_self_play(model, num_games=3):
    """自我對弈測試（簡化版）"""
    logger.info("\n" + "=" * 60)
    logger.info("開始自我對弈測試")
    logger.info("=" * 60)

    results = {
        'num_games': num_games,
        'games': []
    }

    # 創建 SelfPlayWorker
    worker = SelfPlayWorker(
        model=model,
        num_simulations=100,  # 降低以加快速度
        board_size=19,
        max_moves=200
    )

    sgf_dir = Path('data/benchmarks/katago_self_play_games')
    sgf_dir.mkdir(parents=True, exist_ok=True)

    for game_idx in range(num_games):
        logger.info(f"\n對弈 {game_idx+1}/{num_games}")

        start_time = time.time()
        game_result = worker.play_game(verbose=False)
        game_time = time.time() - start_time

        # 從 game_result.moves 獲取著手列表
        move_list = []
        if game_result.moves:
            for move, color in game_result.moves:
                if move is None:
                    move_list.append({'color': color, 'move': 'pass'})
                else:
                    row, col = move
                    move_list.append({'color': color, 'move': [row, col]})

        game_info = {
            'game_idx': game_idx,
            'num_moves': game_result.num_moves,
            'game_time': game_time,
            'winner': game_result.winner,
            'moves': move_list  # 保存完整著手列表
        }

        results['games'].append(game_info)

        logger.info(f"  著手數: {game_result.num_moves}")
        logger.info(f"  對弈時間: {game_time:.1f}s")
        logger.info(f"  勝者: {game_result.winner}")

        # 保存完整對弈記錄（包含著手）
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        record_path = sgf_dir / f"game_{game_idx+1}_{timestamp}.json"

        with open(record_path, 'w', encoding='utf-8') as f:
            json.dump(game_info, f, indent=2, ensure_ascii=False)

        logger.info(f"  記錄已保存: {record_path}")

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


def main():
    """主程序"""
    model_path = 'data/katago/trained_models/2026-01-12_10epochs_1000files/model.pt'
    output_dir = Path('data/benchmarks/katago_baseline')
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("KataGo 模型評估（簡化版）")
    logger.info("=" * 60)
    logger.info(f"模型路徑: {model_path}")
    logger.info(f"輸出目錄: {output_dir}")

    # 載入模型
    try:
        model, device = load_model(model_path)
    except Exception as e:
        logger.error(f"載入模型失敗: {e}")
        return

    # 1. MCTS 質量分析
    try:
        logger.info("\n" + "=" * 60)
        logger.info("任務 1/2: MCTS 質量分析")
        logger.info("=" * 60)

        mcts_results = test_mcts_quality(
            model=model,
            num_simulations=200,
            num_positions=5
        )

        mcts_output = output_dir / 'mcts_quality_analysis.json'
        with open(mcts_output, 'w', encoding='utf-8') as f:
            json.dump(mcts_results, f, indent=2, ensure_ascii=False)
        logger.info(f"\n✅ MCTS 分析結果已保存: {mcts_output}")

    except Exception as e:
        logger.error(f"MCTS 分析失敗: {e}", exc_info=True)

    # 2. 自我對弈測試
    try:
        logger.info("\n" + "=" * 60)
        logger.info("任務 2/2: 自我對弈測試")
        logger.info("=" * 60)

        self_play_results = test_self_play(
            model=model,
            num_games=3
        )

        self_play_output = output_dir / 'self_play_results.json'
        with open(self_play_output, 'w', encoding='utf-8') as f:
            json.dump(self_play_results, f, indent=2, ensure_ascii=False)
        logger.info(f"\n✅ 自我對弈結果已保存: {self_play_output}")

    except Exception as e:
        logger.error(f"自我對弈測試失敗: {e}", exc_info=True)

    # 完成
    logger.info("\n" + "=" * 60)
    logger.info("✅ KataGo 模型評估完成！")
    logger.info("=" * 60)
    logger.info(f"結果保存在: {output_dir}")
    logger.info(f"  - MCTS 分析: mcts_quality_analysis.json")
    logger.info(f"  - 自我對弈: self_play_results.json")
    logger.info(f"  - 對弈記錄: data/benchmarks/katago_self_play_games/")


if __name__ == '__main__':
    main()

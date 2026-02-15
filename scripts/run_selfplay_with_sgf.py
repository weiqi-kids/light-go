#!/usr/bin/env python
"""快速生成 self-play 對局並保存 SGF 檔案"""

import sys
import logging
from pathlib import Path

from hf_models.modeling_go_ai import GoAIModel
from core.self_play import SelfPlayWorker

# 設定日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_selfplay(num_games=1, num_simulations=50):
    """執行 self-play 並生成 SGF 檔案

    Parameters
    ----------
    num_games : int
        要生成的對局數量（預設 1）
    num_simulations : int
        每步 MCTS 模擬次數（預設 50，越大越慢但品質越好）
    """

    logger.info("=" * 60)
    logger.info("開始生成 Self-Play 對局")
    logger.info("=" * 60)

    # 建立一個小型模型（速度較快）
    logger.info("\n建立模型...")
    num_filters = 32
    num_blocks = 2
    model = GoAIModel(
        num_input_planes=22,
        num_filters=num_filters,      # 較小的網路，速度快
        num_blocks=num_blocks,        # 較少的殘差塊
        board_size=19,
        device='cpu'         # 使用 CPU（如果有 GPU 可改成 'cuda'）
    )
    logger.info(f"  模型大小: num_filters={num_filters}, num_blocks={num_blocks}")

    # 建立 SelfPlayWorker，啟用 SGF 生成
    logger.info("\n建立 SelfPlayWorker...")
    worker = SelfPlayWorker(
        model=model,
        num_simulations=num_simulations,
        board_size=19,
        komi=7.5,
        temperature_threshold=30,
        save_sgf=True,                    # 啟用 SGF 生成
        sgf_dir='data/sgf/selfplay',      # SGF 保存目錄
        worker_id=0
    )
    logger.info(f"  MCTS 模擬次數: {num_simulations}")
    logger.info(f"  SGF 保存目錄: data/sgf/selfplay/")

    # 生成對局
    results = []
    for i in range(num_games):
        logger.info(f"\n{'-' * 60}")
        logger.info(f"對局 {i + 1}/{num_games}")
        logger.info(f"{'-' * 60}")

        result = worker.play_game(verbose=True)
        results.append(result)

        logger.info(f"\n對局完成:")
        logger.info(f"  勝者: {result.winner}")
        logger.info(f"  手數: {result.num_moves}")
        logger.info(f"  訓練樣本: {len(result.examples)}")

    # 顯示結果摘要
    logger.info("\n" + "=" * 60)
    logger.info("所有對局完成！")
    logger.info("=" * 60)
    logger.info(f"\n總共生成 {num_games} 局對局")

    # 列出生成的 SGF 檔案
    sgf_dir = Path('data/sgf/selfplay')
    sgf_files = sorted(sgf_dir.glob('*.sgf'))

    logger.info(f"\nSGF 檔案保存在: {sgf_dir.absolute()}")
    logger.info(f"共有 {len(sgf_files)} 個 SGF 檔案:")

    for sgf_file in sgf_files[-num_games:]:  # 只顯示最新生成的
        size = sgf_file.stat().st_size
        logger.info(f"  - {sgf_file.name} ({size} bytes)")

    logger.info("\n你可以使用以下工具查看 SGF 檔案:")
    logger.info("  - Sabaki: https://sabaki.yichuanshen.de/")
    logger.info("  - GoGui: http://gogui.sourceforge.net/")
    logger.info("  - 線上檢視器: https://online-go.com/demo")

    return results


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='生成 self-play 對局並保存 SGF')
    parser.add_argument(
        '--games',
        type=int,
        default=1,
        help='要生成的對局數量（預設: 1）'
    )
    parser.add_argument(
        '--simulations',
        type=int,
        default=50,
        help='每步 MCTS 模擬次數（預設: 50，建議 10-100）'
    )

    args = parser.parse_args()

    try:
        results = run_selfplay(
            num_games=args.games,
            num_simulations=args.simulations
        )
        sys.exit(0)
    except KeyboardInterrupt:
        logger.info("\n\n使用者中斷")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\n錯誤: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

"""互動式圍棋對弈界面

允許人類玩家與 AI 進行終端機對弈。

使用方法：
    # 使用訓練好的模型
    python examples/play_interactive.py --model-path data/models/from_katago/model.pt

    # 使用未訓練模型（快速測試）
    python examples/play_interactive.py --no-model

    # 調整 MCTS 模擬次數
    python examples/play_interactive.py --mcts-simulations 400

    # AI 下黑棋，人類下白棋
    python examples/play_interactive.py --ai-color black
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import Optional, Tuple, Dict

import torch
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
    level=logging.WARNING,  # 降低日誌級別，避免干擾對弈
    format='%(message)s'
)
logger = logging.getLogger(__name__)


class InteractiveGoGame:
    """互動式圍棋對弈管理器"""

    def __init__(
        self,
        model: GoAIModel,
        board_size: int = 19,
        mcts_simulations: int = 200,
        ai_color: str = 'b',
        use_optimized_mcts: bool = True
    ):
        """初始化對弈管理器

        Parameters
        ----------
        model : GoAIModel
            AI 模型
        board_size : int
            棋盤大小
        mcts_simulations : int
            MCTS 模擬次數
        ai_color : str
            AI 執子顏色（'b' 或 'w'）
        use_optimized_mcts : bool
            是否使用優化版 MCTS
        """
        self.model = model
        self.board_size = board_size
        self.ai_color = ai_color
        self.human_color = 'w' if ai_color == 'b' else 'b'

        # 創建 MCTS
        if use_optimized_mcts:
            self.mcts = MCTSOptimized(
                model=model,
                num_simulations=mcts_simulations,
                batch_size=4
            )
        else:
            self.mcts = MCTS(
                model=model,
                num_simulations=mcts_simulations
            )

        self.board = GoBoard(size=board_size)
        self.move_history = []

    def display_board(self, last_move: Optional[Tuple[int, int]] = None):
        """顯示棋盤

        Parameters
        ----------
        last_move : Optional[Tuple[int, int]]
            最後一手的座標（用於標記）
        """
        # 列標籤（跳過 I）
        col_labels = 'ABCDEFGHJKLMNOPQRST'[:self.board_size]

        print("\n   " + " ".join(col_labels))

        for row in range(self.board_size - 1, -1, -1):
            row_num = row + 1
            row_str = f"{row_num:2d} "

            for col in range(self.board_size):
                stone = self.board.board[row][col]

                # 星位標記
                is_star = self._is_star_point(row, col)

                if stone == 'b':
                    row_str += "X "
                elif stone == 'w':
                    row_str += "O "
                elif last_move and last_move == (row, col):
                    row_str += "* "  # 標記最後一手
                elif is_star:
                    row_str += "+ "
                else:
                    row_str += ". "

            row_str += f" {row_num:2d}"
            print(row_str)

        print("   " + " ".join(col_labels))

    def _is_star_point(self, row: int, col: int) -> bool:
        """判斷是否為星位"""
        if self.board_size == 19:
            star_points = [(3, 3), (3, 9), (3, 15),
                          (9, 3), (9, 9), (9, 15),
                          (15, 3), (15, 9), (15, 15)]
            return (row, col) in star_points
        elif self.board_size == 13:
            star_points = [(3, 3), (3, 9), (6, 6), (9, 3), (9, 9)]
            return (row, col) in star_points
        elif self.board_size == 9:
            star_points = [(2, 2), (2, 6), (4, 4), (6, 2), (6, 6)]
            return (row, col) in star_points
        return False

    def parse_move(self, move_str: str) -> Optional[Tuple[int, int]]:
        """解析玩家輸入的著手

        支持格式：
        - D4, d4 (字母 + 數字)
        - 3,3, 3 3 (數字 + 數字)
        - pass, PASS (認輸或 pass)

        Parameters
        ----------
        move_str : str
            玩家輸入

        Returns
        -------
        Optional[Tuple[int, int]]
            座標 (row, col)，如果是 pass 則返回 None
        """
        move_str = move_str.strip().upper()

        if move_str in ['PASS', 'P', 'RESIGN', 'R']:
            return None

        # 嘗試字母+數字格式（如 D4）
        if len(move_str) >= 2:
            col_labels = 'ABCDEFGHJKLMNOPQRST'
            col_char = move_str[0]
            row_str = move_str[1:]

            if col_char in col_labels:
                try:
                    col = col_labels.index(col_char)
                    row = int(row_str) - 1

                    if 0 <= row < self.board_size and 0 <= col < self.board_size:
                        return (row, col)
                except ValueError:
                    pass

        # 嘗試數字,數字格式（如 3,3 或 3 3）
        parts = move_str.replace(',', ' ').split()
        if len(parts) == 2:
            try:
                row = int(parts[0])
                col = int(parts[1])
                if 0 <= row < self.board_size and 0 <= col < self.board_size:
                    return (row, col)
            except ValueError:
                pass

        return None

    def get_human_move(self) -> Optional[Tuple[int, int]]:
        """獲取人類玩家的著手

        Returns
        -------
        Optional[Tuple[int, int]]
            座標，如果是 pass 則返回 None
        """
        while True:
            move_str = input(f"\n您的著手 ({self.human_color.upper()}，如 D4 或 3,3，'pass' 跳過，'resign' 認輸): ")

            if move_str.strip().upper() in ['RESIGN', 'R']:
                return 'resign'

            move = self.parse_move(move_str)

            if move is None:
                return None  # Pass

            row, col = move

            if self.board.is_legal_move(row, col, self.human_color):
                return (row, col)
            else:
                print(f"❌ 非法著手！請選擇其他位置。")

    def get_ai_move(self) -> Tuple[Optional[Tuple[int, int]], float]:
        """獲取 AI 的著手

        Returns
        -------
        Tuple[Optional[Tuple[int, int]], float]
            (著手座標, 勝率預測)
        """
        print(f"\n🤖 AI 思考中...")

        try:
            move_probs, root = self.mcts.search(self.board, self.ai_color)

            # 獲取勝率預測
            win_rate = (root.value_sum / root.visit_count + 1) / 2 if root.visit_count > 0 else 0.5

            # 選擇概率最高的合法著法
            legal_moves = []
            for move, prob in move_probs.items():
                if move == 'pass':
                    legal_moves.append(('pass', prob))
                else:
                    row, col = move
                    if self.board.is_legal_move(row, col, self.ai_color):
                        legal_moves.append(((row, col), prob))

            if not legal_moves:
                return None, win_rate

            # 選擇概率最高的著法
            best_move, best_prob = max(legal_moves, key=lambda x: x[1])

            if best_move == 'pass':
                return None, win_rate

            return best_move, win_rate

        except Exception as e:
            logger.error(f"AI 思考失敗：{e}")
            import traceback
            traceback.print_exc()
            return None, 0.5

    def play(self):
        """開始對弈"""
        print("\n" + "=" * 70)
        print("  Light-Go 互動對弈")
        print("=" * 70)
        print(f"\n棋盤大小：{self.board_size}x{self.board_size}")
        print(f"AI 執子：{self.ai_color.upper()} ({'黑' if self.ai_color == 'b' else '白'})")
        print(f"您執子：{self.human_color.upper()} ({'黑' if self.human_color == 'b' else '白'})")
        print("\n輸入格式：")
        print("  - 字母+數字：D4, d16")
        print("  - 數字座標：3,3 或 3 3")
        print("  - Pass：pass 或 p")
        print("  - 認輸：resign 或 r")

        current_player = 'b'  # 黑棋先行
        consecutive_passes = 0
        move_count = 0

        while move_count < 500:  # 最多 500 手
            self.display_board()

            if current_player == self.human_color:
                # 人類玩家
                move = self.get_human_move()

                if move == 'resign':
                    print(f"\n您認輸了。AI 獲勝！")
                    break

                if move is None:
                    print(f"您選擇 pass")
                    consecutive_passes += 1
                else:
                    row, col = move
                    self.board.play_move(row, col, current_player)
                    consecutive_passes = 0
                    move_count += 1
                    self.move_history.append((current_player, move))

            else:
                # AI 玩家
                move, win_rate = self.get_ai_move()

                if move is None:
                    print(f"✅ AI 選擇 pass（勝率預測：{win_rate*100:.1f}%）")
                    consecutive_passes += 1
                else:
                    row, col = move
                    col_labels = 'ABCDEFGHJKLMNOPQRST'
                    move_str = f"{col_labels[col]}{row+1}"
                    print(f"✅ AI 下在 {move_str} (勝率預測：{win_rate*100:.1f}%)")

                    self.board.play_move(row, col, current_player)
                    consecutive_passes = 0
                    move_count += 1
                    self.move_history.append((current_player, move))

            # 檢查是否終局
            if consecutive_passes >= 2:
                print("\n雙方連續 pass，對弈結束。")
                self.display_board()
                self._calculate_winner()
                break

            # 切換玩家
            current_player = 'w' if current_player == 'b' else 'b'

        if move_count >= 500:
            print("\n達到最大手數限制，對弈結束。")
            self._calculate_winner()

    def _calculate_winner(self):
        """計算勝負"""
        black_score = len(self.board.get_territory('b'))
        white_score = len(self.board.get_territory('w'))

        print(f"\n最終得分：")
        print(f"  黑棋：{black_score}")
        print(f"  白棋：{white_score}")

        if black_score > white_score:
            winner = '黑'
            margin = black_score - white_score
        elif white_score > black_score:
            winner = '白'
            margin = white_score - black_score
        else:
            winner = None
            margin = 0

        if winner:
            print(f"\n{winner}棋獲勝！（{margin} 目）")
        else:
            print(f"\n平局！")


def load_model(
    model_path: Optional[str] = None,
    num_blocks: int = 10,
    num_filters: int = 128,
    device: str = 'cpu'
) -> GoAIModel:
    """載入模型

    Parameters
    ----------
    model_path : Optional[str]
        模型文件路徑，如果為 None 則使用未訓練模型
    num_blocks : int
        殘差塊數量
    num_filters : int
        濾波器數量
    device : str
        設備

    Returns
    -------
    GoAIModel
        載入的模型
    """
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

    # 載入權重（如果提供）
    if model_path and os.path.exists(model_path):
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        print(f"✅ 模型已載入：{model_path}")
    else:
        if model_path:
            print(f"⚠️  模型文件不存在：{model_path}")
        print(f"ℹ️  使用未訓練模型（僅用於測試結構）")

    model.eval()
    return model


def main():
    """主函數"""
    parser = argparse.ArgumentParser(
        description='Light-Go 互動對弈界面',
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
        help='使用未訓練模型（快速測試）'
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
        '--board-size',
        type=int,
        default=19,
        help='棋盤大小（默認：19）'
    )
    parser.add_argument(
        '--mcts-simulations',
        type=int,
        default=200,
        help='MCTS 模擬次數（默認：200）'
    )
    parser.add_argument(
        '--ai-color',
        type=str,
        choices=['black', 'white', 'b', 'w'],
        default='black',
        help='AI 執子顏色（默認：black）'
    )
    parser.add_argument(
        '--use-standard-mcts',
        action='store_true',
        help='使用標準 MCTS（默認：優化版）'
    )

    args = parser.parse_args()

    # 處理 AI 顏色
    ai_color = 'b' if args.ai_color in ['black', 'b'] else 'w'

    # 載入模型
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用裝置：{device}")

    model_path = None if args.no_model else args.model_path
    model = load_model(
        model_path=model_path,
        num_blocks=args.num_blocks,
        num_filters=args.num_filters,
        device=device
    )

    # 創建對弈管理器
    game = InteractiveGoGame(
        model=model,
        board_size=args.board_size,
        mcts_simulations=args.mcts_simulations,
        ai_color=ai_color,
        use_optimized_mcts=not args.use_standard_mcts
    )

    # 開始對弈
    try:
        game.play()
    except KeyboardInterrupt:
        print("\n\n對弈中斷。")
    except Exception as e:
        print(f"\n❌ 發生錯誤：{e}")
        import traceback
        traceback.print_exc()

    print("\n感謝對弈！")


if __name__ == "__main__":
    main()

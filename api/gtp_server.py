"""GTP (Go Text Protocol) 服務器 - 連接實際 AI 模型

完整的 GTP 協議實現，可與 GoGUI 等圖形界面配合使用。

使用方法：
    # 啟動 GTP 服務器（使用訓練好的模型）
    python api/gtp_server.py --model-path data/models/from_katago/model.pt

    # 使用未訓練模型測試
    python api/gtp_server.py --no-model

    # 調整 MCTS 模擬次數
    python api/gtp_server.py --mcts-simulations 400

    # 指定端口
    python api/gtp_server.py --port 2718

連接 GoGUI：
    1. 啟動此服務器
    2. 打開 GoGUI
    3. Program -> New Program
    4. Command: python api/gtp_server.py --model-path <path>
"""

from __future__ import annotations

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import List, Tuple, Optional

import torch

# 添加專案根目錄到路徑
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from hf_models.modeling_go_ai import GoAIModel
from core.architecture_genome import ArchitectureGenome
from core.game_rules import GoBoard
from core.mcts_optimized import MCTSOptimized

logger = logging.getLogger(__name__)


class GTPEngine:
    """GTP 引擎 - 集成 Light-Go AI"""

    def __init__(
        self,
        model: GoAIModel,
        mcts_simulations: int = 200,
        use_optimized: bool = True
    ):
        """初始化 GTP 引擎

        Parameters
        ----------
        model : GoAIModel
            AI 模型
        mcts_simulations : int
            MCTS 模擬次數
        use_optimized : bool
            是否使用優化版 MCTS
        """
        self.model = model
        self.board_size = 19
        self.komi = 6.5
        self.board: Optional[GoBoard] = None
        self.mcts_simulations = mcts_simulations

        # 創建 MCTS
        self.mcts = MCTSOptimized(
            model=model,
            num_simulations=mcts_simulations,
            batch_size=4
        )

        self.clear_board()

        # GTP 命令處理器映射
        self.handlers = {
            "protocol_version": self.handle_protocol_version,
            "name": self.handle_name,
            "version": self.handle_version,
            "known_command": self.handle_known_command,
            "list_commands": self.handle_list_commands,
            "boardsize": self.handle_boardsize,
            "clear_board": self.handle_clear_board,
            "komi": self.handle_komi,
            "play": self.handle_play,
            "genmove": self.handle_genmove,
            "undo": self.handle_undo,
            "quit": self.handle_quit,
            "showboard": self.handle_showboard,
        }

    def clear_board(self):
        """清空棋盤"""
        self.board = GoBoard(size=self.board_size)

    def gtp_to_coords(self, gtp_move: str) -> Optional[Tuple[int, int]]:
        """將 GTP 格式座標轉換為內部座標

        Parameters
        ----------
        gtp_move : str
            GTP 格式（如 "D4", "PASS"）

        Returns
        -------
        Optional[Tuple[int, int]]
            (row, col) 或 None (pass)
        """
        gtp_move = gtp_move.upper().strip()

        if gtp_move == "PASS":
            return None

        if len(gtp_move) < 2:
            return None

        # 列（跳過 I）
        col_labels = "ABCDEFGHJKLMNOPQRST"
        col_char = gtp_move[0]
        if col_char not in col_labels:
            return None

        col = col_labels.index(col_char)

        # 行
        try:
            row = int(gtp_move[1:]) - 1
        except ValueError:
            return None

        if 0 <= row < self.board_size and 0 <= col < self.board_size:
            return (row, col)

        return None

    def coords_to_gtp(self, coords: Optional[Tuple[int, int]]) -> str:
        """將內部座標轉換為 GTP 格式

        Parameters
        ----------
        coords : Optional[Tuple[int, int]]
            (row, col) 或 None (pass)

        Returns
        -------
        str
            GTP 格式（如 "D4", "PASS"）
        """
        if coords is None:
            return "PASS"

        row, col = coords
        col_labels = "ABCDEFGHJKLMNOPQRST"
        return f"{col_labels[col]}{row + 1}"

    # ----------------------------------------------------------------
    # GTP 命令處理器
    # ----------------------------------------------------------------

    def handle_protocol_version(self, args: List[str]) -> str:
        """返回 GTP 協議版本"""
        return "2"

    def handle_name(self, args: List[str]) -> str:
        """返回引擎名稱"""
        return "Light-Go"

    def handle_version(self, args: List[str]) -> str:
        """返回引擎版本"""
        return "0.1.0"

    def handle_known_command(self, args: List[str]) -> str:
        """檢查命令是否支持"""
        if not args:
            raise ValueError("missing command name")
        return "true" if args[0] in self.handlers else "false"

    def handle_list_commands(self, args: List[str]) -> str:
        """列出所有支持的命令"""
        return "\n".join(sorted(self.handlers.keys()))

    def handle_boardsize(self, args: List[str]) -> str:
        """設置棋盤大小"""
        if not args:
            raise ValueError("missing size")

        size = int(args[0])
        if size not in [9, 13, 19]:
            raise ValueError(f"unsupported board size: {size}")

        self.board_size = size
        self.clear_board()
        return ""

    def handle_clear_board(self, args: List[str]) -> str:
        """清空棋盤"""
        self.clear_board()
        return ""

    def handle_komi(self, args: List[str]) -> str:
        """設置貼目"""
        if not args:
            raise ValueError("missing komi")

        self.komi = float(args[0])
        return ""

    def handle_play(self, args: List[str]) -> str:
        """執行著手"""
        if len(args) < 2:
            raise ValueError("missing color or move")

        color = args[0].lower()
        if color not in ['b', 'w', 'black', 'white']:
            raise ValueError(f"invalid color: {color}")

        # 標準化顏色
        color = 'b' if color in ['b', 'black'] else 'w'

        # 解析著手
        coords = self.gtp_to_coords(args[1])

        if coords is None:
            # Pass
            self.board.pass_move(color)
        else:
            row, col = coords
            if not self.board.is_legal_move(row, col, color):
                raise ValueError(f"illegal move: {args[1]}")

            self.board.play_move(row, col, color)

        return ""

    def handle_genmove(self, args: List[str]) -> str:
        """生成著手（AI 下棋）"""
        if not args:
            raise ValueError("missing color")

        color = args[0].lower()
        if color not in ['b', 'w', 'black', 'white']:
            raise ValueError(f"invalid color: {color}")

        # 標準化顏色
        color = 'b' if color in ['b', 'black'] else 'w'

        try:
            # 使用 MCTS 搜索
            move_probs, root = self.mcts.search(self.board, color)

            # 選擇概率最高的合法著法
            legal_moves = []
            for move, prob in move_probs.items():
                if move == 'pass':
                    legal_moves.append(('pass', prob))
                else:
                    row, col = move
                    if self.board.is_legal_move(row, col, color):
                        legal_moves.append(((row, col), prob))

            if not legal_moves:
                # 無合法著法，pass
                self.board.pass_move(color)
                return "PASS"

            # 選擇最佳著法
            best_move, best_prob = max(legal_moves, key=lambda x: x[1])

            if best_move == 'pass':
                self.board.pass_move(color)
                return "PASS"

            row, col = best_move
            self.board.play_move(row, col, color)

            return self.coords_to_gtp((row, col))

        except Exception as e:
            logger.error(f"生成著手失敗：{e}", exc_info=True)
            # 發生錯誤時 pass
            self.board.pass_move(color)
            return "PASS"

    def handle_undo(self, args: List[str]) -> str:
        """撤銷上一手（目前不支持）"""
        raise ValueError("undo not implemented")

    def handle_quit(self, args: List[str]) -> str:
        """退出"""
        return ""

    def handle_showboard(self, args: List[str]) -> str:
        """顯示棋盤（調試用）"""
        lines = []
        for row in range(self.board_size - 1, -1, -1):
            line = f"{row + 1:2d} "
            for col in range(self.board_size):
                stone = self.board.board[row][col]
                if stone == 'b':
                    line += "X "
                elif stone == 'w':
                    line += "O "
                else:
                    line += ". "
            lines.append(line)

        col_labels = "ABCDEFGHJKLMNOPQRST"[:self.board_size]
        lines.append("   " + " ".join(col_labels))

        return "\n" + "\n".join(lines)

    def handle_command(self, command: str) -> Tuple[str, bool]:
        """處理 GTP 命令

        Parameters
        ----------
        command : str
            GTP 命令行

        Returns
        -------
        Tuple[str, bool]
            (響應, 是否退出)
        """
        command = command.strip()
        if not command or command.startswith("#"):
            return "", False

        # 解析命令
        tokens = command.split()
        if not tokens:
            return "? syntax error\n\n", False

        # 提取命令 ID（可選）
        cmd_id = None
        if tokens[0].isdigit():
            cmd_id = tokens.pop(0)

        if not tokens:
            return "? syntax error\n\n", False

        cmd_name = tokens[0]
        cmd_args = tokens[1:]

        # 查找處理器
        handler = self.handlers.get(cmd_name)

        if handler is None:
            response = self._format_response("?", cmd_id, "unknown command")
            return response + "\n\n", False

        # 執行命令
        try:
            result = handler(cmd_args)
            response = self._format_response("=", cmd_id, result)
            should_quit = (cmd_name == "quit")
            return response + "\n\n", should_quit

        except Exception as e:
            error_msg = str(e)
            response = self._format_response("?", cmd_id, error_msg)
            return response + "\n\n", False

    def _format_response(self, prefix: str, cmd_id: Optional[str], msg: str) -> str:
        """格式化 GTP 響應

        Parameters
        ----------
        prefix : str
            '=' 或 '?'
        cmd_id : Optional[str]
            命令 ID
        msg : str
            響應消息

        Returns
        -------
        str
            格式化的響應
        """
        parts = [prefix]
        if cmd_id:
            parts.append(cmd_id)
        if msg:
            parts.append(msg)
        return " ".join(parts).rstrip()


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
        模型文件路徑
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
        logger.info(f"模型已載入：{model_path}")
    else:
        if model_path:
            logger.warning(f"模型文件不存在：{model_path}")
        logger.info("使用未訓練模型")

    model.eval()
    return model


def run_gtp_server(engine: GTPEngine):
    """運行 GTP 服務器（標準輸入/輸出模式）

    Parameters
    ----------
    engine : GTPEngine
        GTP 引擎
    """
    logger.info("GTP 服務器已啟動（標準輸入/輸出模式）")
    logger.info("等待 GTP 命令...")

    try:
        while True:
            try:
                command = input()
                response, should_quit = engine.handle_command(command)
                print(response, end='', flush=True)

                if should_quit:
                    break

            except EOFError:
                break
            except KeyboardInterrupt:
                break

    except Exception as e:
        logger.error(f"服務器錯誤：{e}", exc_info=True)

    logger.info("GTP 服務器已關閉")


def main():
    """主函數"""
    parser = argparse.ArgumentParser(
        description='Light-Go GTP 服務器',
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
        '--mcts-simulations',
        type=int,
        default=200,
        help='MCTS 模擬次數（默認：200）'
    )
    parser.add_argument(
        '--log-level',
        type=str,
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='WARNING',
        help='日誌級別（默認：WARNING）'
    )

    args = parser.parse_args()

    # 配置日誌
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('gtp_server.log'),
            logging.StreamHandler(sys.stderr)  # GTP 響應使用 stdout，日誌使用 stderr
        ]
    )

    # 載入模型
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"使用設備：{device}")

    model_path = None if args.no_model else args.model_path
    model = load_model(
        model_path=model_path,
        num_blocks=args.num_blocks,
        num_filters=args.num_filters,
        device=device
    )

    # 創建 GTP 引擎
    engine = GTPEngine(
        model=model,
        mcts_simulations=args.mcts_simulations,
        use_optimized=True
    )

    # 運行服務器
    run_gtp_server(engine)


if __name__ == "__main__":
    main()

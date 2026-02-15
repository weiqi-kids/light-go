"""演化流水線

整合完整的演化循環：
1. 自我對弈生成訓練數據
2. 訓練所有架構
3. 評估適應度（錦標賽對弈）
4. 演化新世代
"""

from __future__ import annotations

import logging
from typing import List, Dict, Optional
from pathlib import Path
import time

import torch
import torch.optim as optim

from core.evolution_manager import EvolutionManager, Individual
from core.self_play import SelfPlayWorker, generate_training_data
from core.game_rules import GoBoard
from core.mcts import MCTS

logger = logging.getLogger(__name__)


class EvolutionPipeline:
    """完整的演化流水線

    Parameters
    ----------
    population_size : int
        種群大小
    num_simulations : int
        MCTS 模擬次數
    board_size : int
        棋盤大小
    save_dir : str
        保存目錄
    """

    def __init__(
        self,
        population_size: int = 10,
        num_simulations: int = 800,
        board_size: int = 19,
        save_dir: str = "data/evolution"
    ):
        self.population_size = population_size
        self.num_simulations = num_simulations
        self.board_size = board_size
        self.save_dir = Path(save_dir)

        # 創建演化管理器
        self.evolution_manager = EvolutionManager(
            population_size=population_size,
            board_size=board_size
        )

        # 訓練配置
        self.training_epochs = 5
        self.batch_size = 64
        self.learning_rate = 0.001

        # 評估配置
        self.eval_games_per_pair = 10  # 每對架構對弈場數

    def initialize(self) -> None:
        """初始化流水線"""
        logger.info("="*70)
        logger.info("  初始化演化流水線")
        logger.info("="*70)

        # 初始化種群
        self.evolution_manager.initialize_population(include_baseline=True)

        # 創建保存目錄
        self.save_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"✅ 流水線已初始化")
        logger.info(f"   種群大小：{self.population_size}")
        logger.info(f"   MCTS 模擬：{self.num_simulations}")
        logger.info(f"   保存目錄：{self.save_dir}")

    def generate_self_play_data(
        self,
        model,
        num_games: int = 100
    ) -> List:
        """生成自我對弈數據

        Parameters
        ----------
        model : torch.nn.Module
            當前最佳模型
        num_games : int
            對弈局數

        Returns
        -------
        List
            訓練樣本列表
        """
        logger.info(f"\n📊 生成自我對弈數據（{num_games} 局）...")

        start_time = time.time()

        examples = generate_training_data(
            model=model,
            num_games=num_games,
            num_simulations=self.num_simulations,
            board_size=self.board_size,
            verbose=False
        )

        elapsed = time.time() - start_time

        logger.info(f"✅ 生成完成")
        logger.info(f"   時間：{elapsed:.1f} 秒")
        logger.info(f"   樣本數：{len(examples)}")
        logger.info(f"   平均每局：{len(examples)/num_games:.1f} 樣本")

        return examples

    def train_individual(
        self,
        individual: Individual,
        training_data: List,
        device: str = 'cpu'
    ) -> None:
        """訓練單個架構

        Parameters
        ----------
        individual : Individual
            要訓練的個體
        training_data : List
            訓練數據
        device : str
            訓練裝置
        """
        logger.info(f"\n🔧 訓練架構 {individual.get_id()[:8]}...")

        # 創建模型
        model = individual.genome.to_pytorch_model(device=device)

        # 創建優化器
        optimizer = optim.Adam(
            model.parameters(),
            lr=self.learning_rate,
            weight_decay=1e-4
        )

        # 訓練循環
        model.train()
        total_loss = 0

        for epoch in range(self.training_epochs):
            epoch_loss = 0

            # 簡化：每個 epoch 只訓練一部分數據
            num_batches = min(len(training_data) // self.batch_size, 10)

            for batch_idx in range(num_batches):
                start_idx = batch_idx * self.batch_size
                end_idx = min(start_idx + self.batch_size, len(training_data))
                batch = training_data[start_idx:end_idx]

                # 準備批次數據
                features = torch.stack([
                    torch.from_numpy(ex.features) for ex in batch
                ]).to(device)

                policies = torch.stack([
                    torch.from_numpy(ex.policy) for ex in batch
                ]).to(device)

                values = torch.tensor([
                    ex.value for ex in batch
                ], dtype=torch.float32).to(device)

                # 前向傳播
                policy_logits, value_pred = model(features)

                # 計算損失
                policy_loss = torch.nn.functional.cross_entropy(
                    policy_logits,
                    policies
                )
                value_loss = torch.nn.functional.mse_loss(
                    value_pred.squeeze(-1),
                    values
                )
                loss = policy_loss + value_loss

                # 反向傳播
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            avg_loss = epoch_loss / num_batches if num_batches > 0 else 0
            total_loss += avg_loss

            logger.info(f"  Epoch {epoch+1}/{self.training_epochs}: Loss={avg_loss:.4f}")

        logger.info(f"✅ 訓練完成，平均 Loss={total_loss/self.training_epochs:.4f}")

        # 保存模型（簡化：只保存 state_dict）
        model_path = self.save_dir / f"model_{individual.get_id()}.pt"
        torch.save(model.state_dict(), model_path)

    def evaluate_pair(
        self,
        individual1: Individual,
        individual2: Individual,
        num_games: int = 10,
        device: str = 'cpu'
    ) -> float:
        """評估兩個架構的對弈勝率

        Parameters
        ----------
        individual1 : Individual
            個體 1
        individual2 : Individual
            個體 2
        num_games : int
            對弈局數
        device : str
            裝置

        Returns
        -------
        float
            個體 1 的勝率
        """
        logger.info(f"\n⚔️  對弈：{individual1.get_id()[:8]} vs {individual2.get_id()[:8]}")

        # 創建模型
        model1 = individual1.genome.to_pytorch_model(device=device)
        model2 = individual2.genome.to_pytorch_model(device=device)

        # 載入權重（如果存在）
        model1_path = self.save_dir / f"model_{individual1.get_id()}.pt"
        model2_path = self.save_dir / f"model_{individual2.get_id()}.pt"

        if model1_path.exists():
            model1.load_state_dict(torch.load(model1_path, map_location=device))
        if model2_path.exists():
            model2.load_state_dict(torch.load(model2_path, map_location=device))

        model1.eval()
        model2.eval()

        # 創建 MCTS 引擎
        mcts1 = MCTS(model=model1, num_simulations=self.num_simulations // 2)
        mcts2 = MCTS(model=model2, num_simulations=self.num_simulations // 2)

        # 對弈
        wins = 0

        for game_num in range(num_games):
            board = GoBoard(size=self.board_size)

            # 輪流先手
            if game_num % 2 == 0:
                black_mcts, white_mcts = mcts1, mcts2
                is_model1_black = True
            else:
                black_mcts, white_mcts = mcts2, mcts1
                is_model1_black = False

            current_mcts = black_mcts
            current_color = 'b'
            move_count = 0
            max_moves = 300

            # 對弈循環
            while not board.is_game_over() and move_count < max_moves:
                # 選擇著手
                move = current_mcts.select_move(board, current_color, temperature=0.1)

                # 下子
                success = board.play_move(move[0], move[1], current_color)

                if not success:
                    # 非法著手，判負
                    logger.warning(f"非法著手，{current_color} 判負")
                    break

                # 切換玩家
                current_color = 'w' if current_color == 'b' else 'b'
                current_mcts = white_mcts if current_mcts == black_mcts else black_mcts
                move_count += 1

            # 判斷勝負
            winner = board.get_winner()

            if winner == 'black' and is_model1_black:
                wins += 1
            elif winner == 'white' and not is_model1_black:
                wins += 1

        win_rate = wins / num_games

        logger.info(f"   結果：{wins}/{num_games} = {win_rate:.2%}")

        return win_rate

    def evaluate_population(self, device: str = 'cpu') -> None:
        """評估整個種群的適應度

        Parameters
        ----------
        device : str
            訓練裝置
        """
        logger.info(f"\n📊 評估種群適應度...")

        # 簡化：每個個體與基準模型對弈
        baseline = self.evolution_manager.population[0]

        for i, individual in enumerate(self.evolution_manager.population):
            if i == 0:
                # 基準模型與自己不對弈，給固定分數
                self.evolution_manager.evaluate_fitness(
                    individual,
                    win_rate=0.5,
                    elo_rating=1000.0
                )
                continue

            # 與基準對弈
            win_rate = self.evaluate_pair(
                individual,
                baseline,
                num_games=self.eval_games_per_pair,
                device=device
            )

            # 簡化 ELO 計算
            elo = 1000 + (win_rate - 0.5) * 400

            self.evolution_manager.evaluate_fitness(
                individual,
                win_rate=win_rate,
                elo_rating=elo
            )

        logger.info(f"✅ 種群評估完成")

    def run_evolution_cycle(
        self,
        num_generations: int = 5,
        games_per_generation: int = 100,
        device: str = 'cpu'
    ) -> None:
        """運行完整的演化循環

        Parameters
        ----------
        num_generations : int
            演化世代數
        games_per_generation : int
            每世代生成的對弈局數
        device : str
            訓練裝置
        """
        logger.info("\n" + "="*70)
        logger.info("  開始演化循環")
        logger.info("="*70)
        logger.info(f"  世代數：{num_generations}")
        logger.info(f"  每世代對弈：{games_per_generation} 局")
        logger.info("="*70)

        for gen in range(num_generations):
            logger.info(f"\n{'#'*70}")
            logger.info(f"# 世代 {gen + 1}/{num_generations}")
            logger.info('#'*70)

            # 1. 獲取當前最佳模型
            best = self.evolution_manager.get_best_individual()
            if best is None:
                best = self.evolution_manager.population[0]

            best_model = best.genome.to_pytorch_model(device=device)
            model_path = self.save_dir / f"model_{best.get_id()}.pt"
            if model_path.exists():
                best_model.load_state_dict(torch.load(model_path, map_location=device))

            # 2. 生成自我對弈數據
            training_data = self.generate_self_play_data(
                best_model,
                num_games=games_per_generation
            )

            # 3. 訓練所有個體
            logger.info(f"\n🔧 訓練種群...")
            for individual in self.evolution_manager.population:
                self.train_individual(individual, training_data, device=device)

            # 4. 評估適應度
            self.evaluate_population(device=device)

            # 5. 演化新世代
            if gen < num_generations - 1:
                self.evolution_manager.evolve_generation()

            # 6. 保存進度
            self.evolution_manager.save_population(
                self.save_dir / f"generation_{self.evolution_manager.generation}"
            )

            # 7. 報告
            best = self.evolution_manager.get_best_individual()
            diversity = self.evolution_manager.get_population_diversity()

            logger.info(f"\n📊 世代 {gen + 1} 總結：")
            logger.info(f"   最佳個體：{best.get_id()[:8]}")
            logger.info(f"   最佳分數：{best.fitness.total_score:.3f}")
            logger.info(f"   勝率：{best.fitness.win_rate:.2%}")
            logger.info(f"   種群多樣性：{diversity:.2%}")

        logger.info("\n" + "="*70)
        logger.info("  ✅ 演化循環完成！")
        logger.info("="*70)

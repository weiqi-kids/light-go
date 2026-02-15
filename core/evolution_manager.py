"""架構演化管理器

實現神經架構搜索（NAS）的演化算法：
- 種群管理
- 適應度評估
- 選擇、突變、雜交
- 多目標優化（性能 vs 效率）
"""

from __future__ import annotations

import random
import logging
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path
import pickle
import json

from core.architecture_genome import ArchitectureGenome

logger = logging.getLogger(__name__)


@dataclass
class FitnessScore:
    """架構的適應度分數

    Attributes
    ----------
    win_rate : float
        對弈勝率 (0.0-1.0)
    elo_rating : float
        ELO 評級
    parameters : int
        參數數量
    inference_time : float
        推理時間（秒）
    efficiency_score : float
        效率分數（勝率 / 參數量的某種函數）
    """

    win_rate: float = 0.0
    elo_rating: float = 1000.0
    parameters: int = 0
    inference_time: float = 0.0

    @property
    def efficiency_score(self) -> float:
        """計算效率分數（考慮性能和參數量）"""
        # 歸一化參數量（假設 1M-10M 參數範圍）
        normalized_params = min(self.parameters / 10_000_000, 1.0)
        # 效率 = 勝率 / 參數量（鼓勵小而強的模型）
        if self.parameters > 0:
            return self.win_rate / (normalized_params + 0.1)
        return 0.0

    @property
    def total_score(self) -> float:
        """綜合分數

        70% 勝率 + 20% 效率 + 10% ELO
        """
        normalized_elo = (self.elo_rating - 1000) / 500  # 假設 1000-1500 範圍
        return (
            0.7 * self.win_rate +
            0.2 * min(self.efficiency_score, 1.0) +
            0.1 * max(0, normalized_elo)
        )


@dataclass
class Individual:
    """進化種群中的個體

    Attributes
    ----------
    genome : ArchitectureGenome
        架構基因組
    fitness : Optional[FitnessScore]
        適應度分數
    generation : int
        世代編號
    """

    genome: ArchitectureGenome
    fitness: Optional[FitnessScore] = None
    generation: int = 0

    def get_id(self) -> str:
        """獲取個體 ID"""
        return self.genome.get_id()


class EvolutionManager:
    """架構演化管理器

    Parameters
    ----------
    population_size : int
        種群大小（默認 10）
    elite_ratio : float
        精英保留比例（默認 0.2）
    mutation_rate : float
        突變率（默認 0.3）
    board_size : int
        棋盤大小（默認 19）
    """

    def __init__(
        self,
        population_size: int = 10,
        elite_ratio: float = 0.2,
        mutation_rate: float = 0.3,
        board_size: int = 19
    ):
        self.population_size = population_size
        self.elite_ratio = elite_ratio
        self.mutation_rate = mutation_rate
        self.board_size = board_size

        self.population: List[Individual] = []
        self.generation = 0
        self.history: List[Dict] = []

    def initialize_population(self, include_baseline: bool = True) -> None:
        """初始化種群

        Parameters
        ----------
        include_baseline : bool
            是否包含基準架構
        """
        logger.info(f"初始化種群（大小：{self.population_size}）")

        self.population = []

        # 添加基準架構
        if include_baseline:
            baseline_genome = ArchitectureGenome.create_baseline(self.board_size)
            self.population.append(Individual(
                genome=baseline_genome,
                generation=0
            ))

        # 添加隨機變體
        remaining = self.population_size - len(self.population)
        for _ in range(remaining):
            if random.random() < 0.5:
                # 基準的突變體
                genome = ArchitectureGenome.create_baseline(self.board_size)
                genome = genome.mutate(mutation_rate=0.3)
            else:
                # 完全隨機
                genome = ArchitectureGenome.create_random(self.board_size)

            self.population.append(Individual(
                genome=genome,
                generation=0
            ))

        logger.info(f"✅ 種群已初始化：{len(self.population)} 個個體")

    def evaluate_fitness(
        self,
        individual: Individual,
        win_rate: float,
        elo_rating: float = 1000.0,
        inference_time: float = 0.0
    ) -> None:
        """評估個體適應度

        Parameters
        ----------
        individual : Individual
            要評估的個體
        win_rate : float
            對弈勝率
        elo_rating : float
            ELO 評級
        inference_time : float
            推理時間
        """
        individual.fitness = FitnessScore(
            win_rate=win_rate,
            elo_rating=elo_rating,
            parameters=individual.genome.estimate_parameters(),
            inference_time=inference_time
        )

        logger.info(
            f"個體 {individual.get_id()[:8]} 適應度："
            f"勝率={win_rate:.2%}, ELO={elo_rating:.0f}, "
            f"參數={individual.fitness.parameters:,}, "
            f"總分={individual.fitness.total_score:.3f}"
        )

    def select_parents(self, num_parents: int = 2) -> List[Individual]:
        """選擇父代個體（錦標賽選擇）

        Parameters
        ----------
        num_parents : int
            要選擇的父代數量

        Returns
        -------
        List[Individual]
            選中的父代
        """
        # 過濾出已評估的個體
        evaluated = [ind for ind in self.population if ind.fitness is not None]

        if len(evaluated) < num_parents:
            logger.warning(f"可用個體不足：需要 {num_parents}，只有 {len(evaluated)}")
            return random.sample(self.population, min(num_parents, len(self.population)))

        # 錦標賽選擇
        parents = []
        tournament_size = 3

        for _ in range(num_parents):
            # 隨機選擇幾個競爭者
            competitors = random.sample(evaluated, min(tournament_size, len(evaluated)))
            # 選擇最佳者
            winner = max(competitors, key=lambda ind: ind.fitness.total_score)
            parents.append(winner)

        return parents

    def create_offspring(
        self,
        parent1: Individual,
        parent2: Individual
    ) -> Individual:
        """創建後代（雜交 + 突變）

        Parameters
        ----------
        parent1, parent2 : Individual
            父代個體

        Returns
        -------
        Individual
            後代個體
        """
        # 雜交
        child_genome = ArchitectureGenome.crossover(
            parent1.genome,
            parent2.genome
        )

        # 突變
        if random.random() < self.mutation_rate:
            child_genome = child_genome.mutate(mutation_rate=0.2)

        return Individual(
            genome=child_genome,
            generation=self.generation + 1
        )

    def evolve_generation(self) -> List[Individual]:
        """演化一個世代

        Returns
        -------
        List[Individual]
            新世代種群
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"開始演化第 {self.generation + 1} 代")
        logger.info('='*60)

        # 確保所有個體都已評估
        unevaluated = [ind for ind in self.population if ind.fitness is None]
        if unevaluated:
            logger.warning(f"⚠️  有 {len(unevaluated)} 個個體未評估")

        # 排序種群（按適應度）
        evaluated = [ind for ind in self.population if ind.fitness is not None]
        evaluated.sort(key=lambda ind: ind.fitness.total_score, reverse=True)

        # 記錄最佳個體
        if evaluated:
            best = evaluated[0]
            logger.info(
                f"當前最佳：{best.get_id()[:8]} "
                f"(總分={best.fitness.total_score:.3f}, "
                f"勝率={best.fitness.win_rate:.2%})"
            )

        # 精英保留
        num_elite = max(1, int(self.population_size * self.elite_ratio))
        elite = evaluated[:num_elite]
        logger.info(f"精英保留：{num_elite} 個")

        # 創建新個體
        new_population = elite.copy()

        while len(new_population) < self.population_size:
            # 選擇父代
            parents = self.select_parents(num_parents=2)

            # 創建後代
            offspring = self.create_offspring(parents[0], parents[1])
            new_population.append(offspring)

        # 更新種群
        self.population = new_population
        self.generation += 1

        # 記錄歷史
        self.history.append({
            'generation': self.generation,
            'population_size': len(self.population),
            'best_fitness': best.fitness.total_score if evaluated else 0.0,
            'best_id': best.get_id() if evaluated else None
        })

        logger.info(f"✅ 第 {self.generation} 代演化完成")
        logger.info(f"   新種群大小：{len(self.population)}")

        return self.population

    def get_best_individual(self) -> Optional[Individual]:
        """獲取最佳個體

        Returns
        -------
        Optional[Individual]
            最佳個體，如果種群為空則返回 None
        """
        evaluated = [ind for ind in self.population if ind.fitness is not None]
        if not evaluated:
            return None

        return max(evaluated, key=lambda ind: ind.fitness.total_score)

    def get_population_diversity(self) -> float:
        """計算種群多樣性

        Returns
        -------
        float
            多樣性分數（0-1）
        """
        if len(self.population) < 2:
            return 0.0

        # 比較所有個體對的基因差異
        total_diff = 0
        comparisons = 0

        for i in range(len(self.population)):
            for j in range(i + 1, len(self.population)):
                g1 = self.population[i].genome
                g2 = self.population[j].genome

                # 計算差異
                diff = 0
                if g1.num_blocks != g2.num_blocks:
                    diff += 1
                if g1.base_filters != g2.base_filters:
                    diff += 1
                if g1.use_se_blocks != g2.use_se_blocks:
                    diff += 1

                total_diff += diff
                comparisons += 1

        if comparisons == 0:
            return 0.0

        # 歸一化（最大差異為 3）
        avg_diff = total_diff / comparisons
        return min(avg_diff / 3.0, 1.0)

    def save_population(self, save_dir: str) -> None:
        """保存種群

        Parameters
        ----------
        save_dir : str
            保存目錄
        """
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        # 保存每個個體的基因組
        for i, individual in enumerate(self.population):
            genome_path = save_path / f"individual_{i}_{individual.get_id()}.pkl"
            with open(genome_path, 'wb') as f:
                pickle.dump(individual.genome, f)

        # 保存種群元數據
        metadata = {
            'generation': self.generation,
            'population_size': len(self.population),
            'history': self.history,
            'individuals': [
                {
                    'id': ind.get_id(),
                    'generation': ind.generation,
                    'fitness': {
                        'win_rate': ind.fitness.win_rate,
                        'elo_rating': ind.fitness.elo_rating,
                        'parameters': ind.fitness.parameters,
                        'total_score': ind.fitness.total_score
                    } if ind.fitness else None
                }
                for ind in self.population
            ]
        }

        metadata_path = save_path / 'population_metadata.json'
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        logger.info(f"✅ 種群已保存：{save_path}")

    def load_population(self, load_dir: str) -> None:
        """載入種群

        Parameters
        ----------
        load_dir : str
            載入目錄
        """
        load_path = Path(load_dir)

        # 載入元數據
        metadata_path = load_path / 'population_metadata.json'
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)

        self.generation = metadata['generation']
        self.history = metadata['history']

        # 載入基因組
        self.population = []
        genome_files = list(load_path.glob('individual_*.pkl'))

        for genome_file in genome_files:
            with open(genome_file, 'rb') as f:
                genome = pickle.load(f)

            # 從元數據中找到對應的適應度
            genome_id = genome.get_id()
            ind_metadata = next(
                (ind for ind in metadata['individuals'] if ind['id'] == genome_id),
                None
            )

            if ind_metadata and ind_metadata['fitness']:
                fitness = FitnessScore(**ind_metadata['fitness'])
            else:
                fitness = None

            individual = Individual(
                genome=genome,
                fitness=fitness,
                generation=ind_metadata['generation'] if ind_metadata else 0
            )

            self.population.append(individual)

        logger.info(f"✅ 種群已載入：{len(self.population)} 個個體")

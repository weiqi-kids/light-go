# 第五課：完整的演化策略

## 課程目標

學習如何組合突變和雜交，建立一個完整的演化系統，包括選擇、繁殖、替換等步驟，以及多樣性維持策略。

## 演化循環總覽

演化是一個持續的循環過程：

```
┌─────────────────────────────────────┐
│  1. 評估（Evaluation）              │
│     • 對弈測試                      │
│     • 計算勝率                      │
│     • 多目標評分                    │
└──────────────┬──────────────────────┘
               ↓
┌─────────────────────────────────────┐
│  2. 選擇（Selection）               │
│     • 精英保留                      │
│     • 錦標賽選擇                    │
│     • 淘汰弱者                      │
└──────────────┬──────────────────────┘
               ↓
┌─────────────────────────────────────┐
│  3. 繁殖（Reproduction）            │
│     • 雜交（Crossover）             │
│     • 突變（Mutation）              │
│     • 隨機注入                      │
└──────────────┬──────────────────────┘
               ↓
┌─────────────────────────────────────┐
│  4. 替換（Replacement）             │
│     • 精英 + 子代組成新世代         │
│     • 保持種群大小                  │
│     • 更新世代編號                  │
└──────────────┬──────────────────────┘
               ↓
             重複...
```

## 1. 評估階段（Evaluation）

### 單目標評估：勝率

最直接的方法是讓架構對弈，計算勝率：

```python
def evaluate_simple(genome, opponent_genome, num_games=100):
    """簡單評估：勝率"""
    model = genome.to_pytorch_model()
    opponent_model = opponent_genome.to_pytorch_model()

    wins = 0
    for _ in range(num_games):
        result = play_game(model, opponent_model)
        if result == 1:  # 勝利
            wins += 1

    win_rate = wins / num_games
    return win_rate
```

### 多目標評估：性能 + 效率 + 新穎性

實際中，我們需要平衡多個目標：

```python
def evaluate_multi_objective(genome, population, num_games=100):
    """
    多目標評估

    Returns:
        score: 綜合分數（0-1）
    """
    # 1. 性能評估（70% 權重）
    win_rate = evaluate_vs_population(genome, population, num_games)
    performance_score = win_rate

    # 2. 效率評估（20% 權重）
    params = genome.estimate_parameters()
    max_params = 50_000_000  # 50M 參數為上限
    efficiency_score = max(0, 1 - params / max_params)

    # 3. 新穎性評估（10% 權重）
    novelty_score = compute_novelty(genome, population)

    # 綜合得分
    total_score = (
        0.7 * performance_score +
        0.2 * efficiency_score +
        0.1 * novelty_score
    )

    return total_score
```

### 新穎性計算

獎勵與其他架構不同的基因組，避免過早收斂：

```python
def compute_novelty(genome, population):
    """
    計算基因組的新穎性分數

    思路：與種群中所有個體計算距離，距離越大越新穎
    """
    distances = []

    for other in population:
        if other.get_id() == genome.get_id():
            continue

        # 計算架構距離
        distance = compute_genome_distance(genome, other)
        distances.append(distance)

    # 平均距離作為新穎性
    if not distances:
        return 0.5

    avg_distance = sum(distances) / len(distances)
    # 歸一化到 [0, 1]
    novelty_score = min(1.0, avg_distance / 10.0)

    return novelty_score


def compute_genome_distance(genome1, genome2):
    """計算兩個基因組的距離"""
    distance = 0.0

    # 深度差異
    distance += abs(genome1.num_blocks - genome2.num_blocks) / 20.0

    # 寬度差異
    distance += abs(genome1.base_filters - genome2.base_filters) / 512.0

    # SE blocks 差異
    if genome1.use_se_blocks != genome2.use_se_blocks:
        distance += 1.0

    # Dropout 差異
    distance += abs(genome1.dropout_rate - genome2.dropout_rate)

    return distance
```

## 2. 選擇階段（Selection）

### 精英保留（Elitism）

保證最好的個體不會丟失：

```python
def select_elite(population, scores, elite_ratio=0.2):
    """
    精英保留

    Args:
        population: 種群
        scores: 每個個體的評分
        elite_ratio: 保留前多少比例

    Returns:
        elite: 精英個體列表
    """
    # 按分數排序
    sorted_pop = sorted(
        zip(population, scores),
        key=lambda x: x[1],
        reverse=True
    )

    # 保留前 elite_ratio
    elite_count = max(1, int(len(population) * elite_ratio))
    elite = [genome for genome, score in sorted_pop[:elite_count]]

    return elite
```

### 錦標賽選擇（Tournament Selection）

隨機配對競爭，勝者進入下一輪：

```python
def tournament_selection(population, scores, tournament_size=3, num_parents=10):
    """
    錦標賽選擇

    Args:
        tournament_size: 每次錦標賽的參與者數量
        num_parents: 需要選擇多少個父代

    Returns:
        selected_parents: 被選中的父代列表
    """
    selected = []

    for _ in range(num_parents):
        # 隨機抽取 tournament_size 個個體
        tournament_indices = random.sample(range(len(population)), tournament_size)
        tournament = [(population[i], scores[i]) for i in tournament_indices]

        # 選擇分數最高者
        winner = max(tournament, key=lambda x: x[1])[0]
        selected.append(winner)

    return selected
```

### 輪盤賭選擇（Roulette Wheel Selection）

按分數比例選擇，分數越高被選中機率越大：

```python
def roulette_selection(population, scores, num_parents=10):
    """
    輪盤賭選擇

    分數越高，被選中的概率越大
    """
    # 計算總分數
    total_score = sum(scores)

    if total_score == 0:
        # 如果所有分數都是 0，均勻隨機選擇
        return random.sample(population, num_parents)

    # 計算每個個體的選擇概率
    probabilities = [score / total_score for score in scores]

    # 按概率選擇
    selected = random.choices(
        population,
        weights=probabilities,
        k=num_parents
    )

    return selected
```

## 3. 繁殖階段（Reproduction）

### 雜交策略

```python
def create_offspring_by_crossover(parents, num_offspring):
    """
    通過雜交創建子代

    Args:
        parents: 父代列表
        num_offspring: 需要創建多少個子代

    Returns:
        offspring: 子代列表
    """
    offspring = []

    for _ in range(num_offspring):
        # 隨機選擇兩個不同的父代
        parent1, parent2 = random.sample(parents, 2)

        # 雜交
        child = ArchitectureGenome.crossover(parent1, parent2)
        offspring.append(child)

    return offspring
```

### 突變策略

```python
def create_offspring_by_mutation(parents, num_offspring, mutation_rate=0.2):
    """
    通過突變創建子代

    Args:
        parents: 父代列表
        num_offspring: 需要創建多少個子代
        mutation_rate: 突變率

    Returns:
        offspring: 子代列表
    """
    offspring = []

    for _ in range(num_offspring):
        # 隨機選擇一個父代
        parent = random.choice(parents)

        # 突變
        child = parent.mutate(mutation_rate=mutation_rate)
        offspring.append(child)

    return offspring
```

### 混合繁殖

結合雜交和突變：

```python
def create_offspring_hybrid(parents, num_offspring, crossover_rate=0.5, mutation_rate=0.2):
    """
    混合繁殖：雜交 + 突變

    Args:
        crossover_rate: 雜交比例（剩餘為突變）
        mutation_rate: 突變率
    """
    offspring = []

    num_crossover = int(num_offspring * crossover_rate)
    num_mutation = num_offspring - num_crossover

    # 雜交
    offspring.extend(create_offspring_by_crossover(parents, num_crossover))

    # 突變
    offspring.extend(create_offspring_by_mutation(parents, num_mutation, mutation_rate))

    return offspring
```

### 隨機注入

定期注入全新的隨機個體，防止種群多樣性喪失：

```python
def inject_random_genomes(num_random=2, board_size=19):
    """
    注入隨機基因組

    用於：
    - 防止過早收斂
    - 探索全新的架構空間
    - 跳出局部最優
    """
    random_genomes = []

    for _ in range(num_random):
        genome = ArchitectureGenome(
            num_blocks=random.randint(3, 15),
            base_filters=random.choice([64, 96, 128, 192, 256, 384]),
            kernel_sizes=[random.choice([3, 5, 7]) for _ in range(random.randint(3, 15))],
            use_se_blocks=random.choice([True, False]),
            dropout_rate=random.choice([0.0, 0.1, 0.2, 0.3]),
            board_size=board_size
        )
        random_genomes.append(genome)

    return random_genomes
```

## 4. 替換階段（Replacement）

### 世代替換（Generational Replacement）

舊世代完全被新世代替換：

```python
def generational_replacement(elite, offspring):
    """
    世代替換

    新世代 = 精英 + 子代
    """
    new_generation = elite + offspring
    return new_generation
```

### 穩定狀態替換（Steady-State Replacement）

每次只替換少數個體：

```python
def steady_state_replacement(population, scores, offspring, num_replace):
    """
    穩定狀態替換

    Args:
        num_replace: 每次替換多少個個體

    策略：淘汰分數最低的 num_replace 個個體
    """
    # 按分數排序
    sorted_pop = sorted(
        zip(population, scores),
        key=lambda x: x[1],
        reverse=True
    )

    # 保留前 (N - num_replace) 個
    survivors = [genome for genome, score in sorted_pop[:-num_replace]]

    # 加入新子代
    new_population = survivors + offspring[:num_replace]

    return new_population
```

## 完整演化循環實現

```python
from core.architecture_genome import ArchitectureGenome, create_initial_population

def evolve_population(
    population_size=20,
    num_generations=50,
    elite_ratio=0.2,
    mutation_rate=0.2,
    board_size=19
):
    """
    完整的演化循環

    Args:
        population_size: 種群大小
        num_generations: 演化世代數
        elite_ratio: 精英保留比例
        mutation_rate: 突變率

    Returns:
        best_genome: 最佳架構
        history: 演化歷史
    """
    # 1. 創建初始種群
    population = create_initial_population(
        size=population_size,
        board_size=board_size,
        include_baseline=True
    )

    history = {
        'best_scores': [],
        'avg_scores': [],
        'diversity': []
    }

    # 演化循環
    for generation in range(num_generations):
        print(f"\n{'='*60}")
        print(f"世代 {generation}")
        print('='*60)

        # 2. 評估
        print("評估中...")
        scores = []
        for genome in population:
            score = evaluate_multi_objective(genome, population)
            scores.append(score)

        # 記錄統計
        best_score = max(scores)
        avg_score = sum(scores) / len(scores)
        diversity = compute_population_diversity(population)

        history['best_scores'].append(best_score)
        history['avg_scores'].append(avg_score)
        history['diversity'].append(diversity)

        print(f"最佳分數: {best_score:.3f}")
        print(f"平均分數: {avg_score:.3f}")
        print(f"種群多樣性: {diversity:.3f}")

        # 3. 選擇
        print("選擇中...")
        elite = select_elite(population, scores, elite_ratio=elite_ratio)
        parents = tournament_selection(population, scores, num_parents=10)

        print(f"精英數量: {len(elite)}")
        print(f"父代數量: {len(parents)}")

        # 4. 繁殖
        print("繁殖中...")
        num_offspring = population_size - len(elite)

        # 80% 雜交 + 20% 突變
        offspring = create_offspring_hybrid(
            parents,
            num_offspring,
            crossover_rate=0.8,
            mutation_rate=mutation_rate
        )

        # 每 10 代注入 2 個隨機個體
        if generation % 10 == 0:
            random_genomes = inject_random_genomes(num_random=2, board_size=board_size)
            # 替換最差的 2 個子代
            offspring = offspring[:-2] + random_genomes

        print(f"子代數量: {len(offspring)}")

        # 5. 替換
        population = generational_replacement(elite, offspring)

        # 顯示前 3 名
        top_3 = sorted(
            zip(population, scores),
            key=lambda x: x[1],
            reverse=True
        )[:3]

        print("\n前 3 名架構:")
        for i, (genome, score) in enumerate(top_3):
            print(f"  {i+1}. 分數={score:.3f}, {genome}")

    # 返回最佳個體
    final_scores = [evaluate_multi_objective(g, population) for g in population]
    best_idx = final_scores.index(max(final_scores))
    best_genome = population[best_idx]

    print(f"\n{'='*60}")
    print("演化完成！")
    print('='*60)
    print(f"最佳架構: {best_genome}")
    print(f"最終分數: {max(final_scores):.3f}")

    return best_genome, history


def compute_population_diversity(population):
    """計算種群多樣性"""
    if len(population) < 2:
        return 0.0

    total_distance = 0.0
    count = 0

    for i in range(len(population)):
        for j in range(i + 1, len(population)):
            distance = compute_genome_distance(population[i], population[j])
            total_distance += distance
            count += 1

    avg_distance = total_distance / count if count > 0 else 0.0
    return avg_distance
```

## 多樣性維持策略

### 1. Novelty Search（新穎性搜索）

獎勵不同的架構，而不僅僅是性能好的：

```python
def novelty_search_selection(population, scores, novelty_scores, novelty_weight=0.3):
    """
    結合性能和新穎性的選擇

    Args:
        novelty_weight: 新穎性權重（0-1）
    """
    combined_scores = []

    for i in range(len(population)):
        combined = (
            (1 - novelty_weight) * scores[i] +
            novelty_weight * novelty_scores[i]
        )
        combined_scores.append(combined)

    return combined_scores
```

### 2. Speciation（物種形成）

維護多個亞群，每個亞群獨立演化：

```python
def speciation(population, num_species=5):
    """
    將種群分成多個物種

    策略：基於架構相似度聚類
    """
    from sklearn.cluster import KMeans

    # 提取特徵
    features = []
    for genome in population:
        feature = [
            genome.num_blocks / 20.0,
            genome.base_filters / 512.0,
            1.0 if genome.use_se_blocks else 0.0,
            genome.dropout_rate
        ]
        features.append(feature)

    # K-means 聚類
    kmeans = KMeans(n_clusters=num_species, random_state=42)
    labels = kmeans.fit_predict(features)

    # 分組
    species = [[] for _ in range(num_species)]
    for genome, label in zip(population, labels):
        species[label].append(genome)

    return species


def evolve_with_speciation(population_size=20, num_species=4, num_generations=50):
    """
    物種化演化

    每個物種內部獨立演化，定期交換個體
    """
    population = create_initial_population(population_size)

    for generation in range(num_generations):
        # 分成物種
        species = speciation(population, num_species=num_species)

        new_population = []

        # 每個物種獨立演化
        for species_pop in species:
            if not species_pop:
                continue

            # 評估
            scores = [evaluate_multi_objective(g, species_pop) for g in species_pop]

            # 選擇、繁殖
            elite = select_elite(species_pop, scores, elite_ratio=0.2)
            offspring_size = len(species_pop) - len(elite)
            offspring = create_offspring_hybrid(elite, offspring_size)

            new_population.extend(elite + offspring)

        # 物種間交流：每個物種貢獻 1 個精英到其他物種
        # （可選實現）

        population = new_population[:population_size]

    return population
```

### 3. 自適應突變率

根據演化進展調整突變率：

```python
def adaptive_mutation_rate(generation, max_generations, performance_stagnation):
    """
    自適應突變率

    Args:
        generation: 當前世代
        max_generations: 總世代數
        performance_stagnation: 性能停滯的世代數

    Returns:
        mutation_rate: 調整後的突變率
    """
    # 基礎策略：早期高，後期低
    base_rate = 0.5 * (1 - generation / max_generations) + 0.1

    # 如果性能停滯，提高突變率跳出局部最優
    if performance_stagnation > 5:
        base_rate *= 1.5

    # 限制範圍
    mutation_rate = max(0.05, min(0.5, base_rate))

    return mutation_rate
```

## 演化效果可視化

```python
import matplotlib.pyplot as plt

def plot_evolution_history(history):
    """繪製演化歷史"""
    generations = range(len(history['best_scores']))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # 1. 分數變化
    axes[0].plot(generations, history['best_scores'], label='Best', linewidth=2)
    axes[0].plot(generations, history['avg_scores'], label='Average', alpha=0.7)
    axes[0].set_xlabel('Generation')
    axes[0].set_ylabel('Score')
    axes[0].set_title('Performance Evolution')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # 2. 多樣性變化
    axes[1].plot(generations, history['diversity'], color='green', linewidth=2)
    axes[1].set_xlabel('Generation')
    axes[1].set_ylabel('Diversity')
    axes[1].set_title('Population Diversity')
    axes[1].grid(True, alpha=0.3)

    # 3. 改進率
    improvements = [0]
    for i in range(1, len(history['best_scores'])):
        improvement = history['best_scores'][i] - history['best_scores'][i-1]
        improvements.append(improvement)

    axes[2].bar(generations, improvements, alpha=0.7)
    axes[2].axhline(y=0, color='r', linestyle='--', alpha=0.5)
    axes[2].set_xlabel('Generation')
    axes[2].set_ylabel('Improvement')
    axes[2].set_title('Generation-over-Generation Improvement')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()
```

## 實踐練習

### 練習 1：運行簡單演化

```python
# 小規模演化測試
best_genome, history = evolve_population(
    population_size=10,
    num_generations=20,
    elite_ratio=0.2,
    mutation_rate=0.2
)

plot_evolution_history(history)
```

### 練習 2：比較不同選擇策略

```python
# 比較精英保留 vs 錦標賽選擇
strategies = [
    ('Elitism only', lambda pop, scores: select_elite(pop, scores, 0.3)),
    ('Tournament only', lambda pop, scores: tournament_selection(pop, scores, num_parents=len(pop)//2)),
    ('Hybrid', lambda pop, scores: select_elite(pop, scores, 0.2) + tournament_selection(pop, scores, num_parents=len(pop)//5))
]

for name, strategy in strategies:
    # 運行演化...
    print(f"{name}: best_score = ...")
```

### 練習 3：分析多樣性影響

```python
# 高多樣性 vs 低多樣性
configs = [
    {'random_injection': True, 'novelty_weight': 0.3},   # 高多樣性
    {'random_injection': False, 'novelty_weight': 0.0},  # 低多樣性
]

for config in configs:
    # 運行演化...
    # 比較最終性能和收斂速度
```

## 重點總結

### 演化四階段

1. **評估**：多目標評分（性能 + 效率 + 新穎性）
2. **選擇**：精英保留 + 錦標賽
3. **繁殖**：雜交（80%）+ 突變（20%）
4. **替換**：精英 + 子代

### 關鍵策略

- **精英保留**：保證不會退化
- **多目標優化**：平衡性能、效率、多樣性
- **隨機注入**：防止過早收斂
- **自適應參數**：根據進展調整策略

### 成功指標

- ✅ 最佳分數持續提升
- ✅ 種群多樣性維持在合理範圍
- ✅ 發現多種有效架構
- ✅ 沒有過早收斂

## 下一課預告

在[第六課：為什麼架構演化是革命性的創新](lesson-6-why-revolutionary.md)中，您將了解：

- 傳統 AI 與 Light-Go 的根本區別
- 自動架構發現的革命性意義
- Meta-Learning（元學習）的概念
- 長期演化的願景

演化系統已經建立，下一課將深入理解**為什麼這是革命性的**！

---

[← 上一課：基因組轉模型](lesson-4-genome-to-model.md) | [返回大綱](index.md) | [下一課：革命性創新 →](lesson-6-why-revolutionary.md)

# 第三課：雜交 - 結合父代的優勢

## 課程目標

學習雜交（Crossover）機制，了解如何結合兩個成功架構的優點創造新架構，以及如何追蹤世代和血統。

## 什麼是雜交？

雜交（Crossover）結合兩個父代的基因，創造新的架構。這類似生物的有性繁殖，可以組合兩個成功架構的優點。

```python
from core.architecture_genome import ArchitectureGenome

# 兩個父代
parent1 = ArchitectureGenome.create_baseline(board_size=19)
parent2 = ArchitectureGenome(
    num_blocks=8,
    base_filters=256,
    board_size=19
)

# 雜交！
child = ArchitectureGenome.crossover(parent1, parent2)

print(f"父代 1: {parent1}")
print(f"父代 2: {parent2}")
print(f"子代:   {child}")
```

## 為什麼需要雜交？

### 突變 vs 雜交

| 特性 | 突變 | 雜交 |
|------|------|------|
| **機制** | 隨機改變單個基因 | 組合兩個父代 |
| **探索方式** | 局部搜索 | 全局組合 |
| **創新性** | 漸進式改進 | 跳躍式創新 |
| **風險** | 低（小步變化）| 中（可能不兼容）|
| **速度** | 慢 | 快 |

### 實際例子

```python
# 父代 1：深而窄（擅長精細計算）
parent1 = ArchitectureGenome(
    num_blocks=15,
    base_filters=96,
    use_se_blocks=True,
    board_size=19
)

# 父代 2：淺而寬（擅長快速評估）
parent2 = ArchitectureGenome(
    num_blocks=6,
    base_filters=384,
    dropout_rate=0.2,
    board_size=19
)

# 雜交：可能產生「中等深度 + 中等寬度 + SE + Dropout」
# 組合兩者優點！
child = ArchitectureGenome.crossover(parent1, parent2)

print(f"子代深度: {child.num_blocks}")      # 可能是 10 左右
print(f"子代寬度: {child.base_filters}")    # 可能是 96 或 384
print(f"SE Blocks: {child.use_se_blocks}")  # 可能繼承自父代 1
print(f"Dropout: {child.dropout_rate}")     # 可能繼承自父代 2
```

## 雜交策略詳解

### 1. 深度繼承（平均法）

**策略**：取兩個父代深度的平均值（四捨五入）

```python
parent1_blocks = 5
parent2_blocks = 8

child_blocks = (parent1_blocks + parent2_blocks) // 2
# child_blocks = 6
```

**為什麼平均？**
- ✅ 平衡兩個父代的特性
- ✅ 避免過深或過淺
- ✅ 保持漸進性（不會突然跳很多層）

**實際範例**：
```python
# 淺父代 + 深父代 = 中等子代
parent1: num_blocks = 4
parent2: num_blocks = 12
child:   num_blocks = 8   # (4 + 12) // 2

# 相近父代 = 相近子代
parent1: num_blocks = 7
parent2: num_blocks = 8
child:   num_blocks = 7   # (7 + 8) // 2
```

### 2. 寬度選擇（隨機選擇法）

**策略**：隨機選擇一個父代的 `base_filters`

```python
import random

child_filters = random.choice([parent1.base_filters, parent2.base_filters])
```

**為什麼不平均？**
- ❌ 平均可能產生奇怪值（如 (128 + 256) // 2 = 192）
- ✅ 直接選擇保證是「好」的寬度值
- ✅ 保持一致性（所有層使用相同寬度）

**實際範例**：
```python
parent1: base_filters = 128
parent2: base_filters = 384

# 子代隨機選擇
child: base_filters = 128  (50% 機率)
    或 base_filters = 384  (50% 機率)

# 不會是 256！
```

### 3. 交叉點策略（Per-Block 繼承）

**核心概念**：在某個 block 位置「切割」，前半段來自父代 1，後半段來自父代 2

```python
def crossover_blocks(parent1, parent2, child_num_blocks):
    # 隨機選擇交叉點
    crossover_point = random.randint(1, child_num_blocks - 1)

    # 前半段來自父代 1
    child_filters = parent1.filters_per_block[:crossover_point]
    child_kernels = parent1.kernel_sizes[:crossover_point]
    child_types = parent1.block_types[:crossover_point]

    # 後半段來自父代 2
    child_filters += parent2.filters_per_block[crossover_point:child_num_blocks]
    child_kernels += parent2.kernel_sizes[crossover_point:child_num_blocks]
    child_types += parent2.block_types[crossover_point:child_num_blocks]

    return child_filters, child_kernels, child_types
```

**圖解**：
```
父代 1: [R, R, R, R, R] (5 blocks)
父代 2: [R, D, D, B, B, B, B, B] (8 blocks)
              ↓
        交叉點 = 3
              ↓
子代:   [R, R, R, | B, B, B] (6 blocks)
        └─ P1 ─┘   └─ P2 ─┘

R = Residual, D = Dense, B = Bottleneck
```

**實際範例**：
```python
parent1 = ArchitectureGenome(
    num_blocks=5,
    filters_per_block=[128, 128, 128, 128, 128],
    kernel_sizes=[3, 3, 3, 3, 3],
    block_types=['residual'] * 5,
    board_size=19
)

parent2 = ArchitectureGenome(
    num_blocks=8,
    filters_per_block=[256, 256, 384, 384, 384, 384, 256, 256],
    kernel_sizes=[3, 3, 5, 5, 5, 5, 3, 3],
    block_types=['residual', 'residual', 'dense', 'dense',
                 'dense', 'dense', 'residual', 'residual'],
    board_size=19
)

# 雜交（假設子代 6 blocks，交叉點 = 3）
child = ArchitectureGenome.crossover(parent1, parent2)

# 可能結果：
# filters:  [128, 128, 128, | 384, 384, 384]
# kernels:  [3,   3,   3,   | 5,   5,   5  ]
# types:    [res, res, res, | dense, dense, dense]
#           └─── P1 ───┘     └───── P2 ─────┘
```

### 4. 特性繼承（隨機或平均）

#### SE Blocks（隨機選擇）

```python
# 隨機選擇一個父代的設置
child.use_se_blocks = random.choice([
    parent1.use_se_blocks,
    parent2.use_se_blocks
])

if child.use_se_blocks:
    # 如果兩個父代都有 SE，可以平均 reduction_ratio
    if parent1.use_se_blocks and parent2.use_se_blocks:
        child.se_reduction_ratio = random.choice([
            parent1.se_reduction_ratio,
            parent2.se_reduction_ratio
        ])
```

#### Dropout Rate（平均法）

```python
# 平均 dropout rate
child.dropout_rate = (parent1.dropout_rate + parent2.dropout_rate) / 2.0
```

**實際範例**：
```python
parent1: use_se_blocks = True,  se_reduction_ratio = 16
parent2: use_se_blocks = False

# 子代可能：
child: use_se_blocks = True,  se_reduction_ratio = 16  (選擇 P1)
  或: use_se_blocks = False                           (選擇 P2)

parent1: dropout_rate = 0.1
parent2: dropout_rate = 0.3

child: dropout_rate = 0.2  # (0.1 + 0.3) / 2
```

## 世代追蹤與血統管理

### 世代編號

```python
# 子代世代 = max(父代世代) + 1
child.generation = max(parent1.generation, parent2.generation) + 1
```

**範例**：
```python
parent1.generation = 0  # 初始世代
parent2.generation = 2  # 經過兩次演化

child.generation = 3    # max(0, 2) + 1
```

### 父代記錄

```python
# 記錄兩個父代的 ID
child.parent_ids = [parent1.get_id(), parent2.get_id()]
```

**用途**：
- ✅ 可追溯演化歷史
- ✅ 分析哪些架構組合成功
- ✅ 避免近親繁殖（可選）
- ✅ 可視化演化樹

### 完整血統範例

```python
# Gen 0: 初始種群
baseline = ArchitectureGenome.create_baseline(board_size=19)
print(f"Gen 0: {baseline.get_id()[:8]}, 父代: {baseline.parent_ids}")
# Gen 0: a1b2c3d4, 父代: []

# Gen 1: 突變
mutated1 = baseline.mutate(mutation_rate=0.2)
mutated2 = baseline.mutate(mutation_rate=0.2)
print(f"Gen 1a: {mutated1.get_id()[:8]}, 父代: {[x[:8] for x in mutated1.parent_ids]}")
print(f"Gen 1b: {mutated2.get_id()[:8]}, 父代: {[x[:8] for x in mutated2.parent_ids]}")
# Gen 1a: e5f6g7h8, 父代: ['a1b2c3d4']
# Gen 1b: i9j0k1l2, 父代: ['a1b2c3d4']

# Gen 2: 雜交
child = ArchitectureGenome.crossover(mutated1, mutated2)
print(f"Gen 2: {child.get_id()[:8]}, 父代: {[x[:8] for x in child.parent_ids]}")
# Gen 2: m3n4o5p6, 父代: ['e5f6g7h8', 'i9j0k1l2']
```

## 雜交實戰範例

### 範例 1：平衡組合

```python
# 父代 1：保守型（小而穩）
conservative = ArchitectureGenome(
    num_blocks=5,
    base_filters=128,
    kernel_sizes=[3] * 5,
    use_se_blocks=False,
    dropout_rate=0.0,
    board_size=19
)

# 父代 2：激進型（大而強）
aggressive = ArchitectureGenome(
    num_blocks=12,
    base_filters=256,
    kernel_sizes=[5] * 12,
    use_se_blocks=True,
    dropout_rate=0.2,
    board_size=19
)

# 雜交：產生平衡型
balanced = ArchitectureGenome.crossover(conservative, aggressive)

print(f"深度: {conservative.num_blocks} + {aggressive.num_blocks} → {balanced.num_blocks}")
# 深度: 5 + 12 → 8 (平均)

print(f"寬度: {balanced.base_filters}")
# 寬度: 128 或 256 (隨機選擇)

print(f"參數: {balanced.estimate_parameters():,}")
# 約 2-4M（介於兩者之間）
```

### 範例 2：專長互補

```python
# 序盤專家：淺而寬，大視野
opening_expert = ArchitectureGenome(
    num_blocks=4,
    base_filters=384,
    kernel_sizes=[7, 7, 5, 5],  # 大卷積核
    board_size=19
)

# 中盤專家：深而窄，精細計算
middle_expert = ArchitectureGenome(
    num_blocks=15,
    base_filters=128,
    kernel_sizes=[3] * 15,      # 小卷積核
    board_size=19
)

# 雜交：產生全能型
all_round = ArchitectureGenome.crossover(opening_expert, middle_expert)

# 可能結果：
# - 中等深度（9-10 blocks）
# - 隨機寬度（128 或 384）
# - 混合卷積核（前幾層大核，後幾層小核）
```

### 範例 3：性能與效率平衡

```python
# 高性能型：參數多，準確
high_performance = ArchitectureGenome(
    num_blocks=20,
    base_filters=512,
    use_se_blocks=True,
    board_size=19
)
# 約 50M 參數

# 高效率型：參數少，快速
high_efficiency = ArchitectureGenome(
    num_blocks=3,
    base_filters=64,
    use_se_blocks=False,
    board_size=19
)
# 約 200K 參數

# 雜交：尋找甜點
sweet_spot = ArchitectureGenome.crossover(high_performance, high_efficiency)
# 約 5-10M 參數，性能與速度平衡
```

## 常見問題與陷阱

### Q1: 兩個父代差異太大怎麼辦？

```python
parent1: num_blocks = 3,  base_filters = 64
parent2: num_blocks = 20, base_filters = 512

# 雜交
child: num_blocks = 11,  base_filters = 64 或 512
```

**問題**：深度和寬度不匹配，可能訓練困難

**解決方案**：
```python
# 選擇相似的父代進行雜交
def select_compatible_parents(population):
    # 計算架構相似度
    similarity = compute_similarity(p1, p2)
    if similarity < 0.3:
        return False  # 太不相似，不配對
    return True
```

### Q2: 雜交會不會產生無效架構？

```python
# 範例：不兼容的 block 類型組合
parent1: block_types = ['residual'] * 5
parent2: block_types = ['dense'] * 8

# 雜交後可能：
child: block_types = ['residual', 'residual', 'dense', 'dense', 'dense', 'dense']
```

**解決**：架構合法性檢查
```python
def crossover(parent1, parent2):
    child = # ... 雜交操作

    # 檢查合法性
    assert len(child.filters_per_block) == child.num_blocks
    assert len(child.kernel_sizes) == child.num_blocks
    assert len(child.block_types) == child.num_blocks

    return child
```

### Q3: 如何避免近親繁殖？

```python
def is_inbreeding(parent1, parent2, max_shared_ancestors=2):
    """檢查是否有太多共同祖先"""
    ancestors1 = get_all_ancestors(parent1)
    ancestors2 = get_all_ancestors(parent2)

    shared = len(ancestors1 & ancestors2)
    return shared > max_shared_ancestors

# 使用
if not is_inbreeding(p1, p2):
    child = ArchitectureGenome.crossover(p1, p2)
else:
    print("近親繁殖！選擇其他父代")
```

## 雜交策略總結

### 核心原則

| 參數類型 | 繼承策略 | 理由 |
|----------|----------|------|
| **深度** | 平均值 | 平衡兩者，避免極端 |
| **寬度** | 隨機選擇 | 保持一致性，避免奇怪值 |
| **Per-block 參數** | 交叉點切割 | 組合局部優勢 |
| **SE Blocks** | 隨機選擇 | 二元特性，無法平均 |
| **Dropout** | 平均值 | 連續值，平均合理 |

### 與突變的配合

```python
# 完整演化流程
def evolve_population(population):
    # 1. 選擇（評估後保留精英）
    elite = select_top_performers(population, top_k=5)

    # 2. 雜交（精英配對）
    offspring = []
    for p1, p2 in pair_elite(elite):
        child = ArchitectureGenome.crossover(p1, p2)
        offspring.append(child)

    # 3. 突變（所有個體）
    for genome in elite + offspring:
        mutated = genome.mutate(mutation_rate=0.2)
        offspring.append(mutated)

    # 4. 新世代
    return elite + offspring
```

## 實踐練習

### 練習 1：觀察雜交效果

```python
parent1 = ArchitectureGenome.create_baseline(board_size=19)
parent2 = parent1.mutate(mutation_rate=0.3)

for i in range(5):
    child = ArchitectureGenome.crossover(parent1, parent2)
    print(f"\n雜交 {i+1}:")
    print(f"  P1 深度={parent1.num_blocks}, 寬度={parent1.base_filters}")
    print(f"  P2 深度={parent2.num_blocks}, 寬度={parent2.base_filters}")
    print(f"  子代 深度={child.num_blocks}, 寬度={child.base_filters}")
```

### 練習 2：追蹤血統

```python
def print_lineage(genome, depth=0):
    """遞歸打印血統"""
    indent = "  " * depth
    print(f"{indent}Gen {genome.generation}: {genome.get_id()[:8]}")

    if genome.parent_ids:
        for parent_id in genome.parent_ids:
            parent = load_genome_by_id(parent_id)  # 需要實現
            print_lineage(parent, depth + 1)

# 使用
gen0 = ArchitectureGenome.create_baseline(board_size=19)
gen1a = gen0.mutate(mutation_rate=0.2)
gen1b = gen0.mutate(mutation_rate=0.2)
gen2 = ArchitectureGenome.crossover(gen1a, gen1b)

print_lineage(gen2)
# Gen 2: m3n4o5p6
#   Gen 1: e5f6g7h8
#     Gen 0: a1b2c3d4
#   Gen 1: i9j0k1l2
#     Gen 0: a1b2c3d4
```

### 練習 3：設計選擇性雜交

```python
def selective_crossover(population, similarity_threshold=0.5):
    """只讓相似的父代雜交"""
    offspring = []

    for p1 in population:
        for p2 in population:
            if p1 == p2:
                continue

            # 計算相似度（簡化版本）
            similarity = 1.0 - abs(p1.num_blocks - p2.num_blocks) / 20.0

            if similarity >= similarity_threshold:
                child = ArchitectureGenome.crossover(p1, p2)
                offspring.append(child)

    return offspring
```

## 下一課預告

在[第四課：基因組變成真正的神經網絡](lesson-4-genome-to-model.md)中，您將學習：

- 如何將抽象的基因組「編譯」成 PyTorch 模型
- Input Convolution、Residual Blocks 的具體構建
- Policy Head 和 Value Head 的實現
- 參數量驗證與模型測試

基因組只是「設計圖」，下一課將它變成**真正可運行的神經網絡**！

---

[← 上一課：突變機制](lesson-2-mutation.md) | [返回大綱](index.md) | [下一課：基因組轉模型 →](lesson-4-genome-to-model.md)

# 第二課：突變 - 架構如何進化

## 課程目標

深入理解突變（Mutation）機制，學習架構如何通過隨機變化產生新的變體，以及如何保持可訓練性。

## 什麼是突變？

突變（Mutation）是進化的核心機制。每個基因都有一定概率被隨機改變，產生新的架構變體。

```python
from core.architecture_genome import ArchitectureGenome

# 創建基準架構
baseline = ArchitectureGenome.create_baseline(board_size=19)

# 突變！
mutated = baseline.mutate(mutation_rate=0.2)

print(f"原始 ID: {baseline.get_id()}")
print(f"突變 ID: {mutated.get_id()}")
print(f"參數量變化: {baseline.estimate_parameters():,} → {mutated.estimate_parameters():,}")
```

## 突變率（Mutation Rate）

### 定義

突變率決定每個基因被改變的概率。

```python
mutated = genome.mutate(mutation_rate=0.2)
# 每個基因有 20% 機率突變
```

### 突變率的影響

| mutation_rate | 變化程度 | 適用場景 | 風險 |
|---------------|----------|----------|------|
| 0.05-0.1 | 輕微變化 | 精細調優、接近最優解 | 進化太慢 |
| 0.2-0.3 | 中等變化 | **標準設置**、平衡探索與利用 | 平衡良好 |
| 0.4-0.5 | 大幅變化 | 跳出局部最優、探索新區域 | 可能破壞好架構 |
| 0.6+ | 劇烈變化 | 隨機重啟（不推薦作為突變）| 失去繼承性 |

### 實驗：觀察不同突變率

```python
baseline = ArchitectureGenome.create_baseline(board_size=19)

for rate in [0.1, 0.2, 0.3, 0.5]:
    mutated = baseline.mutate(mutation_rate=rate)

    # 計算差異
    diff_count = sum([
        baseline.num_blocks != mutated.num_blocks,
        baseline.base_filters != mutated.base_filters,
        baseline.use_se_blocks != mutated.use_se_blocks,
        baseline.dropout_rate != mutated.dropout_rate
    ])

    print(f"Rate={rate}: {diff_count} 個參數改變")
```

**典型輸出**:
```
Rate=0.1: 0-1 個參數改變
Rate=0.2: 1-2 個參數改變  ← 推薦
Rate=0.3: 2-3 個參數改變
Rate=0.5: 3-5 個參數改變
```

## 5 種突變類型

### 1. 深度突變（Network Depth）

**機制**：±1 block（不會一次跳太多）

```python
# 原始
num_blocks = 5

# 突變後可能：
num_blocks = 4  # -1
num_blocks = 5  # 不變
num_blocks = 6  # +1
```

**限制**：
- 最小值：3 blocks
- 最大值：20 blocks

**為什麼 ±1？**
- ✅ 漸進式變化，保持穩定性
- ✅ 權重可以部分繼承（Net2Net）
- ❌ 如果一次 ±5，架構變化太大，訓練困難

**實際範例**：
```python
# 淺網絡逐步變深
gen_0: num_blocks = 5   (baseline)
gen_1: num_blocks = 6   (+1 突變)
gen_2: num_blocks = 7   (+1 突變)
gen_3: num_blocks = 8   (+1 突變)
# 經過多代，深度逐步增加
```

### 2. 寬度突變（Network Width）

**機制**：在預設值中隨機選擇

```python
# 可選值
FILTER_OPTIONS = [64, 96, 128, 192, 256, 384, 512]

# 突變
if random.random() < mutation_rate:
    genome.base_filters = random.choice(FILTER_OPTIONS)
    # 同時更新 filters_per_block
    genome.filters_per_block = [genome.base_filters] * genome.num_blocks
```

**為什麼使用離散值？**
- ✅ 保證是「好」的寬度（2 的倍數，GPU 友好）
- ✅ 避免奇怪的值（如 127, 253）
- ✅ 便於權重繼承

**典型變化**：
```python
# 變窄
base_filters: 128 → 96   (-25%)
base_filters: 256 → 128  (-50%)

# 變寬
base_filters: 128 → 192  (+50%)
base_filters: 128 → 256  (+100%)
```

### 3. 卷積核突變（Kernel Size）

**機制**：在 {3, 5, 7} 中選擇

```python
KERNEL_OPTIONS = [3, 5, 7]

if random.random() < mutation_rate:
    for i in range(len(genome.kernel_sizes)):
        if random.random() < 0.3:  # 每層 30% 機率
            genome.kernel_sizes[i] = random.choice(KERNEL_OPTIONS)
```

**不同核大小的策略意義**：

```python
# 保守型：全部小核
kernel_sizes = [3, 3, 3, 3, 3]
# 特點：細節豐富，計算快

# 平衡型：中等核
kernel_sizes = [3, 5, 5, 5, 3]
# 特點：中距離模式識別

# 激進型：混合大核
kernel_sizes = [3, 5, 7, 7, 5, 3]
# 特點：全局視野，計算慢
```

**實際變化範例**：
```python
# 突變前
kernel_sizes = [3, 3, 3, 3, 3]

# 突變後（可能）
kernel_sizes = [3, 5, 3, 3, 3]  # 第 2 層變大
kernel_sizes = [3, 3, 3, 5, 3]  # 第 4 層變大
kernel_sizes = [5, 3, 3, 3, 3]  # 第 1 層變大
```

### 4. Block 類型突變

**機制**：切換 block 類型，有偏向性

```python
BLOCK_TYPES = ['residual', 'dense', 'bottleneck']

# 偏向 residual（已被驗證有效）
BLOCK_TYPE_WEIGHTS = {
    'residual': 0.7,    # 70% 機率
    'dense': 0.2,       # 20% 機率
    'bottleneck': 0.1   # 10% 機率
}

if random.random() < mutation_rate:
    for i in range(len(genome.block_types)):
        if random.random() < 0.2:
            genome.block_types[i] = weighted_choice(BLOCK_TYPE_WEIGHTS)
```

**為什麼有偏向性？**
- Residual blocks 被 AlphaGo Zero 驗證有效
- Dense/Bottleneck 是探索性選項
- 平衡**利用**（proven methods）和**探索**（new methods）

**變化範例**：
```python
# 全 residual（保守）
block_types = ['residual'] * 5

# 突變：混入 dense
block_types = ['residual', 'residual', 'dense', 'residual', 'residual']

# 突變：混入 bottleneck
block_types = ['residual', 'bottleneck', 'residual', 'residual', 'residual']
```

### 5. 特性突變（Features）

#### SE Blocks 開關

```python
if random.random() < mutation_rate:
    genome.use_se_blocks = not genome.use_se_blocks

    # 如果啟用，同時設置壓縮比
    if genome.use_se_blocks:
        genome.se_reduction_ratio = random.choice([8, 16, 32])
```

**效果**：
```python
# 禁用 → 啟用
use_se_blocks: False → True
參數量: +5%
性能: +1-2%

# 啟用 → 禁用
use_se_blocks: True → False
參數量: -5%
速度: +10%
```

#### Dropout 調整

```python
DROPOUT_OPTIONS = [0.0, 0.1, 0.15, 0.2, 0.3]

if random.random() < mutation_rate:
    genome.dropout_rate = random.choice(DROPOUT_OPTIONS)
```

**策略**：
- 0.0: 不使用（數據充足）
- 0.1-0.2: 標準正則化
- 0.3+: 強正則化（小數據集）

## 突變實戰範例

### 範例 1：輕微調整

```python
baseline = ArchitectureGenome.create_baseline(board_size=19)
# num_blocks=5, base_filters=128, use_se_blocks=False

mutated = baseline.mutate(mutation_rate=0.15)
# 可能變化：
# num_blocks=6 (+1)
# 其他參數不變

print(f"Params: {baseline.estimate_parameters():,} → {mutated.estimate_parameters():,}")
# Params: 1,538,858 → 1,847,210 (+20%)
```

### 範例 2：中等變化

```python
baseline = ArchitectureGenome.create_baseline(board_size=19)

mutated = baseline.mutate(mutation_rate=0.25)
# 可能變化：
# num_blocks=5 → 6
# base_filters=128 → 192
# use_se_blocks=False → True

print(f"Params: {baseline.estimate_parameters():,} → {mutated.estimate_parameters():,}")
# Params: 1,538,858 → 4,156,234 (+170%)
```

### 範例 3：激進探索

```python
baseline = ArchitectureGenome.create_baseline(board_size=19)

mutated = baseline.mutate(mutation_rate=0.4)
# 可能變化：
# num_blocks=5 → 4
# base_filters=128 → 256
# kernel_sizes=[3,3,3,3,3] → [5,5,3,7,5]
# dropout_rate=0.0 → 0.2

# 架構面目全非！
```

## 保持可訓練性的策略

### 1. 漸進式變化

```python
# ❌ 錯誤：一次跳太多
num_blocks: 5 → 15  # +200%，權重無法繼承

# ✅ 正確：逐步增加
gen_0: num_blocks = 5
gen_1: num_blocks = 6  (+1)
gen_2: num_blocks = 7  (+1)
gen_3: num_blocks = 8  (+1)
```

### 2. 參數範圍限制

```python
# 深度限制
MIN_BLOCKS = 3   # 太淺無法學習
MAX_BLOCKS = 20  # 太深訓練困難

# 寬度限制
MIN_FILTERS = 32   # 太窄表達能力不足
MAX_FILTERS = 512  # 太寬計算爆炸

# Dropout 限制
MAX_DROPOUT = 0.5  # 太高會欠擬合
```

### 3. 合法性檢查

```python
def mutate(self, mutation_rate=0.2):
    # 創建副本
    mutated = copy.deepcopy(self)

    # 突變操作...

    # 檢查合法性
    assert 3 <= mutated.num_blocks <= 20
    assert mutated.base_filters in [64, 96, 128, 192, 256, 384, 512]
    assert all(k in [3, 5, 7] for k in mutated.kernel_sizes)
    assert 0.0 <= mutated.dropout_rate <= 0.5

    # 更新世代信息
    mutated.generation = self.generation + 1
    mutated.parent_ids = [self.get_id()]

    return mutated
```

### 4. 世代追蹤

```python
# 每次突變記錄父代
mutated.parent_ids = [parent.get_id()]
mutated.generation = parent.generation + 1

# 可追溯演化歷史
print(f"世代 {mutated.generation}")
print(f"父代: {mutated.parent_ids}")
```

## 突變策略總結

### 核心原則

1. **漸進式**：小步快跑，不要大跳
2. **有界性**：限制參數範圍，保證合法
3. **偏向性**：已驗證的方法（residual）機率高
4. **可繼承**：新架構能複用舊權重

### 推薦設置

```python
# 標準演化
mutation_rate = 0.2

# 精細調優（接近最優）
mutation_rate = 0.1

# 跳出局部最優
mutation_rate = 0.35

# 隨機重啟（不推薦作為突變）
mutation_rate = 0.8  # 太高！
```

### 常見錯誤

❌ **錯誤 1**：突變率太高
```python
mutated = genome.mutate(mutation_rate=0.9)
# 問題：幾乎所有參數都變了，失去父代優勢
```

❌ **錯誤 2**：無限制變化
```python
num_blocks = random.randint(1, 100)
# 問題：100 層網絡訓練不了
```

❌ **錯誤 3**：忽略世代追蹤
```python
mutated.generation = 0  # 錯！應該 +1
mutated.parent_ids = []  # 錯！應該記錄父代
```

## 實踐練習

### 練習 1：觀察多次突變

```python
baseline = ArchitectureGenome.create_baseline(board_size=19)

print(f"原始: {baseline}")

for i in range(5):
    mutated = baseline.mutate(mutation_rate=0.2)
    print(f"\n突變 {i+1}:")
    print(f"  深度: {baseline.num_blocks} → {mutated.num_blocks}")
    print(f"  寬度: {baseline.base_filters} → {mutated.base_filters}")
    print(f"  參數: {baseline.estimate_parameters():,} → {mutated.estimate_parameters():,}")
```

### 練習 2：測試突變率影響

```python
baseline = ArchitectureGenome.create_baseline(board_size=19)

for rate in [0.1, 0.2, 0.3, 0.5]:
    mutations = []
    for _ in range(10):
        m = baseline.mutate(mutation_rate=rate)
        mutations.append(m.estimate_parameters())

    avg_params = sum(mutations) / len(mutations)
    std_params = (sum((x - avg_params)**2 for x in mutations) / len(mutations)) ** 0.5

    print(f"Rate {rate}: Avg={avg_params:,.0f}, Std={std_params:,.0f}")
```

### 練習 3：設計自適應突變率

```python
def adaptive_mutation_rate(generation, max_gen=100):
    """
    早期：高突變率（探索）
    後期：低突變率（利用）
    """
    return 0.5 * (1 - generation / max_gen) + 0.1
    # gen_0: 0.5
    # gen_50: 0.3
    # gen_100: 0.1

# 測試
for gen in [0, 25, 50, 75, 100]:
    rate = adaptive_mutation_rate(gen)
    print(f"Gen {gen}: mutation_rate = {rate:.2f}")
```

## 下一課預告

在[第三課：雜交 - 結合父代的優勢](lesson-3-crossover.md)中，您將學習：

- 如何結合兩個成功架構的優點
- 交叉點選擇策略
- 深度、寬度、特性的繼承方式
- 世代追蹤與血統管理

突變產生**變異**，雜交產生**組合** —— 兩者結合，演化才強大！

---

[← 上一課：基因組 DNA](lesson-1-genome-dna.md) | [返回大綱](index.md) | [下一課：雜交機制 →](lesson-3-crossover.md)

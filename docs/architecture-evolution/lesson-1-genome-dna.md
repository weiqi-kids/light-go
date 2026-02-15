# 第一課：架構基因組的 DNA

## 課程目標

理解架構基因組（ArchitectureGenome）的基本組成，以及每個參數如何影響神經網絡的結構和性能。

## 什麼是架構基因組？

架構基因組（ArchitectureGenome）是一組「基因」，描述了神經網絡的結構。就像生物的 DNA 決定生物的特徵，基因組決定了神經網絡的架構。

```python
from core.architecture_genome import ArchitectureGenome

# 創建基準基因組
baseline = ArchitectureGenome.create_baseline(board_size=19)
print(baseline)
```

## 基因組的 7 個核心參數

### 1. num_blocks（網絡深度）

**定義**: 決定有多少個 residual blocks

```python
num_blocks = 5  # 範圍: 3-20
```

**影響**:
- **更深**（如 20 blocks）
  - ✅ 更強的表達能力
  - ✅ 能學習更複雜的模式
  - ❌ 訓練更困難
  - ❌ 需要更多訓練數據
  - ❌ 推理速度較慢

- **更淺**（如 5 blocks）
  - ✅ 訓練快速
  - ✅ 推理快速
  - ❌ 表達能力受限
  - ❌ 可能欠擬合

**典型值**:
- 9x9 棋盤：5-8 blocks
- 19x19 棋盤：10-15 blocks
- 資源受限：3-5 blocks

### 2. base_filters（基礎寬度）

**定義**: 每層卷積的 filter 數量

```python
base_filters = 128  # 典型值: 64, 128, 256, 512
```

**影響**:
- **更寬**（如 512 filters）
  - ✅ 更多特徵通道
  - ✅ 表達能力更強
  - ❌ 計算量大幅增加
  - ❌ 內存消耗高
  - ❌ 訓練速度慢

- **更窄**（如 64 filters）
  - ✅ 計算快速
  - ✅ 內存效率高
  - ❌ 特徵表達受限

**參數量關係**:
```
參數量 ≈ base_filters² × num_blocks × 9
```

例如：
- 128 filters, 5 blocks ≈ 740K 參數
- 256 filters, 10 blocks ≈ 5.9M 參數
- 512 filters, 20 blocks ≈ 47M 參數

### 3. filters_per_block（每層的具體寬度）

**定義**: 可以讓每層有不同寬度的 filter 配置

```python
filters_per_block = [128, 256, 256, 256, 128]
# 先變寬，再變窄（瓶頸結構）
```

**常見模式**:

1. **均勻寬度**
   ```python
   [128, 128, 128, 128, 128]  # 所有層相同
   ```

2. **漸進增寬**
   ```python
   [128, 192, 256, 320, 384]  # 逐層增加
   ```

3. **瓶頸結構**
   ```python
   [128, 256, 256, 256, 128]  # 中間寬，兩端窄
   ```

4. **金字塔結構**
   ```python
   [384, 256, 192, 128, 64]  # 逐層減少
   ```

### 4. kernel_sizes（卷積核大小）

**定義**: 每層卷積的感受野大小

```python
kernel_sizes = [3, 3, 3, 3, 3]  # 可選: 3, 5, 7
```

**不同核大小的特性**:

| 核大小 | 感受野 | 計算量 | 適用場景 |
|--------|--------|--------|----------|
| 3×3 | 局部模式 | 低 | 基礎特徵、細節 |
| 5×5 | 中等範圍 | 中 | 棋形、小規模戰鬥 |
| 7×7 | 大範圍 | 高 | 全局判斷、大局觀 |

**混合策略範例**:
```python
kernel_sizes = [3, 3, 5, 5, 7, 5, 3, 3]
# 前層看局部 → 中層看中距離 → 後層整合全局
```

### 5. block_types（模塊類型）

**定義**: 每個 block 使用的架構類型

```python
block_types = ['residual', 'residual', 'dense', ...]
```

**可用類型**:

1. **residual**（ResNet block）
   ```
   Input → Conv → BN → ReLU → Conv → BN → (+Input) → ReLU
   ```
   - 最常用，訓練穩定
   - 適合深層網絡

2. **dense**（DenseNet block）
   ```
   每層連接到所有前層
   ```
   - 特徵重用
   - 參數效率高
   - 內存消耗大

3. **bottleneck**（瓶頸結構）
   ```
   Input → Conv 1×1 (降維) → Conv 3×3 → Conv 1×1 (升維)
   ```
   - 減少計算量
   - 保持表達能力

**典型配置**:
```python
# 保守策略：全部使用 residual
block_types = ['residual'] * num_blocks

# 混合策略：前層 residual，後層 dense
block_types = ['residual'] * 5 + ['dense'] * 3
```

### 6. use_se_blocks（Squeeze-Excitation）

**定義**: 是否啟用注意力機制

```python
use_se_blocks = True
se_reduction_ratio = 16  # SE block 的壓縮比
```

**SE Block 原理**:
```
特徵圖 → Global Average Pooling → FC(壓縮) → ReLU → FC(恢復) → Sigmoid → 重新加權
```

**效果**:
- ✅ 讓網絡學會關注重要特徵
- ✅ 通常提升 1-2% 準確率
- ❌ 增加約 5% 計算量
- ❌ 增加少量參數（< 1%）

**建議**:
- 資源充足：啟用
- 資源受限：禁用
- 實驗性架構：啟用並觀察效果

### 7. dropout_rate（隨機丟棄率）

**定義**: 訓練時隨機丟棄神經元的比例，防止過擬合

```python
dropout_rate = 0.1  # 範圍: 0.0-0.5
```

**影響**:

| dropout_rate | 效果 | 適用場景 |
|--------------|------|----------|
| 0.0 | 不使用 dropout | 數據充足、模型不過擬合 |
| 0.1-0.2 | 輕度正則化 | 標準設置 |
| 0.3-0.4 | 中度正則化 | 小數據集、複雜模型 |
| 0.5+ | 強正則化 | 極小數據集（不推薦太高）|

## 完整範例

### 創建基準基因組

```python
from core.architecture_genome import ArchitectureGenome

baseline = ArchitectureGenome.create_baseline(board_size=19)

print(f"網絡深度: {baseline.num_blocks}")
print(f"基礎寬度: {baseline.base_filters}")
print(f"每層寬度: {baseline.filters_per_block}")
print(f"卷積核: {baseline.kernel_sizes}")
print(f"Block 類型: {baseline.block_types}")
print(f"SE Blocks: {baseline.use_se_blocks}")
print(f"Dropout: {baseline.dropout_rate}")
print(f"估算參數量: {baseline.estimate_parameters():,}")
```

**輸出範例**:
```
網絡深度: 5
基礎寬度: 128
每層寬度: [128, 128, 128, 128, 128]
卷積核: [3, 3, 3, 3, 3]
Block 類型: ['residual', 'residual', 'residual', 'residual', 'residual']
SE Blocks: False
Dropout: 0.0
估算參數量: 1,538,858
```

### 自定義基因組

```python
# 創建一個深而窄的架構（擅長精細計算）
deep_narrow = ArchitectureGenome(
    num_blocks=15,
    base_filters=96,
    filters_per_block=[96] * 15,
    kernel_sizes=[3] * 15,
    block_types=['residual'] * 15,
    use_se_blocks=True,
    dropout_rate=0.1,
    board_size=19
)

# 創建一個淺而寬的架構（擅長快速評估）
shallow_wide = ArchitectureGenome(
    num_blocks=6,
    base_filters=384,
    filters_per_block=[384] * 6,
    kernel_sizes=[5, 5, 5, 5, 5, 5],
    block_types=['residual'] * 6,
    use_se_blocks=False,
    dropout_rate=0.0,
    board_size=19
)

print(f"Deep-Narrow 參數量: {deep_narrow.estimate_parameters():,}")
print(f"Shallow-Wide 參數量: {shallow_wide.estimate_parameters():,}")
```

## 參數量估算

基因組提供參數量估算功能：

```python
params = genome.estimate_parameters()
print(f"估算參數量: {params:,}")
```

**估算公式**（簡化）:
```python
# Input conv
params = 7 * base_filters * 9 + base_filters

# Residual blocks
for i in range(num_blocks):
    f = filters_per_block[i]
    k = kernel_sizes[i]
    params += f * f * k * k * 2  # 兩個卷積層
    params += f * 2              # 兩個 BatchNorm

    if use_se_blocks:
        params += f * (f // se_reduction_ratio) * 2

# Policy head
params += filters_per_block[-1] * 2 * 1 * 1  # 1x1 conv
params += 2 * 19 * 19 * 362                  # FC layer

# Value head
params += filters_per_block[-1] * 1 * 1      # 1x1 conv
params += 19 * 19 * 256 + 256                # FC layers
```

## 重點總結

### 核心概念
1. **架構基因組** = 神經網絡的 DNA
2. **7 個核心參數**決定網絡結構
3. **不同參數組合**產生不同性能特性

### 設計原則
- **深度 vs 寬度**：深度學複雜模式，寬度學多特徵
- **大核 vs 小核**：大核看全局，小核看細節
- **SE blocks**：提升性能但增加計算
- **Dropout**：防止過擬合

### 參數量控制
```
參數量 ≈ (base_filters)² × num_blocks × kernel_size²
```

### 實際建議
1. **初學者**：使用 `create_baseline()` 開始
2. **實驗者**：逐個調整參數，觀察效果
3. **進階者**：設計專門化架構（序盤、中盤、官子）

## 實踐練習

### 練習 1：創建基因組
```python
# 創建一個適合快速推理的基因組
# 目標：< 500K 參數，推理速度快

fast_genome = ArchitectureGenome(
    num_blocks=4,
    base_filters=64,
    # ... 完成其他參數
)
```

### 練習 2：比較參數量
```python
# 觀察深度和寬度如何影響參數量
for depth in [5, 10, 15]:
    for width in [64, 128, 256]:
        genome = ArchitectureGenome(
            num_blocks=depth,
            base_filters=width,
            # ...
        )
        print(f"Depth={depth}, Width={width}: {genome.estimate_parameters():,} params")
```

### 練習 3：設計專門化架構
```python
# 設計一個「序盤專家」架構
# 特點：看大局，少看細節

opening_expert = ArchitectureGenome(
    # 思考：需要深還是淺？寬還是窄？大核還是小核？
)
```

## 下一課預告

在[第二課：突變 - 架構如何進化](lesson-2-mutation.md)中，您將學習：

- 如何隨機改變基因組參數
- 突變率的選擇策略
- 5 種突變類型詳解
- 如何保持架構的可訓練性

這些基因組不是固定的 —— 它們會**進化**！

---

[返回課程大綱](index.md) | [下一課：突變機制 →](lesson-2-mutation.md)

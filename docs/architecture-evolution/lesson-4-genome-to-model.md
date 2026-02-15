# 第四課：基因組變成真正的神經網絡

## 課程目標

探索如何將抽象的架構基因組「編譯」成可訓練的 PyTorch 模型，理解模型的具體結構和參數分布。

## 從設計圖到實體

架構基因組就像建築的設計圖紙，而 PyTorch 模型是真正可運行的建築。這一課將展示這個「編譯」過程。

```python
from core.architecture_genome import ArchitectureGenome

# 創建基因組（設計圖）
genome = ArchitectureGenome.create_baseline(board_size=19)

print(f"📋 基因組規格:")
print(f"   • {genome.num_blocks} blocks")
print(f"   • {genome.base_filters} filters")
print(f"   • 估算參數: ~{genome.estimate_parameters():,}")

# 構建模型（實體）
model = genome.to_pytorch_model(device='cpu')

print(f"\n✅ 模型構建完成！")
print(f"   • 實際參數量: {sum(p.numel() for p in model.parameters()):,}")
```

## 模型架構總覽

### AlphaGo Zero 風格架構

Light-Go 採用類似 AlphaGo Zero 的雙頭架構：

```
輸入 (7×19×19)
    ↓
Input Convolution (Conv 3×3)
    ↓
Residual Blocks ×N
    ↓
    ├─→ Policy Head → Move Probabilities (362)
    └─→ Value Head  → Win Probability (1)
```

### 完整結構

```python
class GoAIModel(nn.Module):
    def __init__(self, genome: ArchitectureGenome):
        # 1. Input Convolution
        self.input_conv = nn.Conv2d(7, base_filters, kernel_size=3, padding=1)
        self.input_bn = nn.BatchNorm2d(base_filters)

        # 2. Residual Blocks
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(filters_per_block[i], kernel_sizes[i])
            for i in range(num_blocks)
        ])

        # 3. Policy Head
        self.policy_head = PolicyHead(filters_per_block[-1])

        # 4. Value Head
        self.value_head = ValueHead(filters_per_block[-1])
```

## 各組件詳解

### 1. Input Convolution（輸入卷積）

**作用**：將棋盤特徵轉換為神經網絡特徵空間

```python
# 輸入：7 個平面
# - 平面 0: 黑子位置 (1 = 有黑子, 0 = 無)
# - 平面 1: 白子位置
# - 平面 2: 黑子氣數
# - 平面 3: 白子氣數
# - 平面 4: 打劫位置
# - 平面 5: 當前顏色 (全 1 或全 0)
# - 平面 6: 空白位置

self.input_conv = nn.Conv2d(
    in_channels=7,
    out_channels=base_filters,  # 如 128
    kernel_size=3,
    stride=1,
    padding=1,
    bias=False
)
self.input_bn = nn.BatchNorm2d(base_filters)

# 前向傳播
x = F.relu(self.input_bn(self.input_conv(board_features)))
# 輸出形狀: (batch, base_filters, 19, 19)
```

**參數量**：
```python
params = 7 * base_filters * 3 * 3 + base_filters
# 如 base_filters=128:
params = 7 * 128 * 9 + 128 = 8,192
```

### 2. Residual Block（殘差模塊）

**作用**：深層特徵提取，保持梯度流動

```python
class ResidualBlock(nn.Module):
    def __init__(self, filters, kernel_size=3):
        super().__init__()

        # 第一層卷積
        self.conv1 = nn.Conv2d(
            filters, filters,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(filters)

        # 第二層卷積
        self.conv2 = nn.Conv2d(
            filters, filters,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            bias=False
        )
        self.bn2 = nn.BatchNorm2d(filters)

    def forward(self, x):
        residual = x  # 保存輸入

        # 第一層
        out = F.relu(self.bn1(self.conv1(x)))

        # 第二層
        out = self.bn2(self.conv2(out))

        # 殘差連接
        out += residual

        # 激活
        out = F.relu(out)

        return out
```

**為什麼需要殘差連接？**

```
無殘差:
x → Conv → ReLU → Conv → ReLU → ...
      ↓梯度消失，難以訓練深層網絡

有殘差:
x → Conv → ReLU → Conv → (+x) → ReLU → ...
     ↑_______________________|
      梯度可以直接流回，訓練穩定
```

**參數量**（每個 block）：
```python
# kernel_size=3, filters=128
params_conv1 = 128 * 128 * 3 * 3 = 147,456
params_bn1 = 128 * 2 = 256  # γ, β
params_conv2 = 128 * 128 * 3 * 3 = 147,456
params_bn2 = 128 * 2 = 256

total_per_block = 295,424
```

### 3. Squeeze-Excitation Block（可選）

**作用**：注意力機制，讓網絡關注重要特徵

```python
class SEBlock(nn.Module):
    def __init__(self, filters, reduction=16):
        super().__init__()
        # Global Average Pooling（自適應）
        self.gap = nn.AdaptiveAvgPool2d(1)

        # 壓縮
        self.fc1 = nn.Linear(filters, filters // reduction)

        # 恢復
        self.fc2 = nn.Linear(filters // reduction, filters)

    def forward(self, x):
        batch, channels, _, _ = x.size()

        # 全局池化: (batch, channels, 19, 19) → (batch, channels, 1, 1)
        squeeze = self.gap(x).view(batch, channels)

        # 計算通道權重
        excitation = self.fc1(squeeze)
        excitation = F.relu(excitation)
        excitation = self.fc2(excitation)
        excitation = torch.sigmoid(excitation).view(batch, channels, 1, 1)

        # 重新加權
        return x * excitation
```

**效果對比**：
```python
# 無 SE：所有特徵平等對待
features = [f1, f2, f3, ..., f128]
output = features

# 有 SE：重要特徵加強，無關特徵削弱
weights = [0.1, 0.9, 0.5, ..., 0.3]
output = features * weights
```

**參數量增加**：
```python
# filters=128, reduction=16
params_se = (128 * (128/16)) + ((128/16) * 128)
          = 128 * 8 + 8 * 128
          = 2,048  # 僅約 0.7% 的一個 residual block
```

### 4. Policy Head（策略頭）

**作用**：預測每個位置的落子概率

```python
class PolicyHead(nn.Module):
    def __init__(self, input_filters, board_size=19):
        super().__init__()

        # 降維卷積
        self.conv = nn.Conv2d(input_filters, 2, kernel_size=1)
        self.bn = nn.BatchNorm2d(2)

        # 全連接層
        self.fc = nn.Linear(2 * board_size * board_size, board_size * board_size + 1)
        # +1 for pass move

    def forward(self, x):
        # x: (batch, filters, 19, 19)

        # 降維
        x = F.relu(self.bn(self.conv(x)))
        # x: (batch, 2, 19, 19)

        # 展平
        batch_size = x.size(0)
        x = x.view(batch_size, -1)
        # x: (batch, 722)  # 2*19*19

        # 輸出 logits
        policy_logits = self.fc(x)
        # policy_logits: (batch, 362)  # 19*19+1

        return policy_logits
```

**輸出解釋**：
```python
policy_logits = model.policy_head(features)
# 形狀: (batch, 362)

# 轉換為概率
policy_probs = F.softmax(policy_logits, dim=1)

# 範例輸出:
# policy_probs[0, 0]   = 0.001  # 位置 (0, 0) 的概率
# policy_probs[0, 180] = 0.025  # 位置 (9, 9) 的概率
# policy_probs[0, 361] = 0.003  # Pass 的概率
```

**參數量**：
```python
# input_filters=128
params_conv = 128 * 2 * 1 * 1 = 256
params_bn = 2 * 2 = 4
params_fc = (2 * 19 * 19) * 362 = 261,524

total = 261,784
```

### 5. Value Head（價值頭）

**作用**：評估當前局面的勝率

```python
class ValueHead(nn.Module):
    def __init__(self, input_filters, board_size=19):
        super().__init__()

        # 降維卷積
        self.conv = nn.Conv2d(input_filters, 1, kernel_size=1)
        self.bn = nn.BatchNorm2d(1)

        # 全連接層
        self.fc1 = nn.Linear(board_size * board_size, 256)
        self.fc2 = nn.Linear(256, 1)

    def forward(self, x):
        # x: (batch, filters, 19, 19)

        # 降維
        x = F.relu(self.bn(self.conv(x)))
        # x: (batch, 1, 19, 19)

        # 展平
        batch_size = x.size(0)
        x = x.view(batch_size, -1)
        # x: (batch, 361)

        # 隱藏層
        x = F.relu(self.fc1(x))
        # x: (batch, 256)

        # 輸出
        value = torch.tanh(self.fc2(x))
        # value: (batch, 1), 範圍 [-1, 1]

        return value
```

**輸出解釋**：
```python
value = model.value_head(features)
# 形狀: (batch, 1)

# 範例輸出:
# value = 0.65  → 黑方勝率約 65%
# value = -0.3  → 白方勝率約 65%（黑方 35%）
# value = 0.0   → 局面均衡
```

**參數量**：
```python
# input_filters=128
params_conv = 128 * 1 * 1 * 1 = 128
params_bn = 1 * 2 = 2
params_fc1 = 361 * 256 = 92,416
params_fc2 = 256 * 1 = 256

total = 92,802
```

## 完整前向傳播流程

```python
def forward(self, x):
    """
    Args:
        x: (batch, 7, 19, 19) - 輸入特徵平面

    Returns:
        policy_logits: (batch, 362) - 每個位置的 logits
        value: (batch, 1) - 勝率評估 [-1, 1]
    """
    # 1. Input Convolution
    x = F.relu(self.input_bn(self.input_conv(x)))
    # x: (batch, 128, 19, 19)

    # 2. Residual Blocks
    for block in self.residual_blocks:
        x = block(x)
    # x: (batch, 128, 19, 19)  # filters 可能變化

    # 3. 雙頭輸出
    policy_logits = self.policy_head(x)  # (batch, 362)
    value = self.value_head(x)           # (batch, 1)

    return policy_logits, value
```

## 參數量分析

### 基準架構（5 blocks, 128 filters）

```python
genome = ArchitectureGenome.create_baseline(board_size=19)
model = genome.to_pytorch_model()

# 參數分布
Input Conv:       8,192      (0.5%)
Residual Blocks:  1,477,120  (96.0%)  # 5 × 295,424
Policy Head:      261,784    (17.0%)
Value Head:       92,802     (6.0%)
────────────────────────────────────
Total:            1,839,898  (100%)
```

### 不同配置的參數量

| 配置 | Blocks | Filters | 參數量 |
|------|--------|---------|--------|
| 極小型 | 3 | 64 | ~250K |
| 小型 | 5 | 96 | ~850K |
| **基準** | **5** | **128** | **~1.8M** |
| 中型 | 8 | 192 | ~7.1M |
| 大型 | 12 | 256 | ~19.7M |
| 超大型 | 20 | 512 | ~210M |

### 參數量估算公式

```python
def estimate_parameters(genome):
    # Input conv
    params = 7 * genome.base_filters * 9 + genome.base_filters

    # Residual blocks
    for i in range(genome.num_blocks):
        f = genome.filters_per_block[i]
        k = genome.kernel_sizes[i]

        # Two conv layers in residual block
        params += 2 * (f * f * k * k + f * 2)

        # SE block (optional)
        if genome.use_se_blocks:
            r = genome.se_reduction_ratio
            params += f * (f // r) + (f // r) * f

    # Policy head
    last_filters = genome.filters_per_block[-1]
    params += last_filters * 2 * 1 * 1  # conv 1x1
    params += 2 * 2                     # bn
    params += 2 * 19 * 19 * (19 * 19 + 1)  # fc

    # Value head
    params += last_filters * 1 * 1 * 1  # conv 1x1
    params += 1 * 2                     # bn
    params += 19 * 19 * 256 + 256       # fc1
    params += 256 * 1 + 1               # fc2

    return params
```

## 測試模型

### 基本推理測試

```python
import torch

# 創建模型
genome = ArchitectureGenome.create_baseline(board_size=19)
model = genome.to_pytorch_model(device='cpu')
model.eval()

# 創建隨機輸入（模擬棋盤）
batch_size = 4
test_input = torch.randn(batch_size, 7, 19, 19)

# 推理
with torch.no_grad():
    policy_logits, value = model(test_input)

# 檢查輸出形狀
assert policy_logits.shape == (batch_size, 362)
assert value.shape == (batch_size, 1)

# 檢查值域
assert value.min() >= -1.0 and value.max() <= 1.0

print("✅ 模型推理測試通過！")
```

### 性能測試

```python
import time

# 推理速度測試
model.eval()
test_input = torch.randn(1, 7, 19, 19)

# 預熱
for _ in range(10):
    with torch.no_grad():
        model(test_input)

# 計時
num_iterations = 100
start_time = time.time()

with torch.no_grad():
    for _ in range(num_iterations):
        model(test_input)

elapsed_time = time.time() - start_time
avg_time = elapsed_time / num_iterations * 1000  # ms

print(f"平均推理時間: {avg_time:.2f} ms")
print(f"吞吐量: {1000/avg_time:.1f} 局/秒")

# 典型結果（CPU）:
# 小型模型（~1M 參數）: ~10 ms/局
# 中型模型（~5M 參數）: ~30 ms/局
# 大型模型（~20M 參數）: ~100 ms/局
```

## 實踐練習

### 練習 1：構建並檢查模型

```python
# 創建不同大小的模型
configs = [
    (3, 64),   # 極小
    (5, 128),  # 基準
    (10, 256), # 大型
]

for blocks, filters in configs:
    genome = ArchitectureGenome(
        num_blocks=blocks,
        base_filters=filters,
        board_size=19
    )

    model = genome.to_pytorch_model()
    params = sum(p.numel() for p in model.parameters())

    print(f"Blocks={blocks}, Filters={filters}")
    print(f"  參數量: {params:,}")
    print(f"  估算: {genome.estimate_parameters():,}")
    print(f"  誤差: {abs(params - genome.estimate_parameters()):,}")
```

### 練習 2：可視化模型結構

```python
from torchsummary import summary

genome = ArchitectureGenome.create_baseline(board_size=19)
model = genome.to_pytorch_model(device='cpu')

# 打印模型結構
summary(model, input_size=(7, 19, 19))

# 輸出範例:
# ----------------------------------------------------------------
#         Layer (type)               Output Shape         Param #
# ================================================================
#             Conv2d-1         [-1, 128, 19, 19]           8,064
#        BatchNorm2d-2         [-1, 128, 19, 19]             256
#   ResidualBlock-3         [-1, 128, 19, 19]         295,424
#   ...
# ================================================================
# Total params: 1,839,898
# ================================================================
```

### 練習 3：比較不同架構的計算量

```python
from thop import profile

# 小模型
small = ArchitectureGenome(num_blocks=3, base_filters=64, board_size=19)
small_model = small.to_pytorch_model()

# 大模型
large = ArchitectureGenome(num_blocks=15, base_filters=256, board_size=19)
large_model = large.to_pytorch_model()

# 計算 FLOPs
input_tensor = torch.randn(1, 7, 19, 19)

small_flops, small_params = profile(small_model, inputs=(input_tensor,))
large_flops, large_params = profile(large_model, inputs=(input_tensor,))

print(f"Small: {small_flops/1e9:.2f} GFLOPs, {small_params/1e6:.2f}M params")
print(f"Large: {large_flops/1e9:.2f} GFLOPs, {large_params/1e6:.2f}M params")
print(f"計算量比: {large_flops/small_flops:.1f}x")
```

## 重點總結

### 核心概念

1. **基因組 → 模型**：`genome.to_pytorch_model()` 將設計圖變成實體
2. **雙頭架構**：Policy Head（下哪）+ Value Head（誰贏）
3. **殘差連接**：解決深層網絡訓練困難
4. **參數分布**：Residual Blocks 佔 90%+

### 參數量與性能

```
參數量 ∝ (filters)² × num_blocks × (kernel_size)²

性能提升 ∝ log(參數量)  # 邊際效益遞減
計算量 ∝ 參數量         # 線性增長
```

### 設計權衡

- **深度 vs 推理速度**：更深 = 更慢
- **寬度 vs 內存**：更寬 = 更多內存
- **大核 vs 計算量**：7×7 ≈ 5.4× 計算於 3×3
- **SE Blocks**：+5% 計算，+1-2% 性能

## 下一課預告

在[第五課：完整的演化策略](lesson-5-evolution-strategy.md)中，您將學習：

- 如何組合突變和雜交建立演化系統
- 選擇、繁殖、替換的完整流程
- 多目標優化（性能、效率、新穎性）
- 多樣性維持策略

現在我們有了**可運行的模型**，下一步是讓它們**競爭演化**！

---

[← 上一課：雜交機制](lesson-3-crossover.md) | [返回大綱](index.md) | [下一課：演化策略 →](lesson-5-evolution-strategy.md)

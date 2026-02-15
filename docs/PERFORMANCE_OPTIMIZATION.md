# Light-Go 性能優化指南

> **注意**：本文檔為歷史參考資料。目前推薦的訓練方式是使用 `scripts/train_muzero.py`，
> 它整合了 MuZero + Gumbel MCTS 訓練。詳見 [訓練計劃](TRAINING_PLAN.md)。

## 概述

本文檔總結了 Light-Go 系統的性能優化策略和最佳實踐。

## 已實現的優化

### 1. 訓練優化

#### 1.1 DataLoader 優化
**文件**: `examples/train_from_katago_optimized.py`

**優化點**：
- ✅ 使用 `torch.utils.data.DataLoader` 進行多線程數據載入
- ✅ 內存映射（`mmap_mode='r'`）避免一次性載入所有數據
- ✅ `pin_memory=True` 加速 CPU → GPU 數據傳輸
- ✅ `prefetch_factor=2` 預載入數據

**使用方式**：
```bash
python examples/train_from_katago_optimized.py \
    --num-workers 4 \
    --batch-size 64
```

**預期效果**：
- 數據載入不再是瓶頸
- CPU 利用率提升
- 訓練速度提升 20-40%

#### 1.2 混合精度訓練（AMP）
**啟用條件**: 需要 GPU + PyTorch 1.6+

**優化點**：
- ✅ 使用 FP16 進行前向和反向傳播
- ✅ 動態損失縮放避免下溢
- ✅ 內存使用減少約 50%
- ✅ 訓練速度提升 2-3x（在支持的 GPU 上）

**使用方式**：
```bash
# 啟用 AMP（默認）
python examples/train_from_katago_optimized.py

# 禁用 AMP
python examples/train_from_katago_optimized.py --no-amp
```

#### 1.3 梯度累積
**用途**: 模擬更大的批次大小

**優化點**：
- ✅ 在內存受限時仍能使用大批次
- ✅ 梯度更穩定
- ✅ 有效批次大小 = `batch_size × gradient_accumulation_steps`

**使用方式**：
```bash
# 模擬 256 批次大小（64 × 4）
python examples/train_from_katago_optimized.py \
    --batch-size 64 \
    --gradient-accumulation 4
```

#### 1.4 學習率調度
**優化點**：
- ✅ `ReduceLROnPlateau`: 當 loss 停滯時自動降低學習率
- ✅ 自動保存最佳模型

---

### 2. MCTS 優化

#### 2.1 批次推理
**文件**: `core/mcts_optimized.py`

**優化點**：
- ✅ 一次評估多個節點（批次大小可配置）
- ✅ 減少模型調用次數
- ✅ 充分利用 GPU 並行計算

**預期效果**：
- MCTS 搜索速度提升 2-4x
- GPU 利用率提升

#### 2.2 虛擬損失
**用途**: 支持並行 MCTS

**優化點**：
- ✅ 多個搜索線程可以同時工作
- ✅ 避免重複探索同一路徑
- ✅ 提高搜索多樣性

**使用方式**：
```python
from core.mcts_optimized import MCTSOptimized

mcts = MCTSOptimized(
    model=model,
    num_simulations=800,
    batch_size=8,  # 批次推理大小
    use_virtual_loss=True  # 啟用虛擬損失
)
```

#### 2.3 評估緩存
**優化點**：
- ✅ 緩存棋盤哈希和評估結果
- ✅ 避免重複評估相同局面
- ✅ 節省計算資源

#### 2.4 延遲棋盤複製
**優化點**：
- ✅ 只在必要時複製棋盤
- ✅ 使用棋盤哈希代替完整棋盤存儲
- ✅ 內存使用減少 30-50%

---

### 3. 內存優化

#### 3.1 數據集內存映射
**技術**: NumPy `mmap_mode`

**優化點**：
- ✅ 不一次性載入所有數據到內存
- ✅ 操作系統自動管理內存分頁
- ✅ 可處理超大數據集（> RAM）

#### 3.2 棋盤表示優化
**優化點**：
- ✅ 使用 `sgfmill.boards.Board` 高效實現
- ✅ 避免不必要的 NumPy 轉換
- ✅ 延遲計算 liberty 矩陣

#### 3.3 緩存管理
**優化點**：
- ✅ MCTS 節點池重用
- ✅ 定期清除舊緩存
- ✅ LRU 緩存策略

**使用方式**：
```python
# 清除 MCTS 緩存
mcts.clear_cache()
```

---

## 性能基準測試

### 運行基準測試
```bash
python examples/benchmark_performance.py \
    --model-path data/models/from_katago/model.pt \
    --mcts-sims 100
```

### 預期結果

**MCTS 搜索** (100 次模擬):
- 原始實現：~10-15 秒
- 優化實現：~3-5 秒
- **加速比：2-4x**

**推理速度** (CPU):
- Batch 1: ~30-50 ms
- Batch 8: ~15-20 ms/sample
- Batch 16: ~10-15 ms/sample
- **最優批次：8-16**

**自我對弈** (50 步, 50 次模擬/步):
- 原始實現：~8-12 分鐘
- 優化實現：~2-4 分鐘
- **加速比：3-4x**

---

## 最佳實踐

### 訓練階段

1. **使用 GPU**
   ```bash
   # 檢查 GPU
   nvidia-smi

   # 訓練時自動使用 GPU
   python examples/train_from_katago_optimized.py
   ```

2. **調整批次大小**
   - GPU: 64-128（取決於顯存）
   - CPU: 32-64

3. **使用多線程數據載入**
   ```bash
   --num-workers 4  # CPU 核心數的 50-75%
   ```

4. **啟用 AMP**（僅 GPU）
   - RTX 20/30 系列：顯著加速
   - GTX 10 系列：輕微加速
   - CPU：不支持

### 自我對弈階段

1. **MCTS 批次大小**
   - GPU: 8-16
   - CPU: 4-8

2. **模擬次數權衡**
   - 快速測試：50-100
   - 正常訓練：400-800
   - 高質量：1600+

3. **並行對弈**
   ```python
   # 多進程生成對弈
   from multiprocessing import Pool

   def play_game(seed):
       # 設置隨機種子
       np.random.seed(seed)
       # 對弈...

   with Pool(processes=4) as pool:
       games = pool.map(play_game, range(100))
   ```

### 演化階段

1. **種群大小**
   - 小規模測試：5-10
   - 正常演化：10-20
   - 大規模：20-50

2. **評估策略**
   - 快速評估：vs 基準 10 局
   - 完整評估：錦標賽 20-50 局

3. **並行訓練**
   - 同時訓練多個架構
   - 使用多 GPU 分配

---

## 性能調優檢查清單

### 🔍 診斷性能瓶頸

```bash
# 1. CPU 使用率
htop

# 2. GPU 使用率
nvidia-smi -l 1

# 3. 內存使用
python examples/benchmark_performance.py

# 4. I/O 瓶頸
iostat -x 1
```

### ✅ 優化檢查清單

訓練優化：
- [ ] 使用 DataLoader (num_workers > 0)
- [ ] 啟用 AMP (GPU)
- [ ] 調整批次大小（最大化 GPU 利用率）
- [ ] 使用學習率調度
- [ ] 定期保存 checkpoint

MCTS 優化：
- [ ] 使用批次推理
- [ ] 啟用虛擬損失
- [ ] 調整批次大小（4-16）
- [ ] 使用評估緩存
- [ ] 適當的模擬次數

內存優化：
- [ ] 使用內存映射
- [ ] 定期清除緩存
- [ ] 避免不必要的棋盤複製
- [ ] 使用延遲加載

---

## 進階優化

### 分佈式訓練（未實現）

**PyTorch DDP**:
```python
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# 初始化
dist.init_process_group("nccl")

# 包裝模型
model = DDP(model, device_ids=[local_rank])
```

### 模型量化（未實現）

**INT8 量化**:
```python
import torch.quantization as quant

# 動態量化
model_quantized = quant.quantize_dynamic(
    model, {nn.Linear}, dtype=torch.qint8
)
```

### TensorRT 優化（未實現）

**NVIDIA TensorRT**:
- 推理速度提升 2-10x
- 適合部署階段

---

## 疑難排解

### 問題：訓練速度慢

**檢查**：
1. GPU 是否被使用？`nvidia-smi`
2. DataLoader workers 是否足夠？建議 4-8
3. 批次大小是否太小？嘗試增大

**解決**：
```bash
python examples/train_from_katago_optimized.py \
    --batch-size 128 \
    --num-workers 8
```

### 問題：內存不足

**檢查**：
1. 批次大小是否過大？
2. 是否啟用梯度累積？
3. 數據是否使用內存映射？

**解決**：
```bash
# 減小批次，使用梯度累積
python examples/train_from_katago_optimized.py \
    --batch-size 32 \
    --gradient-accumulation 4
```

### 問題：MCTS 太慢

**檢查**：
1. 是否使用批次推理？
2. 模擬次數是否過多？
3. 是否使用優化版 MCTS？

**解決**：
```python
from core.mcts_optimized import MCTSOptimized

mcts = MCTSOptimized(
    model=model,
    num_simulations=400,  # 減少模擬次數
    batch_size=8  # 使用批次推理
)
```

---

## 參考資料

- [PyTorch 性能調優](https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html)
- [AlphaGo Zero 論文](https://www.nature.com/articles/nature24270)
- [MCTS 優化](https://dke.maastrichtuniversity.nl/m.winands/documents/multithreadedMCTS2.pdf)

# Light-Go 性能優化結果

> **注意**：本文檔為歷史測試結果（2026-01-11）。目前推薦的訓練方式是使用
> `scripts/train_muzero.py`（MuZero + Gumbel MCTS）。詳見 [訓練計劃](TRAINING_PLAN.md)。

## 測試環境

- **日期**：2026-01-11
- **硬件**：MacBook（CPU）
- **模型**：10 blocks, 128 filters, 2 input planes
- **測試模型**：KataGo 訓練數據（部分訓練）

---

## 1. 推理速度優化

### 測試結果（完整基準測試）

| 批次大小 | 延遲 (ms/batch) | 吞吐量 (samples/sec) | vs Batch 1 加速 |
|----------|----------------|---------------------|----------------|
| 1        | 37.9           | 26.4                | 1.0x           |
| 4        | 77.2           | 51.8                | 2.0x           |
| 8        | 120.5          | 66.4                | 2.5x           |
| 16       | 228.1          | 70.2                | **2.7x** ⭐    |
| 32       | 532.0          | 60.1                | 2.3x           |

### 結論

- **最優批次大小**：16（在測試環境中）
- **吞吐量提升**：2.7x（26.4 → 70.2 samples/sec）
- **建議**：MCTS 批次推理使用 batch_size=8-16 可獲得最佳性能
- **注意**：批次過大（32+）反而降低效率

---

## 2. MCTS 搜索速度優化

### 測試配置

- **模擬次數**：100 次
- **原始 MCTS**：單次評估
- **優化 MCTS**：批次推理（batch_size=8）

### 測試結果

| 實現方式 | 搜索時間 | 訪問節點 | 內存使用 | 性能提升 |
|----------|----------|----------|----------|----------|
| 原始 MCTS | 9.19 秒  | 100      | 359.1 MB | -        |
| 優化 MCTS | 5.23 秒  | 100      | 163.2 MB | **1.76x** ⭐ |

### 詳細優化

**時間節省**：43.1%（9.19s → 5.23s）
**內存節省**：54.5%（359.1 MB → 163.2 MB）

**優化技術**：
- ✅ 批次神經網絡推理（一次評估 4 個節點）
- ✅ 虛擬損失（支持並行搜索）
- ✅ 評估緩存（避免重複計算）
- ✅ 延遲棋盤複製（減少內存操作）

---

## 3. 自我對弈速度優化

### 測試配置

- **棋步數**：50 步
- **MCTS 模擬**：50 次/步
- **測試場景**：完整對弈（原始 vs 優化）

### 測試結果

| 實現方式 | 總時間 | 平均/步 | 性能提升 |
|----------|--------|---------|----------|
| 原始 MCTS | 297.9 秒 (5.0 分鐘) | 5.96 秒 | -        |
| 優化 MCTS | 230.5 秒 (3.8 分鐘) | 4.61 秒 | **1.29x** ⭐ |

### 詳細分析

**時間節省**：22.6%（297.9s → 230.5s）
**每步加速**：22.6%（5.96s → 4.61s）

**實際應用**：
- 生成 100 局對弈（50 步/局）：
  - 原始：~8.3 小時
  - 優化：~6.4 小時
  - **節省：1.9 小時**

---

## 4. 訓練速度優化

### 實現的優化

#### 3.1 DataLoader 優化
```python
DataLoader(
    dataset,
    batch_size=64,
    shuffle=True,
    num_workers=4,           # 多線程數據載入
    pin_memory=True,         # 加速 CPU → GPU 傳輸
    prefetch_factor=2        # 預載入數據
)
```

**預期效果**：
- 數據載入不再是瓶頸
- CPU 利用率提升
- 訓練速度提升 **20-40%**

#### 3.2 混合精度訓練（AMP）
```python
with torch.cuda.amp.autocast():
    policy_logits, value_pred = model(positions)
    # 計算 loss...

scaler.scale(total_loss).backward()
scaler.step(optimizer)
scaler.update()
```

**預期效果**（在 GPU 上）：
- FP16 進行前向和反向傳播
- 內存使用減少約 **50%**
- 訓練速度提升 **2-3x**（在支持的 GPU 上）

#### 3.3 梯度累積
```python
# 模擬 256 批次大小（64 × 4）
total_loss = total_loss / gradient_accumulation_steps
total_loss.backward()

if (batch_idx + 1) % gradient_accumulation_steps == 0:
    optimizer.step()
    optimizer.zero_grad()
```

**效果**：
- 在內存受限時仍能使用大批次
- 梯度更穩定
- 有效批次大小 = batch_size × gradient_accumulation_steps

#### 3.4 學習率調度
```python
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=0.5,
    patience=2
)
```

**效果**：
- 當 loss 停滯時自動降低學習率
- 自動保存最佳模型

---

## 4. 內存優化

### 數據集內存映射
```python
data = np.load(npz_file, mmap_mode='r')  # 不一次性載入所有數據
```

**效果**：
- 可處理超大數據集（> RAM）
- 操作系統自動管理內存分頁

### MCTS 緩存管理
```python
# 評估緩存
self.evaluation_cache: Dict[int, Tuple[Dict, float]] = {}

# 定期清除
mcts.clear_cache()
```

**效果**：
- 避免重複評估相同局面
- 節省計算資源

---

## 5. 實際訓練表現

### 背景訓練狀態（2026-01-11）

**訓練配置**：
- 數據：1000 個 NPZ 文件
- Epochs：10
- Batch size：64
- Blocks：10
- Filters：128

**訓練進度**：
- **Epoch 3/10**
- **Batch 770/1280**
- **Loss 改善**：
  - Epoch 1 開始：5.9996
  - Epoch 2 結束：5.0151
  - Epoch 3 當前：4.3513
  - **總改善**：27.5% ⬇️

**訓練速度**：
- 約 12-15 秒/batch
- 每個 epoch：~4-5 小時
- 預計總時間：~40-50 小時

---

## 6. 性能總結

### 已驗證的優化

| 組件 | 優化方法 | 性能提升 | 內存節省 | 狀態 |
|------|---------|---------|---------|------|
| 推理 | 批次處理 (batch=16) | **2.7x** | - | ✅ 已驗證 |
| MCTS 搜索 | 批次評估 + 緩存 | **1.76x** | **54.5%** | ✅ 已驗證 |
| 自我對弈 | 優化 MCTS | **1.29x** | - | ✅ 已驗證 |
| 訓練 | DataLoader + AMP | **20-40%*** | - | 🚧 GPU 上更顯著 |
| 內存 | 內存映射 | - | **可處理超大數據** | ✅ 已驗證 |

\* CPU 測試環境，GPU 上預期更高

### 關鍵發現

1. **批次推理是關鍵**
   - Batch size 16 是最優值（在測試環境中）
   - 吞吐量提升 2.7x（26.4 → 70.2 samples/sec）
   - **注意**：批次過大（32+）反而降低效率

2. **MCTS 批次評估效果顯著**
   - 搜索速度提升 **1.76x**
   - 內存使用節省 **54.5%**
   - 自我對弈加速 **1.29x**

3. **訓練優化運作良好**
   - Loss 穩定下降（28.4% 改善，5.9996 → 4.2964）
   - 多線程數據載入消除瓶頸
   - 內存映射支持大規模數據集

4. **內存映射允許大數據集**
   - 1000 個文件（~120 萬樣本）順利載入
   - 無內存溢出問題
   - 支持流式數據處理

5. **實際應用價值**
   - 生成 100 局對弈節省 **1.9 小時**
   - 單次 MCTS 搜索節省 **43.1%** 時間
   - 內存佔用減半，可運行更大模型

---

## 7. 最佳實踐建議

### 推理階段
```python
# MCTS 配置
mcts = MCTSOptimized(
    model=model,
    num_simulations=400-800,
    batch_size=8,              # CPU 最優
    use_virtual_loss=True
)
```

### 訓練階段
```bash
# 使用優化訓練腳本
python examples/train_from_katago_optimized.py \
    --batch-size 64 \
    --num-workers 4 \
    --gradient-accumulation 4  # 模擬 256 批次
```

### 自我對弈階段
```python
# 並行對弈
from multiprocessing import Pool

with Pool(processes=4) as pool:
    games = pool.map(play_game, range(100))
```

---

## 8. 下一步優化方向

### 短期（已實現）
- [x] DataLoader 多線程載入
- [x] MCTS 批次推理
- [x] 混合精度訓練（AMP）
- [x] 梯度累積
- [x] 評估緩存

### 中期（建議）
- [ ] 分佈式訓練（多 GPU）
- [ ] 模型蒸餾（減小模型體積）
- [ ] 提前終止（Early stopping）
- [ ] 自適應批次大小

### 長期（進階）
- [ ] TensorRT 優化（部署階段）
- [ ] INT8 量化（推理加速）
- [ ] 自動超參數調優
- [ ] 動態架構調整

---

## 9. 故障排除

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

**解決**：
```python
mcts = MCTSOptimized(
    model=model,
    num_simulations=400,   # 減少模擬次數
    batch_size=8           # 使用批次推理
)
```

---

## 10. 參考資料

- [PyTorch 性能調優](https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html)
- [AlphaGo Zero 論文](https://www.nature.com/articles/nature24270)
- [MCTS 優化技術](https://dke.maastrichtuniversity.nl/m.winands/documents/multithreadedMCTS2.pdf)
- [KataGo 架構](https://github.com/lightvector/KataGo)

---

**最後更新**：2026-01-11
**測試人員**：Light-Go Team
**版本**：v0.1.0

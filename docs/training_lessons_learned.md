# MuZero 訓練經驗教訓

> 記錄日期：2026-02-14
> 訓練配置：b28c512 (108M 參數)

---

## 問題總結

### 1. 訓練速度過慢

**現象**：
- 每 100 batches 需要 7.5 分鐘
- 每 epoch 需要 6.5 天
- 50 epochs 需要約 11 個月

**原因**：
- 流式載入 (StreamingNPZDataset) 造成 I/O 瓶頸
- `num_workers=0` 無法並行載入資料
- `shuffle=False` 導致連續讀取不同檔案，快取命中率低

**教訓**：
- 流式載入適合記憶體受限環境，但會大幅降低訓練速度
- 如果資料能放入記憶體，優先使用預載模式

---

### 2. MPS Shader 編譯延遲

**現象**：
- 訓練開始後卡住約 10-15 分鐘無輸出
- 程序 CPU 使用率接近 0%

**原因**：
- 108M 參數模型首次在 Apple MPS 運行時
- PyTorch 需要編譯大量 Metal shaders
- 28 個殘差塊 = 大量運算核心

**教訓**：
- MPS 首次編譯需要耐心等待，不要誤判為卡住
- 可以考慮先用小 batch 「暖機」觸發編譯

---

### 3. macOS DataLoader 多進程問題

**現象**：
- `num_workers=4` 造成程序卡住或 crash
- DataLoader 初始化時間過長

**原因**：
- macOS 的 fork() 與 Python multiprocessing 相容性問題
- 特別是使用 MPS 時更容易出問題

**解決方案**：
```python
num_workers = 0  # macOS 上避免多進程問題
```

**教訓**：
- 在 macOS + MPS 環境下，建議 `num_workers=0`
- 如需多進程，考慮使用 `multiprocessing_context='spawn'`

---

### 4. 大資料集的 random_split 問題

**現象**：
- 對 300 萬+ 樣本做 `random_split()` 耗時過長
- 記憶體使用量暴增

**原因**：
- PyTorch `random_split` 會建立完整的索引列表
- 對大資料集效率極低

**解決方案**：
- 流式模式下跳過 train/val 分割
- 或預先分割檔案而非樣本

---

### 5. 記憶體需求估算

**實測數據**：

| 配置 | 記憶體使用 |
|------|-----------|
| 模型 (108M params) | ~1.7 GB |
| 預載 620 萬樣本 | ~26 GB |
| 流式載入 | ~4 GB |
| 訓練激活 (batch=32) | ~2 GB |

**教訓**：
- 16GB RAM 無法預載 620 萬樣本
- 必須使用流式載入或限制樣本數

---

## 建議的訓練配置

### 方案 A：速度優先（需要 32GB+ RAM）

```python
{
    'streaming': False,           # 預載到記憶體
    'batch_size': 64,
    'gradient_accumulation': 2,
    'num_workers': 4,             # 可以用多進程
    'max_samples': 2_000_000,     # 限制樣本數
}
```

預估速度：每 epoch 約 2-3 小時

### 方案 B：記憶體優先（16GB RAM）

```python
{
    'streaming': True,
    'batch_size': 32,
    'gradient_accumulation': 4,
    'num_workers': 0,
    'prefetch_files': 50,         # 預取更多檔案到快取
}
```

預估速度：每 epoch 約 6-8 天（太慢，不建議）

### 方案 C：雲端訓練（推薦）

使用 Google Colab Pro / AWS / GCP：
- 32GB+ RAM
- NVIDIA GPU (比 MPS 快 5-10 倍)
- 可以預載全部資料

預估速度：每 epoch 約 1-2 小時

---

## 未來改進方向

### 1. 資料載入優化

- [ ] 實作 `ChunkedDataset`：將多個小 NPZ 合併為大檔案
- [ ] 使用 `mmap` 記憶體映射而非每次讀取
- [ ] 預先 shuffle 檔案順序以提高快取命中率

### 2. 模型優化

- [ ] 使用較小模型 (b14c256, ~27M) 進行初步實驗
- [ ] 實作 gradient checkpointing 減少記憶體
- [ ] 考慮使用 bfloat16 混合精度

### 3. 分散式訓練

- [ ] 實作多 GPU 訓練
- [ ] 考慮使用 PyTorch DDP
- [ ] 支援斷點續訓

---

## 檔案路徑參考

| 檔案 | 用途 |
|------|------|
| `scripts/train_muzero.py` | 訓練腳本 |
| `scripts/convert_katago_npz.py` | KataGo 資料轉換 |
| `core/muzero/trainer.py` | MuZero 訓練器 |
| `core/muzero/networks.py` | MuZero 網路架構 |
| `data/lightgo/training_data/katago_v2/` | 轉換後的訓練資料 |

---

## 訓練日誌位置

```bash
# 訓練日誌
/tmp/muzero_streaming.log

# 轉檔日誌
/tmp/katago_convert_v2.log
```

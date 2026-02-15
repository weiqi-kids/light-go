# Light-Go Examples

這個目錄包含展示 Light-Go 功能的範例腳本，已按功能分類組織。

## 🎯 快速導航

### 主要訓練（推薦）
- **MuZero 訓練（Phase 1 + Phase 2）** → `../scripts/train_muzero.py`

### 範例腳本
- **學習演化系統** → `tutorials/00_minimal_self_evolution.py`
- **與 AI 對弈** → `interactive/play_interactive.py`
- **評估性能** → `evaluation/evaluate_model.py`
- **運行架構演化** → `evolution/run_evolution.py`

## 📁 目錄結構

```
examples/
├── tutorials/              # 教學範例
├── training/               # 訓練腳本
├── evolution/              # 架構演化
├── analysis/               # 分析工具
├── evaluation/             # 模型評估
├── testing/                # 質量測試
├── benchmarking/           # 性能基準
└── interactive/            # 交互式工具
```

## 📚 分類說明

### tutorials/ - 教學範例（推薦從這裡開始！）

**`00_minimal_self_evolution.py`** - 最小化自我演化範例

展示 Light-Go Phase 1 核心功能：
- Architecture Genome Evolution - 架構基因組演化
- Model Construction - 模型構建
- Board Encoding - 棋盤編碼
- Model Inference - 模型推理
- Training - 訓練
- Serialization - 序列化
- Architecture Comparison - 架構對比

```bash
python examples/tutorials/00_minimal_self_evolution.py
```

**預期運行時間**: ~2-5 分鐘（GPU 更快）

**`01_architecture_evolution.py`** - 架構演化系統教程

完整的架構演化流程示範。

### training/ - 舊版訓練腳本（參考用）

> **注意**：主要訓練請使用 `scripts/train_muzero.py`

這些腳本保留供參考：
- `train_from_katago*.py` - KataGo 22-plane 格式訓練（舊方案）
- `train_lightgo_*.py` - LightGo 2/4-plane 格式訓練（舊方案）

**推薦的訓練方式**：
```bash
# MuZero Phase 1: 監督預訓練
python scripts/train_muzero.py --phase 1

# MuZero Phase 2: Gumbel MCTS 自我對弈（只需 16 次模擬）
python scripts/train_muzero.py --phase 2 --resume data/models/muzero/phase1_best.pt
```

### evolution/ - 架構演化

- `run_evolution.py` - 運行架構演化系統

### analysis/ - 分析工具

- `analyze_evolution.py` - 分析演化結果和性能

### evaluation/ - 模型評估

- `evaluate_model.py` - 評估模型性能

### testing/ - 質量測試

- `test_mcts_quality.py` - MCTS 質量測試
- `test_self_play.py` - 自我對弈測試
- `test_trained_model.py` - 訓練模型測試

### benchmarking/ - 性能基準測試

- `benchmark_performance.py` - 性能基準測試
- `quick_self_play_test.py` - 快速自我對弈測試

### interactive/ - 交互式工具

- `play_interactive.py` - 與 AI 對弈的交互式界面

## 🚀 快速開始

### 前置要求

```bash
pip install -r requirements.txt
```

或安裝最小依賴：

```bash
pip install torch>=2.0.0 numpy sgfmill
```

### 1. 學習基礎（推薦起點）

```bash
python examples/tutorials/00_minimal_self_evolution.py
```

### 2. MuZero 訓練（推薦）

```bash
# Phase 1: 監督預訓練（使用 KataGo 棋譜）
python scripts/train_muzero.py --phase 1 --epochs 50

# Phase 2: Gumbel MCTS 自我對弈
python scripts/train_muzero.py --phase 2 --resume data/models/muzero/phase1_best.pt

# 完整訓練（Phase 1 + Phase 2）
python scripts/train_muzero.py
```

### 3. 與 AI 對弈

```bash
python examples/interactive/play_interactive.py \
  --model-path data/models/muzero/phase1_best.pt
```

### 4. 評估模型性能

```bash
python examples/evaluation/evaluate_model.py \
  --model-path data/models/muzero/phase1_best.pt
```

## 📊 訓練策略對比

| 特性 | MuZero + Gumbel MCTS（推薦）| 舊 LightGo 訓練 |
|------|---------------------------|-----------------|
| **Input Planes** | 2 | 2/4 |
| **模型架構** | b28c512 (~108M 參數) | 可自訂 |
| **搜索算法** | Gumbel MCTS (16 次模擬) | 傳統 MCTS (800+) |
| **訓練方式** | Phase 1 監督 + Phase 2 自我對弈 | 僅監督學習 |
| **訓練腳本** | `scripts/train_muzero.py` | `examples/training/` |
| **推薦場景** | 生產使用 | 參考/實驗 |

## 💡 最佳實踐

1. **使用 `scripts/train_muzero.py`** 進行訓練
2. **Phase 1 完成後再進行 Phase 2**
3. **從小 epochs 開始**，驗證訓練流程
4. **定期評估模型**，追蹤性能改進
5. **保留訓練日誌**，便於後續分析

## ⚙️ 配置建議

### MuZero 訓練超參數（推薦）

**Phase 1 監督預訓練**：
- Epochs: 50
- Batch size: 32
- Gradient accumulation: 4 (等效 batch=128)
- Learning rate: 0.001
- 模型: b28c512 (28 blocks, 512 hidden)

**Phase 2 Gumbel MCTS 自我對弈**：
- Iterations: 50
- Games per iteration: 100
- Num simulations: 16 (Gumbel MCTS 只需 16 次)
- Learning rate: 0.0001 (較小！)

## 🐛 常見問題

### "No module named 'torch'"

```bash
pip install torch>=2.0.0
```

### "No module named 'sgfmill'"

```bash
pip install sgfmill>=1.1
```

### CUDA/GPU Issues

檢查 CUDA 是否可用：
```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

### 訓練時內存不足

減小 batch size 或使用更少的 filters/blocks

### Loss 不下降

- 檢查學習率
- 確認數據格式正確
- 查看是否需要更長訓練時間

### 該用 MuZero 還是舊訓練方式？

推薦使用 MuZero + Gumbel MCTS（`scripts/train_muzero.py`）。
舊方式的腳本保留在 `examples/training/` 僅供參考。

## 📚 更多資源

- [訓練計劃](../docs/TRAINING_PLAN.md)
- [專案文檔](../docs/)
- [數據目錄說明](../data/README.md)
- [訓練經驗教訓](../docs/training_lessons_learned.md)

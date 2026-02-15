# Light-Go 訓練計劃

> 最後更新：2026-02-14

---

## 目標

訓練一個強大的圍棋 AI，使用：
- **模型架構**：b28c512（28 blocks, 512 hidden size, ~108M 參數）
- **輸入編碼**：2-plane（Signed Liberties + Forbidden Points）
- **搜索算法**：Gumbel MCTS（2-16 次模擬，相當於傳統 800+ 次）

---

## 訓練階段

### Phase 1：監督預訓練

使用 KataGo 棋譜資料進行監督學習。

**配置**：
```python
{
    'board_size': 19,
    'num_input_planes': 2,
    'hidden_size': 512,
    'num_blocks': 28,
    'supervised_epochs': 50,
    'batch_size': 32,
    'gradient_accumulation': 4,  # 等效 batch_size = 128
    'learning_rate': 0.001,
    'loading_mode': 'batch',
    'batch_max_samples': 500_000,
}
```

**資料**：
- 路徑：`data/lightgo/training_data/katago_v2/`
- 檔案數：132,870 個 NPZ
- 總大小：1.5 GB

**執行**：
```bash
python scripts/train_muzero.py --phase 1
```

---

### Phase 2：Gumbel MCTS 自我對弈

使用 Gumbel MCTS 進行自我對弈強化學習。

**Gumbel MCTS 優勢**：
| 特性 | 傳統 MCTS | Gumbel MCTS |
|------|-----------|-------------|
| 模擬次數 | 800+ | 2-16 |
| 採樣方式 | UCB | Gumbel-Top-k |
| 淘汰機制 | 無 | Sequential Halving |

**配置**：
```python
{
    'selfplay_iterations': 50,
    'games_per_iteration': 100,
    'num_simulations': 16,  # Gumbel MCTS 只需 16 次
    'temperature_threshold': 30,
}
```

**執行**：
```bash
python scripts/train_muzero.py --phase 2 --resume data/models/muzero/phase1_best.pt
```

---

## 檔案結構

```
scripts/
└── train_muzero.py        # 唯一訓練腳本

core/
├── muzero/
│   ├── networks.py        # MuZero 網路（h, g, f）
│   ├── trainer.py         # 訓練器
│   └── replay_buffer.py   # 經驗回放
├── gumbel_mcts.py         # Gumbel MCTS（Phase 2 使用）
├── sequential_halving.py  # Sequential Halving 算法
└── game_rules.py          # 圍棋規則

data/
├── lightgo/training_data/katago_v2/  # 訓練資料
└── models/muzero/                     # 訓練產出
    ├── phase1_best.pt
    └── phase2_final.pt
```

---

## 硬體需求

| 配置 | 記憶體 | 預估時間 |
|------|--------|----------|
| 模型 (108M) | ~1.7 GB | - |
| Phase 1（batch 載入） | ~4 GB | 每 epoch 2-3 小時 |
| Phase 2（Gumbel MCTS） | ~4 GB | 每局 1-2 分鐘 |

**建議**：
- 本機：macOS MPS，16GB+ RAM
- 雲端：NVIDIA GPU，32GB+ RAM（推薦）

---

## 待辦事項

- [x] 資料轉換：KataGo → LightGo 2-plane
- [x] Phase 1 監督預訓練腳本
- [x] Phase 2 整合 Gumbel MCTS
- [x] 刪除重複訓練腳本
- [ ] 評估腳本

## 訓練狀態

- [ ] Phase 1 監督預訓練
- [ ] Phase 2 Gumbel MCTS 自我對弈

---

## 參考

- [Gumbel MCTS 論文](https://arxiv.org/abs/2201.03167)：Policy improvement by planning with Gumbel
- [MuZero 論文](https://arxiv.org/abs/1911.08265)：Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model

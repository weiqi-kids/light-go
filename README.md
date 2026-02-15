# Light-Go

**Light-Go** 是一個輕量級圍棋 AI 訓練框架，使用 MuZero 演算法與 Gumbel MCTS 進行高效訓練。

## 核心特點

- **2-Plane 輕量編碼**：只使用 Signed Liberties + Forbidden Points，比 KataGo 22-plane 簡潔
- **MuZero 架構**：學習環境動態模型，無需人工規則
- **Gumbel MCTS**：只需 16 次模擬達到傳統 800+ 次的效果
- **架構演化**：神經網路架構可透過基因操作自動演化

## 快速開始

### 安裝

```bash
pip install -r requirements.txt
```

### 訓練

```bash
# Phase 1: 監督預訓練（使用 KataGo 棋譜）
python scripts/train_muzero.py --phase 1 --epochs 50

# Phase 2: Gumbel MCTS 自我對弈
python scripts/train_muzero.py --phase 2 --resume data/models/muzero/phase1_best.pt

# 完整訓練（Phase 1 + Phase 2）
python scripts/train_muzero.py
```

### 測試

```bash
pytest                                    # 全部測試
pytest tests/unit/test_game_rules.py -v   # 單一檔案
```

## 訓練配置

| 參數 | 值 |
|------|-----|
| 模型架構 | b28c512（28 blocks, 512 hidden）|
| 參數量 | ~108M |
| 輸入編碼 | 2-plane（Signed Liberties + Forbidden）|
| 搜索算法 | Gumbel MCTS（16 次模擬）|

## 專案結構

```
light-go/
├── scripts/
│   └── train_muzero.py      # 主要訓練腳本
├── core/
│   ├── muzero/              # MuZero 實作
│   │   ├── networks.py      # 三大網路 (h, g, f)
│   │   ├── trainer.py       # 訓練器
│   │   └── replay_buffer.py # 經驗回放
│   ├── gumbel_mcts.py       # Gumbel MCTS
│   └── game_rules.py        # 圍棋規則
├── input/
│   └── lightgo_encoder.py   # 2-plane 編碼器
├── data/
│   ├── lightgo/training_data/katago_v2/  # 訓練資料（132,870 個 NPZ）
│   └── models/muzero/       # 訓練產出
└── docs/
    ├── TRAINING_PLAN.md     # 訓練計劃
    └── training_lessons_learned.md  # 經驗教訓
```

## Gumbel MCTS 優勢

| 特性 | 傳統 MCTS | Gumbel MCTS |
|------|-----------|-------------|
| 模擬次數 | 800+ | 2-16 |
| 採樣方式 | UCB | Gumbel-Top-k |
| 淘汰機制 | 無 | Sequential Halving |

## 文件

- [訓練計劃](docs/TRAINING_PLAN.md)
- [訓練經驗教訓](docs/training_lessons_learned.md)
- [資料目錄說明](data/README.md)
- [CLAUDE.md](CLAUDE.md) - Claude Code 指引

## API 服務

```bash
python api/gtp_server.py      # GTP 協定
python -m api.rest_api        # REST API
python -m api.websocket_api   # WebSocket API
```

## 參考

- [Gumbel MCTS 論文](https://arxiv.org/abs/2201.03167)
- [MuZero 論文](https://arxiv.org/abs/1911.08265)

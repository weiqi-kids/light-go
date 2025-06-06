# ⚫️⚪️ Weiqi.kids Light-Go 計畫

---

## 📁 專案目錄結構

```
Light-Go/
├── main.py
├── hf_models/                     # HF模型目錄
│   ├── __init__.py
│   ├── configuration_go_ai.py     # HF配置類
│   ├── modeling_go_ai.py          # HF模型類
│   ├── tokenization_go_ai.py      # 位置編碼器
│   └── processing_go_ai.py        # 數據預處理器
│
├── input/                         # 🧠 將輸入端資料統一變成棋盤輸入矩陣 liberty 與 forbidden
│ ├── sgf_to_input.py              # 處理 SGF 檔案輸入
│ ├── katago_to_input.py           # 處理 KataGo trainingdata 檔案輸入
│ ├── gtp_to_input.py              # 處理 Go Text Protocol 輸入
│ └── streaming_to_input.py        # 處理 棋盤串流影像 輸入
│
├── core/                          # 新增核心邏輯
│ ├── engine.py                    # 主引擎
│ ├── strategy_manager.py          # 自動管理所有策略
│ └── auto_learner.py              # 自動學習和發現
│
├── data/
│ ├── models                       # 保存訓練結果
│ │ ├── strategies                 # 學習到的各種策略
│ │ │ ├── a.pkl + a.pt             # 策略知識 a 模型與分配器權重
│ │ │ ├── b.pkl + b.pt             # 策略知識 b 模型與分配器權重
│ │ │ └── ...
│ │ │
│ │ ├── narrator                   # 解釋各種策略的的內容
│ │ │ ├── beginner_style.pkl
│ │ │ ├── professional_style.pkl
│ │ │ └── humorous_style.pkl
│ │ │
│ │ └── time # 學習到的時間分配
│ │   ├── v1.pkl + v1.pt           # v1 學習到的時間分配模型與分配器權重
│ │   └── playerA.pkl + playerA.pt # playerA 學習到的時間分配模型與分配器權重
│ │ 
│ ├── memory                       # 盤面狀況記錄
│ │ └── qdrant                     # Qdrant 紀錄檔
│ │ 
│ └── sgf                          # 對局紀錄
│
├── tests/                         # 🆕 測試架構
│ ├── unit_tests/
│ ├── integration_tests/
│ └── performance_tests/
│
├── tools/                         # 🆕 工具腳本
│ ├── train_strategy.py            # 訓練新策略
│ ├── evaluate_models.py           # 模型評估
│ ├── self_play_generator.py       # 生成自對弈數據
│ └── deploy_to_production.py      # 部署工具
│
├── monitoring/
│ ├── performance.py               # 效能監控
│ ├── logging.py                   # 日誌管理
│ └── health_check.py              # 健康檢查
│
└── api/                           # 🆕 API接口
  ├── rest_api.py                  # REST API
  ├── websocket_api.py             # WebSocket API
  └── gtp_interface.py             # GTP協議接口
```

---

## 🚀 啟動模式設計

### 🕹 各種服務模式

1. API服務模式（最常用）

```
python main.py --mode api --port 8080
```

啟動後可以接收HTTP請求：
 - POST /api/move
 - POST /api/analyze
 - GET /api/strategies

2. GTP協議模式（與其他軟件對接）

```
python main.py --mode gtp
```

啟動後等待GTP命令：
 - boardsize 19
 - play black D4
 - genmove white

3. 單次決策模式（快速測試）

```
python main.py --mode single --input game.sgf --move 50
```

或者

```
python main.py --mode single --liberty "[(3,4,3),(5,6,-2)]" --forbidden "[(9,10)]"
```

4. 學習/訓練模式

```
python main.py --mode train --data data/training/ --output data/models/strategies/
```
自動學習新策略

---

## 🧠 棋盤輸入矩陣介紹 liberty 與 forbidden （靜態特徵） input/

輸入：
- liberty 每個棋子位置(x座標, y座標, 氣)的無序集合，黑棋正數、白棋負數 → 簡單的3維向量
- forbidden 禁著點位置
- 隨機洗牌：消除順序偏見
- 使用 `core.liberty.count_liberties()` 可以在取得棋盤後重新計算各棋子的氣數

```
liberty = [
  (3, 4, 3),  # 黑棋3氣
  (5, 6, -2), # 白棋2氣
  // ... 只包含棋子的位置，空點不會出現
]
```

```
forbidden = [
  (9, 10),  # 位置(9,10)是禁著點
  // ... 只包含禁著的位置
]
```

```
metadata = {
  "rules": {                 # 🆕 規則設定
    "ruleset": "chinese",    # 中國規則/日本規則/韓國規則
    "komi": 7.5,             # 貼目
    "board_size": 19,        # 棋盤大小
    "handicap": 0            # 讓子數
  },
  "capture": {
    "black": 1,              # 黑棋提子
    "white": 1,              # 白棋提子
  },
  "next_move": "black",      # 下一步顏色
  "step": []                 # 對局步驟,
  "time_control": {
    "main_time_seconds": 600,
    "byo_yomi": {
      "period_time_seconds": 30,
      "periods": 3
    }
  },
  "time": [{
    "player": "black",       # 黑方
    "main_time_seconds": 100 # 基本時限剩餘秒數
    "periods": 3             # byo_yomi 剩餘次數
  },{
    "player": "white",       # 白方
    "main_time_seconds": 100 # 基本時限剩餘秒數
    "periods": 2             # byo_yomi 剩餘次數
  }]
}
```

---

## 🧪 動態萃取發現層：系統自動生成的狀態 data/models/strategies

系統運行時：
- 自動嘗試所有可用策略
- 自動評估哪個最適合當前局面
- 自動學習和更新策略
- 自動發現新策略並保存為新文件

```
特徵發現網絡：
├── 編碼器：將盤面編碼為高維潛在空間
├── 探索器：在潛在空間中尋找有意義的方向
├── 解碼器：將潛在特徵轉回可解釋的概念
└── 驗證器：測試新特徵的預測能力
```

新策略自動出現：
a.pkl → b.pkl → c.pkl → d.pkl → ...

```
系統自動學會：
├── 什麼時候用哪個策略
├── 如何分配思考時間
├── 如何發現新的模式
└── 如何持續改進
```

---

## ⏳ 用時分配  data/models/time

依照不同「strategies 耗時」與「當下剩餘時間」，決定的用時分配策略。

--- 

## 🧑‍🏫 不同風格的解說 data/narrator

--- 

## 📦 曾經下過的局面與講解紀錄，方便調閱歷史資料 data/memory/qdrant

---

# Light-Go 組件輸入/輸出規格說明

本文檔詳細說明每個組件的輸入、輸出格式，以及如何驗證正確性。

---

## 1. Liberty 編碼器 (`core/liberty.py`)

計算棋盤上每個棋子的氣數。

### 函數: `neighbors(x, y, size)`

計算相鄰座標。

| 項目 | 說明 |
|------|------|
| **輸入** | `x`: 橫座標 (0-based)<br>`y`: 縱座標 (0-based)<br>`size`: 棋盤大小 |
| **輸出** | 生成器，產生相鄰的 `(x, y)` 座標 |

```python
# 輸入
x, y, size = 1, 1, 9

# 輸出
[(0, 1), (2, 1), (1, 0), (1, 2)]  # 上下左右四個鄰居
```

**驗證方式**: 檢查輸出座標都在棋盤範圍內，且與輸入座標相鄰。

---

### 函數: `group_and_liberties(board, x, y)`

找出連接的棋子群組及其氣。

| 項目 | 說明 |
|------|------|
| **輸入** | `board`: 棋盤矩陣 (0=空, 1=黑, -1=白)<br>`x, y`: 起始座標 |
| **輸出** | `(group, liberties)` - 群組座標集合, 氣的座標集合 |

```python
# 輸入: 5x5 棋盤，中心有十字形黑棋
board = [
    [0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0],
    [0, 1, 1, 1, 0],
    [0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0],
]
x, y = 2, 2

# 輸出
group = {(2,1), (1,2), (2,2), (3,2), (2,3)}  # 5顆連接的黑棋
liberties = {(2,0), (0,2), (4,2), (2,4), ...}  # 8個氣
```

**驗證方式**:
- 群組內所有棋子顏色相同且相連
- 氣數 = 群組周圍的空點數量

---

### 函數: `count_liberties(board)`

計算整個棋盤上所有棋子的氣。

| 項目 | 說明 |
|------|------|
| **輸入** | `board`: 棋盤矩陣 |
| **輸出** | `[(x, y, liberties), ...]` - 每個棋子的座標和氣數 |

```python
# 輸入
board = [
    [0, 0, 0],
    [0, 1, 0],   # 黑棋在 (1,1)
    [0, 0, -1],  # 白棋在 (2,2)
]

# 輸出 (注意: 座標是 1-based，氣數黑正白負)
[(2, 2, 4), (3, 3, -2)]
# 黑棋有4氣，白棋有2氣（角落）
```

**驗證方式**:
- 黑棋氣數為正數
- 白棋氣數為負數
- 每個棋子都有記錄

---

## 2. 圍棋規則引擎 (`input/sgf_to_input.py`)

解析 SGF 檔案並提取棋局資訊。

### 函數: `parse_sgf(src, step, from_string)`

| 項目 | 說明 |
|------|------|
| **輸入** | `src`: SGF 檔案路徑或字串<br>`step`: 解析到第幾手 (None=全部)<br>`from_string`: 是否為字串輸入 |
| **輸出** | `(matrix, metadata, board)` |

```python
# 輸入
sgf = "(;GM[1]FF[4]SZ[9]KM[7.5]RU[Chinese];B[ee];W[gc];B[cg])"

# 輸出
matrix = [
    [0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,-1,0,0],  # 白棋 gc
    [0,0,0,0,0,0,0,0,0],
    [0,0,0,0,1,0,0,0,0],   # 黑棋 ee
    [0,0,0,0,0,0,0,0,0],
    [0,0,1,0,0,0,0,0,0],   # 黑棋 cg
    [0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0],
]

metadata = {
    "rules": {
        "ruleset": "chinese",
        "komi": 7.5,
        "board_size": 9,
        "handicap": 0
    },
    "next_move": "white",  # 下一手輪白
    "step": [("black", (4, 4)), ("white", (6, 2)), ("black", (2, 6))],
    ...
}
```

**驗證方式**:
- `matrix` 尺寸符合 `board_size`
- `step` 數量符合棋譜手數
- `next_move` 正確交替

---

### 函數: `convert(src, step, from_string)`

高階封裝，回傳完整結構化資料。

| 項目 | 說明 |
|------|------|
| **輸入** | 同 `parse_sgf` |
| **輸出** | `{"liberty": [...], "forbidden": [...], "metadata": {...}}` |

```python
# 輸出
{
    "liberty": [(4, 4, 4), (6, 2, -4), (2, 6, 3)],  # 每顆棋的氣
    "forbidden": [(3, 3), ...],  # 禁著點 (自殺點)
    "metadata": {...}
}
```

**驗證方式**:
- `liberty` 包含所有棋子
- `forbidden` 只包含真正的禁著點

---

## 3. 神經網絡模型 (`core/engine.py:GoAIModel`)

### 方法: `train(data)`

| 項目 | 說明 |
|------|------|
| **輸入** | `data`: 訓練資料列表 `[{"board": [...], "move": (x,y)}, ...]` |
| **輸出** | 無 (內部更新權重) |

```python
# 輸入
data = [
    {"board": [[0]*9 for _ in range(9)], "move": (4, 4)},
    {"board": [[0]*9 for _ in range(9)], "move": (3, 3)},
]
model.train(data)
```

**驗證方式**: 訓練後模型能產生預測。

---

### 方法: `predict(sample)`

| 項目 | 說明 |
|------|------|
| **輸入** | `sample`: `{"board": [...], "color": "black"}` |
| **輸出** | 預測的著點或 None |

```python
# 輸入
sample = {"board": [[0]*9 for _ in range(9)], "color": "black"}

# 輸出
(4, 4)  # 預測的著點座標
```

**驗證方式**: 輸出座標在棋盤範圍內且為空位。

---

### 方法: `save_pretrained(path)` / `from_pretrained(path)`

| 項目 | 說明 |
|------|------|
| **輸入** | `path`: 檔案路徑 |
| **輸出** | 儲存/載入模型 |

**驗證方式**: 儲存後檔案存在，載入後模型可用。

---

## 4. 策略管理器 (`core/strategy_manager.py`)

### 類別: `StrategyManager`

管理多個策略的註冊、載入、融合。

### 方法: `register_strategy(name, strategy)`

| 項目 | 說明 |
|------|------|
| **輸入** | `name`: 策略名稱<br>`strategy`: 實現 StrategyProtocol 的物件 |
| **輸出** | 無 |

---

### 方法: `converge(input_data, method)`

融合多個策略的預測。

| 項目 | 說明 |
|------|------|
| **輸入** | `input_data`: 棋盤資料<br>`method`: "majority_vote" / "weighted_average" / "meta" |
| **輸出** | 融合後的著點 `(x, y)` |

```python
# 輸入
input_data = {"board": [[0]*9 for _ in range(9)]}
method = "majority_vote"

# 輸出 (假設三個策略分別預測 (3,3), (3,3), (4,4))
(3, 3)  # 多數決結果
```

**驗證方式**:
- `majority_vote`: 回傳最多策略選擇的著點
- `weighted_average`: 考慮權重的平均

---

### 方法: `strategy_accepts(name, state)`

檢查策略是否接受某盤面。

| 項目 | 說明 |
|------|------|
| **輸入** | `name`: 策略名稱<br>`state`: 盤面特徵字典 |
| **輸出** | `bool` - 是否接受 |

```python
# 輸入
state = {"total_black_stones": 50, "total_white_stones": 48}

# 輸出
True  # 策略接受此盤面
```

---

## 5. 自動學習器 (`core/auto_learner.py`)

### 類別: `AutoLearner`

管理策略發現、訓練分配、性能回饋。

### 靜態方法: `_game_stats(board)`

| 項目 | 說明 |
|------|------|
| **輸入** | `board`: 棋盤矩陣 |
| **輸出** | 統計資訊字典 |

```python
# 輸入
board = [
    [0, 0, 0],
    [0, 1, 1],
    [-1, -1, 0],
]

# 輸出
{
    "black_stones": 2,
    "white_stones": 2,
    "avg_liberties_black": 4.0,
    "avg_liberties_white": 3.0
}
```

**驗證方式**: 棋子數量與棋盤一致。

---

### 方法: `discover_strategy(data)`

| 項目 | 說明 |
|------|------|
| **輸入** | `data`: 訓練資料 |
| **輸出** | `str` - 新策略名稱 |

```python
# 輸入
data = {"sample": "training_data"}

# 輸出
"a"  # 自動生成的策略名稱
```

---

### 方法: `assign_training(board_features)`

| 項目 | 說明 |
|------|------|
| **輸入** | `board_features`: 盤面特徵字典 |
| **輸出** | `list[str]` - 適用的策略名稱列表 |

---

### 方法: `receive_feedback(strategy_name, score)`

| 項目 | 說明 |
|------|------|
| **輸入** | `strategy_name`: 策略名稱<br>`score`: 0.0-1.0 的分數 |
| **輸出** | 無 (內部更新分數) |

**驗證方式**: 分數會影響後續的訓練分配。

---

## 6. 訓練循環 / Engine (`core/engine.py`)

### 方法: `train(data_dir, output_dir)`

| 項目 | 說明 |
|------|------|
| **輸入** | `data_dir`: SGF 檔案目錄<br>`output_dir`: 模型輸出目錄 |
| **輸出** | `str` - 新策略名稱 |

```python
# 輸入
data_dir = "/path/to/sgf_files/"
output_dir = "/path/to/models/"

# 輸出
"a"  # 策略名稱
# 同時產生 /path/to/models/a.pt 模型檔案
```

**驗證方式**: 輸出目錄有新的 `.pt` 檔案。

---

### 方法: `decide_move(board, color, use_mcts, mcts_iterations)`

| 項目 | 說明 |
|------|------|
| **輸入** | `board`: 棋盤矩陣<br>`color`: "black" 或 "white"<br>`use_mcts`: 是否使用 MCTS<br>`mcts_iterations`: 迭代次數 |
| **輸出** | `(x, y)` 或 `None` |

```python
# 輸入
board = [
    [0, 0, 0],
    [0, 1, 0],
    [0, 0, 0],
]
color = "black"

# 輸出
(0, 0)  # 建議的著點
```

**驗證方式**:
- 輸出座標在範圍內
- 輸出位置是空的
- 不是自殺著點

---

## 7. MCTS 搜索 (`core/mcts.py`)

### 類別: `GoGameState`

遊戲狀態管理。

### 方法: `get_legal_moves()`

| 項目 | 說明 |
|------|------|
| **輸入** | 無 (使用內部狀態) |
| **輸出** | `list[(x, y)]` - 合法著點列表 |

```python
# 5x5 空棋盤
state = GoGameState([[0]*5 for _ in range(5)], current_color=1)

# 輸出
[(0,0), (1,0), (2,0), ..., (4,4)]  # 25 個合法著點
```

**驗證方式**:
- 不包含已有棋子的位置
- 不包含劫點
- 不包含自殺點

---

### 方法: `play_move(move)`

| 項目 | 說明 |
|------|------|
| **輸入** | `move`: `(x, y)` 或 `None` (虛手) |
| **輸出** | 新的 `GoGameState` |

```python
# 輸入: 黑棋下在 (1,1) 完成提子
capture_board = [
    [0, 1, 0],
    [1, -1, 0],  # 白棋被圍
    [0, 1, 0],
]
state = GoGameState(capture_board, current_color=1)
new_state = state.play_move((2, 1))  # 黑下 (2,1)

# 輸出: 白棋被提走
new_state.board[1][1] == 0  # True，白棋消失
```

**驗證方式**:
- 棋子正確放置
- 被吃的棋子被移除
- 劫點正確設置

---

### 類別: `MCTSNode`

### 方法: `ucb1(exploration)`

| 項目 | 說明 |
|------|------|
| **輸入** | `exploration`: 探索常數 (預設 √2 ≈ 1.414) |
| **輸出** | `float` - UCB1 值 |

```
UCB1 = wins/visits + exploration * sqrt(ln(parent_visits) / visits)
```

```python
# 輸入
parent = MCTSNode(visits=100)
child = MCTSNode(parent=parent, visits=10, wins=6)

# 輸出
ucb1 = child.ucb1(1.414)
# ≈ 0.6 + 1.414 * sqrt(ln(100)/10) ≈ 1.56
```

**驗證方式**: 值在合理範圍 (通常 0-3)。

---

### 類別: `MCTS`

### 方法: `search(board, color, komi)`

| 項目 | 說明 |
|------|------|
| **輸入** | `board`: 棋盤矩陣<br>`color`: 1=黑, -1=白<br>`komi`: 貼目 |
| **輸出** | `(x, y)` 或 `None` |

```python
# 輸入
board = [[0]*9 for _ in range(9)]
mcts = MCTS(iterations=1000)

# 輸出
move = mcts.search(board, color=1, komi=7.5)
# 可能是 (4, 4) - 天元，或其他好點
```

**驗證方式**:
- 輸出是合法著點
- 增加迭代次數應該讓結果更穩定

---

### 方法: `get_move_probabilities(board, color, komi)`

| 項目 | 說明 |
|------|------|
| **輸入** | 同 `search` |
| **輸出** | `dict[(x,y), float]` - 著點機率分佈 |

```python
# 輸出
{
    (4, 4): 0.15,  # 天元被訪問 15%
    (2, 2): 0.08,  # 星位被訪問 8%
    (3, 3): 0.12,
    ...
}
# 所有機率總和 = 1.0
```

**驗證方式**:
- 所有機率總和 ≈ 1.0
- 機率最高的通常是好的著點

---

## 8. 自我對弈引擎 (`api/gtp_interface.py`)

### 類別: `GTPServer`

Go Text Protocol 伺服器。

### GTP 指令: `play color vertex`

| 項目 | 說明 |
|------|------|
| **輸入** | `color`: "black" 或 "white"<br>`vertex`: "D4" 或 "pass" |
| **輸出** | `= ` (成功) 或 `? error` (失敗) |

```
輸入: play black D4
輸出: =

輸入: play white Q16
輸出: =
```

---

### GTP 指令: `genmove color`

| 項目 | 說明 |
|------|------|
| **輸入** | `color`: 要下的顏色 |
| **輸出** | `= vertex` - AI 選擇的著點 |

```
輸入: genmove black
輸出: = E5

輸入: genmove white
輸出: = Q4
```

**驗證方式**:
- 回傳的著點是合法的
- 著點格式正確 (字母+數字)

---

## 快速驗證指南

執行驗證腳本：

```bash
# 驗證所有組件
python validate_components.py

# 驗證單一組件
python validate_components.py liberty
python validate_components.py rules
python validate_components.py neural_network
python validate_components.py strategy_manager
python validate_components.py auto_learner
python validate_components.py engine
python validate_components.py mcts
python validate_components.py self_play
```

每個測試會顯示：
- **Input**: 測試輸入
- **Output**: 實際輸出
- **Status**: PASS/FAIL

---

## 資料流程圖

```
SGF 檔案
    ↓
┌─────────────────────┐
│  sgf_to_input.py    │  → matrix, metadata, liberty, forbidden
└─────────────────────┘
    ↓
┌─────────────────────┐
│  liberty.py         │  → 計算每顆棋子的氣
└─────────────────────┘
    ↓
┌─────────────────────┐
│  Engine.train()     │  → 訓練模型
│  AutoLearner        │  → 發現/管理策略
│  StrategyManager    │  → 儲存策略
└─────────────────────┘
    ↓
┌─────────────────────┐
│  MCTS.search()      │  → 搜索最佳著點
│  GoGameState        │  → 模擬對局
└─────────────────────┘
    ↓
┌─────────────────────┐
│  GTPServer          │  → 提供 GTP 介面
│  (自我對弈)          │  → 生成訓練資料
└─────────────────────┘
```

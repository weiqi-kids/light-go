# Light-Go 開發規範

本文件定義了 Light-Go 專案的開發規範，確保程式碼一致性和可維護性。

## 目錄

1. [測試開發規範](#測試開發規範)
2. [命名規則](#命名規則)
3. [檔案結構](#檔案結構)
4. [程式碼風格](#程式碼風格)

---

## 測試開發規範

### 測試目錄結構

```
tests/
├── conftest.py              # 根層級共享 fixtures（所有測試繼承）
├── components/              # 元件測試（使用真實實現）
│   ├── conftest.py          # 元件專用 fixtures
│   └── test_*.py
├── unit/                    # 單元測試（使用真實實現，輕量級）
│   └── test_*.py
├── integration/             # 整合測試
│   └── test_*.py
├── e2e/                     # 端對端測試
│   └── test_full_pipeline.py
└── performance/             # 效能測試
    └── test_performance.py
```

### 測試原則

1. **使用真實實現**：所有測試應使用真實的類別和方法，避免 Mock
2. **Fixtures 共享**：共用的 fixtures 放在 `conftest.py`，避免重複定義
3. **參數化測試**：使用 `@pytest.mark.parametrize` 減少重複程式碼
4. **類型標註**：所有 fixture 參數應加上類型標註

---

## 命名規則

### Fixture 命名

| 類型 | 命名規則 | 範例 |
|------|---------|------|
| 臨時目錄 | `tmp_path`（pytest 內建）| `tmp_path: Path` |
| 空棋盤 | `empty_board_{size}` | `empty_board_9x9` |
| 帶棋子棋盤 | `board_with_{pattern}` | `board_with_cross_pattern` |
| 棋盤工廠 | `make_board` | `make_board(size, stones)` |
| SGF 內容 | `{desc}_sgf_content` | `simple_sgf_content` |
| 臨時目錄（含資料）| `temp_{domain}_dir` | `temp_sgf_dir` |
| 管理器 | `{domain}_manager` | `strategy_manager` |
| 帶狀態管理器 | `{domain}_manager_with_{state}` | `strategy_manager_with_strategies` |
| 伺服器實例 | `{domain}_server` | `gtp_server` |
| Mock 工廠 | `mock_{domain}` | `mock_strategy` |
| 共享實例（優化）| `shared_{domain}` | `shared_engine`（class scope）|
| 預訓練工廠 | `{domain}_factory` | `trained_engine_factory`（module scope）|

### 參數命名

| 概念 | 標準名稱 | 避免使用 |
|------|---------|---------|
| 棋盤大小 | `size` | `board_size`, `sz` |
| 棋子顏色 | `color` | `current_color`, `to_move`, `next_move` |
| 棋步 | `move` | `coord`, `position` |
| 輸入資料 | `input_data` | `data`, `sample` |
| 輸出目錄 | `output_dir` | `out_dir`, `outdir` |
| 資料目錄 | `data_dir` | `input_dir`, `sgf_dir` |
| 策略目錄 | `strategy_dir` | `strat_dir` |
| 貼目 | `komi` | `komi_value` |
| 預測結果 | `prediction` | `pred`, `result` |

### 測試類別命名

```python
# 格式：Test{Component}{Feature}
class TestGTPServerInstantiation:    # 元件 + 功能
class TestGTPBasicCommands:          # 元件 + 命令類別
class TestStrategyManagerSaveLoad:   # 元件 + 操作
class TestConvergence:               # 概念/演算法
class TestEdgeCases:                 # 邊界情況（每個檔案最多一個）
```

### 測試函數命名

```python
# 格式：test_{action}_{detail}
def test_create_server(self):
def test_protocol_version(self, gtp_server: GTPServer):
def test_boardsize_valid(self, gtp_server: GTPServer, size: int):
def test_play_multiple_moves(self, gtp_server: GTPServer):
def test_genmove_after_play(self, gtp_server: GTPServer):
```

---

## 檔案結構

### 測試檔案結構

```python
"""模組說明文件（必須）。

描述測試的元件和功能。
"""
from __future__ import annotations

import pytest
from typing import TYPE_CHECKING

# 專案內部 imports
from api.gtp_interface import GTPServer

if TYPE_CHECKING:
    from core.strategy_manager import StrategyManager


# ---------------------------------------------------------------------------
# Fixtures（如果有本地 fixtures）
# ---------------------------------------------------------------------------

@pytest.fixture
def local_fixture() -> SomeType:
    """Fixture 說明。"""
    return SomeType()


# ---------------------------------------------------------------------------
# Test Classes
# ---------------------------------------------------------------------------

class TestComponentFeature:
    """測試類別說明。"""

    def test_basic_functionality(self, fixture: Type):
        """測試基本功能。"""
        result = fixture.method()
        assert result == expected

    @pytest.mark.parametrize("param", [1, 2, 3], ids=["case1", "case2", "case3"])
    def test_parametrized(self, fixture: Type, param: int):
        """參數化測試。"""
        assert fixture.compute(param) > 0
```

### Import 順序

```python
# 1. Future imports
from __future__ import annotations

# 2. 標準函式庫
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

# 3. 第三方套件
import pytest
import numpy as np

# 4. 專案內部模組
from core.engine import Engine
from api.gtp_interface import GTPServer
```

**重要**：不要在測試檔案中手動操作 `sys.path`，根層級 `conftest.py` 已處理。

---

## 程式碼風格

### Pytest Fixture 範例

```python
# 基本 fixture
@pytest.fixture
def gtp_server() -> GTPServer:
    """Return a fresh GTPServer instance."""
    return GTPServer()

# 工廠 fixture
@pytest.fixture
def mock_strategy():
    """Factory fixture to create MockStrategy instances."""
    def _create(
        prediction: Any = None,
        stable: bool = False,
    ) -> MockStrategy:
        return MockStrategy(prediction=prediction, stable=stable)
    return _create

# 帶 scope 的 fixture（用於優化）
@pytest.fixture(scope="class")
def shared_engine(tmp_path_factory) -> Engine:
    """Class-scoped Engine for tests that don't modify state."""
    tmp_dir = tmp_path_factory.mktemp("shared_engine")
    return Engine(str(tmp_dir))
```

### 參數化測試範例

```python
class TestGTPBoardConfiguration:
    """Tests for board configuration commands."""

    @pytest.mark.parametrize("size", [9, 13, 19], ids=["9x9", "13x13", "19x19"])
    def test_boardsize(self, gtp_server: GTPServer, size: int):
        """boardsize sets board size correctly."""
        response, should_quit = gtp_server.handle_boardsize([str(size)])

        assert gtp_server.board_size == size
        assert should_quit is False

    @pytest.mark.parametrize(
        "color,move",
        [
            ("black", "D4"),
            ("white", "Q16"),
            ("black", "PASS"),
        ],
        ids=["black_d4", "white_q16", "black_pass"],
    )
    def test_play(self, gtp_server: GTPServer, color: str, move: str):
        """play records moves correctly."""
        response, should_quit = gtp_server.handle_play([color, move])

        assert (color, move) in gtp_server.moves
        assert should_quit is False
```

### 整合測試範例

```python
class TestGTPIntegration:
    """Integration tests simulating real game sequences."""

    def test_full_game_sequence(self, gtp_server: GTPServer):
        """Simulate a complete game setup and play sequence."""
        # Setup
        gtp_server.handle_boardsize(["19"])
        gtp_server.handle_komi(["6.5"])
        gtp_server.handle_clear_board([])

        # Play opening moves
        gtp_server.handle_play(["black", "D4"])
        gtp_server.handle_play(["white", "Q16"])

        # Assertions
        assert gtp_server.board_size == 19
        assert gtp_server.komi == 6.5
        assert len(gtp_server.moves) == 2
```

---

## Fixture Scope 使用指南

| Scope | 使用時機 | 範例 |
|-------|---------|------|
| `function`（預設）| 每個測試需要新實例 | 大部分 fixtures |
| `class` | 同一類別內的測試可共享 | `shared_engine` |
| `module` | 整個模組內可共享，初始化成本高 | `trained_engine_factory` |
| `session` | 整個測試 session 共享 | 很少使用 |

---

## 檢查清單

新增測試時，請確認：

- [ ] 使用真實實現而非 Mock
- [ ] Fixture 命名符合規範
- [ ] 參數命名符合規範
- [ ] 測試類別和函數命名符合規範
- [ ] 參數化測試使用 `ids` 參數
- [ ] 所有 fixture 參數都有類型標註
- [ ] Import 順序正確
- [ ] 沒有手動操作 `sys.path`
- [ ] 測試函數有 docstring

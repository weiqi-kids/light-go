# 架構演化系統深度教學

## 課程簡介

這個深度教學將帶您完整理解 Light-Go 的核心創新：**如何讓神經網絡架構自己演化**。

與傳統圍棋 AI（如 KataGo）不同，Light-Go 不僅訓練模型的權重，更能演化模型的架構本身。這是一個革命性的創新，讓 AI 能夠「學習如何學習」。

## 學習目標

完成本課程後，您將理解：

- ✅ 架構基因組的結構與組成
- ✅ 突變（Mutation）如何產生新架構
- ✅ 雜交（Crossover）如何結合父代優勢
- ✅ 基因組如何轉換成真正的 PyTorch 神經網絡
- ✅ 完整的演化策略（選擇、繁殖、替換）
- ✅ 為什麼架構演化是革命性創新

## 課程大綱

### [第一課：架構基因組的 DNA](lesson-1-genome-dna.md)
**學習時間**: 5 分鐘

了解架構基因組（ArchitectureGenome）的基本組成，包括網絡深度、寬度、卷積核大小、模塊類型等關鍵參數。

**關鍵概念**:
- 基因組的 7 個核心參數
- 每個參數如何影響神經網絡性能
- 參數量估算

### [第二課：突變 - 架構如何進化](lesson-2-mutation.md)
**學習時間**: 5 分鐘

深入理解突變機制，了解架構如何通過隨機變化產生新的變體，以及如何保持可訓練性。

**關鍵概念**:
- 突變率的選擇
- 5 種突變類型
- 漸進式變化策略
- 保持架構合法性

### [第三課：雜交 - 結合父代的優勢](lesson-3-crossover.md)
**學習時間**: 5 分鐘

學習如何結合兩個成功架構的優點，創造出新的架構，以及如何追蹤世代和血統。

**關鍵概念**:
- 深度繼承策略
- 交叉點選擇
- 特性繼承
- 世代追蹤與血統管理

### [第四課：基因組變成真正的神經網絡](lesson-4-genome-to-model.md)
**學習時間**: 5 分鐘

探索如何將抽象的基因組「編譯」成可訓練的 PyTorch 模型，以及模型的具體結構。

**關鍵概念**:
- Input Convolution
- Residual Blocks 構建
- Policy Head 與 Value Head
- 參數量驗證

### [第五課：完整的演化策略](lesson-5-evolution-strategy.md)
**學習時間**: 5 分鐘

了解如何組合突變和雜交，建立一個完整的演化系統，包括選擇、繁殖、替換等步驟。

**關鍵概念**:
- 多目標優化（性能、效率、新穎性）
- 精英保留策略
- 錦標賽選擇
- 多樣性維持
- 演化循環流程

### [第六課：為什麼架構演化是革命性的創新](lesson-6-why-revolutionary.md)
**學習時間**: 5 分鐘

對比傳統方法與 Light-Go 方法，理解架構演化的革命性優勢和長期願景。

**關鍵概念**:
- 傳統 AI vs Light-Go
- 自動架構發現
- Meta-Learning（元學習）
- 專門化架構的發現
- 長期演化願景

## 實踐教學

### 互動式教學腳本

運行完整的互動式教學：

```bash
cd /Users/lightman/weiqi.kids/light-go
python examples/architecture_evolution_tutorial.py
```

這個腳本會引導您完成所有 6 課，包含實際範例和互動演示。

### 前置需求

確保已安裝依賴：

```bash
pip install torch>=2.0.0 numpy>=1.21.0 sgfmill>=1.1
```

檢查環境：

```bash
python check_dependencies.py
```

## 推薦學習路徑

### 初學者
1. 按順序完成第一課到第六課
2. 運行互動式教學腳本
3. 閱讀源碼 `core/architecture_genome.py`
4. 嘗試修改基因組參數

### 進階學習者
1. 快速瀏覽課程大綱
2. 深入研究感興趣的特定課程
3. 實驗自定義演化策略
4. 參與 Phase 2 開發（MCTS + 自我對弈）

## 相關資源

### 核心代碼文件
- `core/architecture_genome.py` - 架構基因組實現（370 行）
- `hf_models/modeling_go_ai.py` - 神經網絡模型（452 行）
- `core/trainer.py` - 訓練循環（316 行）
- `examples/minimal_self_evolution.py` - Phase 1 完整演示

### 文檔
- [CLAUDE.md](../../CLAUDE.md) - 專案完整指南
- [訓練計劃](../TRAINING_PLAN.md) - MuZero + Gumbel MCTS 訓練計劃

### 學術背景
- Neural Architecture Search (NAS)
- Evolutionary Algorithms
- Meta-Learning
- AlphaGo Zero 架構

## 常見問題

### Q: 為什麼要演化架構，而不是只訓練權重？
A: 固定架構可能不是最優的。通過演化架構，系統可以發現人類想不到的更好結構，實現真正的「學習如何學習」。

### Q: 演化會不會很慢？
A: 我們使用權重繼承（Net2Net）技術，新架構不需要從零訓練，可以節省 50-70% 的時間。

### Q: 如何防止演化停滯？
A: 通過多樣性維持機制（Novelty Search、Speciation、隨機注入）確保種群持續探索新架構。

### Q: 這和 AutoML 有什麼不同？
A: Light-Go 是在線演化系統，架構會在使用過程中持續改進。AutoML 通常是離線搜索，找到架構後就固定了。

## 下一步

完成課程後，建議：

1. **實驗基因組** - 創建自定義架構，觀察訓練效果
   ```python
   from core.architecture_genome import ArchitectureGenome

   custom_genome = ArchitectureGenome(
       num_blocks=10,
       base_filters=256,
       use_se_blocks=True,
       # ... 更多參數
   )
   model = custom_genome.to_pytorch_model()
   ```

2. **參與開發** - 實現 Phase 2（MCTS + 自我對弈）

3. **閱讀計劃** - 了解完整的 8 週實現路徑

4. **貢獻代碼** - 提交 PR 改進演化算法

---

**準備好開始了嗎？** 從[第一課：架構基因組的 DNA](lesson-1-genome-dna.md) 開始您的學習之旅！

或者直接運行互動式教學：
```bash
python examples/architecture_evolution_tutorial.py
```

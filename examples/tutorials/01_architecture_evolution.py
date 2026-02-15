"""架構演化系統深度教學
=========================

這個互動教學將帶您深入理解 Light-Go 的核心創新：
如何讓神經網絡架構自己演化。

運行這個腳本，您將學習：
1. 架構基因組的 DNA
2. 突變如何改變架構
3. 雜交如何結合父代
4. 從基因組構建實際模型
5. 為什麼這是革命性的創新
"""

import torch
import json
from typing import Dict, Any
from core.architecture_genome import ArchitectureGenome, create_initial_population


def print_header(title: str, level: int = 1):
    """打印格式化標題"""
    if level == 1:
        print("\n" + "=" * 80)
        print(f"  {title}")
        print("=" * 80)
    else:
        print(f"\n{'─' * 80}")
        print(f"  {title}")
        print("─" * 80)


def print_genome_details(genome: ArchitectureGenome, name: str = "Genome"):
    """詳細顯示基因組內容"""
    print(f"\n🧬 {name}:")
    print(f"   ID: {genome.get_id()}")
    print(f"   世代: {genome.generation}")
    print(f"   父代: {genome.parent_ids if genome.parent_ids else '無（初始）'}")
    print(f"\n   架構參數:")
    print(f"   • 網絡深度: {genome.num_blocks} 個 residual blocks")
    print(f"   • 基礎寬度: {genome.base_filters} filters")
    print(f"   • 每層寬度: {genome.filters_per_block}")
    print(f"   • 卷積核大小: {genome.kernel_sizes}")
    print(f"   • Block 類型: {genome.block_types}")
    print(f"   • SE blocks: {'啟用' if genome.use_se_blocks else '禁用'} (reduction={genome.se_reduction_ratio})")
    print(f"   • Dropout: {genome.dropout_rate}")
    print(f"\n   估算參數量: ~{genome.estimate_parameters():,}")


def compare_genomes(genome1: ArchitectureGenome, genome2: ArchitectureGenome):
    """比較兩個基因組的差異"""
    print("\n📊 基因組比較:")
    print(f"\n   {'屬性':<20} {'基因組 1':<20} {'基因組 2':<20} {'差異'}")
    print("   " + "-" * 70)

    # 比較各項參數
    attrs = [
        ('num_blocks', '網絡深度'),
        ('base_filters', '基礎寬度'),
        ('use_se_blocks', 'SE Blocks'),
        ('dropout_rate', 'Dropout率')
    ]

    for attr, label in attrs:
        val1 = getattr(genome1, attr)
        val2 = getattr(genome2, attr)
        if val1 == val2:
            diff = "相同"
        else:
            diff = f"{val1} → {val2}"
        print(f"   {label:<20} {str(val1):<20} {str(val2):<20} {diff}")

    # 參數量比較
    params1 = genome1.estimate_parameters()
    params2 = genome2.estimate_parameters()
    ratio = params2 / params1 if params1 > 0 else 0
    print(f"\n   參數量: {params1:,} → {params2:,} ({ratio:.2f}x)")


def lesson_1_understanding_genome():
    """第一課：理解架構基因組"""
    print_header("第一課：架構基因組的 DNA", level=1)

    print("""
架構基因組（ArchitectureGenome）是一組「基因」，描述了神經網絡的結構。
就像生物的 DNA 決定生物的特徵，基因組決定了神經網絡的架構。

讓我們創建一個基準基因組，看看它包含什麼：
    """)

    # 創建基準基因組
    baseline = ArchitectureGenome.create_baseline(board_size=19)
    print_genome_details(baseline, "基準基因組")

    print("""
🔍 關鍵概念解釋：

1. **num_blocks** (網絡深度)
   - 決定有多少個 residual blocks
   - 更深 = 更強表達能力，但訓練更難

2. **base_filters** (基礎寬度)
   - 每層卷積的 filter 數量
   - 更寬 = 更多特徵，但計算更慢

3. **filters_per_block** (每層的具體寬度)
   - 可以讓每層有不同寬度
   - 例如：[128, 256, 256, 128] 先變寬再變窄

4. **kernel_sizes** (卷積核大小)
   - 3x3：看局部模式
   - 5x5：看更大範圍
   - 7x7：看更廣的棋形

5. **block_types** (模塊類型)
   - 'residual': 標準 ResNet block
   - 'dense': DenseNet 風格（更多連接）
   - 'bottleneck': 瓶頸結構（減少計算）

6. **use_se_blocks** (Squeeze-Excitation)
   - 加入注意力機制
   - 讓網絡關注重要特徵

7. **dropout_rate** (隨機丟棄率)
   - 防止過擬合
   - 0.0 = 不用，0.3 = 丟棄 30% 神經元
    """)

    input("按 Enter 繼續下一課...")
    return baseline


def lesson_2_mutation(baseline: ArchitectureGenome):
    """第二課：突變機制"""
    print_header("第二課：突變 - 架構如何進化", level=1)

    print("""
突變（Mutation）是進化的核心機制。
每個基因都有一定概率被隨機改變，產生新的架構變體。

讓我們觀察幾次突變，看看會發生什麼：
    """)

    # 進行多次突變展示
    for i in range(3):
        print_header(f"突變實驗 {i+1}", level=2)
        mutated = baseline.mutate(mutation_rate=0.3)

        print(f"\n原始基因組 ID: {baseline.get_id()}")
        print(f"突變基因組 ID: {mutated.get_id()}")

        compare_genomes(baseline, mutated)

        if i < 2:
            input("\n按 Enter 看下一次突變...")

    print("""
🔍 突變策略分析：

1. **突變率** (mutation_rate)
   - 0.2 表示每個基因有 20% 機率突變
   - 太高：變化太大，可能破壞好的架構
   - 太低：進化太慢

2. **突變類型**
   - 深度突變：±1 block
   - 寬度突變：在預設值中選擇 (64, 128, 256, 512...)
   - 卷積核突變：在 3, 5, 7 中選擇
   - 類型突變：切換 block 類型
   - 特性突變：開關 SE blocks、調整 dropout

3. **漸進式變化**
   - 深度變化：±1 (不會一次跳太多)
   - 有偏向性：residual block 機率更高（已被驗證有效）

4. **保持可訓練性**
   - 突變後的架構仍然是合法的
   - 參數範圍有限制（3-20 blocks，64-512 filters）
    """)

    input("按 Enter 繼續下一課...")
    return mutated


def lesson_3_crossover(baseline: ArchitectureGenome, mutated: ArchitectureGenome):
    """第三課：雜交機制"""
    print_header("第三課：雜交 - 結合父代的優勢", level=1)

    print("""
雜交（Crossover）結合兩個父代的基因，創造新的架構。
這類似生物的有性繁殖，可以組合兩個成功架構的優點。

讓我們看看兩個架構如何雜交：
    """)

    # 創建另一個不同的架構作為第二個父代
    parent2 = ArchitectureGenome(
        num_blocks=8,
        base_filters=256,
        filters_per_block=[256, 256, 384, 384, 384, 384, 256, 256],
        kernel_sizes=[3, 3, 5, 5, 5, 5, 3, 3],
        block_types=['residual'] * 8,
        use_se_blocks=True,
        board_size=19
    )

    print_genome_details(baseline, "父代 1（基準）")
    print_genome_details(parent2, "父代 2（自定義）")

    # 雜交
    print("\n🧬 執行雜交...")
    child = ArchitectureGenome.crossover(baseline, parent2)

    print_genome_details(child, "子代（雜交結果）")

    print("""
🔍 雜交策略分析：

1. **深度繼承**
   - 取兩個父代深度的平均值
   - 父代 1: 5 blocks, 父代 2: 8 blocks → 子代: 6 blocks

2. **寬度選擇**
   - 隨機選擇一個父代的 base_filters
   - 保持一致性，不混合

3. **交叉點（Crossover Point）**
   - 在某個 block 位置切割
   - 前半段來自父代 1，後半段來自父代 2
   - 例如：[P1, P1, P1, P2, P2, P2]

4. **特性繼承**
   - SE blocks, dropout 等特性隨機選擇一方
   - 或取平均（如 dropout rate）

5. **世代追蹤**
   - 子代世代 = max(父代世代) + 1
   - 記錄兩個父代的 ID，可追溯血統
    """)

    # 展示血統追蹤
    print("\n📜 血統追蹤:")
    print(f"   父代 1 ID: {baseline.get_id()} (世代 {baseline.generation})")
    print(f"   父代 2 ID: {parent2.get_id()} (世代 {parent2.generation})")
    print(f"   子代 ID: {child.get_id()} (世代 {child.generation})")
    print(f"   父代記錄: {child.parent_ids}")

    input("按 Enter 繼續下一課...")
    return child, parent2


def lesson_4_genome_to_model(genome: ArchitectureGenome):
    """第四課：從基因組構建模型"""
    print_header("第四課：基因組變成真正的神經網絡", level=1)

    print("""
基因組只是一組參數描述。真正神奇的是：
我們可以從基因組「編譯」出一個真正可訓練的 PyTorch 模型！

讓我們看看這個轉換過程：
    """)

    print(f"\n📋 基因組規格:")
    print(f"   • {genome.num_blocks} blocks")
    print(f"   • 平均 {sum(genome.filters_per_block)//len(genome.filters_per_block)} filters")
    print(f"   • 估算參數: ~{genome.estimate_parameters():,}")

    print("\n🏗️  正在構建 PyTorch 模型...")
    model = genome.to_pytorch_model(device='cpu')

    # 計算實際參數
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\n✅ 模型構建完成！")
    print(f"   • 實際參數量: {total_params:,}")
    print(f"   • 可訓練參數: {trainable_params:,}")
    print(f"   • 估算誤差: {abs(total_params - genome.estimate_parameters()):,}")

    print("\n🔍 模型結構:")
    print(model)

    print("""
🔍 轉換過程解析：

1. **Input Convolution**
   - 7 個輸入平面 → base_filters 個特徵圖
   - 3x3 卷積 + BatchNorm

2. **Residual Blocks** (根據 num_blocks)
   - 每個 block: Conv → BN → ReLU → Conv → BN → (+residual) → ReLU
   - 使用 filters_per_block 決定每層寬度
   - kernel_sizes 決定卷積核大小

3. **Policy Head**
   - 特徵 → 2 filters (Conv 1x1)
   - Flatten
   - FC → 362 個輸出 (19x19+1 for pass)

4. **Value Head**
   - 特徵 → 1 filter (Conv 1x1)
   - Flatten
   - FC(256) → FC(1) → Tanh
   - 輸出 [-1, 1] 表示勝率

5. **為什麼這很強大？**
   - 同樣的「模板」可以生成無限種架構
   - 每個基因組都是獨一無二的
   - 可以自動搜索最佳架構
    """)

    # 測試模型
    print("\n🧪 測試模型推理...")
    test_input = torch.randn(1, 7, 19, 19)
    with torch.no_grad():
        policy, value = model(test_input)

    print(f"   輸入形狀: {test_input.shape}")
    print(f"   Policy 輸出: {policy.shape} (每個位置的機率)")
    print(f"   Value 輸出: {value.shape} (勝率預測)")
    print(f"   Value 範例值: {value.item():.4f}")

    input("按 Enter 繼續下一課...")
    return model


def lesson_5_evolution_strategy():
    """第五課：演化策略"""
    print_header("第五課：完整的演化策略", level=1)

    print("""
現在我們了解了突變和雜交，讓我們看看如何組合這些操作，
建立一個完整的演化系統。

模擬一個世代的演化過程：
    """)

    # 創建初始種群
    print("\n📊 初始種群（世代 0）:")
    population = create_initial_population(size=5, board_size=19, include_baseline=True)

    for i, genome in enumerate(population):
        print(f"   [{i}] {genome}")

    input("\n按 Enter 開始演化...")

    # 模擬評估（隨機分數）
    print("\n📈 評估所有架構...")
    import random
    scores = {genome.get_id(): random.uniform(0.3, 0.7) for genome in population}

    print("\n   評估結果（勝率）:")
    for genome in population:
        score = scores[genome.get_id()]
        print(f"   [{genome.get_id()[:8]}] 勝率: {score:.1%} - {genome}")

    # 選擇
    print("\n🏆 選擇階段 - 保留前 40%")
    sorted_pop = sorted(population, key=lambda g: scores[g.get_id()], reverse=True)
    elite = sorted_pop[:2]  # 前 2 個

    print(f"   精英保留:")
    for genome in elite:
        score = scores[genome.get_id()]
        print(f"   • [{genome.get_id()[:8]}] 勝率: {score:.1%}")

    # 雜交
    print("\n🧬 雜交階段 - 精英配對")
    offspring = []
    if len(elite) >= 2:
        child1 = ArchitectureGenome.crossover(elite[0], elite[1])
        offspring.append(child1)
        print(f"   • 子代 1: {child1}")

    # 突變
    print("\n🔀 突變階段 - 精英突變")
    for genome in elite:
        mutated = genome.mutate(mutation_rate=0.2)
        offspring.append(mutated)
        print(f"   • 突變自 [{genome.get_id()[:8]}]: {mutated}")

    # 新世代
    print("\n🌱 新世代（世代 1）:")
    new_generation = elite + offspring

    for i, genome in enumerate(new_generation):
        print(f"   [{i}] Gen={genome.generation}, ID={genome.get_id()[:8]}, Params≈{genome.estimate_parameters()//1000}K")

    print(f"""
🔍 演化策略總結：

**選擇（Selection）**
• 多目標優化：
  - 70% 性能（勝率）
  - 20% 效率（參數量）
  - 10% 新穎性（多樣性）
• 精英保留（Elitism）：保留前 20%
• 錦標賽選擇：隨機配對競爭

**繁殖（Reproduction）**
• 雜交：精英之間配對
• 突變：每個精英產生變體
• 隨機注入：防止收斂到局部最優

**替換（Replacement）**
• 精英 + 子代組成新世代
• 保持種群大小穩定
• 弱者自然淘汰

**多樣性維持**
• Novelty Search：獎勵不同的架構
• Speciation：維護多個亞群
• 定期隨機注入新基因

**這個循環會持續進行：**
1. 評估（對弈測試）
2. 選擇（保留強者）
3. 繁殖（雜交 + 突變）
4. 替換（形成新世代）
5. 重複...

經過數十個世代後，架構會越來越強！
    """)

    input("按 Enter 繼續最後一課...")


def lesson_6_why_revolutionary():
    """第六課：為什麼這是革命性的"""
    print_header("第六課：為什麼架構演化是革命性的創新", level=1)

    print("""
讓我們對比傳統方法和 Light-Go 的方法：
    """)

    print("\n" + "─" * 80)
    print("傳統圍棋 AI（如 KataGo）")
    print("─" * 80)
    print("""
1. **人工設計架構**
   • 研究員設計：「用 20 個 ResNet blocks，256 filters」
   • 基於經驗和直覺
   • 需要深厚的專業知識

2. **固定架構**
   • 架構一旦確定就不變
   • 只訓練權重，不改變結構
   • 所有改進都在權重空間

3. **手動調整**
   • 性能不好 → 人工改架構 → 重新訓練
   • 耗時、勞力密集
   • 依賴人類智慧

4. **局限性**
   • 可能錯過最佳架構
   • 受限於人類的想像力
   • 無法自動適應新問題
    """)

    print("\n" + "─" * 80)
    print("Light-Go 方法 ⭐")
    print("─" * 80)
    print("""
1. **自動架構發現**
   • 系統自己探索架構空間
   • 不需要人工設計
   • 可以發現人類想不到的架構

2. **架構會進化**
   • 策略 a, b, c... 各有不同架構
   • 好的架構保留並繁衍
   • 持續優化結構

3. **自動優化**
   • 性能不好 → 自動突變/雜交 → 產生新架構
   • 24/7 自動運行
   • 無需人工干預

4. **革命性優勢**
   ✅ 可能找到比人類設計更好的架構
   ✅ 自動適應不同棋盤大小、規則
   ✅ 持續演化，永不停止改進
   ✅ 多樣性：不同策略有不同專長
    """)

    print("\n" + "=" * 80)
    print("  實際例子：架構演化可能發現...")
    print("=" * 80)
    print("""
• **專門化架構**
  - 策略 A：深而窄（20 blocks, 128 filters）→ 擅長序盤
  - 策略 B：淺而寬（8 blocks, 384 filters）→ 擅長中盤戰鬥
  - 策略 C：混合核大小 [3,3,5,7,5,3,3] → 擅長全局判斷

• **意外發現**
  - 某個「怪異」的架構在特定局面表現超好
  - 人類永遠不會嘗試這種組合
  - 但演化會自動找到它

• **自動適應**
  - 9x9 棋盤：演化出淺而快的架構
  - 19x19 棋盤：演化出深而強的架構
  - 無需人工調整
    """)

    print("\n" + "=" * 80)
    print("  這就是為什麼 Light-Go 獨特！")
    print("=" * 80)
    print("""
🎯 **核心理念**：
   讓 AI 不僅學習「如何下棋」（訓練權重）
   還要學習「如何學習下棋」（演化架構）

🚀 **長期願景**：
   經過數月的演化，系統可能發現
   比所有人類設計更優秀的神經網絡架構

🌟 **Meta-Learning（元學習）**：
   這不是普通的機器學習
   這是「學習如何學習」的系統
    """)


def main():
    """主教學流程"""
    print("\n" + "🎓" * 40)
    print("   Light-Go 架構演化系統 - 深度教學")
    print("🎓" * 40)

    print("""
這個互動教學將帶您完整理解 Light-Go 的核心創新。
每一課都會有實際範例和詳細解釋。

請確保在一個安靜的環境，預留 15-20 分鐘。
隨時可以按 Ctrl+C 退出。

準備好了嗎？讓我們開始！
    """)

    input("按 Enter 開始第一課...")

    try:
        # 第一課：理解基因組
        baseline = lesson_1_understanding_genome()

        # 第二課：突變
        mutated = lesson_2_mutation(baseline)

        # 第三課：雜交
        child, parent2 = lesson_3_crossover(baseline, mutated)

        # 第四課：構建模型
        model = lesson_4_genome_to_model(baseline)

        # 第五課：演化策略
        lesson_5_evolution_strategy()

        # 第六課：為什麼革命性
        lesson_6_why_revolutionary()

        # 結束
        print("\n" + "🎉" * 40)
        print("   恭喜！您已經完成架構演化系統的深度學習")
        print("🎉" * 40)

        print("""
📚 您現在理解了：
   ✅ 架構基因組的結構
   ✅ 突變如何產生新架構
   ✅ 雜交如何結合父代優勢
   ✅ 基因組如何變成真正的神經網絡
   ✅ 完整的演化策略
   ✅ 為什麼這是革命性創新

🚀 下一步建議：
   • 閱讀源碼：core/architecture_genome.py
   • 實驗自定義基因組
   • 開始實現 Phase 2（MCTS + 自我對弈）

💡 記住：
   Light-Go 不僅是一個圍棋 AI
   它是一個會自己設計神經網絡的系統！
        """)

    except KeyboardInterrupt:
        print("\n\n教學中斷。隨時可以重新運行！")
    except Exception as e:
        print(f"\n\n❌ 錯誤: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

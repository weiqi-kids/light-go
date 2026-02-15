"""Sequential Halving 演算法

用於 Gumbel MCTS 的動作選擇，透過逐步淘汰較弱的候選動作，
以最小化簡單遺憾（Simple Regret）。

參考論文：
- "Policy improvement by planning with Gumbel" (Danihelka et al., 2022)
"""

from __future__ import annotations

import math
from typing import Callable, List, Tuple, Dict, Any
from dataclasses import dataclass


@dataclass
class HalvingResult:
    """Sequential Halving 結果"""
    best_action: int
    rounds_completed: int
    final_candidates: List[int]
    action_scores: Dict[int, float]


def sequential_halving(
    evaluate_fn: Callable[[int], float],
    candidates: List[int],
    total_budget: int,
    return_details: bool = False
) -> int | HalvingResult:
    """Sequential Halving 演算法

    透過多輪評估，逐步淘汰表現較差的候選動作，
    直到只剩下一個最佳動作。

    Parameters
    ----------
    evaluate_fn : Callable[[int], float]
        評估候選動作的函數。
        - 輸入：動作索引
        - 輸出：該動作的 Q 值估計（越高越好）
        每次呼叫都會執行一次模擬並回傳當前估計。
    candidates : List[int]
        候選動作列表（動作索引）
    total_budget : int
        總模擬次數預算
    return_details : bool
        是否回傳詳細結果

    Returns
    -------
    int | HalvingResult
        最佳動作索引，或包含詳細資訊的 HalvingResult

    Examples
    --------
    >>> def mock_evaluate(action):
    ...     # 模擬評估函數
    ...     return np.random.randn() + (action == 5) * 0.5  # action 5 略優
    >>> best = sequential_halving(mock_evaluate, list(range(10)), total_budget=50)
    >>> print(f"Best action: {best}")
    """
    n = len(candidates)

    if n == 0:
        raise ValueError("候選動作列表不能為空")

    if n == 1:
        if return_details:
            return HalvingResult(
                best_action=candidates[0],
                rounds_completed=0,
                final_candidates=candidates,
                action_scores={candidates[0]: 0.0}
            )
        return candidates[0]

    # 計算所需的輪數：ceil(log2(n))
    num_rounds = max(1, math.ceil(math.log2(n)))

    # 每輪的預算
    budget_per_round = max(1, total_budget // num_rounds)

    remaining = list(candidates)
    action_scores: Dict[int, float] = {}
    rounds_completed = 0

    for round_idx in range(num_rounds):
        if len(remaining) == 1:
            break

        rounds_completed += 1

        # 每個候選動作分配的模擬次數
        sims_per_action = max(1, budget_per_round // len(remaining))

        # 評估所有剩餘候選
        scores: List[Tuple[int, float]] = []
        for action in remaining:
            # 執行多次模擬
            total_value = 0.0
            for _ in range(sims_per_action):
                total_value += evaluate_fn(action)

            # 取平均值作為最終分數
            avg_score = total_value / sims_per_action
            scores.append((action, avg_score))
            action_scores[action] = avg_score

        # 按分數排序（高到低）
        scores.sort(key=lambda x: x[1], reverse=True)

        # 保留前一半（至少保留 1 個）
        half = max(1, len(remaining) // 2)
        remaining = [action for action, _ in scores[:half]]

    best_action = remaining[0]

    if return_details:
        return HalvingResult(
            best_action=best_action,
            rounds_completed=rounds_completed,
            final_candidates=remaining,
            action_scores=action_scores
        )

    return best_action


def adaptive_sequential_halving(
    evaluate_fn: Callable[[int], float],
    candidates: List[int],
    total_budget: int,
    confidence_threshold: float = 0.9
) -> int:
    """自適應 Sequential Halving

    當領先者有足夠信心時提前終止。

    Parameters
    ----------
    evaluate_fn : Callable[[int], float]
        評估函數
    candidates : List[int]
        候選動作列表
    total_budget : int
        總預算
    confidence_threshold : float
        提前終止的信心閾值（分數差距比例）

    Returns
    -------
    int
        最佳動作
    """
    n = len(candidates)

    if n <= 1:
        return candidates[0] if candidates else -1

    num_rounds = max(1, math.ceil(math.log2(n)))
    budget_per_round = max(1, total_budget // num_rounds)

    remaining = list(candidates)
    action_visits: Dict[int, int] = {a: 0 for a in candidates}
    action_values: Dict[int, float] = {a: 0.0 for a in candidates}

    for round_idx in range(num_rounds):
        if len(remaining) == 1:
            break

        sims_per_action = max(1, budget_per_round // len(remaining))

        for action in remaining:
            for _ in range(sims_per_action):
                value = evaluate_fn(action)
                action_visits[action] += 1
                action_values[action] += value

        # 計算平均分數
        avg_scores = [
            (a, action_values[a] / action_visits[a] if action_visits[a] > 0 else 0.0)
            for a in remaining
        ]
        avg_scores.sort(key=lambda x: x[1], reverse=True)

        # 檢查是否可以提前終止
        if len(avg_scores) >= 2:
            best_score = avg_scores[0][1]
            second_score = avg_scores[1][1]

            if second_score != 0:
                margin = (best_score - second_score) / abs(second_score)
                if margin > confidence_threshold:
                    return avg_scores[0][0]

        # 保留前一半
        half = max(1, len(remaining) // 2)
        remaining = [action for action, _ in avg_scores[:half]]

    return remaining[0]

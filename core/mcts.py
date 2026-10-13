"""Monte Carlo Tree Search (MCTS) implementation for Go.

The ``'puct'`` implementation of :class:`core.search.Search`. This module
implements AlphaGo Zero style MCTS that combines:
- Policy and value predictions from a :class:`~core.evaluator.PositionEvaluator`
- UCB (Upper Confidence Bound) for exploration
- Dirichlet noise for root exploration

Value perspective convention
----------------------------
``node.color`` is the colour that just moved into the node, and
``node.q_value`` is stored from **that** player's perspective (i.e. the
parent's side to move). Selection therefore maximises ``child.q_value``
directly. Evaluator values are side-to-move perspective, so a leaf evaluated
at value ``v`` (for ``node.to_play``) is backed up as ``-v`` into the leaf and
alternates sign on the way up.
"""

from __future__ import annotations

import math
import numpy as np
from typing import Optional, List, Tuple, Dict, Hashable, Union
from dataclasses import dataclass

from core.evaluator import PositionEvaluator, as_evaluator
from core.game_rules import GoBoard
from core.search import find_matching_node, policy_to_move, position_key


def _opponent(color: str) -> str:
    return 'w' if color == 'b' else 'b'


@dataclass
class MCTSNode:
    """Node in the MCTS search tree.

    Attributes
    ----------
    board : Optional[GoBoard]
        Current board state; ``None`` for a child means "derive lazily from
        ``parent.board`` + ``move``" (most expanded children are never visited)
    parent : Optional[MCTSNode]
        Parent node
    move : Optional[Tuple[int, int]]
        Move that led to this node
    color : str
        Color that just moved ('b' or 'w'); the root uses the opponent of
        the side to move
    prior : float
        Prior probability from neural network
    children : Dict[Tuple[int, int], MCTSNode]
        Child nodes
    visit_count : int
        Number of times this node was visited
    total_value : float
        Sum of value estimates from simulations, from ``color``'s perspective
    """

    board: Optional[GoBoard]
    parent: Optional[MCTSNode] = None
    move: Optional[Tuple[int, int]] = None
    color: str = 'b'
    prior: float = 0.0

    def __post_init__(self):
        """Initialize mutable attributes."""
        if self.board is None:
            # 延遲建立：第一次讀取 ``board`` 時才由父節點盤面＋``move`` 算出
            del self.__dict__['board']
        self.children: Dict[Tuple[int, int], MCTSNode] = {}
        self.visit_count: int = 0
        self.total_value: float = 0.0
        self.virtual_loss: int = 0  # 批次搜尋中尚未回傳的路徑數

    def __getattr__(self, name: str):
        """Materialise a lazily created child board on first access."""
        if name == 'board' and self.__dict__.get('parent') is not None and self.__dict__.get('move') is not None:
            board = self.parent.board.copy()
            board.play_move(self.move[0], self.move[1], self.color)
            self.board = board
            return board
        raise AttributeError(name)

    @property
    def to_play(self) -> str:
        """Colour to move at this node."""
        return _opponent(self.color)

    @property
    def q_value(self) -> float:
        """Average action value (Q) from the parent's side-to-move perspective.

        Returns
        -------
        float
            Mean value from simulations
        """
        if self.visit_count == 0:
            return 0.0
        return self.total_value / self.visit_count

    @property
    def u_value(self) -> float:
        """Exploration bonus (U).

        Returns
        -------
        float
            UCB exploration term
        """
        if self.parent is None:
            return 0.0

        # UCB formula: U(s,a) = c_puct * P(s,a) * sqrt(sum_b N(s,b)) / (1 + N(s,a))
        c_puct = 1.4  # Exploration constant
        parent_visits = self.parent.visit_count
        return c_puct * self.prior * math.sqrt(parent_visits) / (1 + self.visit_count)

    @property
    def ucb_score(self) -> float:
        """UCB score for action selection.

        未訪問的子節點用父節點的平均價值（輪走方視角，即 ``-parent.q_value``）
        當 Q（first-play urgency）；若用 0，當所有著手價值都 > 0 時，第一個被
        訪問的子節點會吃掉全部模擬。

        虛擬損失把每條進行中的路徑視為一次「輸」（Q 下降、U 縮小），
        讓同一批次的選擇分散到不同葉節點；沒有虛擬損失時就是 Q + U。

        Returns
        -------
        float
            Q(s,a) + U(s,a)
        """
        if self.virtual_loss == 0:
            if self.visit_count == 0 and self.parent is not None and self.parent.visit_count > 0:
                return -self.parent.q_value + self.u_value
            return self.q_value + self.u_value
        visits = self.visit_count + self.virtual_loss
        q_value = (self.total_value - self.virtual_loss) / visits
        c_puct = 1.4
        parent_visits = self.parent.visit_count + self.parent.virtual_loss if self.parent else 0
        return q_value + c_puct * self.prior * math.sqrt(parent_visits) / (1 + visits)

    def is_leaf(self) -> bool:
        """Check if this is a leaf node.

        Returns
        -------
        bool
            True if node has no children
        """
        return len(self.children) == 0

    def select_child(self) -> MCTSNode:
        """Select child with highest UCB score.

        Returns
        -------
        MCTSNode
            Child node to explore
        """
        return max(self.children.values(), key=lambda node: node.ucb_score)

    def select_child_deterministic(self) -> MCTSNode:
        """Select child with highest visit count (deterministic).

        Used by Gumbel MCTS for non-root node selection.

        Returns
        -------
        MCTSNode
            Child node with most visits
        """
        return max(self.children.values(), key=lambda node: node.visit_count)

    def expand(self, policy: Dict[Tuple[int, int], float]) -> None:
        """Expand node by adding children for legal moves.

        Parameters
        ----------
        policy : Dict[Tuple[int, int], float]
            Policy probabilities from neural network
        """
        next_color = self.to_play
        legal_moves = self.board.get_legal_moves(next_color)

        for move in legal_moves:
            if move in self.children:
                continue

            # Get prior probability (use uniform prior as fallback)
            board_size = self.board.size
            default_prior = 1.0 / (board_size * board_size + 1)
            prior = policy.get(move, default_prior)

            # Create child node (board is built lazily on first access)
            child = MCTSNode(
                board=None,
                parent=self,
                move=move,
                color=next_color,
                prior=prior
            )

            self.children[move] = child

    def update(self, value: float) -> None:
        """Update node statistics after simulation.

        Parameters
        ----------
        value : float
            Value estimate (-1 to 1) from ``self.color``'s perspective
        """
        self.visit_count += 1
        self.total_value += value


class MCTS:
    """PUCT Monte Carlo Tree Search（:class:`~core.search.Search` 的 ``'puct'`` 實作）。

    也吸收了舊 ``MCTSOptimized`` 的實作細節（皆為選項）：批次葉節點評估＋虛擬損失、
    動態批次大小、以 ``(盤面, 輪走方)`` 為鍵的評估快取。

    Parameters
    ----------
    model : torch.nn.Module | PositionEvaluator, optional
        Network returning ``(policy_logits, value)`` (wrapped in a
        :class:`~core.evaluator.NetworkEvaluator`), or an evaluator directly
    num_simulations : int
        Number of simulations per search (default 800)
    temperature : float
        Temperature for action selection (default 1.0)
    dirichlet_alpha : float
        Dirichlet noise alpha for exploration (default 0.03)
    dirichlet_epsilon : float
        Fraction of Dirichlet noise to add (default 0.25)
    board_size : int
        Nominal board size (the searched board's own size is what is used)
    reuse_tree : bool
        Whether to reuse search tree between moves (default True). A cached
        tree is only reused when a node in it (root, child or grandchild)
        matches the searched position; otherwise a fresh tree is built.
    max_tree_nodes : int
        Maximum number of nodes to keep in cached tree (default 100000)
        When exceeded, less visited branches are pruned
    evaluator : PositionEvaluator, optional
        Position evaluator (takes precedence over ``model``)
    batch_size : int
        Leaves evaluated per evaluator call (default 1 = sequential search)
    use_virtual_loss : bool
        Apply virtual loss while collecting a batch so leaves diversify
    dynamic_batch : bool
        Shrink the batch from ``max_batch_size`` to ``min_batch_size`` as the
        search progresses (only when ``batch_size > 1``)
    min_batch_size, max_batch_size : int
        Bounds for ``dynamic_batch``
    eval_cache_size : int
        Max entries of the ``(position, colour)`` evaluation cache; 0 disables it
    device : optional
        Inference device when wrapping ``model``
    """

    def __init__(
        self,
        model=None,
        num_simulations: int = 800,
        temperature: float = 1.0,
        dirichlet_alpha: float = 0.03,
        dirichlet_epsilon: float = 0.25,
        board_size: int = 19,
        reuse_tree: bool = True,
        max_tree_nodes: int = 100000,
        evaluator: Optional[PositionEvaluator] = None,
        batch_size: int = 1,
        use_virtual_loss: bool = True,
        dynamic_batch: bool = False,
        min_batch_size: int = 4,
        max_batch_size: int = 32,
        eval_cache_size: int = 0,
        device=None,
    ):
        self.model = model
        self.evaluator = as_evaluator(model, evaluator, device)
        self.num_simulations = num_simulations
        self.temperature = temperature
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_epsilon = dirichlet_epsilon
        self.board_size = board_size
        self.reuse_tree = reuse_tree
        self.max_tree_nodes = max_tree_nodes
        self.batch_size = max(1, batch_size)
        self.use_virtual_loss = use_virtual_loss
        self.dynamic_batch = dynamic_batch
        self.min_batch_size = max(1, min_batch_size)
        self.max_batch_size = max(self.min_batch_size, max_batch_size)
        self.eval_cache_size = eval_cache_size

        # Tree reuse: 緩存上一次搜索的根節點
        self._cached_root: Optional[MCTSNode] = None
        # 評估快取：position_key(board, color) -> (policy dict, value)
        self.evaluation_cache: Dict[Hashable, Tuple[Dict, float]] = {}

    # ------------------------------------------------------------------
    # Search interface
    # ------------------------------------------------------------------
    def search(self, board: GoBoard, color: str) -> Tuple[np.ndarray, MCTSNode]:
        """Run MCTS search from current position.

        Parameters
        ----------
        board : GoBoard
            Current board state
        color : str
            Color to move ('b' or 'w')

        Returns
        -------
        Tuple[np.ndarray, MCTSNode]
            - Move probabilities (board_size^2 + 1 for pass)
            - Root node of search tree
        """
        root = self._find_reusable_root(board, color) if self.reuse_tree else None

        if root is None:
            root = MCTSNode(board=board.copy(), color=_opponent(color))
            policy, _ = self._evaluate(root.board, color)
            root.expand(self._add_dirichlet_noise(policy, board.size))
        else:
            # 重用子樹：斷開舊父節點，以新的根評估＋噪聲重設子節點先驗
            root.parent = None
            policy, _ = self._evaluate(root.board, color)
            noisy_policy = self._add_dirichlet_noise(policy, board.size)
            if root.is_leaf():
                root.expand(noisy_policy)
            else:
                for move, child in root.children.items():
                    child.prior = noisy_policy.get(move, child.prior)

        self._run_simulations(root)

        move_probs = self._get_move_probabilities(root, board.size)

        # 緩存根節點供下次使用
        self._cached_root = root if self.reuse_tree else None
        self._maybe_prune_tree()

        return move_probs, root

    def select_move(
        self,
        board: GoBoard,
        color: str,
        temperature: Optional[float] = None
    ) -> Optional[Tuple[int, int]]:
        """Select move using MCTS.

        Parameters
        ----------
        board : GoBoard
            Current board state
        color : str
            Color to move
        temperature : Optional[float]
            Override default temperature (0 = most visited)

        Returns
        -------
        Optional[Tuple[int, int]]
            Selected move (row, col), or None for pass
        """
        old_temp = self.temperature
        if temperature is not None:
            self.temperature = temperature
        try:
            move_probs, _ = self.search(board, color)
            # move_probs 已套用溫度，直接依比例採樣（溫度 0 時為 one-hot）
            sample_temp = 0.0 if self.temperature == 0 else 1.0
        finally:
            self.temperature = old_temp
        return policy_to_move(move_probs, board.size, sample_temp)

    def advance_tree(self, move: Optional[Tuple[int, int]]) -> bool:
        """根據落子推進搜索樹

        將緩存的根移動到對應的子節點，實現樹重用。下一次 :meth:`search`
        仍會驗證盤面是否一致，不一致就重建。

        Parameters
        ----------
        move : Optional[Tuple[int, int]]
            落子位置 (row, col)，None 表示 pass

        Returns
        -------
        bool
            是否成功推進（如果動作不在樹中則返回 False）
        """
        if self._cached_root is None:
            return False

        move_key = move if move is not None else 'pass'
        child = self._cached_root.children.get(move_key)
        if child is None:
            self._cached_root = None
            return False

        child.parent = None  # 斷開父節點連結（幫助 GC 回收舊樹）
        self._cached_root = child
        return True

    def clear_tree(self) -> None:
        """清除緩存的搜索樹"""
        self._cached_root = None

    def clear_cache(self) -> None:
        """清除評估快取"""
        self.evaluation_cache.clear()

    # ------------------------------------------------------------------
    # Tree reuse
    # ------------------------------------------------------------------
    def _find_reusable_root(self, board: GoBoard, color: str) -> Optional[MCTSNode]:
        """在快取樹的根、子、孫節點中找與 (board, color) 相同的局面；找不到就丟掉快取樹。"""
        node = find_matching_node(self._cached_root, board, color)
        if node is None:
            self._cached_root = None
        return node

    def get_tree_stats(self) -> Dict[str, int]:
        """獲取緩存樹的統計信息

        Returns
        -------
        Dict[str, int]
            包含 'visit_count', 'num_children', 'tree_depth' 等信息
        """
        if self._cached_root is None:
            return {'cached': False, 'visit_count': 0, 'num_children': 0}

        total_nodes = 0
        max_depth = 0
        stack = [(self._cached_root, 0)]
        while stack:
            node, depth = stack.pop()
            total_nodes += 1
            max_depth = max(max_depth, depth)
            stack.extend((child, depth + 1) for child in node.children.values())

        return {
            'cached': True,
            'visit_count': self._cached_root.visit_count,
            'num_children': len(self._cached_root.children),
            'total_nodes': total_nodes,
            'tree_depth': max_depth
        }

    def _prune_tree(self, keep_ratio: float = 0.5) -> int:
        """剪枝搜索樹，移除訪問次數較少的分支

        Parameters
        ----------
        keep_ratio : float
            保留訪問次數前 keep_ratio 比例的子節點

        Returns
        -------
        int
            剪枝後的節點數
        """
        if self._cached_root is None:
            return 0

        def prune_node(node: MCTSNode, depth: int = 0) -> int:
            """遞歸剪枝節點"""
            if not node.children:
                return 1

            sorted_children = sorted(
                node.children.items(),
                key=lambda x: x[1].visit_count,
                reverse=True
            )

            # 計算要保留的子節點數量（至少保留 1 個）；深度越深，保留越少
            keep_count = max(1, int(len(sorted_children) * keep_ratio))
            if depth > 3:
                keep_count = max(1, keep_count // 2)

            node.children = dict(sorted_children[:keep_count])
            return 1 + sum(prune_node(child, depth + 1) for child in node.children.values())

        return prune_node(self._cached_root)

    def _maybe_prune_tree(self) -> None:
        """如果樹太大，執行剪枝"""
        if self._cached_root is None:
            return

        stats = self.get_tree_stats()
        if stats.get('total_nodes', 0) > self.max_tree_nodes:
            self._prune_tree(keep_ratio=0.5)

    # ------------------------------------------------------------------
    # Simulation
    # ------------------------------------------------------------------
    def _run_simulations(self, root: MCTSNode) -> None:
        """執行 ``num_simulations`` 次模擬（每批收集多個葉節點一次評估）。"""
        done = 0
        while done < self.num_simulations:
            batch = min(self._get_batch_size(done), self.num_simulations - done)
            apply_vl = self.use_virtual_loss and batch > 1

            leaves: List[MCTSNode] = []
            for _ in range(batch):
                node = root
                while not node.is_leaf():
                    node = node.select_child()
                    if apply_vl:
                        node.virtual_loss += 1
                leaves.append(node)

            evaluations = self._evaluate_batch(
                [leaf.board for leaf in leaves],
                [leaf.to_play for leaf in leaves],
            )

            for leaf, (policy, value) in zip(leaves, evaluations):
                if apply_vl:
                    node = leaf
                    while node is not root:
                        node.virtual_loss -= 1
                        node = node.parent
                if leaf.is_leaf():
                    leaf.expand(policy)
                self._backup(leaf, value)

            done += batch

    def _get_batch_size(self, simulations_done: int) -> int:
        """本輪批次大小；``dynamic_batch`` 時前期大、後期小。"""
        if self.batch_size <= 1 or not self.dynamic_batch:
            return self.batch_size

        progress = simulations_done / max(self.num_simulations, 1)
        if progress < 0.3:
            return self.max_batch_size
        if progress < 0.7:
            ratio = (progress - 0.3) / 0.4
            size = int(self.max_batch_size - ratio * (self.max_batch_size - self.min_batch_size))
            return max(self.min_batch_size, min(self.max_batch_size, size))
        return self.min_batch_size

    def _evaluate(
        self,
        board: GoBoard,
        color: str
    ) -> Tuple[Dict[Union[Tuple[int, int], str], float], float]:
        """Evaluate one position; value is from ``color``'s perspective."""
        return self._evaluate_batch([board], [color])[0]

    def _evaluate_batch(
        self,
        boards: List[GoBoard],
        colors: List[str]
    ) -> List[Tuple[Dict[Union[Tuple[int, int], str], float], float]]:
        """批次評估（有開快取時只評估未命中的盤面）。

        Returns
        -------
        List[Tuple[Dict, float]]
            (policy dict, 輪走方視角 value)
        """
        if self.eval_cache_size <= 0:
            evaluations = self.evaluator.evaluate(boards, colors)
            return [(e.policy_dict(), e.value) for e in evaluations]

        keys = [position_key(board, color) for board, color in zip(boards, colors)]
        missing = [i for i, key in enumerate(keys) if key not in self.evaluation_cache]
        if missing:
            evaluations = self.evaluator.evaluate(
                [boards[i] for i in missing], [colors[i] for i in missing]
            )
            if len(self.evaluation_cache) + len(missing) > self.eval_cache_size:
                self.evaluation_cache.clear()
            for i, evaluation in zip(missing, evaluations):
                self.evaluation_cache[keys[i]] = (evaluation.policy_dict(), evaluation.value)
        return [self.evaluation_cache[key] for key in keys]

    def _add_dirichlet_noise(
        self,
        policy: Dict[Tuple[int, int], float],
        board_size: int
    ) -> Dict[Tuple[int, int], float]:
        """Add Dirichlet noise to policy for exploration.

        Parameters
        ----------
        policy : Dict[Tuple[int, int], float]
            Original policy
        board_size : int
            Size of the board

        Returns
        -------
        Dict[Tuple[int, int], float]
            Policy with Dirichlet noise
        """
        moves = list(policy.keys())
        if not moves or self.dirichlet_epsilon == 0:
            return policy

        noise = np.random.dirichlet([self.dirichlet_alpha] * len(moves))

        return {
            move: (1 - self.dirichlet_epsilon) * policy[move] + self.dirichlet_epsilon * noise[i]
            for i, move in enumerate(moves)
        }

    def _backup(self, node: MCTSNode, value: float) -> None:
        """Propagate value up the tree.

        Parameters
        ----------
        node : MCTSNode
            Leaf node to start backup from
        value : float
            Value estimate from ``node.to_play``'s perspective
        """
        # 每個節點存「走進該節點那一方」的視角：葉節點存 -value，往上逐層變號
        value = -value
        current = node

        while current is not None:
            current.update(value)
            value = -value
            current = current.parent

    def _get_move_probabilities(
        self,
        root: MCTSNode,
        board_size: int
    ) -> np.ndarray:
        """Compute move probabilities from visit counts.

        沒有合法著手時回傳 pass；尚無訪問（``num_simulations=0``）時用先驗。

        Returns
        -------
        np.ndarray
            Move probabilities (board_size^2 + 1), sums to 1
        """
        num_actions = board_size * board_size + 1
        move_probs = np.zeros(num_actions, dtype=np.float32)

        if not root.children:
            move_probs[-1] = 1.0
            return move_probs

        def move_index(move) -> int:
            return board_size * board_size if move == 'pass' else move[0] * board_size + move[1]

        visits = np.zeros(num_actions, dtype=np.float64)
        for move, child in root.children.items():
            visits[move_index(move)] = child.visit_count
        if visits.sum() == 0:
            for move, child in root.children.items():
                visits[move_index(move)] = child.prior

        if self.temperature == 0:
            move_probs[int(np.argmax(visits))] = 1.0
            return move_probs

        visits = (visits / visits.max()) ** (1.0 / self.temperature)
        return (visits / visits.sum()).astype(np.float32)



# =============================================================================
# Backwards Compatibility
# =============================================================================

def mcts_search(
    board: List[List[int]],
    color: int = 1,
    iterations: int = 1000,
    komi: float = 7.5,
) -> Optional[Tuple[int, int]]:
    """Legacy MCTS search function (backwards compatibility).

    Note: This is a stub for backwards compatibility. The neural network-based
    MCTS requires a model to be provided. For proper MCTS, use the MCTS class
    directly with a trained model.

    Parameters
    ----------
    board : List[List[int]]
        Current board state (0=empty, 1=black, -1=white).
    color : int
        Color to play (1=black, -1=white).
    iterations : int
        Number of search iterations (not used in stub).
    komi : float
        Komi value (not used in stub).

    Returns
    -------
    Optional[Tuple[int, int]]
        First empty position found, or None if board is full.
    """
    import warnings
    warnings.warn(
        "mcts_search() is deprecated. Use MCTS class with a neural network model.",
        DeprecationWarning,
        stacklevel=2
    )

    # Simple fallback: return first empty position
    size = len(board)
    for y in range(size):
        for x in range(size):
            if board[y][x] == 0:
                return (x, y)
    return None

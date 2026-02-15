"""Monte Carlo Tree Search (MCTS) implementation for Go.

This module implements AlphaGo Zero style MCTS that combines:
- Neural network policy and value predictions
- UCB (Upper Confidence Bound) for exploration
- Dirichlet noise for root exploration
"""

from __future__ import annotations

import math
import numpy as np
from typing import Optional, List, Tuple, Dict, Union
from dataclasses import dataclass
import torch

from core.game_rules import GoBoard
from input.lightgo_encoder import LightGoEncoder


@dataclass
class MCTSNode:
    """Node in the MCTS search tree.

    Attributes
    ----------
    board : GoBoard
        Current board state
    parent : Optional[MCTSNode]
        Parent node
    move : Optional[Tuple[int, int]]
        Move that led to this node
    color : str
        Color that just moved ('b' or 'w')
    prior : float
        Prior probability from neural network
    children : Dict[Tuple[int, int], MCTSNode]
        Child nodes
    visit_count : int
        Number of times this node was visited
    total_value : float
        Sum of value estimates from simulations
    """

    board: GoBoard
    parent: Optional[MCTSNode] = None
    move: Optional[Tuple[int, int]] = None
    color: str = 'b'
    prior: float = 0.0

    def __post_init__(self):
        """Initialize mutable attributes."""
        self.children: Dict[Tuple[int, int], MCTSNode] = {}
        self.visit_count: int = 0
        self.total_value: float = 0.0

    @property
    def q_value(self) -> float:
        """Average action value (Q).

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

        Returns
        -------
        float
            Q(s,a) + U(s,a)
        """
        return self.q_value + self.u_value

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
        legal_moves = self.board.get_legal_moves(
            'b' if self.color == 'w' else 'w'  # Opponent's turn
        )

        next_color = 'w' if self.color == 'b' else 'b'

        for move in legal_moves:
            if move in self.children:
                continue

            # Create child board
            child_board = self.board.copy()
            child_board.play_move(move[0], move[1], next_color)

            # Get prior probability (use uniform prior as fallback)
            board_size = self.board.size
            default_prior = 1.0 / (board_size * board_size + 1)
            prior = policy.get(move, default_prior)

            # Create child node
            child = MCTSNode(
                board=child_board,
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
            Value estimate from simulation (-1 to 1)
        """
        self.visit_count += 1
        self.total_value += value


class MCTS:
    """Monte Carlo Tree Search engine.

    Parameters
    ----------
    model : torch.nn.Module
        Neural network model for policy and value prediction
    num_simulations : int
        Number of simulations per search (default 800)
    temperature : float
        Temperature for action selection (default 1.0)
    dirichlet_alpha : float
        Dirichlet noise alpha for exploration (default 0.03)
    dirichlet_epsilon : float
        Fraction of Dirichlet noise to add (default 0.25)
    reuse_tree : bool
        Whether to reuse search tree between moves (default True)
    max_tree_nodes : int
        Maximum number of nodes to keep in cached tree (default 100000)
        When exceeded, less visited branches are pruned
    """

    def __init__(
        self,
        model,
        num_simulations: int = 800,
        temperature: float = 1.0,
        dirichlet_alpha: float = 0.03,
        dirichlet_epsilon: float = 0.25,
        board_size: int = 19,
        reuse_tree: bool = True,
        max_tree_nodes: int = 100000
    ):
        self.model = model
        self.num_simulations = num_simulations
        self.temperature = temperature
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_epsilon = dirichlet_epsilon
        self.board_size = board_size
        self.reuse_tree = reuse_tree
        self.max_tree_nodes = max_tree_nodes

        # 獲取模型設備
        self.device = next(model.parameters()).device

        # LightGo 2-plane encoder
        self.encoder = LightGoEncoder(board_size=board_size)

        # Tree reuse: 緩存上一次搜索的根節點
        self._cached_root: Optional[MCTSNode] = None

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
        # 嘗試使用緩存的樹
        root = None
        if self.reuse_tree and self._cached_root is not None:
            root = self._cached_root
            # 驗證緩存的樹是否匹配當前棋盤（簡單檢查顏色）
            if root.color != color:
                root = None  # 顏色不匹配，需要新建

        # 如果沒有緩存或緩存無效，創建新的根節點
        if root is None:
            root = MCTSNode(board=board.copy(), color=color)

            # Get initial policy from neural network
            policy, _ = self._evaluate(root.board, color)

            # Add Dirichlet noise for exploration at root
            policy = self._add_dirichlet_noise(policy, board.size)

            # Expand root
            root.expand(policy)
        else:
            # 使用緩存的樹，但仍要添加 Dirichlet 噪聲到根節點
            # 重新獲取策略並混合噪聲
            if root.children:
                policy, _ = self._evaluate(root.board, color)
                policy = self._add_dirichlet_noise(policy, board.size)
                # 更新子節點的先驗概率
                for move, child in root.children.items():
                    if move in policy:
                        # 混合舊先驗和新噪聲
                        child.prior = 0.5 * child.prior + 0.5 * policy[move]

        # Run simulations
        for _ in range(self.num_simulations):
            node = root

            # Selection: traverse tree to leaf
            while not node.is_leaf():
                node = node.select_child()

            # Expansion & Evaluation
            if node.visit_count == 0:
                # First visit: get value from network
                _, value = self._evaluate(node.board, node.color)
            else:
                # Expand and evaluate
                policy, value = self._evaluate(node.board, node.color)
                node.expand(policy)

            # Backup: propagate value up the tree
            self._backup(node, value)

        # Compute move probabilities from visit counts
        move_probs = self._get_move_probabilities(root, board.size)

        # 緩存根節點供下次使用
        self._cached_root = root

        # 如果樹太大，執行剪枝
        self._maybe_prune_tree()

        return move_probs, root

    def advance_tree(self, move: Optional[Tuple[int, int]]) -> bool:
        """根據落子推進搜索樹

        將緩存的根移動到對應的子節點，實現樹重用。

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

        # 將 None 轉換為 'pass'
        move_key = move if move is not None else 'pass'

        if move_key in self._cached_root.children:
            # 找到對應的子節點
            child = self._cached_root.children[move_key]

            # 斷開父節點連結（幫助 GC 回收舊樹）
            child.parent = None

            # 設置為新的根
            self._cached_root = child
            return True
        else:
            # 動作不在樹中，清除緩存
            self._cached_root = None
            return False

    def clear_tree(self) -> None:
        """清除緩存的搜索樹"""
        self._cached_root = None

    def get_tree_stats(self) -> Dict[str, int]:
        """獲取緩存樹的統計信息

        Returns
        -------
        Dict[str, int]
            包含 'visit_count', 'num_children', 'tree_depth' 等信息
        """
        if self._cached_root is None:
            return {'cached': False, 'visit_count': 0, 'num_children': 0}

        def count_nodes(node: MCTSNode, depth: int = 0) -> Tuple[int, int]:
            """遞歸計算節點數和最大深度"""
            count = 1
            max_depth = depth
            for child in node.children.values():
                c, d = count_nodes(child, depth + 1)
                count += c
                max_depth = max(max_depth, d)
            return count, max_depth

        total_nodes, max_depth = count_nodes(self._cached_root)

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

            # 按訪問次數排序
            sorted_children = sorted(
                node.children.items(),
                key=lambda x: x[1].visit_count,
                reverse=True
            )

            # 計算要保留的子節點數量（至少保留 1 個）
            keep_count = max(1, int(len(sorted_children) * keep_ratio))

            # 深度越深，保留越少
            if depth > 3:
                keep_count = max(1, keep_count // 2)

            # 剪枝
            new_children = {}
            total_nodes = 1
            for i, (move, child) in enumerate(sorted_children):
                if i < keep_count:
                    new_children[move] = child
                    total_nodes += prune_node(child, depth + 1)

            node.children = new_children
            return total_nodes

        return prune_node(self._cached_root)

    def _maybe_prune_tree(self) -> None:
        """如果樹太大，執行剪枝"""
        if self._cached_root is None:
            return

        stats = self.get_tree_stats()
        if stats.get('total_nodes', 0) > self.max_tree_nodes:
            self._prune_tree(keep_ratio=0.5)

    def _evaluate(
        self,
        board: GoBoard,
        color: str
    ) -> Tuple[Dict[Union[Tuple[int, int], str], float], float]:
        """Evaluate position using neural network.

        Parameters
        ----------
        board : GoBoard
            Board state to evaluate
        color : str
            Current player color

        Returns
        -------
        Tuple[Dict[Union[Tuple[int, int], str], float], float]
            - Policy dictionary: {(row, col): probability, 'pass': probability}
            - Value estimate (-1 to 1)
        """
        # Convert board to neural network input
        features = self._board_to_features(board, color)

        # Get network predictions
        with torch.no_grad():
            policy_logits, value = self.model(features)

            # Apply softmax to policy
            policy_probs = torch.softmax(policy_logits, dim=1)[0]

        # Convert policy to dictionary (include all 362 outputs)
        board_size = board.size
        policy_dict = {}

        # Board positions (0 to board_size^2 - 1)
        for i in range(board_size * board_size):
            row = i // board_size
            col = i % board_size
            policy_dict[(row, col)] = policy_probs[i].item()

        # Pass move (index board_size^2)
        policy_dict['pass'] = policy_probs[board_size * board_size].item()

        # Extract value
        value_estimate = value[0].item()

        return policy_dict, value_estimate

    def _board_to_features(self, board: GoBoard, color: str) -> torch.Tensor:
        """Convert board to neural network input features using LightGo 2-plane encoding.

        Parameters
        ----------
        board : GoBoard
            Board state
        color : str
            Current player color

        Returns
        -------
        torch.Tensor
            Feature tensor (1, 2, board_size, board_size)

        Notes
        -----
        LightGo 2-plane encoding:
        - Plane 0: Signed liberties (black=+, white=-)
        - Plane 1: Forbidden points (occupied, suicide, ko)
        """
        # Create encoder with correct board size if needed
        if board.size != self.encoder.board_size:
            self.encoder = LightGoEncoder(board_size=board.size)

        # Use LightGo 2-plane encoder
        features = self.encoder.encode(board, color)

        # Add batch dimension and move to device
        features_tensor = features.unsqueeze(0).to(self.device)

        return features_tensor

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
        if not moves:
            return policy

        # Generate Dirichlet noise
        noise = np.random.dirichlet([self.dirichlet_alpha] * len(moves))

        # Mix policy with noise
        noisy_policy = {}
        for i, move in enumerate(moves):
            noisy_policy[move] = (
                (1 - self.dirichlet_epsilon) * policy[move] +
                self.dirichlet_epsilon * noise[i]
            )

        return noisy_policy

    def _backup(self, node: MCTSNode, value: float) -> None:
        """Propagate value up the tree.

        Parameters
        ----------
        node : MCTSNode
            Leaf node to start backup from
        value : float
            Value estimate
        """
        current = node

        while current is not None:
            # Alternate value sign (opponent's perspective)
            current.update(value)
            value = -value
            current = current.parent

    def _get_move_probabilities(
        self,
        root: MCTSNode,
        board_size: int
    ) -> np.ndarray:
        """Compute move probabilities from visit counts.

        Parameters
        ----------
        root : MCTSNode
            Root node of search tree
        board_size : int
            Size of the board

        Returns
        -------
        np.ndarray
            Move probabilities (board_size^2 + 1)
        """
        move_probs = np.zeros(board_size * board_size + 1, dtype=np.float32)

        if self.temperature == 0:
            # Greedy: select most visited move
            best_move = max(root.children.items(), key=lambda x: x[1].visit_count)
            move = best_move[0]
            if move == 'pass':
                move_idx = board_size * board_size
            else:
                move_idx = move[0] * board_size + move[1]
            move_probs[move_idx] = 1.0
        else:
            # Temperature-based selection
            visits = np.zeros(board_size * board_size + 1, dtype=np.float32)

            for move, child in root.children.items():
                if move == 'pass':
                    move_idx = board_size * board_size
                else:
                    move_idx = move[0] * board_size + move[1]
                visits[move_idx] = child.visit_count

            # Apply temperature
            visits = visits ** (1.0 / self.temperature)

            # Normalize
            if visits.sum() > 0:
                move_probs = visits / visits.sum()

        return move_probs

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
            Override default temperature

        Returns
        -------
        Optional[Tuple[int, int]]
            Selected move (row, col), or None for pass
        """
        if temperature is not None:
            old_temp = self.temperature
            self.temperature = temperature

        move_probs, root = self.search(board, color)

        # Sample move from distribution
        move_idx = np.random.choice(len(move_probs), p=move_probs)

        if temperature is not None:
            self.temperature = old_temp

        # Convert index to (row, col) or None for pass
        board_size = board.size
        if move_idx == board_size * board_size:
            return None  # Pass move

        row = move_idx // board_size
        col = move_idx % board_size

        return (row, col)


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

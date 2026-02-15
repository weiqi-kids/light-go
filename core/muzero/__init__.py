"""MuZero 實作

MuZero 是一種結合 MCTS 和學習環境模型的強化學習方法。
與 AlphaZero 不同，MuZero 不需要完美的遊戲規則，而是學習三個函數：

1. h (representation): 觀察 → 隱藏狀態
2. g (dynamics): (隱藏狀態, 動作) → (下一隱藏狀態, 即時獎勵)
3. f (prediction): 隱藏狀態 → (策略, 價值)

參考論文：
- "Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model" (Schrittwieser et al., 2020)
"""

from core.muzero.networks import MuZeroNetworks, ResBlock
from core.muzero.mcts import MuZeroMCTS, MuZeroNode
from core.muzero.trainer import MuZeroTrainer
from core.muzero.replay_buffer import MuZeroReplayBuffer, Trajectory

__all__ = [
    'MuZeroNetworks',
    'ResBlock',
    'MuZeroMCTS',
    'MuZeroNode',
    'MuZeroTrainer',
    'MuZeroReplayBuffer',
    'Trajectory',
]

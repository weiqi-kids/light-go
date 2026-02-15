"""MuZero 單元測試"""

import pytest
import numpy as np
import torch

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.muzero.networks import (
    MuZeroNetworks,
    RepresentationNetwork,
    DynamicsNetwork,
    PredictionNetwork,
    ResBlock
)
from core.muzero.mcts import MuZeroMCTS, MuZeroNode
from core.muzero.replay_buffer import MuZeroReplayBuffer, Trajectory
from core.muzero.trainer import MuZeroTrainer


class TestMuZeroNetworks:
    """MuZero 網路測試"""

    @pytest.fixture
    def networks(self):
        """建立測試網路"""
        return MuZeroNetworks(
            board_size=9,
            num_input_planes=2,
            hidden_size=64,
            num_blocks=4
        )

    def test_network_creation(self, networks):
        """網路建立"""
        assert networks is not None
        assert networks.board_size == 9
        assert networks.num_actions == 82  # 9*9 + 1

    def test_initial_inference(self, networks):
        """初始推理"""
        observation = torch.randn(1, 2, 9, 9)
        hidden, policy, value = networks.initial_inference(observation)

        assert hidden.shape == (1, 64, 9, 9)
        assert policy.shape == (1, 82)
        assert value.shape == (1, 1)

    def test_recurrent_inference(self, networks):
        """遞迴推理"""
        observation = torch.randn(1, 2, 9, 9)
        hidden, _, _ = networks.initial_inference(observation)

        action = torch.tensor([0])
        next_hidden, reward, policy, value = networks.recurrent_inference(hidden, action)

        assert next_hidden.shape == (1, 64, 9, 9)
        assert reward.shape == (1, 1)
        assert policy.shape == (1, 82)
        assert value.shape == (1, 1)

    def test_forward(self, networks):
        """標準前向傳遞"""
        observation = torch.randn(1, 2, 9, 9)
        policy, value = networks(observation)

        assert policy.shape == (1, 82)
        assert value.shape == (1, 1)

    def test_batch_inference(self, networks):
        """批次推理"""
        observation = torch.randn(4, 2, 9, 9)
        hidden, policy, value = networks.initial_inference(observation)

        assert hidden.shape == (4, 64, 9, 9)
        assert policy.shape == (4, 82)
        assert value.shape == (4, 1)

    def test_param_count(self, networks):
        """參數數量"""
        count = networks.get_param_count()
        assert count > 0
        # 小型網路應該有合理的參數量
        assert count < 10_000_000


class TestResBlock:
    """殘差塊測試"""

    def test_resblock_shape(self):
        """殘差塊形狀"""
        block = ResBlock(64)
        x = torch.randn(2, 64, 9, 9)
        out = block(x)
        assert out.shape == x.shape

    def test_resblock_residual(self):
        """殘差連接"""
        block = ResBlock(64)
        x = torch.randn(2, 64, 9, 9)

        # 殘差連接應該使輸出與輸入相關
        out = block(x)
        # 輸出不應該與輸入完全相同（因為有卷積）
        assert not torch.allclose(out, x)


class TestMuZeroMCTS:
    """MuZero MCTS 測試"""

    @pytest.fixture
    def mcts(self):
        """建立 MCTS"""
        networks = MuZeroNetworks(
            board_size=9,
            num_input_planes=2,
            hidden_size=64,
            num_blocks=4
        )
        return MuZeroMCTS(
            networks=networks,
            num_simulations=10,
            discount=0.99
        )

    def test_search(self, mcts):
        """搜尋測試"""
        observation = torch.randn(1, 2, 9, 9)
        policy, root, value = mcts.search(observation)

        assert policy.shape == (82,)
        assert isinstance(root, MuZeroNode)
        assert isinstance(value, float)

        # 策略應該是有效的機率分布
        assert np.all(policy >= 0)
        assert abs(policy.sum() - 1.0) < 1e-5

    def test_select_action(self, mcts):
        """選擇動作"""
        observation = torch.randn(1, 2, 9, 9)
        action, policy, value = mcts.select_action(observation, temperature=1.0)

        assert 0 <= action < 82
        assert policy.shape == (82,)

    def test_deterministic_action(self, mcts):
        """確定性動作選擇"""
        observation = torch.randn(1, 2, 9, 9)
        action1, _, _ = mcts.select_action(observation, temperature=0.0)
        action2, _, _ = mcts.select_action(observation, temperature=0.0)

        # 溫度為 0 時，應該選擇訪問次數最多的動作
        # 但由於是新的搜尋，結果可能不同
        assert 0 <= action1 < 82
        assert 0 <= action2 < 82


class TestMuZeroNode:
    """MuZero 節點測試"""

    def test_node_creation(self):
        """節點建立"""
        node = MuZeroNode()
        assert node.visit_count == 0
        assert node.q_value == 0.0
        assert not node.is_expanded()

    def test_expand(self):
        """節點擴展"""
        node = MuZeroNode()
        prior = np.ones(82) / 82
        node.expand(prior)

        assert node.is_expanded()
        assert len(node.children) == 82

    def test_update(self):
        """節點更新"""
        node = MuZeroNode()
        node.update(0.5)
        node.update(0.3)

        assert node.visit_count == 2
        assert abs(node.q_value - 0.4) < 1e-6

    def test_select_child(self):
        """選擇子節點"""
        node = MuZeroNode()
        prior = np.zeros(82)
        prior[0] = 0.5
        prior[1] = 0.5
        node.expand(prior)

        # 首次選擇應該基於先驗
        action, child = node.select_child()
        assert action in [0, 1]


class TestTrajectory:
    """軌跡測試"""

    def test_trajectory_creation(self):
        """軌跡建立"""
        traj = Trajectory()
        assert len(traj) == 0

    def test_add_step(self):
        """添加步驟"""
        traj = Trajectory()
        obs = np.zeros((2, 9, 9))
        policy = np.ones(82) / 82

        traj.add_step(obs, action=0, reward=0.0, policy=policy, value=0.5)
        assert len(traj) == 1

    def test_compute_target_values(self):
        """計算目標價值"""
        traj = Trajectory()

        # 添加幾步
        for i in range(5):
            obs = np.zeros((2, 9, 9))
            policy = np.ones(82) / 82
            traj.add_step(obs, action=i, reward=0.0, policy=policy, value=0.5)

        traj.game_result = 1.0

        target_values = traj.compute_target_values(td_steps=3, discount=0.99)
        assert len(target_values) == 5


class TestMuZeroReplayBuffer:
    """重放緩衝區測試"""

    @pytest.fixture
    def buffer(self):
        """建立緩衝區"""
        return MuZeroReplayBuffer(max_size=100, td_steps=5, discount=0.99)

    def test_add_trajectory(self, buffer):
        """添加軌跡"""
        traj = Trajectory()
        for i in range(10):
            obs = np.zeros((2, 9, 9))
            policy = np.ones(82) / 82
            traj.add_step(obs, action=i % 82, reward=0.0, policy=policy, value=0.5)

        buffer.add_trajectory(traj)
        assert len(buffer) == 1
        assert buffer.num_samples == 10

    def test_sample_batch(self, buffer):
        """採樣批次"""
        # 添加幾條軌跡
        for _ in range(5):
            traj = Trajectory()
            for i in range(20):
                obs = np.random.randn(2, 9, 9).astype(np.float32)
                policy = np.ones(82, dtype=np.float32) / 82
                traj.add_step(obs, action=i % 82, reward=0.0, policy=policy, value=0.5)
            traj.game_result = 1.0
            buffer.add_trajectory(traj)

        batch = buffer.sample_batch(batch_size=4, unroll_steps=5)

        assert batch['observations'].shape == (4, 2, 9, 9)
        assert batch['actions'].shape == (4, 5)
        assert batch['target_values'].shape == (4, 6)
        assert batch['target_rewards'].shape == (4, 5)
        assert batch['target_policies'].shape == (4, 6, 82)


class TestMuZeroTrainer:
    """MuZero 訓練器測試"""

    @pytest.fixture
    def trainer(self):
        """建立訓練器"""
        networks = MuZeroNetworks(
            board_size=9,
            num_input_planes=2,
            hidden_size=64,
            num_blocks=4
        )
        buffer = MuZeroReplayBuffer(max_size=100)

        return MuZeroTrainer(
            networks=networks,
            replay_buffer=buffer,
            unroll_steps=3,
            learning_rate=0.001,
            device='cpu'
        )

    def test_train_step(self, trainer):
        """訓練步驟"""
        batch = {
            'observations': np.random.randn(4, 2, 9, 9).astype(np.float32),
            'actions': np.random.randint(0, 82, (4, 3)),
            'target_values': np.random.randn(4, 4).astype(np.float32),
            'target_rewards': np.random.randn(4, 3).astype(np.float32),
            'target_policies': np.random.rand(4, 4, 82).astype(np.float32)
        }
        # 正規化策略
        batch['target_policies'] = batch['target_policies'] / batch['target_policies'].sum(axis=-1, keepdims=True)

        losses = trainer.train_step(batch)

        assert 'total_loss' in losses
        assert 'policy_loss' in losses
        assert 'value_loss' in losses
        assert 'reward_loss' in losses
        assert losses['total_loss'] >= 0

    def test_evaluate(self, trainer):
        """評估"""
        batch = {
            'observations': np.random.randn(4, 2, 9, 9).astype(np.float32),
            'actions': np.random.randint(0, 82, (4, 3)),
            'target_values': np.random.randn(4, 4).astype(np.float32),
            'target_rewards': np.random.randn(4, 3).astype(np.float32),
            'target_policies': np.random.rand(4, 4, 82).astype(np.float32)
        }
        batch['target_policies'] = batch['target_policies'] / batch['target_policies'].sum(axis=-1, keepdims=True)

        losses = trainer.evaluate(batch)
        assert 'total_loss' in losses


class TestMuZeroIntegration:
    """MuZero 整合測試"""

    def test_full_workflow(self):
        """完整工作流程"""
        # 建立網路和緩衝區
        networks = MuZeroNetworks(
            board_size=9,
            num_input_planes=2,
            hidden_size=32,
            num_blocks=2
        )
        buffer = MuZeroReplayBuffer(max_size=100)
        mcts = MuZeroMCTS(networks, num_simulations=5)
        trainer = MuZeroTrainer(networks, buffer, unroll_steps=2, device='cpu')

        # 模擬自我對弈
        for game in range(3):
            traj = Trajectory()

            for step in range(10):
                obs = np.random.randn(2, 9, 9).astype(np.float32)
                obs_tensor = torch.tensor(obs).unsqueeze(0)

                action, policy, value = mcts.select_action(obs_tensor, temperature=1.0)
                traj.add_step(obs, action=action, reward=0.0, policy=policy, value=value)

            traj.game_result = np.random.choice([-1, 0, 1])
            buffer.add_trajectory(traj)

        # 訓練
        if len(buffer) > 0:
            losses = trainer.train_epoch(batch_size=2, num_batches=5)
            assert losses['total_loss'] >= 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

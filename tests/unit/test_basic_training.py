"""Basic tests for neural network training functionality."""

import pytest
import torch
import numpy as np
from hf_models.modeling_go_ai import GoAIModel
from hf_models.board_encoder import BoardEncoder
from core.architecture_genome import ArchitectureGenome
from core.trainer import ModelTrainer
from core.game_rules import GoBoard


def test_goai_model_creation():
    """Test creating a GoAIModel instance."""
    model = GoAIModel.create_baseline(board_size=19, device='cpu')

    assert model is not None
    assert model.board_size == 19
    assert model.num_filters == 128
    assert model.num_blocks == 5


def test_model_forward_pass():
    """Test forward pass through the model."""
    model = GoAIModel.create_baseline(board_size=19, device='cpu')

    # Create dummy input (2 planes for LightGo encoding)
    batch_size = 2
    input_tensor = torch.randn(batch_size, 2, 19, 19)

    # Forward pass
    policy, value = model(input_tensor)

    # Check output shapes
    assert policy.shape == (batch_size, 19 * 19 + 1)  # 361 + pass
    assert value.shape == (batch_size, 1)


def test_board_encoder():
    """Test liberty matrix to feature encoding."""
    encoder = BoardEncoder(board_size=19)

    # Create simple liberty matrix
    liberty = [(3, 3, 3), (4, 4, -2)]  # Black at (3,3), White at (4,4)
    forbidden = []
    metadata = {"next_move": "black"}

    features = encoder.encode(liberty, forbidden, metadata)

    # Check output shape
    assert features.shape == (7, 19, 19)

    # Check black stone presence
    assert features[0, 2, 2] == 1.0  # Black stone (0-indexed)

    # Check white stone presence
    assert features[1, 3, 3] == 1.0  # White stone (0-indexed)


def test_architecture_genome_creation():
    """Test creating and mutating architecture genomes."""
    # Create baseline genome
    genome = ArchitectureGenome.create_baseline()

    assert genome.num_blocks == 5
    assert genome.base_filters == 128

    # Test mutation
    mutated = genome.mutate(mutation_rate=0.5)

    assert mutated.generation == genome.generation + 1
    assert len(mutated.parent_ids) == 1

    # Test that genome can build a model
    model = genome.to_pytorch_model(device='cpu')
    assert isinstance(model, GoAIModel)


def test_genome_crossover():
    """Test genome crossover operation."""
    parent1 = ArchitectureGenome.create_baseline()
    parent2 = ArchitectureGenome.create_baseline()
    parent2.base_filters = 256  # Make it different

    child = ArchitectureGenome.crossover(parent1, parent2)

    assert child.generation == max(parent1.generation, parent2.generation) + 1
    assert len(child.parent_ids) == 2
    assert child.base_filters in [parent1.base_filters, parent2.base_filters]


def test_go_board_basic():
    """Test basic Go board operations."""
    board = GoBoard(size=19)

    # Test legal move
    assert board.is_legal(3, 3, 'b')

    # Play a move
    success = board.play_move(3, 3, 'b')
    assert success

    # Check stone is placed
    assert board.get(3, 3) == 'b'

    # Cannot play on occupied point
    assert not board.is_legal(3, 3, 'w')


def test_go_board_capture():
    """Test stone capture."""
    board = GoBoard(size=19)

    # Set up a capture scenario (white surrounded by black)
    board.play_move(1, 0, 'w')
    board.play_move(0, 0, 'b')
    board.play_move(1, 1, 'b')
    board.play_move(2, 0, 'b')

    # Capturing move
    board.play_move(0, 1, 'b')  # Should capture white at (1,0)

    # Check white stone was captured
    assert board.get(1, 0) is None
    assert board.prisoners['white'] == 1


def test_model_training_step():
    """Test a single training step."""
    model = GoAIModel.create_baseline(board_size=19, device='cpu')
    trainer = ModelTrainer(model, device='cpu')

    # Create dummy training data (2 planes for LightGo encoding)
    positions = torch.randn(4, 2, 19, 19)
    policies = torch.randn(4, 19 * 19 + 1)
    policies = torch.softmax(policies, dim=1)  # Normalize to probabilities
    values = torch.randn(4)

    # Training step
    losses = trainer.train_step(positions, policies, values)

    # Check losses are returned
    assert 'policy_loss' in losses
    assert 'value_loss' in losses
    assert 'total_loss' in losses
    assert all(v > 0 for v in losses.values())


def test_model_save_load():
    """Test saving and loading model."""
    import tempfile
    import os

    model = GoAIModel.create_baseline(board_size=19, device='cpu')

    # Save model
    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = os.path.join(tmpdir, 'test_model.pt')
        model.save_pretrained(save_path)

        # Check file exists
        assert os.path.exists(save_path)

        # Load model
        loaded_model = GoAIModel.from_pretrained(save_path, device='cpu')

        # Test that loaded model works (2 planes for LightGo encoding)
        test_input = torch.randn(1, 2, 19, 19)
        policy1, value1 = model(test_input)
        policy2, value2 = loaded_model(test_input)

        # Should produce same output
        assert torch.allclose(policy1, policy2, atol=1e-6)
        assert torch.allclose(value1, value2, atol=1e-6)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

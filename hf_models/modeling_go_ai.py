"""PyTorch-based Go AI model with policy and value heads.

This module implements the core neural network for Go game playing, featuring:
- Convolutional neural network body
- Policy head for move prediction
- Value head for position evaluation
- Support for architecture evolution (genome-based construction)
"""

from __future__ import annotations

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, Any, List
import logging

logger = logging.getLogger(__name__)


class ResidualBlock(nn.Module):
    """Basic residual block with two convolutional layers."""

    def __init__(self, num_filters: int, kernel_size: int = 3):
        super().__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv2d(num_filters, num_filters, kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm2d(num_filters)
        self.conv2 = nn.Conv2d(num_filters, num_filters, kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm2d(num_filters)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = F.relu(out)
        return out


class PolicyHead(nn.Module):
    """Policy head for predicting move probabilities.

    Outputs probability distribution over all possible moves (board_size^2 + 1 for pass).
    """

    def __init__(self, num_filters: int, board_size: int = 19):
        super().__init__()
        self.board_size = board_size
        self.conv = nn.Conv2d(num_filters, 2, kernel_size=1)
        self.bn = nn.BatchNorm2d(2)
        self.fc = nn.Linear(2 * board_size * board_size, board_size * board_size + 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass producing move probabilities.

        Parameters
        ----------
        x : torch.Tensor
            Feature tensor from network body, shape (batch, filters, height, width)

        Returns
        -------
        torch.Tensor
            Logits for all moves, shape (batch, board_size^2 + 1)
        """
        batch_size = x.shape[0]
        out = F.relu(self.bn(self.conv(x)))
        out = out.view(batch_size, -1)
        out = self.fc(out)
        return out


class ValueHead(nn.Module):
    """Value head for predicting game outcome.

    Outputs a single value in [-1, 1] representing win probability.
    """

    def __init__(self, num_filters: int, board_size: int = 19):
        super().__init__()
        self.board_size = board_size
        self.conv = nn.Conv2d(num_filters, 1, kernel_size=1)
        self.bn = nn.BatchNorm2d(1)
        self.fc1 = nn.Linear(board_size * board_size, 256)
        self.fc2 = nn.Linear(256, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass producing value prediction.

        Parameters
        ----------
        x : torch.Tensor
            Feature tensor from network body

        Returns
        -------
        torch.Tensor
            Value prediction in [-1, 1], shape (batch, 1)
        """
        batch_size = x.shape[0]
        out = F.relu(self.bn(self.conv(x)))
        out = out.view(batch_size, -1)
        out = F.relu(self.fc1(out))
        out = torch.tanh(self.fc2(out))
        return out


class GoAIModel(nn.Module):
    """Main Go AI model with policy and value heads.

    This model follows the AlphaGo Zero architecture:
    - Input: Multi-plane board representation
    - Body: Residual convolutional network
    - Heads: Policy (move probabilities) + Value (win probability)

    Parameters
    ----------
    num_input_planes : int
        Number of input feature planes (default 2 for LightGo format)
    num_filters : int
        Number of filters in convolutional layers (default 128)
    num_blocks : int
        Number of residual blocks (default 5)
    board_size : int
        Size of Go board (default 19)
    device : str
        Device to run model on ('cuda' or 'cpu')
    """

    def __init__(
        self,
        num_input_planes: int = 2,  # LightGo 2-plane format (signed liberties + forbidden)
        num_filters: int = 128,
        num_blocks: int = 5,
        board_size: int = 19,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        super().__init__()

        self.num_input_planes = num_input_planes
        self.num_filters = num_filters
        self.num_blocks = num_blocks
        self.board_size = board_size
        self.device = device

        # Initial convolution
        self.conv_input = nn.Conv2d(num_input_planes, num_filters, kernel_size=3, padding=1)
        self.bn_input = nn.BatchNorm2d(num_filters)

        # Residual blocks (body)
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(num_filters) for _ in range(num_blocks)
        ])

        # Policy and value heads
        self.policy_head = PolicyHead(num_filters, board_size)
        self.value_head = ValueHead(num_filters, board_size)

        # Move to device
        self.to(device)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the network.

        Parameters
        ----------
        x : torch.Tensor
            Input features, shape (batch, num_planes, board_size, board_size)

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            - policy_logits: Move probabilities, shape (batch, board_size^2 + 1)
            - value: Win probability, shape (batch, 1)
        """
        # Initial convolution
        out = F.relu(self.bn_input(self.conv_input(x)))

        # Residual blocks
        for block in self.residual_blocks:
            out = block(out)

        # Compute policy and value
        policy = self.policy_head(out)
        value = self.value_head(out)

        return policy, value

    def predict(self, board_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict move probabilities and value for a board position.

        Parameters
        ----------
        board_features : torch.Tensor
            Encoded board features

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            - policy: Probability distribution over moves (softmax applied)
            - value: Estimated win probability
        """
        self.eval()
        with torch.no_grad():
            if not isinstance(board_features, torch.Tensor):
                board_features = torch.from_numpy(board_features).float()

            if board_features.dim() == 3:
                board_features = board_features.unsqueeze(0)  # Add batch dimension

            board_features = board_features.to(self.device)

            policy_logits, value = self.forward(board_features)
            policy = F.softmax(policy_logits, dim=1)

            return policy, value

    def train_step(
        self,
        positions: torch.Tensor,
        target_policies: torch.Tensor,
        target_values: torch.Tensor,
        optimizer: torch.optim.Optimizer
    ) -> Dict[str, float]:
        """Perform one training step.

        Parameters
        ----------
        positions : torch.Tensor
            Batch of board positions
        target_policies : torch.Tensor
            Target move probabilities from MCTS
        target_values : torch.Tensor
            Target values (game outcomes)
        optimizer : torch.optim.Optimizer
            Optimizer for updating weights

        Returns
        -------
        Dict[str, float]
            Dictionary with 'policy_loss', 'value_loss', 'total_loss'
        """
        self.train()

        positions = positions.to(self.device)
        target_policies = target_policies.to(self.device)
        target_values = target_values.to(self.device)

        # Forward pass
        policy_logits, value = self.forward(positions)

        # Compute losses
        policy_loss = F.cross_entropy(policy_logits, target_policies)
        value_loss = F.mse_loss(value.squeeze(), target_values)
        total_loss = policy_loss + value_loss

        # Backward pass
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        return {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'total_loss': total_loss.item()
        }

    def train_on_batch(
        self,
        batch_data: List[Dict[str, Any]],
        optimizer: torch.optim.Optimizer
    ) -> Dict[str, float]:
        """Train on a batch of game positions.

        Parameters
        ----------
        batch_data : List[Dict[str, Any]]
            List of training samples, each containing:
            - 'position': board features
            - 'policy': target policy
            - 'value': target value
        optimizer : torch.optim.Optimizer
            Optimizer instance

        Returns
        -------
        Dict[str, float]
            Training metrics
        """
        # Collate batch
        positions = torch.stack([torch.from_numpy(sample['position']).float()
                                for sample in batch_data])
        policies = torch.stack([torch.from_numpy(sample['policy']).float()
                               for sample in batch_data])
        values = torch.tensor([sample['value'] for sample in batch_data], dtype=torch.float32)

        return self.train_step(positions, policies, values, optimizer)

    def save_pretrained(self, save_path: str) -> None:
        """Save model weights and configuration.

        Parameters
        ----------
        save_path : str
            Path to save model (can be directory or .pt file)
        """
        if os.path.isdir(save_path):
            save_path = os.path.join(save_path, "model.pt")

        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)

        torch.save({
            'model_state_dict': self.state_dict(),
            'num_input_planes': self.num_input_planes,
            'num_filters': self.num_filters,
            'num_blocks': self.num_blocks,
            'board_size': self.board_size
        }, save_path)

        logger.info(f"Model saved to {save_path}")

    @classmethod
    def from_pretrained(cls, load_path: str, device: Optional[str] = None) -> "GoAIModel":
        """Load model from saved checkpoint.

        Parameters
        ----------
        load_path : str
            Path to saved model (.pt file or directory containing model.pt)
        device : Optional[str]
            Device to load model on

        Returns
        -------
        GoAIModel
            Loaded model instance
        """
        if os.path.isdir(load_path):
            load_path = os.path.join(load_path, "model.pt")

        if not os.path.exists(load_path):
            raise FileNotFoundError(f"Model file not found: {load_path}")

        checkpoint = torch.load(load_path, map_location='cpu')

        # Create model with saved configuration
        model = cls(
            num_input_planes=checkpoint.get('num_input_planes', 2),
            num_filters=checkpoint.get('num_filters', 128),
            num_blocks=checkpoint.get('num_blocks', 5),
            board_size=checkpoint.get('board_size', 19),
            device=device or ('cuda' if torch.cuda.is_available() else 'cpu')
        )

        # Load weights
        model.load_state_dict(checkpoint['model_state_dict'])

        logger.info(f"Model loaded from {load_path}")
        return model

    @staticmethod
    def create_baseline(board_size: int = 19, device: Optional[str] = None) -> "GoAIModel":
        """Create a baseline model with standard hyperparameters.

        Parameters
        ----------
        board_size : int
            Size of Go board
        device : Optional[str]
            Device to create model on

        Returns
        -------
        GoAIModel
            New model instance with baseline configuration
        """
        return GoAIModel(
            num_input_planes=2,  # LightGo 2-plane format
            num_filters=128,
            num_blocks=5,
            board_size=board_size,
            device=device or ('cuda' if torch.cuda.is_available() else 'cpu')
        )


# Backward compatibility: expose as module-level function
def create_model(board_size: int = 19) -> GoAIModel:
    """Create a new Go AI model.

    Parameters
    ----------
    board_size : int
        Size of the Go board

    Returns
    -------
    GoAIModel
        New model instance
    """
    return GoAIModel.create_baseline(board_size=board_size)

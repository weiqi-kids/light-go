"""Training utilities for Go AI models.

This module provides training loops and utilities for supervised and reinforcement learning.
"""

from __future__ import annotations

import os
import logging
from typing import List, Dict, Any, Optional, Tuple
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader

logger = logging.getLogger(__name__)


class GoDataset(Dataset):
    """PyTorch Dataset for Go positions.

    Parameters
    ----------
    samples : List[Dict[str, Any]]
        List of training samples with 'position', 'policy', 'value' keys
    """

    def __init__(self, samples: List[Dict[str, Any]]):
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        sample = self.samples[idx]

        position = torch.from_numpy(sample['position']).float()
        policy = torch.from_numpy(sample['policy']).float()
        value = torch.tensor(sample['value'], dtype=torch.float32)

        return position, policy, value


class ModelTrainer:
    """Trainer for Go AI models with support for supervised and RL training.

    Parameters
    ----------
    model : nn.Module
        Neural network model to train
    device : str
        Device to train on ('cuda' or 'cpu')
    learning_rate : float
        Learning rate for optimizer
    """

    def __init__(
        self,
        model: nn.Module,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        learning_rate: float = 0.001
    ):
        self.model = model
        self.device = device
        self.learning_rate = learning_rate

        # Move model to device
        self.model.to(device)

        # Initialize optimizer
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        # Training history
        self.history = {
            'policy_loss': [],
            'value_loss': [],
            'total_loss': []
        }

    def train_step(
        self,
        positions: torch.Tensor,
        target_policies: torch.Tensor,
        target_values: torch.Tensor
    ) -> Dict[str, float]:
        """Perform one training step.

        Parameters
        ----------
        positions : torch.Tensor
            Batch of board positions
        target_policies : torch.Tensor
            Target move distributions
        target_values : torch.Tensor
            Target game outcomes

        Returns
        -------
        Dict[str, float]
            Dictionary with loss values
        """
        self.model.train()

        # Move to device
        positions = positions.to(self.device)
        target_policies = target_policies.to(self.device)
        target_values = target_values.to(self.device)

        # Forward pass
        policy_logits, value = self.model(positions)

        # Compute losses
        # Policy loss: cross entropy with target distribution
        policy_loss = -(target_policies * torch.log_softmax(policy_logits, dim=1)).sum(dim=1).mean()

        # Value loss: MSE with target value
        value_loss = torch.nn.functional.mse_loss(value.squeeze(), target_values)

        # Total loss
        total_loss = policy_loss + value_loss

        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        # Record history
        losses = {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'total_loss': total_loss.item()
        }

        for key, value in losses.items():
            self.history[key].append(value)

        return losses

    def train_epoch(
        self,
        train_loader: DataLoader,
        verbose: bool = True
    ) -> Dict[str, float]:
        """Train for one epoch.

        Parameters
        ----------
        train_loader : DataLoader
            DataLoader providing training batches
        verbose : bool
            Whether to print progress

        Returns
        -------
        Dict[str, float]
            Average losses for the epoch
        """
        epoch_losses = {
            'policy_loss': 0.0,
            'value_loss': 0.0,
            'total_loss': 0.0
        }

        num_batches = 0

        for batch_idx, (positions, policies, values) in enumerate(train_loader):
            losses = self.train_step(positions, policies, values)

            for key in epoch_losses:
                epoch_losses[key] += losses[key]

            num_batches += 1

            if verbose and (batch_idx + 1) % 10 == 0:
                logger.info(
                    f"Batch {batch_idx + 1}/{len(train_loader)}: "
                    f"Loss={losses['total_loss']:.4f}"
                )

        # Average losses
        for key in epoch_losses:
            epoch_losses[key] /= num_batches

        return epoch_losses

    def train(
        self,
        dataset: List[Dict[str, Any]],
        epochs: int = 10,
        batch_size: int = 64,
        validation_split: float = 0.1,
        verbose: bool = True
    ) -> Dict[str, List[float]]:
        """Train model on dataset.

        Parameters
        ----------
        dataset : List[Dict[str, Any]]
            Training samples
        epochs : int
            Number of epochs to train
        batch_size : int
            Batch size
        validation_split : float
            Fraction of data to use for validation
        verbose : bool
            Whether to print progress

        Returns
        -------
        Dict[str, List[float]]
            Training history
        """
        # Split into train/val
        val_size = int(len(dataset) * validation_split)
        train_dataset = dataset[val_size:]
        val_dataset = dataset[:val_size]

        # Create data loaders
        train_loader = DataLoader(
            GoDataset(train_dataset),
            batch_size=batch_size,
            shuffle=True
        )

        val_loader = DataLoader(
            GoDataset(val_dataset),
            batch_size=batch_size,
            shuffle=False
        ) if val_dataset else None

        # Training loop
        for epoch in range(epochs):
            # Train
            train_losses = self.train_epoch(train_loader, verbose=False)

            # Validate
            if val_loader:
                val_losses = self.evaluate(val_loader)
            else:
                val_losses = {}

            if verbose:
                logger.info(
                    f"Epoch {epoch + 1}/{epochs}: "
                    f"Train Loss={train_losses['total_loss']:.4f}, "
                    f"Val Loss={val_losses.get('total_loss', 0):.4f}"
                )

        return self.history

    def evaluate(self, data_loader: DataLoader) -> Dict[str, float]:
        """Evaluate model on data.

        Parameters
        ----------
        data_loader : DataLoader
            DataLoader providing evaluation data

        Returns
        -------
        Dict[str, float]
            Average losses
        """
        self.model.eval()

        total_losses = {
            'policy_loss': 0.0,
            'value_loss': 0.0,
            'total_loss': 0.0
        }

        num_batches = 0

        with torch.no_grad():
            for positions, policies, values in data_loader:
                positions = positions.to(self.device)
                policies = policies.to(self.device)
                values = values.to(self.device)

                # Forward pass
                policy_logits, value = self.model(positions)

                # Compute losses
                policy_loss = -(policies * torch.log_softmax(policy_logits, dim=1)).sum(dim=1).mean()
                value_loss = torch.nn.functional.mse_loss(value.squeeze(), values)
                total_loss = policy_loss + value_loss

                total_losses['policy_loss'] += policy_loss.item()
                total_losses['value_loss'] += value_loss.item()
                total_losses['total_loss'] += total_loss.item()

                num_batches += 1

        # Average losses
        for key in total_losses:
            total_losses[key] /= num_batches

        return total_losses

    def save_checkpoint(self, path: str, epoch: int, metadata: Optional[Dict] = None) -> None:
        """Save training checkpoint.

        Parameters
        ----------
        path : str
            Path to save checkpoint
        epoch : int
            Current epoch number
        metadata : Optional[Dict]
            Additional metadata to save
        """
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'history': self.history,
            'metadata': metadata or {}
        }

        torch.save(checkpoint, path)
        logger.info(f"Checkpoint saved to {path}")

    def load_checkpoint(self, path: str) -> Dict[str, Any]:
        """Load training checkpoint.

        Parameters
        ----------
        path : str
            Path to checkpoint file

        Returns
        -------
        Dict[str, Any]
            Checkpoint metadata
        """
        checkpoint = torch.load(path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.history = checkpoint.get('history', self.history)

        logger.info(f"Checkpoint loaded from {path}")

        return {
            'epoch': checkpoint['epoch'],
            'metadata': checkpoint.get('metadata', {})
        }


def create_training_sample_from_game(
    game_record: Dict[str, Any],
    board_encoder
) -> List[Dict[str, Any]]:
    """Convert game record to training samples.

    Parameters
    ----------
    game_record : Dict[str, Any]
        Game record with moves and outcome
    board_encoder : BoardEncoder
        Encoder for converting positions to features

    Returns
    -------
    List[Dict[str, Any]]
        List of training samples
    """
    # TODO: Implement full game record parsing
    # This is a placeholder for the integration with self-play
    samples = []

    # Extract positions, moves, and outcome from game_record
    # For each position:
    # - Encode board features
    # - Create policy target (from MCTS or actual move)
    # - Create value target (from game outcome)

    return samples


def train_from_sgf_directory(
    model: nn.Module,
    sgf_directory: str,
    epochs: int = 10,
    batch_size: int = 64
) -> Dict[str, List[float]]:
    """Train model on SGF files.

    Parameters
    ----------
    model : nn.Module
        Model to train
    sgf_directory : str
        Directory containing SGF files
    epochs : int
        Number of training epochs
    batch_size : int
        Batch size

    Returns
    -------
    Dict[str, List[float]]
        Training history
    """
    from input.sgf_to_input import parse_sgf
    from hf_models.board_encoder import BoardEncoder

    encoder = BoardEncoder()
    trainer = ModelTrainer(model)

    # Load and convert SGF files to training data
    training_samples = []

    for filename in os.listdir(sgf_directory):
        if not filename.endswith('.sgf'):
            continue

        sgf_path = os.path.join(sgf_directory, filename)

        try:
            # Parse SGF
            matrix, metadata, board = parse_sgf(sgf_path)

            # Encode position
            features = encoder.from_sgf_parse((matrix, metadata, board))

            # Create training sample (simplified - needs proper policy/value targets)
            sample = {
                'position': features,
                'policy': np.zeros(19 * 19 + 1),  # Placeholder
                'value': 0.0  # Placeholder
            }

            training_samples.append(sample)

        except Exception as e:
            logger.warning(f"Failed to process {filename}: {e}")
            continue

    if not training_samples:
        logger.error("No training samples generated")
        return {}

    # Train
    logger.info(f"Training on {len(training_samples)} samples")
    return trainer.train(training_samples, epochs=epochs, batch_size=batch_size)

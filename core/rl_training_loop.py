"""AlphaGo Zero-style reinforcement learning training loop.

This module implements the complete RL training cycle:
1. Self-play generates training examples → ReplayBuffer
2. Sample batch from buffer (with augmentation)
3. Train model on batch
4. Every N iterations, verify improvement against best model
5. If improved, replace best model
"""

from __future__ import annotations

import os
import time
import logging
import copy
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
import numpy as np
import torch
import torch.optim as optim

from core.replay_buffer import ReplayBuffer
from core.data_augmentation import BoardAugmentation
from core.policy_improvement import PolicyImprovementVerifier
from core.learning_scheduler import create_scheduler, GoTrainingScheduler
from core.self_play import SelfPlayWorker, TrainingExample

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for RL training loop.

    Attributes
    ----------
    batch_size : int
        Training batch size
    learning_rate : float
        Initial learning rate
    weight_decay : float
        L2 regularization weight
    gradient_clip : float
        Gradient clipping threshold
    use_augmentation : bool
        Whether to use data augmentation
    improvement_check_frequency : int
        Batches between improvement checks
    save_checkpoint_frequency : int
        Batches between checkpoint saves
    games_per_iteration : int
        Self-play games per iteration
    mcts_simulations : int
        MCTS simulations during self-play
    eval_mcts_simulations : int
        MCTS simulations during evaluation
    win_threshold : float
        Win rate threshold for improvement
    num_eval_games : int
        Games for improvement verification
    """

    batch_size: int = 64
    learning_rate: float = 0.01
    weight_decay: float = 1e-4
    gradient_clip: float = 1.0
    use_augmentation: bool = True
    improvement_check_frequency: int = 1000
    save_checkpoint_frequency: int = 5000
    games_per_iteration: int = 100
    mcts_simulations: int = 800
    eval_mcts_simulations: int = 400
    win_threshold: float = 0.55
    num_eval_games: int = 100
    board_size: int = 19
    device: str = 'cpu'


@dataclass
class TrainingHistory:
    """History of training metrics.

    Attributes
    ----------
    iterations : List[int]
        Iteration numbers
    policy_losses : List[float]
        Policy loss values
    value_losses : List[float]
        Value loss values
    total_losses : List[float]
        Total loss values
    learning_rates : List[float]
        Learning rates
    improvements : List[Dict]
        Improvement check results
    """

    iterations: List[int] = field(default_factory=list)
    policy_losses: List[float] = field(default_factory=list)
    value_losses: List[float] = field(default_factory=list)
    total_losses: List[float] = field(default_factory=list)
    learning_rates: List[float] = field(default_factory=list)
    improvements: List[Dict] = field(default_factory=list)


class AlphaGoZeroTrainingLoop:
    """Complete AlphaGo Zero-style reinforcement learning loop.

    Flow:
    1. Self-play generates training examples → ReplayBuffer
    2. Sample batch from buffer (with augmentation)
    3. Train model on batch
    4. Every N iterations, verify improvement against best model
    5. If improved, replace best model

    Parameters
    ----------
    model : torch.nn.Module
        Initial neural network model
    config : TrainingConfig
        Training configuration
    save_dir : str
        Directory for saving checkpoints
    """

    def __init__(
        self,
        model: torch.nn.Module,
        config: Optional[TrainingConfig] = None,
        save_dir: str = "data/rl_training"
    ):
        self.config = config or TrainingConfig()
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # Device setup
        self.device = torch.device(self.config.device)

        # Models
        self.model = model.to(self.device)
        self.best_model = copy.deepcopy(model).to(self.device)

        # Optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )

        # Learning rate scheduler
        self.scheduler: Optional[GoTrainingScheduler] = None

        # Replay buffer
        self.replay_buffer = ReplayBuffer(
            max_size=500_000,
            buffer_path=str(self.save_dir / "replay_buffer")
        )

        # Data augmentation
        self.augmenter = BoardAugmentation(board_size=self.config.board_size)

        # Policy improvement verifier
        self.verifier = PolicyImprovementVerifier(
            win_threshold=self.config.win_threshold,
            num_eval_games=self.config.num_eval_games,
            mcts_simulations=self.config.eval_mcts_simulations,
            board_size=self.config.board_size,
            device=self.config.device
        )

        # Training state
        self.current_iteration = 0
        self.total_batches = 0
        self.history = TrainingHistory()
        self.best_model_version = 0

    def set_scheduler(self, scheduler_config: Dict[str, Any]) -> None:
        """Set learning rate scheduler from config.

        Parameters
        ----------
        scheduler_config : Dict[str, Any]
            Scheduler configuration with 'type' and parameters
        """
        self.scheduler = create_scheduler(self.optimizer, scheduler_config)

    def run_training_cycle(
        self,
        num_iterations: int = 1000,
        games_per_iteration: Optional[int] = None,
        batches_per_iteration: int = 1000,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """Run complete RL training cycle.

        Parameters
        ----------
        num_iterations : int
            Number of training iterations
        games_per_iteration : Optional[int]
            Self-play games per iteration (default from config)
        batches_per_iteration : int
            Training batches per iteration
        verbose : bool
            Print progress

        Returns
        -------
        Dict[str, Any]
            Training summary
        """
        games_per_iter = games_per_iteration or self.config.games_per_iteration
        start_time = time.time()

        for iteration in range(num_iterations):
            self.current_iteration += 1
            iter_start = time.time()

            if verbose:
                logger.info(f"\n{'='*60}")
                logger.info(f"Iteration {self.current_iteration}")
                logger.info('='*60)

            # 1. Generate self-play data
            if verbose:
                logger.info(f"Generating {games_per_iter} self-play games...")

            examples = self._generate_self_play_data(games_per_iter)
            self.replay_buffer.add_examples(examples, generation=self.current_iteration)

            if verbose:
                logger.info(f"Added {len(examples)} examples to replay buffer")
                logger.info(f"Buffer size: {len(self.replay_buffer)}")

            # 2. Train on batches
            if self.replay_buffer.is_ready:
                if verbose:
                    logger.info(f"Training for {batches_per_iteration} batches...")

                metrics = self._train_iteration(batches_per_iteration, verbose)

                if verbose:
                    logger.info(
                        f"Iteration metrics - Policy loss: {metrics['policy_loss']:.4f}, "
                        f"Value loss: {metrics['value_loss']:.4f}"
                    )
            else:
                if verbose:
                    logger.info("Buffer not ready, skipping training")
                continue

            # 3. Check for improvement
            if self.total_batches % self.config.improvement_check_frequency == 0:
                if verbose:
                    logger.info("Checking for improvement...")

                improved = self._check_and_update_best(verbose)

                if improved and verbose:
                    logger.info(f"New best model! Version: {self.best_model_version}")

            # 4. Save checkpoint
            if self.total_batches % self.config.save_checkpoint_frequency == 0:
                self.save_checkpoint()
                if verbose:
                    logger.info("Checkpoint saved")

            iter_time = time.time() - iter_start
            if verbose:
                logger.info(f"Iteration completed in {iter_time:.1f}s")

        total_time = time.time() - start_time

        return {
            'iterations': self.current_iteration,
            'total_batches': self.total_batches,
            'best_model_version': self.best_model_version,
            'buffer_size': len(self.replay_buffer),
            'total_time': total_time,
            'history': self.history
        }

    def _generate_self_play_data(self, num_games: int) -> List[TrainingExample]:
        """Generate training data using current best model.

        Parameters
        ----------
        num_games : int
            Number of games to play

        Returns
        -------
        List[TrainingExample]
            Training examples from self-play
        """
        # Use best model for self-play
        self.best_model.eval()

        worker = SelfPlayWorker(
            model=self.best_model,
            num_simulations=self.config.mcts_simulations,
            board_size=self.config.board_size
        )

        results = worker.generate_games(num_games, verbose=False)

        # Flatten examples
        all_examples = []
        for result in results:
            all_examples.extend(result.examples)

        return all_examples

    def _train_iteration(
        self,
        num_batches: int,
        verbose: bool = False
    ) -> Dict[str, float]:
        """Train for one iteration (multiple batches).

        Parameters
        ----------
        num_batches : int
            Number of batches to train
        verbose : bool
            Print progress

        Returns
        -------
        Dict[str, float]
            Average metrics for iteration
        """
        self.model.train()

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_loss = 0.0

        for batch_idx in range(num_batches):
            # Sample batch from replay buffer
            features, policies, values = self.replay_buffer.sample_batch(
                self.config.batch_size
            )

            # Apply data augmentation
            if self.config.use_augmentation:
                features, policies, values = self.augmenter.augment_batch(
                    features, policies, values, num_augments=1
                )

            # Convert to tensors
            features_t = torch.tensor(features, dtype=torch.float32, device=self.device)
            policies_t = torch.tensor(policies, dtype=torch.float32, device=self.device)
            values_t = torch.tensor(values, dtype=torch.float32, device=self.device)

            # Forward pass
            policy_logits, value_pred = self.model(features_t)

            # Compute losses
            # Policy loss: cross-entropy with MCTS policy
            log_probs = torch.log_softmax(policy_logits, dim=1)
            policy_loss = -(policies_t * log_probs).sum(dim=1).mean()

            # Value loss: MSE
            value_loss = torch.nn.functional.mse_loss(
                value_pred.squeeze(), values_t
            )

            # Total loss
            loss = policy_loss + value_loss

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            if self.config.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.gradient_clip
                )

            self.optimizer.step()

            # Update scheduler
            if self.scheduler is not None:
                self.scheduler.step()

            # Accumulate metrics
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_loss += loss.item()
            self.total_batches += 1

            # Log progress
            if verbose and (batch_idx + 1) % 100 == 0:
                avg_loss = total_loss / (batch_idx + 1)
                logger.info(f"  Batch {batch_idx + 1}/{num_batches}: loss={avg_loss:.4f}")

        # Average metrics
        avg_policy_loss = total_policy_loss / num_batches
        avg_value_loss = total_value_loss / num_batches
        avg_total_loss = total_loss / num_batches

        # Record history
        self.history.iterations.append(self.total_batches)
        self.history.policy_losses.append(avg_policy_loss)
        self.history.value_losses.append(avg_value_loss)
        self.history.total_losses.append(avg_total_loss)

        lr = self.scheduler.get_lr() if self.scheduler else self.config.learning_rate
        self.history.learning_rates.append(lr)

        return {
            'policy_loss': avg_policy_loss,
            'value_loss': avg_value_loss,
            'total_loss': avg_total_loss,
            'learning_rate': lr
        }

    def _check_and_update_best(self, verbose: bool = False) -> bool:
        """Verify improvement and update best model if better.

        Parameters
        ----------
        verbose : bool
            Print progress

        Returns
        -------
        bool
            True if model improved
        """
        is_improved, win_rate, stats = self.verifier.verify_improvement(
            new_model=self.model,
            old_model=self.best_model,
            verbose=verbose
        )

        # Record improvement check
        self.history.improvements.append({
            'iteration': self.total_batches,
            'win_rate': win_rate,
            'is_improved': is_improved,
            **stats
        })

        if is_improved:
            # Update best model
            self.best_model = copy.deepcopy(self.model)
            self.best_model_version += 1

            # Save best model
            best_path = self.save_dir / "best_model.pt"
            torch.save({
                'model_state_dict': self.best_model.state_dict(),
                'version': self.best_model_version,
                'win_rate': win_rate
            }, best_path)

            logger.info(
                f"New best model saved (v{self.best_model_version}, "
                f"win_rate={win_rate:.1%})"
            )

        return is_improved

    def save_checkpoint(self, filename: Optional[str] = None) -> str:
        """Save training checkpoint.

        Parameters
        ----------
        filename : Optional[str]
            Custom filename (default: auto-generated)

        Returns
        -------
        str
            Path to saved checkpoint
        """
        if filename is None:
            filename = f"checkpoint_{self.total_batches}.pt"

        filepath = self.save_dir / filename

        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'best_model_state_dict': self.best_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'current_iteration': self.current_iteration,
            'total_batches': self.total_batches,
            'best_model_version': self.best_model_version,
            'config': self.config,
            'history': self.history
        }

        torch.save(checkpoint, filepath)

        # Save replay buffer
        self.replay_buffer.save_to_disk()

        logger.info(f"Checkpoint saved to {filepath}")
        return str(filepath)

    def load_checkpoint(self, filepath: str) -> None:
        """Load training checkpoint.

        Parameters
        ----------
        filepath : str
            Path to checkpoint file
        """
        checkpoint = torch.load(filepath, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.best_model.load_state_dict(checkpoint['best_model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if checkpoint['scheduler_state_dict'] and self.scheduler:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        self.current_iteration = checkpoint['current_iteration']
        self.total_batches = checkpoint['total_batches']
        self.best_model_version = checkpoint['best_model_version']
        self.history = checkpoint['history']

        # Load replay buffer
        self.replay_buffer.load_latest()

        logger.info(f"Checkpoint loaded from {filepath}")
        logger.info(f"Resuming from iteration {self.current_iteration}, batch {self.total_batches}")

    def get_best_model(self) -> torch.nn.Module:
        """Get the current best model.

        Returns
        -------
        torch.nn.Module
            Best model
        """
        return self.best_model


def create_training_loop(
    model: torch.nn.Module,
    config_dict: Dict[str, Any],
    save_dir: str = "data/rl_training"
) -> AlphaGoZeroTrainingLoop:
    """Create training loop from configuration dictionary.

    Parameters
    ----------
    model : torch.nn.Module
        Neural network model
    config_dict : Dict[str, Any]
        Configuration dictionary
    save_dir : str
        Save directory

    Returns
    -------
    AlphaGoZeroTrainingLoop
        Configured training loop
    """
    config = TrainingConfig(
        batch_size=config_dict.get('batch_size', 64),
        learning_rate=config_dict.get('learning_rate', 0.01),
        weight_decay=config_dict.get('weight_decay', 1e-4),
        gradient_clip=config_dict.get('gradient_clip', 1.0),
        use_augmentation=config_dict.get('use_augmentation', True),
        improvement_check_frequency=config_dict.get('improvement_check_frequency', 1000),
        save_checkpoint_frequency=config_dict.get('save_checkpoint_frequency', 5000),
        games_per_iteration=config_dict.get('games_per_iteration', 100),
        mcts_simulations=config_dict.get('mcts_simulations', 800),
        eval_mcts_simulations=config_dict.get('eval_mcts_simulations', 400),
        win_threshold=config_dict.get('win_threshold', 0.55),
        num_eval_games=config_dict.get('num_eval_games', 100),
        board_size=config_dict.get('board_size', 19),
        device=config_dict.get('device', 'cpu')
    )

    loop = AlphaGoZeroTrainingLoop(
        model=model,
        config=config,
        save_dir=save_dir
    )

    # Set scheduler if specified
    if 'lr_schedule' in config_dict:
        loop.set_scheduler(config_dict['lr_schedule'])

    return loop

"""Learning rate schedulers for Go AI training.

This module implements various learning rate scheduling strategies,
following patterns used in AlphaGo Zero and similar systems.
"""

from __future__ import annotations

import math
from typing import List, Optional, Dict, Any, Union
from abc import ABC, abstractmethod
import logging

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

logger = logging.getLogger(__name__)


class GoTrainingScheduler(ABC):
    """Abstract base class for Go training schedulers.

    Parameters
    ----------
    optimizer : Optimizer
        PyTorch optimizer
    initial_lr : float
        Initial learning rate
    """

    def __init__(self, optimizer: Optimizer, initial_lr: float = 0.01):
        self.optimizer = optimizer
        self.initial_lr = initial_lr
        self.current_step = 0
        self._current_lr = initial_lr

        # Set initial LR
        self._set_lr(initial_lr)

    @abstractmethod
    def step(self) -> float:
        """Advance scheduler by one step.

        Returns
        -------
        float
            Current learning rate after step
        """
        pass

    def get_lr(self) -> float:
        """Get current learning rate.

        Returns
        -------
        float
            Current learning rate
        """
        return self._current_lr

    def _set_lr(self, lr: float) -> None:
        """Set learning rate in optimizer.

        Parameters
        ----------
        lr : float
            Learning rate to set
        """
        self._current_lr = lr
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def state_dict(self) -> Dict[str, Any]:
        """Get scheduler state for checkpointing.

        Returns
        -------
        Dict[str, Any]
            Scheduler state
        """
        return {
            'current_step': self.current_step,
            'current_lr': self._current_lr,
            'initial_lr': self.initial_lr
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load scheduler state from checkpoint.

        Parameters
        ----------
        state_dict : Dict[str, Any]
            Scheduler state
        """
        self.current_step = state_dict['current_step']
        self._current_lr = state_dict['current_lr']
        self.initial_lr = state_dict['initial_lr']
        self._set_lr(self._current_lr)


class StepScheduler(GoTrainingScheduler):
    """Step decay learning rate scheduler.

    AlphaGo Zero style:
    - Start at 0.01
    - Decay to 0.001 after 400K steps
    - Decay to 0.0001 after 600K steps

    Parameters
    ----------
    optimizer : Optimizer
        PyTorch optimizer
    initial_lr : float
        Initial learning rate (default 0.01)
    decay_steps : List[int]
        Steps at which to decay LR (default [400000, 600000])
    decay_factor : float
        Factor to multiply LR by at each decay step (default 0.1)
    """

    def __init__(
        self,
        optimizer: Optimizer,
        initial_lr: float = 0.01,
        decay_steps: Optional[List[int]] = None,
        decay_factor: float = 0.1
    ):
        super().__init__(optimizer, initial_lr)
        self.decay_steps = sorted(decay_steps or [400_000, 600_000])
        self.decay_factor = decay_factor
        self._decay_count = 0

    def step(self) -> float:
        """Advance scheduler by one step.

        Returns
        -------
        float
            Current learning rate
        """
        self.current_step += 1

        # Check if we should decay
        while (self._decay_count < len(self.decay_steps) and
               self.current_step >= self.decay_steps[self._decay_count]):
            new_lr = self._current_lr * self.decay_factor
            self._set_lr(new_lr)
            logger.info(f"LR decay at step {self.current_step}: {self._current_lr:.6f}")
            self._decay_count += 1

        return self._current_lr

    def state_dict(self) -> Dict[str, Any]:
        state = super().state_dict()
        state['decay_count'] = self._decay_count
        return state

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        super().load_state_dict(state_dict)
        self._decay_count = state_dict.get('decay_count', 0)


class CosineScheduler(GoTrainingScheduler):
    """Cosine annealing learning rate scheduler.

    LR follows cosine curve from initial_lr to min_lr.

    Parameters
    ----------
    optimizer : Optimizer
        PyTorch optimizer
    initial_lr : float
        Initial learning rate (default 0.01)
    min_lr : float
        Minimum learning rate (default 0.0001)
    total_steps : int
        Total number of training steps (default 1000000)
    """

    def __init__(
        self,
        optimizer: Optimizer,
        initial_lr: float = 0.01,
        min_lr: float = 0.0001,
        total_steps: int = 1_000_000
    ):
        super().__init__(optimizer, initial_lr)
        self.min_lr = min_lr
        self.total_steps = total_steps

    def step(self) -> float:
        """Advance scheduler by one step.

        Returns
        -------
        float
            Current learning rate
        """
        self.current_step += 1

        if self.current_step >= self.total_steps:
            new_lr = self.min_lr
        else:
            progress = self.current_step / self.total_steps
            new_lr = self.min_lr + 0.5 * (self.initial_lr - self.min_lr) * (
                1 + math.cos(math.pi * progress)
            )

        self._set_lr(new_lr)
        return self._current_lr


class WarmupCosineScheduler(GoTrainingScheduler):
    """Cosine scheduler with linear warmup.

    Parameters
    ----------
    optimizer : Optimizer
        PyTorch optimizer
    initial_lr : float
        Initial learning rate (default 0.01)
    min_lr : float
        Minimum learning rate (default 0.0001)
    warmup_steps : int
        Number of warmup steps (default 1000)
    total_steps : int
        Total training steps (default 1000000)
    """

    def __init__(
        self,
        optimizer: Optimizer,
        initial_lr: float = 0.01,
        min_lr: float = 0.0001,
        warmup_steps: int = 1000,
        total_steps: int = 1_000_000
    ):
        super().__init__(optimizer, initial_lr)
        self.min_lr = min_lr
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps

        # Start with very small LR
        self._set_lr(min_lr)

    def step(self) -> float:
        """Advance scheduler by one step.

        Returns
        -------
        float
            Current learning rate
        """
        self.current_step += 1

        if self.current_step < self.warmup_steps:
            # Linear warmup
            progress = self.current_step / self.warmup_steps
            new_lr = self.min_lr + progress * (self.initial_lr - self.min_lr)
        elif self.current_step >= self.total_steps:
            new_lr = self.min_lr
        else:
            # Cosine decay after warmup
            progress = (self.current_step - self.warmup_steps) / (
                self.total_steps - self.warmup_steps
            )
            new_lr = self.min_lr + 0.5 * (self.initial_lr - self.min_lr) * (
                1 + math.cos(math.pi * progress)
            )

        self._set_lr(new_lr)
        return self._current_lr


class ExponentialScheduler(GoTrainingScheduler):
    """Exponential decay learning rate scheduler.

    Parameters
    ----------
    optimizer : Optimizer
        PyTorch optimizer
    initial_lr : float
        Initial learning rate (default 0.01)
    decay_rate : float
        Decay rate per step (default 0.99999)
    min_lr : float
        Minimum learning rate (default 0.0001)
    """

    def __init__(
        self,
        optimizer: Optimizer,
        initial_lr: float = 0.01,
        decay_rate: float = 0.99999,
        min_lr: float = 0.0001
    ):
        super().__init__(optimizer, initial_lr)
        self.decay_rate = decay_rate
        self.min_lr = min_lr

    def step(self) -> float:
        """Advance scheduler by one step.

        Returns
        -------
        float
            Current learning rate
        """
        self.current_step += 1

        new_lr = max(
            self.min_lr,
            self.initial_lr * (self.decay_rate ** self.current_step)
        )

        self._set_lr(new_lr)
        return self._current_lr


class PlateauScheduler(GoTrainingScheduler):
    """Reduce LR on plateau scheduler.

    Reduces LR when a metric stops improving.

    Parameters
    ----------
    optimizer : Optimizer
        PyTorch optimizer
    initial_lr : float
        Initial learning rate (default 0.01)
    patience : int
        Steps to wait before reducing LR (default 10000)
    factor : float
        Factor to reduce LR by (default 0.5)
    min_lr : float
        Minimum learning rate (default 0.0001)
    threshold : float
        Minimum improvement to reset patience (default 0.01)
    """

    def __init__(
        self,
        optimizer: Optimizer,
        initial_lr: float = 0.01,
        patience: int = 10000,
        factor: float = 0.5,
        min_lr: float = 0.0001,
        threshold: float = 0.01
    ):
        super().__init__(optimizer, initial_lr)
        self.patience = patience
        self.factor = factor
        self.min_lr = min_lr
        self.threshold = threshold

        self._best_metric = float('inf')
        self._steps_without_improvement = 0

    def step(self, metric: Optional[float] = None) -> float:
        """Advance scheduler, optionally with metric update.

        Parameters
        ----------
        metric : Optional[float]
            Current metric value (lower is better)

        Returns
        -------
        float
            Current learning rate
        """
        self.current_step += 1

        if metric is not None:
            if metric < self._best_metric - self.threshold:
                self._best_metric = metric
                self._steps_without_improvement = 0
            else:
                self._steps_without_improvement += 1

                if self._steps_without_improvement >= self.patience:
                    new_lr = max(self.min_lr, self._current_lr * self.factor)
                    if new_lr < self._current_lr:
                        self._set_lr(new_lr)
                        logger.info(
                            f"LR reduced on plateau at step {self.current_step}: "
                            f"{self._current_lr:.6f}"
                        )
                    self._steps_without_improvement = 0

        return self._current_lr


def create_scheduler(
    optimizer: Optimizer,
    config: Dict[str, Any]
) -> GoTrainingScheduler:
    """Create scheduler from configuration.

    Parameters
    ----------
    optimizer : Optimizer
        PyTorch optimizer
    config : Dict[str, Any]
        Configuration with 'type' and scheduler-specific parameters

    Returns
    -------
    GoTrainingScheduler
        Configured scheduler

    Raises
    ------
    ValueError
        If scheduler type is unknown
    """
    scheduler_type = config.get('type', 'step').lower()

    if scheduler_type == 'step':
        return StepScheduler(
            optimizer=optimizer,
            initial_lr=config.get('initial_lr', 0.01),
            decay_steps=config.get('decay_steps', [400_000, 600_000]),
            decay_factor=config.get('decay_factor', 0.1)
        )
    elif scheduler_type == 'cosine':
        return CosineScheduler(
            optimizer=optimizer,
            initial_lr=config.get('initial_lr', 0.01),
            min_lr=config.get('min_lr', 0.0001),
            total_steps=config.get('total_steps', 1_000_000)
        )
    elif scheduler_type in ('warmup_cosine', 'warmup-cosine'):
        return WarmupCosineScheduler(
            optimizer=optimizer,
            initial_lr=config.get('initial_lr', 0.01),
            min_lr=config.get('min_lr', 0.0001),
            warmup_steps=config.get('warmup_steps', 1000),
            total_steps=config.get('total_steps', 1_000_000)
        )
    elif scheduler_type == 'exponential':
        return ExponentialScheduler(
            optimizer=optimizer,
            initial_lr=config.get('initial_lr', 0.01),
            decay_rate=config.get('decay_rate', 0.99999),
            min_lr=config.get('min_lr', 0.0001)
        )
    elif scheduler_type == 'plateau':
        return PlateauScheduler(
            optimizer=optimizer,
            initial_lr=config.get('initial_lr', 0.01),
            patience=config.get('patience', 10000),
            factor=config.get('factor', 0.5),
            min_lr=config.get('min_lr', 0.0001),
            threshold=config.get('threshold', 0.01)
        )
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")


class SchedulerWrapper(_LRScheduler):
    """Wrapper to make GoTrainingScheduler compatible with PyTorch LR schedulers.

    This allows using GoTrainingScheduler with training loops that expect
    standard PyTorch scheduler interface.

    Parameters
    ----------
    scheduler : GoTrainingScheduler
        GoTrainingScheduler instance to wrap
    """

    def __init__(self, scheduler: GoTrainingScheduler):
        self._go_scheduler = scheduler
        self.optimizer = scheduler.optimizer
        self._step_count = 0

    def step(self, epoch: Optional[int] = None) -> None:
        """Advance scheduler."""
        self._go_scheduler.step()
        self._step_count += 1

    def get_last_lr(self) -> List[float]:
        """Get last computed learning rate."""
        return [self._go_scheduler.get_lr()]

    def state_dict(self) -> Dict[str, Any]:
        """Get state dict."""
        return self._go_scheduler.state_dict()

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load state dict."""
        self._go_scheduler.load_state_dict(state_dict)

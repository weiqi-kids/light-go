"""Data augmentation for Go board positions.

This module implements board symmetry transformations for training data augmentation.
Go boards have 8-fold symmetry (D4 dihedral group): 4 rotations + 4 reflections.
"""

from __future__ import annotations

import numpy as np
from typing import List, Tuple, Callable, Optional, Any
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class AugmentedExample:
    """Training example with augmentation metadata.

    Attributes
    ----------
    features : np.ndarray
        Transformed board features (2, board_size, board_size)
    policy : np.ndarray
        Transformed MCTS policy targets (board_size^2 + 1)
    value : float
        Game outcome (unchanged by augmentation)
    transform_idx : int
        Index of transformation applied (0-7)
    """

    features: np.ndarray
    policy: np.ndarray
    value: float
    transform_idx: int


class BoardAugmentation:
    """Data augmentation via board symmetries.

    Go boards have 8-fold symmetry (D4 dihedral group):
    - Identity (idx=0)
    - 90° rotation (idx=1)
    - 180° rotation (idx=2)
    - 270° rotation (idx=3)
    - Horizontal flip (idx=4)
    - Vertical flip (idx=5)
    - Diagonal flip (idx=6)
    - Anti-diagonal flip (idx=7)

    Parameters
    ----------
    board_size : int
        Size of the Go board (default 19)
    include_identity : bool
        Whether to include identity transform (default True)
    """

    def __init__(self, board_size: int = 19, include_identity: bool = True):
        self.board_size = board_size
        self.include_identity = include_identity

        # Pre-compute index mappings for each transformation
        self._index_mappings = self._precompute_index_mappings()

    def _precompute_index_mappings(self) -> List[np.ndarray]:
        """Precompute index mappings for all 8 transformations.

        Returns
        -------
        List[np.ndarray]
            List of index mapping arrays for each transformation
        """
        size = self.board_size
        total_moves = size * size + 1  # Including pass move

        mappings = []

        for transform_idx in range(8):
            mapping = np.zeros(total_moves, dtype=np.int32)

            for row in range(size):
                for col in range(size):
                    old_idx = row * size + col
                    new_row, new_col = self._transform_coord(row, col, transform_idx)
                    new_idx = new_row * size + new_col
                    mapping[old_idx] = new_idx

            # Pass move stays at the same index
            mapping[size * size] = size * size

            mappings.append(mapping)

        return mappings

    def _transform_coord(
        self,
        row: int,
        col: int,
        transform_idx: int
    ) -> Tuple[int, int]:
        """Transform a single coordinate.

        Parameters
        ----------
        row : int
            Row index
        col : int
            Column index
        transform_idx : int
            Transformation index (0-7)

        Returns
        -------
        Tuple[int, int]
            Transformed (row, col)
        """
        size = self.board_size
        last = size - 1

        if transform_idx == 0:
            # Identity
            return row, col
        elif transform_idx == 1:
            # 90° clockwise rotation
            return col, last - row
        elif transform_idx == 2:
            # 180° rotation
            return last - row, last - col
        elif transform_idx == 3:
            # 270° clockwise rotation
            return last - col, row
        elif transform_idx == 4:
            # Horizontal flip (left-right)
            return row, last - col
        elif transform_idx == 5:
            # Vertical flip (top-bottom)
            return last - row, col
        elif transform_idx == 6:
            # Diagonal flip (transpose)
            return col, row
        elif transform_idx == 7:
            # Anti-diagonal flip
            return last - col, last - row
        else:
            raise ValueError(f"Invalid transform_idx: {transform_idx}")

    def transform_features(
        self,
        features: np.ndarray,
        transform_idx: int
    ) -> np.ndarray:
        """Apply transformation to feature planes.

        Parameters
        ----------
        features : np.ndarray
            Feature planes (num_planes, board_size, board_size)
        transform_idx : int
            Transformation index (0-7)

        Returns
        -------
        np.ndarray
            Transformed features
        """
        if transform_idx == 0:
            return features.copy()

        # Apply transformation to each plane
        transformed = np.empty_like(features)

        for plane_idx in range(features.shape[0]):
            plane = features[plane_idx]
            transformed[plane_idx] = self._transform_plane(plane, transform_idx)

        return transformed

    def _transform_plane(
        self,
        plane: np.ndarray,
        transform_idx: int
    ) -> np.ndarray:
        """Apply transformation to a single plane.

        Parameters
        ----------
        plane : np.ndarray
            Single plane (board_size, board_size)
        transform_idx : int
            Transformation index (0-7)

        Returns
        -------
        np.ndarray
            Transformed plane
        """
        if transform_idx == 0:
            return plane.copy()
        elif transform_idx == 1:
            # 90° clockwise: transpose then flip horizontally
            return np.flip(plane.T, axis=1)
        elif transform_idx == 2:
            # 180°: flip both axes
            return np.flip(np.flip(plane, axis=0), axis=1)
        elif transform_idx == 3:
            # 270° clockwise: flip horizontally then transpose
            return np.flip(plane, axis=1).T
        elif transform_idx == 4:
            # Horizontal flip
            return np.flip(plane, axis=1)
        elif transform_idx == 5:
            # Vertical flip
            return np.flip(plane, axis=0)
        elif transform_idx == 6:
            # Diagonal flip (transpose)
            return plane.T
        elif transform_idx == 7:
            # Anti-diagonal flip
            return np.flip(np.flip(plane.T, axis=0), axis=1)
        else:
            raise ValueError(f"Invalid transform_idx: {transform_idx}")

    def transform_policy(
        self,
        policy: np.ndarray,
        transform_idx: int
    ) -> np.ndarray:
        """Apply transformation to policy vector.

        Parameters
        ----------
        policy : np.ndarray
            Policy vector (board_size^2 + 1)
        transform_idx : int
            Transformation index (0-7)

        Returns
        -------
        np.ndarray
            Transformed policy (pass probability preserved)
        """
        if transform_idx == 0:
            return policy.copy()

        mapping = self._index_mappings[transform_idx]
        transformed = np.zeros_like(policy)

        # Apply mapping
        for old_idx, new_idx in enumerate(mapping):
            transformed[new_idx] = policy[old_idx]

        return transformed

    def augment_example(
        self,
        features: np.ndarray,
        policy: np.ndarray,
        value: float
    ) -> List[AugmentedExample]:
        """Generate all 8 symmetry variants of an example.

        Parameters
        ----------
        features : np.ndarray
            Board features (num_planes, board_size, board_size)
        policy : np.ndarray
            MCTS policy (board_size^2 + 1)
        value : float
            Game outcome

        Returns
        -------
        List[AugmentedExample]
            8 augmented examples (including identity if include_identity=True)
        """
        start_idx = 0 if self.include_identity else 1

        examples = []
        for transform_idx in range(start_idx, 8):
            aug_features = self.transform_features(features, transform_idx)
            aug_policy = self.transform_policy(policy, transform_idx)

            examples.append(AugmentedExample(
                features=aug_features,
                policy=aug_policy,
                value=value,
                transform_idx=transform_idx
            ))

        return examples

    def augment_training_example(
        self,
        example: Any
    ) -> List[Any]:
        """Augment a TrainingExample object.

        Parameters
        ----------
        example : Any
            TrainingExample with features, policy, value attributes

        Returns
        -------
        List[Any]
            List of augmented examples (same type as input)
        """
        augmented = self.augment_example(
            features=example.features,
            policy=example.policy,
            value=example.value
        )

        # Try to create same type as input
        try:
            from core.self_play import TrainingExample
            return [
                TrainingExample(
                    features=aug.features,
                    policy=aug.policy,
                    value=aug.value
                )
                for aug in augmented
            ]
        except ImportError:
            return augmented

    def augment_batch(
        self,
        features: np.ndarray,
        policies: np.ndarray,
        values: np.ndarray,
        num_augments: int = 1
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Augment a batch of examples with random transformations.

        Parameters
        ----------
        features : np.ndarray
            Batch of features (batch_size, num_planes, board_size, board_size)
        policies : np.ndarray
            Batch of policies (batch_size, board_size^2 + 1)
        values : np.ndarray
            Batch of values (batch_size,)
        num_augments : int
            Number of random augmentations per example (default 1)

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray]
            Augmented (features, policies, values)
        """
        batch_size = features.shape[0]
        total_size = batch_size * num_augments

        aug_features = np.empty((total_size,) + features.shape[1:], dtype=features.dtype)
        aug_policies = np.empty((total_size,) + policies.shape[1:], dtype=policies.dtype)
        aug_values = np.empty(total_size, dtype=values.dtype)

        idx = 0
        for i in range(batch_size):
            for _ in range(num_augments):
                # Random transformation
                transform_idx = np.random.randint(0, 8)

                aug_features[idx] = self.transform_features(features[i], transform_idx)
                aug_policies[idx] = self.transform_policy(policies[i], transform_idx)
                aug_values[idx] = values[i]
                idx += 1

        return aug_features, aug_policies, aug_values

    def get_random_transform(self) -> int:
        """Get a random transformation index.

        Returns
        -------
        int
            Random transformation index (0-7)
        """
        return np.random.randint(0, 8)

    @staticmethod
    def get_transform_name(transform_idx: int) -> str:
        """Get human-readable name for transformation.

        Parameters
        ----------
        transform_idx : int
            Transformation index (0-7)

        Returns
        -------
        str
            Transformation name
        """
        names = [
            "identity",
            "rotate_90",
            "rotate_180",
            "rotate_270",
            "flip_horizontal",
            "flip_vertical",
            "flip_diagonal",
            "flip_antidiagonal"
        ]
        return names[transform_idx] if 0 <= transform_idx < 8 else "unknown"


def augment_dataset(
    examples: List[Any],
    board_size: int = 19,
    include_original: bool = True
) -> List[Any]:
    """Augment an entire dataset with all symmetries.

    Parameters
    ----------
    examples : List[Any]
        List of TrainingExample objects
    board_size : int
        Board size (default 19)
    include_original : bool
        Whether to include original examples (default True)

    Returns
    -------
    List[Any]
        Augmented dataset (8x size if include_original=True, 7x if False)
    """
    augmenter = BoardAugmentation(board_size=board_size, include_identity=include_original)

    augmented = []
    for example in examples:
        augmented.extend(augmenter.augment_training_example(example))

    logger.info(f"Augmented {len(examples)} examples to {len(augmented)} examples")
    return augmented


def apply_random_augmentation(
    features: np.ndarray,
    policy: np.ndarray,
    board_size: int = 19
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply a random augmentation to a single example.

    Parameters
    ----------
    features : np.ndarray
        Board features
    policy : np.ndarray
        Policy vector
    board_size : int
        Board size (default 19)

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        (augmented_features, augmented_policy)
    """
    augmenter = BoardAugmentation(board_size=board_size)
    transform_idx = augmenter.get_random_transform()

    return (
        augmenter.transform_features(features, transform_idx),
        augmenter.transform_policy(policy, transform_idx)
    )

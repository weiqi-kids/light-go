"""Architecture genome for evolvable neural network structures.

This module implements the core innovation of Light-Go: neural architectures that can
evolve through genetic operations (mutation, crossover). Each genome encodes a complete
network architecture that can be converted to a PyTorch model.
"""

from __future__ import annotations

import random
import hashlib
import json
import pickle
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict, field
import torch.nn as nn

# Will be imported when needed to avoid circular dependency
# from hf_models.modeling_go_ai import GoAIModel


@dataclass
class ArchitectureGenome:
    """Genetic representation of a neural network architecture.

    Attributes
    ----------
    num_blocks : int
        Number of residual blocks in the network body (3-20 range)
    base_filters : int
        Base number of filters, blocks may scale from this (64-512 range)
    filters_per_block : List[int]
        Specific filter count for each block
    kernel_sizes : List[int]
        Kernel size for each block (3, 5, or 7)
    block_types : List[str]
        Type of each block ('residual', 'dense', 'bottleneck')
    use_se_blocks : bool
        Whether to use Squeeze-and-Excitation blocks
    se_reduction_ratio : int
        Reduction ratio for SE blocks (4, 8, or 16)
    dropout_rate : float
        Dropout rate for regularization (0.0-0.5)
    num_input_planes : int
        Number of input feature planes
    board_size : int
        Size of the Go board
    generation : int
        Generation number in evolution
    parent_ids : List[str]
        IDs of parent genomes (for tracking lineage)
    """

    num_blocks: int = 5
    base_filters: int = 128
    filters_per_block: List[int] = field(default_factory=list)
    kernel_sizes: List[int] = field(default_factory=list)
    block_types: List[str] = field(default_factory=list)
    use_se_blocks: bool = False
    se_reduction_ratio: int = 8
    dropout_rate: float = 0.0
    num_input_planes: int = 2  # LightGo format (signed liberties + forbidden)
    board_size: int = 19
    generation: int = 0
    parent_ids: List[str] = field(default_factory=list)

    def __post_init__(self):
        """Initialize derived fields if not provided."""
        if not self.filters_per_block:
            self.filters_per_block = [self.base_filters] * self.num_blocks
        if not self.kernel_sizes:
            self.kernel_sizes = [3] * self.num_blocks
        if not self.block_types:
            self.block_types = ['residual'] * self.num_blocks

        # Ensure list lengths match num_blocks
        self._ensure_consistency()

    def _ensure_consistency(self):
        """Ensure all lists have correct length matching num_blocks."""
        # Truncate or extend to match num_blocks
        for attr_name in ['filters_per_block', 'kernel_sizes', 'block_types']:
            attr_list = getattr(self, attr_name)
            if len(attr_list) > self.num_blocks:
                setattr(self, attr_name, attr_list[:self.num_blocks])
            elif len(attr_list) < self.num_blocks:
                # Extend by repeating the pattern
                pattern = attr_list if attr_list else [getattr(self, attr_name.replace('_per_block', '').replace('s', ''), 128 if 'filter' in attr_name else 3)]
                extended = (attr_list + pattern * ((self.num_blocks // len(pattern)) + 1))[:self.num_blocks]
                setattr(self, attr_name, extended)

    def get_id(self) -> str:
        """Generate unique ID for this genome based on its genes.

        Returns
        -------
        str
            SHA256 hash of genome parameters (first 8 characters)
        """
        # Create canonical representation
        genes_dict = {
            'num_blocks': self.num_blocks,
            'filters': tuple(self.filters_per_block),
            'kernels': tuple(self.kernel_sizes),
            'blocks': tuple(self.block_types),
            'se': self.use_se_blocks,
            'se_ratio': self.se_reduction_ratio,
            'dropout': self.dropout_rate
        }
        genes_str = json.dumps(genes_dict, sort_keys=True)
        return hashlib.sha256(genes_str.encode()).hexdigest()[:8]

    def mutate(self, mutation_rate: float = 0.2) -> "ArchitectureGenome":
        """Create a mutated copy of this genome.

        Parameters
        ----------
        mutation_rate : float
            Probability of mutating each gene (default 0.2)

        Returns
        -------
        ArchitectureGenome
            New genome with mutations applied
        """
        # Create a deep copy
        mutated = ArchitectureGenome(**asdict(self))
        mutated.generation = self.generation + 1
        mutated.parent_ids = [self.get_id()]

        # Mutate num_blocks
        if random.random() < mutation_rate:
            delta = random.choice([-1, 0, 0, 1])  # Bias toward no change
            mutated.num_blocks = max(3, min(20, self.num_blocks + delta))

        # Mutate base_filters
        if random.random() < mutation_rate:
            mutated.base_filters = random.choice([64, 96, 128, 160, 192, 256, 320, 384, 512])

        # Mutate individual block properties
        mutated.filters_per_block = list(self.filters_per_block)
        mutated.kernel_sizes = list(self.kernel_sizes)
        mutated.block_types = list(self.block_types)

        for i in range(min(len(mutated.filters_per_block), mutated.num_blocks)):
            if random.random() < mutation_rate:
                # Mutate filter count
                mutated.filters_per_block[i] = random.choice([64, 128, 192, 256, 384, 512])

            if random.random() < mutation_rate:
                # Mutate kernel size
                mutated.kernel_sizes[i] = random.choice([3, 5, 7])

            if random.random() < mutation_rate:
                # Mutate block type
                mutated.block_types[i] = random.choice(['residual', 'residual', 'dense'])  # Bias toward residual

        # Mutate SE blocks
        if random.random() < mutation_rate:
            mutated.use_se_blocks = not mutated.use_se_blocks

        if random.random() < mutation_rate:
            mutated.se_reduction_ratio = random.choice([4, 8, 16])

        # Mutate dropout
        if random.random() < mutation_rate:
            mutated.dropout_rate = random.choice([0.0, 0.1, 0.2, 0.3, 0.4])

        mutated._ensure_consistency()
        return mutated

    @staticmethod
    def crossover(parent1: "ArchitectureGenome", parent2: "ArchitectureGenome") -> "ArchitectureGenome":
        """Create offspring by combining two parent genomes.

        Parameters
        ----------
        parent1, parent2 : ArchitectureGenome
            Parent genomes to combine

        Returns
        -------
        ArchitectureGenome
            Child genome with genes from both parents
        """
        # Average num_blocks (rounded)
        num_blocks = (parent1.num_blocks + parent2.num_blocks) // 2

        # Choose base_filters from one parent randomly
        base_filters = random.choice([parent1.base_filters, parent2.base_filters])

        # Combine block properties through crossover
        child = ArchitectureGenome(
            num_blocks=num_blocks,
            base_filters=base_filters,
            num_input_planes=parent1.num_input_planes,  # Assume same
            board_size=parent1.board_size,  # Assume same
            generation=max(parent1.generation, parent2.generation) + 1,
            parent_ids=[parent1.get_id(), parent2.get_id()]
        )

        # Crossover block properties (take random segments from each parent)
        crossover_point = random.randint(1, num_blocks - 1)

        child.filters_per_block = (
            parent1.filters_per_block[:crossover_point] +
            parent2.filters_per_block[crossover_point:num_blocks]
        )

        child.kernel_sizes = (
            parent1.kernel_sizes[:crossover_point] +
            parent2.kernel_sizes[crossover_point:num_blocks]
        )

        child.block_types = (
            parent1.block_types[:crossover_point] +
            parent2.block_types[crossover_point:num_blocks]
        )

        # Inherit SE and dropout from random parent
        if random.random() < 0.5:
            child.use_se_blocks = parent1.use_se_blocks
            child.se_reduction_ratio = parent1.se_reduction_ratio
        else:
            child.use_se_blocks = parent2.use_se_blocks
            child.se_reduction_ratio = parent2.se_reduction_ratio

        child.dropout_rate = (parent1.dropout_rate + parent2.dropout_rate) / 2.0

        child._ensure_consistency()
        return child

    def to_pytorch_model(self, device: Optional[str] = None):
        """Build PyTorch model from this genome.

        Parameters
        ----------
        device : Optional[str]
            Device to place model on ('cuda' or 'cpu')

        Returns
        -------
        GoAIModel
            Neural network model built according to genome specification
        """
        # Import here to avoid circular dependency
        from hf_models.modeling_go_ai import GoAIModel

        # For now, use simplified mapping to GoAIModel parameters
        # In future, this could build custom architectures based on block_types
        avg_filters = int(sum(self.filters_per_block) / len(self.filters_per_block))

        return GoAIModel(
            num_input_planes=self.num_input_planes,
            num_filters=avg_filters,
            num_blocks=self.num_blocks,
            board_size=self.board_size,
            device=device
        )

    def estimate_parameters(self) -> int:
        """Estimate number of trainable parameters.

        Returns
        -------
        int
            Estimated parameter count
        """
        # Rough estimation based on architecture
        params = 0

        # Input conv: num_input_planes * first_filters * 3 * 3
        first_filters = self.filters_per_block[0] if self.filters_per_block else self.base_filters
        params += self.num_input_planes * first_filters * 9

        # Residual blocks: filters^2 * 9 * 2 (two convs per block)
        for filters in self.filters_per_block:
            params += filters * filters * 9 * 2

        # Policy head: ~filters * board_size^2
        last_filters = self.filters_per_block[-1] if self.filters_per_block else self.base_filters
        params += last_filters * self.board_size * self.board_size

        # Value head: similar size
        params += last_filters * self.board_size * self.board_size

        return params

    def to_dict(self) -> Dict[str, Any]:
        """Convert genome to dictionary for serialization.

        Returns
        -------
        Dict[str, Any]
            Dictionary representation of genome
        """
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ArchitectureGenome":
        """Create genome from dictionary.

        Parameters
        ----------
        data : Dict[str, Any]
            Dictionary containing genome data

        Returns
        -------
        ArchitectureGenome
            Reconstructed genome
        """
        return cls(**data)

    @classmethod
    def create_baseline(cls, board_size: int = 19) -> "ArchitectureGenome":
        """Create a baseline genome with standard configuration.

        Parameters
        ----------
        board_size : int
            Size of the Go board

        Returns
        -------
        ArchitectureGenome
            Baseline genome suitable for initial population
        """
        return cls(
            num_blocks=5,
            base_filters=128,
            filters_per_block=[128] * 5,
            kernel_sizes=[3] * 5,
            block_types=['residual'] * 5,
            use_se_blocks=False,
            se_reduction_ratio=8,
            dropout_rate=0.0,
            num_input_planes=2,  # LightGo format
            board_size=board_size,
            generation=0,
            parent_ids=[]
        )

    @classmethod
    def create_random(cls, board_size: int = 19) -> "ArchitectureGenome":
        """Create a random genome for diversity injection.

        Parameters
        ----------
        board_size : int
            Size of the Go board

        Returns
        -------
        ArchitectureGenome
            Randomly generated genome
        """
        num_blocks = random.randint(3, 12)
        base_filters = random.choice([64, 96, 128, 192, 256])

        return cls(
            num_blocks=num_blocks,
            base_filters=base_filters,
            filters_per_block=[random.choice([64, 128, 192, 256, 384]) for _ in range(num_blocks)],
            kernel_sizes=[random.choice([3, 5, 7]) for _ in range(num_blocks)],
            block_types=[random.choice(['residual', 'residual', 'dense']) for _ in range(num_blocks)],
            use_se_blocks=random.choice([True, False]),
            se_reduction_ratio=random.choice([4, 8, 16]),
            dropout_rate=random.choice([0.0, 0.1, 0.2, 0.3]),
            num_input_planes=2,  # LightGo format
            board_size=board_size,
            generation=0,
            parent_ids=[]
        )

    def __str__(self) -> str:
        """String representation of genome."""
        return (
            f"ArchitectureGenome(id={self.get_id()}, "
            f"blocks={self.num_blocks}, "
            f"filters={self.base_filters}, "
            f"params≈{self.estimate_parameters():,}, "
            f"gen={self.generation})"
        )

    def __repr__(self) -> str:
        return self.__str__()


# Utility functions

def create_initial_population(
    size: int = 5,
    board_size: int = 19,
    include_baseline: bool = True
) -> List[ArchitectureGenome]:
    """Create initial population for evolution.

    Parameters
    ----------
    size : int
        Number of genomes to create
    board_size : int
        Size of the Go board
    include_baseline : bool
        Whether to include baseline genome as first member

    Returns
    -------
    List[ArchitectureGenome]
        Initial population
    """
    population = []

    if include_baseline:
        population.append(ArchitectureGenome.create_baseline(board_size))
        size -= 1

    # Add random variations
    for _ in range(size):
        if random.random() < 0.5:
            # Mutate baseline
            genome = ArchitectureGenome.create_baseline(board_size).mutate(mutation_rate=0.3)
        else:
            # Fully random
            genome = ArchitectureGenome.create_random(board_size)
        population.append(genome)

    return population

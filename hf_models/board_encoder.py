"""Convert liberty matrix representation to neural network feature planes.

The BoardEncoder transforms the Light-Go liberty representation (list of tuples with
position and liberty count) into multi-channel feature planes suitable for CNN input.
"""

from __future__ import annotations

import numpy as np
from typing import List, Tuple, Dict, Any


class BoardEncoder:
    """Encoder for converting liberty matrices to neural network features.

    The encoder produces multiple feature planes:
    - Plane 0: Black stones (binary)
    - Plane 1: White stones (binary)
    - Plane 2: Black liberties (normalized)
    - Plane 3: White liberties (normalized)
    - Plane 4: Forbidden points (binary)
    - Plane 5-6: Current player indicators

    Parameters
    ----------
    board_size : int
        Size of the Go board (default 19 for standard board)
    num_feature_planes : int
        Number of feature planes to generate (default 7)
    """

    def __init__(self, board_size: int = 19, num_feature_planes: int = 7):
        self.board_size = board_size
        self.num_feature_planes = num_feature_planes

    def encode(
        self,
        liberty: List[Tuple[int, int, int]],
        forbidden: List[Tuple[int, int]],
        metadata: Dict[str, Any]
    ) -> np.ndarray:
        """Convert liberty matrix and metadata to feature planes.

        Parameters
        ----------
        liberty : List[Tuple[int, int, int]]
            List of (x, y, liberty_count) where positive = black, negative = white
        forbidden : List[Tuple[int, int]]
            List of (x, y) coordinates that are illegal moves
        metadata : Dict[str, Any]
            Metadata including 'next_move' (color to play)

        Returns
        -------
        np.ndarray
            Feature tensor of shape (num_feature_planes, board_size, board_size)
        """
        # Initialize feature planes
        features = np.zeros(
            (self.num_feature_planes, self.board_size, self.board_size),
            dtype=np.float32
        )

        # Plane 0: Black stones (binary)
        # Plane 1: White stones (binary)
        # Plane 2: Black liberties (normalized)
        # Plane 3: White liberties (normalized)
        for x, y, lib_count in liberty:
            # Convert from 1-indexed to 0-indexed
            x_idx = x - 1
            y_idx = y - 1

            if not (0 <= x_idx < self.board_size and 0 <= y_idx < self.board_size):
                continue

            if lib_count > 0:  # Black stone
                features[0, y_idx, x_idx] = 1.0  # Black stone presence
                features[2, y_idx, x_idx] = min(lib_count / 8.0, 1.0)  # Normalized liberties
            else:  # White stone (lib_count < 0)
                features[1, y_idx, x_idx] = 1.0  # White stone presence
                features[3, y_idx, x_idx] = min(abs(lib_count) / 8.0, 1.0)  # Normalized liberties

        # Plane 4: Forbidden points
        for x, y in forbidden:
            x_idx = x - 1
            y_idx = y - 1
            if 0 <= x_idx < self.board_size and 0 <= y_idx < self.board_size:
                features[4, y_idx, x_idx] = 1.0

        # Plane 5-6: Current player indicators (全盤填充)
        next_move = metadata.get("next_move", "black")
        if next_move == "black":
            features[5, :, :] = 1.0  # Black to play
        else:
            features[6, :, :] = 1.0  # White to play

        return features

    def encode_batch(
        self,
        batch: List[Tuple[List[Tuple[int, int, int]], List[Tuple[int, int]], Dict[str, Any]]]
    ) -> np.ndarray:
        """Encode a batch of positions.

        Parameters
        ----------
        batch : List[Tuple[liberty, forbidden, metadata]]
            List of position tuples

        Returns
        -------
        np.ndarray
            Batch of features with shape (batch_size, num_planes, board_size, board_size)
        """
        return np.stack([
            self.encode(liberty, forbidden, metadata)
            for liberty, forbidden, metadata in batch
        ])

    @classmethod
    def from_sgf_parse(cls, parsed_sgf: Tuple[Any, Dict[str, Any], Any]) -> np.ndarray:
        """Create features directly from parsed SGF output.

        Parameters
        ----------
        parsed_sgf : Tuple[matrix, metadata, board]
            Output from input.sgf_to_input.parse_sgf()
            Note: The matrix format needs to be converted to liberty format

        Returns
        -------
        np.ndarray
            Feature tensor ready for neural network
        """
        from core.liberty import count_liberties

        matrix, metadata, _ = parsed_sgf

        # Convert matrix to liberty format
        liberty = count_liberties(matrix)

        # Compute forbidden points (simplified - would need full board state)
        forbidden = []  # TODO: Extract from metadata if available

        encoder = cls(board_size=len(matrix))
        return encoder.encode(liberty, forbidden, metadata)

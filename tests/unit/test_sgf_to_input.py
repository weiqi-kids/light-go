"""Unit tests for SGF to input conversion (input/sgf_to_input.py).

Tests SGF parsing and conversion to model input format
using real implementation.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from input import sgf_to_input


# ---------------------------------------------------------------------------
# Test Classes
# ---------------------------------------------------------------------------

class TestSGFConversion:
    """Tests for SGF file conversion."""

    def test_convert_from_file(self, tmp_path: Path):
        """Convert SGF file to input format."""
        sgf_content = "(;FF[4]SZ[5];B[aa];W[bb];B[cc];W[dd])"
        sgf_file = tmp_path / "game.sgf"
        sgf_file.write_text(sgf_content)

        result = sgf_to_input.convert(str(sgf_file), step=4)

        assert set(result.keys()) == {"liberty", "forbidden", "metadata"}
        metadata = result["metadata"]
        assert metadata["rules"]["board_size"] == 5
        assert metadata["next_move"] == "black"

        liberty = result["liberty"]
        assert (0, 0, 2) in liberty  # corner stone liberties (0-based)
        assert len(liberty) == 4

    def test_convert_from_string(self):
        """Convert SGF string to input format."""
        sgf_content = "(;FF[4]SZ[5];B[aa];W[bb];B[cc];W[dd])"

        result = sgf_to_input.convert_from_string(sgf_content, step=4)

        assert set(result.keys()) == {"liberty", "forbidden", "metadata"}
        assert result["metadata"]["next_move"] == "black"

    @pytest.mark.parametrize("size", [5, 9, 13, 19], ids=["5x5", "9x9", "13x13", "19x19"])
    def test_various_board_sizes(self, tmp_path: Path, size: int):
        """Convert SGF with different board sizes."""
        sgf_content = f"(;FF[4]SZ[{size}];B[aa])"
        sgf_file = tmp_path / "game.sgf"
        sgf_file.write_text(sgf_content)

        result = sgf_to_input.convert(str(sgf_file), step=1)

        assert result["metadata"]["rules"]["board_size"] == size

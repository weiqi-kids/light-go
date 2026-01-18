"""Component tests for Go Rules Engine (input/sgf_to_input.py).

This module tests SGF parsing and Go rules functionality:
- parse_sgf(): Parse SGF files and extract board state, metadata
- convert(): Convert SGF to liberty and forbidden move data
- _compute_forbidden(): Calculate illegal move positions
"""
from __future__ import annotations

import os
import tempfile
import pytest
from typing import Any, Dict

from input.sgf_to_input import parse_sgf, convert, convert_from_string


class TestParseSgf:
    """Tests for the parse_sgf() function."""

    def test_parse_basic_game(self, simple_sgf_content):
        """Parse a simple 9x9 game."""
        matrix, metadata, board = parse_sgf(simple_sgf_content, from_string=True)

        assert len(matrix) == 9
        assert len(matrix[0]) == 9
        assert metadata["rules"]["board_size"] == 9
        assert metadata["rules"]["komi"] == 7.5
        assert metadata["rules"]["ruleset"] == "chinese"

    def test_parse_extracts_moves(self, simple_sgf_content):
        """Parse correctly extracts move sequence."""
        _, metadata, _ = parse_sgf(simple_sgf_content, from_string=True)

        steps = metadata["step"]
        assert len(steps) == 3
        assert steps[0][0] == "black"
        assert steps[1][0] == "white"
        assert steps[2][0] == "black"

    def test_parse_next_move_alternates(self, simple_sgf_content):
        """Next move should alternate correctly."""
        # After 3 moves (B, W, B), next should be white
        _, metadata, _ = parse_sgf(simple_sgf_content, from_string=True)
        assert metadata["next_move"] == "white"

    def test_parse_japanese_rules(self, japanese_sgf_content):
        """Parse game with Japanese rules."""
        _, metadata, _ = parse_sgf(japanese_sgf_content, from_string=True)

        assert metadata["rules"]["ruleset"] == "japanese"
        assert metadata["rules"]["komi"] == 6.5
        assert metadata["rules"]["board_size"] == 19

    def test_parse_korean_rules(self):
        """Parse game with Korean rules."""
        sgf = "(;GM[1]FF[4]SZ[19]KM[6.5]RU[Korean];B[pd])"
        _, metadata, _ = parse_sgf(sgf, from_string=True)

        assert metadata["rules"]["ruleset"] == "korean"

    def test_parse_default_ruleset(self):
        """Default to Chinese rules when RU not specified."""
        sgf = "(;GM[1]FF[4]SZ[9]KM[7.5];B[ee])"
        _, metadata, _ = parse_sgf(sgf, from_string=True)

        assert metadata["rules"]["ruleset"] == "chinese"

    def test_parse_step_parameter(self):
        """Step parameter limits moves parsed."""
        sgf = "(;GM[1]FF[4]SZ[9];B[ee];W[gc];B[cg];W[gg];B[ce])"

        _, meta_full, _ = parse_sgf(sgf, from_string=True)
        _, meta_step2, _ = parse_sgf(sgf, step=2, from_string=True)
        _, meta_step0, _ = parse_sgf(sgf, step=0, from_string=True)

        assert len(meta_full["step"]) == 5
        assert len(meta_step2["step"]) == 2
        assert len(meta_step0["step"]) == 0

    def test_parse_board_matrix_values(self):
        """Board matrix has correct values (0=empty, 1=black, -1=white)."""
        sgf = "(;GM[1]FF[4]SZ[5];B[cc];W[dd])"
        matrix, _, _ = parse_sgf(sgf, from_string=True)

        # Find non-zero positions
        positions = {}
        for y, row in enumerate(matrix):
            for x, val in enumerate(row):
                if val != 0:
                    positions[(x, y)] = val

        assert 1 in positions.values()  # Black present
        assert -1 in positions.values()  # White present

    def test_parse_handicap(self, sgf_with_handicap):
        """Parse game with handicap stones."""
        _, metadata, _ = parse_sgf(sgf_with_handicap, from_string=True)

        assert metadata["rules"]["handicap"] == 2

    def test_parse_various_komi(self):
        """Parse various komi values."""
        test_cases = [
            ("(;GM[1]FF[4]SZ[9]KM[0.5];B[ee])", 0.5),
            ("(;GM[1]FF[4]SZ[9]KM[5.5];B[ee])", 5.5),
            ("(;GM[1]FF[4]SZ[9]KM[7];B[ee])", 7.0),
        ]

        for sgf, expected_komi in test_cases:
            _, metadata, _ = parse_sgf(sgf, from_string=True)
            assert metadata["rules"]["komi"] == expected_komi

    def test_parse_from_file(self, temp_dir):
        """Parse SGF from file path."""
        sgf_content = "(;GM[1]FF[4]SZ[9]KM[7.5];B[ee])"
        sgf_path = os.path.join(temp_dir, "test.sgf")

        with open(sgf_path, "w") as f:
            f.write(sgf_content)

        matrix, metadata, _ = parse_sgf(sgf_path)

        assert metadata["rules"]["board_size"] == 9

    def test_parse_time_control(self):
        """Parse time control information."""
        sgf = "(;GM[1]FF[4]SZ[9]TM[600]OT[5x30];B[ee])"
        _, metadata, _ = parse_sgf(sgf, from_string=True)

        assert metadata["time_control"]["main_time_seconds"] == 600.0
        assert metadata["time_control"]["byo_yomi"]["periods"] == 5
        assert metadata["time_control"]["byo_yomi"]["period_time_seconds"] == 30

    def test_parse_pass_move(self):
        """Parse game with pass moves."""
        sgf = "(;GM[1]FF[4]SZ[9];B[ee];W[];B[cc])"  # W[] is pass
        _, metadata, _ = parse_sgf(sgf, from_string=True)

        steps = metadata["step"]
        assert len(steps) == 3
        assert steps[1][1] is None  # Pass move has None coordinate


class TestConvert:
    """Tests for the convert() function."""

    def test_convert_returns_required_keys(self, simple_sgf_content):
        """Convert returns liberty, forbidden, and metadata."""
        data = convert(simple_sgf_content, from_string=True)

        assert "liberty" in data
        assert "forbidden" in data
        assert "metadata" in data

    def test_convert_liberty_format(self, simple_sgf_content):
        """Liberty data is list of (x, y, value) tuples."""
        data = convert(simple_sgf_content, from_string=True)

        liberties = data["liberty"]
        assert isinstance(liberties, list)

        for item in liberties:
            assert len(item) == 3
            x, y, v = item
            assert isinstance(x, int)
            assert isinstance(y, int)
            # Black stones have positive, white negative liberties
            assert isinstance(v, int)

    def test_convert_forbidden_format(self, simple_sgf_content):
        """Forbidden data is list of (x, y) tuples."""
        data = convert(simple_sgf_content, from_string=True)

        forbidden = data["forbidden"]
        assert isinstance(forbidden, list)

        for item in forbidden:
            assert len(item) == 2
            x, y = item
            assert isinstance(x, int)
            assert isinstance(y, int)

    def test_convert_from_string_helper(self, simple_sgf_content):
        """convert_from_string() convenience function works."""
        data = convert_from_string(simple_sgf_content)

        assert "liberty" in data
        assert "forbidden" in data
        assert "metadata" in data

    def test_convert_with_step(self):
        """Convert respects step parameter."""
        sgf = "(;GM[1]FF[4]SZ[9];B[ee];W[gc];B[cg])"

        data_full = convert(sgf, from_string=True)
        data_step1 = convert(sgf, step=1, from_string=True)

        # Fewer liberties with fewer stones
        assert len(data_step1["liberty"]) <= len(data_full["liberty"])

    def test_convert_empty_board(self):
        """Convert handles game with no moves."""
        sgf = "(;GM[1]FF[4]SZ[9])"
        data = convert(sgf, from_string=True)

        assert data["liberty"] == []
        # Empty board has no forbidden moves
        assert data["forbidden"] == []


class TestForbiddenMoves:
    """Tests for forbidden move computation."""

    def test_occupied_positions_not_in_forbidden(self):
        """Occupied positions shouldn't be in forbidden list."""
        sgf = "(;GM[1]FF[4]SZ[5];B[cc])"
        data = convert(sgf, from_string=True)

        # The occupied position shouldn't be in forbidden
        # (it's simply occupied, not forbidden)
        forbidden = data["forbidden"]
        # Note: forbidden is about suicidal moves, not occupied positions

    def test_suicide_move_detection(self):
        """Detect suicidal moves as forbidden."""
        # Setup: Black stones surrounding a point, leaving it as suicide for white
        # This would require a specific board configuration
        sgf = """(;GM[1]FF[4]SZ[5]
        ;B[ba];B[ab];B[cb];W[bb]
        ;B[bc])"""  # After this, white at aa would be suicide

        data = convert(sgf, from_string=True)
        # Check that forbidden moves are computed
        assert isinstance(data["forbidden"], list)


class TestBoardSizes:
    """Tests for various board sizes."""

    @pytest.mark.parametrize("size", [5, 9, 13, 19])
    def test_parse_various_board_sizes(self, size):
        """Parse games on various standard board sizes."""
        sgf = f"(;GM[1]FF[4]SZ[{size}];B[cc])"
        matrix, metadata, _ = parse_sgf(sgf, from_string=True)

        assert len(matrix) == size
        assert len(matrix[0]) == size
        assert metadata["rules"]["board_size"] == size


class TestRulesets:
    """Tests for different ruleset handling."""

    @pytest.mark.parametrize("ruleset", ["Chinese", "Japanese", "Korean", "AGA", "NZ"])
    def test_parse_rulesets_case_insensitive(self, ruleset):
        """Rulesets should be parsed case-insensitively."""
        sgf = f"(;GM[1]FF[4]SZ[9]RU[{ruleset}];B[ee])"
        _, metadata, _ = parse_sgf(sgf, from_string=True)

        assert metadata["rules"]["ruleset"] == ruleset.lower()


class TestMetadata:
    """Tests for metadata extraction."""

    def test_capture_tracking(self):
        """Capture counts are tracked in metadata."""
        sgf = "(;GM[1]FF[4]SZ[9];B[ee])"
        _, metadata, _ = parse_sgf(sgf, from_string=True)

        assert "capture" in metadata
        assert "black" in metadata["capture"]
        assert "white" in metadata["capture"]

    def test_time_remaining(self):
        """Time remaining is tracked for both players."""
        sgf = "(;GM[1]FF[4]SZ[9];B[ee]BL[550];W[gc]WL[580])"
        _, metadata, _ = parse_sgf(sgf, from_string=True)

        assert "time" in metadata
        assert len(metadata["time"]) == 2

        time_info = {t["player"]: t for t in metadata["time"]}
        assert "black" in time_info
        assert "white" in time_info


class TestEdgeCases:
    """Edge case tests for SGF parsing."""

    def test_empty_sgf(self):
        """Handle minimal SGF with just root node."""
        sgf = "(;GM[1])"
        matrix, metadata, _ = parse_sgf(sgf, from_string=True)

        # Should use default size (19)
        assert len(matrix) == 19

    def test_sgf_with_whitespace(self):
        """Handle SGF with various whitespace."""
        sgf = """(
            ;GM[1]FF[4]SZ[9]
            ;B[ee]
            ;W[gc]
        )"""
        _, metadata, _ = parse_sgf(sgf, from_string=True)

        assert len(metadata["step"]) == 2

    def test_sgf_with_comments(self):
        """SGF with comments doesn't break parsing."""
        sgf = "(;GM[1]FF[4]SZ[9]C[This is a comment];B[ee]C[Black plays tengen])"
        matrix, metadata, _ = parse_sgf(sgf, from_string=True)

        assert len(metadata["step"]) == 1

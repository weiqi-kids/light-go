"""Unit tests for KataGo input conversion (input/katago_to_input.py).

Tests coordinate conversion and file processing for KataGo format data.
"""
import json
import pathlib
import sys
from unittest.mock import patch, mock_open

import pytest

# Add project root to path
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from input.katago_to_input import katago_to_coords, process_katago_file


class TestKatagoToCoords:
    """Tests for katago_to_coords() function."""

    @pytest.mark.parametrize("move,board_size,expected", [
        ("A19", 19, (0, 0)),
        ("T1", 19, (18, 18)),
        ("K10", 19, (9, 9)),
    ], ids=["top_left", "bottom_right", "center"])
    def test_valid_coordinates(self, move, board_size, expected):
        """Valid move strings convert to correct coordinates."""
        assert katago_to_coords(move, board_size) == expected

    @pytest.mark.parametrize("move,board_size", [
        ("PASS", 19),
        ("A20", 19),     # Out of bounds
        ("Z10", 19),     # Invalid column
        ("A", 19),       # Incomplete move string
        (None, 19),      # None input
        (123, 19),       # Wrong type
    ], ids=["pass", "out_of_bounds", "invalid_col", "incomplete", "none", "wrong_type"])
    def test_invalid_coordinates_return_none(self, move, board_size):
        """Invalid move strings return None."""
        assert katago_to_coords(move, board_size) is None


class TestProcessKatagoFile:
    """Tests for process_katago_file() function."""

    def test_valid_data(self):
        """Process file with valid KataGo data."""
        mock_data = [
            json.dumps({
                "board": ["E", "E", "B", "E", "E", "E", "E", "E", "E"],
                "liberties": [0, 0, 4, 0, 0, 0, 0, 0, 0],
                "boardXSize": 3,
                "boardYSize": 3,
                "whiteCaptures": 0,
                "blackCaptures": 0,
                "pla": "B",
                "moves": ["C1"],
                "rules": {
                    "rules": "chinese",
                    "komi": 7.5,
                    "mainTime": 3600,
                    "byoYomiTime": 30,
                    "byoYomiPeriods": 3,
                },
                "bTime": 3500, "bPeriodsLeft": 3,
                "wTime": 3500, "wPeriodsLeft": 3,
                "illegalMoves": ["A1"]
            }),
            json.dumps({
                "board": ["E", "E", "B", "E", "W", "E", "E", "E", "E"],
                "liberties": [0, 0, 3, 0, 2, 0, 0, 0, 0],
                "boardXSize": 3,
                "boardYSize": 3,
                "whiteCaptures": 0,
                "blackCaptures": 0,
                "pla": "W",
                "moves": ["C1", "B2"],
                "rules": {
                    "rules": "chinese",
                    "komi": 7.5,
                    "mainTime": 3600,
                    "byoYomiTime": 30,
                    "byoYomiPeriods": 3,
                },
                "bTime": 3400, "bPeriodsLeft": 3,
                "wTime": 3400, "wPeriodsLeft": 3,
                "illegalMoves": ["A1", "C2"]
            })
        ]

        with patch('builtins.open', mock_open()) as mock_file:
            mock_file.return_value.__enter__.return_value = mock_data
            result = process_katago_file("dummy_path.jsonl")

        assert len(result) == 2

        # Test first move
        assert result[0]['liberty'] == [(2, 2, 4)]
        assert result[0]['forbidden'] == [(0, 2)]
        assert result[0]['metadata']['rules']['ruleset'] == 'chinese'
        assert result[0]['metadata']['capture']['black'] == 0
        assert result[0]['metadata']['next_move'] == 'black'
        assert result[0]['metadata']['step'] == ['C1']
        assert result[0]['metadata']['time_control']['main_time_seconds'] == 3600
        assert result[0]['metadata']['time'][0]['player'] == 'black'

        # Test second move
        assert result[1]['liberty'] == [(2, 2, 3), (1, 1, -2)]
        assert result[1]['forbidden'] == [(0, 2), (2, 1)]
        assert result[1]['metadata']['next_move'] == 'white'

    def test_empty_file(self):
        """Process empty file returns empty list."""
        with patch('builtins.open', mock_open()) as mock_file:
            mock_file.return_value.__enter__.return_value = []
            result = process_katago_file("empty.jsonl")

        assert result == []

    def test_invalid_json_lines_skipped(self):
        """Invalid JSON lines are skipped, valid ones processed."""
        mock_data = ['{"key": "value"}', "invalid json", '{"another": 1}']

        with patch('builtins.open', mock_open()) as mock_file:
            mock_file.return_value.__enter__.return_value = mock_data
            result = process_katago_file("invalid.jsonl")

        assert len(result) == 2  # Only valid JSON lines processed

    def test_missing_fields_uses_defaults(self):
        """Missing fields use default values."""
        mock_data = [
            json.dumps({
                "board": ["E", "B"],
                "liberties": [0, 1],
                "boardXSize": 2,
                "boardYSize": 1,
                "pla": "B"
            })
        ]

        with patch('builtins.open', mock_open()) as mock_file:
            mock_file.return_value.__enter__.return_value = mock_data
            result = process_katago_file("missing_fields.jsonl")

        assert len(result) == 1
        assert 'liberty' in result[0]
        assert 'forbidden' in result[0]
        assert 'metadata' in result[0]
        assert result[0]['metadata']['rules']['handicap'] == 0  # Default value
        assert result[0]['metadata']['capture']['black'] == 0   # Default value

    def test_io_error_returns_empty(self):
        """IO error returns empty list."""
        with patch('builtins.open', side_effect=IOError("File not found")):
            result = process_katago_file("non_existent.jsonl")

        assert result == []

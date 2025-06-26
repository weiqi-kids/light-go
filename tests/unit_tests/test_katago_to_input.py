import unittest
import json
import os
import sys
from unittest.mock import patch, mock_open

# Add the parent directory of input to the sys.path to allow importing katago_to_input
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from input.katago_to_input import katago_to_coords, process_katago_file

class TestKatagoToInput(unittest.TestCase):

    def test_katago_to_coords(self):
        self.assertEqual(katago_to_coords("A19", 19), (0, 0))
        self.assertEqual(katago_to_coords("T1", 19), (18, 18))
        self.assertEqual(katago_to_coords("K10", 19), (9, 9))
        self.assertIsNone(katago_to_coords("PASS", 19))
        self.assertIsNone(katago_to_coords("A20", 19)) # Out of bounds
        self.assertIsNone(katago_to_coords("Z10", 19)) # Invalid column
        self.assertIsNone(katago_to_coords("A", 19)) # Incomplete move string
        self.assertIsNone(katago_to_coords(None, 19))
        self.assertIsNone(katago_to_coords(123, 19))

    @patch('builtins.open', new_callable=mock_open)
    def test_process_katago_file_valid_data(self, mock_file_open):
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
                "rules": {"rules": "chinese", "komi": 7.5, "mainTime": 3600, "byoYomiTime": 30, "byoYomiPeriods": 3},
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
                "rules": {"rules": "chinese", "komi": 7.5, "mainTime": 3600, "byoYomiTime": 30, "byoYomiPeriods": 3},
                "bTime": 3400, "bPeriodsLeft": 3,
                "wTime": 3400, "wPeriodsLeft": 3,
                "illegalMoves": ["A1", "C2"]
            })
        ]
        mock_file_open.return_value.__enter__.return_value = mock_data

        result = process_katago_file("dummy_path.jsonl")

        self.assertEqual(len(result), 2)

        # Test first move
        self.assertEqual(result[0]['liberty'], [(2, 2, 4)])
        self.assertEqual(result[0]['forbidden'], [(0, 2)])
        self.assertEqual(result[0]['metadata']['rules']['ruleset'], 'chinese')
        self.assertEqual(result[0]['metadata']['capture']['black'], 0)
        self.assertEqual(result[0]['metadata']['next_move'], 'black')
        self.assertEqual(result[0]['metadata']['step'], ['C1'])
        self.assertEqual(result[0]['metadata']['time_control']['main_time_seconds'], 3600)
        self.assertEqual(result[0]['metadata']['time'][0]['player'], 'black')

        # Test second move
        self.assertEqual(result[1]['liberty'], [(2, 2, 3), (1, 1, -2)])
        self.assertEqual(result[1]['forbidden'], [(0, 2), (2, 1)])
        self.assertEqual(result[1]['metadata']['next_move'], 'white')

    @patch('builtins.open', new_callable=mock_open)
    def test_process_katago_file_empty(self, mock_file_open):
        mock_file_open.return_value.__enter__.return_value = []
        result = process_katago_file("empty.jsonl")
        self.assertEqual(result, [])

    @patch('builtins.open', new_callable=mock_open)
    def test_process_katago_file_invalid_json(self, mock_file_open):
        mock_file_open.return_value.__enter__.return_value = ['{"key": "value"}', "invalid json", '{"another": 1}']
        result = process_katago_file("invalid.jsonl")
        self.assertEqual(len(result), 2) # Only valid JSON lines should be processed

    @patch('builtins.open', new_callable=mock_open)
    def test_process_katago_file_missing_fields(self, mock_file_open):
        mock_data = [
            json.dumps({
                "board": ["E", "B"],
                "liberties": [0, 1],
                "boardXSize": 2,
                "boardYSize": 1,
                "pla": "B"
            })
        ]
        mock_file_open.return_value.__enter__.return_value = mock_data
        result = process_katago_file("missing_fields.jsonl")
        self.assertEqual(len(result), 1)
        self.assertIn('liberty', result[0])
        self.assertIn('forbidden', result[0])
        self.assertIn('metadata', result[0])
        self.assertEqual(result[0]['metadata']['rules']['handicap'], 0) # Default value
        self.assertEqual(result[0]['metadata']['capture']['black'], 0) # Default value

    @patch('builtins.open', new_callable=mock_open)
    def test_process_katago_file_io_error(self, mock_file_open):
        mock_file_open.side_effect = IOError("File not found")
        result = process_katago_file("non_existent.jsonl")
        self.assertEqual(result, [])

if __name__ == '__main__':
    unittest.main()

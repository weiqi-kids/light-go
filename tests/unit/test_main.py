"""Unit tests for CLI entry point (main.py).

Tests command-line argument parsing and mode execution.

Note: main.py is loaded via importlib since it's a CLI entry point,
not a standard module.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# Load main.py as a module using importlib
ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("cli_main", ROOT / "main.py")
cli_main = importlib.util.module_from_spec(spec)
sys.modules["cli_main"] = cli_main
spec.loader.exec_module(cli_main)


# ---------------------------------------------------------------------------
# Test Classes
# ---------------------------------------------------------------------------

class TestTrainMode:
    """Tests for train mode."""

    def test_train_sgf_calls_engine_train(self, tmp_path: Path, capsys):
        """Train mode with SGF data calls engine.train()."""
        argv = [
            'main.py',
            '--mode', 'train',
            '--data', 'data_dir',
            '--output', str(tmp_path)
        ]
        engine_mock = MagicMock()
        engine_mock.strategy_manager.strategies_path = str(tmp_path)
        engine_mock.train.return_value = 's1'

        with patch.object(sys, 'argv', argv), \
             patch('cli_main.detect_data_type', return_value='sgf'), \
             patch('cli_main.Engine', return_value=engine_mock) as engine_cls:
            cli_main.main()

            engine_cls.assert_called_with(str(tmp_path))
            engine_mock.train.assert_called_with('data_dir', str(tmp_path))
            assert capsys.readouterr().out.strip().endswith('s1.pkl')

    def test_train_npz_calls_engine(self, tmp_path: Path, capsys):
        """Train mode with NPZ data calls engine.train_npz_directory()."""
        argv = [
            'main.py', '--mode', 'train',
            '--data', str(tmp_path),
            '--output', str(tmp_path / 'out')
        ]
        (tmp_path / 'out').mkdir()
        np.savez(tmp_path / 'd.npz', data=['{}'])
        engine_mock = MagicMock()
        engine_mock.strategy_manager.strategies_path = str(tmp_path)
        engine_mock.train_npz_directory.return_value = 's1'

        with patch.object(sys, 'argv', argv), \
             patch('cli_main.Engine', return_value=engine_mock) as engine_cls:
            cli_main.main()

            engine_cls.assert_called_with(str(tmp_path / 'out'))
            engine_mock.train_npz_directory.assert_called_with(
                str(tmp_path), str(tmp_path / 'out')
            )
            assert capsys.readouterr().out.strip().endswith('s1.pkl')


class TestEvaluateMode:
    """Tests for evaluate mode."""

    def test_calls_auto_learner_train(self, tmp_path: Path, capsys):
        """Evaluate mode calls auto_learner.train()."""
        argv = [
            'main.py',
            '--mode', 'evaluate',
            '--data', 'eval_dir',
            '--output', str(tmp_path)
        ]
        engine_mock = MagicMock()
        engine_mock.strategy_manager.strategies_path = str(tmp_path)
        engine_mock.auto_learner.train.return_value = {'score': 1}

        with patch.object(sys, 'argv', argv), \
             patch('cli_main.Engine', return_value=engine_mock):
            cli_main.main()

            engine_mock.auto_learner.train.assert_called_with('eval_dir')
            assert 'score' in capsys.readouterr().out


class TestPlayMode:
    """Tests for play mode."""

    def test_calls_decide_move(self, tmp_path: Path, capsys):
        """Play mode calls engine.decide_move()."""
        argv = [
            'main.py',
            '--mode', 'play',
            '--data', 'game.sgf',
            '--output', str(tmp_path)
        ]
        engine_mock = MagicMock()
        engine_mock.strategy_manager.strategies_path = str(tmp_path)
        engine_mock.decide_move.return_value = (3, 3)

        with patch.object(sys, 'argv', argv), \
             patch('cli_main.Engine', return_value=engine_mock), \
             patch('input.sgf_to_input.parse_sgf',
                   return_value=([[0]], {'next_move': 'black'}, None)) as parser:
            cli_main.main()

            parser.assert_called_with('game.sgf')
            engine_mock.decide_move.assert_called_with([[0]], 'black')
            assert '(3, 3)' in capsys.readouterr().out


class TestArgumentValidation:
    """Tests for argument validation."""

    def test_missing_data_errors(self, tmp_path: Path):
        """Missing --data argument causes SystemExit."""
        argv = ['main.py', '--mode', 'train', '--output', str(tmp_path)]

        with patch.object(sys, 'argv', argv), patch('cli_main.Engine'):
            with pytest.raises(SystemExit):
                cli_main.main()

    def test_strategy_argument_loads_strategy(self, tmp_path: Path):
        """--strategy argument triggers load_strategy()."""
        argv = [
            'main.py',
            '--mode', 'train',
            '--data', 'dir',
            '--output', str(tmp_path),
            '--strategy', 'strat1'
        ]
        engine_mock = MagicMock()
        engine_mock.strategy_manager.strategies_path = str(tmp_path)
        engine_mock.train.return_value = 's1'

        with patch.object(sys, 'argv', argv), \
             patch('cli_main.detect_data_type', return_value='sgf'), \
             patch('cli_main.Engine', return_value=engine_mock):
            cli_main.main()

            engine_mock.load_strategy.assert_called_with('strat1')


class TestErrorHandling:
    """Tests for error handling."""

    def test_exception_logged(self, tmp_path: Path):
        """Runtime exceptions are logged."""
        argv = ['main.py', '--mode', 'train', '--data', 'dir', '--output', str(tmp_path)]
        engine_mock = MagicMock()
        engine_mock.strategy_manager.strategies_path = str(tmp_path)
        engine_mock.train.side_effect = RuntimeError('boom')

        with patch.object(sys, 'argv', argv), \
             patch('cli_main.detect_data_type', return_value='sgf'), \
             patch('cli_main.Engine', return_value=engine_mock), \
             patch('logging.exception') as log_exc:
            cli_main.main()

            log_exc.assert_called()
            assert 'Unhandled error' in log_exc.call_args.args[0]


class TestDetectDataType:
    """Tests for detect_data_type() function."""

    def test_detects_sgf(self, tmp_path: Path):
        """Detects SGF data type."""
        sgf_file = tmp_path / 'a.sgf'
        sgf_file.write_text('')

        assert cli_main.detect_data_type(str(tmp_path)) == 'sgf'

    def test_detects_npz(self, tmp_path: Path):
        """Detects NPZ data type."""
        npz_path = tmp_path / 'b.npz'
        np.savez(npz_path, data=['{}'])

        assert cli_main.detect_data_type(str(tmp_path)) == 'npz'

    def test_mixed_formats_raises(self, tmp_path: Path):
        """Mixed formats raise ValueError."""
        (tmp_path / 'a.sgf').write_text('')
        np.savez(tmp_path / 'b.npz', data=['{}'])

        with pytest.raises(ValueError):
            cli_main.detect_data_type(str(tmp_path))

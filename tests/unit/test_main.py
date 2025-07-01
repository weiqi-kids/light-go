import pathlib
import sys
from unittest.mock import MagicMock, patch
import numpy as np

# Put project root on sys.path so we can load the CLI module
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import importlib.util

spec = importlib.util.spec_from_file_location("cli_main", ROOT / "main.py")
cli_main = importlib.util.module_from_spec(spec)
sys.modules["cli_main"] = cli_main
spec.loader.exec_module(cli_main)

import pytest


def test_argparse_train_calls_engine_train(tmp_path, capsys):
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


def test_evaluate_calls_auto_learner(tmp_path, capsys):
    argv = [
        'main.py',
        '--mode', 'evaluate',
        '--data', 'eval_dir',
        '--output', str(tmp_path)
    ]
    engine_mock = MagicMock()
    engine_mock.strategy_manager.strategies_path = str(tmp_path)
    engine_mock.auto_learner.train.return_value = {'score': 1}
    with patch.object(sys, 'argv', argv), patch('cli_main.Engine', return_value=engine_mock):
        cli_main.main()
        engine_mock.auto_learner.train.assert_called_with('eval_dir')
        assert 'score' in capsys.readouterr().out


def test_play_calls_decide_move(tmp_path, capsys):
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
         patch('input.sgf_to_input.parse_sgf', return_value=([[0]], {'next_move': 'black'}, None)) as parser:
        cli_main.main()
        parser.assert_called_with('game.sgf')
        engine_mock.decide_move.assert_called_with([[0]], 'black')
        assert '(3, 3)' in capsys.readouterr().out


def test_missing_data_errors(tmp_path):
    argv = ['main.py', '--mode', 'train', '--output', str(tmp_path)]
    with patch.object(sys, 'argv', argv), patch('cli_main.Engine'):
        with pytest.raises(SystemExit):
            cli_main.main()


def test_exception_logged(tmp_path):
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


def test_strategy_argument_loads_strategy(tmp_path):
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


def test_detect_data_type(tmp_path):
    sgf_file = tmp_path / 'a.sgf'
    sgf_file.write_text('')
    assert cli_main.detect_data_type(str(tmp_path)) == 'sgf'
    sgf_file.unlink()
    npz_path = tmp_path / 'b.npz'
    np.savez(npz_path, data=['{}'])
    assert cli_main.detect_data_type(str(tmp_path)) == 'npz'
    (tmp_path / 'c.sgf').write_text('')
    with pytest.raises(ValueError):
        cli_main.detect_data_type(str(tmp_path))


def test_argparse_train_npz_calls_engine(tmp_path, capsys):
    argv = [
        'main.py', '--mode', 'train', '--data', str(tmp_path), '--output', str(tmp_path / 'out')
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
        engine_mock.train_npz_directory.assert_called_with(str(tmp_path), str(tmp_path / 'out'))
        assert capsys.readouterr().out.strip().endswith('s1.pkl')

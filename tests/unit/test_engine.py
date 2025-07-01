import os
import pathlib
import sys
from unittest.mock import Mock, patch

# Add project root to path
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from core.engine import Engine


def _create_engine(tmp_path: pathlib.Path):
    """Create an Engine with mocks for its dependencies."""
    sm_patch = patch('core.engine.StrategyManager')
    al_patch = patch('core.engine.AutoLearner')
    mkdir_patch = patch('core.engine.os.makedirs')
    with sm_patch as MockSM, al_patch as MockAL, mkdir_patch:
        engine = Engine(str(tmp_path))
        sm_inst = MockSM.return_value
        al_inst = MockAL.return_value
    return engine, sm_inst, al_inst, MockSM, MockAL


def test_engine_initialization(tmp_path: pathlib.Path):
    engine, sm_inst, al_inst, MockSM, MockAL = _create_engine(tmp_path)

    MockSM.assert_called_with(os.path.join(str(tmp_path), 'strategies'))
    MockAL.assert_called_with(sm_inst)
    assert engine.strategy_manager is sm_inst
    assert engine.auto_learner is al_inst


def test_engine_train(tmp_path: pathlib.Path):
    engine, sm_inst, al_inst, _, _ = _create_engine(tmp_path)
    data_dir = tmp_path / 'data'
    out_dir = tmp_path / 'out'
    data_dir.mkdir()
    out_dir.mkdir()

    with patch.object(Engine, '_convert_directory', return_value=['d1']) as conv, \
         patch('core.engine.GoAIModel') as MockModel:
        model_inst = Mock()
        MockModel.return_value = model_inst
        al_inst.train_and_save.return_value = 's1'

        name = engine.train(str(data_dir), str(out_dir))

        conv.assert_called_once_with(str(data_dir))
        al_inst.train_and_save.assert_called_once_with(str(data_dir))
        model_inst.train.assert_called_once_with(['d1'])
        model_inst.save_pretrained.assert_called_once_with(os.path.join(str(out_dir), 's1.pt'))
        assert name == 's1'


def test_engine_train_npz(tmp_path: pathlib.Path):
    engine, sm_inst, al_inst, _, _ = _create_engine(tmp_path)
    data_dir = tmp_path / 'data'
    out_dir = tmp_path / 'out'
    data_dir.mkdir()
    out_dir.mkdir()
    npz_file = data_dir / 'd.npz'
    npz_file.touch()

    with patch('numpy.load') as np_load, \
         patch('core.engine.GoAIModel') as MockModel, \
         patch('input.katago_to_input.process_katago_lines', return_value=['d1']) as proc:
        class Dummy:
            files = ['arr']
            def __getitem__(self, k):
                return ['{}']
        np_load.return_value = Dummy()
        model_inst = Mock()
        MockModel.return_value = model_inst
        al_inst.train_and_save.return_value = 's1'

        name = engine.train_npz_directory(str(data_dir), str(out_dir))

        proc.assert_called()
        model_inst.train.assert_called_with(['d1'])
        model_inst.save_pretrained.assert_called_once_with(os.path.join(str(out_dir), 's1.pt'))
        assert name == 's1'


def test_engine_evaluate(tmp_path: pathlib.Path):
    engine, sm_inst, al_inst, _, _ = _create_engine(tmp_path)
    data_dir = tmp_path / 'data'
    out_dir = tmp_path / 'out'
    data_dir.mkdir()
    out_dir.mkdir()

    sm_inst.list_strategies.return_value = ['a', 'b']
    with patch.object(Engine, '_convert_directory', return_value=['c1']) as conv, \
         patch.object(Engine, '_fuse_predictions', return_value=['fused']) as fuse, \
         patch('core.engine.GoAIModel') as MockModel:
        model_inst = Mock()
        MockModel.from_pretrained.return_value = model_inst
        model_inst.predict.return_value = 'p'

        metrics = engine.evaluate(str(data_dir), str(out_dir))

        conv.assert_called_once_with(str(data_dir))
        assert MockModel.from_pretrained.call_count == 2
        model_inst.predict.assert_called_with('c1')
        fuse.assert_called_once()
        al_inst.train.assert_called_once_with(str(data_dir))
        assert metrics == {'samples': 1, 'strategies_tested': 2}


def test_engine_play_latest(tmp_path: pathlib.Path):
    engine, sm_inst, _, _, _ = _create_engine(tmp_path)
    data_dir = tmp_path / 'data'
    out_dir = tmp_path / 'out'
    data_dir.mkdir()
    out_dir.mkdir()

    sm_inst.list_strategies.return_value = ['a', 'b']
    with patch.object(Engine, '_convert_directory', return_value=['c1']) as conv, \
         patch('core.engine.GoAIModel') as MockModel:
        model_inst = Mock()
        MockModel.from_pretrained.return_value = model_inst
        model_inst.predict.return_value = 'r'

        result = engine.play(str(data_dir), str(out_dir))

        conv.assert_called_once_with(str(data_dir))
        MockModel.from_pretrained.assert_called_once_with(os.path.join(str(out_dir), 'b.pt'))
        model_inst.predict.assert_called_once_with('c1')
        assert result == ['r']


def test_engine_play_specific_strategy(tmp_path: pathlib.Path):
    engine, sm_inst, _, _, _ = _create_engine(tmp_path)
    data_dir = tmp_path / 'data'
    out_dir = tmp_path / 'out'
    data_dir.mkdir()
    out_dir.mkdir()

    sm_inst.list_strategies.return_value = ['a', 'b']
    with patch.object(Engine, '_convert_directory', return_value=['c1']) as conv, \
         patch('core.engine.GoAIModel') as MockModel:
        model_inst = Mock()
        MockModel.from_pretrained.return_value = model_inst
        model_inst.predict.return_value = 'r'

        result = engine.play(str(data_dir), str(out_dir), strategy='a')

        conv.assert_called_once_with(str(data_dir))
        MockModel.from_pretrained.assert_called_once_with(os.path.join(str(out_dir), 'a.pt'))
        model_inst.predict.assert_called_once_with('c1')
        assert result == ['r']


def test_predict_lazy_engine(tmp_path: pathlib.Path, monkeypatch):
    import core.engine as engine_mod
    monkeypatch.setattr(engine_mod, "_engine_instance", None, raising=False)
    engine_inst = Mock()
    engine_inst.decide_move.return_value = (0, 0)
    engine_cls = Mock(return_value=engine_inst)
    monkeypatch.setattr(engine_mod, "Engine", engine_cls)
    monkeypatch.setenv("LIGHTGO_MODEL_DIR", str(tmp_path))

    result1 = engine_mod.predict({"board": [[0]], "color": "black"})
    result2 = engine_mod.predict({"board": [[0]], "color": "black"})

    engine_cls.assert_called_once_with(str(tmp_path))
    assert result1 == (0, 0)
    assert result2 == (0, 0)


def test_predict_empty_move_list(monkeypatch):
    import core.engine as engine_mod
    engine_inst = Mock()
    engine_mod._engine_instance = engine_inst
    engine_inst.decide_move.return_value = (1, 2)

    result = engine_mod.predict({"board": [], "color": "black", "size": 5})

    board_arg, color_arg = engine_inst.decide_move.call_args[0]
    assert len(board_arg) == 5
    assert len(board_arg[0]) == 5
    assert color_arg == "black"
    assert result == (1, 2)

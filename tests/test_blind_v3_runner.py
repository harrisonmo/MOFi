"""Synthetic end-to-end execution tests with injected CLI and GPU adapters."""
import json
import pytest
from rsi import blind_v3_runner as runner
from rsi import blind_campaign_v3 as core
from rsi.blind_g1 import write, read


@pytest.fixture
def campaign(tmp_path, monkeypatch):
    root = tmp_path / 'run'
    root.mkdir()
    package = tmp_path / 'package'
    public = package / 'public'
    public.mkdir(parents=True)
    (public / 'a.npz').write_text('synthetic')
    (public / 'a.cif').write_text('synthetic')
    (public / 'TASK.md').write_text('Synthetic task')
    write(public / 'train.json', [dict(id='a', graph='a.npz', cif='a.cif', target=2)])
    write(package / 'private/development.json', [dict(id='dev', target=2)])
    state = core.initialize({'train.py': 'pass'})
    state.update(package=str(package), image='sha256:' + 'a'*64, model_calls=0,
                 call_pending=False, stopped=False, confirmation=None,
                 metrics={'mean_mae': 2, 'seeds': {'17': {'mae': 2}, '29': {'mae': 2}}})
    state['policy']['max_prompt_bytes'] = 2_000_000
    runner.save(root, state)
    monkeypatch.setattr(runner, 'load', lambda root: read(root / 'state.json'))
    return root


def fake_gpu(public, candidate, image, seed, cap, epochs=50):
    if epochs == 1:
        assert not (public / 'private').exists()
        assert 'target' not in read(public / 'validation.json')[0]
        return dict(status='completed', predictions={'a': 2},
                    metadata={'diagnostic_features': {'a': [1, 2]}})
    return dict(status='completed', predictions={'dev': 2.1})


def test_step_promotes_and_confirms(campaign):
    def invoke(prompt, directory, timeout):
        assert 'diagnostic_features' in prompt
        return {'passed_transport': True}, json.dumps(dict(action='evaluate', base='g1',
            hypothesis='Synthetic improvement', files={'train.py': 'x = 1'}))
    runner.step(campaign, invoke, fake_gpu)
    state = runner.load(campaign)
    assert state['incumbent'] == 'action-1'
    assert state['diagnostics_used'] == state['candidates_used'] == state['model_calls'] == 1
    runner.confirm(campaign, fake_gpu)
    assert len(runner.load(campaign)['confirmation']['fits']) == 6
    with pytest.raises(ValueError):
        runner.confirm(campaign, fake_gpu)


def test_bad_features_prevent_training(campaign):
    calls = []
    def gpu(*args, **kwargs):
        calls.append(kwargs)
        return dict(status='completed', predictions={'a': 2}, metadata={})
    state = runner.load(campaign)
    runner.run_action(campaign, state, dict(action='evaluate', base='g1', hypothesis='Broken'), gpu)
    assert len(calls) == 1
    assert state['history'][-1]['status'] == 'failed'
    assert state['incumbent'] == 'g1'
    assert state['candidates_used'] == 1


def test_transport_failure_consumes_call(campaign):
    with pytest.raises(ValueError):
        runner.step(campaign, lambda *a: ({'passed_transport': False}, '{}'), fake_gpu)
    state = runner.load(campaign)
    assert state['model_calls'] == 1 and not state['call_pending']


def test_crash_reservation_persists(campaign):
    def crash(*args, **kwargs):
        raise KeyboardInterrupt()
    state = runner.load(campaign)
    with pytest.raises(KeyboardInterrupt):
        runner.run_action(campaign, state, dict(action='repair', base='g1', hypothesis='Attempt'), crash)
    saved = runner.load(campaign)
    assert saved['pending'] == 'action-1'
    assert saved['diagnostics_used'] == 1

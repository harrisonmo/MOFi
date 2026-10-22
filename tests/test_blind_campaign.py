"""Synthetic campaign tests: no model calls or scientific training."""
import json
from pathlib import Path
import pytest
from rsi.blind_g1 import read, sha, write
from rsi import blind_campaign as campaign

@pytest.fixture
def prepared(tmp_path):
    package = tmp_path / 'package'
    write(package / 'public/train.json', [{'id': 'a', 'graph': 'a.npz', 'cif': 'a.cif', 'target': 1}])
    write(package / 'public/validation.json', [{'id': 'b', 'graph': 'b.npz', 'cif': 'b.cif'}])
    (package / 'public/TASK.md').write_text('Synthetic task')
    write(package / 'private/development.json', [{'id': 'b', 'target': 2, 'family': 'f'}])
    write(package / 'manifest.json', {'public_files': {p.name: sha(p) for p in (package / 'public').iterdir()},
          'development_sha256': sha(package / 'private/development.json')})
    calibration = tmp_path / 'calibration'
    (calibration / 'parent').mkdir(parents=True)
    (calibration / 'parent/train.py').write_text('print("parent")')
    write(calibration / 'report.json', {'passed': True, 'package_sha256': sha(package / 'manifest.json'),
          'parent_files': {'train.py': sha(calibration / 'parent/train.py')}, 'image_id': 'sha256:' + 'a'*64,
          'fits': [{'seed': s, 'mae': 1, 'rmse': 1} for s in (17, 29)], 'fit_wall_seconds': 180})
    root = tmp_path / 'campaign'
    probe = tmp_path / 'probe.json'
    write(probe, {'passed': True, 'image_id': 'sha256:' + 'a'*64, 'package_sha256': sha(package / 'manifest.json')})
    campaign.initialize(package, calibration, root, probe)
    for provider in campaign.PROVIDERS:
        campaign.register(root, provider, 'synthetic-model', 'app-file-exchange')
    return root


def response(packet, path, code='print("candidate")'):
    value = {'packet_id': packet['packet_id'], 'hypothesis': 'Synthetic change', 'files': {'train.py': code},
             'session': {'fresh_context': True, 'tools_disabled': True, 'external_materials_used': False,
                         'model': 'synthetic-model', 'usage': None}}
    write(path, value)
    return path


def test_feedback_inheritance_and_provider_separation(prepared, monkeypatch):
    calls = []
    def fake(*args):
        calls.append(args)
        return {'status': 'completed', 'predictions': {'b': 2.1}}
    monkeypatch.setattr(campaign, 'run_container', fake)
    first = campaign.packet(prepared, 'chatgpt', 'rsi')
    result = campaign.submit(prepared, 'chatgpt', 'rsi', response(first, prepared / 'response.json'))
    assert result['accepted']
    second = campaign.packet(prepared, 'chatgpt', 'rsi', reserve=False)
    assert second['source_files']['train.py'] == 'print("candidate")'
    assert len(second['history']) == 1
    assert campaign.packet(prepared, 'claude', 'rsi', reserve=False)['history'] == []
    independent = campaign.packet(prepared, 'chatgpt', 'independent', reserve=False)
    assert independent['history'] == []
    assert independent['source_files']['train.py'] == 'print("parent")'
    assert len(calls) == 2


def test_failure_consumes_slot(prepared):
    first = campaign.packet(prepared, 'chatgpt', 'rsi')
    with pytest.raises(ValueError):
        campaign.packet(prepared, 'chatgpt', 'rsi')
    path = response(first, prepared / 'response.json')
    value = read(path)
    value['files'] = {'../evil.py': 'pass', 'train.py': 'pass'}
    write(path, value)
    assert campaign.submit(prepared, 'chatgpt', 'rsi', path)['status'] == 'failed'
    with pytest.raises(ValueError):
        campaign.submit(prepared, 'chatgpt', 'rsi', path)
    assert campaign.packet(prepared, 'chatgpt', 'rsi', reserve=False)['history'][0]['status'] == 'failed'


def test_budget_and_deadline(prepared, monkeypatch):
    for _ in range(3):
        value = campaign.packet(prepared, 'claude', 'independent')
        path = response(value, prepared / 'response.json')
        ledger = read(prepared / 'ledger.json')
        ledger['providers']['claude']['arms']['independent']['slots'][-1]['started_at'] = 0
        write(prepared / 'ledger.json', ledger)
        assert campaign.submit(prepared, 'claude', 'independent', path)['status'] == 'failed'
    with pytest.raises(ValueError):
        campaign.packet(prepared, 'claude', 'independent')


def test_policy_tampering(prepared):
    policy = read(prepared / 'policy.json')
    policy['fit_wall_seconds'] *= 100
    write(prepared / 'policy.json', policy)
    with pytest.raises(ValueError, match='binding'):
        campaign.load(prepared)


@pytest.mark.parametrize('files', [{'train.py': 'bad syntax ***'}, {'train.py': 'pass', '/etc/x.py': 'pass'},
                                  {'train.py': '#' * 128001}])
def test_bad_sources(files):
    with pytest.raises((ValueError, SyntaxError)):
        campaign.validate_files(files)


def test_acceptance_guard():
    parent = {'mean_mae': 1, 'seeds': {'17': {'mae': 1}, '29': {'mae': 1}}}
    assert not campaign.accepted({'mean_mae': .9, 'seeds': {'17': {'mae': .6}, '29': {'mae': 1.2}}}, parent)

def test_complete_campaign_and_confirmation(prepared, monkeypatch):
    calls = []
    def fake(*args):
        calls.append(args)
        return {'status': 'completed', 'predictions': {'b': 2.1}, 'wall_seconds': 1}
    monkeypatch.setattr(campaign, 'run_container', fake)
    with pytest.raises(ValueError):
        campaign.confirm(prepared, 'chatgpt')
    for arm in campaign.ARMS:
        for _ in range(3):
            packet = campaign.packet(prepared, 'chatgpt', arm)
            campaign.submit(prepared, 'chatgpt', arm, response(packet, prepared / 'response.json'))
    result = campaign.confirm(prepared, 'chatgpt')
    assert result['status'] == 'completed'
    assert len(result['rows']) == 9
    assert len(calls) == 21
    with pytest.raises(ValueError):
        campaign.confirm(prepared, 'chatgpt')
    assert 'confirmation' not in campaign.load(prepared)[1]['providers']['claude']


def test_interruption_recovery_consumes_attempt(prepared):
    campaign.packet(prepared, 'claude', 'rsi')
    campaign.recover(prepared, 'claude', 'rsi')
    assert len(campaign.packet(prepared, 'claude', 'rsi', reserve=False)['history']) == 1
    with pytest.raises(ValueError):
        campaign.recover(prepared, 'claude', 'rsi')

def test_rejected_candidate_does_not_replace_parent(prepared, monkeypatch):
    monkeypatch.setattr(campaign, 'run_container', lambda *args: {'status': 'completed', 'predictions': {'b': 5.0}})
    value = campaign.packet(prepared, 'chatgpt', 'rsi')
    result = campaign.submit(prepared, 'chatgpt', 'rsi', response(value, prepared / 'response.json'))
    assert result['accepted'] is False
    next_packet = campaign.packet(prepared, 'chatgpt', 'rsi', reserve=False)
    assert next_packet['source_files']['train.py'] == 'print("parent")'
    assert next_packet['history'][0]['feedback']['accepted'] is False


def test_runtime_failure_is_counted(prepared, monkeypatch):
    monkeypatch.setattr(campaign, 'run_container', lambda *args: {'status': 'timeout'})
    value = campaign.packet(prepared, 'claude', 'rsi')
    result = campaign.submit(prepared, 'claude', 'rsi', response(value, prepared / 'response.json'))
    assert result['status'] == 'failed'
    assert campaign.load(prepared)[1]['providers']['claude']['arms']['rsi']['incumbent'] == 'parent'


def test_metric_overflow_rejected():
    from rsi.blind_calibrate import score
    with pytest.raises(ValueError, match='overflow'):
        score({'a': 1e308}, [{'id': 'a', 'target': 1}])


def test_app_context_attestation_required(prepared):
    value = campaign.packet(prepared, 'claude', 'rsi')
    path = response(value, prepared / 'response.json')
    content = read(path)
    content['session']['tools_disabled'] = False
    write(path, content)
    assert campaign.submit(prepared, 'claude', 'rsi', path)['status'] == 'failed'


def test_mismatched_runtime_probe_rejected(prepared, tmp_path):
    probe = tmp_path / 'probe.json'
    data = read(probe)
    data['image_id'] = 'sha256:' + 'f' * 64
    write(probe, data)
    with pytest.raises(ValueError, match='Runtime probe'):
        campaign.initialize(tmp_path / 'package', tmp_path / 'calibration', tmp_path / 'bad-campaign', probe)

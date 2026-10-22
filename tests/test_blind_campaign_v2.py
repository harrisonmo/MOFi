"""Synthetic second-version campaign tests: no model calls or scientific training."""
import pytest
from rsi.blind_g1 import read, sha, write
from rsi import blind_campaign_v2 as campaign


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
          'parent_files': {'train.py': sha(calibration / 'parent/train.py')}, 'image_id': 'sha256:' + 'a' * 64,
          'fits': [{'seed': s, 'mae': 1, 'rmse': 1} for s in (17, 29)], 'fit_wall_seconds': 270})
    probe = tmp_path / 'probe.json'
    write(probe, {'passed': True, 'image_id': 'sha256:' + 'a' * 64, 'package_sha256': sha(package / 'manifest.json')})
    root = tmp_path / 'campaign'
    campaign.initialize(package, calibration, root, probe)
    for provider in campaign.PROVIDERS:
        campaign.register(root, provider, 'synthetic-model', 'cli-file-exchange')
    return root


@pytest.fixture
def fits(monkeypatch):
    calls = []
    def fake(*args):
        calls.append(args)
        return {'status': 'completed', 'predictions': {'b': 2.1}, 'wall_seconds': 1}
    monkeypatch.setattr(campaign, 'run_container', fake)
    return calls


def response(packet, path, hypothesis='Synthetic change', code='print("candidate")', model='anything'):
    write(path, {'packet_id': packet['packet_id'], 'hypothesis': hypothesis, 'files': {'train.py': code},
                 'session': {'fresh_context': True, 'tools_disabled': True, 'external_materials_used': False,
                             'model': model, 'usage': None}})
    return path


def test_policy_records_relaxed_budget(prepared):
    policy = read(prepared / 'policy.json')
    assert policy['rounds'] == 3 and policy['proposal_wall_seconds'] == 2700 and policy['fit_wall_seconds'] == 360
    assert policy['maximum_formal_fits_per_provider'] == 21 and policy['calibrated_fit_wall_seconds'] == 270
    assert 'blind_campaign_v2.py' in policy['controller_hashes']


def test_one_percent_promotion_with_regression_guard():
    parent = {'mean_mae': 1, 'seeds': {'17': {'mae': 1}, '29': {'mae': 1}}}
    assert campaign.accepted({'mean_mae': .985, 'seeds': {'17': {'mae': .98}, '29': {'mae': .99}}}, parent)
    assert not campaign.accepted({'mean_mae': .995, 'seeds': {'17': {'mae': .99}, '29': {'mae': 1}}}, parent)
    assert not campaign.accepted({'mean_mae': .9, 'seeds': {'17': {'mae': .74}, '29': {'mae': 1.06}}}, parent)


def test_check_reports_errors_without_running_code_and_is_limited(prepared, fits):
    first = campaign.packet(prepared, 'claude', 'rsi')
    path = response(first, prepared / 'draft.json', code='def broken(:\n')
    result = campaign.check(prepared, 'claude', 'rsi', path)
    assert not result['valid'] and 'Syntax error' in result['errors'][0] and result['checks_remaining'] == 1
    path.write_text('not json')
    assert 'Not valid JSON' in campaign.check(prepared, 'claude', 'rsi', path)['errors'][0]
    with pytest.raises(ValueError, match='allowance'):
        campaign.check(prepared, 'claude', 'rsi', path)
    assert fits == []
    slot = read(prepared / 'ledger.json')['providers']['claude']['arms']['rsi']['slots'][-1]
    assert slot['status'] == 'awaiting' and slot['format_checks'] == 2


@pytest.mark.parametrize('payload', ['null', '[]', '[{"packet_id": "x"}]', '"text"', '3', 'true', '{}'])
def test_non_object_json_is_invalid_at_check_and_submit(prepared, fits, payload):
    campaign.packet(prepared, 'claude', 'rsi')
    path = prepared / 'draft.json'
    path.write_text(payload)
    result = campaign.check(prepared, 'claude', 'rsi', path)
    assert not result['valid'] and result['errors']
    feedback = campaign.submit(prepared, 'claude', 'rsi', path)
    assert feedback['status'] == 'failed' and feedback['reason'].startswith('Invalid submission')
    assert fits == []
    slot = read(prepared / 'ledger.json')['providers']['claude']['arms']['rsi']['slots'][-1]
    assert slot['status'] == 'failed'


@pytest.mark.parametrize('hypothesis', [None, 7, ['a'], {'a': 1}, '   '])
def test_wrong_typed_fields_are_reported_not_raised(prepared, fits, hypothesis):
    first = campaign.packet(prepared, 'claude', 'rsi')
    path = response(first, prepared / 'draft.json')
    value = read(path)
    value['hypothesis'] = hypothesis
    value['files'] = {'train.py': 5}
    value['session'] = 'yes'
    write(path, value)
    result = campaign.check(prepared, 'claude', 'rsi', path)
    assert not result['valid'] and len(result['errors']) == 3
    assert campaign.submit(prepared, 'claude', 'rsi', path)['status'] == 'failed'
    assert fits == []


def test_undecodable_bytes_are_invalid(prepared, fits):
    campaign.packet(prepared, 'claude', 'rsi')
    path = prepared / 'draft.json'
    path.write_bytes(b'\xff\xfe\x00{')
    assert not campaign.check(prepared, 'claude', 'rsi', path)['valid']
    assert campaign.submit(prepared, 'claude', 'rsi', path)['status'] == 'failed'


def test_long_hypothesis_is_truncated_not_rejected(prepared, fits):
    first = campaign.packet(prepared, 'claude', 'independent')
    path = response(first, prepared / 'r.json', hypothesis='x' * 2041)
    assert campaign.check(prepared, 'claude', 'independent', path)['warnings']
    assert campaign.submit(prepared, 'claude', 'independent', path)['status'] == 'completed'
    slot = read(prepared / 'ledger.json')['providers']['claude']['arms']['independent']['slots'][-1]
    assert slot['hypothesis_truncated'] and slot['hypothesis_chars'] == 2041
    assert len(read(prepared / 'submissions/claude/independent-1.json')['hypothesis']) == 2000


def test_model_field_is_a_self_report(prepared, fits):
    first = campaign.packet(prepared, 'claude', 'rsi')
    path = response(first, prepared / 'r.json', model='claude-opus-5-5 (not independently verified)')
    assert campaign.submit(prepared, 'claude', 'rsi', path)['accepted']
    slot = read(prepared / 'ledger.json')['providers']['claude']['arms']['rsi']['slots'][-1]
    assert slot['registered_model'] == 'synthetic-model'
    assert slot['self_reported_model'].startswith('claude-opus-5-5')


def test_attestation_and_syntax_still_consume_the_slot_at_submit(prepared, fits):
    first = campaign.packet(prepared, 'chatgpt', 'rsi')
    path = response(first, prepared / 'r.json')
    value = read(path)
    value['session']['tools_disabled'] = False
    write(path, value)
    assert campaign.submit(prepared, 'chatgpt', 'rsi', path)['status'] == 'failed'
    assert campaign.packet(prepared, 'chatgpt', 'rsi', reserve=False)['history'][0]['status'] == 'failed'
    assert fits == []


def test_three_rounds_then_confirmation(prepared, fits):
    for arm in campaign.ARMS:
        for _ in range(3):
            packet = campaign.packet(prepared, 'chatgpt', arm)
            assert campaign.submit(prepared, 'chatgpt', arm, response(packet, prepared / 'r.json'))['status'] == 'completed'
        with pytest.raises(ValueError):
            campaign.packet(prepared, 'chatgpt', arm)
    result = campaign.confirm(prepared, 'chatgpt')
    assert result['status'] == 'completed' and len(result['rows']) == 9
    assert len(fits) == 21
    assert all(call[-1] == 360 for call in fits)


def test_deadline_applies_to_check_and_submit(prepared, fits):
    first = campaign.packet(prepared, 'claude', 'rsi')
    path = response(first, prepared / 'r.json')
    ledger = read(prepared / 'ledger.json')
    ledger['providers']['claude']['arms']['rsi']['slots'][-1]['started_at'] = 0
    write(prepared / 'ledger.json', ledger)
    with pytest.raises(ValueError, match='deadline'):
        campaign.check(prepared, 'claude', 'rsi', path)
    assert campaign.submit(prepared, 'claude', 'rsi', path)['reason'] == 'Proposal deadline exceeded'


def test_v1_campaigns_are_refused(prepared):
    policy = read(prepared / 'policy.json')
    policy['schema'] = 'blind-g1-campaign-v2'
    write(prepared / 'policy.json', policy)
    with pytest.raises(ValueError):
        campaign.load(prepared)

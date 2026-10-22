"""Synthetic lineage tests: no model calls or scientific training."""
import pytest
from rsi.blind_g1 import read, sha, write
from rsi import blind_campaign_v2 as campaign
from rsi import blind_lineage as lineage


@pytest.fixture
def setup(tmp_path):
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
    first = tmp_path / 'campaign-02'
    campaign.initialize(package, calibration, first, probe)
    campaign.register(first, 'claude', 'synthetic-model', 'cli-file-exchange')
    return tmp_path, first, probe


def run_generation(root, quality, monkeypatch):
    """Complete a campaign where each source's prediction for 'b' is looked up by its code."""
    def fake(public, candidate, image, seed, wall):
        code = (candidate / 'train.py').read_text()
        return {'status': 'completed', 'predictions': {'b': quality.get(code, 3.0)}, 'wall_seconds': 5}
    monkeypatch.setattr(campaign, 'run_container', fake)
    for arm in campaign.ARMS:
        for generation in range(3):
            packet = campaign.packet(root, 'claude', arm)
            code = f'print("{arm}-{generation}")'
            write(root / 'r.json', {'packet_id': packet['packet_id'], 'hypothesis': 'h', 'files': {'train.py': code},
                                    'session': {'fresh_context': True, 'tools_disabled': True,
                                                'external_materials_used': False, 'model': 'm', 'usage': None}})
            campaign.submit(root, 'claude', arm, root / 'r.json')
    assert campaign.confirm(root, 'claude')['status'] == 'completed'


def test_champion_becomes_next_parent(setup, monkeypatch):
    tmp_path, first, probe = setup
    # Parent predicts 3.0 (error 1.0); rsi-1 predicts 2.4 (error 0.4); independent-2 predicts 2.2 (error 0.2).
    run_generation(first, {'print("rsi-0")': 2.4, 'print("independent-1")': 2.2}, monkeypatch)
    best = lineage.champion(first, 'claude')
    assert best['configuration'] == 'independent' and best['improved']
    assert best['files'] == {'train.py': 'print("independent-1")'}
    second = tmp_path / 'campaign-03'
    decision = lineage.successor(first, second, probe, 'claude', 'synthetic-model', 'cli-file-exchange')
    assert decision['continued']
    policy = read(second / 'policy.json')
    assert policy['baseline']['mean_mae'] == pytest.approx(0.2)
    assert read(second / 'parent.json') == {'train.py': 'print("independent-1")'}
    packet = campaign.packet(second, 'claude', 'independent', reserve=False)
    assert packet['source_files'] == {'train.py': 'print("independent-1")'} and packet['history'] == []


def test_lineage_stops_without_confirmed_improvement(setup, monkeypatch):
    tmp_path, first, probe = setup
    # Error 0.995 against the parent's 1.0 is a 0.5% gain, below the 1% continuation margin.
    run_generation(first, {'print("rsi-0")': 2.995}, monkeypatch)
    decision = lineage.successor(first, tmp_path / 'campaign-03', probe, 'claude', 'synthetic-model', 'cli-file-exchange')
    assert not decision['continued'] and not (tmp_path / 'campaign-03').exists()


def test_one_percent_gain_continues_the_lineage(setup, monkeypatch):
    tmp_path, first, probe = setup
    # Error 0.985 is a 1.5% gain: promoted under the 1% margin and enough to continue the lineage.
    run_generation(first, {'print("rsi-0")': 2.985}, monkeypatch)
    best = lineage.champion(first, 'claude')
    assert best['configuration'] == 'rsi' and best['improved']
    assert lineage.IMPROVEMENT == .01


def test_successor_requires_completed_confirmation(setup):
    tmp_path, first, probe = setup
    with pytest.raises(ValueError, match='confirmation'):
        lineage.champion(first, 'claude')

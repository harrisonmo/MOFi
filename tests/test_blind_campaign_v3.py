"""Offline tests; no GPU, model, or candidate execution."""
import pytest
from rsi import blind_campaign_v3 as v3


def test_rejected_route_remains_available():
    state = v3.initialize({'train.py': 'x = 1\n'})
    key = v3.propose(state, 'repair', 'g1', 'Add features', {'train.py': 'x = 2\n'})
    v3.finish(state, {'feature_coverage': 0}, status='failed')
    exported = v3.packet(state)
    assert exported['incumbent'] == 'g1'
    assert exported['sources'][key]['train.py'] == 'x = 2\n'
    assert '+x = 2' in exported['history'][0]['diff']
    assert exported['history'][0]['health_errors']
    v3.propose(state, 'repair', key, 'Fix failed route')


def test_budget_and_pending():
    state = v3.initialize({'train.py': 'pass'})
    state['policy']['diagnostics'] = 1
    v3.propose(state, 'diagnose', 'g1', 'Check input')
    with pytest.raises(ValueError):
        v3.propose(state, 'diagnose', 'g1', 'Duplicate')
    v3.finish(state, {}, status='interrupted')
    with pytest.raises(ValueError):
        v3.propose(state, 'diagnose', 'g1', 'Retry')


@pytest.mark.parametrize('coverage', [None, 0, float('nan'), True, 0.99])
def test_missing_features_fail(coverage):
    assert v3.health_errors(dict(train_only=True, finite_features=True,
                                forward_pass=True, dimensions_valid=True,
                                feature_coverage=coverage))


def test_execution_disabled_and_packet_isolated():
    state = v3.initialize({'train.py': 'pass'})
    v3.packet(state)['sources']['g1']['train.py'] = 'changed'
    assert state['sources']['g1']['train.py'] == 'pass'
    with pytest.raises(RuntimeError):
        v3.execute(state)

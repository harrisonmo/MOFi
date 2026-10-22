"""Synthetic lineage continuation and restart tests."""
import json
from test_blind_v3_runner import campaign, fake_gpu
from rsi import blind_v3_lineage as lineage
from rsi import blind_v3_runner as runner
from rsi.blind_g1 import read


def test_two_generations_and_idempotent_resume(campaign):
    path = campaign.parent / 'lineage.json'
    lineage.initialize(path, campaign, 2)
    calls = []
    def invoke(prompt, directory, timeout):
        calls.append(prompt)
        return {'passed_transport': True}, json.dumps(dict(action='stop', base='g1', hypothesis='Stop search'))
    result = lineage.run(path, invoke, fake_gpu)
    assert result['status'] == 'completed'
    assert len(calls) == 2
    child = campaign.parent / 'generation-02'
    state = runner.load(child)
    assert state['history'][0]['id'] == 'g2-action-1'
    assert state['previous_generations'][0]['history'][0]['hypothesis'] == 'Stop search'
    assert state['sources']['g1'] == {'train.py': 'pass'}
    lineage.run(path, invoke, fake_gpu)
    assert len(calls) == 2
    lineage.successor(campaign, child, 2)
    assert read(path)['campaigns'][-1] == str(child.resolve())

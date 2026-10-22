"""Regressions for the observed Research type confusion, independent of prompts."""
from copy import deepcopy
import jsonschema
import pytest
from rsi.q2c.data import dataset
from rsi.q2c.runtime import manual_program, TOOL
from rsi.q2c.scoring import score, evidence_values, target
from rsi.q2c_v2.contract import typecheck, execute, validate_results, manifest, SCHEMAS
from rsi.q2c_v2.protocol import research_prompt


def registered(task):
    fields = {'G': ['graph_a', 'graph_b'], 'M': ['mic'], 'S': ['mappings']}[task['family']]
    return dict(version=1, nodes=[dict(id='result', op=TOOL, inputs=['A', 'B'])],
                outputs=['result.' + field for field in fields])


@pytest.mark.parametrize('task', dataset(), ids=lambda t: t['id'])
def test_typed_registered_results_against_unchanged_reference(task):
    program = registered(task)
    out = execute(program, task['A'], task['B'], manual_program())
    handles = program['outputs']
    values = evidence_values(task['family'], out['outputs'], handles)
    assert score(task, out, dict(uncertain=False, evidence=handles, values=values))['passed']


def test_reject_old_double_lattice_before_any_scientific_execution(monkeypatch):
    program = registered(dataset()[0])
    program['nodes'].append(dict(id='wrong', op='integer_lattice', inputs=['result.graph_a']))
    def forbidden(*args):
        raise AssertionError('Must reject before executing a primitive')
    monkeypatch.setattr('rsi.q2c_v2.contract.raw_execute', forbidden)
    with pytest.raises(ValueError, match='expects.*CycleGroups.*got.*LatticeRows'):
        execute(program, {}, {}, manual_program())


def test_reject_type_invalid_author_and_removal():
    tool = manual_program()
    tool['nodes'][2]['op'] = 'cycle_groups'
    tool['nodes'][2]['inputs'] = ['A']
    with pytest.raises(ValueError, match='EvidenceBundle'):
        typecheck(tool, author=True)
    with pytest.raises(ValueError, match='Unavailable'):
        typecheck(registered(dataset()[0]))


def test_unknown_projection_and_structural_mismatch():
    program = registered(dataset()[0])
    program['outputs'] = ['result.graph_a.cycles']
    with pytest.raises(ValueError, match='projectable'):
        typecheck(program, manual_program())
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate([dict(atoms=[0], cycles=[[1, 0, 0]])], SCHEMAS['LatticeRows'])


def test_contract_and_prompt_share_the_same_output_shapes():
    meta = manifest(manual_program())
    assert meta['fields']['EvidenceBundle']['graph_a'] == 'LatticeRows'
    for task in dataset():
        prompt = research_prompt(task, True)
        assert 'graph_a' in prompt and 'LatticeRows' in prompt and 'exactly' in prompt
        assert 'EvidenceBundle' not in research_prompt(task, False)


def test_scalar_plus_result_still_fails_frozen_score():
    task = next(t for t in dataset() if t['id'] == 'D1-S')
    program = registered(task)
    program['outputs'].append('result.mappings.multiplicity')
    output = execute(program, task['A'], task['B'], manual_program())
    report = dict(uncertain=False, evidence=program['outputs'], values=target(task))
    assert not score(task, output, report)['passed']
    report['evidence'] = [program['outputs'][0]]
    assert score(task, output, report)['passed']


def test_plan_cannot_forge_result_shape():
    task = dataset()[0]
    program = registered(task)
    output = execute(program, task['A'], task['B'], manual_program())
    corrupted = deepcopy(output)
    corrupted['outputs']['result.graph_a'][0]['index'] = 0
    with pytest.raises(jsonschema.ValidationError):
        validate_results(program, corrupted, manual_program())

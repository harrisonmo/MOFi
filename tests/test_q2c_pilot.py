"""Evaluator-independent analytic values and actual registry wiring tests."""
from copy import deepcopy
import json
import jsonschema
import pytest
from rsi.q2c.data import dataset
from rsi.q2c.runtime import check,execute,manual_program,TOOL
from rsi.q2c.protocol import plan_schema,research_prompt,AUTHOR_SCHEMA
from rsi.q2c.scoring import target,score
from rsi.q2c.worker import run
from rsi.q2c.primitives import validate,cycle_groups,integer_lattice
from validation.q2c_reference import periodic_signature

TASKS=dataset()
ANALYTIC={'D1-G':[1,1,1,2],'D2-G':[2,1,1,1],'E1-G':[3,1,2,1],
          'E2-G':[0,1,1,1,0,1,1,2],
          'D1-M':[2273/200000,1],'D2-M':[1,2],'E1-M':[.3125,1],'E2-M':[.10036,2],
          'D1-S':[2],'D2-S':[1],'E1-S':[2],'E2-S':[1]}


@pytest.mark.parametrize('task',TASKS,ids=lambda t:t['id'])
def test_analytic_targets_and_fixed_reachability(task):
    assert target(task)==ANALYTIC[task['id']]
    assert run(dict(action='validate',task=task))['passed']


@pytest.mark.parametrize('task',TASKS,ids=lambda t:t['id'])
def test_registered_end_to_end(task):
    assert run(dict(action='validate',task=task,registered=True))['passed']


def test_removal_unknown_name_and_id_contract():
    p=dict(version=1,nodes=[dict(id='audit',op=TOOL,inputs=['A','B'])],outputs=['audit'])
    jsonschema.validate(dict(program=p),plan_schema(True))
    with pytest.raises(ValueError):check(p)
    with pytest.raises(jsonschema.ValidationError):jsonschema.validate(dict(program=p),plan_schema(False))
    for name in ['Audit','components_A','a-b']:
        changed=deepcopy(p);changed['nodes'][0]['id']=name
        with pytest.raises(ValueError):check(changed,manual_program())
        with pytest.raises(jsonschema.ValidationError):jsonschema.validate(dict(program=changed),plan_schema(True))
    p['nodes'][0]['op']='registered'
    with pytest.raises(ValueError):check(p,manual_program())


def test_nested_output_and_projected_handles():
    task=TASKS[0];tool=manual_program()
    p=dict(version=1,nodes=[dict(id='audit',op=TOOL,inputs=['A','B'])],outputs=['audit','audit.graph_a'])
    before=json.dumps(task,sort_keys=True)
    out=execute(p,task['A'],task['B'],tool)
    assert out['outputs']['audit']['graph_a']==out['outputs']['audit.graph_a']
    assert out['operations']==6 and json.dumps(task,sort_keys=True)==before
    with pytest.raises(ValueError):check(p,tool,author=True)


def test_scoring_rejects_guesses_wrong_binding_and_false_numbers():
    t=TASKS[0];out=execute(manual_program(),t['A'],t['B'])
    report=dict(uncertain=False,evidence=['graph_a','graph_b'],values=ANALYTIC[t['id']])
    assert score(t,out,report)['passed']
    assert not score(t,out,dict(report,evidence=[]))['passed']
    assert not score(t,out,dict(report,values=[0,1,0,1]))['passed']
    assert not score(t,dict(out,input_hash='forged'),report)['passed']
    assert not score(t,out,dict(report,uncertain=True))['passed']


@pytest.mark.parametrize('field,value',[('unit','nm'),('occupancy',[1,.5]),('lattice',[[0,0,0]]*3),('frac',[[float('nan'),0,0],[.5,0,0]])])
def test_domain_rejections(field,value):
    a=deepcopy(TASKS[0]['A']);a[field]=value
    with pytest.raises(ValueError):validate(a)


@pytest.mark.parametrize('task',TASKS[::3])
def test_union_find_invariance_vs_independent_reference(task):
    a=deepcopy(task['A']);n=len(a['frac']);g=[[i,-i,2*i] for i in range(n)]
    a['edges']=[[n-1-j,n-1-i,-x-g[i][0]+g[j][0],-y-g[i][1]+g[j][1],-z-g[i][2]+g[j][2]] for i,j,x,y,z in a['edges']]
    actual=sorted((r['rank'],r['index']) for r in integer_lattice(cycle_groups(a)))
    expected=sorted((r['rank'],r['saturation_index']) for r in periodic_signature(n,task['A']['edges']))
    assert actual==expected


def test_split_and_prompt_nonleakage():
    from collections import Counter
    assert Counter(t['block'] for t in TASKS)==dict(development=2,source_transfer=2,task_transfer=4,joint_transfer=4)
    for task in TASKS:
        prompt=research_prompt(task,False)
        assert 'expected' not in prompt and 'periodic_evidence(A,B)' not in prompt
    assert TOOL not in AUTHOR_SCHEMA['properties']['program']['properties']['nodes']['items']['properties']['op']['enum']


def test_budget_unknown_usage_and_stage_caps(tmp_path):
    from rsi.q2c.run import LIMITS
    from rsi.q2b_v21.budget import RunBudget,BudgetStop
    budget=RunBudget(tmp_path/'budget.json',LIMITS)
    for i in range(2):
        row=budget.reserve_model('probe',str(i))
        budget.finish_model(row,dict(status='ok',usage=dict(input_tokens=1,output_tokens=1)))
    with pytest.raises(BudgetStop):budget.reserve_model('probe','third')
    row=budget.reserve_model('author','failed')
    budget.finish_model(row,dict(status='timeout',usage=None))
    with pytest.raises(BudgetStop):budget.reserve_model('research','after_unknown')
    assert LIMITS['total_model_calls']==2+3+12*4*2


def test_data_freeze_mutation_is_rejected(tmp_path):
    from rsi.q2c.run import Pilot,fingerprints
    from rsi.q2c.runtime import digest
    from rsi.q2b_v21.budget import BudgetStop
    pilot=object.__new__(Pilot);pilot.frozen=fingerprints();pilot.gate=dict(task_hash=digest(TASKS))
    pilot.read=lambda name:deepcopy(TASKS)
    pilot.unchanged()
    modified=deepcopy(TASKS);modified[0]['A']['edges'][0][2]=1
    pilot.read=lambda name:modified
    with pytest.raises(BudgetStop):pilot.unchanged()

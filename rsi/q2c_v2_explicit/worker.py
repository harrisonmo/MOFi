"""One bounded scientific action per worker process."""
import json
import sys
from rsi.q2c_v2.contract import execute,manual_program,digest
from rsi.q2c.scoring import target,score,evidence_values,equivalent


def run(payload):
    action=payload['action']
    if action=='tests':
        import contextlib,io,os
        import xml.etree.ElementTree as ET
        import pytest
        os.environ['PYTEST_DISABLE_PLUGIN_AUTOLOAD']='1'
        stream=io.StringIO()
        with contextlib.redirect_stdout(stream),contextlib.redirect_stderr(stream):
            code=pytest.main(['tests/test_q2c_v2_explicit.py','-q',
                             '--basetemp',payload['temp'],'--junitxml',payload['xml']])
        if code:raise ValueError(stream.getvalue())
        return dict(passed=True,checks=len(ET.parse(payload['xml']).findall('.//testcase')),log=stream.getvalue())
    if action=='execute':return execute(payload['program'],payload['A'],payload['B'],payload.get('tool'))
    if action=='score':return score(payload['task'],payload['output'],payload['report'])
    if action=='validate':
        task=payload['task'];tool=payload.get('tool',manual_program())
        program=dict(version=1,nodes=[dict(id='audit',op='periodic_evidence',inputs=['A','B'])],
                     outputs=['audit.graph_a','audit.graph_b','audit.mic','audit.mappings']) if payload.get('registered') else tool
        output=execute(program,task['A'],task['B'],tool if payload.get('registered') else None)
        prefix='audit.' if payload.get('registered') else ''
        handles={'G':[prefix+'graph_a',prefix+'graph_b'],'M':[prefix+'mic'],'S':[prefix+'mappings']}[task['family']]
        expected=target(task)
        values=evidence_values(task['family'],output['outputs'],handles)
        report=dict(uncertain=False,evidence=handles,values=values)
        checked=score(task,output,report)
        if not checked['passed']:raise ValueError('Independent reference mismatch')
        for family,refs in {'G':['graph_a','graph_b'],'M':['mic'],'S':['mappings']}.items():
            observed=evidence_values(family,output['outputs'],[prefix+r for r in refs])
            if not equivalent(observed,target(dict(task,family=family))):
                raise ValueError('Generic evidence contract mismatch: '+family)
        return dict(id=task['id'],values=values,expected=expected,passed=True,operations=output['operations'])
    raise ValueError('Unknown worker action')


if __name__=='__main__':
    try:result=dict(ok=True,value=run(json.load(sys.stdin)))
    except Exception as exc:result=dict(ok=False,error=f'{type(exc).__name__}: {exc}')
    print(json.dumps(result,allow_nan=False))

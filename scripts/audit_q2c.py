"""Read-only Q2C result audit and compact export, including incomplete runs."""
from collections import Counter
import hashlib
import json
import math
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
RUN=ROOT/'runs/q2c-v1'


def read(name,default=None):
    path=RUN/name
    return json.loads(path.read_text(encoding='utf-8')) if path.exists() else default


def digest(x):
    return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def semantic_program(program):
    """Canonical dataflow expressions, ignoring node names and ordering."""
    values={'A':'A','B':'B'}
    def ref(name):
        first,*fields=name.split('.')
        value=values[first]
        for field in fields:value=['field',value,field]
        return value
    for node in program['nodes']:values[node['id']]=[node['op'],*[ref(r) for r in node['inputs']]]
    return {key:ref(key) for key in sorted(program['outputs'])}


def main():
    manifest=read('manifest.json');completion=read('completion.json')
    if completion is None:raise RuntimeError('Only audit a stopped/completed run')
    model=read('model_budget.json');science=read('science_budget.json');episodes=read('episodes.json',[])
    tasks=read('tasks.json');candidate=read('candidate.json');authors=read('authors.json',[])
    checks=[]
    def check(name,ok):
        checks.append(dict(name=name,passed=bool(ok)))
        if not ok:raise AssertionError(name)
    check('frozen_files',all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h for p,h in manifest['files'].items()))
    check('dataset_hash',digest(tasks)==manifest['gate']['task_hash'])
    check('task_ids_unique',len({t['id'] for t in tasks})==12)
    check('clean_start',not manifest['root_git']['status'] and not manifest['crystal_builder_git']['status'])
    check('calls_bounded',len(model['entries'])<=101)
    check('science_bounded',sum(r.get('wall_seconds',0) for r in science['entries'])<=600)
    check('checks_bounded',sum(r.get('evaluations',1) for r in science['entries'])<=1000)
    check('no_unresolved_science',not any(r['status']=='reserved' for r in science['entries']))
    calls=[];diagnostics=[]
    for row in model['entries']:
        raw=read('model/'+row['label']+'/result.json')
        check('result_'+row['label'],raw is not None)
        usage=raw['usage']
        tokens=usage['input_tokens']+usage['output_tokens'] if usage is not None else None
        check('tokens_'+row['label'],row['tokens']==tokens)
        check('status_'+row['label'],row['status']==raw['status'])
        check('no_tool_use_'+row['label'],not raw['unexpected_tool_types'])
        calls.append(row)
        stderr=(RUN/'model'/row['label']/'stderr.log').read_bytes()
        if stderr:diagnostics.append(dict(label=row['label'],bytes=len(stderr),sha256=hashlib.sha256(stderr).hexdigest()))
    schedule=read('schedule.json',[])
    check('attempted_prefix',[(r['id'],r['arm']) for r in episodes]==[(r['id'],r['arm']) for r in schedule[:len(episodes)]])
    check('attempted_unique',len({(r['id'],r['arm']) for r in episodes})==len(episodes))
    index={t['id']:t for t in tasks}
    for row in episodes:
        label=row['id']+'-'+row['arm'];task=index[row['id']]
        cc=[c for c in calls if c['stage']=='research' and c['label'].startswith(label+'-')]
        check('call_count_'+label,len(cc)==row['calls'] and len(cc)<=2)
        expected=None if any(c['tokens'] is None for c in cc) else sum(c['tokens'] for c in cc)
        check('episode_tokens_'+label,expected==row['tokens'])
        evidence=read('evidence/'+label+'.json')
        report_result=read('model/'+label+'-report/result.json',{})
        response=report_result.get('response')
        row['audit_uncertain']=response.get('uncertain') if isinstance(response,dict) else None
        expected=next(r['expected'] for r in manifest['gate']['results'] if r['id']==row['id'])
        reported=response.get('values',[]) if isinstance(response,dict) else []
        row['audit_numeric_correct']=len(reported)==len(expected) and all(type(x) in (int,float) and math.isfinite(x) and abs(x-y)<=1e-10 for x,y in zip(reported,expected))
        if evidence:
            plan=read('model/'+label+'-plan/result.json')['response']['program']
            check('evidence_input_'+label,evidence['input_hash']==digest([task['A'],task['B']]))
            check('evidence_program_'+label,evidence['program_hash']==digest(plan))
        if row['passed']:check('passed_has_evidence_'+label,row['certificate_valid'] and bool(evidence) and row['completed'])
    names=['fixed','outer_generated','manual_control','removed'];arms={};blocks={};paired={}
    for arm in names:
        rows=[r for r in episodes if r['arm']==arm]
        cs=[c for c in calls if c['stage']=='research' and '-'+arm+'-' in c['label']]
        arms[arm]=dict(attempted=len(rows),planned=12,completed=sum(r['completed'] for r in rows),passed=sum(r['passed'] for r in rows),
            tool_errors=sum('tool_error' in r for r in rows),certificate_valid=sum(r['certificate_valid'] for r in rows),
            incorrect_completed=sum(r['completed'] and not r['passed'] for r in rows),
            abstained=sum(r.get('audit_uncertain') is True for r in rows),
            confident_rejected=sum(r['completed'] and not r['passed'] and r.get('audit_uncertain') is False for r in rows),
            confident_numerically_wrong=sum(r['completed'] and r.get('audit_uncertain') is False and not r['audit_numeric_correct'] for r in rows),
            numerically_correct_contract_rejected=sum(r['completed'] and not r['passed'] and r.get('audit_uncertain') is False and r['audit_numeric_correct'] for r in rows),
            calls=len(cs),known_tokens=sum(c['tokens'] or 0 for c in cs),unknown_usage=sum(c['tokens'] is None for c in cs),
            requested_registered_uses=sum(r['registered_uses'] for r in rows),
            successful_registered_uses=sum(r['registered_uses'] for r in rows if r['operations']>0),
            retained_primitive_operations_lower_bound=sum(r['operations'] for r in rows),
            failed_batches_with_unreported_operation_count=sum('tool_error' in r for r in rows),
            reused_sources=sorted({r['source'] for r in rows if r['registered_uses'] and r['operations']>0}),
            reused_families=sorted({r['family'] for r in rows if r['registered_uses'] and r['operations']>0}))
        blocks[arm]={b:dict(passed=sum(r['passed'] for r in rows if r['block']==b),attempted=sum(r['block']==b for r in rows),planned=n)
                     for b,n in [('development',2),('source_transfer',2),('task_transfer',4),('joint_transfer',4)]}
    for control in ['fixed','manual_control','removed']:
        rows=[]
        for task in tasks:
            g=next((r for r in episodes if r['id']==task['id'] and r['arm']=='outer_generated'),None)
            c=next((r for r in episodes if r['id']==task['id'] and r['arm']==control),None)
            if g and c:rows.append(dict(id=task['id'],block=task['block'],generated_pass=g['passed'],control_pass=c['passed'],generated_tokens=g['tokens'],control_tokens=c['tokens']))
        known=[r for r in rows if r['generated_tokens'] is not None and r['control_tokens'] is not None]
        paired[control]=dict(matched=len(rows),missing=12-len(rows),wins=sum(r['generated_pass']>r['control_pass'] for r in rows),
            losses=sum(r['generated_pass']<r['control_pass'] for r in rows),ties=sum(r['generated_pass']==r['control_pass'] for r in rows),
            known_cost_pairs=len(known),generated_use_tokens=sum(r['generated_tokens'] for r in known),control_use_tokens=sum(r['control_tokens'] for r in known),rows=rows)
    semantic_equal=None
    if candidate:
        from rsi.q2c.runtime import manual_program
        check('candidate_hash',digest(candidate['program'])==candidate['sha256'])
        semantic_equal=semantic_program(candidate['program'])==semantic_program(manual_program())
    stages={s:dict(actions=sum(r['stage']==s for r in science['entries']),seconds=sum(r.get('wall_seconds',0) for r in science['entries'] if r['stage']==s)) for s in sorted({r['stage'] for r in science['entries']})}
    result=dict(version='q2c-v1',visibility='public_engineering_validation',manifest=manifest,completion=completion,arms=arms,blocks=blocks,
        paired_generated_vs=paired,author_tokens=sum(c['tokens'] or 0 for c in calls if c['stage']=='author'),
        probe_tokens=sum(c['tokens'] or 0 for c in calls if c['stage']=='probe'),
        known_token_lower_bound=sum(c['tokens'] or 0 for c in calls),unknown_usage=sum(c['tokens'] is None for c in calls),
        model_wall_seconds=sum(c.get('wall_seconds') or 0 for c in calls),science_stages=stages,
        science_seconds=sum(r.get('wall_seconds',0) for r in science['entries']),
        candidate_semantically_equal_to_manual=semantic_equal,candidate=candidate,authors=authors,episodes=episodes,calls=calls,
        stderr_files=diagnostics,audit=dict(checks=len(checks),passed=True),
        limits='Synthetic source families only; no real-material generalization. Engineering/development labor and tokens unmetered, not zero.')
    (ROOT/'benchmarks/q2c/astra-summary.json').write_text(json.dumps(result,indent=2)+'\n')
    (RUN/'audit.json').write_text(json.dumps(checks,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ['completion','arms','known_token_lower_bound','unknown_usage','audit']},indent=2))


if __name__=='__main__':main()

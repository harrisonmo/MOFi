"""Frozen four-arm Astra pilot with all-stage accounting and no resumption."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import random
import sys
from .limits import ROOT,Ledger
from .runtime import digest,manual_program,check,TOOL
from .protocol import CATALOG,AUTHOR,AUTHOR_SCHEMA,REPORT_SCHEMA,plan_schema,research_prompt
from rsi.q2b_v21.transport import CodexCLI
from rsi.q2b_v21.budget import RunBudget,BudgetStop
from rsi.experiment import git_state,cli_identity,write_json

LIMITS=dict(total_model_calls=101,probe_calls=2,generation_and_debug_calls=3,research_calls=96,
            observed_token_soft_limit=3_232_000,tokens_per_call_soft_limit=32_000)
ARMS=['fixed','outer_generated','manual_control','removed']


def fingerprints():
    paths=list((ROOT/'rsi/q2c').glob('*.py'))+[ROOT/p for p in (
        'validation/q2c_reference.py','rsi/q2b_v21/transport.py','rsi/q2b_v21/budget.py',
        'rsi/backends/codex_cli.py','rsi/experiment.py','docs/Q2C_PILOT_PROTOCOL.md')]
    return {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


class Pilot:
    def __init__(self,executable):
        self.output=ROOT/'runs/q2c-v1'
        self.read=lambda name:json.loads((self.output/name).read_text(encoding='utf-8'))
        if (self.output/'model_budget.json').exists():raise RuntimeError('Never overwrite or silently resume')
        self.tasks=self.read('tasks.json');self.gate=self.read('gate.json');self.manual=manual_program()
        if not self.gate['passed'] or digest(self.tasks)!=self.gate['task_hash'] or digest(self.manual)!=self.gate['tool_hash']:
            raise RuntimeError('Calibration admission failed')
        root,cb=git_state(ROOT),git_state(ROOT/'Crystal-Builder')
        if root['status'] or cb['status']:raise RuntimeError('Require clean independent repositories')
        self.frozen=fingerprints();self.backend=CodexCLI('gpt-6-astra','high',240,executable)
        self.manifest=dict(version='q2c-v1',visibility='public_engineering_validation',root_git=root,crystal_builder_git=cb,
            files=self.frozen,gate=self.gate,limits=LIMITS,requested_model='gpt-6-astra',actual_model=None,reasoning='high',
            cli=cli_identity(executable),python=sys.version,
            packages={p:importlib.metadata.version(p) for p in ('numpy','scipy','ase','jsonschema')},
            workflow='one-plan-one-batch-one-report',source_scope='synthetic_periodic_record_families')
        self.budget=RunBudget(self.output/'model_budget.json',LIMITS)
        self.science=Ledger(self.output/'science_budget.json');self.rows=[];self.authors=[]
        write_json(self.output/'manifest.json',self.manifest)

    def unchanged(self):
        if fingerprints()!=self.frozen or digest(self.read('tasks.json'))!=self.gate['task_hash']:
            raise BudgetStop('Frozen files or dataset changed')

    def model(self,stage,label,prompt,schema):
        self.unchanged();reservation=self.budget.reserve_model(stage,label)
        result=self.backend.run(prompt,schema,self.output/'model'/label)
        tokens=self.budget.finish_model(reservation,result)
        if result['status'] in ('unexpected_tool_use','auth_error','model_unavailable'):
            self.budget.blocked=result['status'];self.budget.save()
        print(json.dumps(dict(label=label,status=result['status'],tokens=tokens,blocked=self.budget.blocked)),flush=True)
        if self.budget.blocked:raise BudgetStop(self.budget.blocked)
        return result

    def author(self):
        development=[t for t in self.tasks if t['block']=='development'];feedback=None
        for attempt in range(3):
            result=self.model('author',f'author-{attempt+1}',AUTHOR+CATALOG+json.dumps(dict(development=development,feedback=feedback)),AUTHOR_SCHEMA)
            record=dict(attempt=attempt+1,accepted=False,status=result['status'],validation=[])
            try:
                if result['status']!='ok':raise ValueError('Invalid author response')
                tool=result['response']['program'];check(tool,author=True)
                record.update(program=tool,hash=digest(tool),hypothesis=result['response']['hypothesis'])
                for task in development:
                    record['validation'].append(self.science.run('author_validation',f'author-{attempt+1}-{task["id"]}',
                        dict(action='validate',task=task,registered=True,tool=tool),seconds=20))
                record.update(accepted=True,identical_to_manual=tool==self.manual)
                self.authors.append(record);write_json(self.output/'authors.json',self.authors)
                candidate=dict(id=TOOL,version=1,origin='outer_generated',program=tool,sha256=digest(tool))
                write_json(self.output/'candidate.json',candidate)
                return tool
            except Exception as exc:
                record['error']=str(exc);feedback=dict(error=str(exc),validation=record['validation'])
                self.authors.append(record);write_json(self.output/'authors.json',self.authors)
        raise BudgetStop('No valid candidate')

    def episode(self,task,arm,candidate):
        label=task['id']+'-'+arm
        tool=candidate if arm=='outer_generated' else self.manual if arm=='manual_control' else None
        prompt=research_prompt(task,tool is not None)
        row=dict(id=task['id'],source=task['source'],family=task['family'],block=task['block'],arm=arm,
                 completed=False,passed=False,certificate_valid=False,registered_uses=0,operations=0)
        self.rows.append(row);start=len(self.budget.entries)
        try:
            plan=self.model('research',label+'-plan',prompt+'\nReturn your plan now.',plan_schema(tool is not None))
            output=None;observation={'error':'Invalid plan'};scientific_start=len(self.science.data['entries'])
            if plan['status']=='ok':
                program=plan['response']['program']
                row['registered_uses']=sum(n['op']==TOOL for n in program['nodes'])
                try:
                    output=self.science.run('research_tools',label,dict(action='execute',program=program,A=task['A'],B=task['B'],tool=tool),seconds=20)
                    row['operations']=output['operations'];observation=output['outputs']
                    write_json(self.output/'evidence'/(label+'.json'),output)
                except Exception as exc:
                    row['tool_error']=str(exc);observation={'error':str(exc)}
            report=self.model('research',label+'-report',prompt+'\nExecuted plan: '+json.dumps(plan.get('response'))+
                '\nRetained evidence handles and values: '+json.dumps(observation)+
                '\nReturn final report, exact retained handle names and requested numeric values. '
                'Use uncertain=true if evidence is unavailable. No further tools or replanning.',REPORT_SCHEMA)
            row['completed']=report['status']=='ok'
            used=sum(r.get('wall_seconds',0) for r in self.science.data['entries'][scientific_start:])
            if row['completed'] and output is not None and used<20:
                row.update(self.science.run('research_score',label,dict(action='score',task=task,output=output,report=report['response']),seconds=20-used))
            else:row['error']='Missing evidence/report or science time exhausted'
        finally:
            calls=self.budget.entries[start:];row['calls']=len(calls)
            row['tokens']=None if any(c['tokens'] is None for c in calls) else sum(c['tokens'] for c in calls)
            write_json(self.output/'episodes.json',self.rows)
            print(json.dumps(row),flush=True)

    def run(self):
        stop=None
        try:
            schema=dict(type='object',properties={'ready':{'type':'boolean'}},required=['ready'],additionalProperties=False)
            for i in range(2):
                result=self.model('probe',f'probe-{i+1}','Return ready=true. Transport-only check. Do not use tools.',schema)
                if result['status']!='ok' or result['response']!={'ready':True}:raise BudgetStop('Transport gate failed')
            candidate=self.author();tasks=list(self.tasks);random.Random(731913).shuffle(tasks)
            schedule=[(t,ARMS[(i+j)%4]) for i,t in enumerate(tasks) for j in range(4)]
            write_json(self.output/'schedule.json',[dict(id=t['id'],arm=a) for t,a in schedule])
            for task,arm in schedule:self.episode(task,arm,candidate)
        except Exception as exc:stop=f'{type(exc).__name__}: {exc}'
        self.unchanged()
        completion=dict(completed=len(self.rows)==48 and stop is None,stop=stop,episodes=len(self.rows),
                        model_calls=len(self.budget.entries),frozen_unchanged=True)
        write_json(self.output/'completion.json',completion);print(json.dumps(completion),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--codex',required=True)
    Pilot(parser.parse_args().codex).run()

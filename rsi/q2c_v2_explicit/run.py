"""One-shot real Research semantic admission, never a tool-benefit estimate."""
import argparse
import hashlib
import importlib.metadata
import json
import sys
from .limits import ROOT, Ledger
from rsi.q2c_v2.contract import manifest as tool_manifest
from rsi.q2c_v2.protocol import research_prompt, REPORT_SCHEMA, plan_schema
from rsi.q2c.runtime import digest, manual_program, TOOL
from rsi.q2b_v21.transport import CodexCLI
from rsi.q2b_v21.budget import RunBudget, BudgetStop
from rsi.experiment import git_state, cli_identity, write_json

LIMITS = dict(total_model_calls=14, probe_calls=2, generation_and_debug_calls=0, research_calls=12,
              observed_token_soft_limit=448000, tokens_per_call_soft_limit=32000)


def validate_parent(tasks, rows, completion, budget):
    expected = {t['id'] for t in tasks}
    fixed = [r for r in rows if r['arm'] == 'fixed']
    if (len(expected) != 6 or len(fixed) != 6 or {r['id'] for r in fixed} != expected
            or not completion['completed'] or completion['stop'] is not None
            or budget['blocked'] or any(c['tokens'] is None for c in budget['entries'])
            or not all(r['completed'] and r['passed'] and r['route_passed'] and not r['registered_uses'] for r in fixed)):
        raise RuntimeError('Parent fixed-path admission is invalid')
    return True


def fingerprints():
    paths = list((ROOT / 'rsi/q2c_v2').glob('*.py')) + list((ROOT / 'rsi/q2c').glob('*.py'))
    paths += list((ROOT / 'rsi/q2c_v2_explicit').glob('*.py'))
    paths += [ROOT / 'docs/Q2C_V2_EXPLICIT_GATE.md', ROOT / 'tests/test_q2c_v2_explicit.py']
    paths += [ROOT / ('runs/q2c-v2-interface/' + p) for p in ('manifest.json','tasks.json','gate.json','episodes.json','model_budget.json','completion.json')]
    paths += [ROOT / p for p in ('validation/q2c_reference.py', 'rsi/q2b_v21/transport.py',
        'rsi/q2b_v21/budget.py', 'rsi/backends/codex_cli.py', 'rsi/experiment.py',
        'docs/Q2C_V2_PROTOCOL.md', 'docs/Q2C_V2_MATERIAL_BENCHMARK.md',
        'tests/test_q2c_v2.py', 'tests/test_q2c_pilot.py', 'tests/test_q2c_reference.py',
        'benchmarks/q2c_v2/material-sources.json')]
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


class InterfacePilot:
    def __init__(self, executable):
        self.output = ROOT / 'runs/q2c-v2-explicit'
        self.read = lambda name: json.loads((self.output / name).read_text(encoding='utf-8'))
        if (self.output / 'model_budget.json').exists():
            raise RuntimeError('Never overwrite or resume a frozen interface experiment')
        self.tasks = self.read('tasks.json')
        self.gate = self.read('gate.json')
        self.manual = manual_program()
        parent = ROOT / 'runs/q2c-v2-interface'
        read_parent = lambda p: json.loads((parent / p).read_text(encoding='utf-8'))
        validate_parent(self.tasks, read_parent('episodes.json'), read_parent('completion.json'), read_parent('model_budget.json'))
        for path, expected in read_parent('manifest.json')['files'].items():
            if hashlib.sha256((ROOT / path).read_bytes()).hexdigest() != expected:
                raise RuntimeError('Parent calibration implementation changed')
        if digest(self.tasks) != digest(read_parent('tasks.json')) or digest(self.gate) != digest(read_parent('gate.json')):
            raise RuntimeError('Parent calibration input mismatch')
        if (not self.gate['passed'] or digest(self.tasks) != self.gate['task_hash']
                or digest(self.manual) != self.gate['tool_hash']
                or digest(self.gate['contract']) != digest(tool_manifest(self.manual))):
            raise RuntimeError('Offline gate or contract mismatch')
        root, cb = git_state(ROOT), git_state(ROOT / 'Crystal-Builder')
        if root['status'] or cb['status']:
            raise RuntimeError('Require clean independent repositories')
        self.frozen = fingerprints()
        self.backend = CodexCLI('gpt-6-astra', 'high', 240, executable)
        self.manifest = dict(version='q2c-v2-explicit', visibility='public_engineering_validation',
            purpose='interface_admission_not_effect_estimate', root_git=root, crystal_builder_git=cb,
            files=self.frozen, gate=self.gate, limits=LIMITS, requested_model='gpt-6-astra', actual_model=None,
            reasoning='high', cli=cli_identity(executable), python=sys.version,
            packages={p: importlib.metadata.version(p) for p in ('numpy', 'scipy', 'ase', 'jsonschema')},
            workflow='one-plan-one-batch-one-report', contract=tool_manifest(self.manual))
        self.budget = RunBudget(self.output / 'model_budget.json', LIMITS)
        self.science = Ledger(self.output / 'science_budget.json')
        self.rows = []
        write_json(self.output / 'manifest.json', self.manifest)

    def unchanged(self):
        if (fingerprints() != self.frozen or digest(self.read('tasks.json')) != self.gate['task_hash']
                or digest(self.read('gate.json')) != digest(self.gate)):
            raise BudgetStop('Frozen inputs changed')

    def model(self, stage, label, prompt, schema):
        self.unchanged()
        reservation = self.budget.reserve_model(stage, label)
        result = self.backend.run(prompt, schema, self.output / 'model' / label)
        tokens = self.budget.finish_model(reservation, result)
        if result['status'] in ('unexpected_tool_use', 'auth_error', 'model_unavailable'):
            self.budget.blocked = result['status']
            self.budget.save()
        print(json.dumps(dict(label=label, status=result['status'], tokens=tokens, blocked=self.budget.blocked)), flush=True)
        if self.budget.blocked:
            raise BudgetStop(self.budget.blocked)
        return result

    def run(self):
        stop = None
        try:
            schema = dict(type='object', properties={'ready': {'type': 'boolean'}}, required=['ready'], additionalProperties=False)
            for i in range(2):
                result = self.model('probe', f'probe-{i+1}', 'Return ready=true. Transport check. Do not use tools.', schema)
                if result['status'] != 'ok' or result['response'] != {'ready': True}:
                    raise BudgetStop('Transport gate failed')
            schedule = [(t, 'registered_required') for t in self.tasks]
            write_json(self.output / 'schedule.json', [dict(id=t['id'], arm=a) for t, a in schedule])
            for task, arm in schedule:
                self.episode(task, arm)
        except Exception as exc:
            stop = f'{type(exc).__name__}: {exc}'
        self.unchanged()
        complete = len(self.rows) == 6 and all(r['completed'] for r in self.rows) and stop is None
        admitted = complete and all(r['passed'] and r.get('route_passed', False) and 'tool_error' not in r for r in self.rows)
        result = dict(completed=complete, admitted=admitted, stop=stop, episodes=len(self.rows),
                      passed=sum(r['passed'] for r in self.rows), model_calls=len(self.budget.entries), frozen_unchanged=True)
        write_json(self.output / 'completion.json', result)
        print(json.dumps(result), flush=True)
    def episode(self,task,arm):
        label=task['id']+'-'+arm
        tool=self.manual if arm=='registered_required' else None
        prompt=research_prompt(task,tool is not None)
        prompt += '\nMANDATORY INTERFACE DIAGNOSTIC: You MUST include a node whose op is exactly periodic_evidence with inputs A,B. Every reported evidence handle MUST be a correctly typed field of that node. A primitive-only solution FAILS this diagnostic even if numerically correct. Do not replace periodic_evidence with its primitive implementation. Decide which result fields satisfy the task using the type contract.' if tool else '\nInterface diagnostic: use shared primitives only.'
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
            registered_ids={n['id'] for n in plan.get('response',{}).get('program',{}).get('nodes',[]) if n['op']==TOOL}
            handles=report.get('response',{}).get('evidence',[])
            row['route_passed']=(bool(handles) and all(h.split('.')[0] in registered_ids for h in handles)) if tool else not registered_ids
            used=sum(r.get('wall_seconds',0) for r in self.science.data['entries'][scientific_start:])
            if row['completed'] and output is not None and used<20:
                row.update(self.science.run('research_score',label,dict(action='score',task=task,output=output,report=report['response']),seconds=20-used))
            else:row['error']='Missing evidence/report or science time exhausted'
        finally:
            calls=self.budget.entries[start:];row['calls']=len(calls)
            row['tokens']=None if any(c['tokens'] is None for c in calls) else sum(c['tokens'] for c in calls)
            write_json(self.output/'episodes.json',self.rows)
            print(json.dumps(row),flush=True)



if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--codex',required=True)
    InterfacePilot(parser.parse_args().codex).run()

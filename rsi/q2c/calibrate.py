"""Metered independent-reference and fixed-tool reachability admission gate."""
import json
from pathlib import Path
from .data import dataset
from .runtime import digest,manual_program
from .limits import Ledger,ROOT
from rsi.experiment import write_json


def main():
    output=ROOT/'runs/q2c-v1'
    if (output/'model_budget.json').exists():raise RuntimeError('No calibration changes after model admission')
    output.mkdir(parents=True,exist_ok=True)
    ledger=Ledger(output/'science_budget.json')
    if sum(r.get('evaluations',1) for r in ledger.data['entries'])+100>1000:
        raise RuntimeError('Insufficient reserved capacity for offline suite')
    tasks=dataset();write_json(output/'tasks.json',tasks)
    iteration=len(ledger.data['entries'])
    tests=ledger.run('offline_checks',f'tests-{iteration}',dict(action='tests',
        temp=str(output/f'pytest-{iteration}'),xml=str(output/f'pytest-{iteration}.xml')),seconds=20)
    ledger.data['entries'][-1]['evaluations']=tests['checks'];ledger.save()
    if sum(r.get('evaluations',1) for r in ledger.data['entries'])>1000:raise RuntimeError('Offline check cap')
    results=[]
    for task in tasks:
        for registered in (False,True):
            checked=ledger.run('calibration',task['id']+f'-registered-{registered}',
                dict(action='validate',task=task,registered=registered),seconds=20)
            results.append(dict(registered=registered,**checked))
    gate=dict(passed=True,version='q2c-v1',task_hash=digest(tasks),tool_hash=digest(manual_program()),
              test_checks=tests['checks'],fixed_and_registered_checks=len(results),results=results,
              model_calls=0,science_seconds=sum(r.get('wall_seconds',0) for r in ledger.data['entries']))
    write_json(output/'gate.json',gate)
    write_json(ROOT/'benchmarks/q2c/p0-summary.json',dict(gate=gate,science_ledger=ledger.data))
    print(json.dumps(gate),flush=True)


if __name__=='__main__':main()

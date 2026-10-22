"""Meter offline interface admission before the real Research gate."""
import json
from rsi.q2c.data import dataset
from rsi.q2c.runtime import digest, manual_program
from rsi.experiment import write_json
from .limits import ROOT, Ledger
from .contract import manifest


def tasks():
    return [t for t in dataset() if t['source'] in ('D1', 'D2')]


def main():
    output = ROOT / 'runs/q2c-v2-interface'
    if (output / 'model_budget.json').exists():
        raise RuntimeError('Never recalibrate a frozen model run')
    output.mkdir(parents=True, exist_ok=True)
    ledger = Ledger(output / 'science_budget.json')
    if sum(r.get('evaluations', 1) for r in ledger.data['entries']) + 160 > 1000:
        raise RuntimeError('Insufficient offline test capacity')
    iteration = len(ledger.data['entries'])
    tested = ledger.run('offline_checks', f'tests-{iteration}', dict(action='tests',
        temp=str(output / f'pytest-{iteration}'), xml=str(output / f'pytest-{iteration}.xml')), seconds=20)
    ledger.data['entries'][-1]['evaluations'] = tested['checks']
    ledger.save()
    records = tasks()
    results = []
    for task in records:
        for registered in (False, True):
            result = ledger.run('calibration', task['id'] + str(registered),
                dict(action='validate', task=task, registered=registered), seconds=20)
            results.append(dict(registered=registered, **result))
    gate = dict(passed=True, test_checks=tested['checks'], task_hash=digest(records),
                tool_hash=digest(manual_program()), contract=manifest(manual_program()), results=results)
    write_json(output / 'tasks.json', records)
    write_json(output / 'gate.json', gate)
    write_json(ROOT / 'benchmarks/q2c_v2/p0-summary.json', dict(gate=gate, science=ledger.data))
    print(json.dumps(dict(passed=True, tests=tested['checks'], checks=len(results))))


if __name__ == '__main__':
    main()

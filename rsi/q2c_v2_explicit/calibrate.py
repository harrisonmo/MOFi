"""Meter new controller checks; reuse unchanged archived scientific calibration."""
import json
from .limits import ROOT, Ledger
from .run import validate_parent
from rsi.experiment import write_json


def main():
    output = ROOT / 'runs/q2c-v2-explicit'
    if (output / 'model_budget.json').exists():
        raise RuntimeError('No offline changes after model freeze')
    parent = ROOT / 'runs/q2c-v2-interface'
    read = lambda p: json.loads((parent / p).read_text(encoding='utf-8'))
    validate_parent(read('tasks.json'), read('episodes.json'), read('completion.json'), read('model_budget.json'))
    ledger = Ledger(output / 'science_budget.json')
    i = len(ledger.data['entries'])
    result = ledger.run('offline_checks', f'tests-{i}', dict(action='tests',
        temp=str(output / f'pytest-{i}'), xml=str(output / f'pytest-{i}.xml')), seconds=20)
    ledger.data['entries'][-1]['evaluations'] = result['checks']
    ledger.save()
    for name in ('tasks.json', 'gate.json'):
        write_json(output / name, read(name))
    write_json(ROOT / 'benchmarks/q2c_v2/explicit-p0-summary.json', dict(controller_checks=result['checks'],
        parent_science_reused=True, science=ledger.data))
    print(json.dumps(dict(passed=True, checks=result['checks'])))


if __name__ == '__main__':
    main()

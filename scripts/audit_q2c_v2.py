"""Read-only interface audit using archived independent targets, not new science."""
import hashlib
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rsi.q2c.runtime import digest
from rsi.q2c.scoring import evidence_values, equivalent

RUN = ROOT / 'runs/q2c-v2-interface'


def read(name, default=None):
    path = RUN / name
    return json.loads(path.read_text(encoding='utf-8')) if path.exists() else default


def main(explicit=False):
    global RUN
    if explicit:
        RUN = ROOT / 'runs/q2c-v2-explicit'
    manifest, completion = read('manifest.json'), read('completion.json')
    if completion is None:
        raise RuntimeError('Audit only a finished or stopped run')
    model, science = read('model_budget.json'), read('science_budget.json')
    tasks, rows = read('tasks.json'), read('episodes.json', [])
    checks = []

    def check(name, ok):
        checks.append(dict(name=name, passed=bool(ok)))
        if not ok:
            raise AssertionError(name)

    check('frozen_files', all(hashlib.sha256((ROOT / p).read_bytes()).hexdigest() == h for p, h in manifest['files'].items()))
    check('frozen_gate', digest(read('gate.json')) == digest(manifest['gate']))
    check('dataset_hash', digest(tasks) == manifest['gate']['task_hash'])
    check('development_only', len(tasks) == 6 and {t['source'] for t in tasks} == {'D1', 'D2'})
    check('clean_start', not manifest['root_git']['status'] and not manifest['crystal_builder_git']['status'])
    check('model_budget', model['limits'] == manifest['limits'] and len(model['entries']) <= (14 if explicit else 26))
    check('science_budget', sum(r.get('wall_seconds', 0) for r in science['entries']) <= 600)
    check('offline_count', sum(r.get('evaluations', 1) for r in science['entries']) <= 1000)
    check('no_science_reservation', not any(r['status'] == 'reserved' for r in science['entries']))
    calls = model['entries']
    for call in calls:
        raw = read('model/' + call['label'] + '/result.json')
        check('raw_' + call['label'], raw is not None)
        usage = raw.get('usage')
        tokens = usage['input_tokens'] + usage['output_tokens'] if usage else None
        check('cost_' + call['label'], tokens == call['tokens'])
        check('status_' + call['label'], raw['status'] == call['status'])
        check('unauthorized_tools_' + call['label'], not raw['unexpected_tool_types'])
    schedule = read('schedule.json', [])
    check('schedule_prefix', [(r['id'], r['arm']) for r in rows] == [(r['id'], r['arm']) for r in schedule[:len(rows)]])
    check('unique_episodes', len({(r['id'], r['arm']) for r in rows}) == len(rows))
    for row in rows:
        label = row['id'] + '-' + row['arm']
        plan = read('model/' + label + '-plan/result.json', {}).get('response') or {}
        report = read('model/' + label + '-report/result.json', {}).get('response') or {}
        evidence = read('evidence/' + label + '.json')
        subset = [c for c in calls if c['label'].startswith(label + '-')]
        check('two_call_workflow_' + label, len(subset) == row['calls'] and len(subset) <= 2)
        tokens = None if any(c['tokens'] is None for c in subset) else sum(c['tokens'] for c in subset)
        check('row_cost_' + label, tokens == row['tokens'])
        registered = {n['id'] for n in plan.get('program', {}).get('nodes', []) if n['op'] == 'periodic_evidence'}
        handles = report.get('evidence', [])
        route = bool(handles) and all(h.split('.')[0] in registered for h in handles) if row['arm'] == 'registered_required' else not registered
        if 'route_passed' in row:
            check('route_' + label, route == row['route_passed'])
        observed_pass = False
        if evidence:
            task = next(t for t in tasks if t['id'] == row['id'])
            check('input_binding_' + label, evidence['input_hash'] == digest([task['A'], task['B']]))
            check('program_binding_' + label, evidence['program_hash'] == digest(plan['program']))
            expected = next(t['expected'] for t in manifest['gate']['results'] if t['id'] == row['id'])
            try:
                observed = evidence_values(row['family'], evidence['outputs'], handles)
                observed_pass = not report['uncertain'] and equivalent(observed, expected) and equivalent(report['values'], expected)
            except (KeyError, TypeError, ValueError):
                pass
        check('independent_archived_target_' + label, bool(observed_pass) == row['passed'])
    inherited = None
    if explicit:
        parent = ROOT / 'runs/q2c-v2-interface'
        parent_read = lambda p: json.loads((parent / p).read_text(encoding='utf-8'))
        from rsi.q2c_v2_explicit.run import validate_parent
        check('parent_fixed_reused', validate_parent(tasks, parent_read('episodes.json'), parent_read('completion.json'), parent_read('model_budget.json')))
        inherited = dict(version='q2c-v2-interface', known_tokens=sum(c['tokens'] or 0 for c in parent_read('model_budget.json')['entries']),
                         science_seconds=sum(r.get('wall_seconds', 0) for r in parent_read('science_budget.json')['entries']))
    admitted = (len(rows) == (6 if explicit else 12) and completion['stop'] is None
                and all(r['completed'] and r['passed'] and r.get('route_passed') and 'tool_error' not in r for r in rows))
    check('admission_recomputed', admitted == completion['admitted'])
    arms = {}
    for name in (('registered_required',) if explicit else ('fixed', 'registered_required')):
        selected = [r for r in rows if r['arm'] == name]
        arms[name] = dict(planned=6, attempted=len(selected), completed=sum(r['completed'] for r in selected),
            passed=sum(r['passed'] for r in selected), tool_errors=sum('tool_error' in r for r in selected),
            route_passed=sum(bool(r.get('route_passed')) for r in selected),
            known_tokens=sum(r['tokens'] or 0 for r in selected),
            retained_operations=sum(r['operations'] for r in selected),
            confirmed_registered_uses=sum(r['registered_uses'] for r in selected if r['passed'] and r.get('route_passed')))
    stages = {s: dict(actions=sum(r['stage'] == s for r in science['entries']),
                     seconds=sum(r.get('wall_seconds', 0) for r in science['entries'] if r['stage'] == s))
              for s in sorted({r['stage'] for r in science['entries']})}
    summary = dict(version=manifest['version'], visibility='public_engineering_validation',
        purpose='interface_admission_not_effect_estimate', manifest=manifest, completion=completion,
        arms=arms, episodes=rows, calls=calls, known_tokens=sum(c['tokens'] or 0 for c in calls),
        unknown_usage=sum(c['tokens'] is None for c in calls),
        probe_tokens=sum(c['tokens'] or 0 for c in calls if c['stage'] == 'probe'),
        model_wall_seconds=sum(c.get('wall_seconds') or 0 for c in calls),
        science_seconds=sum(r.get('wall_seconds', 0) for r in science['entries']), science_stages=stages,
        scientific_evaluation_count=sum(r.get('evaluations', 1) for r in science['entries']),
        audit=dict(passed=True, checks=len(checks)))
    if inherited:
        summary.update(inherited_parent_cost=inherited, combined_known_tokens=summary['known_tokens'] + inherited['known_tokens'],
                       combined_science_seconds=summary['science_seconds'] + inherited['science_seconds'])
    filename = 'explicit-summary.json' if explicit else 'interface-summary.json'
    (ROOT / 'benchmarks/q2c_v2' / filename).write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    (RUN / 'audit.json').write_text(json.dumps(checks, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({k: summary[k] for k in ('completion', 'arms', 'known_tokens', 'unknown_usage', 'science_seconds', 'audit')}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--explicit', action='store_true')
    main(parser.parse_args().explicit)

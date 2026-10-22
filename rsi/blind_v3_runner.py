"""Trusted, persistent v3 execution controller. Explicit CLI commands only."""
import argparse
import json
import math
import os
from pathlib import Path
import shutil
import subprocess

from rsi import blind_campaign_v3 as core
from rsi.blind_campaign import files_at, fingerprint
from rsi.blind_campaign_v2 import accepted
from rsi.blind_calibrate import score
from rsi.blind_g1 import ROOT, read, sha, verify, write
from rsi.blind_runtime import run_container
from rsi.hpc.core import locked


def gpu(public, candidate, image, seed, cap, epochs=50):
    """Use the existing WSL Docker runtime when the controller is on Windows."""
    with locked(ROOT / 'runs/blind-g1-gpu', timeout=10800, stale_seconds=14400):
        if os.name != 'nt':
            return run_container(public, candidate, image, seed, cap, epochs)
        def linux(path):
            path = Path(path).resolve()
            return '/mnt/' + path.drive[0].lower() + path.as_posix()[2:]
        script = ('import json,sys; from pathlib import Path; '
                  'from rsi.blind_runtime import run_container; '
                  'a=json.load(sys.stdin); a[0]=Path(a[0]); a[1]=Path(a[1]); '
                  'print(json.dumps(run_container(*a)))')
        result = subprocess.run(['wsl', '-d', 'Ubuntu', '--cd', linux(ROOT), '--',
                                 'python3', '-c', script],
            input=json.dumps([linux(public), linux(candidate), image, seed, cap, epochs]),
            capture_output=True, text=True, encoding='utf-8', timeout=cap + 120, check=True)
        return json.loads(result.stdout)


def save(root, state):
    temporary = root / 'state.next.json'
    write(temporary, state)
    temporary.replace(root / 'state.json')


def prepare(root, package, calibration, probe):
    verify(package)
    report, isolation = read(calibration / 'report.json'), read(probe)
    if not report['passed'] or not isolation['passed']:
        raise ValueError('Calibration and isolation admission required')
    for record in (report, isolation):
        if record['package_sha256'] != sha(package / 'manifest.json'):
            raise ValueError('Package mismatch')
    if isolation['image_id'] != report['image_id']:
        raise ValueError('Image mismatch')
    if report['fit_wall_seconds'] > core.POLICY['fit_seconds']:
        raise ValueError('Calibration exceeds fit budget')
    if report['parent_files'] != {p.name: sha(p) for p in (calibration / 'parent').iterdir()}:
        raise ValueError('Calibrated parent changed')
    parent = files_at(calibration / 'parent')
    expected = {name: (package / 'public' / name).read_text(encoding='utf-8-sig')
                for name in ('g1_model.py', 'g1_features.py')}
    expected['train.py'] = (ROOT / 'templates/blind_g1/train.py').read_text(encoding='utf-8-sig')
    if parent != expected:
        raise ValueError('Only original calibrated G1 is admitted')
    state = core.initialize(parent)
    state.update(package=str(package.resolve()), image=report['image_id'],
                 package_sha=sha(package / 'manifest.json'), model_calls=0,
                 stopped=False, confirmation=None, call_pending=False)
    seeds = {str(r['seed']): {k: r[k] for k in ('mae', 'rmse')} for r in report['fits'][:2]}
    state['metrics'] = {'seeds': seeds, 'mean_mae': sum(r['mae'] for r in seeds.values()) / 2}
    state['policy'].update(schema='blind-g1-v3', execution_enabled=True,
                           max_prompt_bytes=2_000_000)
    state['policy_fingerprint'] = fingerprint(state['policy'])
    state['controller_hashes'] = {str(p.relative_to(ROOT)): sha(p) for p in (
        Path(__file__), ROOT / 'rsi/blind_campaign_v3.py', ROOT / 'rsi/blind_runtime.py',
        ROOT / 'rsi/blind_campaign.py', ROOT / 'rsi/blind_campaign_v2.py',
        ROOT / 'rsi/blind_calibrate.py', ROOT / 'rsi/blind_g1.py', ROOT / 'rsi/hpc/core.py',
        ROOT / 'scripts/run_blind_codex_cli.py')}
    root.mkdir(parents=True, exist_ok=False)
    save(root, state)


def load(root):
    state = read(root / 'state.json')
    if fingerprint(state['policy']) != state['policy_fingerprint']:
        raise ValueError('Frozen policy changed')
    for name, digest in state['controller_hashes'].items():
        if sha(ROOT / name) != digest:
            raise ValueError('Frozen controller changed')
    package = Path(state['package'])
    verify(package)
    if sha(package / 'manifest.json') != state['package_sha']:
        raise ValueError('Package changed')
    if fingerprint(state['sources']['g1']) != state['g1_fingerprint']:
        raise ValueError('G1 changed')
    return state


def diagnostic_input(package, output):
    """Mount only 16 training structures; no development structures or labels."""
    rows = read(package / 'public/train.json')[:16]
    output.mkdir(parents=True)
    for row in rows:
        for field in ('graph', 'cif'):
            source = (package / 'public' / row[field]).resolve()
            if not source.is_relative_to((package / 'public').resolve()):
                raise ValueError('Unsafe input path')
            dest = output / row[field]
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, dest)
    write(output / 'train.json', rows)
    write(output / 'validation.json', [{k: v for k, v in r.items() if k != 'target'} for r in rows])
    return rows


def health(result, rows):
    """Validate emitted feature arrays independently of boolean self-reports."""
    score(result.get('predictions'), rows)
    metadata = result.get('metadata') or {}
    features = metadata.get('diagnostic_features')
    expected = {r['id'] for r in rows}
    valid = isinstance(features, dict) and set(features) == expected
    widths = set()
    if valid:
        for vector in features.values():
            if not isinstance(vector, list) or not vector or any(
                    type(v) not in (int, float) or not math.isfinite(v) for v in vector):
                valid = False
                break
            widths.add(len(vector))
    return dict(train_only=True, finite_features=valid,
                forward_pass=result['status'] == 'completed',
                dimensions_valid=valid and len(widths) == 1,
                feature_coverage=1.0 if valid else 0.0)


def run_action(root, state, response, runner=gpu):
    if state['stopped'] or state['confirmation'] is not None:
        raise ValueError('Search is closed')
    if response.get('action') == 'evaluate' and state['diagnostics_used'] >= state['policy']['diagnostics']:
        raise ValueError('Diagnostic budget exhausted')
    key = core.propose(state, **response)
    if response['action'] == 'evaluate':
        state['diagnostics_used'] += 1
    save(root, state)
    candidate = root / 'candidates' / key
    candidate.mkdir(parents=True)
    for name, content in state['sources'][key].items():
        (candidate / name).write_text(content, encoding='utf-8')
    report = {}
    try:
        if response['action'] == 'stop':
            state['stopped'] = True
            core.finish(state, {})
            return
        package = Path(state['package'])
        subset = root / 'diagnostics' / key / 'input'
        rows = diagnostic_input(package, subset)
        result = runner(subset, candidate, state['image'], 17, 60, epochs=1)
        write(root / 'diagnostics' / key / 'result.json', result)
        report = health(result, rows)
        if result['status'] != 'completed' or core.health_errors(report):
            raise ValueError('Diagnostic failed: ' + json.dumps(core.health_errors(report)))
        metrics = None
        if response['action'] == 'evaluate':
            seeds = {}
            for seed in state['policy']['search_seeds']:
                result = runner(package / 'public', candidate, state['image'], seed, 360)
                write(root / 'fits' / key / f'{seed}.json', result)
                if result['status'] != 'completed':
                    raise ValueError('Training runtime failed: ' + result['status'])
                seeds[str(seed)] = score(result['predictions'], read(package / 'private/development.json'))
            metrics = {'seeds': seeds, 'mean_mae': sum(r['mae'] for r in seeds.values()) / 2}
            if accepted(metrics, state['metrics']):
                state['incumbent'], state['metrics'] = key, metrics
        core.finish(state, report, metrics=metrics)
    except (ValueError, TypeError, KeyError, OSError) as exc:
        # Only train-only runtime logs are returned to the author.
        report['error'] = str(exc)[:2000]
        if response['action'] != 'evaluate' or not report.get('forward_pass'):
            report['train_traceback'] = locals().get('result', {}).get('logs', '')[-8000:]
        core.finish(state, report, status='failed')
    finally:
        save(root, state)


def step(root, invoke, runner=gpu):
    """Reserve model calls before invocation; no invisible free retries."""
    with locked(root / 'operation', timeout=1, stale_seconds=20000):
        state = load(root)
        if state['pending'] or state['call_pending'] or state['stopped'] or state['confirmation'] is not None:
            raise ValueError('Recover pending work or search is closed')
        if state['model_calls'] >= state['policy']['decisions']:
            raise ValueError('Model decision budget exhausted')
        payload = core.packet(state)
        payload['call_failures'] = state.get('call_failures', [])
        payload['task'] = (Path(state['package']) / 'public/TASK.md').read_text(encoding='utf-8')
        payload['interface'] = 'python train.py --input /input --output /output --seed INT --epochs INT'
        payload['response_contract'] = {'action': 'diagnose|repair|evaluate|stop', 'base': 'source ID',
            'hypothesis': 'Full rationale', 'files': 'Complete Python source mapping including train.py'}
        payload['diagnostics_contract'] = ('For epochs=1 emit metadata.json with diagnostic_features: '
            'mapping each validation ID to the actual finite nonempty feature vector used by the model. '
            'Emit predictions.json for all validation IDs. No fallback or fabricated diagnostic arrays. '
            'All calls use the same input layout as G1; validation records never include targets.')
        prompt = 'Use no tools. Return exactly one JSON object.\n' + json.dumps(payload)
        if len(prompt.encode()) > state['policy']['max_prompt_bytes']:
            raise ValueError('Context budget exceeded; no automatic history truncation')
        state['model_calls'] += 1
        state['call_pending'] = True
        save(root, state)
        directory = root / 'calls' / str(state['model_calls'])
        try:
            record, text = invoke(prompt, directory, state['policy']['proposal_seconds'])
            if not record.get('passed_transport'):
                raise ValueError('CLI transport admission failed')
            response = json.loads(text)
            if not isinstance(response, dict):
                raise ValueError('Response must be an object')
            run_action(root, state, response, runner)
        except (ValueError, TypeError, OSError) as exc:
            state.setdefault('call_failures', []).append(
                {'call': state['model_calls'], 'error': str(exc)[:2000]})
            raise
        finally:
            state['call_pending'] = False
            save(root, state)


def confirm(root, runner=gpu):
    with locked(root / 'operation', timeout=1, stale_seconds=20000):
        state = load(root)
        if state['pending'] or state['call_pending'] or state['confirmation'] is not None:
            raise ValueError('Pending work or confirmation already reserved')
        state['stopped'] = True
        state['confirmation'] = {'status': 'running', 'selected': state['incumbent'], 'fits': []}
        save(root, state)
        try:
            for key in ('g1', state['incumbent']):
                candidate = root / 'confirmation_sources' / key
                candidate.mkdir(parents=True, exist_ok=True)
                for name, content in state['sources'][key].items():
                    (candidate / name).write_text(content, encoding='utf-8')
                for seed in state['policy']['confirmation_seeds']:
                    result = runner(Path(state['package']) / 'public', candidate, state['image'], seed, 360)
                    index = len(state['confirmation']['fits'])
                    write(root / 'confirmation' / f'{index}.json', result)
                    if result['status'] != 'completed':
                        raise ValueError('Confirmation fit failed')
                    metrics = score(result['predictions'], read(Path(state['package']) / 'private/development.json'))
                    state['confirmation']['fits'].append(dict(source=key, seed=seed, **metrics))
                    save(root, state)
            state['confirmation']['status'] = 'completed'
        finally:
            if state['confirmation']['status'] == 'running':
                state['confirmation']['status'] = 'interrupted'
            save(root, state)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'status', 'step', 'recover', 'confirm'])
    parser.add_argument('--campaign', type=Path, required=True)
    for name in ('package', 'calibration', 'probe'):
        parser.add_argument('--' + name, type=Path)
    args = parser.parse_args()
    root = args.campaign.resolve()
    if args.action == 'prepare':
        prepare(root, args.package, args.calibration, args.probe)
    elif args.action == 'step':
        from scripts.run_blind_codex_cli import invoke
        step(root, invoke)
    elif args.action == 'confirm':
        confirm(root)
    elif args.action == 'recover':
        with locked(root / 'operation', timeout=1, stale_seconds=20000):
            state = load(root)
            if state['pending']:
                core.finish(state, {'error': 'Interrupted; reservation retained'}, status='interrupted')
            if state['confirmation'] and state['confirmation']['status'] == 'running':
                state['confirmation']['status'] = 'interrupted'
            state['call_pending'] = False
            save(root, state)
    print(json.dumps(load(root), indent=2))


if __name__ == '__main__':
    main()

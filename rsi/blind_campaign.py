"""Provider-neutral blind GNN campaign controller; run only on the trusted host."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import time

from rsi.blind_g1 import ROOT, read, sha, verify, write
from rsi.blind_calibrate import score
from rsi.blind_runtime import run_container
from rsi.hpc.core import locked

ARMS = ('independent', 'rsi')
PROVIDERS = ('chatgpt', 'claude')
MAX_SOURCE_BYTES = 128000


def files_at(path):
    return {p.name: p.read_text(encoding='utf-8-sig') for p in sorted(path.glob('*.py'))}


def validate_files(files):
    if not isinstance(files, dict) or 'train.py' not in files or not 1 <= len(files) <= 16:
        raise ValueError('Submit 1-16 Python files including train.py')
    if any(not isinstance(v, str) for v in files.values()) or sum(len(v.encode()) for v in files.values()) > MAX_SOURCE_BYTES:
        raise ValueError('Source budget or type invalid')
    for name, content in files.items():
        if not re.fullmatch(r'[A-Za-z][A-Za-z0-9_]{0,63}\.py', name) or not isinstance(content, str):
            raise ValueError('Invalid source file')
        compile(content, name, 'exec')


def fingerprint(files):
    return hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()


def accepted(candidate, parent):
    return (candidate['mean_mae'] <= .98 * parent['mean_mae'] and
            all(candidate['seeds'][seed]['mae'] <= 1.05 * parent['seeds'][seed]['mae'] for seed in ('17', '29')))


def initialize(package, calibration, output, probe):
    if output.exists():
        raise FileExistsError('Campaign directory already exists')
    verify(package)
    report = read(calibration / 'report.json')
    if not report['passed'] or report['package_sha256'] != sha(package / 'manifest.json'):
        raise ValueError('Calibration not admitted for this package')
    isolation = read(probe)
    if (not isolation['passed'] or isolation['image_id'] != report['image_id'] or
            isolation['package_sha256'] != report['package_sha256']):
        raise ValueError('Runtime probe does not admit this image and package')
    parent = files_at(calibration / 'parent')
    validate_files(parent)
    if {p.name: sha(p) for p in (calibration / 'parent').iterdir()} != report['parent_files']:
        raise ValueError('Calibrated source changed')
    seeds = {str(row['seed']): {k: row[k] for k in ('mae', 'rmse')} for row in report['fits'][:2]}
    baseline = {'seeds': seeds, 'mean_mae': sum(v['mae'] for v in seeds.values()) / 2}
    output.mkdir(parents=True)
    write(output / 'parent.json', parent)
    policy = {'schema': 'blind-g1-campaign-v2', 'package': str(package.resolve()),
              'package_sha256': sha(package / 'manifest.json'), 'image_id': report['image_id'],
              'parent_sha256': sha(output / 'parent.json'), 'baseline': baseline,
              'fit_wall_seconds': report['fit_wall_seconds'], 'epochs': 50,
              'search_seeds': [17, 29], 'confirmation_seeds': [101, 202, 303],
              'rounds': 3, 'maximum_formal_fits_per_provider': 21,
              'proposal_wall_seconds': 1800, 'proposal_source_bytes': MAX_SOURCE_BYTES,
              'transport': 'fresh-session-file-exchange', 'equal_token_budget_claim': False,
              'runtime_probe_sha256': sha(probe), 'calibration_sha256': sha(calibration / 'report.json'),
              'development_execution_admitted': True, 'app_blindness_requires_operator_attestation': True,
              'final_test_enabled': False,
              'controller_hashes': {name: sha(ROOT / 'rsi' / name) for name in
                  ('blind_g1.py', 'blind_calibrate.py', 'blind_campaign.py', 'blind_runtime.py', 'hpc/core.py')}}
    write(output / 'policy.json', policy)
    write(output / 'ledger.json', {'policy_sha256': sha(output / 'policy.json'), 'providers': {}})
    return policy


def load(campaign):
    policy = read(campaign / 'policy.json')
    ledger = read(campaign / 'ledger.json')
    if sha(campaign / 'policy.json') != ledger['policy_sha256'] or sha(campaign / 'parent.json') != policy['parent_sha256']:
        raise ValueError('Campaign binding changed')
    for name, expected in policy['controller_hashes'].items():
        if sha(ROOT / 'rsi' / name) != expected:
            raise ValueError('Controller changed after campaign freeze')
    package = Path(policy['package'])
    verify(package)
    if sha(package / 'manifest.json') != policy['package_sha256']:
        raise ValueError('Package binding changed')
    return policy, ledger


def register(campaign, provider, model, transport):
    if provider not in PROVIDERS or not model.strip() or len(model) > 160:
        raise ValueError('Record the actual provider and model identifier')
    if transport not in ('app-file-exchange', 'cli-file-exchange'):
        raise ValueError('Only file exchange is admitted')
    with locked(campaign / 'ledger.json'):
        policy, ledger = load(campaign)
        if provider in ledger['providers']:
            raise ValueError('Provider already registered; do not overwrite attempts')
        ledger['providers'][provider] = {'model': model, 'transport': transport,
             'arms': {arm: {'slots': [], 'incumbent': 'parent', 'metrics': policy['baseline']} for arm in ARMS}}
        write(campaign / 'ledger.json', ledger)


def incumbent_files(campaign, provider, arm_state):
    if arm_state['incumbent'] == 'parent':
        return read(campaign / 'parent.json')
    return read(campaign / 'submissions' / provider / (arm_state['incumbent'] + '.json'))['files']


def packet(campaign, provider, arm, reserve=True):
    with locked(campaign / 'ledger.json'):
        policy, ledger = load(campaign)
        state = ledger['providers'][provider]['arms'][arm]
        if any(s['status'] in ('awaiting', 'running') for s in state['slots']) or len(state['slots']) >= 3:
            raise ValueError('Finish the current slot or arm budget exhausted')
        generation = len(state['slots']) + 1
        sources = read(campaign / 'parent.json') if arm == 'independent' else incumbent_files(campaign, provider, state)
        history = [] if arm == 'independent' else [{k: s[k] for k in ('generation', 'status', 'feedback') if k in s} for s in state['slots']]
        packet_id = hashlib.sha256(f'{ledger["policy_sha256"]}:{provider}:{arm}:{generation}'.encode()).hexdigest()
        content = {'packet_id': packet_id, 'task': (Path(policy['package']) / 'public/TASK.md').read_text(),
                   'interface': 'python /candidate/train.py --input /input --output /output --seed INTEGER --epochs 50',
                   'input_contract': {'train.json': 'List of id, graph, cif, target records',
                       'validation.json': 'List of id, graph, cif records; no labels',
                       'paths': 'Relative to /input; NPZ allow_pickle=False; CIF contains raw structure',
                       'output': '/output/predictions.json: object mapping every validation ID to a finite number'},
                   'budget': {k: policy[k] for k in ('fit_wall_seconds', 'epochs', 'search_seeds', 'proposal_wall_seconds', 'proposal_source_bytes')},
                   'environment': {'python': '3.11', 'torch': '2.8.0+cu128', 'numpy': '2.2.6', 'ase': '3.26.0',
                                   'networkx': '3.5', 'scipy': '1.15.3', 'cpu': 4, 'ram_gb': 8, 'gpu': 'RTX 4090'},
                   'source_files': sources, 'history': history,
                   'current_metrics': policy['baseline'] if arm == 'independent' else state['metrics'],
                   'response_contract': {'packet_id': packet_id, 'hypothesis': 'Explain your proposed change in <= 2000 characters',
                                         'files': 'Complete replacement mapping filename.py -> source text; include train.py',
                                         'session': {'fresh_context': True, 'tools_disabled': True, 'external_materials_used': False,
                                                     'model': ledger['providers'][provider]['model'], 'usage': None}},
                   'instructions': 'Use a new conversation containing only this packet. Do not browse, use connectors, read other files, or run host commands. Return one JSON object, without Markdown fences. No extra datasets, pretrained models, hardcoded validation predictions, or development-label queries. The controller executes all code. Submit a single candidate, not a search over multiple configurations. Training-only diagnostics are allowed inside the fixed compute cap. Independent proposals receive no earlier outcomes. Do not request historical experiments.'}
        if reserve:
            path = campaign / 'outbox' / provider / f'{arm}-{generation}.json'
            write(path, content)
            state['slots'].append({'generation': generation, 'status': 'awaiting', 'packet_id': packet_id,
                                   'started_at': time.time(), 'packet_sha256': sha(path)})
            write(campaign / 'ledger.json', ledger)
        return content


def submit(campaign, provider, arm, submission):
    with locked(campaign / f'{provider}-{arm}.active', timeout=1, stale_seconds=14400):
        return _submit(campaign, provider, arm, submission)


def _submit(campaign, provider, arm, submission):
    # Reserve the slot before running candidate code; failures consume it.
    with locked(campaign / 'ledger.json'):
        policy, ledger = load(campaign)
        state = ledger['providers'][provider]['arms'][arm]
        if not state['slots'] or state['slots'][-1]['status'] != 'awaiting':
            raise ValueError('No pending slot')
        slot = state['slots'][-1]
        generation = slot['generation']
        slot['status'] = 'running'
        write(campaign / 'ledger.json', ledger)
    feedback = {'status': 'failed', 'reason': 'invalid_submission'}
    try:
        if time.time() - slot['started_at'] > policy['proposal_wall_seconds']:
            raise ValueError('Proposal deadline exceeded')
        if submission.stat().st_size > 200000:
            raise ValueError('Submission too large')
        response = read(submission)
        if set(response) != {'packet_id', 'hypothesis', 'files', 'session'} or response['packet_id'] != slot['packet_id']:
            raise ValueError('Submission contract mismatch')
        if not isinstance(response['hypothesis'], str) or not 1 <= len(response['hypothesis']) <= 2000:
            raise ValueError('Hypothesis length invalid')
        session = response['session']
        if (session.get('fresh_context') is not True or session.get('tools_disabled') is not True or
                session.get('external_materials_used') is not False or session.get('model') != ledger['providers'][provider]['model']):
            raise ValueError('Fresh tool-free session attestation required')
        validate_files(response['files'])
        key = f'{arm}-{generation}'
        write(campaign / 'submissions' / provider / (key + '.json'), response)
        candidate = campaign / 'candidates' / provider / key
        candidate.mkdir(parents=True)
        for name, source in response['files'].items():
            (candidate / name).write_text(source, encoding='utf-8')
        results = {}
        reference = read(Path(policy['package']) / 'private/development.json')
        with locked(ROOT / 'runs/blind-g1-gpu', timeout=10800, stale_seconds=14400):
            for seed in policy['search_seeds']:
                result = run_container(Path(policy['package']) / 'public', candidate, policy['image_id'], seed, policy['fit_wall_seconds'])
                write(campaign / 'runs' / provider / key / f'{seed}.json', result)
                if result['status'] != 'completed':
                    raise ValueError('Candidate runtime failed: ' + result['status'])
                results[str(seed)] = score(result['predictions'], reference)
        metrics = {'seeds': results, 'mean_mae': sum(v['mae'] for v in results.values()) / 2}
        reference_metrics = policy['baseline'] if arm == 'independent' else state['metrics']
        promoted = accepted(metrics, reference_metrics)
        feedback = {'status': 'completed', 'metrics': metrics, 'accepted': promoted}
        # Independent trials always restart from G1; the best accepted one is retained only for confirmation.
        if promoted and (arm == 'rsi' or metrics['mean_mae'] < state['metrics']['mean_mae']):
            state['incumbent'], state['metrics'] = key, metrics
    except (ValueError, SyntaxError, OSError) as exc:
        feedback = {'status': 'failed', 'reason': str(exc)[:300]}
    finally:
        with locked(campaign / 'ledger.json'):
            _, latest = load(campaign)
            current = latest['providers'][provider]['arms'][arm]
            current['slots'][-1].update(status=feedback['status'], feedback=feedback)
            current['incumbent'], current['metrics'] = state['incumbent'], state['metrics']
            write(campaign / 'ledger.json', latest)
            write(campaign / 'feedback' / provider / f'{arm}-{generation}.json', feedback)
    return feedback


def recover(campaign, provider, arm):
    """Close an interrupted slot without refunding its budget."""
    with locked(campaign / f'{provider}-{arm}.active', timeout=1, stale_seconds=14400):
        return _recover(campaign, provider, arm)


def _recover(campaign, provider, arm):
    with locked(campaign / 'ledger.json'):
        _, ledger = load(campaign)
        slots = ledger['providers'][provider]['arms'][arm]['slots']
        if not slots or slots[-1]['status'] not in ('awaiting', 'running'):
            raise ValueError('No unfinished slot')
        slots[-1].update(status='failed', feedback={'status': 'failed', 'reason': 'operator_closed_interrupted_slot'})
        write(campaign / 'ledger.json', ledger)


def confirm(campaign, provider):
    """Lock selection, then evaluate three fresh seeds without further proposals."""
    with locked(campaign / 'ledger.json'):
        policy, ledger = load(campaign)
        state = ledger['providers'][provider]
        if any(len(a['slots']) != 3 or any(s['status'] in ('awaiting', 'running') for s in a['slots']) for a in state['arms'].values()):
            raise ValueError('Complete both search arms first')
        if 'confirmation' in state:
            raise ValueError('Confirmation already reserved; no automatic rerun')
        selected = {'parent': read(campaign / 'parent.json')}
        selected.update({arm: incumbent_files(campaign, provider, state['arms'][arm]) for arm in ARMS})
        selection = {name: fingerprint(files) for name, files in selected.items()}
        state['confirmation'] = {'status': 'running', 'selection': selection, 'fits_reserved': 9}
        write(campaign / 'ledger.json', ledger)
        write(campaign / 'confirmation' / provider / 'selection.json', selection)
    results = []
    reference = read(Path(policy['package']) / 'private/development.json')
    try:
        with locked(ROOT / 'runs/blind-g1-gpu', timeout=10800, stale_seconds=14400):
            for name, files in selected.items():
                directory = campaign / 'confirmation' / provider / name
                directory.mkdir(parents=True)
                for filename, text in files.items():
                    (directory / filename).write_text(text, encoding='utf-8')
                for seed in policy['confirmation_seeds']:
                    result = run_container(Path(policy['package']) / 'public', directory, policy['image_id'], seed, policy['fit_wall_seconds'])
                    write(directory / f'{seed}.json', result)
                    if result['status'] != 'completed':
                        raise ValueError('Confirmation fit failed; reserved budget is not refunded')
                    results.append({'configuration': name, 'seed': seed, **score(result['predictions'], reference),
                                    'wall_seconds': result['wall_seconds']})
        summary = {'status': 'completed', 'rows': results, 'heldout_evaluated': False,
                   'scope': 'fresh-seed development confirmation; no unseen-task claim',
                   'aggregate': {name: sum(r['mae'] for r in results if r['configuration'] == name)/3 for name in selected}}
    except Exception as exc:
        summary = {'status': 'failed', 'rows': results, 'reason': str(exc)[:300], 'heldout_evaluated': False}
    with locked(campaign / 'ledger.json'):
        _, latest = load(campaign)
        latest['providers'][provider]['confirmation'].update(summary)
        write(campaign / 'ledger.json', latest)
        write(campaign / 'confirmation' / provider / 'report.json', summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['init', 'register', 'export', 'preview', 'submit', 'status', 'recover', 'confirm'])
    parser.add_argument('--campaign', type=Path, required=True)
    parser.add_argument('--package', type=Path)
    parser.add_argument('--calibration', type=Path)
    parser.add_argument('--probe', type=Path)
    parser.add_argument('--provider', choices=PROVIDERS)
    parser.add_argument('--arm', choices=ARMS)
    parser.add_argument('--model')
    parser.add_argument('--transport', default='app-file-exchange')
    parser.add_argument('--submission', type=Path)
    args = parser.parse_args()
    if args.action == 'init':
        result = initialize(args.package, args.calibration, args.campaign, args.probe)
    elif args.action == 'register':
        result = register(args.campaign, args.provider, args.model, args.transport)
    elif args.action in ('export', 'preview'):
        result = packet(args.campaign, args.provider, args.arm, reserve=args.action == 'export')
    elif args.action == 'submit':
        result = submit(args.campaign, args.provider, args.arm, args.submission)
    elif args.action == 'recover':
        result = recover(args.campaign, args.provider, args.arm)
    elif args.action == 'confirm':
        result = confirm(args.campaign, args.provider)
    else:
        result = load(args.campaign)[1]
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()

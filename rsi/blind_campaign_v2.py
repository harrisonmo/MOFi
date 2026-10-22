"""Blind-G1 campaign controller, second version; run only on the trusted host.

Differences from rsi.blind_campaign (which stays frozen for campaign 01):
- Larger time caps: 2700 s per proposal and a 360 s fit cap; three slots per arm as before.
- A free pre-submission format check. It runs no candidate code, reveals no
  scores and allows one repair per slot, so slots are spent on science rather
  than on formatting mistakes.
- A soft hypothesis limit: text beyond 2000 characters is truncated and the
  truncation is recorded, instead of rejecting the whole candidate.
- The controller records the registered model itself; the session model field
  is optional free text kept as a self-report.
- Promotion needs a 1% mean MAE improvement (v1: 2%), with the same <=5%
  regression guard on either seed, so smaller gains can accumulate.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

from rsi.blind_g1 import ROOT, read, sha, verify, write
from rsi.blind_calibrate import score
from rsi.blind_campaign import ARMS, PROVIDERS, MAX_SOURCE_BYTES, files_at, fingerprint, validate_files
from rsi.blind_runtime import run_container
from rsi.hpc.core import locked

ROUNDS = 3
PROPOSAL_SECONDS = 2700
FIT_SECONDS = 360
HYPOTHESIS_CHARS = 2000
CHECKS_PER_SLOT = 2
PROMOTION_MARGIN = .01
REGRESSION_LIMIT = .05
CONTROLLER_FILES = ('blind_g1.py', 'blind_calibrate.py', 'blind_campaign.py', 'blind_campaign_v2.py',
                    'blind_runtime.py', 'hpc/core.py')


def accepted(candidate, parent, margin=PROMOTION_MARGIN):
    return (candidate['mean_mae'] <= (1 - margin) * parent['mean_mae'] and
            all(candidate['seeds'][seed]['mae'] <= (1 + REGRESSION_LIMIT) * parent['seeds'][seed]['mae']
                for seed in ('17', '29')))


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
    if FIT_SECONDS < report['fit_wall_seconds']:
        raise ValueError('Fit cap may not be below the calibrated cap')
    output.mkdir(parents=True)
    write(output / 'parent.json', parent)
    policy = {'schema': 'blind-g1-campaign-02', 'package': str(package.resolve()),
              'package_sha256': sha(package / 'manifest.json'), 'image_id': report['image_id'],
              'parent_sha256': sha(output / 'parent.json'), 'baseline': baseline,
              'calibrated_fit_wall_seconds': report['fit_wall_seconds'], 'fit_wall_seconds': FIT_SECONDS,
              'epochs': 50, 'search_seeds': [17, 29], 'confirmation_seeds': [101, 202, 303], 'rounds': ROUNDS,
              'maximum_formal_fits_per_provider': ROUNDS * 2 * 2 + 9,
              'proposal_wall_seconds': PROPOSAL_SECONDS, 'proposal_source_bytes': MAX_SOURCE_BYTES,
              'hypothesis_soft_chars': HYPOTHESIS_CHARS, 'format_checks_per_slot': CHECKS_PER_SLOT,
              'promotion_margin': PROMOTION_MARGIN, 'regression_limit': REGRESSION_LIMIT,
              'transport': 'fresh-session-file-exchange', 'equal_token_budget_claim': False,
              'runtime_probe_sha256': sha(probe), 'calibration_sha256': sha(calibration / 'report.json'),
              'development_execution_admitted': True, 'app_blindness_requires_operator_attestation': True,
              'final_test_enabled': False,
              'controller_hashes': {name: sha(ROOT / 'rsi' / name) for name in CONTROLLER_FILES}}
    write(output / 'policy.json', policy)
    write(output / 'ledger.json', {'policy_sha256': sha(output / 'policy.json'), 'providers': {}})
    return policy


def load(campaign):
    policy = read(campaign / 'policy.json')
    ledger = read(campaign / 'ledger.json')
    if policy.get('schema') != 'blind-g1-campaign-02':
        raise ValueError('Not a second-version campaign')
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
        if any(s['status'] in ('awaiting', 'running') for s in state['slots']) or len(state['slots']) >= policy['rounds']:
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
                   'budget': {k: policy[k] for k in ('fit_wall_seconds', 'epochs', 'search_seeds', 'proposal_wall_seconds',
                                                     'proposal_source_bytes', 'rounds')},
                   'environment': {'python': '3.11', 'torch': '2.8.0+cu128', 'numpy': '2.2.6', 'ase': '3.26.0',
                                   'networkx': '3.5', 'scipy': '1.15.3', 'cpu': 4, 'ram_gb': 8, 'gpu': 'RTX 4090'},
                   'source_files': sources, 'history': history,
                   'current_metrics': policy['baseline'] if arm == 'independent' else state['metrics'],
                   'response_contract': {'packet_id': packet_id,
                                         'hypothesis': f'Explain your proposed change; aim for at most {policy["hypothesis_soft_chars"]} characters. Longer text is truncated, not rejected.',
                                         'files': 'Complete replacement mapping filename.py -> source text; include train.py',
                                         'session': {'fresh_context': True, 'tools_disabled': True, 'external_materials_used': False,
                                                     'model': 'Optional free text; the controller records the registered model', 'usage': None}},
                   'instructions': ('Use a new conversation containing only this packet. Do not browse, use connectors, read other files, '
                                    'or run host commands. Return one JSON object, without Markdown fences. Before execution the controller '
                                    'runs a free format check (no code is run and no score is revealed); if it fails you may receive the '
                                    'errors once and return a corrected object. No extra datasets, pretrained models, hardcoded validation '
                                    'predictions, or development-label queries. Submit a single candidate, not a search over multiple '
                                    'configurations. Training-only diagnostics are allowed inside the fixed compute cap. Independent '
                                    'proposals receive no earlier outcomes. Do not request historical experiments.')}
        if reserve:
            path = campaign / 'outbox' / provider / f'{arm}-{generation}.json'
            write(path, content)
            state['slots'].append({'generation': generation, 'status': 'awaiting', 'packet_id': packet_id,
                                   'started_at': time.time(), 'packet_sha256': sha(path), 'format_checks': 0})
            write(campaign / 'ledger.json', ledger)
        return content


def contract_errors(response, packet_id):
    """Return format errors without executing candidate code."""
    if not isinstance(response, dict):
        return ['Response must be one JSON object']
    errors = []
    if set(response) != {'packet_id', 'hypothesis', 'files', 'session'}:
        errors.append('Top-level keys must be exactly packet_id, hypothesis, files, session')
    if response.get('packet_id') != packet_id:
        errors.append('packet_id does not match the packet')
    if not isinstance(response.get('hypothesis'), str) or not response.get('hypothesis', '').strip():
        errors.append('hypothesis must be a non-empty string')
    session = response.get('session')
    if (not isinstance(session, dict) or session.get('fresh_context') is not True or
            session.get('tools_disabled') is not True or session.get('external_materials_used') is not False):
        errors.append('session must attest fresh_context=true, tools_disabled=true, external_materials_used=false')
    try:
        validate_files(response.get('files'))
    except SyntaxError as error:
        errors.append(f'Syntax error in {error.filename} line {error.lineno}: {error.msg}')
    except (ValueError, TypeError) as error:
        errors.append(str(error))
    return errors


def validate_submission(submission, packet_id):
    """Parse and validate a response file; return (response, errors, warnings) without raising on bad content.

    JSON null, arrays, strings and numbers parse successfully but are not responses,
    so validity is decided only by the error list, never by the parsed value.
    """
    if submission.stat().st_size > 200000:
        return None, ['Submission larger than 200000 bytes'], []
    try:
        response = read(submission)
    except (ValueError, UnicodeDecodeError, RecursionError) as error:
        return None, [f'Not valid JSON: {str(error)[:200]}'], []
    errors = contract_errors(response, packet_id)
    warnings = []
    if isinstance(response, dict) and isinstance(response.get('hypothesis'), str) and len(response['hypothesis']) > HYPOTHESIS_CHARS:
        warnings.append(f'hypothesis has {len(response["hypothesis"])} characters and will be truncated to {HYPOTHESIS_CHARS}')
    return response, errors, warnings


def check(campaign, provider, arm, submission):
    """Free format check of a draft response for the current slot; runs no candidate code."""
    with locked(campaign / 'ledger.json'):
        policy, ledger = load(campaign)
        state = ledger['providers'][provider]['arms'][arm]
        if not state['slots'] or state['slots'][-1]['status'] != 'awaiting':
            raise ValueError('No pending slot')
        slot = state['slots'][-1]
        if slot['format_checks'] >= policy['format_checks_per_slot']:
            raise ValueError('Format check allowance used')
        if time.time() - slot['started_at'] > policy['proposal_wall_seconds']:
            raise ValueError('Proposal deadline exceeded')
        slot['format_checks'] += 1
        _, errors, warnings = validate_submission(submission, slot['packet_id'])
        result = {'valid': not errors, 'errors': errors, 'warnings': warnings,
                  'checks_remaining': policy['format_checks_per_slot'] - slot['format_checks']}
        write(campaign / 'checks' / provider / f'{arm}-{slot["generation"]}-{slot["format_checks"]}.json',
              {**result, 'submission_sha256': sha(submission)})
        write(campaign / 'ledger.json', ledger)
    return result


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
    record = {}
    try:
        if time.time() - slot['started_at'] > policy['proposal_wall_seconds']:
            raise ValueError('Proposal deadline exceeded')
        response, errors, _ = validate_submission(submission, slot['packet_id'])
        if errors:
            raise ValueError('Invalid submission: ' + '; '.join(errors))
        hypothesis = response['hypothesis']
        record = {'hypothesis_chars': len(hypothesis), 'hypothesis_truncated': len(hypothesis) > policy['hypothesis_soft_chars'],
                  'self_reported_model': response['session'].get('model'),
                  'registered_model': ledger['providers'][provider]['model']}
        response = {**response, 'hypothesis': hypothesis[:policy['hypothesis_soft_chars']]}
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
        promoted = accepted(metrics, reference_metrics, policy['promotion_margin'])
        feedback = {'status': 'completed', 'metrics': metrics, 'accepted': promoted}
        # Independent trials always restart from G1; the best accepted one is retained only for confirmation.
        if promoted and (arm == 'rsi' or metrics['mean_mae'] < state['metrics']['mean_mae']):
            state['incumbent'], state['metrics'] = key, metrics
    # Malformed content must end as a recorded failure, never as an uncaught exception.
    except (ValueError, SyntaxError, OSError, TypeError, KeyError, AttributeError) as exc:
        feedback = {'status': 'failed', 'reason': str(exc)[:300]}
    finally:
        with locked(campaign / 'ledger.json'):
            _, latest = load(campaign)
            current = latest['providers'][provider]['arms'][arm]
            current['slots'][-1].update(status=feedback['status'], feedback=feedback, **record)
            current['incumbent'], current['metrics'] = state['incumbent'], state['metrics']
            write(campaign / 'ledger.json', latest)
            write(campaign / 'feedback' / provider / f'{arm}-{generation}.json', feedback)
    return feedback


def recover(campaign, provider, arm):
    """Close an interrupted slot without refunding its budget."""
    with locked(campaign / f'{provider}-{arm}.active', timeout=1, stale_seconds=14400):
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
        if any(len(a['slots']) != policy['rounds'] or any(s['status'] in ('awaiting', 'running') for s in a['slots'])
               for a in state['arms'].values()):
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
                   'aggregate': {name: sum(r['mae'] for r in results if r['configuration'] == name) / 3 for name in selected}}
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
    parser.add_argument('action', choices=['init', 'register', 'export', 'preview', 'check', 'submit', 'status', 'recover', 'confirm'])
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
    elif args.action == 'check':
        result = check(args.campaign, args.provider, args.arm, args.submission)
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

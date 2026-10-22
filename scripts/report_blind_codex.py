"""Summarize the ChatGPT pilot without changing its selection or budgets."""
import json
from pathlib import Path
import statistics
from rsi.blind_g1 import ROOT, read, sha, write
from rsi.blind_campaign import fingerprint


def main():
    campaign = ROOT / 'runs/blind-g1-campaign-01'
    ledger = read(campaign / 'ledger.json')
    policy = read(campaign / 'policy.json')
    state = ledger['providers']['chatgpt']
    calls = ROOT / 'runs/blind-codex-sol-high'
    rows = []
    for arm, value in state['arms'].items():
        for slot in value['slots']:
            generation = slot['generation']
            record_path = calls / f'{arm}-{generation}/call.json'
            record = read(record_path) if record_path.exists() else {}
            row = {'arm': arm, 'generation': generation, 'status': slot['status'],
                   'feedback': slot.get('feedback'), 'proposal_seconds': record.get('seconds'),
                   'usage': record.get('usage'), 'unexpected_tool_types': record.get('unexpected_item_types')}
            submission = campaign / 'submissions/chatgpt' / f'{arm}-{generation}.json'
            if submission.exists():
                value = read(submission)
                row['hypothesis'] = value['hypothesis']
                row['source_sha256'] = fingerprint(value['files'])
                row['submission_sha256'] = sha(submission)
            fits = []
            directory = campaign / 'runs/chatgpt' / f'{arm}-{generation}'
            if directory.exists():
                for file in sorted(directory.glob('*.json')):
                    result = read(file)
                    fits.append({'seed': result['seed'], 'status': result['status'], 'seconds': result['wall_seconds']})
            row['fits'] = fits
            rows.append(row)
    confirmation = state.get('confirmation')
    report = {'schema': 'blind-g1-sol-pilot-v1', 'model': state['model'], 'reasoning_effort': 'high',
              'policy_sha256': sha(campaign / 'policy.json'), 'baseline_search_mae': policy['baseline']['mean_mae'],
              'fit_wall_seconds': policy['fit_wall_seconds'], 'maximum_fits': 21, 'rows': rows,
              'incumbents': {a: {'candidate': v['incumbent'], 'mean_mae': v['metrics']['mean_mae']} for a,v in state['arms'].items()},
              'confirmation': confirmation, 'heldout_evaluated': False,
              'scope': 'Single evolutionary run; fresh-seed development confirmation, not unseen-task generalization.'}
    if confirmation and confirmation['status'] == 'completed':
        by_key = {(r['configuration'], r['seed']):r['mae'] for r in confirmation['rows']}
        differences = [by_key['rsi', seed]-by_key['independent', seed] for seed in policy['confirmation_seeds']]
        report['paired_rsi_minus_independent_mae'] = {'seeds': policy['confirmation_seeds'], 'differences': differences,
             'mean': statistics.mean(differences), 'sample_std': statistics.stdev(differences),
             'interpretation': 'Negative values favor RSI; seed replicates do not replicate the evolution process.'}
    write(ROOT / 'docs/reports/BLIND_G1_SOL_RESULT.json', report)
    lines = ['# Blind-G1 GPT-6.1 Sol high pilot', '',
             'Frozen data, evaluator and compute budget; independent reset versus recursive code inheritance.', '',
             '| Arm | Round | Status | Mean MAE | Accepted |', '|---|---:|---|---:|---|']
    for row in rows:
        feedback = row['feedback'] or {}
        mean = feedback.get('metrics',{}).get('mean_mae')
        lines.append(f"| {row['arm']} | {row['generation']} | {row['status']} | {mean if mean is not None else '-'} | {feedback.get('accepted','-')} |")
    lines += ['', f"Search parent G1-F50 MAE: {policy['baseline']['mean_mae']:.6f} mmol/g.", '']
    if confirmation:
        lines += ['Confirmation status: ' + confirmation['status'] + '.', '']
        if confirmation['status'] == 'completed':
            lines += ['| Configuration | Three fresh-seed mean MAE |', '|---|---:|']
            for name, value in confirmation['aggregate'].items():
                lines.append(f'| {name} | {value:.6f} |')
    lines += ['', 'This is one evolutionary pilot. Three training seeds measure training variability, not independent repetitions of self-improvement. Heldout evaluation was not performed. No unseen-task generalization or matched human-versus-RSI conclusion follows from this pilot.', '',
              'The first model preflight used an old CLI by mistake; a second probe succeeded with the App-bundled CLI. See [version diagnosis](BLIND_G1_CLI_VERSION.md). The host disables executable I/O and audits tool events, while mandatory nonfunctional framework wrappers remain listed. Actual backend model metadata was not supplied by CLI events; the requested model, high setting and exact executable are retained in each raw call record.', '',
              'Raw source, hypotheses, predictions, usage and container records remain ignored under `runs/blind-codex-sol-high` and `runs/blind-g1-campaign-01`. The machine-readable report is [here](BLIND_G1_SOL_RESULT.json).']
    (ROOT / 'docs/reports/BLIND_G1_SOL_RESULT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({'rows':len(rows),'confirmation_status':confirmation.get('status') if confirmation else None}))

if __name__ == '__main__':
    main()

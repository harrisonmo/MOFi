"""Chain Blind-G1 v2 campaigns: the confirmed champion of one campaign becomes the next parent.

The frozen v2 controller initializes a campaign from a calibration record
(parent sources plus seed-17/29 metrics). This module writes such a record for
a finished campaign's champion, so the controller itself never changes. The
champion is the configuration with the lowest fresh-seed confirmation MAE;
the lineage continues only if it beats that campaign's parent by at least 1%
(the user's choice; promotion inside a campaign keeps the 2% margin).
"""
import argparse
import json
from pathlib import Path

from rsi.blind_g1 import read, sha, write
from rsi import blind_campaign_v2 as controller

IMPROVEMENT = .01


def champion(campaign, provider):
    policy, ledger = controller.load(campaign)
    state = ledger['providers'][provider]
    confirmation = state.get('confirmation', {})
    if confirmation.get('status') != 'completed':
        raise ValueError('Campaign confirmation has not completed')
    aggregate = confirmation['aggregate']
    best = min(aggregate, key=lambda name: (aggregate[name], name != 'parent'))
    if best == 'parent':
        files, seeds = read(campaign / 'parent.json'), policy['baseline']['seeds']
    else:
        arm = state['arms'][best]
        key = arm['incumbent']
        files = read(campaign / 'submissions' / provider / (key + '.json'))['files']
        seeds = arm['metrics']['seeds']
    walls = [row['wall_seconds'] for row in confirmation['rows'] if row['configuration'] == best]
    return {'configuration': best, 'source': best if best == 'parent' else state['arms'][best]['incumbent'],
            'files': files, 'search_seeds': {s: {k: seeds[s][k] for k in ('mae', 'rmse')} for s in ('17', '29')},
            'confirmation_aggregate': aggregate, 'parent_confirmation_mae': aggregate['parent'],
            'relative_improvement': (aggregate['parent'] - aggregate[best]) / aggregate['parent'],
            'improved': aggregate[best] <= (1 - IMPROVEMENT) * aggregate['parent'],
            'max_confirmation_wall_seconds': max(walls), 'fit_wall_seconds': policy['fit_wall_seconds'],
            'package': policy['package'], 'package_sha256': policy['package_sha256'], 'image_id': policy['image_id'],
            'runtime_probe_sha256': policy['runtime_probe_sha256'], 'policy_sha256': ledger['policy_sha256']}


def successor(previous, output, probe, provider, model, transport):
    """Write the champion as a calibration record and initialize the next campaign; return the decision."""
    best = champion(previous, provider)
    decision = {k: v for k, v in best.items() if k != 'files'}
    decision.update(previous=str(previous), next=str(output))
    if not best['improved']:
        decision['continued'] = False
        return decision
    if sha(probe) != best['runtime_probe_sha256']:
        raise ValueError('Runtime probe differs from the previous campaign')
    record = output.parent / (output.name + '-parent')
    if record.exists() or output.exists():
        raise FileExistsError('Successor already prepared')
    (record / 'parent').mkdir(parents=True)
    for name, text in best['files'].items():
        (record / 'parent' / name).write_text(text, encoding='utf-8')
    write(record / 'report.json', {
        'schema': 'blind-g1-lineage-parent', 'passed': True, 'package_sha256': best['package_sha256'],
        'image_id': best['image_id'], 'parent_files': {p.name: sha(p) for p in (record / 'parent').iterdir()},
        'fits': [{'seed': int(s), **best['search_seeds'][s]} for s in ('17', '29')],
        'fit_wall_seconds': best['fit_wall_seconds'], 'heldout_evaluated': False, 'lineage': decision})
    controller.initialize(Path(best['package']), record, output, probe)
    controller.register(output, provider, model, transport)
    decision['continued'] = True
    write(output / 'lineage.json', decision)
    return decision


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['champion', 'successor'])
    parser.add_argument('--previous', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--probe', type=Path)
    parser.add_argument('--provider', default='claude')
    parser.add_argument('--model')
    parser.add_argument('--transport', default='cli-file-exchange')
    args = parser.parse_args()
    if args.action == 'champion':
        result = {k: v for k, v in champion(args.previous, args.provider).items() if k != 'files'}
    else:
        result = successor(args.previous, args.output, args.probe, args.provider, args.model, args.transport)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()

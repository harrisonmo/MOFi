"""Bounded, restartable v3 lineage scheduling; no execution on import."""
import argparse
from copy import deepcopy
from pathlib import Path

from rsi import blind_v3_runner as runner
from rsi.blind_g1 import read, write, sha
from rsi.hpc.core import locked


def checkpoint(path, value):
    temporary = path.with_suffix('.next.json')
    write(temporary, value)
    temporary.replace(path)


def initialize(path, campaign, generations):
    if path.exists() or not 1 <= generations <= 10:
        raise ValueError('Use a new lineage and 1-10 generations')
    state = runner.load(campaign)
    if state.get('action_prefix') or state.get('previous_generations'):
        raise ValueError('Start the lineage at its original G1 campaign')
    checkpoint(path, dict(schema='blind-v3-lineage', maximum_generations=generations,
        campaigns=[str(campaign.resolve())], status='ready',
        first_g1=state['g1_fingerprint'], scheduler_sha=sha(Path(__file__))))


def successor(parent, child, generation):
    state = runner.load(parent)
    if state['confirmation'] is None or state['confirmation']['status'] != 'completed':
        raise ValueError('Completed confirmation is required')
    if child.exists():
        existing = runner.load(child)
        if existing.get('parent_state_sha') != sha(parent / 'state.json'):
            raise ValueError('Existing successor has a different parent')
        return
    new = deepcopy(state)
    new.setdefault('previous_generations', []).append(dict(
        generation=generation - 1, history=state['history'],
        call_failures=state.get('call_failures', []), selected=state['incumbent']))
    new.update(history=[], pending=None, model_calls=0, call_pending=False,
               diagnostics_used=0, candidates_used=0, stopped=False, confirmation=None,
               call_failures=[], action_prefix=f'g{generation}-',
               parent_state_sha=sha(parent / 'state.json'))
    # Keep all source versions and original G1, but continue from the incumbent.
    staging = child.with_name(child.name + '.preparing')
    if staging.exists():
        raise ValueError('Incomplete successor staging requires inspection')
    staging.mkdir(parents=True)
    runner.save(staging, new)
    staging.replace(child)


def run(path, invoke, gpu=runner.gpu):
    with locked(path, timeout=1, stale_seconds=604800):
        lineage = read(path)
        if lineage['scheduler_sha'] != sha(Path(__file__)):
            raise ValueError('Frozen scheduler changed')
        if lineage['status'] == 'completed':
            return lineage
        while True:
            campaign = Path(lineage['campaigns'][-1])
            state = runner.load(campaign)
            if state['g1_fingerprint'] != lineage['first_g1']:
                raise ValueError('Original G1 binding changed')
            if state['pending'] or state['call_pending']:
                raise ValueError('Recover interrupted campaign before resuming lineage')
            if state['confirmation'] is None:
                while not state['stopped'] and state['model_calls'] < state['policy']['decisions']:
                    runner.step(campaign, invoke, gpu)
                    state = runner.load(campaign)
                    print(f'{campaign.name}: decision {state["model_calls"]}; incumbent {state["incumbent"]}', flush=True)
                runner.confirm(campaign, gpu)
                state = runner.load(campaign)
            if state['confirmation']['status'] != 'completed':
                raise ValueError('Confirmation incomplete; no automatic replay')
            if len(lineage['campaigns']) >= lineage['maximum_generations']:
                lineage['status'] = 'completed'
                checkpoint(path, lineage)
                return lineage
            generation = len(lineage['campaigns']) + 1
            child = path.parent / f'generation-{generation:02d}'
            successor(campaign, child, generation)
            lineage['campaigns'].append(str(child.resolve()))
            checkpoint(path, lineage)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'run', 'status'])
    parser.add_argument('--lineage', type=Path, required=True)
    parser.add_argument('--campaign', type=Path)
    parser.add_argument('--generations', type=int, default=1)
    args = parser.parse_args()
    if args.action == 'prepare':
        initialize(args.lineage, args.campaign, args.generations)
    elif args.action == 'run':
        from scripts.run_blind_codex_cli import invoke
        run(args.lineage, invoke)
    print(read(args.lineage))


if __name__ == '__main__':
    main()

"""One separately audited transport segment; never retry attempted episodes."""
import argparse
import hashlib
import json
from pathlib import Path

from rsi.experiment import git_state, write_json
from rsi.q2b_v21.run import Pilot, fingerprints
from rsi.q2b_v21.budget import RunBudget, BudgetStop
from rsi.q2b_v21.limits import Ledger, ROOT
from rsi.q2b_v21.runtime import digest
from rsi.q2b_http_transport import CodexCLI

LIMITS = dict(total_model_calls=104, probe_calls=2, generation_and_debug_calls=0,
              research_calls=102, observed_token_soft_limit=3_328_000,
              tokens_per_call_soft_limit=32_000)


def remaining(schedule, attempted):
    keys = [(r['id'], r['arm']) for r in schedule]
    done = [(r['id'], r['arm']) for r in attempted]
    if len(set(keys)) != len(keys) or len(set(done)) != len(done) or keys[:len(done)] != done:
        raise ValueError('Parent attempts must be a unique schedule prefix')
    return schedule[len(done):]


def recovery_fingerprints():
    return {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in
            ('rsi/q2b_recovery.py', 'rsi/q2b_http_transport.py', 'docs/Q2B_HTTP_RECOVERY.md')}


class Recovery(Pilot):
    def __init__(self, parent, output, executable):
        self.output = Path(output).resolve()
        if self.output.exists():
            raise RuntimeError('Recovery output must be new')
        self.parent = Path(parent).resolve()
        read_parent = lambda n: json.loads((self.parent / n).read_text(encoding='utf-8'))
        parent_manifest = read_parent('manifest.json')
        self.frozen = fingerprints()
        if parent_manifest['files'] != self.frozen:
            raise RuntimeError('Parent scientific implementation changed')
        models = read_parent('model_budget.json')['entries']
        completion = read_parent('completion.json')
        attempted = read_parent('episodes.json')
        self.schedule = remaining(read_parent('schedule.json'), attempted)
        if len(models) != 93 or len(attempted) != 45 or len(self.schedule) != 51 or completion['completed']:
            raise RuntimeError('Unexpected parent coverage or budget')
        root_git, cb_git = git_state(ROOT), git_state(ROOT / 'Crystal-Builder')
        if root_git['status'] or cb_git['status']:
            raise RuntimeError('Require clean independent repositories')
        self.parent_hashes = {n: hashlib.sha256((self.parent / n).read_bytes()).hexdigest() for n in
            ('manifest.json', 'candidate.json', 'gate.json', 'pairs.json', 'episodes.json',
             'model_budget.json', 'science_budget.json', 'schedule.json', 'completion.json')}
        self.recovery_hashes = recovery_fingerprints()
        self.output.mkdir(parents=True)
        for n in ('pairs.json', 'gate.json', 'candidate.json'):
            (self.output / n).write_bytes((self.parent / n).read_bytes())
        self.read = lambda n: json.loads((self.output / n).read_text(encoding='utf-8'))
        self.pairs, self.gate = self.read('pairs.json'), self.read('gate.json')
        self.manual = json.loads((ROOT / 'tools/q2b/manual-v2.json').read_text())
        self.candidate = self.read('candidate.json')
        if digest(self.candidate['program']) != self.candidate['sha256']:
            raise RuntimeError('Candidate hash mismatch')
        self.backend = CodexCLI('gpt-6-astra', 'high', 240, executable)
        self.manifest = dict(parent_manifest, root_git=root_git, crystal_builder_git=cb_git,
            experiment_version='q2b-v2.1-http-recovery', limits=LIMITS,
            parent_path=str(self.parent), parent_hashes=self.parent_hashes,
            recovery_files=self.recovery_hashes, parent_calls=len(models),
            parent_known_tokens=sum(r['tokens'] or 0 for r in models),
            parent_unknown_usage=sum(r['tokens'] is None for r in models),
            transport='CLI custom OpenAI provider, HTTP/SSE, no internal retries')
        write_json(self.output / 'manifest.json', self.manifest)
        write_json(self.output / 'schedule.json', self.schedule)
        self.budget = RunBudget(self.output / 'model_budget.json', LIMITS)
        self.science = Ledger(self.output / 'science_budget.json')
        carried = sum(r.get('wall_seconds', 0) for r in read_parent('science_budget.json')['entries'])
        self.science.data['entries'].append(dict(stage='parent_carried', label='prior_segment',
            status='ok', wall_seconds=carried, evaluations=0, measured=True))
        self.science.save()
        self.rows, self.authors = [], []
        self.unchanged()

    def unchanged(self):
        super().unchanged()
        if recovery_fingerprints() != self.recovery_hashes:
            raise BudgetStop('Recovery controller changed')
        for n, expected in self.parent_hashes.items():
            if hashlib.sha256((self.parent / n).read_bytes()).hexdigest() != expected:
                raise BudgetStop('Parent segment changed')
        if digest(self.read('candidate.json')) != digest(self.candidate):
            raise BudgetStop('Candidate changed')

    def run(self):
        stop = None
        try:
            schema = dict(type='object', properties={'ready': {'type': 'boolean'}},
                          required=['ready'], additionalProperties=False)
            for i in range(2):
                result = self.model('probe', f'probe-{i + 1}',
                    'Return ready=true. This is a transport-only check. Do not use tools.', schema)
                if self.budget.blocked or result['status'] != 'ok' or result['response'] != {'ready': True}:
                    raise BudgetStop(self.budget.blocked or 'HTTP admission failed')
            pairs = {p['id']: p for p in self.pairs}
            for row in self.schedule:
                self.episode(pairs[row['id']], row['arm'], self.candidate['program'])
        except Exception as exc:
            stop = f'{type(exc).__name__}: {exc}'
        self.unchanged()
        write_json(self.output / 'completion.json', dict(completed=len(self.rows) == 51 and stop is None,
            stop=stop, episodes=len(self.rows), model_calls=len(self.budget.entries), frozen_unchanged=True))
        print(json.dumps(self.read('completion.json')), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--parent', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--codex', required=True)
    args = parser.parse_args()
    Recovery(args.parent, args.output, args.codex).run()

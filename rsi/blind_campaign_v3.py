"""Offline v3 research-state framework. No model or training execution."""
from copy import deepcopy
from difflib import unified_diff
import math

from rsi.blind_campaign import validate_files, fingerprint

POLICY = {
    'schema': 'blind-g1-v3-draft', 'model': 'gpt-6.1-sol', 'reasoning_effort': 'high',
    'starting_point': 'G1-F50', 'independent_arm': False,
    'decisions': 12, 'diagnostics': 12, 'diagnostic_seconds': 60,
    'search_candidates': 6, 'search_seeds': [17, 29],
    'confirmation_seeds': [101, 202, 303], 'confirmation_fits': 6,
    'fit_seconds': 360, 'epochs': 50, 'proposal_seconds': 2700,
    'execution_enabled': False, 'heldout_enabled': False,
}


def initialize(g1_files):
    """Create offline state; production admission must verify calibrated G1."""
    validate_files(g1_files)
    return {'policy': deepcopy(POLICY), 'incumbent': 'g1',
            'sources': {'g1': deepcopy(g1_files)}, 'history': [],
            'pending': None, 'diagnostics_used': 0, 'candidates_used': 0,
            'g1_fingerprint': fingerprint(g1_files)}


def packet(state):
    """Include rejected source and exact differences, not only scalar scores."""
    return deepcopy({'policy': state['policy'], 'incumbent': state['incumbent'],
                     'sources': state['sources'], 'history': state['history'],
                     'current_metrics': state.get('metrics'),
                     'budget_used': {k: state.get(k, 0) for k in
                                     ('model_calls', 'diagnostics_used', 'candidates_used')},
                     'previous_generations': state.get('previous_generations', []),
                     'instructions': 'The original ancestor is G1 only; continue from this lineage incumbent or an archived route. '
                     'Use no prior G2/G3 or other provider results. '
                     'Choose diagnose, repair, evaluate, or stop. Explain the evidence. '
                     'Diagnostics use training samples only. Never request development labels.'})


def propose(state, action, base, hypothesis, files=None):
    """Reserve an action before execution; failed actions retain their cost."""
    if state['pending'] is not None or len(state['history']) >= state['policy']['decisions']:
        raise ValueError('Pending action or exhausted decision budget')
    if action not in ('diagnose', 'repair', 'evaluate', 'stop') or base not in state['sources']:
        raise ValueError('Unknown action or source version')
    if not isinstance(hypothesis, str) or not hypothesis.strip():
        raise ValueError('A research rationale is required')
    source = deepcopy(state['sources'][base] if files is None else files)
    validate_files(source)
    counter = 'candidates_used' if action == 'evaluate' else 'diagnostics_used'
    limit = 'search_candidates' if action == 'evaluate' else 'diagnostics'
    if action != 'stop' and state[counter] >= state['policy'][limit]:
        raise ValueError('Action budget exhausted')
    key = state.get('action_prefix', '') + f'action-{len(state["history"]) + 1}'
    diff = ''.join(''.join(unified_diff(state['sources'][base].get(name, '').splitlines(True),
                                      source.get(name, '').splitlines(True),
                                      fromfile=f'{base}/{name}', tofile=f'{key}/{name}'))
                   for name in sorted(set(source) | set(state['sources'][base])))
    state['sources'][key] = source
    state['history'].append({'id': key, 'action': action, 'base': base,
                             'hypothesis': hypothesis, 'diff': diff, 'status': 'pending'})
    state['pending'] = key
    if action != 'stop':
        state[counter] += 1
    return key


def health_errors(report):
    """Require explicit train-only feature health; never accept silent fallback."""
    if not isinstance(report, dict):
        return ['Missing diagnostic report']
    errors = []
    for field in ('train_only', 'finite_features', 'forward_pass', 'dimensions_valid'):
        if report.get(field) is not True:
            errors.append(f'{field} must be true')
    coverage = report.get('feature_coverage')
    if isinstance(coverage, bool) or not isinstance(coverage, (int, float)) or not math.isfinite(coverage) or coverage != 1:
        errors.append('Complete feature coverage is required')
    return errors


def finish(state, diagnostics, status='completed', metrics=None):
    """Record controller-sanitized feedback; promotion requires a future evaluator."""
    if state['pending'] is None or status not in ('completed', 'failed', 'interrupted'):
        raise ValueError('No pending action or invalid status')
    row = state['history'][-1]
    row.update(status=status, diagnostics=deepcopy(diagnostics), metrics=deepcopy(metrics))
    row['health_errors'] = health_errors(diagnostics) if row['action'] != 'stop' else []
    state['pending'] = None


def execute(*args, **kwargs):
    """Fail closed until runtime admission and bounded feedback are implemented."""
    raise RuntimeError('V3 is preparation-only; model calls and training are disabled')

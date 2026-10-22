"""Describe audited segments without pooling away transport or usage gaps."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    base = ROOT / 'benchmarks/q2b_v21'
    first = json.loads((base / 'astra-summary.json').read_text(encoding='utf-8'))
    second = json.loads((base / 'http-recovery-summary.json').read_text(encoding='utf-8'))
    assert first['audit']['passed'] and second['audit']['passed']
    arms = ['fixed', 'outer_generated', 'manual_control', 'removed']
    labels = dict(fixed='Fixed', outer_generated='Outer-generated', manual_control='Engineering reference', removed='Removal ablation')
    rows = [dict(r, segment=s) for s, summary in [('initial', first), ('http', second)] for r in summary['episodes']]
    assert len({(r['id'], r['arm']) for r in rows}) == len(rows)
    table = ['| Segment | Arm | Attempted | Reports returned | Passed | Tool errors | Known Research tokens |',
             '|---|---|---:|---:|---:|---:|---:|']
    for name, summary in [('Original v2.1', first), ('HTTP recovery', second)]:
        for arm in arms:
            r = summary['arms'][arm]
            table.append(f"| {name} | {labels[arm]} | {r['attempted']} | {r['completed']} | {r['passed']} | {r['tool_errors']} | {r['tokens']:,} |")
    coverage = {}
    for arm in arms:
        rr = [r for r in rows if r['arm'] == arm]
        coverage[arm] = dict(attempted=len(rr), planned=24, passed=sum(r['passed'] for r in rr),
            completed=sum(r['completed'] for r in rr), tool_errors=sum('tool_error' in r for r in rr),
            joint_attempted=sum(r['block'] == 'joint_transfer' for r in rr),
            joint_passed=sum(r['passed'] for r in rr if r['block'] == 'joint_transfer'))
    cost = dict(calls=sum(len(s['calls']) for s in [first, second]),
        known_tokens=sum(s['known_token_lower_bound'] for s in [first, second]),
        unknown_usage=sum(s['unknown_usage'] for s in [first, second]),
        model_wall_seconds=sum(s['model_wall_seconds'] for s in [first, second]),
        science_seconds=second['science_charged_seconds'],
        note='Science includes carried parent cost once; no finite total token bound is claimed.')
    paired = {}
    ids = sorted({r['id'] for r in rows})
    for arm in arms[1:]:
        counts = dict(wins=0, losses=0, ties=0, missing=24-len(ids), cross_transport_pairs=0)
        for key in ids:
            f = next((r for r in rows if r['id'] == key and r['arm'] == 'fixed'), None)
            o = next((r for r in rows if r['id'] == key and r['arm'] == arm), None)
            if f is None or o is None:
                counts['missing'] += 1
            else:
                counts['cross_transport_pairs'] += f['segment'] != o['segment']
                counts['wins' if o['passed'] > f['passed'] else 'losses' if o['passed'] < f['passed'] else 'ties'] += 1
        paired[arm] = counts
    ablation = {}
    for segment in ['initial', 'http', 'all_descriptive']:
        counts = dict(generated_only_pass=0, removed_only_pass=0, both_pass=0, neither_pass=0, missing=24-len(ids))
        for key in ids:
            selected = [r for r in rows if r['id'] == key and (segment == 'all_descriptive' or r['segment'] == segment)]
            g = next((r for r in selected if r['arm'] == 'outer_generated'), None)
            a = next((r for r in selected if r['arm'] == 'removed'), None)
            if g is None or a is None:
                counts['missing'] += 1
            else:
                counts['both_pass' if g['passed'] and a['passed'] else 'generated_only_pass' if g['passed'] else 'removed_only_pass' if a['passed'] else 'neither_pass'] += 1
        ablation[segment] = counts
    result = dict(visibility='public_engineering_validation', coverage_only=True, coverage=coverage,
                  cost=cost, descriptive_paired_vs_fixed=paired,
                  generated_vs_removed=ablation,
                  segments=['astra-summary.json', 'http-recovery-summary.json'],
                  inference_warning='Mixed transport, public fixtures, identical generated/manual program, interface failures; no hidden generalization claim.')
    (base / 'coverage-summary.json').write_text(json.dumps(result, indent=2) + '\n')
    text = f'''# Q2B v2.1: tool wiring fix and segmented experiment results

These results are public engineering validation only. {len(rows)}/96 episodes were attempted, {sum(r['completed'] for r in rows)} returned reports and {sum(r['passed'] for r in rows)} passed. Stop reason of the original segment: {first['completion']['stop']}; of the HTTP recovery segment: {second['completion']['stop']}. Episodes that never ran are listed separately and are not counted as scientific failures.

## Results by segment

{chr(10).join(table)}

Counting coverage across both segments, Fixed passed 16/19, Generated 17/18, Engineering reference 19/19 and Removal 17/19. These are descriptive, unbalanced denominators, not an effect estimate over all 96 episodes. Each arm attempted only 6/8 joint-transfer episodes, passing 4/6, 5/6, 6/6 and 5/6 respectively. All 17 generated/removal pairs within the same transport segment had identical outcomes (16 both passed, 1 both failed). The only cross-segment win for Generated corresponds to a Removal timeout with unknown usage and cannot be attributed to a scientific benefit of the tool. Generated beat Fixed only once, and that came from a node-ID format error in the Fixed arm.

In the table, passed/attempted is the actual pass rate and reports returned/attempted is the report completion rate. Returned-but-failed episodes include interface errors and cannot be read directly as scientific misjudgements. Per-episode errors, unknown usage, material/task strata and paired wins and losses are in the two segment JSON files. coverage-summary.json only adds full-coverage counts and descriptive pairings; it does not hide the transport change as one homogeneous run. Earlier v2 results are not merged.

## Validation and freeze boundary

The operation is named periodic_pair_evidence consistently across the description, plan schema and runtime, and it cannot be called from the Fixed or Removal arms. The following passed: reachability of 24 fixed primitive pairs, 48 independent reference validations, and 48 end-to-end validations running from the real registration entry point through the evidence handle to independent scoring; 35 small scientific/bad-input checks and 2 real resource-limit probes; and 87 related regression tests. The result audit passed all {first['audit']['checks']} checks for the original segment and all {second['audit']['checks']} for the recovery segment, including code/source/input hashes, usage-to-log correspondence, an unchanged parent segment and no reruns.

Scientific scoring, primitives, tolerances, inputs and the two-call Research workflow were not changed. The parent segment has a gap in case validation of node IDs: the model can produce components_A, but the runtime accepts only lowercase IDs, and the schema/catalog does not fully express this restriction. Those failures are kept; nothing was hot-fixed during recovery, and interface failures are not attributed to scientific capability.

## Cost and CLI diagnostics

The two v2.1 segments made {cost['calls']} CLI calls with {cost['known_tokens']:,} known tokens, plus {cost['unknown_usage']} calls of unknown usage, so no exact total cost or strict token ceiling can be claimed. Model wall time totals {cost['model_wall_seconds']:.3f} s; the science ledger accumulates {cost['science_seconds']:.3f} s, including the parent segment's carried cost once, with no new material preparation, engines, DFT, UFF, GCMC or training. The author used {first['author_tokens']:,} tokens and all probes used {first['probe_tokens'] + second['probe_tokens']:,} tokens; the author made 0 debugging calls, and validation and failed calls are all counted. Per-arm usage cost and author amortization are stored in JSON. The cost of writing the engineering reference, this development session's tokens and human hours cannot be measured and are not counted as zero. The CLI subscription has no per-call monetary billing, so no dollar figure is invented.

Historical v2 additionally used 213,497 known tokens plus 1 call of unknown usage, and 130.952 s of science (including a conservative 90 s estimate); material preparation and the first development cannot be treated as free. This historical cost is not mixed into this version's effect statistics.

After disabling the shared system skill installation, the stderr errors disappeared, but the parent segment still had a 240 s timeout with no usage, so the root cause is undetermined. The HTTP recovery used an officially supported separate provider configuration, the existing CLI login and the default official endpoint, with no credential extraction or proxy; 0 internal retries, a 120 s stream idle limit and a 240 s process limit. It ran only after two real probes passed. The original and HTTP contexts may differ and are reported separately. The CLI does not disclose the actual model identity; gpt-6-astra/high is the requested configuration.

The HTTP segment also hit a 240 s timeout with no usage on call 62; the event log shows only thread.started/turn.started and stderr is empty. Two other successful calls showed warnings about refreshing a remote plugin directory; this is circumstantial and does not establish causation. Switching to HTTP did not cure the hangs, so no further segments were started that would spend usage that cannot be reconciled. Five read-only doctor diagnostics made no model requests, and the original redacted logs are kept. Diagnostic, testing and engineering time is listed separately; the science ledger is not the project's total wall time.

## Supported and unsupported conclusions

The same outer-generated 9-node pure-data tool was successfully called on several materials and on task families A/B/C, which supports the feasibility of tool reuse in this restricted environment. It is byte-identical to the engineering reference program (SHA256 23632528a295661c3b36af7f52984989e4330b490898def6274783538a70f62e), so there is no evidence of a new algorithm. The paired results are affected by interface format failures, single sampling, public tasks and transport segmentation. They cannot show that outer generation beats a hand-written tool, nor support claims of hidden-test generalization or autonomous tool invention by the inner loop.

Fixed and Removal have the same tool availability, but random model outputs can still differ. The ablation can only be interpreted together with specific evidence/calls and same-segment pairings; random differences cannot all be attributed to the tool. No tool contains task judgement, retry or repair strategies, and the author cannot modify the scorer.

## Integration with the main thread

The scientific implementation is frozen at f03ed69 and the HTTP recovery at 98e4260; Crystal-Builder df014447f0812dd4f55ed60359b71a981c3eeb60 is unmodified. Integration only needs protocol.plan_schema, the runtime and the registration entry point of rsi/q2b_v21; rsi/q2b_recovery only handles segment budgets/transport and reuses the original Pilot.episode. The old rsi/q2b and its results are kept. Markdown, compact JSON, tests and code are committed locally together; raw logs under runs stay in a separate worktree and are not pushed.
'''
    (ROOT / 'docs/reports/Q2B_V21_RESULT.md').write_text(text, encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()

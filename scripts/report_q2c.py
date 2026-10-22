"""Render a conservative Markdown account of the frozen Q2C pilot."""
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def main():
    s=json.loads((ROOT/'benchmarks/q2c/astra-summary.json').read_text())
    names=dict(fixed='Fixed',outer_generated='Outer-generated',manual_control='Engineering reference',removed='Removed')
    table=['| Arm | Attempted/planned | Reports completed | Independently passed | Tool errors | Known Research tokens | Registered calls succeeded/requested |',
           '|---|---:|---:|---:|---:|---:|---:|']
    for arm,r in s['arms'].items():
        table.append(f"| {names[arm]} | {r['attempted']}/{r['planned']} | {r['completed']} | {r['passed']} | {r['tool_errors']} | {r['known_tokens']:,} | {r['successful_registered_uses']}/{r['requested_registered_uses']} |")
    blocks=['| Arm | Development | Source transfer | Task transfer | Joint transfer |','|---|---:|---:|---:|---:|']
    for arm,bb in s['blocks'].items():
        vals=[f"{bb[b]['passed']}/{bb[b]['attempted']} (planned {bb[b]['planned']})" for b in ['development','source_transfer','task_transfer','joint_transfer']]
        blocks.append('| '+names[arm]+' | '+' | '.join(vals)+' |')
    paired=[]
    for control,r in s['paired_generated_vs'].items():
        g=r['generated_use_tokens']+s['author_tokens'];c=r['control_use_tokens']
        fraction=f'{(g-c)/c:+.2%}' if c else 'not computable'
        paired.append(f"- Generated vs {names[control]}: {r['matched']} complete pairs, wins/losses/ties={r['wins']}/{r['losses']}/{r['ties']}, {r['missing']} missing; usage known for {r['known_cost_pairs']} pairs, generated usage plus one full author cost={g:,} vs control usage={c:,} tokens, difference {fraction}.")
    rows=s['episodes'];ops=sum(r['operations'] for r in rows);errors=[r for r in rows if 'tool_error' in r]
    actual_uses={arm:(s['arms'][arm]['reused_sources'],s['arms'][arm]['reused_families']) for arm in ['outer_generated','manual_control']}
    text=f'''# Q2C-v1: shared-primitive reachability and Astra four-arm results

Date: 2026-10-13. These results are **public engineering validation on synthetic periodic structures** only; they are neither cross-material generalization on real MOFs nor autonomous tool invention by the inner loop.

## Completion status

Frozen implementation: {s['manifest']['root_git']['commit']}; Crystal-Builder: {s['manifest']['crystal_builder_git']['commit']}. The latter was not modified, and this round runs none of its material building. Run status: {'fully completed' if s['completion']['completed'] else s['completion']['stop']}. {len(rows)}/48 attempted, {sum(r['completed'] for r in rows)} reports completed, {sum(r['passed'] for r in rows)} independently passed. Episodes that never ran are not filled with zeros or counted as scientific failures.

{chr(10).join(table)}

Report completion rate = completed/attempted and pass rate = passed/attempted; tool errors are kept separate from scientific judgement errors. A registered request counts as successful only if a batch actually completed; tool availability or the model's choice to use it is not counted as scientific success.

The audit also lists abstained, confident_numerically_wrong and numerically_correct_contract_rejected. After all 8 tool errors the model abstained honestly. Two more episodes (D1-S engineering reference, D2-S removed) were numerically correct but submitted two handles at once (including a scalar projection), while the task requires one complete result handle, so the frozen scorer rejected them. No clearly wrong numerical answer was observed. Contract failures must not be read as scientific numerical errors, nor back-filled as formal passes.

## Splits and pairing

4 analytic sources × 3 tasks G/M/S: the D1 chain and D2 layer are development sources; the E1 three-dimensional net and E2 periodic chain with a finite component are evaluation sources. G audits the ring translation rank/saturation index, M computes the non-orthogonal shortest image and the number of equidistant images, and S enumerates same-species mappings under same-cell translations. The author saw only G on D1/D2. All data are public and strict hidden isolation was not implemented. The 3D-net/finite-component differences are structurally meaningful, but they cannot support a claim of held-out real-material sources.

{chr(10).join(blocks)}

{chr(10).join(paired)}

These costs compare only known tokens on matched tasks, charging one author cost per independent comparison; the rows must not be summed into a total. Shared probes/calibration, failures and unknown usage are listed separately below. There are no repeated independent runs and no significance claims.

## Admission and independent validation

Before freezing, 77 tests passed, and 12 fixed-tool solutions and 12 registered end-to-end runs all agreed with independent references. Development iterations account for 200 tests/calibration counts and 7.047 s of science. The shared cycle_groups uses a weighted union-find, integer_lattice uses integer cross/dot products, mic_probe uses ASE MIC with an SVD out-of-domain bound, and translation_maps uses floating-point modulo-integer residuals; the independent references use BFS, rational minors, exact rational out-of-domain bounds and exact modulo-integer matching respectively. Candidates cannot change the evaluator; all arms share the same primitives, and the final report must be bound to real retained evidence.

The four arms were frozen to one plan, one batch and one report each, sharing the gpt-6-astra/high request, scoring, inputs and budget. After the run, {s['audit']['checks']} read-only audit checks passed, covering frozen files, tasks, raw usage, evidence input/program binding and plan order. No old scoring or prompts were hot-fixed.

## Tool reuse and discovered limitations

The normalized data flow of the candidate and the engineering reference is {'identical' if s['candidate_semantically_equal_to_manual'] else 'different'} (ignoring intermediate node names and declaration order). A rename must not be read as a new algorithm. The Generated and Engineering reference arms each requested the registered tool 4 times, and none retained final tool evidence; this round shows no confirmed reuse of a registered tool across sources or task families. Success of a tool arm that did not call the tool only shows that the shared base primitives work, not a benefit from reusing a generated tool.

The author space required one bundled tool that outputs all four kinds of evidence at once, whereas the M/S tasks could call a single shared primitive directly. This limited the opportunity for real savings; this round does not represent more general on-demand small-tool design, algorithm invention or the difficulty of real material tasks. All G/M/S tools are pure computation and contain no next-action selection or retry strategy.

In the first G task, the Generated and Reference arms passed periodic_evidence.graph_a back into integer_lattice; graph_a already was that result, causing KeyError: cycles. The author contract and offline end-to-end tests stated the type, but the Research tool description listed only the returned fields without making their structure clear. Later errors of the same kind were kept and cannot be read as a difference in periodic-graph scientific capability. Adding a tool can mislead planning; passing fixed reachability/registration wiring tests does not mean the model understands the interface.

This round had {len(errors)} tool errors; the raw JSON keeps each one. The lower bound on primitive calls behind retained evidence is {ops}. Failed batches may have completed part of their computation, so the primitive count is unknown and cannot be reported as 0. Science wall time is counted in full, including failures. G is only a descriptive audit and does not decide full graph isomorphism; S does not support noise, partial occupancy or supercells; M is used only within its declared range.

## Full-stage cost and budget

{s['completion']['model_calls']} CLI calls in total; {s['known_token_lower_bound']:,} known tokens and {s['unknown_usage']} calls of unknown usage. Author {s['author_tokens']:,}, probes {s['probe_tokens']:,}, Research per arm as above. Model wall time {s['model_wall_seconds']:.3f} s; this round's science ledger accumulates {s['science_seconds']:.3f} s (including P0, author validation, use, scoring and failures). The earlier reference preparation of 116 items/1.922 s is listed separately. Engineering-session tokens, reference-tool authoring and human hours were not measured; they are not counted as zero, and no dollar cost is invented.

Limits: 101 calls / 3,232,000 observed tokens / 600 science seconds; 2 probes, at most 3 author calls and 96 Research calls; a 32k soft limit and a 240 s timeout per call. Unknown usage stops the run, with no new recovery segment. If usage is unknown, the known tokens are only a lower bound. When the CLI does not disclose the actual model identity it is recorded as unknown; request parameters are not treated as a verified snapshot.

## Integration and next steps

The new module rsi/q2c is independent of the old Q2/B and the main thread M10. The manifest, candidate, call/science ledgers and raw evidence stay in runs/q2c-v1; the compact summary is in benchmarks/q2c/astra-summary.json and can be rechecked with scripts/audit_q2c.py and scripts/report_q2c.py.

The next version should first generate one unified contract for the registered return field types, their purpose and the legal downstream inputs, and validate the semantic wiring with real Research probes; old scores cannot be back-filled. Only then should larger, more complex real-material/topology splits, equally reachable by fixed tools, be considered. This round claims no improvement in hidden generalization or scientific completion rate. Code, protocol and results are committed locally, not pushed, and the main thread is not modified.
'''
    (ROOT/'docs/reports/Q2C_ASTRA_RESULT.md').write_text(text,encoding='utf-8')


if __name__=='__main__':main()

"""Independently audit crossed-field work and summarize paired diagnostics."""
import json
import math
from pathlib import Path
import numpy as np
import torch
from scipy.special import logsumexp
from run_qk12_diagnostic import sha

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'runs/qk8-cross-01'
r = json.loads((RUN / 'results.json').read_text())
traces = json.loads((RUN / 'traces.json').read_text())
cfg = r['provenance']['config']
assert r['status'] == 'completed' and r['seconds'] < cfg['seconds_limit']
assert sha(ROOT / 'scripts/run_qk8_cross.py') == r['provenance']['script_sha256']
assert sha(ROOT / 'configs/qk8-cross-fields.json') == r['provenance']['config_sha256']
assert sha(ROOT / 'scripts/run_qk12_diagnostic.py') == r['provenance']['analytic_sha256']
for checkpoint in r['checkpoints']:
    assert sha(Path(checkpoint['path'])) == checkpoint['sha256']
keys = ['seed', 'stream', 'split', 'arm', 'graph']
assert len(r['rows']) == len({tuple(x[k] for k in keys) for x in r['rows']}) == 1080
assert len(traces) == len({tuple(x[k] for k in keys + ['time']) for x in traces}) == 35640
for x in traces:
    np.testing.assert_allclose(x['forward_mse'], x['velocity_mse'] + .0625*x['score_mse'] + .5*x['cross'], atol=1e-10)
    np.testing.assert_allclose(x['backward_mse'], x['velocity_mse'] + .0625*x['score_mse'] - .5*x['cross'], atol=1e-10)
stats = []
count = 0
for seed in cfg['seeds']:
    for stream in cfg['streams']:
        for split, ng in [('test', 24), ('ood', 12)]:
            for arm in cfg['arms']:
                chunks = torch.load(RUN / f'{seed}-{stream}-{split}-{arm}.pt', weights_only=True)
                arrays = {k: np.concatenate([c[k].numpy() for c in chunks]).reshape(ng, cfg['paths']) for k in chunks[0]}
                work = arrays['work']
                assert np.isfinite(work).all()
                np.testing.assert_allclose(work, arrays['delta_energy'] + arrays['log_kernel_ratio'], atol=1e-10)
                np.testing.assert_allclose(work, arrays['dimensionless_work'], atol=1e-10)
                je = -(logsumexp(-work, axis=1) - np.log(cfg['paths']))
                weights = np.exp(-work - logsumexp(-work, axis=1)[:, None])
                ess = 1 / (weights**2).sum(1)
                rr = sorted([x for x in r['rows'] if x['seed']==seed and x['stream']==stream and x['split']==split and x['arm']==arm], key=lambda x:x['graph'])
                assert [x['graph'] for x in rr] == list(range(ng))
                np.testing.assert_allclose(je, [x['prediction'] for x in rr], atol=1e-10, rtol=1e-10)
                np.testing.assert_allclose(ess, [x['ess'] for x in rr], atol=1e-10, rtol=1e-10)
                if arm in ['PP', 'JJ', 'PS']:
                    recipe = {'PP':'parent', 'JJ':'joint', 'PS':'score_only'}[arm]
                    old = torch.load(ROOT / f'runs/qk7-frozen-01/seed-{seed}-{recipe}/stream-{stream}/{split}-work.pt', weights_only=True)
                    historical = np.concatenate([x['feat_gnn_jarzynski'].numpy() for x in old])
                    np.testing.assert_allclose(work, historical, atol=1e-10, rtol=1e-10)
                count += work.size
                for i, row in enumerate(rr):
                    stats.append({**row, 'error': abs(row['prediction']-row['truth']),
                                  'work_std': float(work[i].std()), 'max_weight': float(weights[i].max()),
                                  'top5_mass': float(np.sort(weights[i])[-math.ceil(.05*cfg['paths']):].sum())})

summary, per_seed, temporal = [], [], []
for split in ['test', 'ood']:
    for arm in cfg['arms']:
        rows = [x for x in stats if x['split']==split and x['arm']==arm]
        summary.append({'split':split, 'arm':arm, 'mae':float(np.mean([x['error'] for x in rows])),
                        'ess_fraction':float(np.mean([x['ess']/cfg['paths'] for x in rows])),
                        'low_ess':sum(x['ess']<.1*cfg['paths'] for x in rows),
                        **{k:float(np.mean([x[k] for x in rows])) for k in ['work_std', 'max_weight', 'top5_mass']}})
        for seed in cfg['seeds']:
            ss = [x for x in rows if x['seed']==seed]
            per_seed.append({'split':split, 'arm':arm, 'seed':seed, 'mae':float(np.mean([x['error'] for x in ss]))})
        for region in ['early', 'middle', 'late']:
            tt = [x for x in traces if x['split']==split and x['arm']==arm and
                  (('early' if x['time']<=.1 else 'late' if x['time']>=.9 else 'middle')==region)]
            temporal.append({'split':split, 'arm':arm, 'region':region,
                             **{k:float(np.mean([x[k] for x in tt])) for k in ['velocity_mse','score_mse','forward_mse','backward_mse','cross']}})
contrasts = []
for split in ['test','ood']:
    for seed in cfg['seeds']:
        m = {x['arm']:x['mae'] for x in per_seed if x['split']==split and x['seed']==seed}
        contrasts.append({'split':split, 'seed':seed, 'score_at_parent_velocity':m['PJ']-m['PP'],
                          'velocity_at_parent_score':m['JP']-m['PP'],
                          'interaction':m['JJ']-m['JP']-m['PJ']+m['PP']})
report = {'summary':summary, 'per_seed':per_seed, 'temporal':temporal, 'contrasts':contrasts,
          'seconds':r['seconds'], 'audit':{'work_values':count, 'historical_anchors':True, 'trace_rows':len(traces)},
          'timings':r['timings'], 'provenance':r['provenance'], 'checkpoints':r['checkpoints']}
(ROOT/'docs/reports/QK8_RESULT.json').write_text(json.dumps(report, indent=2)+'\n')
lines = ['# QK8 crossed fields and sampler trajectories', '', '2026-10-19. Development diagnostic; no fitting or final test.', '',
         f"Completed in {r['seconds']:.3f}/900 seconds. Independently audited {count} work values and {len(traces)} trajectory summaries. PP/JJ/PS work reproduces QK7 within 1e-10 tolerance. All nine checkpoints, source hashes, analytic identities, drift expansions and instrumentation admission checks passed.", '',
         '| Split | V/S | MAE | ESS fraction | Work std | Max weight | Top 5% mass |',
         '|---|---|---:|---:|---:|---:|---:|']
for x in summary:
    lines.append(f"| {x['split']} | {x['arm']} | {x['mae']:.6f} | {x['ess_fraction']:.3f} | {x['work_std']:.3f} | {x['max_weight']:.4f} | {x['top5_mass']:.3f} |")
lines += ['', 'P: original parent; J: joint continuation; S: score-only continuation. The first letter is velocity. All arms have the same path/step budget; mixed arms use two forward passes. Runtime includes instrumentation and is not deployment latency.', '',
          '## Findings and next decision', '',
          'On six-node development graphs, replacing only velocity raises MAE from 0.077637 to 0.087893 (+13.21%); replacing only score raises it to 0.080764 (+4.03%); joint replacement raises it to 0.093344 (+20.23%). Neither output alone fully explains the joint result. The signed MAE interaction is positive on average but changes sign across seeds, so a universal incompatibility mechanism is not established.', '',
          'The joint velocity has lower reference MSE in all three time regions on both splits. On six-node trajectories its late velocity MSE falls from 0.015467 to 0.004006, while early/middle score MSE also falls. Nevertheless, free-energy MAE worsens. The parent has a negative early velocity-score error cross term; improving a field can remove cancellation in the forward drift. This is a descriptive observation on different trajectories, not proof that cancellation caused the MAE change.', '',
          'Six-node joint ESS rises from 0.465 to 0.499 and average maximum weight falls from 0.0616 to 0.0535 despite worse MAE. Empirical ESS and marginal field losses are therefore insufficient selection criteria in this sample. All arms have zero graph-stream ESS fractions below 0.1. The PJ four-node average improvement is driven by seed 17; seeds 29 and 43 slightly worsen. PS also regresses substantially on six-node seed 43. No cross-seed robust successor is established.', '',
          'Retain the original parent and stop unrestricted weight tuning. Before a human/RSI pilot, freeze a shared evaluator with paired repeated sampling, graph-level development partitions and explicit inference costs; use free-energy error as the primary criterion and field/weight diagnostics as secondary evidence. A bounded next intervention may preserve parent velocity and constrain score updates, but QK8 does not establish it will help. Fresh final graphs must remain sealed until both arms lock their recipes. No additional run is authorized by this report itself.', '',
          '## Interpretation limits', '',
          'All graphs were previously viewed. On-policy coordinates differ between arms. Analytic targets describe the independent-endpoint interpolant, not the actual sampler density. The MAE factorial interaction is descriptive on a nonlinear metric, not an independent causal mechanism estimate. Three training seeds with two paired MC streams do not establish broad generalization. No automatic promotion; retain the parent.', '',
          'Frozen implementation/protocol commit: dcd4672. Qiankun source: bce0ffe801040a78442140b28f3d07963eafe598.', '',
          '[Frozen protocol](../QK8_CROSS_FIELDS_PROTOCOL.md). [Per-seed contrasts and temporal diagnostics](QK8_RESULT.json). Raw work and trajectory summaries: runs/qk8-cross-01. No Crystal-Builder execution or remote push.']
(ROOT/'docs/reports/QK8_RESULT.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
print(json.dumps({'summary':summary,'contrasts':contrasts,'temporal':temporal,'seconds':r['seconds']}, indent=2))

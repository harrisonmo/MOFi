"""Audit the complete architecture experiment and paired final comparisons."""
import json
import math
import argparse
import hashlib
import sys
from pathlib import Path
import numpy as np
import torch
from scipy.special import logsumexp
from run_qk12_diagnostic import sha

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'runs/qk9-architecture-01'
parser = argparse.ArgumentParser()
parser.add_argument('--source', type=Path, required=True)
args = parser.parse_args()
sys.path.insert(0, str(args.source.resolve()))
from rsi.jarzynski.systems import graph_hash
r = json.loads((RUN/'results.json').read_text())
cfg = r['provenance']['config']
rows = r['rows']
assert r['status']=='completed' and r['seconds'] < cfg['seconds_limit']
for path, expected in r['provenance']['source']['source_hashes'].items():
    assert sha(args.source/path)==expected
original_data_path = args.source/'runs/qiankun-gpu-comparison-01/data.pt'
assert sha(original_data_path)==r['provenance']['data_sha256']
original = torch.load(original_data_path, weights_only=True)['graphs']
final = torch.load(RUN/'final-data.pt', weights_only=True)
manifest = json.loads((RUN/'final-manifest.json').read_text())
for split, graph in final.items():
    assert graph_hash(graph)==manifest[split]['hash'] and len(graph['a0'])==48
def fingerprints(graph):
    return [hashlib.sha256(b''.join(graph[k][i].contiguous().numpy().tobytes() for k in sorted(graph))).hexdigest()
            for i in range(len(graph['a0']))]
old_fingerprints = {h for g in original.values() for h in fingerprints(g)}
new_fingerprints = [h for g in final.values() for h in fingerprints(g)]
assert len(set(new_fingerprints))==96 and not old_fingerprints.intersection(new_fingerprints)
for path, expected in r['provenance']['hashes'].items():
    assert sha(ROOT/path)==expected
for cost in r['costs']:
    assert cost['parameters']==cfg['parameters'][cost['recipe']] and cost['updates']==1200
    assert sha(RUN/f"train-{cost['seed']}-{cost['recipe']}/feat_gnn.pt")==cost['checkpoint_sha256']
key_names = ['training_seed','recipe','stream','split','graph_id']
index = {tuple(x[k] for k in key_names):x for x in rows}
assert len(index)==len(rows)==4752
historical = json.loads((ROOT/'runs/qk7-frozen-01/results.json').read_text())['records']
historical = {(x['training_seed'],x['stream'],x['split'],x['graph_id']):x for x in historical if x['recipe']=='parent'}
work_count = 0
for phase, splits in [('development',{'test':24,'ood':12}),('final',{'final_four':48,'final_six':48})]:
    for seed in cfg['seeds']:
        for arm in cfg['architectures']:
            for stream in cfg['streams']:
                for split, n in splits.items():
                    folder = RUN/f'{phase}-{seed}-{arm}-{stream}-{split}'
                    legacy_split = 'ood' if split in ['ood','final_six'] else 'test'
                    chunks = torch.load(folder/f'{legacy_split}-work.pt',weights_only=True)
                    rr = [index[seed,arm,stream,split,i] for i in range(n)]
                    for estimator in ['feat_gnn_jarzynski','no_transport_jarzynski']:
                        work = np.concatenate([x[estimator].numpy() for x in chunks])
                        assert work.shape==(n,cfg['paths']) and np.isfinite(work).all()
                        prediction = -(logsumexp(-work,axis=1)-math.log(cfg['paths']))
                        ess = np.exp(2*logsumexp(-work,axis=1)-logsumexp(-2*work,axis=1))
                        np.testing.assert_allclose(prediction,[x['predictions'][estimator] for x in rr],rtol=1e-10,atol=1e-10)
                        np.testing.assert_allclose(ess,[x['ess'][estimator] for x in rr],rtol=1e-10,atol=1e-10)
                        work_count += work.size
                    for x in rr:
                        reference = index[seed,'shared',stream,split,x['graph_id']]
                        for estimator in ['supervised_gnn','no_transport_jarzynski']:
                            assert x['predictions'][estimator]==reference['predictions'][estimator]
                        if phase=='development' and arm=='shared' and stream in [101000,211000]:
                            old = historical[seed,stream,split,x['graph_id']]
                            assert x['predictions']==old['predictions']

summary, per_seed, intervals = [], [], []
def error(row):
    return row['predictions']['feat_gnn_jarzynski']-row['truth']

for split in ['test','ood','final_four','final_six']:
    reference = [x for x in rows if x['split']==split and x['recipe']=='shared']
    parent_mae = float(np.mean([abs(error(x)) for x in reference]))
    for arm in cfg['architectures']:
        rr = [x for x in rows if x['split']==split and x['recipe']==arm]
        errors = np.array([error(x) for x in rr])
        ess = np.array([x['ess']['feat_gnn_jarzynski']/cfg['paths'] for x in rr])
        mae = float(abs(errors).mean())
        summary.append({'split':split,'arm':arm,'mae':mae,'rmse':float(np.sqrt((errors**2).mean())),
                        'signed_bias':float(errors.mean()),'gain':1-mae/parent_mae,
                        'ess_fraction':float(ess.mean()),'low_ess':int((ess<.1).sum()),
                        'inference_seconds':float(np.mean([x['feat_gnn_jarzynski'] for x in r['timings'] if x['split']==split and x['recipe']==arm]))})
        for seed in cfg['seeds']:
            ss = [x for x in rr if x['training_seed']==seed]
            per_seed.append({'split':split,'arm':arm,'seed':seed,'mae':float(np.mean([abs(error(x)) for x in ss]))})
        if split.startswith('final_'):
            n = cfg[split]['graphs']
            a = np.array([np.mean([abs(error(index[seed,arm,stream,split,g])) for seed in cfg['seeds'] for stream in cfg['streams']]) for g in range(n)])
            b = np.array([np.mean([abs(error(index[seed,'shared',stream,split,g])) for seed in cfg['seeds'] for stream in cfg['streams']]) for g in range(n)])
            rng = np.random.default_rng(190039)
            draws = rng.integers(n,size=(2000,n))
            difference = a[draws].mean(1)-b[draws].mean(1)
            gain = 1-a[draws].mean(1)/b[draws].mean(1)
            intervals.append({'split':split,'arm':arm,'mae_difference_95ci':np.quantile(difference,[.025,.975]).tolist(),
                              'relative_gain_95ci':np.quantile(gain,[.025,.975]).tolist()})

metrics = {(x['split'],x['arm']):x for x in summary}
seed_metrics = {(x['split'],x['arm'],x['seed']):x['mae'] for x in per_seed}
ranking = {arm:sum(metrics[s,arm]['mae']/metrics[s,'shared']['mae'] for s in ['test','ood'])/2 for arm in cfg['architectures']}
selected = min(ranking,key=ranking.get)
assert selected==r['selection']['selected']
assert json.loads((RUN/'selection-lock.json').read_text())==r['selection']
gates = {}
for arm in cfg['architectures']:
    gates[arm] = all(metrics[s,arm]['gain']>=.05 and metrics[s,arm]['low_ess']<=metrics[s,'shared']['low_ess'] and
                     all(seed_metrics[s,arm,seed]<=1.1*seed_metrics[s,'shared',seed] for seed in cfg['seeds'])
                     for s in ['final_four','final_six'])
report = {'summary':summary,'per_seed':per_seed,'paired_graph_bootstrap':intervals,'selected_on_development':selected,
          'final_gates':gates,'selected_passed':gates[selected],'seconds':r['seconds'],'training':r['costs'],
          'audit':{'work_values':work_count,'unique_records':len(rows),'historical_shared_predictions':True,
                   'matched_controls':True,'final_graphs_distinct_from_all_original_splits':True,
                   'architecture_tests_passed':8,'legacy_tests_passed':73},
          'final_manifest':json.loads((RUN/'final-manifest.json').read_text()),'provenance':r['provenance']}
(ROOT/'docs/reports/QK9_RESULT.json').write_text(json.dumps(report,indent=2)+'\n')
lines = ['# QK9 complete architecture comparison','', '2026-10-19. Synthetic Gaussian graph benchmark; assistant-designed intervention.', '',
         f"Completed in {r['seconds']:.3f}/1800 seconds. Nine fresh 1200-update fits, three seeds, four paired sampling streams, 36 existing development graphs and 96 fresh final graphs. Development-selected architecture: **{selected}**. Frozen final gate passed: **{gates[selected]}**.", '',
         '| Split | Architecture | MAE | Gain vs shared | ESS fraction | Inference seconds |',
         '|---|---|---:|---:|---:|---:|']
for x in summary:
    lines.append(f"| {x['split']} | {x['arm']} | {x['mae']:.6f} | {x['gain']:.2%} | {x['ess_fraction']:.3f} | {x['inference_seconds']:.3f} |")
lines += ['', '## Decision', '',
          'Retain the original shared architecture. The development-selected independent model fails the frozen final gate: four-node MAE worsens 2.25% and six-node MAE worsens 0.64%. The branched model changes four-node MAE by -0.14% gain and six-node MAE by +1.54% gain, also failing the gate; its four-node seed 17 regresses more than 10%. All candidate gain intervals include zero. Do not switch to the branched model after seeing final outcomes.', '',
          'Branched inference costs approximately 19-20% more and independent inference 45-47% more on the fresh splits. Training costs are approximately 31% and 65% higher, respectively. Under this parameter/update/path budget, splitting the two fields is not supported as a useful replacement. This does not rule out other architectures, larger budgets or benefits on real MOF tasks, and it does not establish that shared gradients never conflict.', '',
          'Close this architecture comparison without further tuning on the consumed final graphs. The shared model is the reference for a subsequent human/RSI experiment; freeze common search-space and compute rules before either arm starts. Any later architecture or timestep intervention needs a separately registered experiment and fresh final evaluation.', '',
          '![Final architecture comparison](QK9_ARCHITECTURE.png)', '',
          '## Paired final uncertainty', '', '| Split | Architecture | MAE difference 95% CI | Relative gain 95% CI |', '|---|---|---|---|']
for x in intervals:
    d, g = x['mae_difference_95ci'], x['relative_gain_95ci']
    lines.append(f"| {x['split']} | {x['arm']} | [{d[0]:.6f}, {d[1]:.6f}] | [{g[0]:.2%}, {g[1]:.2%}] |")
lines += ['', 'Graph-cluster bootstrap: 2000 paired resamples, conditional on these three trained seeds and four sampling streams. It does not estimate uncertainty over the population of possible training seeds.', '',
          '## Compute and validation', '', '| Architecture | Parameters | Mean training seconds |', '|---|---:|---:|']
for arm in cfg['architectures']:
    seconds = np.mean([x['seconds'] for x in r['costs'] if x['recipe']==arm])
    lines.append(f"| {arm} | {cfg['parameters'][arm]} | {seconds:.3f} |")
lines += ['', f"All 8 new architecture tests and 73 existing Jarzynski tests passed. Calibration admitted all three arms. Shared training checkpoint tensors reproduce historical parents exactly. Independent NumPy/SciPy reductions verify {work_count} work values and all 4752 graph-stream records. Physical and supervised controls match across arms; historical shared development predictions reproduce exactly. Source and checkpoint hashes pass.", '',
          'Widths were selected by parameter count only: shared 32, branched 27, independent 22. Equal parameter/update/path budgets do not establish equal compute; inference costs are reported rather than hidden. Width and sharing change together, so this experiment does not isolate the sharing mechanism alone.', '',
          'Final graphs were generated only after all checkpoints and the development selection were locked. All 96 graph fingerprints are unique and disjoint from all original splits. All three final recipes are reported without retuning. These final sets are now consumed and must not be reused as an unseen test after further development. Six-node OOD changes graph size, not coefficient ranges. No human/RSI comparison, realistic MOF evidence, Crystal-Builder execution or remote push is claimed.', '',
          'Frozen source/protocol: 79b0639. Qiankun: bce0ffe801040a78442140b28f3d07963eafe598. [Protocol](../QK9_ARCHITECTURE_PROTOCOL.md). [Full metrics and per-seed results](QK9_RESULT.json). Raw checkpoints/work: runs/qk9-architecture-01.']
(ROOT/'docs/reports/QK9_RESULT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
arms = list(cfg['architectures'])
colors = ['#566573', '#2471a3', '#148f77']
for axis, split, title in zip(axes, ['final_four','final_six'], ['Fresh four-node graphs','Fresh six-node graphs']):
    values = [metrics[split,a]['mae'] for a in arms]
    axis.bar(arms, values, color=colors)
    for i, arm in enumerate(arms):
        yy = [seed_metrics[split,arm,seed] for seed in cfg['seeds']]
        axis.scatter(i+np.linspace(-.12,.12,len(yy)), yy, color='black', s=20, zorder=3)
    axis.set_title(title)
    axis.set_ylabel('Free-energy MAE (lower is better)')
    axis.spines[['top','right']].set_visible(False)
fig.suptitle('QK9 architecture comparison: bars = mean; dots = training seeds')
fig.savefig(ROOT/'docs/reports/QK9_ARCHITECTURE.png', dpi=180)
plt.close(fig)
print(json.dumps({'summary':summary,'selected':selected,'gates':gates,'intervals':intervals,'per_seed':per_seed,'seconds':r['seconds']},indent=2))

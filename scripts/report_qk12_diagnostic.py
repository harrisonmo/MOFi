"""Audit the frozen diagnostic grid and render development cost curves."""
import hashlib
import itertools
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'runs/qk12-diagnostic-02'
r=json.loads((RUN/'results.json').read_text())
c=r['evidence']['config']
assert r['status']=='completed' and r['budget_seconds']<1800
assert hashlib.sha256((ROOT/'scripts/run_qk12_diagnostic.py').read_bytes()).hexdigest()==r['evidence']['script_sha256']
assert hashlib.sha256((ROOT/'configs/qk12-diagnostic.json').read_bytes()).hexdigest()==r['evidence']['config_sha256']
rows=[x for x in r['records'] if not x['calibration']]
expected={(s,m,split,g,p,k,a) for s,m,p,k,a in itertools.product(c['training_seeds'],c['sampling_seeds'],c['paths'],c['steps'],c['arms']) for split,n in [('test',24),('ood',12)] for g in range(n)}
keys=[(x['seed'],x['mc'],x['split'],x['graph'],x['paths'],x['steps'],x['arm']) for x in rows]
assert len(keys)==len(set(keys)) and set(keys)==expected
assert all(np.isfinite([x['prediction'],x['truth'],x['ess']]).all() and 1-1e-9<=x['ess']<=x['paths']+1e-9 for x in rows)
summary=[]
for split,p,k,a in itertools.product(['test','ood'],c['paths'],c['steps'],c['arms']):
 subset=[x for x in rows if (x['split'],x['paths'],x['steps'],x['arm'])==(split,p,k,a)]
 errors=np.array([x['prediction']-x['truth'] for x in subset]);ess=np.array([x['ess']/p for x in subset])
 costs=[x for x in r['timings'] if not x['calibration'] and (x['split'],x['paths'],x['steps'],x['arm'])==(split,p,k,a)]
 assert len(costs)==6
 grouped=np.array([[np.mean([abs(x['prediction']-x['truth']) for x in subset if x['seed']==s and x['mc']==m]) for m in c['sampling_seeds']] for s in c['training_seeds']])
 summary.append({'split':split,'paths':p,'steps':k,'arm':a,'mae':float(abs(errors).mean()),'rmse':float(np.sqrt((errors**2).mean())),
 'bias':float(errors.mean()),'ess_fraction':float(ess.mean()),'low_ess_count':int((ess<.1).sum()),
 'mae_training_seed_sd':float(grouped.mean(1).std(ddof=1)),'mean_sampling_seed_sd':float(grouped.std(1,ddof=1).mean()),
 'standalone_seconds':float(np.mean([x['seconds']+x['shared_preparation_seconds'] for x in costs])),
 'execution_seconds':float(np.mean([x['seconds'] for x in costs]))})
# Repeating oracle arms across training seeds changes the sampling stream; this is MC, not learning variance.
report={'summary':summary,'budget_seconds':r['budget_seconds'],'failure_seconds':r['prior_failure_seconds'],
 'audit':{'complete_unique_graph_rows':len(rows),'finite_values':True,'source_hash_matches':True},
 'evidence':r['evidence'],'historical_training':r['historical_training']}
(ROOT/'docs/reports/QK12_RESULT.json').write_text(json.dumps(report,indent=2)+'\n')
labels={'learned':'Learned V / learned S','zero_exact':'Zero V / exact S','learned_exact':'Learned V / exact S','analytic_exact':'Analytic V / exact S','analytic_learned':'Analytic V / learned S','physical':'Zero V / physical S'}
fig,axes=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
for j,split in enumerate(['test','ood']):
 for arm,label in labels.items():
  ss=[x for x in summary if x['split']==split and x['arm']==arm and x['steps']==32]
  axes[0,j].plot([x['paths'] for x in ss],[x['mae'] for x in ss],'-o',label=label)
  axes[1,j].plot([x['standalone_seconds'] for x in ss],[x['mae'] for x in ss],'-o',label=label)
 axes[0,j].set(xscale='log',xlabel='Paths (32 steps)',ylabel='Development MAE (kBT)',title=split)
 axes[1,j].set(xscale='log',xlabel='Measured execution + shared preparation (s)',ylabel='Development MAE (kBT)')
axes[0,0].legend(fontsize=8)
fig.suptitle('QK1/QK2: fixed checkpoints; reused development graphs; cost excludes historical training')
fig.savefig(ROOT/'docs/reports/QK12_CURVES.png',dpi=150)
lines=['# QK1/QK2 diagnostic results','',f"Date: 2026-10-17. Budget used: {r['budget_seconds']:.3f} / 1800 seconds, including {r['prior_failure_seconds']:.3f} seconds for the checkpoint-loading failure. No retraining or new heldout evaluation.",'',
 'Six arms, three frozen training seeds, two sampling streams, three path counts and two step counts; 24 previously viewed four-node graphs and 12 previously viewed six-node graphs. All are development diagnostics. Calibration passed before the full grid. The 30-minute ceiling is an execution wall-time bound with synchronized CUDA timing, not measured GPU kernel occupancy.','',
 '## Illustrative slice: 128 paths, 32 steps (full grid below)','', '| Split | Arm | MAE | Bias | ESS fraction | Standalone seconds |','|---|---|---:|---:|---:|---:|']
for x in summary:
 if x['paths']==128 and x['steps']==32:lines.append(f"| {x['split']} | {labels[x['arm']]} | {x['mae']:.5f} | {x['bias']:.5f} | {x['ess_fraction']:.3f} | {x['standalone_seconds']:.4f} |")
lines+=['', '## Full registered grid', '', '| Split | Paths | Steps | Arm | MAE | ESS fraction | Low ESS rows |', '|---|---:|---:|---|---:|---:|---:|']
for x in summary:
 lines.append(f"| {x['split']} | {x['paths']} | {x['steps']} | {labels[x['arm']]} | {x['mae']:.5f} | {x['ess_fraction']:.3f} | {x['low_ess_count']} |")
lines+=['','![Cost curves](QK12_CURVES.png)','','## Interpretation', '', 'At 32 steps, replacing the learned score with the exact interpolant score improves learned-velocity MAE on both development splits at all three path counts. Analytic velocity with exact score improves further. At 128 paths, learned/exact reduces MAE relative to learned/learned by about16% on four-node graphs and20% on six-node graphs. This supports score error as one limitation, not the sole limitation.', '', 'At8 steps, even analytic velocity/exact score performs poorly: at512 paths, six-node MAE is0.24038 and48 of72 graph-stream rows have ESS below10%, versus MAE0.02581 at32 steps with no low-ESS rows. Analytic fields are not sufficient for low variance under coarse Euler transitions. Increasing path count alone does not repair this setting.', '', 'The exact physical-score comparator remains strong and cheaper in this implementation. No learned-arm compute-efficiency superiority is established. Next prioritize a separately frozen discretization/interpolant-schedule diagnostic and score-target intervention, retaining the physical comparator. Do not silently enlarge the network or retune on these results. Human/RSI search has not started.', '', '## Audit and limitations','',f"Verified {len(rows)} unique graph-level rows cover the exact registered grid, with finite estimates and valid ESS; source/config hashes match. Mathematical preflight checked covariance continuity, finite differences, endpoint scores and independent Gaussian conditional regression. These checks do not independently replay every simulated path: work arrays were not saved by this diagnostic runner.",'',
 'The analytic reference follows the independent-endpoint stochastic interpolant used by training; it is not a globally optimal transport bound. The exact physical score follows a different density path. Finite-step discretization can leave variance even for analytic fields. Shared preparation computes all reference tables and is charged to every arm as a conservative common overhead; timings are not optimized standalone implementations and do not include original model training. Inference records include endpoint-energy and network/physical-score call counts. No new training cost is incurred; original FEAT training costs are retained separately in the JSON.','',
 'Within each training-seed/sampling-seed/graph/grid cell, arms share initial samples and noise. Different training seeds also change the sampling stream, so between-training-seed variation is not pure training variance; sampling variation is separately estimated conditional on each checkpoint using two streams. Oracle repetitions quantify sampling effects, not independent models. Three checkpoints and two streams are exploratory, not significance tests.','',
 'Initial attempt failed before calibration due to a checkpoint payload interface mismatch. The corrected attempt reads state_dict and verifies seed/update metadata; original weights and settings are unchanged. Failure preserved at runs/qk12-diagnostic-01/failure.json; complete raw rows and costs at runs/qk12-diagnostic-02/results.json. Frozen source commits:6750eee, interface correction5d83394; original Qiankun bce0ffe801040a78442140b28f3d07963eafe598. Crystal-Builder not used. No LLM calls, G1/G2 holdout access, branch merge or remote push.','',
 '[Protocol](../QK12_PROTOCOL.md). [Full grid summary](QK12_RESULT.json).']
(ROOT/'docs/reports/QK12_RESULT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print(json.dumps({'seconds':r['budget_seconds'],'rows':len(rows),'primary':[x for x in summary if x['paths']==128 and x['steps']==32]},indent=2))

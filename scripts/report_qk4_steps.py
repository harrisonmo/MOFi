"""Audit the QK4 grid, reproduce its QK12 anchor, and report cost curves."""
import hashlib
import itertools
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
r=json.loads((ROOT/'runs/qk4-steps-01/results.json').read_text());c=r['evidence']['config']
assert r['status']=='completed' and r['budget_seconds']<900
for file,key in [('scripts/run_qk4_steps.py','script_sha256'),('configs/qk4-steps.json','config_sha256')]:
 assert hashlib.sha256((ROOT/file).read_bytes()).hexdigest()==r['evidence'][key]
rows=[x for x in r['records'] if not x['calibration']]
def key(x):return tuple(x[k] for k in ['seed','mc','split','graph','paths','steps','arm'])
expected={(s,m,sp,g,128,k,a) for s,m,k,a in itertools.product(c['training_seeds'],c['sampling_seeds'],c['steps'],c['arms']) for sp,n in [('test',24),('ood',12)] for g in range(n)}
assert len(rows)==len(expected) and {key(x) for x in rows}==expected
old=json.loads((ROOT/'runs/qk12-diagnostic-02/results.json').read_text())
anchors={key(x):x for x in old['records'] if not x['calibration'] and x['paths']==128 and x['steps']==32}
maxdiff=0.
for x in rows:
 assert np.isfinite([x['prediction'],x['truth'],x['ess']]).all() and 1-1e-9<=x['ess']<=128+1e-9
 if x['steps']==32:
  y=anchors[key(x)]
  for field in ['prediction','truth','ess']:
   diff=abs(x[field]-y[field]);maxdiff=max(maxdiff,diff);assert diff<1e-10
summary=[]
for split,arm,steps in itertools.product(['test','ood'],c['arms'],c['steps']):
 rr=[x for x in rows if (x['split'],x['arm'],x['steps'])==(split,arm,steps)]
 err=np.array([x['prediction']-x['truth'] for x in rr]);ess=np.array([x['ess']/128 for x in rr])
 tt=[x for x in r['timings'] if not x['calibration'] and (x['split'],x['arm'],x['steps'])==(split,arm,steps)]
 assert len(tt)==6
 summary.append({'split':split,'arm':arm,'steps':steps,'mae':float(abs(err).mean()),'rmse':float(np.sqrt((err**2).mean())),
 'bias':float(err.mean()),'ess_fraction':float(ess.mean()),'low_ess':int((ess<.1).sum()),
 'execution_seconds':float(np.mean([x['seconds'] for x in tt])),
 'standalone_seconds':float(np.mean([x['seconds']+x['shared_preparation_seconds'] for x in tt]))})
report={'summary':summary,'seconds':r['budget_seconds'],'audit':{'unique_rows':len(rows),'anchor_rows':len(anchors),'anchor_max_abs_difference':maxdiff,'finite':True},'evidence':r['evidence']}
(ROOT/'docs/reports/QK4_STEPS_RESULT.json').write_text(json.dumps(report,indent=2)+'\n')
labels={'learned':'Learned V/S','zero_exact':'Zero V / exact S','learned_exact':'Learned V / exact S','analytic_exact':'Analytic V/S','analytic_learned':'Analytic V / learned S','physical':'Physical score'}
fig,axes=plt.subplots(2,2,figsize=(11,8),constrained_layout=True)
for j,split in enumerate(['test','ood']):
 for arm in c['arms']:
  ss=[x for x in summary if x['split']==split and x['arm']==arm]
  axes[0,j].plot([x['steps'] for x in ss],[x['mae'] for x in ss],'-o',label=labels[arm])
  axes[1,j].plot([x['standalone_seconds'] for x in ss],[x['mae'] for x in ss],'-o')
 axes[0,j].set(title=split,xlabel='Steps (128 paths)',ylabel='MAE (kBT)')
 axes[1,j].set(xlabel='Execution + common preparation (seconds)',ylabel='MAE (kBT)')
axes[0,0].legend(fontsize=8)
fig.suptitle('QK4: existing checkpoints and previously viewed development graphs')
fig.savefig(ROOT/'docs/reports/QK4_STEPS_CURVES.png',dpi=150)
lines=['# QK4 step-count diagnostic','', 'Date:2026-10-17. No retraining, new holdout, model selection or external science calls.', '',
 f"Completed six arms, three checkpoints, two sampling streams,128paths and32/64/128steps in {r['budget_seconds']:.3f}/900seconds including calibration. Analytic identity checks and finite calibration passed. All {len(rows)} graph rows cover the registered grid. All {len(anchors)}32-step anchor rows reproduce QK12 within1e-10 (maximum difference {maxdiff:.3g}). Source/config hashes and finite-value/ESS bounds pass.", '',
 '| Split | Arm | Steps | MAE | Bias | ESS fraction | Low ESS rows | Execution seconds | Standalone seconds |', '|---|---|---:|---:|---:|---:|---:|---:|---:|']
for x in summary:lines.append(f"| {x['split']} | {labels[x['arm']]} | {x['steps']} | {x['mae']:.6f} | {x['bias']:.6f} | {x['ess_fraction']:.3f} | {x['low_ess']} | {x['execution_seconds']:.4f} | {x['standalone_seconds']:.4f} |")
lines+=['', '![Curves](QK4_STEPS_CURVES.png)', '', '## Findings', '', 'Learned-field MAE falls from0.06332 to0.04317 to0.04276 on four-node development graphs, and0.08014 to0.06749 to0.05477 on six-node graphs. Learned ESS fractions increase from0.609 to0.725 and0.471 to0.598 respectively. Moving from32 to128steps costs roughly3.6-3.8times the measured learned-field execution time. The four-node MAE gain from64 to128steps is small in these samples; this is not a statistical plateau claim.', '', 'Analytic velocity and score improve much more:128-step MAE0.01517/0.01778 and ESS fractions0.954/0.933. Thus finer stepping exposes remaining learned-field limitations; discretization sensitivity alone does not explain the performance gap. The physical-score comparator changes little with step count. No equal-time superiority of learned transport is established.', '', 'Next propose a separately frozen time-sampling intervention with unchanged loss weight and inference budget, or a deliberately budget-matched paths-versus-steps comparison. Do not silently adopt128steps as a free accuracy improvement or claim optimality from these three settings. No new experiment or model promotion follows automatically.', '', '## Interpretation limits', '',
 'All graphs were previously inspected and are development diagnostics. Initial samples and seeds are matched; arms at each step count share all noise, but different step counts do not use a coupled Brownian path. Finite-sample Monte Carlo fluctuations can obscure monotone convergence. The analytic interpolant fields are not a globally optimal transport bound. These results describe this Euler-kernel implementation, not universal superiority of a step count.', '',
 'Standalone cost includes common preparation of all analytic tables, conservatively charged to each arm, and is not an optimized implementation benchmark. Original training cost is sunk and excluded from current inference curves. The physical comparator uses another density path. Shared checkpoints with only two sampling streams do not establish significance. Work arrays were not saved in this diagnostic; independent checks cover recorded predictions/metrics and the previous anchor, not full trajectory replay.', '',
 'Full records and operation counts: runs/qk4-steps-01/results.json. Frozen runner commit:b7af850. Qiankun base:bce0ffe801040a78442140b28f3d07963eafe598. No G1/G2 heldout access, branch merge or push. [Protocol](../QK4_STEPS_PROTOCOL.md), [machine-readable summary](QK4_STEPS_RESULT.json).']
(ROOT/'docs/reports/QK4_STEPS_RESULT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print(json.dumps({'seconds':r['budget_seconds'],'audit':report['audit'],'selected':[x for x in summary if x['arm'] in ['learned','analytic_exact','physical']]},indent=2))

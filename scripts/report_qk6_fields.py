"""Audit paired field errors and summarize the registered time grid."""
import hashlib,itertools,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
r=json.loads((ROOT/'runs/qk6-fields-01/results.json').read_text());c=r['provenance']['config'];rows=r['rows']
assert r['status']=='completed' and r['seconds']<300
for f,k in [('scripts/run_qk6_fields.py','script_sha256'),('scripts/run_qk12_diagnostic.py','analytic_module_sha256'),('configs/qk6-field-errors.json','config_sha256')]:assert hashlib.sha256((ROOT/f).read_bytes()).hexdigest()==r['provenance'][k]
expected={(sp,g,mc,t,recipe,seed) for sp,n in [('test',24),('ood',12)] for g in range(n) for mc,t,recipe,seed in itertools.product(c['sampling_seeds'],c['times'],c['recipes'],c['seeds'])}
key=lambda x:tuple(x[k] for k in ['split','graph','mc','time','recipe','seed'])
assert len(rows)==len(expected) and {key(x) for x in rows}==expected
hashes={(x['split'],x['mc'],x['time']):x['sha256'] for x in r['samples']}
metrics=['velocity_mse','score_mse','weighted_score_mse','forward_drift_mse','backward_drift_mse','velocity_reference_norm','score_reference_norm']
for x in rows:
 assert x['sample_sha256']==hashes[x['split'],x['mc'],x['time']]
 assert np.isfinite([x[k] for k in metrics]).all() and all(x[k]>=0 for k in metrics)
 np.testing.assert_allclose(x['forward_drift_mse'],x['velocity_mse']+.0625*x['score_mse']+.5*x['cross_error'],rtol=1e-10,atol=1e-10)
summary=[]
for split,recipe,t in itertools.product(['test','ood'],c['recipes'],c['times']):
 ss=[x for x in rows if (x['split'],x['recipe'],x['time'])==(split,recipe,t)]
 summary.append({'split':split,'recipe':recipe,'time':t,**{k:float(np.mean([x[k] for x in ss])) for k in metrics}})
regions=[]
for split,recipe,region in itertools.product(['test','ood'],c['recipes'],['early','middle','late']):
 ss=[x for x in summary if x['split']==split and x['recipe']==recipe and (('early' if x['time']<=.1 else 'late' if x['time']>=.9 else 'middle')==region)]
 regions.append({'split':split,'recipe':recipe,'region':region,**{k:float(np.mean([x[k] for x in ss])) for k in metrics}})
for x in summary+regions:
 for field in ['velocity','score']:
  den=x[field+'_reference_norm']
  x[field+'_normalized_mse']=x[field+'_mse']/den if den>1e-12 else None
report={'seconds':r['seconds'],'summary':summary,'regions':regions,'audit':{'unique_rows':len(rows),'paired_sample_batches':len(hashes),'drift_identity':True,'finite':True},'checks':r['checks'],'checkpoints':r['checkpoints'],'provenance':r['provenance']}
(ROOT/'docs/reports/QK6_FIELDS_RESULT.json').write_text(json.dumps(report,indent=2)+'\n')
fig,axes=plt.subplots(3,2,figsize=(11,10),constrained_layout=True)
for j,split in enumerate(['test','ood']):
 for recipe in c['recipes']:
  ss=[x for x in summary if x['split']==split and x['recipe']==recipe]
  for i,metric in enumerate(['velocity_mse','score_mse','forward_drift_mse']):
   axes[i,j].plot([x['time'] for x in ss],[x[metric] for x in ss],'-o',markersize=3,label=recipe)
   axes[i,j].set(yscale='log',xlabel='Interpolant time',ylabel=metric,title=split)
axes[0,0].legend();fig.suptitle('QK6: exact marginal states, paired across all recipes and checkpoints')
fig.savefig(ROOT/'docs/reports/QK6_FIELD_CURVES.png',dpi=150)
lines=['# QK6: time-resolved field diagnostics','', 'Date:2026-10-17. Existing checkpoints only; no training or new final evaluation.', '',f"Execution {r['seconds']:.3f}/300seconds. Audit covers {len(rows)} unique graph/checkpoint/time/stream rows and {len(hashes)} sample batches. Every recipe and checkpoint shares the same states within each graph/time/stream. Nine model hashes, source/config hashes, analytic identities, empirical covariance and forward/backward drift expansions were checked. Empirical covariance relative error: {r['checks']['sample_covariance_relative_error']:.5f}.",'',
 '| Split | Recipe | Region | Velocity MSE | Score MSE | Weighted score MSE | Forward drift MSE |','|---|---|---|---:|---:|---:|---:|']
for x in regions:lines.append(f"| {x['split']} | {x['recipe']} | {x['region']} | {x['velocity_mse']:.5f} | {x['score_mse']:.5f} | {x['weighted_score_mse']:.5f} | {x['forward_drift_mse']:.5f} |")
lines+=['','![Field curves](QK6_FIELD_CURVES.png)','','## Findings and next hypothesis', '', 'Score4 improves early/middle score MSE but raises early velocity MSE from0.02579 to0.03847 on four-node graphs and0.02506 to0.03787 on six-node graphs. Late score MSE also worsens (0.05617 to0.07693 and0.05879 to0.08396). A global score multiplier does not improve all times or both fields.', '', 'Endpoint sampling improves early/late velocity but worsens middle velocity MSE by about54% and43% on the two splits. Late score MSE rises about92% and59%. More endpoint samples do not automatically resolve the vanishing gamma-weighted DSM signal. The forward/backward drift errors include velocity-score cross terms, so separate field MSE alone is not a JE-quality guarantee.', '', 'A targeted next hypothesis is to separate field optimization or use a staged update that explicitly preserves the parent velocity while fitting score. Freezing a shared encoder or adjusting only a score head must be evaluated for capacity limitations; merely naming an update score-only is insufficient. This is a proposed next experiment, not executed here. Before claiming causal improvement, evaluate on actual sampler trajectories as well as exact marginals and retain the original JE gate.', '', '## Scope and interpretation','',
 'Regions are early t<=0.1, middle0.1<t<0.9, and late t>=0.9. Means are discrete registered-grid averages, not uniform-time integrals. Raw per-coordinate errors are compared within a field and graph-size group; velocity and score have different scales. Gamma-weighted score error approaches zero at endpoints regardless of raw score accuracy. The JSON includes reference squared norms for context.', '',
 'States come from the exact Gaussian interpolant marginal, not from the learned sampler trajectory. Therefore these diagnostics do not directly measure on-policy error accumulation or establish the cause of a free-energy error. A smaller field error need not imply lower finite-path JE MAE. All graph families were already viewed; no untouched holdout claims.', '',
 'Original parent, rejected score4, and rejected endpoint-time recipes use the same architecture. The latter two interventions can change both velocity and score through shared parameters or changed time weighting. No new candidate is promoted here. Sampling seeds are now shared across training checkpoints, enabling a crossed diagnostic; previous JE experiments had some checkpoint/sampling mixing.', '',
 'Frozen runner:716c6ba. Qiankun source:bce0ffe801040a78442140b28f3d07963eafe598. No Crystal-Builder, LLM or G1/G2 holdout use. Raw rows/sample hashes: runs/qk6-fields-01/results.json. [Protocol](../QK6_FIELD_PROTOCOL.md), [summary and provenance](QK6_FIELDS_RESULT.json).']
(ROOT/'docs/reports/QK6_FIELDS_RESULT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print(json.dumps({'seconds':r['seconds'],'audit':report['audit'],'regions':regions},indent=2))

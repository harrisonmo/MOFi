"""Audit QK7 work arrays, matched controls and frozen development gates."""
import hashlib,json
from pathlib import Path
import numpy as np
import torch
from scipy.special import logsumexp
ROOT=Path(__file__).resolve().parents[1];RUN=ROOT/'runs/qk7-frozen-01'
r=json.loads((RUN/'results.json').read_text());cfg=r['provenance']['config'];rows=r['records']
assert r['status']=='completed' and r['seconds']<900 and len(rows)==648
assert hashlib.sha256((ROOT/'scripts/run_qk7_frozen_velocity.py').read_bytes()).hexdigest()==r['provenance']['script_sha256']
assert hashlib.sha256((ROOT/'configs/qk7-frozen-velocity.json').read_bytes()).hexdigest()==r['provenance']['config_sha256']
keys=[tuple(x[k] for k in ['training_seed','recipe','stream','split','graph_id']) for x in rows];assert len(set(keys))==648
metrics=[];count=0
old=json.loads((ROOT/'runs/qk5-time-01/results.json').read_text())['records']
for seed in cfg['seeds']:
 for arm in cfg['arms']:
  folder=RUN/f'seed-{seed}-{arm}'
  cost=next(x for x in r['costs'] if x['seed']==seed and x['arm']==arm)
  if arm!='parent':
   assert cost['updates']==600
   assert hashlib.sha256((folder/'student.pt').read_bytes()).hexdigest()==cost['checkpoint_sha256']
  for split,n in [('test',24),('ood',12)]:
   rr=[x for x in rows if x['training_seed']==seed and x['recipe']==arm and x['split']==split]
   assert len(rr)==2*n
   for stream in cfg['streams']:
    ss=sorted([x for x in rr if x['stream']==stream],key=lambda x:x['graph_id']);assert [x['graph_id'] for x in ss]==list(range(n))
    chunks=torch.load(folder/f'stream-{stream}'/f'{split}-work.pt',weights_only=True)
    for model in ['feat_gnn_jarzynski','no_transport_jarzynski']:
     w=np.concatenate([x[model].numpy() for x in chunks]);assert w.shape==(n,128)
     je=-(logsumexp(-w,axis=1)-np.log(128));ess=np.exp(2*logsumexp(-w,axis=1)-logsumexp(-2*w,axis=1))
     np.testing.assert_allclose(je,[x['predictions'][model] for x in ss],rtol=1e-10,atol=1e-10)
     np.testing.assert_allclose(ess,[x['ess'][model] for x in ss],rtol=1e-10,atol=1e-10);count+=w.size
    for x in ss:
     parent=next(y for y in rows if y['recipe']=='parent' and all(y[k]==x[k] for k in ['training_seed','split','stream','graph_id']))
     for model in ['supervised_gnn','no_transport_jarzynski']:assert x['predictions'][model]==parent['predictions'][model]
     if arm=='parent':
      historical=next(y for y in old if y['recipe']=='uniform' and all(y[k]==x[k] for k in ['training_seed','split','stream','graph_id']))
      assert x['predictions']==historical['predictions']
   err=np.array([x['predictions']['feat_gnn_jarzynski']-x['truth'] for x in rr]);ess=np.array([x['ess']['feat_gnn_jarzynski']/128 for x in rr])
   metrics.append({'seed':seed,'arm':arm,'split':split,'mae':float(abs(err).mean()),'rmse':float(np.sqrt((err**2).mean())), 'ess_fraction':float(ess.mean()),'low_ess':int((ess<.1).sum())})
summary=[];gates={}
for arm in cfg['arms']:
 gates[arm]=True
 for split in ['test','ood']:
  a=[x for x in metrics if x['arm']=='parent' and x['split']==split];b=[x for x in metrics if x['arm']==arm and x['split']==split]
  ma=np.mean([x['mae'] for x in a]);mb=np.mean([x['mae'] for x in b]);gain=1-mb/ma
  passed=bool(gain>=.05 and all(y['mae']<=1.1*x['mae'] for x,y in zip(a,b)) and sum(x['low_ess'] for x in b)<=sum(x['low_ess'] for x in a));gates[arm]&=passed
  tt=[x['feat_gnn_jarzynski'] for x in r['timings'] if x['recipe']==arm and x['split']==split]
  summary.append({'arm':arm,'split':split,'mae':float(mb),'gain':float(gain),'gate':passed,'ess_fraction':float(np.mean([x['ess_fraction'] for x in b])),'inference_seconds':float(np.mean(tt))})
report={'summary':summary,'gates':gates,'per_seed':metrics,'seconds':r['seconds'],'audit':{'work_values':count,'parent_predictions_reproduced':True,'matched_controls':True},'costs':r['costs'],'provenance':r['provenance']}
(ROOT/'docs/reports/QK7_RESULT.json').write_text(json.dumps(report,indent=2)+'\n')
lines=['# QK7 frozen velocity continuation','', 'Date:2026-10-17. Assistant-designed development intervention, not a human/RSI comparison.', '',f"Completed in {r['seconds']:.3f}/900seconds including calibration. Three frozen parents, three600-update joint continuations and three600-update score-only continuations, each with two paired evaluation streams. Gate results: joint={gates['joint']}, score-only={gates['score_only']}.",'',
 '| Arm | Split | MAE | Gain over parent | ESS fraction | Inference seconds | Gate |','|---|---|---:|---:|---:|---:|---|']
for x in summary:lines.append(f"| {x['arm']} | {x['split']} | {x['mae']:.6f} | {100*x['gain']:.2f}% | {x['ess_fraction']:.3f} | {x['inference_seconds']:.3f} | {x['gate']} |")
lines+=['','## Decision', '', 'Retain the original parent. Score-only improves four-node MAE by8.67%, but six-node MAE by only0.91%, missing the two-split gate. Joint continuation improves four-node MAE by4.86% and worsens six-node MAE by20.23%. Score isolation is descriptively better than this matched-update joint control, but neither candidate passes. Do not relax the gate after observing the result.', '', 'The composite increases measured inference time by roughly48% and doubles deployed network parameters. The present evidence does not establish equal-cost benefit or unseen-family transfer. A next diagnostic could cross the same frozen parent velocity with the parent/joint/score-only scores to distinguish output mixing from training-objective effects, but that experiment is not run here. No additional tuning is warranted under this completed budget.', '', '## Controls and limitations','',
 'Score-only inference uses parent velocity exactly and student score. Parent parameters remain frozen with no gradients; calibration and formal checks verify velocity equality. Both continuations reset Adam and use identical training draws, update count, model initialization and clipping; their losses differ. Uniform time sampling and128paths/32steps remain fixed. Both losses start from the original parent, not the previously rejected candidates.', '',
 'Composite inference stores two17412-parameter networks (34824 total) and performs two network forwards per field evaluation; parent and joint use17412parameters and one forward. This is a fixed-velocity attribution experiment, not a matched-memory or matched-inference-compute superiority test. Additional600updates are charged, and historical1200updates are shared sunk cost. Timing includes the existing evaluator overhead; repeated microtimings are not a hardware benchmark.', '',
 f"Independent NumPy/SciPy reduction verifies {count} saved work values and all JE/ESS predictions. All648graph-stream records are unique and cover the registered splits. Unchanged physical/supervised control predictions match; parent predictions reproduce QK5 uniform results exactly. Source/config and six continued-checkpoint hashes match. Raw model/work files remain in runs/qk7-frozen-01.",'',
 'All24four-node and12six-node graphs were already viewed and are development diagnostics; no unseen-task claims or final test access. Frozen acceptance gate requires5% improvement on both splits, no seed/split regression above10%, and no extra low-ESS rows. Passing parent comparison alone does not prove score isolation caused improvement; the joint continuation is the matched-update control. No tuning, LLM calls, Crystal-Builder calls, branch merges or remote push.', '',
 'Frozen source:f8b015d; Qiankun base:bce0ffe801040a78442140b28f3d07963eafe598. [Protocol](../QK7_FROZEN_VELOCITY_PROTOCOL.md). [Full summary](QK7_RESULT.json).']
(ROOT/'docs/reports/QK7_RESULT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print(json.dumps({'summary':summary,'gates':gates,'seconds':r['seconds'],'work_values':count},indent=2))

"""Independently audit saved work arrays and apply the frozen QK3 gate."""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from scipy.special import logsumexp
ROOT=Path(__file__).resolve().parents[1];RUN=ROOT/'runs/qk3-score-01'
r=json.loads((RUN/'results.json').read_text());cfg=r['provenance']['config']
assert r['status']=='completed' and r['seconds']<cfg['total_seconds_limit']
assert hashlib.sha256((ROOT/'scripts/run_qk3_score.py').read_bytes()).hexdigest()==r['provenance']['script_sha256']
assert hashlib.sha256((ROOT/'configs/qk3-score-weight.json').read_bytes()).hexdigest()==r['provenance']['config_sha256']
assert len(r['training'])==6 and len(r['records'])==432
keys=[(x['training_seed'],x['score_weight'],x['stream'],x['split'],x['graph_id']) for x in r['records']];assert len(set(keys))==432
count=0;metrics=[]
for s in cfg['seeds']:
 for weight in cfg['score_weights']:
  folder=RUN/f'seed-{s}-weight-{weight}'
  cost=next(x for x in r['training'] if x['seed']==s and x['score_weight']==weight)
  assert cost['updates']==1200 and cost['seconds']<900
  assert hashlib.sha256((folder/'feat_gnn.pt').read_bytes()).hexdigest()==cost['checkpoint_sha256']
  for split,n in [('test',24),('ood',12)]:
   rows=[x for x in r['records'] if x['training_seed']==s and x['score_weight']==weight and x['split']==split]
   assert len(rows)==2*n
   for stream in cfg['evaluation_stream_offsets']:
    rr=sorted([x for x in rows if x['stream']==stream],key=lambda x:x['graph_id'])
    assert [x['graph_id'] for x in rr]==list(range(n))
    work=torch.load(folder/f'stream-{stream}'/f'{split}-work.pt',weights_only=True)
    for arm in ['feat_gnn_jarzynski','no_transport_jarzynski']:
     w=np.concatenate([x[arm].numpy() for x in work]);assert w.shape==(n,128)
     pred=-(logsumexp(-w,axis=1)-np.log(128));ess=np.exp(2*logsumexp(-w,axis=1)-logsumexp(-2*w,axis=1))
     np.testing.assert_allclose(pred,[x['predictions'][arm] for x in rr],rtol=1e-10,atol=1e-10)
     np.testing.assert_allclose(ess,[x['ess'][arm] for x in rr],rtol=1e-10,atol=1e-10)
     count+=w.size
   err=np.array([x['predictions']['feat_gnn_jarzynski']-x['truth'] for x in rows]);ess=np.array([x['ess']['feat_gnn_jarzynski'] for x in rows])
   metrics.append({'seed':s,'weight':weight,'split':split,'mae':float(abs(err).mean()),'rmse':float(np.sqrt((err**2).mean())),
                   'bias':float(err.mean()),'ess_fraction':float(ess.mean()/128),'low_ess':int((ess<12.8).sum())})
# Unchanged supervised/physical arms confirm the evaluator uses matching samples and graphs.
for x in r['records']:
 if x['score_weight']==1:
  y=next(y for y in r['records'] if y['score_weight']==4 and all(y[k]==x[k] for k in ['training_seed','stream','split','graph_id']))
  assert x['truth']==y['truth']
  for arm in ['supervised_gnn','no_transport_jarzynski']:assert x['predictions'][arm]==y['predictions'][arm]
summary=[];accept=True
for split in ['test','ood']:
 a=[x for x in metrics if x['split']==split and x['weight']==1];b=[x for x in metrics if x['split']==split and x['weight']==4]
 mean_a=np.mean([x['mae'] for x in a]);mean_b=np.mean([x['mae'] for x in b])
 gains=1-mean_b/mean_a
 gate=bool(gains>=.05 and all(y['mae']<=1.1*x['mae'] for x,y in zip(a,b)) and sum(y['low_ess'] for y in b)<=sum(x['low_ess'] for x in a))
 accept=accept and gate
 summary.append({'split':split,'parent_mae':float(mean_a),'candidate_mae':float(mean_b),'relative_gain':float(gains),'gate':gate,
                 'parent_low_ess':sum(x['low_ess'] for x in a),'candidate_low_ess':sum(x['low_ess'] for x in b)})
report={'accepted_for_development':accept,'summary':summary,'per_seed':metrics,'seconds':r['seconds'],
        'audit':{'independently_recomputed_work_values':count,'paired_physical_and_supervised_predictions_exact':True,'parent_replay_exact':all(x['parent_replay_exact'] for x in r['training'] if x['score_weight']==1)},
        'training':r['training'],'timings':r['timings'],'provenance':r['provenance']}
(ROOT/'docs/reports/QK3_SCORE_RESULT.json').write_text(json.dumps(report,indent=2)+'\n')
lines=['# QK3: score-loss reweighting result','', 'Date:2026-10-17. Assistant-designed intervention, not a human-versus-RSI comparison.', '',
 f"Decision: **{'accept for continued development' if accept else 'reject candidate; retain parent'}** under the predeclared gate. Total execution {r['seconds']:.3f}/900seconds, including calibration. All six runs finished1200updates without tuning or retries.",'',
 '| Split | Parent MAE | Weight4 MAE | Relative improvement | Gate |','|---|---:|---:|---:|---|']
for x in summary:lines.append(f"| {x['split']} | {x['parent_mae']:.6f} | {x['candidate_mae']:.6f} | {100*x['relative_gain']:.2f}% | {x['gate']} |")
lines+=['', '| Seed | Split | Weight | MAE | ESS fraction | Low ESS rows |', '|---:|---|---:|---:|---:|---:|']
for x in metrics:lines.append(f"| {x['seed']} | {x['split']} | {x['weight']} | {x['mae']:.6f} | {x['ess_fraction']:.3f} | {x['low_ess']} |")
lines+=['','## Method and evidence','',
 'Only the existing weighted DSM loss coefficient changed from1 to4; flow coefficient, model, data, optimizer,1200updates and training random draws are unchanged. Shared parameters mean velocity can change indirectly. Two paired evaluation streams use128paths/32steps on the previously viewed24four-node and12six-node graphs. The original test/ood names denote development diagnostics here; no new holdout or G1/G2 data was opened.', '',
 f"All three fresh parent models exactly reproduced the original checkpoint tensors. Loss identity and finite-gradient checks preceded a20-update-per-arm calibration. Independent NumPy/SciPy reduction verified {count} saved work values, JE estimates and ESS. Six checkpoint hashes match; unchanged physical/supervised predictions match exactly between recipes, supporting sampling pairing. Complete432graph-stream records remain in runs/qk3-score-01/results.json; raw weights and work arrays remain local.",'',
 'Training seeds and sampling variability remain partly mixed across checkpoints; this is a small development intervention, not a statistically established generalization gain. Weight4 was one fixed candidate. No follow-up tuning was permitted by this gate. Calibration/failures and costs are retained; no LLM or external science-engine calls occurred.', '',
 'Frozen implementation:ed2eb8c. Qiankun source:bce0ffe801040a78442140b28f3d07963eafe598. Original source/config hashes, GPU identity and training costs are in the JSON. Crystal-Builder not used. No branch merge or remote push.', '',
 '[Protocol](../QK3_SCORE_PROTOCOL.md). [Full summary](QK3_SCORE_RESULT.json).']
(ROOT/'docs/reports/QK3_SCORE_RESULT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print(json.dumps({'summary':summary,'accepted':accept,'seconds':r['seconds'],'audit':report['audit']},indent=2))

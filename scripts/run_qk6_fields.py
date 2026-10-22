"""Evaluate fixed field checkpoints on paired exact Gaussian marginal states."""
import argparse
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import torch
from run_qk12_diagnostic import fields, validate_math
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def tensor_sha(x):return hashlib.sha256(x.cpu().numpy().tobytes()).hexdigest()

def main():
 p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--prior',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 cfg=json.loads((ROOT/'configs/qk6-field-errors.json').read_text());sys.path.insert(0,str(a.source.resolve()))
 from rsi.jarzynski.models import TransportGNN
 from rsi.jarzynski.systems import precision
 assert torch.cuda.is_available()
 torch.set_num_threads(2);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 prior=json.loads((a.prior/'results.json').read_text())
 for n,v in prior['provenance']['source_hashes'].items():assert sha(a.source/n)==v
 a.output.mkdir(parents=True,exist_ok=False)
 def save(n,v):(a.output/n).write_text(json.dumps(v,indent=2)+'\n')
 start=time.perf_counter();deadline=start+cfg['seconds_limit'];rows=[];samples=[];checkpoints=[]
 def guard():
  if time.perf_counter()>deadline:raise TimeoutError('QK6 budget exceeded')
 provenance={'config':cfg,'script_sha256':sha(Path(__file__)),'analytic_module_sha256':sha(ROOT/'scripts/run_qk12_diagnostic.py'),
 'config_sha256':sha(ROOT/'configs/qk6-field-errors.json'),'prior_sha256':sha(a.prior/'results.json'),
 'data_sha256':sha(a.prior/'data.pt'),'torch':torch.__version__,'gpu':torch.cuda.get_device_name(0)}
 save('provenance.json',provenance)
 try:
  checks=validate_math();data=torch.load(a.prior/'data.pt',weights_only=True)
  # Independent empirical covariance check at an interior time on one training graph.
  graph={k:v[:1] for k,v in data['graphs']['train'].items()}
  c0,c1=[torch.linalg.inv(precision(graph,e)) for e in [0,1]]
  cc=fields(c0,c1,.37)[0][0];z=torch.randn(8192,4,dtype=torch.float64,generator=torch.Generator().manual_seed(991))
  xx=z@torch.linalg.cholesky(cc).T;emp=xx.T@xx/len(xx)
  rel=float(torch.linalg.norm(emp-cc)/torch.linalg.norm(cc));assert rel<.1
  checks['sample_covariance_relative_error']=rel
  models={}
  for recipe in cfg['recipes']:
   for seed in cfg['seeds']:
    if recipe=='parent':path=a.prior/f'seed-{seed}'/'feat_gnn.pt';expected=next(x for x in prior['seeds'] if x['seed']==seed)['training']['feat_gnn']['checkpoint_sha256']
    else:
     folder=ROOT/('runs/qk3-score-01' if recipe=='score4' else 'runs/qk5-time-01')
     result=json.loads((folder/'results.json').read_text())
     path=folder/(f'seed-{seed}-weight-4.0' if recipe=='score4' else f'seed-{seed}-recipe-endpoint')/'feat_gnn.pt'
     expected=next(x for x in result['training'] if x['seed']==seed and (x.get('score_weight')==4 if recipe=='score4' else x.get('recipe')=='endpoint'))['checkpoint_sha256']
    assert sha(path)==expected
    model=TransportGNN(32,2).cuda().eval();payload=torch.load(path,weights_only=True,map_location='cuda');model.load_state_dict(payload['state_dict'])
    models[recipe,seed]=model;checkpoints.append({'recipe':recipe,'seed':seed,'sha256':expected})
  save('checks.json',checks)
  with torch.no_grad():
   for split in ['test','ood']:
    graph=data['graphs'][split];ng=len(graph['a0']);n=cfg['samples_per_graph']
    c0,c1=[torch.linalg.inv(precision(graph,e)) for e in [0,1]]
    expanded={k:v.repeat_interleave(n,0).cuda().float() for k,v in graph.items()}
    for mc in cfg['sampling_seeds']:
     for ti,t in enumerate(cfg['times']):
      guard();cov,dc,vmat,smat=fields(c0,c1,t)
      generator=torch.Generator().manual_seed(mc*1000+ti+(1000000 if split=='ood' else 0))
      z=torch.randn(ng,n,cov.shape[-1],3,dtype=torch.float64,generator=generator)
      x=(torch.linalg.cholesky(cov)[:,None]@z).reshape(ng*n,cov.shape[-1],3)
      h=tensor_sha(x);samples.append({'split':split,'mc':mc,'time':t,'sha256':h})
      refv=(vmat[:,None]@(x.reshape(ng,n,cov.shape[-1],3))).reshape_as(x).cuda()
      refs=(smat[:,None]@(x.reshape(ng,n,cov.shape[-1],3))).reshape_as(x).cuda()
      x=x.cuda();tt=torch.full((len(x),),t,device='cuda',dtype=torch.float32)
      def mean(y):return y.reshape(ng,n,-1).mean((1,2)).cpu().tolist()
      for (recipe,seed),model in models.items():
       v,s=model(x.float(),tt,expanded);ev=v.double()-refv;es=s.double()-refs
       vm,sm,cross=[mean(y) for y in [ev.square(),es.square(),ev*es]]
       forward,backward=[mean((ev+sign*cfg['diffusion']*es).square()) for sign in [1,-1]]
       np.testing.assert_allclose(forward,np.array(vm)+.0625*np.array(sm)+.5*np.array(cross),rtol=1e-10,atol=1e-10)
       np.testing.assert_allclose(backward,np.array(vm)+.0625*np.array(sm)-.5*np.array(cross),rtol=1e-10,atol=1e-10)
       vn,sn=mean(refv.square()),mean(refs.square())
       for i in range(ng):rows.append({'split':split,'mc':mc,'time':t,'recipe':recipe,'seed':seed,'graph':i,'sample_sha256':h,
        'velocity_mse':vm[i],'score_mse':sm[i],'weighted_score_mse':(.5*math.sin(math.pi*t))**2*sm[i],
        'forward_drift_mse':forward[i],'backward_drift_mse':backward[i],'cross_error':cross[i],'velocity_reference_norm':vn[i],'score_reference_norm':sn[i]})
      torch.cuda.synchronize();guard()
     save('partial.json',{'rows':rows,'samples':samples})
     print(f'Completed {split} sampling seed {mc}',flush=True)
  save('results.json',{'status':'completed','seconds':time.perf_counter()-start,'rows':rows,'samples':samples,'checks':checks,'checkpoints':checkpoints,'provenance':provenance})
 except Exception as err:
  save('failure.json',{'exception':repr(err),'seconds':time.perf_counter()-start});raise
if __name__=='__main__':main()

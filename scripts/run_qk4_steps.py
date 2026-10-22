"""Frozen Gaussian interpolant diagnostics using existing Qiankun checkpoints."""
import argparse
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fields(c0, c1, t, scale=.5):
    eye = torch.eye(c0.shape[-1], dtype=c0.dtype, device=c0.device)
    gamma = scale * math.sin(math.pi*t)
    gd = scale * math.pi * math.cos(math.pi*t)
    c = (1-t)**2*c0+t*t*c1+gamma*gamma*eye
    dc = -2*(1-t)*c0+2*t*c1+2*gamma*gd*eye
    score = -torch.linalg.inv(c)
    velocity = -.5*dc@score
    return c, dc, velocity, score


def validate_math():
    rng = np.random.default_rng(19)
    a, b = rng.normal(size=(4,4)), rng.normal(size=(4,4))
    c0, c1 = [torch.tensor(x@x.T+np.eye(4), dtype=torch.float64) for x in [a,b]]
    for t in [0.,.13,.5,.87,1.]:
        c, dc, v, sc = fields(c0,c1,t)
        torch.testing.assert_close(v@c+c@v.T,dc,rtol=1e-11,atol=1e-11)
        torch.testing.assert_close(sc@c,-torch.eye(4,dtype=c.dtype),rtol=1e-11,atol=1e-11)
        derivative=(fields(c0,c1,t+1e-6)[0]-fields(c0,c1,t-1e-6)[0])/2e-6
        torch.testing.assert_close(dc,derivative,rtol=1e-7,atol=1e-7)
    torch.testing.assert_close(fields(c0,c1,0)[3],-torch.linalg.inv(c0))
    torch.testing.assert_close(fields(c0,c1,1)[3],-torch.linalg.inv(c1))
    # Independent block Gaussian conditional regression of dX/dt on X.
    t=.37; g=.5*math.sin(math.pi*t); gd=.5*math.pi*math.cos(math.pi*t)
    cov=torch.block_diag(c0,c1,torch.eye(4,dtype=c0.dtype))
    eye=torch.eye(4,dtype=c0.dtype)
    h=torch.cat([(1-t)*eye,t*eye,g*eye],dim=1)
    dh=torch.cat([-eye,eye,gd*eye],dim=1)
    reference=dh@cov@h.T@torch.linalg.inv(h@cov@h.T)
    torch.testing.assert_close(reference,fields(c0,c1,t)[2],rtol=1e-11,atol=1e-11)
    return {"covariance_continuity":True,"finite_difference":True,"endpoint_score":True,"conditional_regression":True}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--prior',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    cfg=json.loads((ROOT/'configs/qk4-steps.json').read_text())
    math_checks=validate_math()
    sys.path.insert(0,str(args.source.resolve()))
    from rsi.jarzynski.systems import precision, sample_endpoint
    from rsi.jarzynski.models import TransportGNN
    from rsi.jarzynski.estimators import simulate_forward, jarzynski, effective_sample_size
    assert torch.cuda.is_available()
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    prior=json.loads((args.prior/'results.json').read_text())
    assert prior['provenance']['git']['commit']==cfg['source_commit']
    for name,value in prior['provenance']['source_hashes'].items():assert sha(args.source/name)==value
    args.output.mkdir(parents=True,exist_ok=False)
    started=time.perf_counter()
    prior_charge=0.0
    deadline=started+cfg['gpu_seconds_limit']-prior_charge
    def guard():
        if time.perf_counter()>deadline:raise TimeoutError('Global diagnostic budget exceeded')
    def save(name,value):
        (args.output/name).write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')
    evidence={'config':cfg,'config_sha256':sha(ROOT/'configs/qk4-steps.json'),'script_sha256':sha(Path(__file__)),
              'prior_results_sha256':sha(args.prior/'results.json'),'math_checks':math_checks,
              'environment':{'torch':torch.__version__,'gpu':torch.cuda.get_device_name(0)},'prior_provenance':prior['provenance']}
    save('provenance.json',evidence)
    data=torch.load(args.prior/'data.pt',weights_only=True)
    records=[];timings=[]
    @torch.no_grad()
    def evaluate(model,graph,seed,mc,split,paths,steps,calibration=False):
        guard();torch.cuda.synchronize();begin=time.perf_counter()
        ngraphs=len(graph['a0']);nodes=graph['a0'].shape[1]
        k0,k1=precision(graph,0),precision(graph,1)
        c0,c1=torch.linalg.inv(k0)/cfg['beta'],torch.linalg.inv(k1)/cfg['beta']
        # Tables contain one matrix per graph, rather than one inverse per path.
        table=[fields(c0,c1,i/steps,cfg['gamma_scale']) for i in range(steps+1)]
        cpu={k:v.repeat_interleave(paths,0) for k,v in graph.items()}
        rs=mc*100000+seed*100+(10000 if split=='ood' else 0)
        initial=sample_endpoint(cpu,0,torch.Generator().manual_seed(rs),cfg['beta']).cuda()
        gpu={k:v.cuda() for k,v in cpu.items()};g32={k:v.float() for k,v in gpu.items()}
        vs=[x[2].repeat_interleave(paths,0).cuda() for x in table]
        ss=[x[3].repeat_interleave(paths,0).cuda() for x in table]
        ps=[(-cfg['beta']*((1-i/steps)*k0+i/steps*k1)).repeat_interleave(paths,0).cuda() for i in range(steps+1)]
        torch.cuda.synchronize();preparation=time.perf_counter()-begin
        # Independent eigenspectrum formula for harmonic reference free energy.
        truth=(graph['c1']-graph['c0']+1.5/cfg['beta']*(torch.linalg.eigvalsh(k1).log().sum(-1)-torch.linalg.eigvalsh(k0).log().sum(-1))).numpy()
        for arm in cfg['arms']:
            guard();torch.cuda.synchronize();begin=time.perf_counter();call=[0]
            def drift(x,t,unused):
                i=call[0];call[0]+=1
                if arm in ['learned','learned_exact','analytic_learned']:
                    v,s=model(x.float(),t.float(),g32);v=v.double();s=s.double()
                if arm in ['zero_exact','physical']:v=torch.zeros_like(x)
                if arm in ['analytic_exact','analytic_learned']:v=vs[i]@x
                if arm in ['zero_exact','learned_exact','analytic_exact']:s=ss[i]@x
                if arm=='physical':s=ps[i]@x
                return v,s
            result=simulate_forward(gpu,initial,drift,steps,cfg['diffusion'],torch.Generator(device='cuda').manual_seed(rs+79),beta=cfg['beta'])
            work=result['work'].reshape(ngraphs,paths)
            prediction=jarzynski(work,cfg['beta']);ess=effective_sample_size(work,cfg['beta'])
            assert torch.isfinite(prediction).all() and call[0]==steps+1
            torch.cuda.synchronize();seconds=time.perf_counter()-begin
            timings.append({'seed':seed,'mc':mc,'split':split,'paths':paths,'steps':steps,'arm':arm,'seconds':seconds,'shared_preparation_seconds':preparation,'calibration':calibration,
                            'energy_evaluations':2*ngraphs*paths,'network_batch_calls':steps+1 if arm in ['learned','learned_exact','analytic_learned'] else 0,
                            'physical_score_evaluations':ngraphs*paths*(steps+1) if arm=='physical' else 0})
            for i,(p,e) in enumerate(zip(prediction.cpu().tolist(),ess.cpu().tolist())):
                records.append({'seed':seed,'mc':mc,'split':split,'graph':i,'paths':paths,'steps':steps,'arm':arm,'prediction':p,'truth':float(truth[i]),'ess':e,'calibration':calibration})
        guard()
    try:
        for seed in cfg['training_seeds']:
            ck=args.prior/f'seed-{seed}'/'feat_gnn.pt'
            cost=next(s for s in prior['seeds'] if s['seed']==seed)['training']['feat_gnn']
            assert sha(ck)==cost['checkpoint_sha256']
            model=TransportGNN(width=32,layers=2).cuda().eval()
            payload=torch.load(ck,weights_only=True,map_location='cuda')
            assert payload['seed']==seed and payload['updates']==1200
            model.load_state_dict(payload['state_dict'])
            if seed==17:
                evaluate(model,{k:v[:1] for k,v in data['graphs']['train'].items()},seed,991,'calibration',32,8,True)
                save('calibration.json',{'admitted':True,'seconds':time.perf_counter()-started,'records':records.copy()})
            for mc in cfg['sampling_seeds']:
                for paths in cfg['paths']:
                    for steps in cfg['steps']:
                        for split in ['test','ood']:
                            evaluate(model,data['graphs'][split],seed,mc,split,paths,steps)
                        save('partial.json',{'records':records,'timings':timings})
                print(f'Completed training seed {seed}, sampling seed {mc}',flush=True)
        save('results.json',{'status':'completed','prior_failure_seconds':prior_charge,'budget_seconds':prior_charge+time.perf_counter()-started,'seconds':time.perf_counter()-started,'records':records,'timings':timings,'evidence':evidence,'historical_training':[s['training']['feat_gnn'] for s in prior['seeds']]})
        print('Completed',time.perf_counter()-started,flush=True)
    except Exception as exc:
        save('failure.json',{'exception':repr(exc),'seconds':time.perf_counter()-started})
        raise


if __name__=='__main__':main()

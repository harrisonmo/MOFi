"""Compare a frozen DSM weight intervention with a fresh parent replay."""
import argparse
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import torch
ROOT=Path(__file__).resolve().parents[1]

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def weighted_loss(model,graph,x0,x1,t,noise,gamma_scale=.5):
    t3=t[:,None,None]
    gamma=gamma_scale*torch.sin(math.pi*t3)
    gd=gamma_scale*math.pi*torch.cos(math.pi*t3)
    x=(1-t3)*x0+t3*x1+gamma*noise
    velocity,score=model(x,t,graph)
    flow=(velocity-(x1-x0+gd*noise)).square().mean()
    dsm=(gamma*score+noise).square().mean()
    return flow+4*dsm,{'flow':flow.detach(),'score':dsm.detach()}

def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--prior',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    cfg=json.loads((ROOT/'configs/qk3-score-weight.json').read_text())
    sys.path.insert(0,str(a.source.resolve()))
    import rsi.jarzynski.experiment as ex
    from rsi.jarzynski.models import FreeEnergyGNN,TransportGNN
    from rsi.jarzynski.systems import make_graphs
    prior=json.loads((a.prior/'results.json').read_text());base=prior['config'];original_loss=ex.interpolant_loss
    assert prior['provenance']['git']['commit']==cfg['source_commit']
    for n,v in prior['provenance']['source_hashes'].items():assert sha(a.source/n)==v
    assert torch.cuda.is_available()
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    a.output.mkdir(parents=True,exist_ok=False)
    def save(n,v):(a.output/n).write_text(json.dumps(v,indent=2)+'\n')
    start=time.monotonic();deadline=start+cfg['total_seconds_limit'];training=[];records=[];timings=[]
    provenance={'config':cfg,'config_sha256':sha(ROOT/'configs/qk3-score-weight.json'),'script_sha256':sha(Path(__file__)),
                'prior_sha256':sha(a.prior/'results.json'),'original_provenance':prior['provenance'],
                'environment':{'torch':torch.__version__,'gpu':torch.cuda.get_device_name(0)}}
    save('provenance.json',provenance)
    try:
        # Numerical identity checks of the candidate loss, before dependent training.
        g=make_graphs(8,4,991);pool=ex.training_pool(g,16,992,1.)
        model=TransportGNN(32,2);torch.manual_seed(33)
        x0,x1=pool[0][:,0].float(),pool[1][:,0].float();gg={k:v.float() for k,v in g.items()}
        tt=torch.linspace(.01,.99,8);noise=torch.randn_like(x0)
        loss,parts=weighted_loss(model,gg,x0,x1,tt,noise)
        _,old=original_loss(model,gg,x0,x1,tt,noise)
        torch.testing.assert_close(parts['flow'],old['flow']);torch.testing.assert_close(parts['score'],old['score'])
        torch.testing.assert_close(loss.detach(),old['flow']+4*old['score']);loss.backward()
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
        for weight in cfg['score_weights']:
            folder=a.output/f'calibration-{weight}';folder.mkdir()
            ex.interpolant_loss=original_loss if weight==1 else weighted_loss
            model,cost=ex.train_arm('feat_gnn',g,pool,base,997,folder,torch.device('cuda'),deadline,updates=20)
            assert all(torch.isfinite(v).all() for v in model.state_dict().values())
        save('calibration.json',{'admitted':True,'seconds':time.monotonic()-start,'loss_identity_and_gradients':True})
        data=torch.load(a.prior/'data.pt',weights_only=True)
        for seed in cfg['seeds']:
            for weight in cfg['score_weights']:
                folder=a.output/f'seed-{seed}-weight-{weight}';folder.mkdir()
                ex.interpolant_loss=original_loss if weight==1 else weighted_loss
                model,cost=ex.train_arm('feat_gnn',data['graphs']['train'],data['training_endpoints'],base,seed,folder,torch.device('cuda'),deadline)
                if weight==1:
                    old=torch.load(a.prior/f'seed-{seed}'/'feat_gnn.pt',weights_only=True)['state_dict']
                    assert all(torch.equal(v.cpu(),old[k].cpu()) for k,v in model.state_dict().items()),'Parent replay mismatch'
                training.append({'seed':seed,'score_weight':weight,**cost,'parent_replay_exact':weight==1})
                sup=FreeEnergyGNN(32,2).cuda().eval()
                sup.load_state_dict(torch.load(a.prior/f'seed-{seed}'/'supervised_gnn.pt',weights_only=True)['state_dict'])
                for offset in cfg['evaluation_stream_offsets']:
                    stream=folder/f'stream-{offset}';stream.mkdir()
                    for split in ['test','ood']:
                        rr,tm=ex.evaluate_split(sup,model,data['graphs'][split],split,{**base,'paths_per_graph':cfg['paths'],'inference_steps':cfg['steps']},seed+offset,stream,torch.device('cuda'),deadline)
                        records.extend([{**row,'training_seed':seed,'score_weight':weight,'stream':offset} for row in rr])
                        timings.append({'seed':seed,'score_weight':weight,'stream':offset,'split':split,**tm})
                save('partial.json',{'training':training,'records':records,'timings':timings})
                print(f'Completed seed {seed}, score weight {weight}',flush=True)
        save('results.json',{'status':'completed','seconds':time.monotonic()-start,'training':training,'records':records,'timings':timings,'provenance':provenance})
    except Exception as err:
        save('failure.json',{'exception':repr(err),'seconds':time.monotonic()-start});raise
    finally:ex.interpolant_loss=original_loss

if __name__=='__main__':main()

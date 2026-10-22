"""Paired continuation with exactly frozen parent velocity."""
import argparse,copy,hashlib,json,math,os,sys,time
from pathlib import Path
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import torch
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
class Composite(torch.nn.Module):
    def __init__(self,parent,student):
        super().__init__();self.parent=parent;self.student=student
    def forward(self,x,t,graph):
        with torch.no_grad():v,_=self.parent(x,t,graph)
        _,s=self.student(x,t,graph)
        return v,s

def main():
 p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--prior',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 cfg=json.loads((ROOT/'configs/qk7-frozen-velocity.json').read_text());sys.path.insert(0,str(a.source.resolve()))
 import rsi.jarzynski.experiment as ex
 from rsi.jarzynski.models import TransportGNN,FreeEnergyGNN,interpolant_loss
 from rsi.jarzynski.systems import make_graphs
 prior=json.loads((a.prior/'results.json').read_text());base=prior['config']
 for n,h in prior['provenance']['source_hashes'].items():assert sha(a.source/n)==h
 assert torch.cuda.is_available();torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
 torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 a.output.mkdir(parents=True,exist_ok=False)
 def save(n,v):(a.output/n).write_text(json.dumps(v,indent=2)+'\n')
 start=time.perf_counter();deadline=start+cfg['seconds_limit'];records=[];costs=[];times=[]
 def guard():
  if time.perf_counter()>deadline:raise TimeoutError('QK7 global budget exceeded')
 save('provenance.json',{'config':cfg,'script_sha256':sha(Path(__file__)),'config_sha256':sha(ROOT/'configs/qk7-frozen-velocity.json'),'prior_sha256':sha(a.prior/'results.json'),'source':prior['provenance'],'torch':torch.__version__,'gpu':torch.cuda.get_device_name(0)})
 def fit(parent,arm,graph_cpu,pool_cpu,seed,updates,folder):
  student=copy.deepcopy(parent).cuda().train()
  for par in student.parameters():par.requires_grad_(True)
  optimizer=torch.optim.Adam(student.parameters(),lr=.001)
  graph={k:v.cuda().float() for k,v in graph_cpu.items()};pool=[x.cuda().float() for x in pool_cpu]
  gen=torch.Generator(device='cuda').manual_seed(seed+104729+2000000);gg=torch.Generator(device='cuda').manual_seed(seed+65537+2000000)
  trace=[];torch.cuda.synchronize();begin=time.perf_counter()
  for step in range(updates):
   guard();idx=torch.randint(len(graph['a0']),(64,),device='cuda',generator=gg);batch={k:v[idx] for k,v in graph.items()}
   left=torch.randint(pool[0].shape[1],(64,),device='cuda',generator=gen);right=torch.randint(pool[0].shape[1],(64,),device='cuda',generator=gen)
   x0,x1=pool[0][idx,left],pool[1][idx,right]
   t=.001+.998*torch.rand(64,device='cuda',generator=gen);noise=torch.randn(x0.shape,device='cuda',generator=gen)
   optimizer.zero_grad(set_to_none=True)
   if arm=='joint':loss,parts=interpolant_loss(student,batch,x0,x1,t,noise,.5)
   else:
    t3=t[:,None,None];gamma=.5*torch.sin(math.pi*t3);x=(1-t3)*x0+t3*x1+gamma*noise
    _,score=student(x,t,batch);loss=(gamma*score+noise).square().mean()
   assert torch.isfinite(loss);loss.backward();norm=torch.nn.utils.clip_grad_norm_(student.parameters(),5,error_if_nonfinite=True);optimizer.step()
   if step==0 or (step+1)%100==0:trace.append({'step':step+1,'loss':float(loss.detach()),'gradient_norm':float(norm)})
  torch.cuda.synchronize();guard();seconds=time.perf_counter()-begin
  torch.save({'state_dict':student.state_dict(),'seed':seed,'arm':arm,'updates':updates},folder/'student.pt')
  (folder/'trace.json').write_text(json.dumps(trace,indent=2)+'\n')
  return student.eval(),{'seconds':seconds,'updates':updates,'checkpoint_sha256':sha(folder/'student.pt')}
 try:
  data=torch.load(a.prior/'data.pt',weights_only=True)
  # A separate synthetic calibration tests the same continuation interface.
  torch.manual_seed(997);cal_parent=TransportGNN(32,2).cuda().eval()
  for par in cal_parent.parameters():par.requires_grad_(False)
  graph=make_graphs(8,4,991);pool=ex.training_pool(graph,16,992,1.)
  for arm in ['joint','score_only']:
   folder=a.output/f'calibration-{arm}';folder.mkdir();student,cc=fit(cal_parent,arm,graph,pool,997,20,folder)
   if arm=='score_only':
    composite=Composite(cal_parent,student);g={k:v.cuda().float() for k,v in graph.items()};x=pool[0][:,0].cuda().float();t=torch.full((8,),.4,device='cuda')
    v,s=composite(x,t,g);pv,_=cal_parent(x,t,g);assert torch.equal(v,pv)
    student.zero_grad(set_to_none=True);s.square().mean().backward();assert all(p.grad is None for p in cal_parent.parameters())
  save('calibration.json',{'admitted':True,'seconds':time.perf_counter()-start,'velocity_preserved':True,'parent_gradients_absent':True})
  for seed in cfg['seeds']:
   checkpoint=a.prior/f'seed-{seed}'/'feat_gnn.pt';pc=next(s for s in prior['seeds'] if s['seed']==seed)['training']['feat_gnn']
   assert sha(checkpoint)==pc['checkpoint_sha256']
   parent=TransportGNN(32,2).cuda().eval();state=torch.load(checkpoint,weights_only=True)['state_dict'];parent.load_state_dict(state)
   for par in parent.parameters():par.requires_grad_(False)
   sup=FreeEnergyGNN(32,2).cuda().eval();sup.load_state_dict(torch.load(a.prior/f'seed-{seed}'/'supervised_gnn.pt',weights_only=True)['state_dict'])
   for arm in cfg['arms']:
    folder=a.output/f'seed-{seed}-{arm}';folder.mkdir()
    if arm=='parent':model=parent;cost={'seconds':0.,'updates':0,'checkpoint_sha256':sha(checkpoint)}
    else:
     student,cost=fit(parent,arm,data['graphs']['train'],data['training_endpoints'],seed,600,folder)
     model=student if arm=='joint' else Composite(parent,student).eval()
    assert all(torch.equal(v.cpu(),state[k].cpu()) for k,v in parent.state_dict().items())
    if arm=='score_only':
     g={k:v[:8].cuda().float() for k,v in data['graphs']['train'].items()};x=data['training_endpoints'][0][:8,0].cuda().float();t=torch.linspace(0,1,8,device='cuda')
     with torch.no_grad():assert torch.equal(model(x,t,g)[0],parent(x,t,g)[0])
    costs.append({'seed':seed,'arm':arm,**cost,'deployed_parameters':sum(p.numel() for p in model.parameters()),'network_calls_per_field':2 if arm=='score_only' else 1})
    for offset in cfg['streams']:
     stream=folder/f'stream-{offset}';stream.mkdir()
     for split in ['test','ood']:
      guard()
      # Legacy evaluator expects a monotonic-clock absolute deadline.
      legacy_deadline=time.monotonic()+max(0,deadline-time.perf_counter())
      rr,tm=ex.evaluate_split(sup,model,data['graphs'][split],split,{**base,'paths_per_graph':128,'inference_steps':32},seed+offset,stream,torch.device('cuda'),legacy_deadline)
      records.extend([{**row,'training_seed':seed,'recipe':arm,'stream':offset} for row in rr]);times.append({'seed':seed,'recipe':arm,'stream':offset,'split':split,**tm})
    save('partial.json',{'costs':costs,'records':records,'timings':times});print(f'Completed {seed} {arm}',flush=True)
  guard();save('results.json',{'status':'completed','seconds':time.perf_counter()-start,'costs':costs,'records':records,'timings':times,'provenance':json.loads((a.output/'provenance.json').read_text())})
 except Exception as e:save('failure.json',{'error':repr(e),'seconds':time.perf_counter()-start});raise
if __name__=='__main__':main()

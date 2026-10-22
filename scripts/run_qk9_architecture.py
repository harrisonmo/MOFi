"""Run the frozen architecture comparison through fresh final evaluation."""
import argparse
import json
import os
import sys
import time
from pathlib import Path
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import torch
from run_qk12_diagnostic import sha

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    for name in ['source', 'prior', 'output']:
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.source.resolve()))
    from qk9_models import MODELS
    import rsi.jarzynski.experiment as ex
    from rsi.jarzynski.models import FreeEnergyGNN
    from rsi.jarzynski.systems import make_graphs
    cfg = json.loads((ROOT/'configs/qk9-architecture.json').read_text())
    prior = json.loads((args.prior/'results.json').read_text())
    base = {**prior['config'], 'updates':cfg['updates'], 'paths_per_graph':cfg['paths'], 'inference_steps':cfg['steps']}
    for name, expected in prior['provenance']['source_hashes'].items():
        assert sha(args.source/name) == expected
    assert torch.cuda.is_available()
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device('cuda')
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    def save(name, value):
        (args.output/name).write_text(json.dumps(value, indent=2)+'\n', encoding='utf-8')
    def deadline():
        remaining = cfg['seconds_limit'] - (time.perf_counter()-started)
        if remaining <= 0:
            raise TimeoutError('QK9 global budget exceeded')
        return time.monotonic()+remaining
    rows, costs, timings = [], [], []
    provenance = {'config':cfg, 'source':prior['provenance'], 'data_sha256':sha(args.prior/'data.pt'),
                  'hashes':{p:sha(ROOT/p) for p in ['configs/qk9-architecture.json', 'scripts/run_qk9_architecture.py', 'scripts/qk9_models.py']},
                  'gpu':torch.cuda.get_device_name(0), 'torch':torch.__version__}
    save('provenance.json', provenance)
    try:
        data = torch.load(args.prior/'data.pt', weights_only=True)
        cal = make_graphs(8, 4, 991)
        pool = ex.training_pool(cal, 16, 992, 1.)
        for arm, width in cfg['architectures'].items():
            folder = args.output/f'calibration-{arm}'
            folder.mkdir()
            ex.TransportGNN = MODELS[arm]
            model, cost = ex.train_arm('feat_gnn', cal, pool, {**base, 'width':width}, 997, folder, device, deadline(), updates=20)
            assert cost['parameters'] == cfg['parameters'][arm]
            torch.manual_seed(998)
            sup = FreeEnergyGNN(32, 2).cuda().eval()
            ex.evaluate_split(sup, model, make_graphs(2,4,998), 'test',
                              {**base,'paths_per_graph':32,'inference_steps':8},997,folder,device,deadline())
        save('calibration.json', {'admitted':True, 'seconds':time.perf_counter()-started})
        models, supervisors = {}, {}
        def evaluate(phase, graphs):
            for seed in cfg['seeds']:
                for arm in cfg['architectures']:
                    for stream in cfg['streams']:
                        for split, graph in graphs.items():
                            folder = args.output/f'{phase}-{seed}-{arm}-{stream}-{split}'
                            folder.mkdir()
                            # Keep original RNG offsets and file names for historical anchors.
                            legacy_split = 'ood' if split in ['ood','final_six'] else 'test'
                            rr, tm = ex.evaluate_split(supervisors[seed], models[seed,arm], graph, legacy_split,
                                                       base,seed+stream,folder,device,deadline())
                            rows.extend([{**x,'split':split,'training_seed':seed,'recipe':arm,'stream':stream,'phase':phase} for x in rr])
                            timings.append({'seed':seed,'recipe':arm,'stream':stream,'split':split,'phase':phase,**tm})
                    save('partial.json', {'rows':rows,'costs':costs,'timings':timings})
                    print(f'Completed {phase}: seed {seed}, {arm}', flush=True)
        for seed in cfg['seeds']:
            sup = FreeEnergyGNN(32,2).cuda().eval()
            sup.load_state_dict(torch.load(args.prior/f'seed-{seed}/supervised_gnn.pt',weights_only=True)['state_dict'])
            supervisors[seed] = sup
            for arm, width in cfg['architectures'].items():
                folder = args.output/f'train-{seed}-{arm}'
                folder.mkdir()
                ex.TransportGNN = MODELS[arm]
                model, cost = ex.train_arm('feat_gnn',data['graphs']['train'],data['training_endpoints'],
                                           {**base,'width':width},seed,folder,device,deadline())
                assert cost['parameters'] == cfg['parameters'][arm]
                if arm == 'shared':
                    old = torch.load(args.prior/f'seed-{seed}/feat_gnn.pt',weights_only=True)['state_dict']
                    assert all(torch.equal(value.cpu(),old[key].cpu()) for key,value in model.state_dict().items())
                models[seed,arm] = model
                costs.append({'seed':seed,'recipe':arm,**cost})
                print(f'Trained seed {seed}, {arm}', flush=True)
        evaluate('development', {k:data['graphs'][k] for k in ['test','ood']})
        def mae(arm, split):
            errors = [abs(x['predictions']['feat_gnn_jarzynski']-x['truth']) for x in rows if x['recipe']==arm and x['split']==split]
            return sum(errors)/len(errors)
        ranking = {arm:sum(mae(arm,s)/mae('shared',s) for s in ['test','ood'])/2 for arm in cfg['architectures']}
        selected = min(ranking, key=ranking.get)
        lock = {'selected':selected,'development_ratios':ranking,'seconds':time.perf_counter()-started,'checkpoint_hashes':costs}
        save('selection-lock.json',lock)
        final = {name:make_graphs(spec['graphs'],spec['nodes'],spec['seed'],ood=spec['ood']) for name,spec in cfg.items() if name.startswith('final_')}
        torch.save(final,args.output/'final-data.pt')
        save('final-manifest.json', {name:{'hash':ex.graph_hash(graph),'graphs':len(graph['a0'])} for name,graph in final.items()})
        evaluate('final',final)
        deadline()
        save('results.json',{'status':'completed','seconds':time.perf_counter()-started,'rows':rows,'costs':costs,
                             'timings':timings,'selection':lock,'provenance':provenance})
    except Exception as error:
        save('failure.json',{'error':repr(error),'seconds':time.perf_counter()-started})
        raise


if __name__ == '__main__':
    main()

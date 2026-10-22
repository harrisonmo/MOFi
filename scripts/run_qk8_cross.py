"""Cross frozen fields and measure errors on actual sampler trajectories."""
import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import torch
from run_qk12_diagnostic import fields, sha, validate_math

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    for name in ['source', 'prior', 'output']:
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    cfg_path = ROOT / 'configs/qk8-cross-fields.json'
    cfg = json.loads(cfg_path.read_text())
    sys.path.insert(0, str(args.source.resolve()))
    from rsi.jarzynski.models import TransportGNN
    from rsi.jarzynski.systems import precision, sample_endpoint, exact_delta_f, make_graphs
    from rsi.jarzynski.estimators import simulate_forward, jarzynski, effective_sample_size
    assert torch.cuda.is_available()
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    def save(name, value):
        (args.output / name).write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')
    def guard():
        if time.perf_counter() - started > cfg['seconds_limit']:
            raise TimeoutError('QK8 budget exceeded')
    try:
        prior = json.loads((args.prior / 'results.json').read_text())
        qk7 = json.loads((ROOT / 'runs/qk7-frozen-01/results.json').read_text())
        for name, expected in prior['provenance']['source_hashes'].items():
            assert sha(args.source / name) == expected
        data = torch.load(args.prior / 'data.pt', weights_only=True)
        provenance = {'config': cfg, 'script_sha256': sha(Path(__file__)),
                      'config_sha256': sha(cfg_path), 'analytic_sha256': sha(ROOT / 'scripts/run_qk12_diagnostic.py'),
                      'data_sha256': sha(args.prior / 'data.pt'), 'source': prior['provenance'],
                      'qk7_sha256': sha(ROOT / 'runs/qk7-frozen-01/results.json'),
                      'gpu': torch.cuda.get_device_name(0), 'torch': torch.__version__}
        save('provenance.json', provenance)
        checks = validate_math()
        rows, traces, timings, checkpoints = [], [], [], []

        @torch.no_grad()
        def sample(models, pair, graph, rng_seed, paths, steps, instrument):
            ng = len(graph['a0'])
            expanded = {k: v.repeat_interleave(paths, 0) for k, v in graph.items()}
            initial = sample_endpoint(expanded, 0, torch.Generator().manual_seed(rng_seed), 1.).cuda()
            gpu = {k: v.cuda() for k, v in expanded.items()}
            g32 = {k: v.float() for k, v in gpu.items()}
            covs = [torch.linalg.inv(precision(graph, e)).cuda() for e in [0, 1]]
            table = [fields(*covs, i / steps)[2:] for i in range(steps + 1)] if instrument else []
            trace = []
            def drift(x, t, unused):
                v, s = models[pair[0]](x.float(), t.float(), g32)
                if pair[0] != pair[1]:
                    _, s = models[pair[1]](x.float(), t.float(), g32)
                v, s = v.double(), s.double()
                if instrument:
                    i = len(trace)
                    assert torch.all(t == i / steps)
                    rv, rs = [m[:, None] @ x.reshape(ng, paths, x.shape[1], 3) for m in table[i]]
                    ev, es = v.reshape_as(rv) - rv, s.reshape_as(rs) - rs
                    values = [ev.square(), es.square(), (ev + .25 * es).square(),
                              (ev - .25 * es).square(), ev * es]
                    trace.append(torch.stack([z.mean((1, 2, 3)) for z in values], -1))
                return v, s
            result = simulate_forward(gpu, initial, drift, steps, .25,
                                      torch.Generator(device='cuda').manual_seed(rng_seed + 99991))
            result = {k: v.cpu() for k, v in result.items() if k != 'endpoint'}
            if instrument:
                assert len(trace) == steps + 1
            return result, torch.stack(trace).cpu() if trace else None

        for seed in cfg['seeds']:
            models = {}
            for recipe in ['parent', 'joint', 'score_only']:
                if recipe == 'parent':
                    path = args.prior / f'seed-{seed}/feat_gnn.pt'
                    expected = next(x for x in prior['seeds'] if x['seed'] == seed)['training']['feat_gnn']['checkpoint_sha256']
                else:
                    path = ROOT / f'runs/qk7-frozen-01/seed-{seed}-{recipe}/student.pt'
                    expected = next(x for x in qk7['costs'] if x['seed'] == seed and x['arm'] == recipe)['checkpoint_sha256']
                assert sha(path) == expected
                model = TransportGNN(32, 2).cuda().eval()
                model.load_state_dict(torch.load(path, weights_only=True)['state_dict'])
                models[recipe] = model
                checkpoints.append({'seed': seed, 'recipe': recipe, 'path': str(path), 'sha256': expected})
            if seed == cfg['seeds'][0]:
                for pair in cfg['arms'].values():
                    graph = make_graphs(2, 4, 991)
                    a, _ = sample(models, pair, graph, 997, 32, 8, False)
                    b, _ = sample(models, pair, graph, 997, 32, 8, True)
                    for key in a:
                        assert torch.equal(a[key], b[key]) and torch.isfinite(b[key]).all()
                save('calibration.json', {'admitted': True, 'instrumentation_preserves_work': True,
                                         'math': checks, 'seconds': time.perf_counter() - started})
                guard()
            for stream in cfg['streams']:
                for split in ['test', 'ood']:
                    graph = data['graphs'][split]
                    truth = exact_delta_f(graph)
                    for arm, pair in cfg['arms'].items():
                        guard()
                        begin = time.perf_counter()
                        chunks = []
                        for start in range(0, len(truth), cfg['chunk_graphs']):
                            chunk = {k: v[start:start + cfg['chunk_graphs']] for k, v in graph.items()}
                            rng = (seed + stream) * 1000003 + start + (10000 if split == 'ood' else 0)
                            result, trace = sample(models, pair, chunk, rng, cfg['paths'], cfg['steps'], True)
                            chunks.append(result)
                            work = result['work'].reshape(-1, cfg['paths'])
                            je, ess = jarzynski(work), effective_sample_size(work)
                            for j in range(len(work)):
                                key = {'seed': seed, 'stream': stream, 'split': split, 'arm': arm, 'graph': start + j}
                                rows.append({**key, 'prediction': float(je[j]), 'truth': float(truth[start + j]), 'ess': float(ess[j])})
                                for i in range(cfg['steps'] + 1):
                                    traces.append({**key, 'time': i / cfg['steps'], **dict(zip(
                                        ['velocity_mse', 'score_mse', 'forward_mse', 'backward_mse', 'cross'], trace[i, j].tolist()))})
                            guard()
                        torch.save(chunks, args.output / f'{seed}-{stream}-{split}-{arm}.pt')
                        timings.append({'seed': seed, 'stream': stream, 'split': split, 'arm': arm,
                                        'seconds': time.perf_counter() - begin, 'forwards_per_field': 1 if pair[0] == pair[1] else 2})
                print(f'Completed seed {seed}, stream {stream}', flush=True)
                save('partial.json', {'rows': rows, 'timings': timings})
        guard()
        save('traces.json', traces)
        save('results.json', {'status': 'completed', 'seconds': time.perf_counter() - started,
                              'rows': rows, 'timings': timings, 'checkpoints': checkpoints, 'provenance': provenance})
    except Exception as error:
        save('failure.json', {'error': repr(error), 'seconds': time.perf_counter() - started})
        raise


if __name__ == '__main__':
    main()

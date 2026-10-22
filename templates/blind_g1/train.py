"""Fixed-epoch G1 parent; development labels are never available here."""
import argparse
import json
import os
from pathlib import Path
import random
import time

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import torch
from g1_model import PeriodicMPNN, pack


def load(root, split):
    rows = json.loads((root / (split + '.json')).read_text())
    graphs = []
    for row in rows:
        with np.load(root / row['graph'], allow_pickle=False) as graph:
            graphs.append({key: graph[key].copy() for key in ('z', 'src', 'dst', 'distance')})
    return rows, graphs


def run(args):
    start = time.monotonic()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required')
    train, graphs = load(args.input, 'train')
    development, dev_graphs = load(args.input, 'validation')
    target = np.array([row['target'] for row in train])
    mean, std = float(target.mean()), float(target.std())
    if not std > 0:
        raise ValueError('Constant target')
    scaled = (target - mean) / std
    model = PeriodicMPNN().cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.00001)
    rng = np.random.default_rng(args.seed)
    history = []
    torch.cuda.reset_peak_memory_stats()
    for epoch in range(args.epochs):
        model.train()
        order = rng.permutation(len(graphs))
        total = 0.0
        for start_index in range(0, len(order), 16):
            indices = order[start_index:start_index + 16]
            batch = pack([graphs[i] for i in indices], 'cuda')
            y = torch.as_tensor(scaled[indices], dtype=torch.float32, device='cuda')
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.mse_loss(model(batch), y)
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite loss')
            loss.backward()
            optimizer.step()
            total += loss.item() * len(indices)
        history.append(total / len(graphs))
    model.eval()
    predictions = []
    with torch.no_grad():
        for start_index in range(0, len(dev_graphs), 16):
            output = model(pack(dev_graphs[start_index:start_index + 16], 'cuda')) * std + mean
            predictions.extend(output.cpu().tolist())
    args.output.mkdir(parents=True, exist_ok=True)
    result = {row['id']: float(value) for row, value in zip(development, predictions)}
    (args.output / 'predictions.json').write_text(json.dumps(result, allow_nan=False))
    metadata = {'seed': args.seed, 'epochs': args.epochs, 'seconds': time.monotonic() - start,
                'torch': torch.__version__, 'cuda': torch.version.cuda,
                'device': torch.cuda.get_device_name(0), 'training_mse': history,
                'peak_gpu_allocated_bytes': torch.cuda.max_memory_allocated()}
    (args.output / 'metadata.json').write_text(json.dumps(metadata, allow_nan=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--epochs', type=int, default=50)
    run(parser.parse_args())

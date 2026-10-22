"""Build and verify a label-minimal G1 package; never execute candidate code."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
SOURCES = ('g1_model.py', 'g1_features.py')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build(output, root=ROOT):
    output, root = Path(output).resolve(), Path(root).resolve()
    if output.exists():
        raise FileExistsError('Use a new output directory')
    data = root / 'datasets/processed/g1'
    report = read(root / 'docs/reports/G1_BASELINE_RESULT.json')
    prior = report['provenance']
    if sha(root / 'configs/g1-comparison.json') != report['config_sha256']:
        raise ValueError('Historical configuration changed')
    if sha(data / 'manifest.json') != prior['dataset_manifest_sha256']:
        raise ValueError('G1 manifest changed')
    manifest = read(data / 'manifest.json')
    if not manifest['training_admitted']:
        raise ValueError('Dataset not admitted')
    splits = {}
    for split in ('train', 'validation'):
        path = data / (split + '.json')
        if sha(path) != manifest['split_sha256'][split]:
            raise ValueError('Split hash mismatch')
        rows = read(path)
        if len({r['id'] for r in rows}) != len(rows) or set(r['id'] for r in rows) != set(manifest['split_ids'][split]):
            raise ValueError('Split membership mismatch')
        splits[split] = rows
    if {r['family'] for r in splits['train']} & {r['family'] for r in splits['validation']}:
        raise ValueError('Family leakage')
    public = output / 'public'
    public.mkdir(parents=True)
    for name in SOURCES:
        src = root / 'scripts' / name
        expected = prior['source_sha256'][name]
        # Historical artifacts may have Windows line endings.
        content = src.read_bytes()
        if expected not in {sha(src), hashlib.sha256(content.replace(b'\r\n', b'\n').replace(b'\n', b'\r\n')).hexdigest()}:
            raise ValueError('Historical source changed: ' + name)
        shutil.copyfile(src, public / name)
    records = {}
    for split, rows in splits.items():
        clean = []
        for row in rows:
            name = hashlib.sha256(row['id'].encode()).hexdigest()
            item = {'id': row['id']}
            for field in ('graph', 'cif'):
                src = (root / row[field]).resolve()
                if not src.is_relative_to(data.resolve()) or sha(src) != row[field + '_sha256']:
                    raise ValueError('Unsafe or changed input')
                dest = public / 'inputs' / (name + ('.npz' if field == 'graph' else '.cif'))
                dest.parent.mkdir(exist_ok=True)
                shutil.copyfile(src, dest)
                item[field] = dest.relative_to(public).as_posix()
            if split == 'train':
                item['target'] = row['target']
            clean.append(item)
        write(public / (split + '.json'), clean)
        records[split] = len(clean)
    config = read(root / 'configs/g1-comparison.json')
    write(public / 'baseline.json', {k: config[k] for k in ('gnn', 'batch_size', 'max_epochs', 'early_stopping_patience')})
    (public / 'TASK.md').write_text('Predict methane working capacity at 298 K, 65 minus 5.8 bar, in mmol/g. Start from the supplied periodic GNN. Improve prediction under the controller budget. Training targets are supplied; development labels are external. You may modify representation, architecture and training. Do not use external data or pretrained weights. Submit finite predictions keyed by structure ID for both registered seeds. The controller owns evaluation and selection.\n', encoding='utf-8')
    write(output / 'private/development.json', [{'id': r['id'], 'target': r['target'], 'family': r['family']} for r in splits['validation']])
    files = {p.relative_to(public).as_posix(): sha(p) for p in public.rglob('*') if p.is_file()}
    write(output / 'manifest.json', {'schema': 'blind-g1-preparation-v1', 'public_files': files,
        'development_sha256': sha(output / 'private/development.json'), 'counts': records,
        'historical_parent_commit': prior['root_commit'], 'dataset_sha256': prior['dataset_manifest_sha256'],
        'heldout_labels_read': False, 'model_calls': 0, 'training_runs': 0,
        'execution_admitted': False, 'pending_gates': ['parent_reproduction', 'runtime_budget_calibration', 'outer_transport_isolation', 'heldout_access_audit']})
    verify(output)
    return read(output / 'manifest.json')


def verify(output):
    output = Path(output)
    manifest = read(output / 'manifest.json')
    public = output / 'public'
    actual = {}
    for path in public.rglob('*'):
        if path.is_symlink():
            raise ValueError('Symlinks prohibited')
        if path.is_file():
            actual[path.relative_to(public).as_posix()] = sha(path)
    if actual != manifest['public_files']:
        raise ValueError('Public package changed')
    if sha(output / 'private/development.json') != manifest['development_sha256']:
        raise ValueError('Development labels changed')
    for split in ('train', 'validation'):
        allowed = {'id', 'graph', 'cif'} | ({'target'} if split == 'train' else set())
        for row in read(public / (split + '.json')):
            if set(row) != allowed:
                raise ValueError('Unexpected public fields')
    return manifest


def evaluate(output, predictions, arm, generation):
    """Controller-only aggregate scoring; caller must isolate candidate execution."""
    from rsi.hpc.core import locked
    output = Path(output)
    manifest = verify(output)
    if arm not in {'independent', 'rsi'} or type(generation) is not int or generation not in (1, 2, 3):
        raise ValueError('Invalid experiment slot')
    reference = read(output / 'private/development.json')
    ids = {r['id'] for r in reference}
    if set(predictions) != {'17', '29'}:
        raise ValueError('Exactly two search seeds required')
    for pred in predictions.values():
        if not isinstance(pred, dict) or set(pred) != ids:
            raise ValueError('Prediction IDs differ')
        if any(type(v) not in (int, float) or not math.isfinite(v) for v in pred.values()):
            raise ValueError('Predictions must be finite numbers')
    ledger_path = output / 'private/evaluation-ledger.json'
    with locked(ledger_path):
        ledger = read(ledger_path) if ledger_path.exists() else {'binding': sha(output / 'manifest.json'), 'entries': []}
        if ledger['binding'] != sha(output / 'manifest.json'):
            raise ValueError('Evaluation binding changed')
        entries = [e for e in ledger['entries'] if e['arm'] == arm]
        if generation != len(entries) + 1:
            raise ValueError('Slot already used or out of order')
        metrics = {}
        for seed, pred in predictions.items():
            errors = [pred[r['id']] - r['target'] for r in reference]
            metrics[seed] = {'mae': sum(abs(e) for e in errors)/len(errors), 'rmse': math.sqrt(sum(e*e for e in errors)/len(errors))}
        result = {'seeds': metrics, 'mean_mae': sum(m['mae'] for m in metrics.values())/2}
        payload_hash = hashlib.sha256(json.dumps(predictions, sort_keys=True, allow_nan=False).encode()).hexdigest()
        write(output / 'private/predictions' / f'{arm}-{generation}.json', predictions)
        ledger['entries'].append({'arm': arm, 'generation': generation, 'predictions_sha256': payload_hash, 'result': result})
        write(ledger_path, ledger)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['build', 'verify'])
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = build(args.output) if args.action == 'build' else verify(args.output)
    print(json.dumps({k: v for k,v in result.items() if k != 'public_files'}, indent=2))

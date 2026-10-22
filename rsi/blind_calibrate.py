"""Calibrate the fixed-epoch parent in the exact candidate GPU sandbox."""
import argparse
import math
from pathlib import Path
import shutil
import subprocess

from rsi.blind_g1 import ROOT, read, sha, verify, write
from rsi.blind_runtime import run_container


def score(predictions, reference):
    ids = {row['id'] for row in reference}
    if not isinstance(predictions, dict) or set(predictions) != ids:
        raise ValueError('Prediction membership differs')
    if any(type(v) not in (int, float) or not math.isfinite(v) for v in predictions.values()):
        raise ValueError('Nonfinite prediction')
    errors = [predictions[row['id']] - row['target'] for row in reference]
    metrics = {'mae': sum(abs(v) for v in errors) / len(errors),
               'rmse': math.sqrt(sum(v*v for v in errors) / len(errors))}
    if not all(math.isfinite(v) for v in metrics.values()):
        raise ValueError('Metric overflow')
    return metrics


def parent_files(package, output):
    output.mkdir(parents=True)
    for name in ('g1_model.py', 'g1_features.py'):
        shutil.copyfile(package / 'public' / name, output / name)
    shutil.copyfile(ROOT / 'templates/blind_g1/train.py', output / 'train.py')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--image', required=True)
    args = parser.parse_args()
    verify(args.package)
    if args.output.exists():
        raise FileExistsError('Use a fresh calibration directory')
    parent_files(args.package, args.output / 'parent')
    reference = read(args.package / 'private/development.json')
    report = {'schema': 'blind-g1-calibration-v2', 'protocol': 'G1-F50',
              'image_id': args.image, 'package_sha256': sha(args.package / 'manifest.json'),
              'parent_files': {p.name: sha(p) for p in (args.output / 'parent').iterdir()},
              'heldout_evaluated': False, 'model_calls': 0, 'fits': []}
    for index, seed in enumerate((17, 29, 17)):
        result = run_container(args.package / 'public', args.output / 'parent', args.image, seed, 900)
        write(args.output / f'fit-{index}.json', result)
        if result['status'] != 'completed':
            write(args.output / 'report.json', report)
            raise RuntimeError(f'Calibration fit failed: {result["status"]}')
        metrics = score(result['predictions'], reference)
        report['fits'].append({'seed': seed, 'wall_seconds': result['wall_seconds'],
                              'metadata': result['metadata'], **metrics})
        write(args.output / 'report.json', report)
        print(f'Calibration {index + 1}/3: seed={seed} MAE={metrics["mae"]:.6f} seconds={result["wall_seconds"]:.1f}', flush=True)
    first, repeated = read(args.output / 'fit-0.json')['predictions'], read(args.output / 'fit-2.json')['predictions']
    report['repeat_max_absolute_delta'] = max(abs(first[k] - repeated[k]) for k in first)
    report['reproducible'] = report['repeat_max_absolute_delta'] <= 1e-5
    report['fit_wall_seconds'] = min(900, max(180, math.ceil(max(r['wall_seconds'] for r in report['fits']) * 2 / 30) * 30))
    report['headroom_sufficient'] = report['fit_wall_seconds'] >= 1.5 * max(r['wall_seconds'] for r in report['fits'])
    report['passed'] = report['reproducible'] and report['headroom_sufficient']
    report['environment_freeze'] = subprocess.check_output(['docker', 'run', '--rm', '--network', 'none', args.image, 'pip', 'freeze'], text=True).splitlines()
    write(args.output / 'report.json', report)
    print('Calibration passed: ' + str(report['passed']), flush=True)


if __name__ == '__main__':
    main()

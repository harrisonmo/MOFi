"""Freeze a prepared development campaign and publish compact controller evidence."""
import argparse
from pathlib import Path
import shutil
import subprocess
from rsi.blind_g1 import ROOT, read, sha, write
from rsi.blind_campaign import initialize


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package', type=Path, required=True)
    parser.add_argument('--calibration', type=Path, required=True)
    parser.add_argument('--probe', type=Path, required=True)
    parser.add_argument('--campaign', type=Path, required=True)
    parser.add_argument('--handoff', type=Path, required=True)
    parser.add_argument('--interrupted-calibration', type=Path)
    args = parser.parse_args()
    report = read(args.calibration / 'report.json')
    if args.handoff.exists():
        raise FileExistsError('Use a new handoff directory')
    policy = initialize(args.package, args.calibration, args.campaign, args.probe)
    args.handoff.mkdir(parents=True)
    for name in ('START_HERE.md', 'response.schema.json'):
        shutil.copyfile(ROOT / 'templates/blind_g1' / name, args.handoff / name)
    freeze = ROOT / 'environments/blind-g1/resolved-requirements.txt'
    freeze.write_text('\n'.join(report['environment_freeze']) + '\n', encoding='utf-8')
    result = {k: v for k, v in report.items() if k not in ('environment_freeze', 'fits')}
    result['fits'] = [{**{k: v for k, v in row.items() if k != 'metadata'},
                       'device': row['metadata']['device'], 'torch': row['metadata']['torch'],
                       'peak_gpu_allocated_bytes': row['metadata']['peak_gpu_allocated_bytes']} for row in report['fits']]
    result.update({'scope': 'development search and fresh-seed confirmation; no formal search launched',
                   'runtime_probe': read(args.probe), 'formal_training_runs': 0,
                   'root_commit_before_preparation_commit': subprocess.check_output(['git', '-c', f'safe.directory={ROOT.as_posix()}', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                   'crystal_builder_reference': 'df014447f0812dd4f55ed60359b71a981c3eeb60',
                   'crystal_builder_executed': False,
                   'policy_sha256': sha(args.campaign / 'policy.json'),
                   'calibration_report_sha256': sha(args.calibration / 'report.json'),
                   'resolved_environment_sha256': sha(freeze),
                   'formal_fit_cap_per_provider': 21, 'formal_fit_cap_both_providers': 42,
                   'formal_container_seconds_cap_per_provider': 21 * policy['fit_wall_seconds'],
                   'app_session_conditions': 'operator-attested fresh context and disabled tools; not OS-verified app isolation',
                   'final_test_enabled': False})
    if args.interrupted_calibration:
        interrupted = read(args.interrupted_calibration)
        result['interrupted_calibration'] = {'path': str(args.interrupted_calibration),
            'sha256': sha(args.interrupted_calibration), 'status': interrupted['status'],
            'seconds': interrupted['wall_seconds'], 'scored': False}
    write(ROOT / 'docs/reports/BLIND_G1_READY.json', result)
    print('Prepared campaign: ' + str(args.campaign))
    print('Neutral app handoff: ' + str(args.handoff))
    print('Per-fit wall cap: ' + str(policy['fit_wall_seconds']))

if __name__ == '__main__':
    main()

"""Exercise the actual candidate runtime without training or label access."""
import argparse
import json
from pathlib import Path
from rsi.blind_g1 import write, sha
from rsi.blind_runtime import run_container

PROBE = '''
import json, os, pathlib, socket
checks = {'non_root': os.getuid() == 65534,
 'private_labels_absent': not pathlib.Path('/input/../private/development.json').exists(),
 'host_workspace_absent': not pathlib.Path('/mnt/c/Users').exists(),
 'repository_absent': not pathlib.Path('/input/.git').exists(),
 'socket_absent': not pathlib.Path('/var/run/docker.sock').exists(),
 'credentials_absent': not any(k.endswith('API_KEY') for k in os.environ)}
for name, path in [('input_readonly','/input/probe'),('candidate_readonly','/candidate/probe'),('root_readonly','/probe')]:
 try:
  pathlib.Path(path).write_text('x')
  checks[name] = False
 except OSError:
  checks[name] = True
try:
 socket.create_connection(('1.1.1.1',443),timeout=1).close()
 checks['network_disabled'] = False
except OSError:
 checks['network_disabled'] = True
checks['development_unlabeled'] = all(set(r)=={'id','graph','cif'} for r in json.loads(pathlib.Path('/input/validation.json').read_text()))
pathlib.Path('/output/predictions.json').write_text(json.dumps(checks))
'''

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--image', required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Use a new probe directory')
    args.output.mkdir(parents=True)
    candidate = args.output / 'candidate'
    candidate.mkdir()
    (candidate / 'train.py').write_text(PROBE)
    good = run_container(args.package / 'public', candidate, args.image, 17, 30)
    write(args.output / 'boundary.json', good)
    (candidate / 'train.py').write_text('import time\ntime.sleep(60)\n')
    timeout = run_container(args.package / 'public', candidate, args.image, 17, 1)
    write(args.output / 'timeout.json', timeout)
    (candidate / 'train.py').write_text("from pathlib import Path\nPath('/output/predictions.json').symlink_to('/input/validation.json')\n")
    try:
        run_container(args.package / 'public', candidate, args.image, 17, 30)
        symlink_rejected = False
    except ValueError:
        symlink_rejected = True
    report = {'boundary_checks': good.get('predictions', {}), 'boundary_status': good['status'],
              'image_id': args.image, 'package_sha256': sha(args.package / 'manifest.json'),
              'timeout_enforced': timeout['status'] == 'timeout', 'symlink_output_rejected': symlink_rejected,
              'training_runs': 0, 'model_calls': 0}
    report['passed'] = good['status'] == 'completed' and all(report['boundary_checks'].values()) and report['timeout_enforced'] and symlink_rejected
    write(args.output / 'report.json', report)
    print(json.dumps(report, indent=2))
    if not report['passed']:
        raise SystemExit(1)

if __name__ == '__main__':
    main()

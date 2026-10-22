"""Check the candidate container boundary without training or model calls."""
import argparse
import json
from pathlib import Path
import subprocess

PROBE = r'''
import json, os, pathlib, socket
checks = {'non_root': os.getuid() != 0,
          'public_readable': pathlib.Path('/input/train.json').is_file(),
          'private_absent': not pathlib.Path('/input/../private/development.json').exists(),
          'repository_absent': not pathlib.Path('/input/.git').exists(),
          'socket_absent': not pathlib.Path('/var/run/docker.sock').exists()}
for name, path in [('public_readonly', '/input/probe'), ('root_readonly', '/probe')]:
    try:
        pathlib.Path(path).write_text('probe')
        checks[name] = False
    except OSError:
        checks[name] = True
try:
    socket.create_connection(('1.1.1.1', 443), timeout=2).close()
    checks['network_disabled'] = False
except OSError:
    checks['network_disabled'] = True
print(json.dumps(checks))
raise SystemExit(0 if all(checks.values()) else 1)
'''

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--image', default='python:3.13-slim')
    args = parser.parse_args()
    digest = subprocess.check_output(['docker', 'image', 'inspect', args.image, '--format', '{{.Id}}'], text=True).strip()
    command = ['docker', 'run', '--rm', '--network', 'none', '--read-only', '--user', '65534:65534',
               '--cap-drop', 'ALL', '--security-opt', 'no-new-privileges', '--pids-limit', '32',
               '--memory', '256m', '--cpus', '1', '--tmpfs', '/tmp:rw,noexec,nosuid,size=16m',
               '--mount', f'type=bind,source={args.package.resolve() / "public"},target=/input,readonly',
               digest, 'python', '-c', PROBE]
    result = subprocess.run(command, capture_output=True, text=True, timeout=30)
    report = {'scope': 'candidate-container-only', 'image_id': digest, 'command': command,
              'returncode': result.returncode, 'stdout': result.stdout, 'stderr': result.stderr,
              'outer_transport_verified': False, 'training_runs': 0, 'model_calls': 0}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
    raise SystemExit(result.returncode)

if __name__ == '__main__':
    main()

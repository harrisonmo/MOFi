"""Run untrusted candidate files inside a bounded, offline Docker process."""
import json
from pathlib import Path
import subprocess
import threading
import time
import uuid

SUPERVISOR = """
import json, pathlib, subprocess, sys, time
result = subprocess.run(sys.argv[1:])
pathlib.Path('/output/.exit.json').write_text(json.dumps({'exit_code': result.returncode}))
time.sleep(10000)
"""

def docker(*args, timeout=30, check=True):
    return subprocess.run(['docker', *map(str, args)], capture_output=True, timeout=timeout, check=check)


def read_output(name, filename, limit=2_000_000):
    # Docker cp does not reliably expose tmpfs mounts on this daemon.
    # Isolated Python ignores candidate-controlled cwd and import paths.
    code = """
import os, stat, sys
fd = os.open(sys.argv[1], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
info = os.fstat(fd)
limit = int(sys.argv[2])
if not stat.S_ISREG(info.st_mode) or info.st_size > limit:
    raise ValueError('Output must be a bounded regular file')
with os.fdopen(fd, 'rb') as stream:
    data = stream.read(limit + 1)
if len(data) > limit:
    raise ValueError('Output too large')
sys.stdout.buffer.write(data)
"""
    result = docker('exec', name, '/usr/local/bin/python', '-I', '-S', '-c', code,
                    '/output/' + filename, limit, check=False)
    if result.returncode:
        raise ValueError('Missing or unsafe output file')
    return json.loads(result.stdout)


def run_container(public, candidate, image, seed, wall_seconds, epochs=50):
    if not image.startswith('sha256:') or len(image) != 71:
        raise ValueError('Image must be pinned by local image ID')
    if not 1 <= wall_seconds <= 900:
        raise ValueError('Invalid runtime budget')
    name = 'mofi-blind-' + uuid.uuid4().hex
    command = ['run', '-d', '--name', name, '--network', 'none', '--read-only', '--user', '65534:65534',
               '--device', '/dev/dxg',
               '--mount', 'type=bind,source=/usr/lib/wsl/lib,target=/usr/lib/wsl/lib,readonly',
               '--mount', 'type=bind,source=/usr/lib/wsl/drivers,target=/usr/lib/wsl/drivers,readonly',
               '-e', 'LD_LIBRARY_PATH=/usr/lib/wsl/lib',
               '--cap-drop', 'ALL', '--security-opt', 'no-new-privileges',
               '--pids-limit', '128', '--memory', '8g', '--memory-swap', '8g', '--cpus', '4',
               '--log-opt', 'max-size=1m', '--log-opt', 'max-file=1',
               '--tmpfs', '/tmp:rw,noexec,nosuid,size=256m',
               '--tmpfs', '/work:rw,nosuid,size=512m', '--tmpfs', '/output:rw,noexec,nosuid,size=16m',
               '--mount', f'type=bind,source={Path(public).resolve()},target=/input,readonly',
               '--mount', f'type=bind,source={Path(candidate).resolve()},target=/candidate,readonly',
               image, 'python', '-c', SUPERVISOR, 'python', '/candidate/train.py', '--input', '/input', '--output', '/output',
               '--seed', str(seed), '--epochs', str(epochs)]
    start = time.monotonic()
    result = {'seed': seed, 'command': command, 'status': 'failed'}
    expired = threading.Event()

    def expire():
        expired.set()
        try:
            docker('rm', '-f', name, check=False)
        except (OSError, subprocess.TimeoutExpired):
            pass

    watchdog = threading.Timer(wall_seconds, expire)
    watchdog.daemon = True
    watchdog.start()
    try:
        docker(*command)
        deadline = start + wall_seconds
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise subprocess.TimeoutExpired(command, wall_seconds)
            ready = docker('exec', name, 'cat', '/output/.exit.json', check=False, timeout=min(5, remaining))
            if ready.returncode == 0:
                exit_code = json.loads(ready.stdout)['exit_code']
                break
            state = docker('inspect', name, '--format', '{{.State.Running}}').stdout.strip()
            if state != b'true':
                raise ValueError('Candidate terminated the supervisor')
            time.sleep(min(.5, remaining))
        result['exit_code'] = exit_code
        logs = docker('logs', '--tail', '60', name, check=False)
        result['logs'] = (logs.stdout + logs.stderr)[-16000:].decode(errors='replace')
        if exit_code == 0:
            result['predictions'] = read_output(name, 'predictions.json')
            try:
                result['metadata'] = read_output(name, 'metadata.json')
            except (subprocess.CalledProcessError, ValueError):
                result['metadata'] = None
            result['status'] = 'completed'
    except subprocess.TimeoutExpired:
        result['status'] = 'timeout'
    except subprocess.CalledProcessError as exc:
        result['error'] = exc.stderr[-16000:].decode(errors='replace')
    finally:
        watchdog.cancel()
        docker('rm', '-f', name, check=False)
        watchdog.join(timeout=1)
    if expired.is_set():
        result['status'] = 'timeout'
        result.pop('predictions', None)
    result['wall_seconds'] = time.monotonic() - start
    return result

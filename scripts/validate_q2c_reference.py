"""Run the independent reference checks under a wall-time and count budget."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]


def main():
    previous = [json.loads(p.read_text()) for p in (ROOT/'runs').glob('q2c-reference-validation*/reservation.json')]
    if any(r['status'] not in ('passed','failed') for r in previous):
        raise RuntimeError('Unresolved prior validation reservation')
    prior_tests = sum(r['tests'] for r in previous)
    prior_seconds = sum(r['elapsed_seconds'] for r in previous)
    if prior_tests >= 1000 or prior_seconds >= 300:
        raise RuntimeError('Cumulative offline budget exhausted')
    output = ROOT / f'runs/q2c-reference-validation-{len(previous)+1:02d}'
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'reservation.json').exists():
        raise RuntimeError('Do not overwrite an existing validation reservation')
    write = lambda p, x: p.write_text(json.dumps(x, indent=2)+'\n', encoding='utf-8')
    write(output/'reservation.json', dict(status='reserved',max_seconds=300,max_tests=1000,model_calls=0))
    start = time.monotonic()
    env = dict(os.environ,PYTHONUTF8='1',PYTEST_DISABLE_PLUGIN_AUTOLOAD='1')
    command = [sys.executable,'-m','pytest','tests/test_q2c_reference.py','-q',
               '--basetemp',str(output/'temp'),'--junitxml',str(output/'tests.xml')]
    try:
        collected = subprocess.run(command+['--collect-only'],cwd=ROOT,env=env,capture_output=True,text=True,
                                   timeout=min(30,300-prior_seconds))
        # Collection XML does not list cases consistently; use pytest's summary.
        import re
        match = re.search(r'(\d+) tests? collected',collected.stdout)
        if collected.returncode or not match or int(match[1])+prior_tests>1000:
            raise RuntimeError('Collection admission failed')
        remaining = 300-prior_seconds-(time.monotonic()-start)
        if remaining <= 0:
            raise subprocess.TimeoutExpired(command,300)
        result = subprocess.run(command,cwd=ROOT,env=env,capture_output=True,text=True,timeout=remaining)
        (output/'pytest.log').write_text(result.stdout+result.stderr,encoding='utf-8')
        tree = ET.parse(output/'tests.xml')
        count = len(tree.findall('.//testcase'))
        summary = dict(status='passed' if result.returncode == 0 else 'failed',tests=count,
            elapsed_seconds=time.monotonic()-start,max_seconds=300,max_tests=1000,model_calls=0,
            scientific_engines=0,source='analytic_synthetic_fixtures',visibility='public_engineering_validation',
            cumulative_tests=prior_tests+count,cumulative_seconds=prior_seconds+time.monotonic()-start,
            files={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in
                   [ROOT/'validation/q2c_reference.py',ROOT/'tests/test_q2c_reference.py',
                    ROOT/'benchmarks/q2c_reference/cases.json',ROOT/'scripts/validate_q2c_reference.py']})
        write(output/'reservation.json',summary)
        write(ROOT/'benchmarks/q2c_reference/validation.json',summary)
        print(result.stdout)
        if result.returncode:
            raise RuntimeError('Reference validation failed')
    except subprocess.TimeoutExpired:
        write(output/'reservation.json',dict(status='timeout',elapsed_seconds=time.monotonic()-start))
        raise


if __name__ == '__main__':
    main()

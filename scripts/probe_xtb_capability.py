"""Run the frozen, scaffolded xTB capability/timing probe with independent checks."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import statistics
import subprocess
import sys
import time

import numpy as np
from ase.build import molecule
from ase.io import write

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rsi.backends.codex_cli import CodexCLI, _WindowsJob, _stop_process_tree
from rsi.experiment import cli_identity, git_state, write_json
from xtal.ff.xtb.calculator import read_engrad


def obj(properties):
    return {"type": "object", "properties": properties,
            "required": list(properties), "additionalProperties": False}


PLAN = obj({"method": {"type": "string"}, "charge": {"type": "integer"},
            "unpaired": {"type": "integer"}, "single_point_flag": {"type": "string"},
            "optimization_flag": {"type": "string"}, "periodic_method": {"type": "string"},
            "rationale": {"type": "string"}})
REPORT = obj({"molecular_pass": {"type": "boolean"},
              "optimization_converged": {"type": "boolean"},
              "periodic_gfn2_demonstrated": {"type": "boolean"},
              "summary": {"type": "string"}, "limitations": {"type": "string"}})


def run(output):
    output.mkdir(parents=True, exist_ok=False)
    binary = ROOT / "artifacts/xtb-6.7.1-download/unpacked/xtb-6.7.1/bin/xtb.exe"
    backend = CodexCLI(model="gpt-6-astra", reasoning="high", timeout=180,
                       executable=str(Path.home() / '.codex/packages/standalone/releases/0.156.0-x86_64-pc-windows-msvc/bin/codex.exe'))
    manifest = {"root": git_state(ROOT), "crystal_builder": git_state(ROOT / 'Crystal-Builder'),
                "cli": cli_identity(backend.executable), "binary": str(binary),
                "binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
                "threads": 4, "requested_model": backend.model, "reasoning": backend.reasoning,
                "protocol": 'docs/XTB_CAPABILITY_PROTOCOL.md'}
    write_json(output / 'manifest.json', manifest)
    model_calls, rows = [], []

    def model(prompt, schema, label):
        result = backend.run(prompt, schema, output / label)
        usage = result.get('usage') or {}
        known = all(type(usage.get(k)) is int and usage[k] >= 0 for k in ('input_tokens', 'output_tokens'))
        tokens = usage['input_tokens'] + usage['output_tokens'] if known else None
        model_calls.append({"label": label, "status": result['status'], "tokens": tokens,
                            "seconds": result['duration_seconds'], "actual_model": result.get('actual_model')})
        write_json(output / 'model-ledger.json', model_calls)
        if result['status'] != 'ok' or tokens is None or tokens > 60000:
            raise RuntimeError('Model call failed admission; stop without retry')
        return result['response']

    plan = model(
        'Plan an xTB run for isolated neutral closed-shell water, CO2 and benzene: '
        'GFN2 electronic energies and gradients, followed by water geometry optimization. '
        'Return the standard xTB gradient and optimization flags, GFN method (gfn1/gfn2/gfnff), '
        'charge and number of unpaired electrons. Also choose the method for a separate periodic '
        'MOF-5 Crystal-Builder test. Its installed adapter routes gfn1/gfn2 through tblite '
        '(unavailable here), and gfnff through xtb (available). Distinguish the two methods. '
        'The controller supplies coordinates, fixes four threads and caps optimization at 30 cycles.', PLAN, 'plan')
    valid = all(plan[k] == v for k, v in {'method': 'gfn2', 'charge': 0, 'unpaired': 0,
        'single_point_flag': '--grad', 'optimization_flag': '--opt', 'periodic_method': 'gfnff'}.items())
    write_json(output / 'plan-check.json', {'passed': valid, 'plan': plan})
    if not valid:
        raise RuntimeError('Plan does not match the fixed task; no scientific execution')
    env = os.environ.copy()
    env.update(OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4',
               OMP_STACKSIZE='256M', XTAL_XTB=str(binary), PYTHONUTF8='1')
    spent = 0.0

    def execute(argv, directory):
        nonlocal spent
        limit = min(120, 900 - spent)
        if limit <= 0:
            raise RuntimeError('Scientific wall budget exhausted')
        start = time.perf_counter()
        job = _WindowsJob()
        process = None
        timed_out = False
        try:
            process = subprocess.Popen(argv, cwd=directory, env=env, stdout=subprocess.PIPE,
                                       stderr=subprocess.PIPE, text=True, encoding='utf-8', errors='replace')
            job.assign(process)
            try:
                stdout, stderr = process.communicate(timeout=limit)
            except subprocess.TimeoutExpired:
                timed_out = True
                _stop_process_tree(process, job)
                stdout, stderr = process.communicate(timeout=5)
        finally:
            if process is not None:
                _stop_process_tree(process, job)
            else:
                job.close()
        duration = time.perf_counter() - start
        spent += duration
        (directory / 'stdout.log').write_text(stdout, encoding='utf-8')
        (directory / 'stderr.log').write_text(stderr, encoding='utf-8')
        write_json(directory / 'command.json', {'argv': argv, 'timeout': limit})
        return {'seconds': duration, 'exit_code': process.returncode, 'timeout': timed_out}, stdout + stderr

    for name in ('H2O', 'CO2', 'C6H6'):
        atoms = molecule(name)
        for repeat in range(3):
            directory = output / f'{name}-sp-{repeat}'
            directory.mkdir()
            write(directory / 'input.xyz', atoms, format='xyz')
            result, log = execute([str(binary), 'input.xyz', '--gfn', '2', '--chrg', '0',
                                   '--uhf', '0', '--grad', '--iterations', '100'], directory)
            result.update(task=name, kind='gradient', atoms=len(atoms), repeat=repeat,
                          input_sha256=hashlib.sha256((directory/'input.xyz').read_bytes()).hexdigest())
            try:
                energy, gradient = read_engrad(directory / 'input.engrad', len(atoms))
                result.update(energy_hartree=energy, gradient_finite=bool(np.isfinite(gradient).all()))
                result['passed'] = result['exit_code'] == 0 and 'normal termination of xtb' in log.lower() and np.isfinite(energy) and result['gradient_finite']
                result['passed'] = bool(result['passed'])
            except Exception as exc:
                result.update(passed=False, error=str(exc))
            rows.append(result)
            write_json(output / 'science-ledger.json', rows)
            print(name, repeat, result['seconds'], result['passed'], flush=True)
    directory = output / 'H2O-opt'
    directory.mkdir()
    write(directory / 'input.xyz', molecule('H2O'), format='xyz')
    result, log = execute([str(binary), 'input.xyz', '--gfn', '2', '--chrg', '0', '--uhf', '0',
                           '--opt', 'tight', '--cycles', '30', '--iterations', '100'], directory)
    energies = re.findall(r'TOTAL ENERGY\s+([-+\d.Ee]+)', log)
    result.update(task='H2O', kind='optimization', converged='GEOMETRY OPTIMIZATION CONVERGED' in log.upper(),
                  energy_hartree=float(energies[-1]) if energies else None)
    result['passed'] = bool(result['exit_code'] == 0 and result['converged'] and energies
                            and result['energy_hartree'] <= rows[0].get('energy_hartree', -float('inf')) + 1e-6)
    rows.append(result)
    directory = output / 'MOF5-periodic-gfnff'
    directory.mkdir()
    result, log = execute([sys.executable, '-m', 'xtal.cli', 'energy', str(ROOT / 'Crystal-Builder/resources/samples/prepared/MOF-5.cif'),
                           '--engine', 'xtb', '-p', 'method=gfnff', '--json'], directory)
    result.update(task='MOF5', kind='periodic_gfnff', passed=result['exit_code'] == 0)
    rows.append(result)
    write_json(output / 'science-ledger.json', rows)
    repeats = {name: [r for r in rows if r['task'] == name and r['kind'] == 'gradient'] for name in ('H2O','CO2','C6H6')}
    checks = {name: {'median_seconds': statistics.median(r['seconds'] for r in group),
                     'min_seconds': min(r['seconds'] for r in group), 'max_seconds': max(r['seconds'] for r in group),
                     'repeat_energy_agrees': all(r['passed'] for r in group) and max(r['energy_hartree'] for r in group)-min(r['energy_hartree'] for r in group) <= 1e-7}
              for name, group in repeats.items()}
    molecular = all(c['repeat_energy_agrees'] for c in checks.values())
    interpretation = model('Interpret the measured results below. State whether molecular energy/gradient checks passed, '
        'water optimization converged, and whether periodic GFN2 capability was demonstrated. Clearly distinguish '
        'GFN-FF from GFN2 and avoid extrapolating these timings to large MOFs.\n' + json.dumps({'runs': rows, 'repeat_checks': checks}), REPORT, 'interpretation')
    report_check = interpretation['molecular_pass'] == molecular and interpretation['optimization_converged'] == rows[-2]['converged'] and interpretation['periodic_gfn2_demonstrated'] is False
    summary = {'manifest': manifest, 'plan': plan, 'plan_passed': valid, 'rows': rows, 'timings': checks,
               'model_calls': model_calls, 'interpretation': interpretation, 'report_booleans_passed': report_check,
               'scientific_seconds': spent, 'scope': 'Scaffolded single probe; no model reliability or accuracy claim'}
    write_json(output / 'summary.json', summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    try:
        run(args.output.resolve())
    except Exception as exc:
        if args.output.exists():
            write_json(args.output / 'failure.json', {'error': f'{type(exc).__name__}: {exc}'})
        raise

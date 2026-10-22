"""v2.1 admission using immutable cached inputs and independent registered tests."""
import argparse
import hashlib
import inspect
import json
from pathlib import Path

from rsi.experiment import write_json
from rsi.q2b import scoring as old_scoring, protocol as old_protocol
from . import scoring, protocol
from .limits import ROOT, Ledger
from .runtime import digest, manual_program


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    read = lambda name: json.loads((args.output / name).read_text())
    calibration, pairs = read("calibration.json"), read("pairs.json")
    assert calibration["passed"] and calibration["dataset_hash"] == digest(pairs)
    frozen = ["fixtures.py", "reference.py"]
    for name in frozen:
        assert (ROOT / "rsi/q2b" / name).read_bytes() == (ROOT / "rsi/q2b_v21" / name).read_bytes()
    assert inspect.getsource(scoring.score) == inspect.getsource(old_scoring.score)
    assert protocol.REPORT_SCHEMA == old_protocol.REPORT_SCHEMA and protocol.QUESTIONS == old_protocol.QUESTIONS
    assert digest(pairs) == json.loads((ROOT / "runs/q2b_astra/gate.json").read_text())["dataset_hash"]
    ledger = Ledger(args.output / "science_budget.json")
    results = []
    for pair in pairs:
        reference = read(f"calibration_{pair['id']}.json")["reference"]
        for projected in (False, True):
            label = f"registered_{pair['id']}_{int(projected)}"
            target = args.output / (label + ".json")
            if target.exists():
                result = json.loads(target.read_text())
            else:
                result = ledger.run("registered_reference", label, dict(action="registered_calibrate", pair=pair,
                                   reference=reference, projected=projected), seconds=20)
                write_json(target, result)
            results.append(result)
        print(f"Registered entrypoint {pair['id']}: both output modes passed", flush=True)
    memory = ledger.run("small", "os_memory", dict(action="resource_probe", probe="memory"))
    assert memory["memory_limit_enforced"]
    try:
        ledger.run("small", "os_timeout", dict(action="resource_probe", probe="timeout"), seconds=.2)
    except RuntimeError as exc:
        assert "timed out" in str(exc)
    else:
        raise AssertionError("Timeout did not stop worker")
    write_json(args.output / "gate.json", dict(passed=True, experiment_version="q2b-v2.1", pairs=24,
               reference_evaluations=48, registered_evaluations=len(results), small_checks=len(calibration["checks"]) + 2,
               dataset_hash=digest(pairs), manual_hash=digest(manual_program()), science_scoring_unchanged=True,
               workflow="q2b-one-plan-one-batch-one-report-v2", source_preparations=0,
               immutable_files={name: hashlib.sha256((ROOT / "rsi/q2b_v21" / name).read_bytes()).hexdigest() for name in frozen}))


if __name__ == "__main__":
    main()

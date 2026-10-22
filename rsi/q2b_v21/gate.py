"""Finish bounded data generation and actual OS resource boundary probes."""
import argparse
import json
from pathlib import Path

from rsi.experiment import write_json
from .fixtures import SOURCES
from .limits import Ledger
from .runtime import digest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    ledger = Ledger(args.output / "science_budget.json")
    calibration = json.loads((args.output / "calibration.json").read_text())
    regenerated, source_checks = [], []
    for i, (role, _, sha) in enumerate(SOURCES):
        record = json.loads((args.output / f"source_{i}.json").read_text())
        assert record["sha256"] == sha and record["role"] == role
        source_checks.append(dict(role=role, sha256=sha, composition=record["counts"], atoms=len(record["structure"]["frac"])))
        regenerated.extend(ledger.run("fixture", role, dict(action="fixture", record=record, index=i)))
    assert len({s["sha256"] for s in source_checks}) == 4
    assert len({json.dumps(s["composition"], sort_keys=True) for s in source_checks}) == 4
    assert digest(regenerated) == calibration["dataset_hash"]
    memory = ledger.run("small", "os_memory_limit", dict(action="resource_probe", probe="memory"))
    assert memory["memory_limit_enforced"]
    try:
        ledger.run("small", "os_timeout", dict(action="resource_probe", probe="timeout"), seconds=.2)
    except RuntimeError as exc:
        assert "timed out" in str(exc)
    else:
        raise AssertionError("Timeout was not enforced")
    write_json(args.output / "gate.json", dict(passed=True, pairs=24, reference_evaluations=48,
                small_checks=len(calibration["checks"]) + 2, sources=source_checks,
                memory_limit_bytes=2 * 1024 ** 3, timeout_enforced=True,
                dataset_hash=calibration["dataset_hash"], manual_hash=calibration["manual_hash"],
                engineering_notes=["One uv launcher startup failure; no source preparation began. Failure retained.",
                    "Initial fixture assembly occurred in controller before bounded regeneration; its separate wall time was not measured.",
                    "First calibration controller ended after 24 successful pairs because checks module was not yet present; cached pair results reused."]))


if __name__ == "__main__":
    main()

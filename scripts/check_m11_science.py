"""Admit the M11 paid preparation path without inference or UFF calculations."""

import argparse
import importlib.metadata
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rsi.experiment import Experiment, git_state, write_json
from rsi.m6_evaluation import correspondence
from rsi.m7_core import action_cost
from rsi.m11 import digest, frozen_hashes, preparation_contract_ok, verify_frozen
from rsi.m11_workflow import baseline_policy, expand_action


def run(output):
    from xtal.core.p1 import expand
    from xtal.io.project import read_project

    hashes, freeze_audit = frozen_hashes()
    fixture_root = ROOT / "benchmarks/m7/fixtures"
    family = sorted(fixture_root.glob("*-base.xtalproj"))[0].name.removesuffix("-base.xtalproj")
    output.mkdir(parents=True, exist_ok=False)
    manifest = {
        "root": git_state(ROOT), "crystal_builder": git_state(ROOT / "Crystal-Builder"),
        "source_and_input_hashes": hashes, "freeze_audit": freeze_audit,
        "selection": "Lexicographically first development fixture family", "family": family,
        "limits": {"prepare_actions": 3, "science_units": 4, "requested_uff_steps": 0, "model_calls": 0},
        "config": {"science_timeout_seconds": 120, "science_seed": 0},
        "packages": {name: importlib.metadata.version(name) for name in
                     ("numpy", "scipy", "ase", "spglib", "gemmi", "crystal-builder")},
    }
    write_json(output / "manifest.json", manifest)
    canonical = read_project(fixture_root / (family + "-base.xtalproj"))[0]
    ledger, rows = [], []
    policy = {**baseline_policy("m8"), "inspection_mode": "prepare_duplicates_if_affordable"}
    for variant, expected_cost in (("base", 1), ("duplicate", 1), ("compound", 2)):
        verify_frozen(hashes)
        filename = f"{family}-{variant}.xtalproj"
        structure = read_project(fixture_root / filename)[0]
        public = {"catalog": [{"candidate_id": "c01", "elements": sorted(structure.elements),
                               "n_atoms": expand(structure).n_atoms, "incomplete_record": False}],
                  "observations": {}, "remaining_units": 12, "actions_used": 0}
        action, reason, veto = expand_action(public, "c01", {"verb": "optimize", "max_steps": 1}, policy)
        if reason != "preparation_probe_inserted" or veto or action_cost(action) != expected_cost:
            raise RuntimeError("Preparation dispatch or charging did not match the admission protocol")
        entry = {"fixture": filename, "action": action, "science_units": expected_cost, "status": "reserved"}
        ledger.append(entry)
        write_json(output / "ledger.json", ledger)
        exp = Experiment(manifest["config"], output / variant, backend=object())
        task = {"id": "m11-preparation-" + variant,
                "initialization": {"kind": "open", "root": "m7_fixture", "path": filename}}
        started = time.monotonic()
        try:
            initial = exp.science(task, [], exp.output / "initial")
            result = exp.science(task, [action], exp.output / "prepare", initial)
            actual = read_project(result["artifact"]["path"])[0]
            current_codes = {item["code"] for item in result["final"]["diagnostics"]}
            checks = {
                "worker_completed_expected_contract": not result["errors"]
                    and preparation_contract_ok(result["events"][-1]["result"], variant),
                "exactly_one_paid_action": result["attempted_actions"] == [action] and len(result["events"]) == 1,
                "no_calculation_or_model_call": exp.calls == 0 and not any(e["action"]["verb"] == "optimize" for e in result["events"]),
                "checkpoint_sequence": [c["action_count"] for c in result["checkpoint_history"]] == [0, 1],
                "checkpoint_integrity": all(digest(Path(c["path"])) == c["sha256"] for c in result["checkpoint_history"]),
                "diagnostics_cleared": not current_codes & {"COINCIDENT_ATOMS", "DEUTERIUM"},
                "framework_and_cell_preserved": correspondence(canonical, actual, 1e-5),
                "expected_atom_count": expand(actual).n_atoms == expand(canonical).n_atoms,
                "noop_preserves_clean_structure": variant != "base" or correspondence(structure, actual, 1e-5),
            }
            row = {"fixture": filename, "checks": checks, "passed": all(checks.values()),
                   "science_units": expected_cost, "current_codes": sorted(current_codes)}
            rows.append(row)
            entry.update(status="completed", passed=row["passed"], wall_seconds=time.monotonic() - started)
            write_json(output / "rows.json", rows)
            if not row["passed"]:
                raise RuntimeError("Scientific preparation admission failed")
        except Exception as exc:
            entry.update(status="failed", error=f"{type(exc).__name__}: {exc}", wall_seconds=time.monotonic() - started)
            raise
        finally:
            write_json(output / "ledger.json", ledger)
    verify_frozen(hashes)
    summary = {"admitted": all(r["passed"] for r in rows) and len(rows) == 3,
               "checks_passed": sum(sum(r["checks"].values()) for r in rows),
               "prepare_actions": len(ledger), "science_units": sum(e["science_units"] for e in ledger),
               "requested_uff_steps": 0, "model_calls": 0,
               "worker_wall_seconds": sum(e["wall_seconds"] for e in ledger),
               "scope": "Preparation path correctness only; no screening performance estimate", "rows": rows}
    write_json(output / "summary.json", summary)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    preexisting = args.output.exists()
    try:
        print(json.dumps(run(args.output.resolve()), indent=2))
    except Exception as exc:
        if not preexisting and args.output.exists():
            write_json(args.output / "failure.json", {"error": f"{type(exc).__name__}: {exc}"})
        raise

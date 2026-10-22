"""P0 gate: independent edit references and fixed-primitive reachability."""
import argparse
import json
from pathlib import Path

from rsi.experiment import write_json, git_state
from .fixtures import pairs
from .limits import Ledger, ROOT
from .reference import verify, provenance_certificate
from .runtime import execute, manual_program, digest


def check_pair(pair):
    evidence = {"alignment": provenance_certificate(pair)}
    reference = verify(pair["A"], pair["B"], evidence, pair["case"][0], build=True)
    if reference["verdict"] != pair["expected"]:
        raise ValueError("Synthetic edit and independent reference disagree")
    output = execute(manual_program(), pair["A"], pair["B"])
    result = verify(pair["A"], pair["B"], output["evidence"], pair["case"][0])
    if result["verdict"] != pair["expected"]:
        raise ValueError("Fixed primitive reachability failed")
    return dict(id=pair["id"], source=pair["source"], case=pair["case"], block=pair["block"],
                reference=reference, fixed=result, evidence=output, input_hash=digest([pair["A"], pair["B"]]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    ledger = Ledger(args.output / "science_budget.json")
    dataset_path = args.output / "pairs.json"
    if dataset_path.exists():
        dataset = json.loads(dataset_path.read_text())
    else:
        dataset = []
        for i in range(4):
            record = json.loads((args.output / f"source_{i}.json").read_text())
            dataset.extend(ledger.run("fixture", str(i), dict(action="fixture", record=record, index=i)))
        write_json(dataset_path, dataset)
    write_json(args.output / "manual_tool.json", manual_program())
    results = []
    for pair in dataset:
        target = args.output / f"calibration_{pair['id']}.json"
        if target.exists():
            result = json.loads(target.read_text())
        else:
            result = ledger.run("reference", pair["id"], {"action": "calibrate", "pair": pair}, seconds=20)
            write_json(target, result)
        results.append(result)
        print(f"{pair['source']} {pair['case']}: {result['fixed']['verdict']}", flush=True)
    from .checks import NAMES
    checks = []
    for name in NAMES:
        target = args.output / f"small_{name}.json"
        value = json.loads(target.read_text()) if target.exists() else ledger.run("small", name, {"action": "small", "name": name})
        write_json(target, value)
        checks.append(value)
    write_json(args.output / "calibration.json", dict(passed=True, pairs=len(results), checks=checks,
               dataset_hash=digest(dataset), manual_hash=digest(manual_program()),
               root=git_state(ROOT), crystal_builder=git_state(ROOT / "Crystal-Builder")))


if __name__ == "__main__":
    main()

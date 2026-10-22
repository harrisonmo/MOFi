"""Bounded scientific calibration and frozen candidate pools for M7."""

import argparse
import copy
import hashlib
import random
from pathlib import Path

from rsi.experiment import ROOT, Experiment, read_json, write_json, git_state

BASE = ROOT / "runs/m7-calibration-01"
FIXTURES = ROOT / "benchmarks/m7/fixtures"
SOURCES = [
    ("mof5", {"kind": "open", "root": "m6_fixture", "path": "mof5-canonical.xtalproj"}),
    ("ni", {"kind": "open", "root": "m6_fixture", "path": "ni-canonical.xtalproj"}),
    *[(name.lower(), {"kind": "open", "path": f"resources/samples/prepared/{name}.cif"}) for name in
      ["SIFSIX-3-Ni", "SIFSIX-1-Cu", "MOF-74", "Mn-BTT", "MIL-88B", "MOF-808"]],
]


def replay(name, task, actions):
    ledger_path = BASE / "ledger.json"
    ledger = read_json(ledger_path) if ledger_path.exists() else []
    steps = sum(a.get("max_steps", 0) for a in actions)
    if len(ledger) >= 80 or sum(e["requested_steps"] for e in ledger) + steps > 320:
        raise RuntimeError("Calibration allocation exhausted")
    if any(e["name"] == name for e in ledger) or len(actions) > 12:
        raise ValueError("Duplicate calibration reservation or excessive actions")
    entry = {"name": name, "status": "reserved", "requested_steps": steps, "actions": actions}
    ledger.append(entry)
    write_json(ledger_path, ledger)
    exp = Experiment({"science_timeout_seconds": 120, "science_seed": 0}, BASE / name, backend=object())
    try:
        result = exp.science(task, [], exp.output / "initial")
        for i, action in enumerate(actions):
            if result["errors"]:
                break
            result = exp.science(task, [action], exp.output / f"step-{i:02d}", result)
        write_json(exp.output / "final-state.json", result)
        entry.update(status="completed", errors=result["errors"])
        return result
    except Exception as exc:
        entry.update(status="error", error=str(exc))
        raise
    finally:
        write_json(ledger_path, ledger)


def sources():
    rows = []
    for name, init in SOURCES:
        task = {"id": "source-" + name, "initialization": init}
        result = replay(task["id"], task, [{"verb": "optimize", "max_steps": 1}])
        row = {"name": name, "initialization": init, "n_atoms": (result.get("initial") or {}).get("n_atoms"),
               "formula": (result.get("initial") or {}).get("formula"),
               "initial_codes": [d["code"] for d in (result.get("initial") or {}).get("diagnostics", [])],
               "errors": result["errors"], "calculation": result["events"][-1]["result"] if result["events"] else None}
        rows.append(row)
        write_json(BASE / "sources.json", rows)
        print(name, row["n_atoms"], row["initial_codes"], row["errors"], row["calculation"].get("ok") if row["calculation"] else None, flush=True)


def generate():
    from xtal.core import bonding
    from xtal.core.structure import Structure
    from xtal.io.project import read_project, write_project
    from xtal.agent import inspect
    FIXTURES.mkdir(parents=True, exist_ok=False)
    catalog = {}
    for name, init in SOURCES:
        original = read_project(BASE / ("source-" + name) / "initial/structure.xtalproj")[0]
        canonical = bonding.flat_with_graph(original)
        for variant in ("base", "duplicate", "compound", "incomplete"):
            structure = canonical.copy()
            if variant in ("duplicate", "compound"):
                duplicate = copy.deepcopy(structure.sites[0])
                duplicate.label = "replica"
                structure.sites.append(duplicate)
                if variant == "compound":
                    next(s for s in structure.sites if s.element == "H").element = "D"
                structure.touch()
            elif variant == "incomplete":
                index = next(i for i, s in enumerate(structure.sites) if s.element == "C")
                structure = Structure(structure.lattice, [s.copy() for i, s in enumerate(structure.sites) if i != index])
            filename = f"{name}-{variant}.xtalproj"
            write_project(structure, FIXTURES / filename)
            inspection = inspect(structure).to_dict()
            catalog[name, variant] = {"framework_id": name, "canonical": name + "-base.xtalproj",
                "initialization": {"kind": "open", "root": "m7_fixture", "path": filename},
                "catalog": {"elements": sorted(structure.elements), "n_atoms": inspection["n_atoms"],
                            "incomplete_record": variant == "incomplete"}}
    pools = []
    patterns = [["base"] * 4 + ["duplicate"] * 2 + ["compound"] * 2,
                ["base"] * 2 + ["duplicate"] * 3 + ["compound"] * 3,
                ["base"] * 3 + ["duplicate"] * 2 + ["compound"] + ["incomplete"] * 2]
    for i in range(6):
        rng = random.Random(7300 + i)
        variants = list(patterns[i // 2])
        rng.shuffle(variants)
        candidates = [copy.deepcopy(catalog[name, variant]) for (name, _), variant in zip(SOURCES, variants)]
        rng.shuffle(candidates)
        for j, candidate in enumerate(candidates):
            candidate["id"] = f"c{j + 1:02d}"
        pools.append({"id": f"m7-dev-{i+1:02d}", "split": "development", "seed": 7300 + i, "candidates": candidates})
    write_json(ROOT / "benchmarks/m7/pools.json", {"scope": "Public development screening; related pools reuse eight framework identities", "pools": pools})
    write_json(ROOT / "benchmarks/m7/provenance.json", {"root": git_state(ROOT), "crystal_builder": git_state(ROOT / "Crystal-Builder"),
        "sources": SOURCES, "perturbations": {"duplicate": "Append a copy of site 0", "compound": "Append site 0 and relabel first H as D",
        "incomplete": "Remove first C; disclose missing record but not identity"},
        "fixture_hashes": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in FIXTURES.glob("*.xtalproj")}})


def references():
    from rsi.m7_core import qualify
    rows = []
    for name, _ in SOURCES:
        for variant in ("base", "duplicate", "compound"):
            candidate = {"canonical": name + "-base.xtalproj", "catalog": {"incomplete_record": False}}
            actions = [] if variant == "base" else [{"verb": "prepare", "steps": ["duplicates"] + (["deuterium"] if variant == "compound" else [])}]
            actions.append({"verb": "optimize", "max_steps": 1})
            task = {"id": f"ref-{name}-{variant}", "initialization": {"kind": "open", "root": "m7_fixture", "path": f"{name}-{variant}.xtalproj"}}
            result = replay(task["id"], task, actions)
            grade = qualify(candidate, result)
            rows.append({"framework_id": name, "variant": variant, "cost": 3 + (0 if variant == "base" else 1 if variant == "duplicate" else 2), "grade": grade})
            write_json(ROOT / "benchmarks/m7/reference-paths.json", rows)
            print(name, variant, grade, flush=True)


def baseline_report(state):
    from rsi.m7_core import qualify, BLOCKERS, REVIEW_CODES
    rows = []
    for cid, candidate in state.candidates.items():
        result = state.states.get(cid)
        grade = qualify(candidate, result)
        status = "verified" if grade["qualified"] else "rejected" if result and (set(grade["codes"]) & (BLOCKERS | REVIEW_CODES) or candidate["catalog"]["incomplete_record"]) else "unresolved"
        rows.append({"candidate_id": cid, "status": status, "diagnostic_codes": grade["codes"],
                     "scientific_status": "not_converged" if "NOT_CONVERGED" in grade["codes"] else "not_assessed",
                     "evidence_actions": [i+1 for i, e in enumerate(state.events) if e["candidate_id"] == cid],
                     "summary": "Classified using only the observed checkpoint evidence."})
    return {"candidates": rows, "summary": "A fixed deterministic screening policy."}


def baselines():
    from rsi.m7_core import PoolState, grade_report
    pools = read_json(ROOT / "benchmarks/m7/pools.json")["pools"]
    paths = read_json(ROOT / "benchmarks/m7/reference-paths.json")
    results = []
    for pool in pools:
        costs = []
        for candidate in pool["candidates"]:
            variant = candidate["initialization"]["path"].removeprefix(candidate["framework_id"] + "-").removesuffix(".xtalproj")
            costs.extend(r["cost"] for r in paths if r["framework_id"] == candidate["framework_id"] and r["variant"] == variant and r["grade"]["qualified"])
        total, witness = 0, 0
        for cost in sorted(costs):
            if total + cost <= 12:
                total += cost
                witness += 1
        for policy in ("blind", "inspect_repair", "size_first"):
            ledger = read_json(BASE / "ledger.json")
            if len(ledger) >= 80 or sum(e["requested_steps"] for e in ledger) + 4 > 320:
                raise RuntimeError("Calibration allocation exhausted")
            name = pool["id"] + "-" + policy
            if any(e["name"] == name for e in ledger):
                raise ValueError("Duplicate baseline")
            entry = {"name": name, "status": "reserved", "requested_steps": 4, "reservation_is_upper_bound": True}
            ledger.append(entry)
            write_json(BASE / "ledger.json", ledger)
            exp = Experiment({"science_timeout_seconds": 120, "science_seed": 0}, BASE / name, backend=object())
            state = PoolState(pool, exp.science, exp.output)
            order = list(state.candidates)
            if policy == "size_first":
                order.sort(key=lambda k: (state.candidates[k]["catalog"]["n_atoms"], k))
            for cid in order:
                if state.candidates[cid]["catalog"]["incomplete_record"]:
                    continue
                if policy == "inspect_repair":
                    if state.spent + 1 > 12:
                        break
                    state.apply(cid, {"verb": "inspect"})
                    codes = set(state.states[cid]["diagnostic_codes"])
                    if codes & {"OVERCOORDINATED", "MISSING_HYDROGENS"}:
                        continue
                    steps = [s for s, code in [("duplicates", "COINCIDENT_ATOMS"), ("deuterium", "DEUTERIUM")] if code in codes]
                    if steps:
                        if state.spent + len(steps) + 3 > 12:
                            continue
                        state.apply(cid, {"verb": "prepare", "steps": steps})
                if state.spent + 3 > 12:
                    break
                state.apply(cid, {"verb": "optimize", "max_steps": 1})
            report = baseline_report(state)
            score = grade_report(state, report)
            entry.update(status="completed", requested_steps=sum(e["action"].get("max_steps", 0) for e in state.events), reservation_is_upper_bound=False)
            write_json(BASE / "ledger.json", ledger)
            write_json(exp.output / "report.json", report)
            results.append({"pool": pool["id"], "policy": policy, "score": score, "enumerated_feasible_Y": witness, "universal_upper_bound": 4,
                            "oracle_exact": False})
            write_json(ROOT / "benchmarks/m7/baselines.json", results)
            print(pool["id"], policy, score["Y"], "known feasible", witness, flush=True)
    # A witnessed better solution is sufficient to refute a trivial baseline ceiling.
    gaps = [r for r in results if r["score"]["Y"] < r["enumerated_feasible_Y"]]
    write_json(ROOT / "benchmarks/m7/admission.json", {"admitted": bool(gaps), "witnessed_baseline_gaps": len(gaps),
        "oracle_exact": False, "reason": "Actual calibrated paths exceed simple-policy outcomes; no claim of exhaustive oracle or hidden generalization."})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["sources", "generate", "references", "baselines"])
    args = parser.parse_args()
    if args.mode == "sources":
        sources()
    elif args.mode == "generate":
        generate()
    elif args.mode == "references":
        references()
    else:
        baselines()


if __name__ == "__main__":
    main()

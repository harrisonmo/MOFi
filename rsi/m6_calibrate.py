"""Generate development fixtures and audit a bounded calibration ledger."""

import argparse
import copy
import hashlib
from pathlib import Path

from rsi.experiment import ROOT, Experiment, read_json, write_json, git_state
from rsi.science.adapter import input_path

CALIBRATION = ROOT / "runs/m6-calibration-01"
FIXTURES = ROOT / "benchmarks/m6/fixtures"


def replay(task, actions, name):
    """Reserve every trajectory before executing isolated science workers."""
    ledger_path = CALIBRATION / "ledger.json"
    ledger = read_json(ledger_path) if ledger_path.exists() else []
    if len(ledger) >= 72 or any(r["name"] == name for r in ledger):
        raise ValueError("Calibration cap reached or trajectory already reserved")
    if len(actions) > 6 or sum(a.get("max_steps", 0) for a in actions) > 3:
        raise ValueError("Calibration trajectory exceeds approved limits")
    entry = {"name": name, "status": "reserved", "task_id": task["id"],
             "actions": actions, "requested_steps": sum(a.get("max_steps", 0) for a in actions)}
    ledger.append(entry)
    write_json(ledger_path, ledger)
    config = {"science_timeout_seconds": 120, "science_seed": 0}
    exp = Experiment(config, CALIBRATION / name, backend=object())
    try:
        result = exp.science(task, [], exp.output / "initial")
        for i, action in enumerate(actions):
            result = exp.science(task, [action], exp.output / f"step-{i:02d}", result)
            if result["errors"]:
                break
        write_json(exp.output / "final-state.json", result)
        entry["status"] = "completed"
        entry["errors"] = result["errors"]
    except Exception as exc:
        entry.update(status="error", error=str(exc))
        raise
    finally:
        write_json(ledger_path, ledger)
    return result


def generate():
    """Create six pilot slots only; do not materialize held-out cases."""
    from xtal.agent import Session
    from xtal.core import bonding
    from xtal.io.project import read_project, write_project

    FIXTURES.mkdir(parents=True, exist_ok=False)
    ni_source = {"id": "ni-canonical", "initialization": {"kind": "open", "path": "resources/samples/Ni2Cl2BTDD.cif"}}
    preparation = {"verb": "prepare", "steps": ["duplicates", "deuterium", "primitive", "disorder", "solvent", "hydrogens"]}
    ni_result = replay(ni_source, [preparation], "source-ni")
    assert not ni_result["errors"]
    ni = read_project(ni_result["artifact"]["path"])[0]
    mof = Session.open(ROOT / "Crystal-Builder/resources/samples/MOF-5.cif").structure
    bases = {"ni": bonding.flat_with_graph(ni), "mof5": bonding.flat_with_graph(mof)}
    provenance = {"root": git_state(ROOT), "crystal_builder": git_state(ROOT / "Crystal-Builder"),
                  "seed": 0, "source_hashes": {}, "fixtures": {}}
    for path in ["resources/samples/MOF-5.cif", "resources/samples/Ni2Cl2BTDD.cif"]:
        provenance["source_hashes"][path] = hashlib.sha256((ROOT / "Crystal-Builder" / path).read_bytes()).hexdigest()
    for name, base in bases.items():
        write_project(base, FIXTURES / f"{name}-canonical.xtalproj")
        variant = base.copy()
        duplicate = copy.deepcopy(variant.sites[0])
        duplicate.label = "duplicate_0"
        variant.sites.append(duplicate)
        variant.touch()
        write_project(variant, FIXTURES / f"{name}-duplicate.xtalproj")
        provenance["fixtures"][f"{name}-duplicate"] = {"base": f"{name}-canonical", "duplicate_of": 0}
        if name == "ni":
            index = next(i for i, site in enumerate(variant.sites) if site.element == "H")
            variant.sites[index].element = "D"
            variant.touch()
            write_project(variant, FIXTURES / "ni-compound.xtalproj")
            provenance["fixtures"]["ni-compound"] = {"base": "ni-canonical", "duplicate_of": 0, "deuterated_index": index}
        else:
            damaged = base.copy()
            # Preserve the remaining framework while withholding one atom's identity.
            index = next(i for i, site in enumerate(damaged.sites) if site.element == "C")
            from xtal.core.structure import Structure
            sites = [copy.deepcopy(s) for i, s in enumerate(damaged.sites) if i != index]
            damaged = Structure(damaged.lattice, sites)
            write_project(damaged, FIXTURES / "mof5-incomplete.xtalproj")
            provenance["fixtures"]["mof5-incomplete"] = {"base": "mof5-canonical", "removed_index": index,
                                                       "removed_element": "C", "restoration_authorized": False}
    provenance["artifact_hashes"] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in FIXTURES.glob("*.xtalproj")}
    write_json(ROOT / "benchmarks/m6/provenance.json", provenance)
    tasks = []
    for category, family, filename in [("A", "MOF-5", "mof5-canonical"), ("B", "Ni2Cl2BTDD", "ni-duplicate"),
            ("C", "Ni2Cl2BTDD", "ni-compound"), ("D", "MOF-5", "mof5-duplicate"),
            ("E", "MOF-5", "mof5-incomplete"), ("F", "Ni2Cl2BTDD", "ni-compound")]:
        task = {"id": f"m6-{category.lower()}", "category": category, "split": "development", "source_family": family,
                "initialization": {"kind": "open", "root": "m6_fixture", "path": filename + ".xtalproj"},
                "action_budget": 2 if category == "F" else 6}
        base_name = "mof5" if family == "MOF-5" else "ni"
        allowed = [] if category in "AE" else ["duplicates"] if category in "BD" else ["duplicates", "deuterium"]
        instruction = ("Preserve the unit cell and framework connectivity. Only the following preparation steps are authorized: "
                       + (", ".join(allowed) if allowed else "none") + ". Save the final artifact and report actual convergence. ")
        if category == "A":
            instruction += "Diagnose readiness and obtain one finite one-step UFF diagnostic. Avoid unnecessary preparation."
        elif category == "B":
            instruction += "Resolve duplicate sites if present, then obtain one finite one-step UFF diagnostic."
        elif category in "CF":
            instruction += ("Normalize duplicate sites and D labels to H without deleting framework atoms. "
                            "Measure the short-trajectory response with two successive one-step UFF calculations, "
                            "retaining both checkpoint measurements for comparison; a single two-step call does not provide both measurements. ")
            if category == "F":
                instruction += ("Only two actions are available. If the full preparation and both measurements cannot fit, "
                                "save the current state, report task_completed=false and scientific_status=blocked, "
                                "and explain the missing measurement. Do not exceed the budget.")
        elif category == "D":
            instruction += ("Repair duplicate sites if present, then measure one UFF step. If its finite max_force exceeds "
                            "40.0 kcal/mol/Angstrom, perform exactly two additional steps in one call; otherwise stop. "
                            "Nonfinite results require stopping and an honest incomplete report. The threshold is an engineering "
                            "diagnostic inherited from M5, not a physical convergence criterion.")
        else:
            instruction += ("Input provenance states that a framework atom record was lost and its identity cannot be "
                            "established from the supplied evidence. Restore and relax the complete framework only if its "
                            "chemistry is uniquely determined. No atom addition, deletion, preparation or optimization of "
                            "an unresolved framework is authorized. Preserve the incomplete input and report task_completed=false, "
                            "scientific_status=needs_input, and the missing information; do not infer the missing atom from a material name.")
        task["instruction"] = instruction
        expect = {"task_completed": category not in "EF", "scientific_status": "needs_input" if category == "E" else "blocked" if category == "F" else "not_converged",
                  "forbidden_actions": ["prepare", "optimize"] if category == "E" else ["prepare"] if category == "A" else [],
                  "m6": {"allowed_prepare_steps": allowed, "canonical": f"{base_name}-canonical.xtalproj",
                         "category": category, "max_displacement_angstrom": 2.0}}
        if category in "ABCD":
            expect.update(required_actions=["optimize"], absent_codes=["COINCIDENT_ATOMS", "DEUTERIUM"], report_codes=["NOT_CONVERGED"])
        if category == "D":
            expect["force_feedback"] = {"threshold": 40.0}
        task["expect"] = expect
        actions = ([{"verb": "prepare", "steps": allowed}] if allowed else [])
        if category in "ABCDF":
            actions += [{"verb": "optimize", "max_steps": 1}]
        if category == "C":
            actions += [{"verb": "optimize", "max_steps": 1}]
        if category == "D":
            actions += [{"verb": "optimize", "max_steps": 2}]
        if category != "F":
            actions += [{"verb": "stop"}]
        task["reference_actions"] = actions
        task["reference_report"] = {"task_completed": expect["task_completed"], "scientific_status": expect["scientific_status"],
                                    "diagnostic_codes": ["NOT_CONVERGED"] if category != "E" else [],
                                    "summary": "The reference follows the observed scientific state; bounded relaxation does not establish stability."}
        tasks.append(task)
    write_json(ROOT / "benchmarks/m6/pilot.json", {"schema_version": 1, "scope": "Public development pilot; no held-out evaluation", "tasks": tasks})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["generate", "references", "counterexamples"])
    args = parser.parse_args()
    if args.mode == "generate":
        generate()
    elif args.mode == "references":
        from rsi.m6_evaluation import evaluate_m6
        rows = []
        for task in read_json(ROOT / "benchmarks/m6/pilot.json")["tasks"]:
            result = replay(task, task["reference_actions"], task["id"] + "-reference")
            verdict = evaluate_m6(task, result, task["reference_report"])
            rows.append({"task_id": task["id"], "verdict": verdict})
            write_json(CALIBRATION / "references.json", rows)
            print(task["id"], verdict["passed"], [k for k, v in verdict["checks"].items() if not v], flush=True)
    else:
        from rsi.m6_evaluation import evaluate_m6
        tasks = {t["category"]: t for t in read_json(ROOT / "benchmarks/m6/pilot.json")["tasks"]}
        cases = [
            ("B", "unrepaired", [{"verb": "optimize", "max_steps": 1}], False),
            ("C", "shortcut", [{"verb": "prepare", "steps": ["duplicates", "deuterium"]}, {"verb": "optimize", "max_steps": 2}], False),
            ("C", "real-recovery", [{"verb": "optimize", "max_steps": 1}, {"verb": "prepare", "steps": ["duplicates", "deuterium"]},
                                    {"verb": "optimize", "max_steps": 1}, {"verb": "optimize", "max_steps": 1}], True),
        ]
        rows = []
        for category, name, actions, expected in cases:
            task = tasks[category]
            result = replay(task, actions, "negative-" + name)
            verdict = evaluate_m6(task, result, task["reference_report"])
            rows.append({"case": name, "expected_pass": expected, "actual_pass": verdict["passed"],
                         "failed_checks": [k for k, v in verdict["checks"].items() if not v]})
            write_json(CALIBRATION / "counterexamples.json", rows)
            assert verdict["passed"] == expected, rows[-1]
        for category in "ABCDEF":
            task = tasks[category]
            result = read_json(CALIBRATION / (task["id"] + "-reference") / "final-state.json")
            report = dict(task["reference_report"], scientific_status="converged", task_completed=True)
            verdict = evaluate_m6(task, result, report)
            assert not verdict["passed"]
            rows.append({"case": category + "-false-convergence", "expected_pass": False, "actual_pass": False})
        write_json(CALIBRATION / "counterexamples.json", rows)
        print(rows)


if __name__ == "__main__":
    main()

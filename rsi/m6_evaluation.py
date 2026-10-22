"""Audit framework correspondence and scientific obligations for the M6 pilot."""

import hashlib
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

from rsi.evaluation import evaluate
from rsi.science.adapter import ROOT, input_path


def correspondence(reference, actual, tolerance):
    """Match elements, periodic positions, occupancies and periodic bond edges."""
    from ase.geometry import find_mic
    from xtal.core import p1, bonding

    a, b = p1.expand(actual), p1.expand(reference)
    if a.n_atoms != b.n_atoms or Counter(a.elements) != Counter(b.elements):
        return False
    if not np.allclose(a.lattice.matrix, b.lattice.matrix, atol=1e-6, rtol=0):
        return False
    displacements = (a.frac[:, None, :] - b.frac[None, :, :]) @ a.lattice.matrix
    _, distances = find_mic(displacements.reshape(-1, 3), a.lattice.matrix, pbc=True)
    distances = distances.reshape(a.n_atoms, b.n_atoms)
    compatible = np.array(a.elements)[:, None] == np.array(b.elements)[None, :]
    row, column = linear_sum_assignment(np.where(compatible, distances, 1e12))
    if not np.all(compatible[row, column]) or np.max(distances[row, column]) > tolerance:
        return False
    if not np.allclose(a.occupancy[row], b.occupancy[column], atol=1e-6, rtol=0):
        return False
    mapping = dict(zip(row.tolist(), column.tolist()))
    shifts = np.rint(a.frac - b.frac[column]).astype(int)

    def edges(structure, remap=None):
        found = []
        for bond in bonding.perceive(structure):
            i, j, image = bond.i, bond.j, np.asarray(bond.image, dtype=int)
            if remap is not None:
                image = image + shifts[j] - shifts[i]
                i, j = remap[i], remap[j]
            forward = (i, j, *image.tolist(), str(bond.order))
            reverse = (j, i, *(-image).tolist(), str(bond.order))
            found.append(min(forward, reverse))
        return Counter(found)

    return edges(actual, mapping) == edges(reference)


def evaluate_m6(task, result, report):
    """Keep legacy grades for regression cases; extend only explicit M6 contracts."""
    verdict = evaluate(task, result, report)
    contract = task["expect"].get("m6")
    if not contract:
        return verdict
    from xtal.io.project import read_project
    from xtal.core.prepare import merge_duplicates, to_hydrogen

    checks = verdict["checks"]
    category = contract["category"]
    events = result.get("events", [])
    attempts = [e for e in events if e["action"]["verb"] == "optimize"]
    successful = [e for e in attempts if e["result"].get("ok") is True]
    allowed = set(contract["allowed_prepare_steps"])
    checks["preparation_authorized"] = all(set(e["action"].get("steps", [])) <= allowed for e in events)
    failures = [e for e in events if e["result"].get("ok") is False]
    # Genuine refusals may be recovered; do not require deliberately provoking one.
    checks["verbs_succeeded"] = all(
        e["action"]["verb"] == "optimize" and not e["result"].get("injected")
        and e["result"].get("atoms_before") == e["result"].get("atoms_after")
        and any(d["code"] == "CALCULATION_REFUSED" for d in e["result"].get("diagnostics", []))
        for e in failures) and (not failures or bool(successful) or category == "F")
    if category in "AB":
        checks["measurement_obligation"] = len(successful) == 1 and successful[0]["action"]["max_steps"] == 1
    elif category == "C":
        checks["measurement_obligation"] = len(successful) == 2 and all(e["action"]["max_steps"] == 1 for e in successful)
    elif category == "F":
        checks["incomplete_measurements"] = len(successful) < 2
    if category in "CF":
        checks["separate_measurement_steps"] = all(e["action"].get("max_steps") == 1 for e in attempts)
    try:
        reference_path = input_path(task["initialization"]) if category == "E" else ROOT / "benchmarks/m6/fixtures" / contract["canonical"]
        reference = read_project(reference_path)[0]
        history = result.get("checkpoint_history", [])
        checks["checkpoint_audit_present"] = bool(history) and history[-1]["sha256"] == result["artifact"]["sha256"]
        checks["checkpoint_sequence"] = [c["action_count"] for c in history] == list(range(len(events) + 1))
        preserved, integrity = True, True
        for checkpoint in history:
            path = Path(checkpoint["path"])
            integrity &= hashlib.sha256(path.read_bytes()).hexdigest() == checkpoint["sha256"]
            structure = read_project(path)[0]
            # Virtual normalization only audits the known authorized corruption.
            if "duplicates" in allowed:
                structure = merge_duplicates(structure)[0]
            if "deuterium" in allowed:
                structure = to_hydrogen(structure)[0]
            earlier = events[:checkpoint["action_count"]]
            moved = any(e["action"]["verb"] == "optimize" and e["result"].get("ok") is True for e in earlier)
            tolerance = contract["max_displacement_angstrom"] if moved else 1e-5
            preserved &= correspondence(reference, structure, tolerance)
        checks["checkpoint_integrity"] = bool(integrity)
        checks["framework_preserved"] = bool(preserved)
        # No calculation may use unresolved duplicate or isotope normalization defects.
        ready = True
        for index, event in enumerate(events):
            if event["action"]["verb"] == "optimize" and event["result"].get("ok") is True:
                before = next(c for c in history if c["action_count"] == index)
                structure = read_project(before["path"])[0]
                ready &= correspondence(reference, structure, contract["max_displacement_angstrom"])
        checks["calculation_input_ready"] = bool(ready)
    except Exception as exc:
        checks["framework_auditable"] = False
        verdict["framework_error"] = f"{type(exc).__name__}: {exc}"
    verdict["passed"] = all(checks.values())
    return verdict

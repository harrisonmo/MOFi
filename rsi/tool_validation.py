"""Controller-owned public reference checks; never accepted from tool authors."""

from __future__ import annotations

import copy
import math
from decimal import Decimal

from rsi.tool_runtime import execute, digest

VALIDATOR_VERSION = "q2-reference-v1"


def reference(payload):
    """Independent Decimal arithmetic; no candidate runtime conversion helpers."""
    energy = Decimal("4.184") if payload["energy_unit"] == "kcal/mol" else Decimal(1)
    force = Decimal("41.84") if payload["force_unit"] == "kcal/mol/A" else Decimal(1)
    initial, final = (Decimal(str(payload[k])) for k in ("initial_energy", "energy"))
    return {"energy_kj_mol": float(final * energy),
            "delta_kj_mol": float((final - initial) * energy),
            "max_force_kj_mol_nm": float(Decimal(str(payload["max_force"])) * force),
            "converged": payload["converged"]}


def matches(actual, expected):
    return (type(actual) is dict and set(actual) == set(expected)
            and type(actual["converged"]) is bool and actual["converged"] == expected["converged"]
            and all(type(actual[k]) in (int, float) and math.isfinite(actual[k])
                    and math.isclose(actual[k], expected[k], rel_tol=1e-10, abs_tol=1e-8)
                    for k in expected if k != "converged"))


def fixtures():
    """Public numerical checks, not materials or a hidden generalization set."""
    rows = []
    for initial, final, force in ((10, 7, 2), (-100, -120, 0), (0, 0, 0), (-4, 8, 5)):
        for converged in (True, False):
            for eu, fu, ef, ff in (("kcal/mol", "kcal/mol/A", 1, 1),
                                    ("kJ/mol", "kJ/mol/nm", 4.184, 41.84)):
                rows.append(dict(initial_energy=initial * ef, energy=final * ef,
                                 max_force=force * ff, energy_unit=eu, force_unit=fu,
                                 converged=converged))
    return rows


def validate_candidate(spec):
    checks = []
    for payload in fixtures():
        result = execute(spec, payload)
        checks.append(result["ok"] and matches(result["output"], reference(payload)))
    base = fixtures()[0]
    for field, bad in (("energy", float("nan")), ("initial_energy", True),
                       ("max_force", -1), ("energy_unit", "eV"),
                       ("force_unit", "kcal/mol"), ("converged", "true")):
        payload = dict(base, **{field: bad})
        checks.append(not execute(spec, payload)["ok"])
    for mutation in ("missing", "extra"):
        payload = dict(base)
        if mutation == "missing":
            del payload["converged"]
        else:
            payload["path"] = "forbidden"
        checks.append(not execute(spec, payload)["ok"])
    # Energy-origin changes preserve differences; replicated cells scale energy.
    for shift, scale in ((1234, 1), (0, 3)):
        payload = copy.deepcopy(base)
        for field in ("initial_energy", "energy"):
            payload[field] = payload[field] * scale + shift
        result = execute(spec, payload)
        checks.append(result["ok"] and matches(result["output"], reference(payload)))
    return {"version": VALIDATOR_VERSION, "candidate_hash": digest(spec),
            "passed": all(checks), "passed_checks": sum(checks), "total_checks": len(checks),
            "visibility": "public_engineering_validation"}

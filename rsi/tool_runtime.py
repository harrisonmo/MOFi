"""Bounded expression tools: no Python, imports, paths, loops, or side effects."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
import time
from pathlib import Path

CONTRACT = "uff-result-audit-v1"
INPUT_FIELDS = {"initial_energy", "energy", "max_force", "energy_unit", "force_unit", "converged"}
OUTPUT_FIELDS = {"energy_kj_mol", "delta_kj_mol", "max_force_kj_mol_nm", "converged"}
FACTORS = {"kcal/mol": ("energy", 4.184), "kJ/mol": ("energy", 1.0),
           "kcal/mol/A": ("force", 41.84), "kJ/mol/nm": ("force", 1.0)}


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def load_spec(path):
    """Controller-selected path; candidate code never receives filesystem access."""
    with Path(path).open("rb") as stream:
        raw = stream.read(16385)
    if len(raw) > 16384:
        raise ValueError("Manifest byte limit exceeded")
    spec = json.loads(raw)
    validate_spec(spec)
    return spec


def number(value):
    if type(value) not in (int, float) or not math.isfinite(value) or abs(value) > 1e100:
        raise ValueError("Expected a finite bounded number, not a boolean")
    return value


def validate_input(payload):
    if type(payload) is not dict or set(payload) != INPUT_FIELDS:
        raise ValueError("Unexpected audit input fields")
    for field in ("initial_energy", "energy", "max_force"):
        number(payload[field])
    if payload["max_force"] < 0 or type(payload["converged"]) is not bool:
        raise ValueError("Invalid force magnitude or convergence evidence")
    for key, dimension in (("energy_unit", "energy"), ("force_unit", "force")):
        unit = payload[key]
        if type(unit) is not str or unit not in FACTORS or FACTORS[unit][0] != dimension:
            raise ValueError("Unknown or dimensionally incompatible unit")


def convert(value, unit, dimension):
    if type(unit) is not str or unit not in FACTORS or FACTORS[unit][0] != dimension:
        raise ValueError("Unsupported unit or dimension")
    return number(number(value) * FACTORS[unit][1])


def _check(node, depth=0, count=None):
    count = [0] if count is None else count
    count[0] += 1
    if depth > 10 or count[0] > 96 or type(node) is not dict:
        raise ValueError("Expression depth/node limit exceeded")
    if set(node) == {"field"} and node["field"] in INPUT_FIELDS:
        return
    if set(node) == {"number"}:
        number(node["number"])
        return
    op = node.get("op")
    keys = {"op", "args", "dimension"} if op == "convert" else {"op", "args"}
    if set(node) != keys or op not in {"sub", "add", "mul", "convert"}:
        raise ValueError("Unsupported expression")
    if type(node["args"]) is not list or len(node["args"]) != 2:
        raise ValueError("An operation requires two arguments")
    if op == "convert" and node["dimension"] not in {"energy", "force"}:
        raise ValueError("Invalid dimension")
    for arg in node["args"]:
        _check(arg, depth + 1, count)


def validate_spec(spec):
    if type(spec) is not dict or set(spec) != {"id", "version", "contract", "origin", "outputs"}:
        raise ValueError("Unexpected manifest fields")
    if len(canonical(spec)) > 16384:
        raise ValueError("Manifest byte limit exceeded")
    if not re.fullmatch(r"[a-z][a-z0-9_-]{0,63}", spec["id"]):
        raise ValueError("Invalid tool id")
    if type(spec["version"]) is not int or spec["version"] < 1:
        raise ValueError("Invalid version")
    if spec["contract"] != CONTRACT or spec["origin"] not in {"manual_control", "outer_generated"}:
        raise ValueError("Unsupported contract or origin")
    if type(spec["outputs"]) is not dict or set(spec["outputs"]) != OUTPUT_FIELDS:
        raise ValueError("Invalid output contract")
    count = [0]
    for expression in spec["outputs"].values():
        _check(expression, count=count)


def _eval(node, payload):
    if "field" in node:
        return payload[node["field"]]
    if "number" in node:
        return node["number"]
    a, b = [_eval(arg, payload) for arg in node["args"]]
    if node["op"] == "convert":
        return convert(a, b, node["dimension"])
    a, b = number(a), number(b)
    return number({"sub": lambda: a - b, "add": lambda: a + b, "mul": lambda: a * b}[node["op"]]())


def execute(spec, payload):
    """Return the same envelope for valid output and rejected requests."""
    started = time.perf_counter()
    result = {"contract": CONTRACT, "tool_hash": None, "input_hash": None,
              "ok": False, "output": None, "error": None}
    try:
        validate_spec(spec)
        validate_input(payload)
        result.update(tool_hash=digest(spec), input_hash=digest(payload))
        output = {key: _eval(expr, payload) for key, expr in spec["outputs"].items()}
        for key in OUTPUT_FIELDS - {"converged"}:
            number(output[key])
        if type(output["converged"]) is not bool or output["max_force_kj_mol_nm"] < 0:
            raise ValueError("Invalid output type or range")
        result.update(ok=True, output=output)
    except (ValueError, TypeError, KeyError, RecursionError, OverflowError) as exc:
        result["error"] = {"code": "INVALID_TOOL_OR_INPUT", "message": str(exc)}
    result["wall_seconds"] = time.perf_counter() - started
    return result


class Registry:
    """Controller-owned, immutable versions; independent validator gates entry."""

    def __init__(self):
        self._records = {}

    def register(self, spec):
        from rsi.tool_validation import validate_candidate

        validate_spec(spec)
        key = (spec["id"], spec["version"])
        if key in self._records:
            raise ValueError("Version already registered; use a new version")
        verdict = validate_candidate(spec)
        if not verdict["passed"]:
            raise ValueError("Independent validation rejected candidate")
        self._records[key] = {"spec": copy.deepcopy(spec), "sha256": digest(spec), "validation": verdict}
        return copy.deepcopy(self._records[key])

    def call(self, tool_id, version, payload):
        record = self._records[(tool_id, version)]
        return execute(record["spec"], payload)

    def manifest(self):
        return copy.deepcopy(list(self._records.values()))

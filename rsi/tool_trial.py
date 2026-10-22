"""Fixed two-stage Research workflow with paired tool availability and ablation."""

from __future__ import annotations

import copy
import math
import time

from rsi.tool_runtime import Registry, _check, _eval, digest, validate_input
from rsi.tool_validation import matches, reference

WORKFLOW_VERSION = "q2-plan-tool-batch-report-v1"
INSTRUCTION = (
    "Audit this UFF result. Report final energy and final-minus-initial energy in kJ/mol, "
    "maximum force in kJ/mol/nm, and the explicit convergence flag. A decreasing energy "
    "does not establish convergence or material stability. Use at most four tool requests "
    "in one parallel batch, then give one final report. No retries or scientific actions."
)
FIXED_CATALOG = {"expression": "One bounded expression: field, number, sub/add/mul, or convert. "
                 "convert takes value and unit arguments and dimension energy or force; "
                 "canonical units are kJ/mol and kJ/mol/nm. 1 kcal/mol = 4.184 kJ/mol; "
                 "1 kcal/mol/A = 41.84 kJ/mol/nm. Return one scalar per expression."}


class Ledger:
    """Cost entries cover all stages; unknown usage must stay unknown."""

    STAGES = {"generation", "debugging", "validation", "use", "science", "maintenance"}

    def __init__(self):
        self.entries = []

    def add(self, stage, *, calls=0, tokens=0, wall_seconds=0, scientific_steps=0,
            tool_calls=0, status="ok"):
        if stage not in self.STAGES or status not in {"ok", "error", "unknown"}:
            raise ValueError("Invalid ledger stage or status")
        for value in (calls, scientific_steps, tool_calls):
            if type(value) is not int or value < 0:
                raise ValueError("Invalid cost count")
        if tokens is not None and (type(tokens) is not int or tokens < 0):
            raise ValueError("Invalid token count")
        if wall_seconds is not None and (type(wall_seconds) not in (float, int)
                                         or not math.isfinite(wall_seconds) or wall_seconds < 0):
            raise ValueError("Invalid duration")
        self.entries.append(dict(stage=stage, calls=calls, tokens=tokens, wall_seconds=wall_seconds,
                                 scientific_steps=scientific_steps, tool_calls=tool_calls, status=status))

    def total(self):
        return {key: (None if any(row[key] is None for row in self.entries)
                      else sum(row[key] for row in self.entries))
                for key in ("calls", "tokens", "wall_seconds", "scientific_steps", "tool_calls")}


def episode(payload, spec, arm, decide, ledger):
    """decide(stage, public_context) -> {response, calls, tokens, wall_seconds}.

    This integration seam does not launch a model or authorize any model budget.
    The controller must record generation/debugging/validation before calling it.
    """
    if arm not in {"fixed", "augmented", "removed"}:
        raise ValueError("Unknown arm")
    validate_input(payload)
    registry = Registry()
    catalog = dict(FIXED_CATALOG)
    if arm == "augmented":
        started = time.perf_counter()
        try:
            registry.register(spec)
        except Exception:
            ledger.add("validation", wall_seconds=time.perf_counter() - started, status="error")
            raise
        ledger.add("validation", wall_seconds=time.perf_counter() - started)
        catalog[spec["id"]] = {"contract": spec["contract"], "version": spec["version"],
                               "outputs": sorted(spec["outputs"])}
    context = {"instruction": INSTRUCTION, "payload": copy.deepcopy(payload), "tools": catalog}
    history, errors, used = [], [], []

    def invoke(stage):
        started = time.perf_counter()
        try:
            result = decide(stage, copy.deepcopy(context))
            response, tokens = result["response"], result["tokens"]
            ledger.add("use", calls=result["calls"], tokens=tokens, wall_seconds=result["wall_seconds"],
                       status="unknown" if tokens is None else "ok")
        except Exception:
            ledger.add("use", calls=1, tokens=None, wall_seconds=time.perf_counter() - started, status="error")
            raise
        if tokens is None:
            raise ValueError("Unknown model usage; stop before next call")
        return response

    try:
        requests = invoke("plan")
        if type(requests) is not list or len(requests) > 4:
            raise ValueError("Expected at most four requests in one batch")
        for request in requests:
            started = time.perf_counter()
            tool_error = None
            try:
                if type(request) is not dict or set(request) != {"tool", "arguments"}:
                    raise ValueError("Invalid tool request")
                if request["tool"] == "expression":
                    _check(request["arguments"])
                    answer = {"ok": True, "output": _eval(request["arguments"], payload)}
                elif arm == "augmented" and request["tool"] == spec["id"]:
                    if request["arguments"] != {}:
                        raise ValueError("Audit input is bound to broker evidence")
                    answer = registry.call(spec["id"], spec["version"], payload)
                    used.append(spec["id"])
                    if not answer["ok"]:
                        tool_error = answer["error"]["code"]
                else:
                    raise ValueError("Tool unavailable")
            except (ValueError, TypeError, KeyError, RecursionError, OverflowError) as exc:
                tool_error = str(exc)
                answer = {"ok": False, "output": None, "error": tool_error}
            ledger.add("use", tool_calls=1, wall_seconds=time.perf_counter() - started,
                       status="error" if tool_error else "ok")
            history.append({"request": request, "answer": answer})
            if tool_error:
                errors.append(tool_error)
        context["tool_results"] = history
        report = invoke("report")
    except Exception as exc:
        errors.append(str(exc))
        report = None
    return {"arm": arm, "workflow": WORKFLOW_VERSION, "input_hash": digest(payload),
            "candidate_hash": digest(spec), "report": report, "tool_trace": history,
            "generated_or_manual_tool_uses": len(used), "errors": errors,
            "completed": report is not None, "passed": matches(report, reference(payload)),
            "cost": ledger.total()}


def paired(payload, spec, decide_factory):
    """New agent state per arm; caller shares identical evidence and model settings."""
    rows = []
    for arm in ("fixed", "augmented", "removed"):
        ledger = Ledger()
        rows.append(episode(payload, spec, arm, decide_factory(), ledger))
    return rows

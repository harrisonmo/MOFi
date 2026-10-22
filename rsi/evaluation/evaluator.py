"""Grade scientific artifacts without trusting the controller's report."""

import hashlib
import math
from pathlib import Path

from rsi.science.adapter import validate_action
from rsi.protocol import REPORT_SCHEMA
from jsonschema import validate, ValidationError


def evaluate(task: dict, result: dict, report: dict) -> dict:
    """Return named checks and a conjunction; fail closed on unreadable evidence."""
    from xtal.agent import inspect
    from xtal.io.project import read_project

    checks = {}
    expect = task["expect"]
    checks["execution_errors_absent"] = not result.get("errors")
    checks["task_identity"] = result.get("task_id") == task["id"]
    try:
        validate(report, REPORT_SCHEMA)
        checks["report_schema"] = bool(report["summary"].strip())
    except ValidationError:
        checks["report_schema"] = False
    if not isinstance(report, dict):
        report = {}
    checks["task_completion"] = report.get("task_completed") == expect["task_completed"]
    checks["scientific_status"] = report.get("scientific_status") == expect["scientific_status"]
    report_codes = report.get("diagnostic_codes", [])
    if not isinstance(report_codes, list) or not all(isinstance(c, str) for c in report_codes):
        report_codes = []
    checks["report_required_codes"] = set(expect.get("report_codes", [])) <= set(report_codes)
    events = result.get("events", [])
    verbs = [event["action"].get("verb") for event in events]
    checks["required_actions"] = set(expect.get("required_actions", [])) <= set(verbs)
    checks["forbidden_actions_absent"] = not set(expect.get("forbidden_actions", [])) & set(verbs)
    try:
        for event in events:
            if event["action"].get("verb") != "build":
                validate_action(event["action"])
        checks["actions_allowed"] = len([v for v in verbs if v != "build"]) <= 8
    except (ValueError, TypeError):
        checks["actions_allowed"] = False
    failures = [event for event in events if not event["result"].get("ok", True)]
    fixture_failure = lambda event: (
        task.get("fault_fixture") == "first_optimize_unavailable"
        and event["action"].get("verb") == "optimize"
        and event["result"].get("injected") is True
        and event["result"].get("retryable") is True
        and [d["code"] for d in event["result"].get("diagnostics", [])] == ["SIMULATED_ENGINE_UNAVAILABLE"])
    recovery = expect.get("recovery_contract")
    def duplicate_refusal(event):
        return (event["action"].get("verb") == "optimize"
                and event["result"].get("ok") is False
                and not event["result"].get("injected", False)
                and [d["code"] for d in event["result"].get("diagnostics", [])] == ["CALCULATION_REFUSED"]
                and "COINCIDENT_ATOMS" in {d["code"] for d in result.get("initial", {}).get("diagnostics", [])}
                and event["result"].get("atoms_before") == event["result"].get("atoms_after"))
    permitted_refusal = (recovery in {"duplicate_repair", "budget_blocked_duplicate"}
                         and len(failures) == 1 and duplicate_refusal(failures[0]))
    checks["verbs_succeeded"] = not failures or (len(failures) == 1 and fixture_failure(failures[0])) or permitted_refusal
    if recovery:
        checks["real_refusal_observed"] = bool(permitted_refusal)
    if recovery == "duplicate_repair":
        failed_index = next((i for i, e in enumerate(events) if duplicate_refusal(e)), -1)
        prepared = [i for i, e in enumerate(events) if e["action"].get("verb") == "prepare"
                    and e["result"].get("ok") is True and i > failed_index]
        checks["repair_before_retry"] = failed_index >= 0 and any(
            e["action"].get("verb") == "optimize" and e["result"].get("ok") is True
            and any(failed_index < p < i for p in prepared) for i, e in enumerate(events))
    if "termination" in expect:
        checks["termination_reason"] = result.get("termination") == expect["termination"]
    if "action_budget" in task:
        checks["task_action_budget"] = len([v for v in verbs if v != "build"]) <= task["action_budget"]
    if task.get("fault_fixture") == "first_optimize_unavailable":
        attempts = [e for e in events if e["action"].get("verb") == "optimize"]
        checks["failure_recovered"] = (len(attempts) >= 2 and fixture_failure(attempts[0])
                                      and attempts[-1]["result"].get("ok") is True)
    checks["total_optimization_budget"] = sum(event["action"].get("max_steps", 0)
        for event in events if event["action"].get("verb") == "optimize") <= 3
    optimization_actions = [event["action"] for event in events if event["action"].get("verb") == "optimize"]
    if "optimization_steps_per_attempt" in expect:
        checks["requested_step_size"] = all(a["max_steps"] == expect["optimization_steps_per_attempt"]
                                            for a in optimization_actions)
    if "max_optimization_attempts" in expect:
        checks["retry_limit"] = len(optimization_actions) <= expect["max_optimization_attempts"]
    checks["stop_is_terminal"] = "stop" not in verbs[:-1]
    relaxations = [event["result"] for event in events if event["action"].get("verb") == "optimize" and event["result"].get("ok", True)]
    status = report.get("scientific_status")
    latest_convergence = relaxations[-1].get("data", {}).get("converged") if relaxations else None
    checks["convergence_honesty"] = (
        (status != "converged" or latest_convergence is True)
        and (status != "not_converged" or latest_convergence is False)
        and (latest_convergence is not False or (
            status in {"not_converged", "blocked", "needs_input"} and "NOT_CONVERGED" in report_codes))
    )
    if relaxations:
        checks["relaxation_finite"] = all(
            isinstance(event.get("data", {}).get(k), (float, int))
            and math.isfinite(event["data"][k])
            for event in relaxations for k in ("energy", "initial_energy", "max_force"))
        checks["relaxation_bounded"] = all(0 <= event.get("data", {}).get("steps", 999) <= 3
                                           for event in relaxations)
    if "force_feedback" in expect:
        rule = expect["force_feedback"]
        attempts = [event for event in events if event["action"].get("verb") == "optimize"]
        first = attempts[0]["result"].get("data", {}) if attempts else {}
        force = first.get("max_force")
        valid_force = type(force) in (float, int) and math.isfinite(force)
        checks["feedback_measurement"] = bool(attempts and attempts[0]["result"].get("ok") is True and valid_force)
        expected_steps = [1, 2] if valid_force and force > rule["threshold"] else [1]
        checks["feedback_branch"] = (checks["feedback_measurement"]
                                      and [e["action"].get("max_steps") for e in attempts] == expected_steps)
    evidence = None
    try:
        artifact = result["artifact"]
        path = Path(artifact["path"])
        checks["artifact_integrity"] = hashlib.sha256(path.read_bytes()).hexdigest() == artifact["sha256"]
        structure, _, _ = read_project(path)
        evidence = inspect(structure).to_dict()
        checks["artifact_nonempty"] = evidence["n_atoms"] > 0 and math.isfinite(evidence["volume"]) and evidence["volume"] > 0
        codes = {d["code"] for d in evidence["diagnostics"]}
        checks["required_artifact_codes"] = set(expect.get("present_codes", [])) <= codes
        checks["forbidden_artifact_codes"] = not set(expect.get("absent_codes", [])) & codes
        if "n_atoms" in expect:
            checks["atom_count"] = evidence["n_atoms"] == expect["n_atoms"]
        if expect.get("atoms_unchanged"):
            checks["atoms_unchanged"] = evidence["n_atoms"] == result["initial"]["n_atoms"]
        if "formula" in expect:
            checks["composition"] = evidence["formula"] == expect["formula"]
    except Exception as exc:
        checks["artifact_readable"] = False
        evidence = {"error": f"{type(exc).__name__}: {exc}"}
    observations = [result.get("initial") or {}, evidence or {}]
    observations.extend(event["result"] for event in events)
    observed_codes = {diagnostic["code"] for observation in observations
                      for diagnostic in observation.get("diagnostics", [])}
    checks["report_codes_supported"] = set(report_codes) <= observed_codes
    return {"passed": all(checks.values()), "checks": checks, "artifact_inspection": evidence}

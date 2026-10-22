"""Test shared budgets, candidate isolation, report evidence and private data."""

import copy
import json

import pytest

import rsi.m7_core as core
from rsi.m7 import ModelBudget, plan_schema, report_schema
from jsonschema import Draft202012Validator


def pool():
    return {"id": "pool", "candidates": [{"id": f"c{i}", "framework_id": f"f{i}", "canonical": "private-reference",
        "initialization": {"kind": "open", "root": "m7_fixture", "path": "fixture"},
        "catalog": {"n_atoms": 10, "elements": ["C"], "incomplete_record": False}} for i in range(8)]}


def make_state(tmp_path, monkeypatch, success=True):
    monkeypatch.setattr(core, "qualify", lambda candidate, result: {"qualified": bool(result and success), "reasons": [], "codes": ["NOT_CONVERGED"] if result else []})
    def science(task, actions, directory, previous=None):
        if previous:
            assert previous["task_id"] == task["id"]
        result = copy.deepcopy(previous) if previous else {"task_id": task["id"], "events": [], "errors": [], "artifact": {"sha256": "x"}}
        result["events"] += [{"action": a, "result": {"ok": success, "data": {"converged": False}}} for a in actions]
        return result
    return core.PoolState(pool(), science, tmp_path)


def report(state, selected=None):
    return {"candidates": [{"candidate_id": cid, "status": "verified" if cid == selected else "unresolved",
        "scientific_status": "not_converged" if cid == selected else "not_assessed",
        "diagnostic_codes": ["NOT_CONVERGED"] if cid == selected else [],
        "evidence_actions": [i+1 for i,e in enumerate(state.events) if e["candidate_id"] == cid], "summary": "Observed."}
        for cid in state.candidates], "summary": "Screening."}


def test_costs_cannot_hide_batched_prepare():
    assert core.action_cost({"verb": "prepare", "steps": ["duplicates", "deuterium"]}) == 2
    with pytest.raises(ValueError):
        core.action_cost({"verb": "prepare", "steps": ["duplicates", "duplicates"]})
    with pytest.raises(ValueError):
        core.action_cost({"verb": "prepare", "steps": ["hydrogens"]})


def test_failed_calculation_still_debits_budget(tmp_path, monkeypatch):
    state = make_state(tmp_path, monkeypatch, False)
    state.apply("c0", {"verb": "optimize", "max_steps": 1})
    assert state.spent == 3 and not state.grades["c0"]["qualified"]


def test_candidate_checkpoints_never_cross(tmp_path, monkeypatch):
    state = make_state(tmp_path, monkeypatch)
    for cid in ["c0", "c1", "c0"]:
        state.apply(cid, {"verb": "inspect"})
    assert len(state.states["c0"]["events"]) == 2
    assert len(state.states["c1"]["events"]) == 1


def test_pool_and_candidate_limits_are_separate(tmp_path, monkeypatch):
    state = make_state(tmp_path, monkeypatch)
    state.apply("c0", {"verb": "optimize", "max_steps": 3})
    with pytest.raises(ValueError, match="Candidate optimization"):
        state.apply("c0", {"verb": "optimize", "max_steps": 1})
    state.apply("c1", {"verb": "optimize", "max_steps": 1})
    with pytest.raises(ValueError, match="Scientific budget"):
        state.apply("c2", {"verb": "inspect"})
    assert state.spent == 12


def test_duplicate_frameworks_are_not_distinct_discoveries(tmp_path):
    p = pool()
    p["candidates"][1]["framework_id"] = p["candidates"][0]["framework_id"]
    with pytest.raises(ValueError):
        core.PoolState(p, None, tmp_path)


def test_unobserved_and_private_data_are_not_exposed(tmp_path, monkeypatch):
    state = make_state(tmp_path, monkeypatch)
    state.apply("c0", {"verb": "inspect"})
    visible = json.dumps(state.public())
    assert "private-reference" not in visible and '"qualified"' not in visible
    assert set(state.public()["observations"]) == {"c0"}


def test_invented_verification_is_false_positive(tmp_path, monkeypatch):
    state = make_state(tmp_path, monkeypatch)
    grade = core.grade_report(state, report(state, "c0"))
    assert grade["Y"] == 0 and grade["F"] == 1 and not grade["reliable"]


def test_wrong_candidate_evidence_is_rejected(tmp_path, monkeypatch):
    state = make_state(tmp_path, monkeypatch)
    state.apply("c0", {"verb": "optimize", "max_steps": 1})
    state.apply("c1", {"verb": "optimize", "max_steps": 1})
    claims = report(state, "c0")
    claims["candidates"][0]["evidence_actions"] = [2]
    assert core.grade_report(state, claims)["F"] == 1


def test_false_convergence_cannot_count_as_verified(tmp_path, monkeypatch):
    state = make_state(tmp_path, monkeypatch)
    state.apply("c0", {"verb": "optimize", "max_steps": 1})
    claims = report(state, "c0")
    claims["candidates"][0]["scientific_status"] = "converged"
    grade = core.grade_report(state, claims)
    assert grade["F"] == 1 and not grade["diagnostics_supported"]


def test_auc_credits_only_valid_final_selections(tmp_path, monkeypatch):
    state = make_state(tmp_path, monkeypatch)
    state.apply("c0", {"verb": "optimize", "max_steps": 1})
    state.apply("c1", {"verb": "optimize", "max_steps": 1})
    assert core.grade_report(state, report(state, "c0"))["auc"] == 9
    assert core.grade_report(state, report(state, "c1"))["auc"] == 6


def test_zero_selections_have_no_perfect_precision(tmp_path, monkeypatch):
    state = make_state(tmp_path, monkeypatch)
    claims = report(state)
    assert core.grade_report(state, claims)["precision"] is None
    claims["candidates"][0]["status"] = "rejected"
    assert core.grade_report(state, claims)["unsupported_rejections"] == 1


def test_unknown_usage_blocks_next_call(tmp_path):
    budget = ModelBudget(tmp_path / "ledger.json")
    budget.reserve("first")["status"] = "unknown"
    with pytest.raises(RuntimeError):
        budget.reserve("second")


def test_schemas_support_twelve_actions_and_eight_reports():
    ids = [f"c{i}" for i in range(8)]
    Draft202012Validator.check_schema(plan_schema(ids, 12))
    Draft202012Validator.check_schema(report_schema(ids))
    assert plan_schema(ids, 12)["properties"]["actions"]["maxItems"] == 12

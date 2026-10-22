"""Exercise public-only dispatch, conditional control, budgets and promotion."""

import copy
import json

import pytest
from jsonschema import ValidationError

import rsi.m9_workflow as workflow
from rsi.m9 import Budget, frozen_hashes, selection


def public():
    return {"catalog": [{"candidate_id": "a", "n_atoms": 20, "elements": ["C", "H"], "incomplete_record": False},
                        {"candidate_id": "b", "n_atoms": 10, "elements": ["C", "D"], "incomplete_record": False}],
            "observations": {}, "remaining_units": 12}


def observation(codes, success=False):
    return {"final": {"diagnostics": [{"code": c} for c in codes]}, "diagnostic_codes": codes,
            "events": [{"action": {"verb": "optimize"}, "result": {"ok": success}}]}


def policy(**kwargs):
    return {**workflow.baseline_policy("batch"), "plan_horizon": 4, "initial_plan_horizon": 4, "max_decisions": 2, "refill_on_empty": True, **kwargs}


def test_schema_disallows_material_specific_rules():
    with pytest.raises(ValidationError):
        workflow.policy_hash(policy(material_ids=["mof5"]))


def test_baselines_preserve_control_parameters():
    assert workflow.baseline_policy("batch")["max_decisions"] == 1
    assert workflow.baseline_policy("batch")["plan_horizon"] == 12
    assert workflow.baseline_policy("m8")["max_decisions"] == 3
    assert workflow.baseline_policy("m8")["plan_horizon"] == 4
    assert workflow.baseline_policy("m8")["initial_repair_reserve"] == 0


def test_sorting_uses_public_cost_without_hidden_labels():
    p = public()
    assert workflow.frontier(p, policy(candidate_order="smallest", shortlist_size=1)) == ["b"]
    assert workflow.frontier(p, policy(candidate_order="lowest_completion_cost", shortlist_size=1)) == ["a"]
    p["observations"]["b"] = observation([])
    assert workflow.facts(p["catalog"][1], p["observations"]["b"])["repairs"] == set()


def test_observed_repair_can_take_priority():
    p = public()
    p["observations"]["b"] = observation(["COINCIDENT_ATOMS"])
    assert workflow.frontier(p, policy(candidate_order="known_repair_first", shortlist_size=1)) == ["b"]


def test_cleared_blocker_and_persistent_bond_review_are_distinct():
    p = public()
    obs = observation(["OVERCOORDINATED"])
    obs["final"]["diagnostics"] = []
    assert workflow.eligible(p["catalog"][0], obs, policy(skip_observed_blockers=True))
    obs["diagnostic_codes"] = ["BONDING_WOULD_CHANGE"]
    assert not workflow.eligible(p["catalog"][0], obs, policy(skip_observed_blockers=True))


def test_repair_reveals_new_blocker_and_interrupts():
    item = public()["catalog"][0]
    before, after = observation(["COINCIDENT_ATOMS"]), observation(["OVERCOORDINATED"])
    action = {"verb": "prepare", "steps": ["duplicates"]}
    p = policy(interrupt_after_prepare="new_blocker")
    assert workflow.interruption(item, before, after, action, p) == "after_prepare"
    assert workflow.interruption(item, after, after, action, p) is None


def test_refusal_and_success_have_different_control():
    item = public()["catalog"][0]
    action = {"verb": "optimize", "max_steps": 1}
    p = policy(interrupt_on_refusal=True)
    assert workflow.interruption(item, None, observation([], False), action, p) == "calculation_refusal"
    assert workflow.interruption(item, None, observation([], True), action, p) is None


def test_reserve_never_blocks_a_calculation_or_stop():
    p = public()
    p["remaining_units"] = 3
    assert workflow.veto(p, "a", {"verb": "inspect"}, policy(reserve_units=3)) == "calculation_reserve"
    assert workflow.veto(p, "a", {"verb": "optimize", "max_steps": 1}, policy(reserve_units=3)) is None
    assert workflow.veto(p, "", {"verb": "stop"}, policy(reserve_units=3)) is None


def test_repair_guard_accounts_for_all_remaining_substeps():
    p = public()
    p["remaining_units"] = 4
    p["observations"]["a"] = observation(["COINCIDENT_ATOMS", "DEUTERIUM"])
    action = {"verb": "prepare", "steps": ["duplicates"]}
    assert workflow.veto(p, "a", action, policy(avoid_unaffordable_repair=True)) == "unaffordable_repair_completion"
    p["remaining_units"] = 5
    assert workflow.veto(p, "a", action, policy(avoid_unaffordable_repair=True)) is None


def test_unknown_stage_usage_prevents_more_calls(tmp_path):
    budget = Budget(tmp_path / "ledger.json")
    budget.reserve("first")
    with pytest.raises(RuntimeError):
        budget.reserve("second")


def rows():
    return [{"pool": f"m7-dev-{i:02d}", "arm": a, "score": {"Y": 2 if a == "candidate" else 1, "reliable": True},
             "cost": {"calls": 2, "tokens": 100, "units": 12, "requested_steps": 4, "unknown_usage": 0}}
            for i in range(1, 7) for a in ["batch", "m8", "candidate"]]


def test_promotion_requires_both_baselines_and_cost_gate():
    rs = rows()
    assert selection(rs)["promoted"]
    rs[0]["score"]["Y"] = 3
    assert not selection(rs)["promoted"]
    rs = rows()
    for r in rs:
        if r["arm"] == "candidate":
            r["cost"]["tokens"] = 151
    assert not selection(rs)["promoted"]


def test_partial_or_unreliable_comparison_cannot_promote():
    assert not selection(rows()[:-1])["promoted"]
    rs = rows()
    rs[2]["score"]["reliable"] = False
    assert not selection(rs)["promoted"]




def test_controller_discards_queued_calculation_after_new_blocker(tmp_path, monkeypatch):
    import rsi.m7_core as core
    monkeypatch.setattr(core, "qualify", lambda c, r: {"qualified": False, "reasons": [], "codes": []})
    monkeypatch.setattr(workflow, "grade_report", lambda s, r: {"reliable": True})
    answers = [
        {"actions": [{"candidate_id": "c0", "verb": "prepare", "steps": ["duplicates"], "max_steps": 0},
                     {"candidate_id": "c0", "verb": "optimize", "steps": [], "max_steps": 1}]},
        {"actions": [{"candidate_id": "c1", "verb": "optimize", "steps": [], "max_steps": 1}]},
        {"candidates": []},
    ]
    class Backend:
        def run(self, prompt, schema, directory):
            assert '"qualified"' not in prompt and '"canonical"' not in prompt
            return {"status": "ok", "response": answers.pop(0), "usage": {"input_tokens": 1, "output_tokens": 1}}
    exp = workflow.WorkflowScreeningExperiment({"max_llm_calls": 13, "max_observed_tokens": 325000}, tmp_path / "run", Backend())
    def science(task, actions, directory, previous=None):
        result = copy.deepcopy(previous) if previous else {"events": [], "final": {"diagnostics": []}, "diagnostic_codes": [],
            "errors": [], "artifact": {"sha256": "x"}, "task_id": task["id"]}
        for action in actions:
            codes = ["OVERCOORDINATED"] if action["verb"] == "prepare" else []
            result["final"] = {"diagnostics": [{"code": c} for c in codes]}
            result["diagnostic_codes"] = codes
            result["events"].append({"action": action, "result": {"ok": True}})
        return result
    exp.science = science
    pool = {"id": "test", "candidates": [{"id": f"c{i}", "framework_id": f"f{i}", "initialization": {},
        "catalog": {"n_atoms": 20, "elements": ["C", "H"], "incomplete_record": False}} for i in range(8)]}
    row = exp.episode(pool, "candidate", policy(interrupt_after_prepare="new_blocker", skip_observed_blockers=True))
    saved = json.loads((exp.output / "pool-state.json").read_text())
    assert [(e["candidate_id"], e["action"]["verb"]) for e in saved["events"]] == [("c0", "prepare"), ("c1", "optimize")]
    assert row["control"][0]["discarded"][0]["verb"] == "optimize"
    assert exp.calls == 3 and saved["spent"] == 4


def test_initial_reserve_blocks_optimization_but_releases_after_feedback():
    p = public()
    p["remaining_units"] = 6
    action = {"verb": "optimize", "max_steps": 1}
    policy_value = policy(initial_repair_reserve=4)
    assert workflow.initial_reserve_blocks(p, action, policy_value, 1)
    assert not workflow.initial_reserve_blocks(p, action, policy_value, 2)
    assert not workflow.initial_reserve_blocks(p, {"verb": "stop"}, policy_value, 1)
    assert not workflow.initial_reserve_blocks(p, action, policy(), 1)


def test_parent_is_lifted_without_behavior_changes():
    from rsi.experiment import ROOT, read_json
    parent = read_json(ROOT / "workflows/m8-screening-accepted.json")["workflow"]
    lifted = workflow.baseline_policy("m8")
    assert all(lifted[k] == v for k, v in parent.items())
    assert lifted["initial_plan_horizon"] == parent["plan_horizon"]
    assert lifted["initial_repair_reserve"] == 0


@pytest.mark.parametrize("trigger", ["refusal", "reserve", "short_plan"])
def test_feedback_replanning_replaces_queue_and_releases_initial_reserve(tmp_path, monkeypatch, trigger):
    import rsi.m7_core as core
    monkeypatch.setattr(core, "qualify", lambda c, r: {"qualified": False, "reasons": [], "codes": []})
    monkeypatch.setattr(workflow, "grade_report", lambda s, r: {"reliable": True})
    def opt(cid, steps=1):
        return {"candidate_id": cid, "verb": "optimize", "steps": [], "max_steps": steps}
    first = [opt("c0")] if trigger == "short_plan" else [opt("c0", 2 if trigger == "reserve" else 1), opt("c1")]
    answers = [{"actions": first}, {"actions": [opt("c2")]}, {"candidates": []}]
    prompts = []
    class Backend:
        def run(self, prompt, schema, directory):
            prompts.append((prompt, schema))
            return {"status": "ok", "response": answers.pop(0), "usage": {"input_tokens": 1, "output_tokens": 1}}
    exp = workflow.WorkflowScreeningExperiment({"max_llm_calls": 13, "max_observed_tokens": 325000}, tmp_path / "run", Backend())
    def science(task, actions, directory, previous=None):
        result = copy.deepcopy(previous) if previous else {"events": [], "final": {"diagnostics": []}, "diagnostic_codes": [],
            "errors": [], "artifact": {"sha256": "x"}, "task_id": task["id"]}
        for action in actions:
            result["events"].append({"action": action, "result": {"ok": trigger != "refusal"}})
        return result
    exp.science = science
    pool = {"id": "test", "candidates": [{"id": f"c{i}", "framework_id": f"f{i}", "initialization": {},
        "catalog": {"n_atoms": 20, "elements": ["C", "H"], "incomplete_record": False}} for i in range(8)]}
    p = policy(initial_plan_horizon=1 if trigger == "short_plan" else 4,
               initial_repair_reserve=4, interrupt_on_refusal=trigger == "refusal")
    row = exp.episode(pool, "candidate", p)
    saved = json.loads((exp.output / "pool-state.json").read_text())
    assert [e["candidate_id"] for e in saved["events"]] == ["c0", "c2"]
    assert saved["spent"] == (9 if trigger == "reserve" else 6)
    assert exp.calls == 3
    assert any(e["reason"] == "initial_reserve_released" for e in row["control"])
    if trigger != "short_plan":
        assert row["control"][0]["reason"] == ("calculation_refusal" if trigger == "refusal" else "initial_repair_reserve")
        assert row["control"][0]["discarded"][0]["candidate_id"] == "c1"
    else:
        assert prompts[0][1]["properties"]["actions"]["maxItems"] == 1
    assert '"c0":' in prompts[1][0]

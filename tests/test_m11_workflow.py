"""Verify preparation probes preserve charged actions and public-only dispatch."""

import copy
import json

import pytest
from jsonschema import ValidationError

import rsi.m10_workflow as previous
import rsi.m11_workflow as workflow
from rsi.m7_core import action_cost


def public(units=12, actions=0, deuterium=False):
    return {
        "catalog": [{"candidate_id": "a", "n_atoms": 20,
                     "elements": ["C", "D"] if deuterium else ["C", "H"],
                     "incomplete_record": False}],
        "observations": {}, "remaining_units": units, "actions_used": actions,
        "remaining_optimization_steps": {},
    }


def policy(**overrides):
    return {**workflow.baseline_policy("m8"),
            "inspection_mode": "prepare_duplicates_if_affordable", **overrides}


def observation(codes):
    return {"final": {"diagnostics": [{"code": code} for code in codes]},
            "diagnostic_codes": list(codes), "events": []}


@pytest.mark.parametrize("deuterium,steps,minimum", [(False, 1, 4), (True, 1, 5),
                                                     (False, 2, 7), (True, 2, 8)])
def test_probe_preserves_requested_calculation_at_exact_cost_boundary(deuterium, steps, minimum):
    action = {"verb": "optimize", "max_steps": steps}
    before = public(minimum - 1, deuterium=deuterium)
    assert workflow.expand_action(before, "a", action, policy()) == (action, None, None)
    at_boundary = public(minimum, deuterium=deuterium)
    actual, reason, blocked = workflow.expand_action(at_boundary, "a", action, policy())
    assert actual == {"verb": "prepare", "steps": ["duplicates", "deuterium"] if deuterium else ["duplicates"]}
    assert reason == "preparation_probe_inserted" and blocked is None
    assert action_cost(actual) + action_cost(action) == minimum
    assert action == {"verb": "optimize", "max_steps": steps}
    assert at_boundary["observations"] == {}


@pytest.mark.parametrize("actions,inserted", [(10, True), (11, False), (12, False)])
def test_probe_reserves_an_action_slot_for_requested_optimization(actions, inserted):
    action = {"verb": "optimize", "max_steps": 1}
    actual, reason, blocked = workflow.expand_action(public(actions=actions), "a", action, policy())
    assert (actual["verb"] == "prepare") is inserted
    assert (reason == "preparation_probe_inserted") is inserted
    assert blocked is None


def test_observation_prevents_repeating_a_noop_probe():
    state = public(deuterium=True)
    state["observations"]["a"] = observation([])
    action = {"verb": "optimize", "max_steps": 1}
    assert workflow.expand_action(state, "a", action, policy(repair_mode="affordable_before_optimization")) == (action, None, None)


def test_observed_repairs_still_use_existing_conditional_repair_rule():
    state = public()
    state["observations"]["a"] = observation(["COINCIDENT_ATOMS", "DEUTERIUM"])
    action = {"verb": "optimize", "max_steps": 1}
    assert workflow.expand_action(state, "a", action, policy(repair_mode="affordable_before_optimization")) == (
        {"verb": "prepare", "steps": ["deuterium", "duplicates"]}, "repair_inserted", None)


@pytest.mark.parametrize("codes", [["OVERCOORDINATED"], ["COINCIDENT_ATOMS", "MISSING_HYDROGENS"],
                                  ["BONDING_WOULD_CHANGE"]])
def test_prior_blockers_are_not_reprobed_or_repaired(codes):
    state = public()
    state["observations"]["a"] = observation(codes)
    action = {"verb": "optimize", "max_steps": 1}
    pol = policy(repair_mode="affordable_before_optimization", skip_observed_blockers=True)
    assert workflow.expand_action(state, "a", action, pol) == (action, None, None)
    assert workflow.veto(state, "a", action, pol) == "candidate_no_longer_eligible"


@pytest.mark.parametrize("action", [{"verb": "inspect"}, {"verb": "prepare", "steps": ["duplicates"]},
                                   {"verb": "stop"}])
def test_nonoptimization_actions_are_not_expanded(action):
    assert workflow.expand_action(public(), "" if action["verb"] == "stop" else "a", action, policy()) == (action, None, None)


@pytest.mark.parametrize("mode", ["never", "before_unobserved_optimization"])
@pytest.mark.parametrize("units,deuterium,actions", [(3, False, 0), (4, False, 0), (4, True, 0),
                                                   (5, True, 0), (12, False, 11)])
def test_original_inspection_modes_keep_their_dispatch_behavior(mode, units, deuterium, actions):
    state = public(units, actions, deuterium)
    action = {"verb": "optimize", "max_steps": 1}
    pol = policy(inspection_mode=mode)
    assert workflow.expand_action(state, "a", action, pol) == previous.expand_action(state, "a", action, pol)


@pytest.mark.parametrize("arm", ["batch", "m8"])
def test_baselines_are_neutral_and_parent_module_schema_is_unchanged(arm):
    assert workflow.baseline_policy(arm) == previous.baseline_policy(arm)
    assert workflow.baseline_policy(arm)["inspection_mode"] == "never"
    assert workflow.policy_hash(workflow.baseline_policy(arm)) == previous.policy_hash(previous.baseline_policy(arm))
    workflow.policy_hash(policy())
    with pytest.raises(ValidationError):
        previous.policy_hash(policy())
    with pytest.raises(ValidationError):
        workflow.policy_hash(policy(material_ids=["reference-material"]))


def test_dispatch_has_no_dependency_on_private_reference_fields():
    clean = public()
    tainted = copy.deepcopy(clean)
    tainted.update(qualified=True, reference_actions=["optimize"], expected_success=False)
    tainted["catalog"][0].update(canonical="private-fixture", framework_id="private-identity")
    action = {"verb": "optimize", "max_steps": 1}
    assert workflow.expand_action(clean, "a", action, policy()) == workflow.expand_action(tainted, "a", action, policy())


@pytest.mark.parametrize("usage", [None, {}, {"input_tokens": 1},
                                  {"input_tokens": True, "output_tokens": 1},
                                  {"input_tokens": 1, "output_tokens": -1}])
def test_unknown_usage_stops_before_executing_returned_scientific_plan(tmp_path, usage):
    class Backend:
        def run(self, prompt, schema, directory):
            return {"status": "ok", "usage": usage,
                    "response": {"actions": [{"candidate_id": "c0", "verb": "optimize",
                                              "steps": [], "max_steps": 1}]}}

    exp = workflow.WorkflowScreeningExperiment(
        {"max_llm_calls": 13, "max_observed_tokens": 325000}, tmp_path / "run", Backend())
    exp.science = lambda *args, **kwargs: pytest.fail("Science ran after unknown model usage")
    pool = {"id": "test", "candidates": [
        {"id": f"c{i}", "framework_id": f"f{i}", "initialization": {},
         "catalog": {"n_atoms": 20, "elements": ["C", "H"], "incomplete_record": False}}
        for i in range(8)]}
    with pytest.raises(RuntimeError, match="Unknown model usage"):
        exp.episode(pool, "candidate", policy())
    assert exp.calls == 1
    assert not (exp.output / "pool-state.json").exists()


@pytest.mark.parametrize("deuterium,new_blocker", [(False, False), (True, False), (False, True)])
def test_episode_charges_probe_once_and_obeys_real_worker_diagnostics(tmp_path, monkeypatch, deuterium, new_blocker):
    import rsi.m7_core as core
    monkeypatch.setattr(core, "qualify", lambda candidate, result: {"qualified": False, "reasons": [], "codes": []})
    monkeypatch.setattr(workflow, "grade_report", lambda state, report: {"reliable": True})
    answers = [
        {"actions": [{"candidate_id": "c0", "verb": "optimize", "steps": [], "max_steps": 1}]},
        {"candidates": []},
    ]
    prompts = []

    class Backend:
        def run(self, prompt, schema, directory):
            assert '"qualified"' not in prompt and "private-evaluator-sentinel" not in prompt
            prompts.append(prompt)
            return {"status": "ok", "response": answers.pop(0),
                    "usage": {"input_tokens": 1, "output_tokens": 1}}

    exp = workflow.WorkflowScreeningExperiment(
        {"max_llm_calls": 13, "max_observed_tokens": 325000}, tmp_path / "run", Backend())
    executed = []

    def science(task, actions, directory, prior=None):
        result = copy.deepcopy(prior) if prior else {
            "events": [], "initial": {"diagnostics": []}, "final": {"diagnostics": []},
            "diagnostic_codes": [], "errors": [], "artifact": {"sha256": "test-artifact"},
            "task_id": task["id"], "canonical": "private-evaluator-sentinel", "qualified": True,
        }
        for action in actions:
            executed.append(action)
            codes = ["OVERCOORDINATED"] if new_blocker and action["verb"] == "prepare" else []
            result["final"] = {"diagnostics": [{"code": code} for code in codes]}
            result["diagnostic_codes"] = codes
            result["events"].append({"action": action, "result": {"ok": True}})
        return result

    exp.science = science
    pool = {"id": "test", "candidates": [
        {"id": f"c{i}", "framework_id": f"f{i}", "initialization": {},
         "canonical": "private-evaluator-sentinel", "expected_success": True,
         "catalog": {"n_atoms": 20, "elements": ["C", "D"] if deuterium else ["C", "H"],
                     "incomplete_record": False}} for i in range(8)]}
    row = exp.episode(pool, "candidate", policy(max_decisions=1, refill_on_empty=False,
                                               repair_mode="affordable_before_optimization",
                                               interrupt_after_prepare="new_blocker"))
    saved = json.loads((exp.output / "pool-state.json").read_text())
    steps = ["duplicates", "deuterium"] if deuterium else ["duplicates"]
    expected = [{"verb": "prepare", "steps": steps}]
    if not new_blocker:
        expected.append({"verb": "optimize", "max_steps": 1})
    assert executed == expected
    assert [event["action"] for event in saved["events"]] == expected
    assert saved["spent"] == len(steps) + (0 if new_blocker else 3)
    assert exp.calls == 2 and not answers
    assert row["control"][0]["reason"] == "preparation_probe_inserted"
    assert sum(event["reason"] == "preparation_probe_inserted" for event in row["control"]) == 1
    if new_blocker:
        assert row["control"][1]["reason"] == "after_prepare"
        assert "OVERCOORDINATED" in prompts[-1]
    assert saved["events"][0]["status"] == "completed"

"""Check three-arm boundaries, scientific identity, and budget accounting."""

import copy
from pathlib import Path

import pytest

import rsi.m6 as m6
from rsi.experiment import ROOT, read_json, public_observation
from rsi.m6_evaluation import correspondence
from rsi.science.adapter import input_path
from rsi.workflow import load_policy


def test_fixture_path_cannot_escape():
    with pytest.raises(ValueError):
        input_path({"root": "m6_fixture", "path": "../pilot.json"})
    with pytest.raises(ValueError):
        input_path({"root": "other", "path": "x"})


def test_only_known_policy_can_be_loaded():
    record = read_json(ROOT / "workflows/m5-accepted.json")
    original = load_policy(record)
    assert m6.policy_for("m5", 6, original) == original
    assert m6.policy_for("batch", 6, original)["max_decisions"] == 1
    assert m6.policy_for("stepwise", 6, original)["max_decisions"] == 6


def test_framework_rejects_composition_preserving_atom_exchange():
    from xtal.io.project import read_project
    reference = read_project(input_path({"root": "m6_fixture", "path": "ni-canonical.xtalproj"}))[0]
    altered = reference.copy()
    i = next(i for i, s in enumerate(altered.sites) if s.element == "Ni")
    j = next(i for i, s in enumerate(altered.sites) if s.element == "C")
    altered.sites[i].element, altered.sites[j].element = altered.sites[j].element, altered.sites[i].element
    altered.touch()
    assert not correspondence(reference, altered, 1e-5)


def test_framework_rejects_same_atoms_with_missing_bond():
    from xtal.io.project import read_project
    from xtal.core import bonding
    reference = read_project(input_path({"root": "m6_fixture", "path": "ni-canonical.xtalproj"}))[0]
    altered = reference.copy()
    bonding.graph(altered)
    from xtal.core.structure import Bond
    bond = bonding.perceive(altered)[0]
    altered.add_bond(Bond(bond.i, bond.j, bond.image, kind="suppressed"))
    assert not correspondence(reference, altered, 1e-5)


def test_reserved_or_unknown_stage_usage_stops_next_call(tmp_path):
    budget = m6.StageBudget(tmp_path / "ledger.json")
    entry = budget.reserve("first")
    with pytest.raises(RuntimeError):
        budget.reserve("second")
    entry.update(status="unknown", tokens=0)
    with pytest.raises(RuntimeError):
        budget.reserve("second")


def test_pilot_allocation_cannot_borrow_regression_tokens(tmp_path):
    budget = m6.StageBudget(tmp_path / "ledger.json")
    budget.entries = [{"suite": "pilot", "status": "known", "tokens": 3150000}]
    with pytest.raises(RuntimeError):
        budget.reserve("exhausted")
    budget.suite = "regression"
    assert budget.reserve("regression")["suite"] == "regression"


class Backend:
    def __init__(self, plans, tokens=1):
        self.plans = iter(plans)
        self.prompts = []
        self.tokens = tokens

    def run(self, prompt, schema, directory):
        self.prompts.append(prompt)
        if "scientific_status" in schema.get("properties", {}):
            response = {"task_completed": True, "scientific_status": "not_assessed", "diagnostic_codes": [], "summary": "Saved."}
        else:
            response = {"actions": [{"verb": a, "steps": [], "max_steps": 0} for a in next(self.plans)], "rationale": "Read feedback."}
        return {"status": "ok", "response": response, "usage": {"input_tokens": self.tokens, "output_tokens": 1}}


def execute(tmp_path, monkeypatch, arm, plans, tokens=1, allowance=175000):
    config = dict(read_json(ROOT / "configs/m6_pilot.json"), max_actions_per_task=6, max_llm_calls=7, max_observed_tokens=allowance)
    backend = Backend(plans, tokens)
    exp = m6.ThreeArmExperiment(config, tmp_path / "episode", backend)
    task = {"id": "unit", "instruction": "Inspect and stop.", "expect": {"secret": "NOT_PUBLIC"}}
    def science(task, actions, directory, previous=None):
        prior = (previous or {}).get("attempted_actions", [])
        done = prior + actions
        return {"task_id": "unit", "errors": [], "artifact": {"sha256": "hash"}, "attempted_actions": done,
                "events": [{"action": a, "result": {"ok": True}} for a in done],
                "final": {"feedback_serial": len(done)}, "stopped": bool(done and done[-1]["verb"] == "stop")}
    monkeypatch.setattr(exp, "science", science)
    monkeypatch.setattr(m6, "evaluate_m6", lambda *args: {"checks": {"science": True}, "passed": True})
    row = exp.episode(task, arm, load_policy(read_json(ROOT / "workflows/m5-accepted.json")))
    return row, exp, backend


@pytest.mark.parametrize("arm,plans,expected", [
    ("batch", [["inspect", "stop"]], 1),
    ("m5", [["inspect"], ["stop"]], 2),
    ("stepwise", [["inspect"], ["inspect"], ["stop"]], 3),
])
def test_one_interpreter_obeys_policy_and_shares_observations(tmp_path, monkeypatch, arm, plans, expected):
    row, exp, backend = execute(tmp_path, monkeypatch, arm, plans)
    assert len(row["decisions"]) == expected
    assert exp.calls == expected + 1
    assert not any("NOT_PUBLIC" in prompt for prompt in backend.prompts)
    if expected > 1:
        assert '"feedback_serial": 1' in backend.prompts[1]


def test_m5_cannot_take_third_planning_call(tmp_path, monkeypatch):
    row, exp, _ = execute(tmp_path, monkeypatch, "m5", [["inspect"], ["inspect"]])
    assert row["termination"] == "decision_budget"
    assert exp.calls == 3


def test_empty_plan_is_terminal_even_for_stepwise(tmp_path, monkeypatch):
    row, exp, _ = execute(tmp_path, monkeypatch, "stepwise", [[]])
    assert row["termination"] == "plan_end"
    assert not row["actions"]
    assert exp.calls == 2


def test_token_overshoot_does_not_allow_final_report_call(tmp_path, monkeypatch):
    with pytest.raises(RuntimeError, match="budget exhausted"):
        execute(tmp_path, monkeypatch, "batch", [["stop"]], tokens=100, allowance=100)


def test_summary_rejects_duplicate_or_missing_pairs():
    tasks = [{"id": "one"}]
    assert not m6.summarize([], tasks)["complete"]


def test_private_checkpoint_paths_are_not_model_observations():
    observation = public_observation({"checkpoint_history": [{"path": "private"}], "expect": "answer"})
    assert "private" not in str(observation)
    assert "answer" not in str(observation)

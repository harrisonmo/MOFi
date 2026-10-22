"""Check executable policy boundaries, feedback triggers, and frozen selection."""

import copy
import json
from pathlib import Path

import pytest
from jsonschema import ValidationError

from rsi.workflow import WorkflowExperiment, PARENT, load_policy, policy_hash, selection
from rsi.evolve_workflow import development_feedback
from rsi.science import run_task

ROOT = Path(__file__).resolve().parents[1]
TASKS = json.loads((ROOT / "benchmarks/workflow/tasks.json").read_text())["tasks"]
CONFIG = json.loads((ROOT / "configs/m4_workflow.json").read_text())


class Backend:
    def __init__(self, plans, report):
        self.answers = iter([{"actions": [{"verb": a["verb"], "steps": a.get("steps", []),
            "max_steps": a.get("max_steps", 0)} for a in plan], "rationale": "Follow observations."} for plan in plans] + [report])

    def run(self, *args):
        return {"status": "ok", "response": next(self.answers), "usage": {"input_tokens": 1, "output_tokens": 1}}


def execute(tmp_path, task, policy, plans):
    limit = min(4, task.get("action_budget", 4))
    e = WorkflowExperiment(dict(CONFIG, max_actions_per_task=limit, max_llm_calls=limit + 1),
                           tmp_path / "episode", Backend(plans, task["reference_report"]))
    e.science = lambda task, actions, directory, previous=None: run_task(task, actions, directory, previous)
    return e.episode(task, policy)


@pytest.mark.parametrize("task", TASKS, ids=lambda t: t["id"])
def test_batch_parent_replays_reference_under_same_interpreter(task, tmp_path):
    row = execute(tmp_path, task, PARENT, [task["reference_actions"]])
    assert row["verdict"]["passed"], row


def test_failure_trigger_discards_stale_stop_and_recovers(tmp_path):
    task = TASKS[2]
    policy = dict(PARENT, replan_on=["failure"], max_decisions=2)
    plans = [[task["reference_actions"][0], {"verb": "stop"}], task["reference_actions"][1:]]
    row = execute(tmp_path, task, policy, plans)
    assert row["verdict"]["passed"]
    assert row["control"][1]["trigger"] == ["failure"]
    assert row["control"][1]["discarded"] == [{"verb": "stop"}]
    assert row["actions"] == task["reference_actions"]


def test_trigger_cannot_bypass_decision_cap(tmp_path):
    task = TASKS[2]
    policy = dict(PARENT, replan_on=["failure"])
    row = execute(tmp_path, task, policy, [task["reference_actions"]])
    assert row["termination"] == "decision_budget"
    assert len(row["decisions"]) == 1 and len(row["actions"]) == 1
    assert not row["verdict"]["passed"]


def test_short_plans_refill_using_current_state(tmp_path):
    task = TASKS[1]
    policy = dict(PARENT, plan_horizon=1, refill_on_empty=True, max_decisions=4)
    row = execute(tmp_path, task, policy, [[a] for a in task["reference_actions"]])
    assert row["verdict"]["passed"]
    assert len(row["decisions"]) == 3


@pytest.mark.parametrize("patch", [{"code": "arbitrary"}, {"max_decisions": 5}, {"replan_on": ["task_id"]}, {"plan_horizon": 0}])
def test_candidate_cannot_expand_capabilities(patch):
    with pytest.raises(ValidationError):
        policy_hash(dict(PARENT, **patch))


def test_parent_version_is_hash_verified():
    record = {"workflow": PARENT, "sha256": policy_hash(PARENT)}
    assert load_policy(record) == PARENT
    record["workflow"] = dict(PARENT, plan_horizon=1)
    with pytest.raises(ValueError, match="hash"):
        load_policy(record)


def test_selection_requires_quality_or_substantial_measured_efficiency():
    row = {"verdict": {"passed": True}, "cost": {"observed_tokens": 100, "calls": 3, "actions": 3, "unknown_usage_calls": 0}}
    before = {"a": row}
    assert not selection(before, before)["promoted"]
    after = copy.deepcopy(before)
    after["a"]["cost"]["observed_tokens"] = 80
    assert not selection(before, after)["promoted"]
    after["a"]["cost"]["calls"] = 2
    assert selection(before, after)["promoted"]
    after["a"]["verdict"]["passed"] = False
    assert not selection(before, after)["promoted"]
    assert not selection(before, {})["promoted"]
    after["a"]["verdict"]["passed"] = True
    after["a"]["cost"]["unknown_usage_calls"] = 1
    assert not selection(before, after)["promoted"]


def test_proposal_feedback_excludes_unapproved_top_level_fields():
    feedback = development_feedback([{"task_id": "dev", "cost": {}, "expect": "private", "reference_actions": "private"}])
    assert feedback == [{"task_id": "dev", "cost": {}}]


def test_wire_schema_omits_unsupported_keyword_but_local_validation_rejects_duplicates():
    from rsi.workflow import CANDIDATE_SCHEMA
    assert "uniqueItems" not in json.dumps(CANDIDATE_SCHEMA)
    with pytest.raises(ValidationError):
        policy_hash(dict(PARENT, replan_on=["failure", "failure"]))


def test_development_reuse_rejects_changed_configuration(tmp_path):
    from rsi.evolve_workflow import reuse_development
    (tmp_path / "manifest.json").write_text("{}")
    (tmp_path / "config.json").write_text("{}")
    with pytest.raises(ValueError, match="identical"):
        reuse_development(tmp_path, CONFIG, TASKS, PARENT)

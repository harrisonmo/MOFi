"""Verify matched budgets, execution continuity, and paired accounting."""

import json
from pathlib import Path

import pytest

from rsi.paired import MatchedExperiment, paired_summary
from rsi.science import run_task

ROOT = Path(__file__).resolve().parents[1]
TASKS = json.loads((ROOT / "benchmarks/paired/tasks.json").read_text())["tasks"]
CONFIG = json.loads((ROOT / "configs/m3_paired.json").read_text())


class Backend:
    def __init__(self, task, arm):
        actions = [{"verb": a["verb"], "steps": a.get("steps", []), "max_steps": a.get("max_steps", 0)} for a in task["reference_actions"]]
        self.answers = iter(([{"actions": actions, "rationale": "Reference"}] if arm == "batch" else
                            [{"action": a, "rationale": "Reference"} for a in actions]) + [task["reference_report"]])
        self.prompts = []

    def run(self, prompt, schema, directory):
        self.prompts.append(prompt)
        return {"status": "ok", "response": next(self.answers), "usage": {"input_tokens": 1, "output_tokens": 1}}


@pytest.mark.parametrize("task", TASKS, ids=lambda t: t["id"])
def test_both_arms_use_identical_checkpoint_execution(task, tmp_path):
    states, experiments = [], []
    for arm in ("batch", "stepwise"):
        limit = min(4, task.get("action_budget", 4))
        config = dict(CONFIG, max_actions_per_task=limit, max_llm_calls=limit + 1)
        experiment = MatchedExperiment(config, tmp_path / arm, Backend(task, arm))
        executed = []

        def science(task, actions, directory, previous=None):
            executed.append((actions, previous is not None))
            return run_task(task, actions, directory, previous)

        experiment.science = science
        row = experiment.episode(task, arm)
        assert row["verdict"]["passed"], row
        assert executed == [([], False)] + [([a], True) for a in task["reference_actions"]]
        states.append(json.loads((experiment.output / "final-state.json").read_text(encoding="utf-8")))
        experiments.append(experiment)
    assert states[0]["final"]["n_atoms"] == states[1]["final"]["n_atoms"]
    assert states[0]["attempted_actions"] == states[1]["attempted_actions"]
    assert experiments[0].calls == 2
    assert experiments[1].calls == len(task["reference_actions"]) + 1
    assert experiments[0].backend.prompts[-1].split("\n")[:-1] == experiments[1].backend.prompts[-1].split("\n")[:-1]
    assert "CALCULATION_REFUSED" not in experiments[0].backend.prompts[0]


def test_missing_pair_and_failure_are_not_dropped():
    cost = {"calls": 1, "observed_tokens": 0, "actions": 0, "requested_optimization_steps": 0,
            "wall_seconds": 1, "unknown_usage_calls": 1}
    rows = [{"task_id": "a", "arm": "batch", "verdict": {"passed": False}, "cost": cost}]
    summary = paired_summary(rows, [{"id": "a"}])
    assert not summary["complete"]
    assert summary["pairs"][0]["protocol_delta"] is None
    assert summary["metrics"]["batch"]["expected_tasks"] == 1
    assert summary["metrics"]["batch"]["protocol_passes"] == 0
    assert summary["metrics"]["batch"]["unknown_usage_calls"] == 1


def test_correct_incompletion_is_separate_from_goal_completion():
    cost = {"calls": 2, "observed_tokens": 10, "actions": 1, "requested_optimization_steps": 1,
            "wall_seconds": 1, "unknown_usage_calls": 0}
    rows = [{"task_id": "a", "arm": arm, "verdict": {"passed": True},
             "report": {"task_completed": False}, "cost": cost} for arm in ("batch", "stepwise")]
    summary = paired_summary(rows, [{"id": "a"}])
    assert summary["complete"]
    assert summary["metrics"]["batch"]["protocol_passes"] == 1
    assert summary["metrics"]["batch"]["verified_completions"] == 0
    assert summary["metrics"]["batch"]["correct_incomplete_reports"] == 1


def test_final_response_token_overshoot_is_not_credited(tmp_path):
    task = TASKS[0]
    backend = Backend(task, "batch")
    original = backend.run

    def expensive_report(*args):
        result = original(*args)
        if len(backend.prompts) == 2:
            result["usage"]["output_tokens"] = 100
        return result

    backend.run = expensive_report
    experiment = MatchedExperiment(dict(CONFIG, max_llm_calls=5, max_observed_tokens=100), tmp_path / "run", backend)
    experiment.science = lambda task, actions, directory, previous=None: run_task(task, actions, directory, previous)
    row = experiment.episode(task, "batch")
    assert not row["verdict"]["checks"]["token_allocation_respected"]
    assert not row["verdict"]["passed"]

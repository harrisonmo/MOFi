"""Grade actual numerical feedback rather than a favored workflow label."""

import copy
import json
from pathlib import Path

import pytest

from rsi.science import run_task
from rsi.evaluation import evaluate

ROOT = Path(__file__).resolve().parents[1]
TASKS = json.loads((ROOT / "benchmarks/feedback/tasks.json").read_text())["tasks"]


@pytest.fixture(scope="module")
def replayed(tmp_path_factory):
    root = tmp_path_factory.mktemp("feedback")
    results = {}
    for task in TASKS:
        result = run_task(task, [], root / task["id"] / "initial")
        for i, action in enumerate(task["reference_actions"]):
            result = run_task(task, [action], root / task["id"] / str(i), result)
        results[task["id"]] = result
    return results


@pytest.mark.parametrize("task", TASKS, ids=lambda t: t["id"])
def test_reference_branch_passes(task, replayed):
    grade = evaluate(task, replayed[task["id"]], task["reference_report"])
    assert grade["passed"], grade


@pytest.mark.parametrize("offset", [0, 2])
def test_same_fixed_schedule_cannot_satisfy_both_thresholds(offset, replayed):
    lower, upper = TASKS[offset:offset + 2]
    for source, target in [(lower, upper), (upper, lower)]:
        result = copy.deepcopy(replayed[source["id"]])
        result["task_id"] = target["id"]
        grade = evaluate(target, result, target["reference_report"])
        assert not grade["checks"]["feedback_branch"]
        assert not grade["passed"]


def test_threshold_equality_does_not_authorize_more_steps(replayed):
    task = copy.deepcopy(TASKS[1])
    result = replayed[task["id"]]
    first = next(e for e in result["events"] if e["action"]["verb"] == "optimize")
    task["expect"]["force_feedback"]["threshold"] = first["result"]["data"]["max_force"]
    assert evaluate(task, result, task["reference_report"])["checks"]["feedback_branch"]


@pytest.mark.parametrize("force", [None, float("nan"), float("inf"), True])
def test_missing_or_invalid_first_measurement_fails(force, replayed):
    task = TASKS[0]
    result = copy.deepcopy(replayed[task["id"]])
    first = next(e for e in result["events"] if e["action"]["verb"] == "optimize")
    first["result"]["data"]["max_force"] = force
    assert not evaluate(task, result, task["reference_report"])["checks"]["feedback_measurement"]

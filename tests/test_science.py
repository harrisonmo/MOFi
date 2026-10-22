"""Real scientific replay plus evaluator counterexamples."""

import copy
import json
from pathlib import Path

import pytest

from rsi.evaluation import evaluate
from rsi.science import run_task

TASKS = json.loads((Path(__file__).resolve().parents[1] / "benchmarks/smoke/tasks.json").read_text(encoding="utf-8"))["tasks"]


@pytest.fixture(scope="module")
def replayed(tmp_path_factory):
    root = tmp_path_factory.mktemp("scientific-replay")
    return {t["id"]: run_task(t, t["reference_actions"], root / t["id"]) for t in TASKS}


@pytest.mark.parametrize("task", TASKS, ids=lambda t: t["id"])
def test_reference_replay_passes(task, replayed):
    grade = evaluate(task, replayed[task["id"]], task["reference_report"])
    assert grade["passed"], grade["checks"]


def test_false_convergence_is_rejected(replayed):
    task = next(t for t in TASKS if t["id"] == "short-relax")
    report = {"task_completed": True, "scientific_status": "converged", "diagnostic_codes": [], "summary": "Converged."}
    grade = evaluate(task, replayed[task["id"]], report)
    assert not grade["passed"]
    assert not grade["checks"]["convergence_honesty"]


def test_invented_report_diagnostic_is_rejected(replayed):
    task = TASKS[0]
    report = dict(task["reference_report"], diagnostic_codes=["INVENTED_DIAGNOSTIC"])
    grade = evaluate(task, replayed[task["id"]], report)
    assert not grade["passed"]
    assert not grade["checks"]["report_codes_supported"]


def test_automatic_inspection_suffices_for_read_task(tmp_path):
    task = TASKS[0]
    result = run_task(task, [{"verb": "stop"}], tmp_path)
    assert evaluate(task, result, task["reference_report"])["passed"]


def test_tampered_artifact_fails(replayed, tmp_path):
    task = TASKS[0]
    result = copy.deepcopy(replayed[task["id"]])
    artifact = tmp_path / "corrupt.xtalproj"
    artifact.write_text("invalid scientific artifact", encoding="utf-8")
    result["artifact"]["path"] = str(artifact)
    assert not evaluate(task, result, task["reference_report"])["passed"]


def test_saved_wrong_structure_fails_even_with_matching_hash(replayed):
    task = TASKS[0]
    result = copy.deepcopy(replayed[task["id"]])
    result["artifact"] = replayed["short-relax"]["artifact"]
    grade = evaluate(task, result, task["reference_report"])
    assert grade["checks"]["artifact_integrity"]
    assert not grade["checks"]["composition"]
    assert not grade["passed"]


@pytest.mark.parametrize("actions", [
    [{"verb": "run", "command": "arbitrary"}],
    [{"verb": "prepare", "steps": ["cap"]}],
    [{"verb": "optimize", "max_steps": 4}],
    [{"verb": "optimize", "max_steps": 2}, {"verb": "optimize", "max_steps": 2}],
])
def test_disallowed_actions_fail(actions, tmp_path):
    task = TASKS[0]
    result = run_task(task, actions, tmp_path)
    assert result["errors"]
    assert result["artifact"] is None
    assert not evaluate(task, result, task["reference_report"])["passed"]


def test_missing_preparation_fails_real_artifact_check(tmp_path):
    task = TASKS[1]
    result = run_task(task, [{"verb": "inspect"}, {"verb": "stop"}], tmp_path)
    grade = evaluate(task, result, task["reference_report"])
    assert not grade["passed"]
    assert not grade["checks"]["forbidden_artifact_codes"]


def test_operational_split_is_predeclared():
    assert sum(t["split"] == "development" for t in TASKS) == 3
    assert sum(t["split"] == "validation" for t in TASKS) == 3
    assert len({t["source_family"] for t in TASKS}) == 6


@pytest.mark.parametrize("task_id", ["open-trimers", "short-relax", "build-overlap"])
def test_completed_task_can_retain_scientific_limitations(task_id, replayed):
    task = next(t for t in TASKS if t["id"] == task_id)
    report = task["reference_report"]
    assert report["task_completed"] is True
    assert report["scientific_status"] != "not_assessed"
    assert evaluate(task, replayed[task_id], report)["passed"]


def test_completion_cannot_hide_unresolved_chemistry(replayed):
    task = next(t for t in TASKS if t["id"] == "open-trimers")
    report = dict(task["reference_report"], scientific_status="not_assessed")
    verdict = evaluate(task, replayed[task["id"]], report)
    assert verdict["checks"]["task_completion"]
    assert not verdict["checks"]["scientific_status"]
    assert not verdict["passed"]


def test_convergence_cannot_be_claimed_without_calculation(replayed):
    task = TASKS[0]
    report = dict(task["reference_report"], scientific_status="converged")
    assert not evaluate(task, replayed[task["id"]], report)["checks"]["convergence_honesty"]


@pytest.mark.parametrize("patch", [{"status": "completed"}, {"task_completed": "true"}, {"summary": "  "}])
def test_malformed_or_legacy_report_fails_closed(patch, replayed):
    task = TASKS[0]
    report = dict(task["reference_report"], **patch)
    verdict = evaluate(task, replayed[task["id"]], report)
    assert not verdict["checks"]["report_schema"]
    assert not verdict["passed"]


def test_earlier_nonfinite_relaxation_is_not_hidden_by_later_event(replayed):
    task = next(t for t in TASKS if t["id"] == "short-relax")
    result = copy.deepcopy(replayed[task["id"]])
    event = next(e for e in result["events"] if e["action"].get("verb") == "optimize")
    invalid = copy.deepcopy(event)
    invalid["result"]["data"]["energy"] = float("nan")
    result["events"].insert(0, invalid)
    assert not evaluate(task, result, task["reference_report"])["checks"]["relaxation_finite"]

"""State continuity, conditional repair, recovery, and cumulative budget checks."""

import json
from pathlib import Path

import pytest

from rsi.experiment import Experiment
from rsi.evaluation import evaluate
from rsi.science import run_task

ROOT = Path(__file__).resolve().parents[1]
TASKS = json.loads((ROOT / "benchmarks/stepwise/tasks.json").read_text())["tasks"]
CONFIG = json.loads((ROOT / "configs/m1_stepwise.json").read_text())


def response(action):
    return {"action": {"verb": action["verb"], "steps": action.get("steps", []),
                       "max_steps": action.get("max_steps", 0)}, "rationale": "Use current evidence."}


class ScriptedBackend:
    def __init__(self, answers):
        self.answers = iter(answers)
        self.prompts = []

    def run(self, prompt, schema, directory):
        self.prompts.append(prompt)
        return {"status": "ok", "response": next(self.answers), "usage": {"input_tokens": 1}}


def controller(tmp_path, task, actions, config=None):
    backend = ScriptedBackend([response(a) for a in actions] + [task["reference_report"]])
    experiment = Experiment(config or CONFIG, tmp_path / "run", backend)
    experiment.science = lambda task, actions, directory, previous=None: run_task(task, actions, directory, previous)
    return experiment, backend


@pytest.mark.parametrize("task", TASKS, ids=lambda t: t["id"])
def test_stepwise_controller_with_real_science(task, tmp_path):
    experiment, backend = controller(tmp_path, task, task["reference_actions"])
    row = experiment.research(task, CONFIG["baseline_prompt"], experiment.output / "task")
    assert row["verdict"]["passed"], row["verdict"]
    assert row["termination"] == "agent_stop"
    assert experiment.calls == len(task["reference_actions"]) + 1
    if task["id"] == "conditional-repair":
        context = json.loads(backend.prompts[1].split("\n")[-1])
        assert context["observation"]["final"]["n_atoms"] == 102
        assert context["remaining_actions"] == 3
    if task.get("fault_fixture"):
        assert "SIMULATED_ENGINE_UNAVAILABLE" in backend.prompts[1]
        context = json.loads(backend.prompts[1].split("\n")[-1])
        assert context["remaining_optimization_steps"] == 2


def test_resume_does_not_repeat_prepare_or_open_original(tmp_path, monkeypatch):
    from xtal.agent import Session

    task = TASKS[1]
    original = Session.prepare
    calls = []

    def prepare(self, **kwargs):
        calls.append(kwargs)
        return original(self, **kwargs)

    monkeypatch.setattr(Session, "prepare", prepare)
    initial = run_task(task, [], tmp_path / "initial")
    prepared = run_task(task, [task["reference_actions"][0]], tmp_path / "prepared", initial)
    resumed = run_task(task, [{"verb": "inspect"}], tmp_path / "resumed", prepared)
    assert len(calls) == 1
    assert resumed["final"]["n_atoms"] == prepared["final"]["n_atoms"] == 102
    assert resumed["initial"] == initial["initial"]
    assert resumed["resumed_from"] == prepared["artifact"]
    assert len([e for e in resumed["events"] if e["action"]["verb"] == "prepare"]) == 1


@pytest.mark.parametrize("change", ["hash", "identity", "stop"])
def test_invalid_checkpoint_fails_closed(change, tmp_path):
    task = TASKS[0]
    previous = run_task(task, [], tmp_path / "initial")
    if change == "hash":
        previous["artifact"]["sha256"] = "bad"
    elif change == "identity":
        previous["task_id"] = "other"
    else:
        previous["stopped"] = True
    result = run_task(task, [{"verb": "inspect"}], tmp_path / "next", previous)
    assert result["errors"] and result["artifact"] is None


def test_failed_attempt_consumes_optimization_budget(tmp_path):
    task = TASKS[2]
    failed = run_task(task, [{"verb": "optimize", "max_steps": 2}], tmp_path / "failed")
    assert not failed["events"][-1]["result"]["ok"]
    result = run_task(task, [{"verb": "optimize", "max_steps": 2}], tmp_path / "next", failed)
    assert result["errors"] and result["artifact"] is None


def test_recovery_cannot_hide_unexpected_or_unrecovered_failure(tmp_path):
    task = TASKS[2]
    failed = run_task(task, [task["reference_actions"][0]], tmp_path / "failed")
    assert not evaluate(task, failed, task["reference_report"])["passed"]
    recovered = run_task(task, [task["reference_actions"][1]], tmp_path / "recovered", failed)
    recovered["events"][0]["result"]["diagnostics"][0]["code"] = "UNEXPECTED_FAILURE"
    assert not evaluate(task, recovered, task["reference_report"])["checks"]["verbs_succeeded"]


def test_action_budget_terminates_without_extra_decision(tmp_path):
    task = TASKS[0]
    config = dict(CONFIG, max_actions_per_task=1)
    experiment, _ = controller(tmp_path, task, [{"verb": "inspect"}], config)
    row = experiment.research(task, "test", experiment.output / "task")
    assert row["termination"] == "action_budget"
    assert experiment.calls == 2


def test_model_call_budget_reserves_final_report(tmp_path):
    task = TASKS[0]
    experiment, _ = controller(tmp_path, task, [], dict(CONFIG, max_llm_calls=1))
    row = experiment.research(task, "test", experiment.output / "task")
    assert row["termination"] == "model_call_budget"
    assert experiment.calls == 1


def test_overbudget_request_is_not_executed(tmp_path):
    task = TASKS[2]
    actions = [{"verb": "optimize", "max_steps": 2}] * 2
    experiment, _ = controller(tmp_path, task, actions)
    row = experiment.research(task, "test", experiment.output / "task")
    assert row["termination"] == "invalid_budget_request"
    assert not row["verdict"]["passed"]
    assert len(row["plan"]["actions"]) == 1
    assert (experiment.output / "task/rejected-action.json").exists()


@pytest.mark.parametrize("violation", ["step_size", "retry_limit"])
def test_recovery_task_enforces_requested_step_and_retry_limits(violation, tmp_path):
    task = TASKS[2]
    result = run_task(task, task["reference_actions"], tmp_path / "result")
    if violation == "step_size":
        result["events"][1]["action"]["max_steps"] = 2
        check = "requested_step_size"
    else:
        result["events"].insert(2, result["events"][1])
        check = "retry_limit"
    assert not evaluate(task, result, task["reference_report"])["checks"][check]


def test_injected_failure_leaves_structure_unchanged(tmp_path):
    import zipfile

    task = TASKS[2]
    initial = run_task(task, [], tmp_path / "initial")
    failed = run_task(task, [{"verb": "optimize", "max_steps": 1}], tmp_path / "failed", initial)
    with zipfile.ZipFile(initial["artifact"]["path"]) as before, zipfile.ZipFile(failed["artifact"]["path"]) as after:
        for member in ("structure.cif", "bonds.json", "sites.json"):
            assert (member in before.namelist()) == (member in after.namelist())
            if member in before.namelist():
                assert before.read(member) == after.read(member)


RECOVERY_TASKS = json.loads((ROOT / "benchmarks/recovery/tasks.json").read_text())["tasks"]


@pytest.mark.parametrize("task", RECOVERY_TASKS, ids=lambda t: t["id"])
def test_real_refusal_and_budget_blocking(task, tmp_path):
    experiment, _ = controller(tmp_path, task, task["reference_actions"])
    row = experiment.research(task, "test", experiment.output / "task")
    assert row["verdict"]["passed"], row["verdict"]
    assert row["report"]["task_completed"] == (task["action_budget"] > 1)
    assert row["termination"] == ("agent_stop" if task["action_budget"] > 1 else "action_budget")
    state = json.loads((experiment.output / "task/final-state.json").read_text())
    refusal = state["events"][0]["result"]
    assert refusal["ok"] is False and not refusal.get("injected")
    assert "on top of one another" in refusal["diagnostics"][0]["message"]


def test_budget_exhaustion_does_not_credit_false_completion(tmp_path):
    task = RECOVERY_TASKS[1]
    result = run_task(task, task["reference_actions"], tmp_path)
    result["termination"] = "action_budget"
    report = dict(task["reference_report"], task_completed=True)
    grade = evaluate(task, result, report)
    assert not grade["checks"]["task_completion"] and not grade["passed"]


def test_real_recovery_rejects_retry_without_intervening_repair(tmp_path):
    task = RECOVERY_TASKS[0]
    result = run_task(task, task["reference_actions"], tmp_path)
    result["events"] = [e for e in result["events"] if e["action"]["verb"] != "prepare"]
    grade = evaluate(task, result, task["reference_report"])
    assert not grade["checks"]["repair_before_retry"] and not grade["passed"]


def test_injected_failure_cannot_masquerade_as_real_refusal(tmp_path):
    task = RECOVERY_TASKS[0]
    result = run_task(task, task["reference_actions"], tmp_path)
    result["events"][0]["result"]["injected"] = True
    assert not evaluate(task, result, task["reference_report"])["checks"]["real_refusal_observed"]

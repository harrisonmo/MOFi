"""Compare open-loop batches and feedback decisions under matched budget ceilings."""

import argparse
import copy
import hashlib
import importlib.metadata
import time
from datetime import datetime, timezone
from pathlib import Path

from rsi.experiment import (ROOT, Experiment, PLAN_SCHEMA, STEP_SCHEMA, cli_identity,
                            git_state, normalize_actions, public_observation, read_json, write_json)
from rsi.protocol import REPORT_SCHEMA, REPORT_INSTRUCTIONS, REPORT_PROTOCOL_VERSION
from rsi.science.adapter import validate_action
from rsi.evaluation import evaluate

import json

TOOLS = (
    "The structure has already been inspected. inspect reads; prepare applies explicitly selected steps; "
    "optimize runs UFF for 1-3 requested steps; stop ends the episode. "
    "Use steps=[] unless preparing and max_steps=0 unless optimizing. "
    "Each action resumes the latest saved checkpoint; prior actions are never replayed. "
    "Failed attempts consume actions and requested optimization steps. "
    "Total requested optimization budget is three steps. Preserve chemistry outside authorized changes. "
)


def paired_summary(rows, tasks):
    """Retain missing pairs and failures; never silently reduce the denominator."""
    metrics = {}
    for arm in ("batch", "stepwise"):
        selected = [r for r in rows if r["arm"] == arm]
        metrics[arm] = {
            "expected_tasks": len(tasks), "recorded_tasks": len(selected),
            "protocol_passes": sum(r["verdict"]["passed"] for r in selected),
            "verified_completions": sum(r["verdict"]["passed"] and r.get("report", {}).get("task_completed") is True for r in selected),
            "correct_incomplete_reports": sum(r["verdict"]["passed"] and r.get("report", {}).get("task_completed") is False for r in selected),
            "calls": sum(r["cost"]["calls"] for r in selected),
            "observed_tokens": sum(r["cost"]["observed_tokens"] for r in selected),
            "actions": sum(r["cost"]["actions"] for r in selected),
            "requested_optimization_steps": sum(r["cost"]["requested_optimization_steps"] for r in selected),
            "wall_seconds": sum(r["cost"]["wall_seconds"] for r in selected),
            "unknown_usage_calls": sum(r["cost"]["unknown_usage_calls"] for r in selected),
        }
    pairs = []
    for task in tasks:
        pair = {r["arm"]: r for r in rows if r["task_id"] == task["id"]}
        complete = set(pair) == {"batch", "stepwise"}
        pairs.append({"task_id": task["id"], "complete": complete,
                      "protocol_delta": int(pair["stepwise"]["verdict"]["passed"]) - int(pair["batch"]["verdict"]["passed"]) if complete else None})
    return {"metrics": metrics, "pairs": pairs,
            "complete": len(rows) == 2 * len(tasks) and all(p["complete"] for p in pairs),
            "interpretation": "Matched allocation ceilings, not matched realized spend; descriptive public pilot only."}


class MatchedExperiment(Experiment):
    """Use the same execution, observations, report prompt, and evaluator in both arms."""

    def episode(self, task, arm):
        limit = self.config["max_actions_per_task"]
        result = self.science(task, [], self.output / "initial")
        actions, decisions, pending = [], [], []
        termination = "action_budget"
        for index in range(limit):
            if result["errors"] or not result["artifact"]:
                termination = "scientific_failure"
                break
            remaining = 3 - sum(a.get("max_steps", 0) for a in actions)
            if arm == "stepwise" or index == 0:
                context = {"instruction": task["instruction"], "observation": public_observation(result),
                           "remaining_actions": limit - index, "remaining_optimization_steps": remaining,
                           "prior_decisions": decisions}
                schema = copy.deepcopy(PLAN_SCHEMA if arm == "batch" else STEP_SCHEMA)
                if arm == "batch":
                    schema["properties"]["actions"]["maxItems"] = limit
                mode = (f"Choose one fixed sequence of at most {limit} actions now. There are no branches or further decisions before the final report. "
                        if arm == "batch" else "Choose exactly one next action using the latest feedback. ")
                decision = self.model(self.config["baseline_prompt"] + "\n" + TOOLS + mode + "\n"
                                      + json.dumps(context, ensure_ascii=False), schema, self.output / f"decision-{index:02d}")
                decisions.append(decision)
                pending = normalize_actions(decision["actions"] if arm == "batch" else [decision["action"]])
                if arm == "batch" and len(pending) > limit:
                    termination = "invalid_budget_request"
                    break
            if not pending:
                termination = "plan_end"
                break
            action = pending.pop(0)
            validate_action(action)
            if action.get("max_steps", 0) > remaining:
                termination = "invalid_budget_request"
                break
            actions.append(action)
            result = self.science(task, [action], self.output / f"step-{index:02d}", previous=result)
            write_json(self.output / "trajectory.json", {"decisions": decisions, "result": result})
            if result["stopped"]:
                termination = "agent_stop"
                break
        result["termination"] = termination
        write_json(self.output / "final-state.json", result)
        report = self.model(self.config["baseline_prompt"] + "\nProduce the final report; no actions remain.\n"
                            + REPORT_INSTRUCTIONS + "\n" + json.dumps({"instruction": task["instruction"],
                                "observation": public_observation(result)}, ensure_ascii=False),
                            REPORT_SCHEMA, self.output / "report")
        verdict = evaluate(task, result, report)
        verdict["checks"]["controller_budget_compliant"] = termination != "invalid_budget_request"
        verdict["checks"]["token_allocation_respected"] = self.observed_tokens <= self.config["max_observed_tokens"]
        verdict["passed"] = all(verdict["checks"].values())
        return {"task_id": task["id"], "arm": arm, "termination": termination,
                "actions": actions, "decisions": decisions, "report": report, "verdict": verdict}


def run(config, output):
    output.mkdir(parents=True, exist_ok=False)
    tasks = read_json(ROOT / config["tasks_path"])["tasks"]
    write_json(output / "config.json", config)
    write_json(output / "tasks.json", tasks)
    schedule = [(t, arm) for i, t in enumerate(tasks)
                for arm in (("batch", "stepwise") if i % 2 == 0 else ("stepwise", "batch"))]
    manifest = {"started_utc": datetime.now(timezone.utc).isoformat(),
                "root": git_state(ROOT), "crystal_builder": git_state(ROOT / "Crystal-Builder"),
                "report_protocol_version": REPORT_PROTOCOL_VERSION,
                "tasks_sha256": hashlib.sha256(json.dumps(tasks, sort_keys=True).encode()).hexdigest(),
                "schedule": [{"task_id": t["id"], "arm": arm} for t, arm in schedule],
                "packages": {n: importlib.metadata.version(n) for n in ["numpy", "scipy", "ase", "spglib", "gemmi", "crystal-builder"]},
                "input_hashes": {t["initialization"]["path"]: hashlib.sha256((ROOT / "Crystal-Builder" / t["initialization"]["path"]).read_bytes()).hexdigest() for t in tasks}}
    rows = []
    for task, arm in schedule:
        episode_config = dict(config)
        episode_config["max_actions_per_task"] = min(config["max_actions_per_task"], task.get("action_budget", 4))
        episode_config["max_llm_calls"] = episode_config["max_actions_per_task"] + 1
        experiment = MatchedExperiment(episode_config, output / task["id"] / arm)
        identity = cli_identity(experiment.backend.executable)
        if "cli" not in manifest:
            manifest["cli"] = identity
            write_json(output / "manifest.json", manifest)
        elif identity != manifest["cli"]:
            raise RuntimeError("CLI identity changed during paired run")
        started = time.monotonic()
        try:
            row = experiment.episode(task, arm)
        except Exception as exc:
            row = {"task_id": task["id"], "arm": arm, "error": f"{type(exc).__name__}: {exc}",
                   "verdict": {"passed": False, "checks": {"episode_completed": False}}}
            write_json(experiment.output / "failure.json", row)
        state_path = experiment.output / "final-state.json"
        trajectory = experiment.output / "trajectory.json"
        state = read_json(state_path) if state_path.exists() else read_json(trajectory)["result"] if trajectory.exists() else {}
        executed = state.get("attempted_actions", [])
        row["cost"] = {"calls": experiment.calls, "observed_tokens": experiment.observed_tokens,
                       "actions": len(executed), "requested_optimization_steps": sum(a.get("max_steps", 0) for a in executed),
                       "wall_seconds": time.monotonic() - started,
                       "unknown_usage_calls": sum("input_tokens" not in e.get("usage", {}) or "output_tokens" not in e.get("usage", {}) for e in experiment.ledger)}
        write_json(experiment.output / "graded.json", row)
        rows.append(row)
        write_json(output / "rows.json", rows)
        write_json(output / "summary.json", paired_summary(rows, tasks))
    return paired_summary(rows, tasks)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/m3_paired.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(read_json(args.config), args.output.resolve()), indent=2))


if __name__ == "__main__":
    main()

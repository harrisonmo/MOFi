"""Run the approved M6 development pilot and legacy regression in three arms."""

import argparse
import copy
import hashlib
import importlib.metadata
import json
import time
from datetime import datetime, timezone
from pathlib import Path

from rsi.experiment import ROOT, Experiment, PLAN_SCHEMA, cli_identity, git_state, normalize_actions, public_observation, read_json, write_json
from rsi.m6_evaluation import evaluate_m6
from rsi.paired import TOOLS
from rsi.protocol import REPORT_INSTRUCTIONS, REPORT_SCHEMA
from rsi.science.adapter import input_path, validate_action
from rsi.workflow import load_policy

ARMS = ("batch", "m5", "stepwise")


def policy_for(arm, limit, accepted):
    if arm == "m5":
        return copy.deepcopy(accepted)
    if arm not in ARMS:
        raise ValueError("Unknown arm")
    return {"plan_horizon": limit if arm == "batch" else 1, "replan_on": [],
            "refill_on_empty": arm == "stepwise", "max_decisions": 1 if arm == "batch" else limit}


class StageBudget:
    """Persist attempted calls and stop on unknown usage or exhausted allocation."""

    def __init__(self, path):
        self.path = Path(path)
        self.entries = []
        self.suite = "pilot"
        self.save()

    def save(self):
        write_json(self.path, self.entries)

    @property
    def tokens(self):
        return sum(e.get("tokens", 0) for e in self.entries)

    def suite_exhausted(self, suite):
        entries = [e for e in self.entries if e["suite"] == suite]
        calls, tokens = (126, 3150000) if suite == "pilot" else (84, 2100000)
        return len(entries) >= calls or sum(e.get("tokens", 0) for e in entries) >= tokens

    def reserve(self, label):
        if len(self.entries) >= 210 or self.tokens >= 5250000 or self.suite_exhausted(self.suite) or any(e["status"] in {"reserved", "unknown"} for e in self.entries):
            raise RuntimeError("Stage budget exhausted or usage unknown")
        entry = {"label": label, "suite": self.suite, "status": "reserved"}
        self.entries.append(entry)
        self.save()
        return entry


class ThreeArmExperiment(Experiment):
    """One interpreter and one prompt template, parameterized only by policy."""

    def __init__(self, config, output, backend=None, stage=None):
        super().__init__(config, output, backend)
        self.stage = stage

    def model(self, prompt, schema, directory):
        if self.calls >= self.config["max_llm_calls"] or self.observed_tokens >= self.config["max_observed_tokens"]:
            raise RuntimeError("Episode model budget exhausted")
        entry = self.stage.reserve(str(directory)) if self.stage else None
        before = self.observed_tokens
        try:
            return super().model(prompt, schema, directory)
        finally:
            if entry is not None:
                usage = self.ledger[-1].get("usage", {}) if self.ledger else {}
                known = all(type(usage.get(k)) is int and usage[k] >= 0 for k in ("input_tokens", "output_tokens"))
                entry.update(status="known" if known else "unknown", tokens=self.observed_tokens - before)
                self.stage.save()

    def episode(self, task, arm, accepted):
        limit = min(self.config["max_actions_per_task"], task.get("action_budget", 6))
        policy = policy_for(arm, limit, accepted)
        result = self.science(task, [], self.output / "initial")
        actions, decisions, pending = [], [], []
        reason = "action_budget"
        for index in range(limit):
            if result["errors"] or not result["artifact"]:
                reason = "scientific_failure"
                break
            if not pending:
                if decisions and not policy["refill_on_empty"]:
                    reason = "plan_end"
                    break
                if len(decisions) >= policy["max_decisions"] or self.calls >= self.config["max_llm_calls"] - 1:
                    reason = "decision_budget"
                    break
                horizon = min(policy["plan_horizon"], limit - index)
                schema = copy.deepcopy(PLAN_SCHEMA)
                schema["properties"]["actions"]["maxItems"] = horizon
                context = {"instruction": task["instruction"], "observation": public_observation(result),
                           "remaining_actions": limit - index, "remaining_optimization_steps": 3 - sum(a.get("max_steps", 0) for a in actions),
                           "prior_decisions": decisions, "workflow": policy}
                decision = self.model(self.config["baseline_prompt"] + "\n" + TOOLS
                    + f"Plan at most {horizon} next actions. Action lists cannot contain branches. "
                    "A terminal stop or an empty plan ends the episode. When a nonempty queue is exhausted, "
                    "the workflow may allow another planning call. No replanning happens while actions remain queued.\n"
                    + json.dumps(context, ensure_ascii=False), schema, self.output / f"decision-{len(decisions):02d}")
                decisions.append(decision)
                pending = normalize_actions(decision["actions"])
                if len(pending) > horizon:
                    reason = "invalid_budget_request"
                    break
            if not pending:
                reason = "plan_end"
                break
            action = pending.pop(0)
            validate_action(action)
            if sum(a.get("max_steps", 0) for a in actions) + action.get("max_steps", 0) > 3:
                reason = "invalid_budget_request"
                break
            actions.append(action)
            result = self.science(task, [action], self.output / f"step-{index:02d}", result)
            write_json(self.output / "trajectory.json", {"result": result, "decisions": decisions})
            if result["stopped"]:
                reason = "agent_stop"
                break
        result["termination"] = reason
        write_json(self.output / "final-state.json", result)
        report = self.model(self.config["baseline_prompt"] + "\nProduce the final report; no actions remain.\n"
                            + REPORT_INSTRUCTIONS + "\n" + json.dumps({"instruction": task["instruction"],
                                "observation": public_observation(result)}, ensure_ascii=False), REPORT_SCHEMA, self.output / "report")
        verdict = evaluate_m6(task, result, report)
        verdict["checks"]["controller_budget_compliant"] = reason != "invalid_budget_request"
        verdict["checks"]["token_allocation_respected"] = self.observed_tokens <= self.config["max_observed_tokens"]
        verdict["passed"] = all(verdict["checks"].values())
        return {"task_id": task["id"], "arm": arm, "workflow": policy, "actions": actions,
                "decisions": decisions, "termination": reason, "report": report, "verdict": verdict}


def summarize(rows, tasks):
    metrics = {}
    for arm in ARMS:
        selected = [r for r in rows if r["arm"] == arm]
        metrics[arm] = {"expected_tasks": len(tasks), "recorded_tasks": len(selected),
            "passes": sum(r["verdict"]["passed"] for r in selected),
            "verified_completions": sum(r["verdict"]["passed"] and r.get("report", {}).get("task_completed") is True for r in selected),
            "correct_incomplete": sum(r["verdict"]["passed"] and r.get("report", {}).get("task_completed") is False for r in selected),
            **{key: sum(r["cost"][key] for r in selected) for key in ("calls", "observed_tokens", "unknown_usage_calls", "actions", "requested_optimization_steps", "actual_optimization_steps", "prepare_substeps", "wall_seconds")}}
    expected = {(t["id"], a) for t in tasks for a in ARMS}
    keys = [(r["task_id"], r["arm"]) for r in rows]
    return {"metrics": metrics, "complete": set(keys) == expected and len(keys) == len(expected),
            "comparable": set(keys) == expected and len(keys) == len(expected) and not any(r.get("error") or r["cost"]["unknown_usage_calls"] for r in rows),
            "interpretation": "Public development pilot; matched ceilings, not matched realized spend; no promotion."}


def run(config, output):
    output.mkdir(parents=True, exist_ok=False)
    pilot = read_json(ROOT / "benchmarks/m6/pilot.json")["tasks"]
    regression = read_json(ROOT / "benchmarks/paired/tasks.json")["tasks"]
    if len(pilot) != 6 or {t.get("category") for t in pilot} != set("ABCDEF") or any(t["split"] != "development" for t in pilot) or len(regression) != 4:
        raise ValueError("Only the approved six development and four legacy cases may run")
    accepted_record = read_json(ROOT / "workflows/m5-accepted.json")
    accepted = load_policy(accepted_record)
    if accepted_record["sha256"] != "7d1fd442d56473cf6669bb1dfff9caf4c94c8a09233178e08c30ce3b563dfdf1":
        raise ValueError("M5 accepted policy changed")
    stage = StageBudget(output / "stage-ledger.json")
    schedule = [(suite, task, arm) for suite, tasks in [("pilot", pilot), ("regression", regression)]
                for i, task in enumerate(tasks) for arm in ARMS[i % 3:] + ARMS[:i % 3]]
    manifest = {"started_utc": datetime.now(timezone.utc).isoformat(), "root": git_state(ROOT),
        "crystal_builder": git_state(ROOT / "Crystal-Builder"), "config": config, "accepted": accepted_record,
        "schedule": [{"suite": s, "task_id": t["id"], "arm": a} for s, t, a in schedule],
        "packages": {n: importlib.metadata.version(n) for n in ["numpy", "scipy", "ase", "spglib", "gemmi", "crystal-builder"]},
        "task_hashes": {name: hashlib.sha256(json.dumps(ts, sort_keys=True).encode()).hexdigest() for name, ts in [("pilot", pilot), ("regression", regression)]},
        "input_hashes": {str(input_path(t["initialization"]).relative_to(ROOT)): hashlib.sha256(input_path(t["initialization"]).read_bytes()).hexdigest() for t in pilot + regression},
        "isolation": "Empty CLI working directory, decision-only prompt, forbidden-tool event rejection; no OS-level held-out isolation claim."}
    write_json(output / "tasks.json", {"pilot": pilot, "regression": regression})
    rows = []
    for suite, task, arm in schedule:
        if stage.tokens >= 5250000 or len(stage.entries) >= 210 or any(e["status"] != "known" for e in stage.entries):
            break
        if stage.suite_exhausted(suite):
            continue
        stage.suite = suite
        limit = task.get("action_budget", 4 if suite == "regression" else 6)
        episode_config = dict(config, max_actions_per_task=limit, max_llm_calls=limit + 1,
                              max_observed_tokens=75000 if task.get("category") == "F" else 175000)
        exp = ThreeArmExperiment(episode_config, output / suite / task["id"] / arm, stage=stage)
        identity = cli_identity(exp.backend.executable)
        if "cli" in manifest and manifest["cli"] != identity:
            raise RuntimeError("CLI identity changed")
        manifest["cli"] = identity
        write_json(output / "manifest.json", manifest)
        started = time.monotonic()
        try:
            row = exp.episode(task, arm, accepted)
        except Exception as exc:
            row = {"task_id": task["id"], "arm": arm, "error": f"{type(exc).__name__}: {exc}",
                   "verdict": {"passed": False, "checks": {"episode_completed": False}}}
        path = exp.output / "final-state.json"
        trajectory = exp.output / "trajectory.json"
        state = read_json(path) if path.exists() else read_json(trajectory)["result"] if trajectory.exists() else {}
        executed = state.get("attempted_actions", [])
        row.update(suite=suite, category=task.get("category"), source_family=task["source_family"])
        row["cost"] = {"calls": exp.calls, "observed_tokens": exp.observed_tokens, "actions": len(executed),
            "prepare_substeps": sum(len(a.get("steps", [])) for a in executed),
            "requested_optimization_steps": sum(a.get("max_steps", 0) for a in executed),
            "actual_optimization_steps": sum(e["result"].get("data", {}).get("steps", 0) for e in state.get("events", []) if e["action"]["verb"] == "optimize"),
            "wall_seconds": time.monotonic() - started,
            "unknown_usage_calls": sum(not all(k in e.get("usage", {}) for k in ("input_tokens", "output_tokens")) for e in exp.ledger)}
        row["recoverable_refusals"] = sum(e["result"].get("ok") is False and any(d["code"] == "CALCULATION_REFUSED" for d in e["result"].get("diagnostics", [])) for e in state.get("events", []))
        write_json(exp.output / "graded.json", row)
        rows.append(row)
        write_json(output / "rows.json", rows)
        summary = {name: summarize([r for r in rows if r["suite"] == name], ts) for name, ts in [("pilot", pilot), ("regression", regression)]}
        summary["total"] = {"episodes": len(rows), "calls": len(stage.entries), "observed_tokens": stage.tokens}
        write_json(output / "summary.json", summary)
        print(suite, task["id"], arm, row["verdict"]["passed"], row["cost"]["observed_tokens"], flush=True)
    return read_json(output / "summary.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/m6_pilot.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(read_json(args.config), args.output.resolve()), indent=2))


if __name__ == "__main__":
    main()

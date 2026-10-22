"""Propose one bounded workflow and gate it on predeclared paired validation."""

import argparse
import hashlib
import importlib.metadata
import sys
import json
import time
from datetime import datetime, timezone

from rsi.experiment import ROOT, Experiment, cli_identity, git_state, read_json, write_json
from rsi.workflow import (WorkflowExperiment, PARENT, POLICY_SCHEMA, CANDIDATE_SCHEMA,
                          load_policy, policy_hash, selection)


def episode(config, task, policy, output, expected_cli=None):
    settings = dict(config)
    settings["max_actions_per_task"] = min(config["max_actions_per_task"], task.get("action_budget", 4))
    settings["max_llm_calls"] = settings["max_actions_per_task"] + 1
    experiment = WorkflowExperiment(settings, output)
    if expected_cli is not None and cli_identity(experiment.backend.executable) != expected_cli:
        raise RuntimeError("CLI identity changed during workflow evolution")
    started = time.monotonic()
    try:
        row = experiment.episode(task, policy)
    except Exception as exc:
        row = {"task_id": task["id"], "error": f"{type(exc).__name__}: {exc}",
               "verdict": {"passed": False, "checks": {"episode_completed": False}}}
        write_json(output / "failure.json", row)
    state = read_json(output / "final-state.json") if (output / "final-state.json").exists() else (
        read_json(output / "trajectory.json")["result"] if (output / "trajectory.json").exists() else {})
    row["cost"] = {"calls": experiment.calls, "observed_tokens": experiment.observed_tokens,
                   "actions": len(state.get("attempted_actions", [])), "wall_seconds": time.monotonic() - started,
                   "unknown_usage_calls": sum("input_tokens" not in e.get("usage", {}) or "output_tokens" not in e.get("usage", {}) for e in experiment.ledger)}
    write_json(output / "graded.json", row)
    return row


def development_feedback(rows):
    """Allow only already executed development outcomes into the proposal context."""
    return [{k: row[k] for k in ("task_id", "actions", "report", "verdict", "cost") if k in row} for row in rows]


def reuse_development(source, config, tasks, parent):
    from rsi.evaluation import evaluate
    manifest = read_json(source / "manifest.json")
    if read_json(source / "config.json") != config or read_json(source / "tasks.json") != tasks:
        raise ValueError("Development reuse requires identical config and task definitions")
    if manifest["parent_sha256"] != policy_hash(parent):
        raise ValueError("Development parent differs")
    if manifest["crystal_builder"] != git_state(ROOT / "Crystal-Builder"):
        raise ValueError("Scientific repository differs from development run")
    rows = read_json(source / "development.json")
    expected = {t["id"]: t for t in tasks if t["split"] == "development"}
    if len(rows) != len(expected) or {r["task_id"] for r in rows} != set(expected):
        raise ValueError("Development outcomes are incomplete")
    for row in rows:
        state = read_json(source / "development" / row["task_id"] / "final-state.json")
        if not evaluate(expected[row["task_id"]], state, row["report"])["passed"]:
            raise ValueError("Reused development evidence did not revalidate")
    return rows


def run(config, output, parent_record=None, development_from=None):
    output.mkdir(parents=True, exist_ok=False)
    parent = load_policy(parent_record) if parent_record else PARENT
    tasks = read_json(ROOT / config["tasks_path"])["tasks"]
    write_json(output / "config.json", config)
    write_json(output / "tasks.json", tasks)
    write_json(output / "parent.json", {"workflow": parent, "sha256": policy_hash(parent)})
    proposer = Experiment(dict(config, max_llm_calls=1), output / "evolution")
    manifest = {"started_utc": datetime.now(timezone.utc).isoformat(), "root": git_state(ROOT),
                "crystal_builder": git_state(ROOT / "Crystal-Builder"), "cli": cli_identity(proposer.backend.executable),
                "tasks_sha256": hashlib.sha256(json.dumps(tasks, sort_keys=True).encode()).hexdigest(),
                "python": sys.version,
                "packages": {n: importlib.metadata.version(n) for n in ["numpy", "scipy", "ase", "spglib", "gemmi", "crystal-builder"]},
                "parent_sha256": policy_hash(parent), "scope": "public_engineering_workflow_evolution_not_hidden_generalization",
                "input_hashes": {t["initialization"]["path"]: hashlib.sha256((ROOT / "Crystal-Builder" / t["initialization"]["path"]).read_bytes()).hexdigest() for t in tasks},
                "validation_schedule": [{"task_id": t["id"], "order": ["parent", "candidate"] if i % 2 == 0 else ["candidate", "parent"]}
                                        for i, t in enumerate(t for t in tasks if t["split"] == "validation")]}
    write_json(output / "manifest.json", manifest)
    development = []
    if development_from:
        development = reuse_development(development_from, config, tasks, parent)
        manifest["development_reused_from"] = str(development_from.resolve())
        manifest["development_source_commit"] = read_json(development_from / "manifest.json")["root"]["commit"]
        manifest["development_source_sha256"] = hashlib.sha256((development_from / "development.json").read_bytes()).hexdigest()
        write_json(output / "manifest.json", manifest)
        write_json(output / "development.json", development)
    else:
        for task in tasks:
            if task["split"] == "development":
                development.append(episode(config, task, parent, output / "development" / task["id"], manifest["cli"]))
                write_json(output / "development.json", development)
    proposal = proposer.model(
        "You are the Evolution agent. Propose exactly one changed workflow policy and an English hypothesis. "
        "Do not change prompts, tools, tasks, grading, budgets, or models. Only the supplied policy fields are editable. "
        "plan_horizon limits the number of actions requested per planning call; max_decisions caps planning calls; "
        "refill_on_empty permits another plan when the queued actions end; replan_on discards remaining queued actions "
        "after a matching event and replans. If a trigger fires after the decision cap, execution stops and reports. "
        "stop is always terminal. All attempts consume the frozen task budget. "
        "Use only these development outcomes. Do not specialize by task identity or material. "
        "The parent is a strong low-cost batch baseline; do not assume additional feedback is beneficial. "
        "Promotion requires no validation regressions and either a protocol success gain, or equal full success with "
        "at least 10% fewer observed tokens plus fewer model calls or scientific actions. "
        "The proposal must differ from the parent.\n" + json.dumps({"parent": parent, "policy_schema": POLICY_SCHEMA,
            "development": development_feedback(development)}, ensure_ascii=False),
        CANDIDATE_SCHEMA, output / "evolution" / "proposal")
    candidate = proposal["workflow"]
    candidate_hash = policy_hash(candidate)
    write_json(output / "candidate.json", dict(proposal, sha256=candidate_hash))
    before, after = {}, {}
    if candidate_hash == policy_hash(parent):
        decision = {"promoted": False, "reason": "unchanged_policy"}
    else:
        by_id = {t["id"]: t for t in tasks}
        for pair in manifest["validation_schedule"]:
            task = by_id[pair["task_id"]]
            for label in pair["order"]:
                row = episode(config, task, parent if label == "parent" else candidate, output / "validation" / task["id"] / label, manifest["cli"])
                (before if label == "parent" else after)[task["id"]] = row
                write_json(output / "validation.json", {"parent": before, "candidate": after})
        decision = selection(before, after)
    selected = candidate if decision["promoted"] else parent
    write_json(output / "accepted.json", {"workflow": selected, "sha256": policy_hash(selected),
        "source": "candidate" if decision["promoted"] else "parent", "parent_sha256": policy_hash(parent)})
    all_rows = development + list(before.values()) + list(after.values())
    summary = {"decision": decision, "development": development, "validation": {"parent": before, "candidate": after},
               "total_calls": proposer.calls + sum(r["cost"]["calls"] for r in all_rows),
               "total_observed_tokens": proposer.observed_tokens + sum(r["cost"]["observed_tokens"] for r in all_rows),
               "proposal_observed_tokens": proposer.observed_tokens}
    write_json(output / "summary.json", summary)
    return summary


def main():
    from pathlib import Path
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/m4_workflow.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--parent", type=Path)
    parser.add_argument("--development-from", type=Path, help="Revalidate and reuse unchanged development evidence")
    args = parser.parse_args()
    preexisting = args.output.exists()
    try:
        result = run(read_json(args.config), args.output.resolve(), read_json(args.parent) if args.parent else None, args.development_from)
    except Exception as exc:
        if not preexisting and args.output.exists():
            write_json(args.output / "failure.json", {"type": type(exc).__name__, "message": str(exc)})
        raise
    print(json.dumps({k: v for k, v in result.items() if k not in {"development", "validation"}}, indent=2))


if __name__ == "__main__":
    main()

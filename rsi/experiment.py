"""Run a small, versioned, independently graded scientific agent experiment."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PREPARE_STEPS = ["duplicates", "deuterium", "primitive", "disorder", "solvent", "hydrogens"]
from rsi.protocol import REPORT_INSTRUCTIONS, REPORT_PROTOCOL_VERSION, REPORT_SCHEMA

PLAN_SCHEMA = {
    "type": "object",
    "properties": {
        "actions": {
            "type": "array", "maxItems": 4,
            "items": {
                "type": "object",
                "properties": {
                    "verb": {"type": "string", "enum": ["inspect", "prepare", "optimize", "stop"]},
                    "steps": {"type": "array", "items": {"type": "string", "enum": PREPARE_STEPS}},
                    "max_steps": {"type": "integer", "minimum": 0, "maximum": 3},
                },
                "required": ["verb", "steps", "max_steps"], "additionalProperties": False,
            },
        },
        "rationale": {"type": "string"},
    },
    "required": ["actions", "rationale"], "additionalProperties": False,
}
STEP_SCHEMA = {
    "type": "object",
    "properties": {"action": PLAN_SCHEMA["properties"]["actions"]["items"], "rationale": {"type": "string"}},
    "required": ["action", "rationale"], "additionalProperties": False,
}
CANDIDATE_SCHEMA = {
    "type": "object",
    "properties": {
        "prompt": {"type": "string", "minLength": 1, "maxLength": 6000},
        "hypothesis": {"type": "string"},
    },
    "required": ["prompt", "hypothesis"], "additionalProperties": False,
}


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_parent(path):
    """Verify an immutable accepted prompt before using it as the next parent."""
    parent = read_json(path)
    prompt = parent.get("prompt")
    if not isinstance(prompt, str) or not prompt or hashlib.sha256(prompt.encode()).hexdigest() != parent.get("sha256"):
        raise ValueError("Accepted parent prompt is missing or its hash does not match")
    return parent


def git_state(path):
    def git(*args):
        return subprocess.check_output(
            ["git", "-c", f"safe.directory={path.as_posix()}", "-C", str(path), *args],
            text=True, encoding="utf-8", stderr=subprocess.DEVNULL,
        ).strip()
    return {"commit": git("rev-parse", "HEAD"), "status": git("status", "--porcelain")}


def cli_identity(executable):
    """Fingerprint the actual backend executable, not an unrelated PATH entry."""
    path = Path(shutil.which(executable) or executable).resolve(strict=True)
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "version": subprocess.check_output([str(path), "--version"], text=True, timeout=15).strip()}


def public_observation(result):
    """Expose scientific results, never task expectations or reference actions."""
    def compact(value):
        if isinstance(value, dict):
            return {key: compact(item) for key, item in value.items()
                    if key not in {"sites", "site_groups", "trajectory", "cart", "positions"}}
        if isinstance(value, list):
            return [compact(item) for item in value]
        return value
    allowed = ("initial", "final", "initial_inspection", "final_inspection", "events", "errors", "diagnostic_codes", "stopped", "termination")
    observation = {key: compact(result[key]) for key in allowed if key in result}
    artifact = result.get("artifact")
    observation["artifact"] = ({"saved": True, "sha256": artifact["sha256"]}
                               if artifact else {"saved": False})
    return observation


def normalize_actions(actions):
    normalized = []
    for action in actions:
        verb = action["verb"]
        item = {"verb": verb}
        if verb == "prepare":
            item["steps"] = action["steps"]
        if verb == "optimize":
            if not 1 <= action["max_steps"] <= 3:
                raise ValueError("Optimization requires one to three steps")
            item["max_steps"] = action["max_steps"]
        normalized.append(item)
    return normalized


def promotion(baseline, candidate, minimum_gain=1):
    """Missing, failed, or regressed task results cannot win promotion."""
    if not baseline or set(baseline) != set(candidate):
        return {"promoted": False, "reason": "incomplete_comparison"}
    regressions = [key for key in baseline if baseline[key] and not candidate[key]]
    gain = sum(candidate.values()) - sum(baseline.values())
    return {"promoted": not regressions and gain >= minimum_gain,
            "reason": "validation_gain" if not regressions and gain >= minimum_gain else "no_qualified_gain",
            "gain": gain, "regressions": regressions}


class Experiment:
    def __init__(self, config, output, backend=None):
        self.config = config
        self.output = Path(output).resolve()
        self.output.mkdir(parents=True, exist_ok=False)
        self.calls = 0
        self.observed_tokens = 0
        if backend is None:
            from rsi.backends.codex_cli import CodexCLI
            backend = CodexCLI(model=config["model"], reasoning=config["reasoning"], timeout=config["cli_timeout_seconds"])
        self.backend = backend
        self.ledger = []
        write_json(self.output / "config.json", config)

    def model(self, prompt, schema, directory):
        if self.calls >= self.config["max_llm_calls"] or self.observed_tokens >= self.config["max_observed_tokens"]:
            raise RuntimeError("Model budget exhausted before next invocation")
        self.calls += 1
        # Persist the reservation before invoking the process, including failed calls.
        entry = {"call": self.calls, "path": str(directory.relative_to(self.output)), "status": "reserved"}
        self.ledger.append(entry)
        write_json(self.output / "ledger.json", self.ledger)
        result = self.backend.run(
            "Return only the requested JSON. Do not use any shell, file, web, MCP, or other tools. "
            "All scientific tools are executed by an external controller.\n" + prompt,
            schema, directory,
        )
        usage = result.get("usage") or {}
        tokens = usage.get("input_tokens", 0) + usage.get("output_tokens", 0)
        self.observed_tokens += tokens
        entry.update(status=result["status"], usage=usage,
                     duration_seconds=result.get("duration_seconds"), cumulative_observed_tokens=self.observed_tokens)
        write_json(self.output / "ledger.json", self.ledger)
        if result["status"] != "ok":
            raise RuntimeError(f"CLI failure: {result['status']}; see {directory}")
        return result["response"]

    def science(self, task, actions, directory, previous=None):
        directory.mkdir(parents=True, exist_ok=False)
        write_json(directory / "task.json", task)
        write_json(directory / "actions.json", actions)
        env = dict(os.environ, PYTHONUTF8="1", OPENBLAS_NUM_THREADS="1")
        if "science_seed" in self.config:
            env["PYTHONHASHSEED"] = str(self.config["science_seed"])
            env["RSI_SCIENCE_SEED"] = str(self.config["science_seed"])
        argv = [sys.executable, "-m", "rsi.worker", str(directory / "task.json"),
                str(directory / "actions.json"), str(directory)]
        if previous is not None:
            write_json(directory / "previous.json", previous)
            argv.append(str(directory / "previous.json"))
        with (directory / "worker.log").open("w", encoding="utf-8") as log:
            process = subprocess.Popen(argv, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
            try:
                code = process.wait(timeout=self.config["science_timeout_seconds"])
            except subprocess.TimeoutExpired:
                import psutil
                try:
                    parent = psutil.Process(process.pid)
                    children = parent.children(recursive=True)
                    for child in children:
                        try:
                            child.kill()
                        except psutil.NoSuchProcess:
                            pass
                    parent.kill()
                except psutil.NoSuchProcess:
                    pass
                process.wait(timeout=10)
                raise RuntimeError(f"Scientific worker timeout: {directory}") from None
        if code:
            raise RuntimeError(f"Scientific worker exit {code}: {directory}")
        return read_json(directory / "result.json")

    def research(self, task, prompt, directory):
        if self.config.get("workflow", "batch") == "stepwise":
            return self.research_stepwise(task, prompt, directory)
        return self.research_batch(task, prompt, directory)

    def research_stepwise(self, task, prompt, directory):
        from rsi.evaluation import evaluate
        from rsi.science.adapter import validate_action

        result = self.science(task, [], directory / "initial")
        decisions = []
        actions = []
        limit = min(self.config["max_actions_per_task"], task.get("action_budget", 8), 8)
        if type(limit) is not int or limit < 1:
            raise ValueError("Task action budget must be a positive integer")
        termination = "action_budget"
        for index in range(limit):
            if result["errors"] or not result["artifact"]:
                termination = "scientific_failure"
                break
            # Leave one model call for the final evidence-based report.
            if self.calls >= self.config["max_llm_calls"] - 1:
                termination = "model_call_budget"
                break
            remaining = 3 - sum(a.get("max_steps", 0) for a in actions)
            decision = self.model(
                prompt + "\nChoose exactly one next action using the latest observation and prior outcomes. "
                "The controller resumes the saved structure; previous actions are not replayed. "
                "inspect reads; prepare applies explicit permitted steps; optimize runs UFF; stop ends the episode. "
                "Use steps=[] unless preparing and max_steps=0 unless optimizing. "
                "Failed attempts consume an action and their requested optimization budget. "
                "Never retry a nonretryable failure blindly. Stop when obligations are fulfilled or no useful "
                "authorized action remains.\n" + json.dumps({"instruction": task["instruction"],
                    "observation": public_observation(result), "prior_decisions": decisions,
                    "remaining_actions": limit - index, "remaining_optimization_steps": remaining}, ensure_ascii=False),
                STEP_SCHEMA, directory / f"decision-{index:02d}",
            )
            decisions.append(decision)
            action = normalize_actions([decision["action"]])[0]
            validate_action(action)
            if action.get("max_steps", 0) > remaining:
                termination = "invalid_budget_request"
                write_json(directory / "rejected-action.json", decision)
                break
            actions.append(action)
            result = self.science(task, [action], directory / f"step-{index:02d}", previous=result)
            write_json(directory / "trajectory.json", {"decisions": decisions, "result": result})
            if result["stopped"]:
                termination = "agent_stop"
                break
        result["termination"] = termination
        write_json(directory / "final-state.json", result)
        report = self.model(prompt + "\nProduce a final report; no further actions are possible.\n"
            + REPORT_INSTRUCTIONS + "\n" + json.dumps({"instruction": task["instruction"],
                "observation": public_observation(result)}, ensure_ascii=False),
            REPORT_SCHEMA, directory / "report")
        verdict = evaluate(task, result, report)
        verdict["checks"]["controller_budget_compliant"] = termination != "invalid_budget_request"
        verdict["passed"] = all(verdict["checks"].values())
        row = {"task_id": task["id"], "split": task["split"], "workflow": "stepwise",
               "plan": {"actions": actions, "decisions": decisions}, "termination": termination,
               "report": report, "verdict": verdict}
        write_json(directory / "graded.json", row)
        return row

    def research_batch(self, task, prompt, directory):
        from rsi.evaluation import evaluate

        initial = self.science(task, [], directory / "initial")
        context = {"instruction": task["instruction"], "observation": public_observation(initial)}
        plan = self.model(
            prompt + "\nChoose up to four actions. inspect reads the structure; prepare executes explicitly selected "
            "steps; optimize runs UFF for 1-3 steps; stop ends the sequence. All work starts from the observed "
            "structure, which has already been inspected. Total optimization budget is three steps. "
            "For non-prepare actions use steps=[]; for non-optimize actions use max_steps=0. "
            "Do not invoke tools yourself.\n" + json.dumps(context, ensure_ascii=False),
            PLAN_SCHEMA, directory / "plan",
        )
        actions = normalize_actions(plan["actions"])
        if len(actions) > self.config["max_actions_per_task"]:
            raise ValueError("Too many scientific actions")
        result = self.science(task, actions, directory / "execution")
        report = self.model(
            prompt + "\nProduce the final report, using the observed tool results. No further actions are possible.\n"
            + REPORT_INSTRUCTIONS + "\nList actual diagnostic codes for important initial problems and remaining issues; never invent codes. "
            + json.dumps({"instruction": task["instruction"], "actions": actions,
                          "observation": public_observation(result)}, ensure_ascii=False),
            REPORT_SCHEMA, directory / "report",
        )
        verdict = evaluate(task, result, report)
        row = {"task_id": task["id"], "split": task["split"], "plan": plan,
               "report": report, "verdict": verdict}
        write_json(directory / "graded.json", row)
        return row

    def run(self, tasks, mode):
        from rsi.evaluation import evaluate

        manifest = {"started_utc": datetime.now(timezone.utc).isoformat(), "mode": mode,
                    "root": git_state(ROOT), "crystal_builder": git_state(ROOT / "Crystal-Builder"),
                    "tasks_sha256": hashlib.sha256(json.dumps(tasks, sort_keys=True).encode()).hexdigest(),
                    "python": sys.version, "model_requested": self.config["model"],
                    "evaluation_scope": "public_engineering_smoke_not_hidden_generalization",
                    "report_protocol_version": REPORT_PROTOCOL_VERSION}
        packages = ["crystal-builder", "numpy", "scipy", "gemmi", "spglib", "ase", "jsonschema", "psutil"]
        manifest["packages"] = {name: importlib.metadata.version(name) for name in packages}
        manifest["cli"] = cli_identity(self.backend.executable)
        manifest["cli_version"] = manifest["cli"]["version"]
        manifest["input_hashes"] = {
            task["initialization"]["path"]: hashlib.sha256(
                (ROOT / "Crystal-Builder" / task["initialization"]["path"]).read_bytes()).hexdigest()
            for task in tasks if task.get("initialization", {}).get("kind") == "open"
        }
        write_json(self.output / "manifest.json", manifest)
        if mode == "probe":
            response = self.model("Return task_completed=true, scientific_status=not_assessed, diagnostic_codes=[], summary='M0_READY'.",
                                  REPORT_SCHEMA, self.output / "probe")
            write_json(self.output / "summary.json", {"probe": response})
            return
        if mode == "replay":
            rows = []
            for task in tasks:
                if self.config.get("workflow") == "stepwise" or self.config.get("checkpointed_replay", False):
                    result = self.science(task, [], self.output / task["id"] / "initial")
                    for index, action in enumerate(task["reference_actions"]):
                        result = self.science(task, [action], self.output / task["id"] / f"step-{index:02d}", previous=result)
                else:
                    result = self.science(task, task["reference_actions"], self.output / task["id"])
                if self.config.get("workflow") == "stepwise" or self.config.get("checkpointed_replay", False):
                    limit = min(self.config["max_actions_per_task"], task.get("action_budget", 8), 8)
                    result["termination"] = "agent_stop" if result["stopped"] else "action_budget" if len(task["reference_actions"]) >= limit else "reference_end"
                report = task["reference_report"]
                rows.append({"task_id": task["id"], "verdict": evaluate(task, result, report)})
                write_json(self.output / "summary.json", {"replay": rows})
            if not all(row["verdict"]["passed"] for row in rows):
                raise RuntimeError("Reference replay did not pass every scientific check")
            return
        baseline = []
        for task in tasks:
            baseline.append(self.research(task, self.config["baseline_prompt"], self.output / "baseline" / task["id"]))
            write_json(self.output / "summary.json", {"baseline": baseline, "calls": self.calls, "observed_tokens": self.observed_tokens})
        if mode == "baseline":
            return
        development = [{"task_id": row["task_id"], "plan": row["plan"], "report": row["report"],
                        "checks": row["verdict"]["checks"], "passed": row["verdict"]["passed"]}
                       for row in baseline if row["split"] == "development"]
        candidate = self.model(
            "You are the Evolution agent. Propose exactly one replacement research system prompt, not code. "
            "Improve general scientific decision making using only these development outcomes. Do not embed "
            "task IDs, material-specific answers, or instructions to change evaluation. The next agent has only "
            "inspect/prepare/optimize/stop. Keep the prompt below 6000 characters. Write the prompt in English.\n"
            + REPORT_INSTRUCTIONS + "\n"
            + json.dumps({"parent_prompt": self.config["baseline_prompt"], "development": development}, ensure_ascii=False),
            CANDIDATE_SCHEMA, self.output / "evolution",
        )
        write_json(self.output / "candidate.json", candidate)
        candidate_rows = []
        for task in tasks:
            if task["split"] == "validation":
                candidate_rows.append(self.research(task, candidate["prompt"], self.output / "candidate" / task["id"]))
        before = {row["task_id"]: row["verdict"]["passed"] for row in baseline if row["split"] == "validation"}
        after = {row["task_id"]: row["verdict"]["passed"] for row in candidate_rows}
        decision = promotion(before, after, self.config["promotion"]["minimum_validation_gain"])
        selected = candidate["prompt"] if decision["promoted"] else self.config["baseline_prompt"]
        write_json(self.output / "accepted.json", {"source": "candidate" if decision["promoted"] else "baseline",
                                                  "prompt": selected, "sha256": hashlib.sha256(selected.encode()).hexdigest()})
        write_json(self.output / "summary.json", {"baseline": baseline, "candidate": candidate_rows,
                                                  "decision": decision, "calls": self.calls,
                                                  "observed_tokens": self.observed_tokens})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["probe", "replay", "baseline", "cycle"])
    parser.add_argument("--config", type=Path, default=ROOT / "configs/m0_astra.json")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--parent", type=Path, help="Verified accepted.json from a previous run")
    args = parser.parse_args()
    config = read_json(args.config)
    if args.parent:
        parent = load_parent(args.parent)
        config["baseline_prompt"] = parent["prompt"]
        config["parent"] = {"path": str(args.parent.resolve()), "sha256": parent["sha256"]}
    tasks = read_json(ROOT / config.get("tasks_path", "benchmarks/smoke/tasks.json"))["tasks"] if args.mode != "probe" else []
    output = args.output or ROOT / "runs" / (datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + args.mode)
    experiment = Experiment(config, output)
    try:
        experiment.run(tasks, args.mode)
    except Exception as exc:
        write_json(experiment.output / "failure.json", {"type": type(exc).__name__, "message": str(exc)})
        raise
    print(json.dumps({"status": "completed", "output": str(experiment.output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()

"""Approved public four-arm pilot; fixed science, workflow and reference scorer."""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from rsi.backends.codex_cli import CodexCLI
from rsi.experiment import ROOT, Experiment, cli_identity, git_state, read_json, write_json
from rsi.q2_budget import RunBudget, BudgetStop
from rsi.q2_model_protocol import AUTHOR_INSTRUCTION, AUTHOR_SCHEMA, PLAN_SCHEMA, FINAL_SCHEMA, RESEARCH_TRANSPORT
from rsi.tool_evidence import from_uff_event
from rsi.tool_runtime import Registry, digest, load_spec
from rsi.tool_trial import Ledger, episode

ARMS = ("fixed", "outer_generated", "manual_control", "removed")
FROZEN = ["rsi/tool_trial.py", "rsi/tool_runtime.py", "rsi/tool_validation.py", "rsi/tool_evidence.py",
          "rsi/science/adapter.py", "rsi/evaluation/evaluator.py", "rsi/m6_evaluation.py", "rsi/workflow.py",
          "rsi/backends/codex_cli.py", "rsi/q2_model_protocol.py", "rsi/q2_budget.py", "rsi/q2_astra.py",
          "tools/q2/manual-audit-v1.json", "configs/q2_tool_astra.json"]


def fingerprints():
    return {path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest() for path in FROZEN}


def science_recipes():
    """Preserve prior chemistry authorization, including no optimization of open trimers."""
    tasks = read_json(ROOT / "benchmarks/smoke/tasks.json")["tasks"]
    identities = [("normal-mof5", "development"), ("copies-and-hydrogens", "development"),
                  ("short-relax", "evaluation"), ("open-trimers", "evaluation")]
    recipes = []
    for task_id, split in identities:
        original = next(t for t in tasks if t["id"] == task_id)
        task = {"id": "q2-" + task_id, "initialization": original["initialization"]}
        actions = [{"verb": "inspect"}]
        actions += [a for a in original["reference_actions"] if a["verb"] == "prepare"]
        blocked = task_id == "open-trimers"
        if not blocked:
            actions.append({"verb": "optimize", "max_steps": 1})
        actions += [{"verb": "inspect"}, {"verb": "stop"}]
        recipes.append(dict(task=task, actions=actions, split=split,
                            source_family=original["source_family"],
                            predeclared_exclusion="unresolved_open_trimer_chemistry" if blocked else None))
    return recipes


def make_tasks(science_records):
    tasks = []
    for record in science_records:
        for units in (["native"] if record["split"] == "development" else ["native", "SI"]):
            task = {key: record[key] for key in ("source_family", "split", "eligible", "reason", "evidence")}
            task.update(id=record["task_id"] + "-" + units, unit_condition=units,
                        payload=copy.deepcopy(record["payload"]))
            if task["eligible"] and units == "SI":
                for key in ("initial_energy", "energy"):
                    task["payload"][key] *= 4.184
                task["payload"]["max_force"] *= 41.84
                task["payload"].update(energy_unit="kJ/mol", force_unit="kJ/mol/nm")
            tasks.append(task)
    return tasks


def summarize(rows):
    summary = {}
    for arm in ARMS:
        arm_rows = [r for r in rows if r["arm"] == arm]
        eligible = [r for r in arm_rows if r["eligible"]]
        completed = [r for r in arm_rows if r["completed"]]
        evaluation = [r for r in eligible if r["split"] == "evaluation"]
        summary[arm] = dict(tasks=len(arm_rows), eligible=len(eligible),
                            completed=len(completed), correct=sum(r["passed"] for r in arm_rows),
                            incorrect_reports=sum(not r["passed"] for r in completed),
                            evaluation_correct=sum(r["passed"] for r in evaluation),
                            evaluation_eligible=len(evaluation),
                            tool_requests=sum(len(r.get("tool_trace", [])) for r in arm_rows),
                            tool_errors=sum(not t["answer"]["ok"] for r in arm_rows for t in r.get("tool_trace", [])),
                            registered_tool_uses=sum(r.get("generated_or_manual_tool_uses", 0) for r in arm_rows),
                            reuse_sources=sorted({r["source_family"] for r in arm_rows
                                                 if r.get("generated_or_manual_tool_uses", 0)}),
                            model_calls=sum(r["cost"]["calls"] for r in arm_rows),
                            tokens=None if any(r["cost"]["tokens"] is None for r in arm_rows)
                            else sum(r["cost"]["tokens"] for r in arm_rows),
                            wall_seconds=sum(r["cost"]["wall_seconds"] or 0 for r in arm_rows))
    pairs = {}
    for comparison in ARMS[1:]:
        counts = dict(wins=0, losses=0, ties=0, missing=0)
        for task_id in {r["task_id"] for r in rows if r["eligible"]}:
            baseline = next((r for r in rows if r["task_id"] == task_id and r["arm"] == "fixed"), None)
            other = next((r for r in rows if r["task_id"] == task_id and r["arm"] == comparison), None)
            if baseline is None or other is None or not baseline["cost"]["calls"] or not other["cost"]["calls"]:
                counts["missing"] += 1
            else:
                counts["wins" if other["passed"] > baseline["passed"] else
                       "losses" if other["passed"] < baseline["passed"] else "ties"] += 1
        pairs[comparison] = counts
    return {"arms": summary, "paired_vs_fixed": pairs}


class Pilot:
    def __init__(self, output, executable=None):
        self.output = Path(output).resolve()
        self.output.mkdir(parents=True, exist_ok=False)
        self.config = read_json(ROOT / "configs/q2_tool_astra.json")
        if self.config["status"] != "approved":
            raise ValueError("Experiment is not approved")
        self.budget = RunBudget(self.output / "budget.json", self.config["budget"])
        self.rows, self.overhead = [], []
        self.backend = CodexCLI("gpt-6-astra", "high", self.config["budget"]["model_timeout_seconds"], executable)
        self.frozen = fingerprints()
        self.manifest = dict(root_git=git_state(ROOT), crystal_builder_git=git_state(ROOT / "Crystal-Builder"),
                             config=self.config, frozen_files=self.frozen, recipes=science_recipes(),
                             python=sys.version, python_executable=sys.executable,
                             packages={p: importlib.metadata.version(p) for p in ("numpy", "scipy", "jsonschema")},
                             cli=cli_identity(self.backend.executable),
                             requested_model="gpt-6-astra", actual_model=None, reasoning="high",
                             science_seed=0, visibility="public_engineering_validation")
        if self.manifest["root_git"]["status"] or self.manifest["crystal_builder_git"]["status"]:
            raise ValueError("Start only from clean frozen repositories")
        self.source_hashes = {}
        for recipe in self.manifest["recipes"]:
            source = ROOT / "Crystal-Builder" / recipe["task"]["initialization"]["path"]
            self.source_hashes[str(source)] = hashlib.sha256(source.read_bytes()).hexdigest()
        self.manifest["source_files"] = self.source_hashes
        write_json(self.output / "manifest.json", self.manifest)

    def verify_frozen(self):
        if fingerprints() != self.frozen:
            raise BudgetStop("Frozen implementation changed")
        if any(hashlib.sha256(Path(path).read_bytes()).hexdigest() != sha for path, sha in self.source_hashes.items()):
            raise BudgetStop("Frozen source structure changed")

    def model(self, stage, label, prompt, schema):
        self.verify_frozen()
        row = self.budget.reserve_model(stage, label)
        started = time.perf_counter()
        try:
            result = self.backend.run(prompt, schema, self.output / "model" / label)
        except Exception as exc:
            result = dict(status="controller_error", usage=None, duration_seconds=time.perf_counter() - started,
                          requested_model="gpt-6-astra", actual_model=None, response=None, error=str(exc))
            write_json(self.output / "model" / label / "controller-error.json", result)
        tokens = self.budget.finish_model(row, result)
        if result["status"] in {"unexpected_tool_use", "auth_error", "model_unavailable"}:
            self.budget.blocked = result["status"]
            self.budget.save()
        print(json.dumps(dict(label=label, status=result["status"], tokens=tokens,
                              blocked=self.budget.blocked)), flush=True)
        return result, tokens

    def science(self):
        records = []
        worker = object.__new__(Experiment)
        worker.config = dict(science_timeout_seconds=self.config["budget"]["science_timeout_seconds"], science_seed=0)
        # Prefer the independent scientific checkout over the installed editable source.
        os.environ["PYTHONPATH"] = str(ROOT / "Crystal-Builder")
        for recipe in self.manifest["recipes"]:
            self.verify_frozen()
            task, actions = recipe["task"], recipe["actions"]
            row = self.budget.reserve_science(task["id"], sum(a.get("max_steps", 0) for a in actions))
            started = time.perf_counter()
            result = None
            try:
                result = worker.science(task, actions, self.output / "science" / task["id"])
                row.update(status="error" if result["errors"] else "ok", errors=result["errors"])
            except Exception as exc:
                row.update(status="error", errors=[str(exc)])
            row["wall_seconds"] = time.perf_counter() - started
            self.budget.save()
            record = dict(task_id=task["id"], source_family=recipe["source_family"], split=recipe["split"],
                          eligible=False, payload=None, evidence=None, reason=recipe["predeclared_exclusion"])
            if result:
                record["initial"] = result["initial"]
                record["final"] = result["final"]
                record["diagnostic_codes"] = result["diagnostic_codes"]
                record["result_sha256"] = digest(result)
                events = [e for e in result["events"] if e["action"]["verb"] == "optimize" and e["result"].get("ok") is True]
                if not result["errors"] and events and not record["reason"]:
                    evidence = from_uff_event(events[-1])
                    record.update(eligible=True, payload=evidence["payload"], evidence=evidence)
                elif not record["reason"]:
                    record["reason"] = "No successful trusted UFF event"
            else:
                record["reason"] = "Scientific worker failed"
            records.append(record)
            write_json(self.output / "science-records.json", records)
            print(json.dumps(dict(science=task["id"], eligible=record["eligible"], reason=record["reason"])), flush=True)
        tasks = make_tasks(records)
        write_json(self.output / "tasks.json", tasks)
        self.manifest["task_manifest_sha256"] = digest(tasks)
        write_json(self.output / "manifest.json", self.manifest)
        return tasks

    def author(self, tasks):
        development = [{"source_family": t["source_family"], "payload": t["payload"]}
                       for t in tasks if t["split"] == "development" and t["eligible"]]
        feedback = None
        for attempt in range(1, self.config["budget"]["generation_and_debug_calls"] + 1):
            prompt = AUTHOR_INSTRUCTION + "\n" + json.dumps(dict(development=development, feedback=feedback))
            result, _ = self.model("author", f"author-{attempt}", prompt, AUTHOR_SCHEMA)
            if result["status"] != "ok":
                feedback = {"status": result["status"]}
                continue
            spec = result["response"]["spec"]
            started = time.perf_counter()
            try:
                registry = Registry()
                record = registry.register(spec)
                feedback = {"accepted_for_evaluation": True}
            except Exception as exc:
                record = None
                feedback = {"accepted_for_evaluation": False, "error": str(exc)}
            self.overhead.append(dict(stage="candidate_validation", attempt=attempt,
                                      wall_seconds=time.perf_counter() - started, verdict=feedback))
            write_json(self.output / "overhead.json", self.overhead)
            write_json(self.output / f"author-validation-{attempt}.json", feedback)
            if record is not None:
                write_json(self.output / "generated-tool.json", spec)
                write_json(self.output / "generated-registration.json", record)
                return spec
        raise BudgetStop("No independently valid candidate within author budget")

    def research(self, tasks, generated):
        manual = load_spec(ROOT / "tools/q2/manual-audit-v1.json")
        for index, task in enumerate(tasks):
            order = ARMS[index % 4:] + ARMS[:index % 4]
            for arm in order:
                if not task["eligible"]:
                    self.rows.append(dict(task_id=task["id"], source_family=task["source_family"], split=task["split"],
                                          arm=arm, eligible=False, reason=task["reason"], completed=False, passed=False,
                                          cost=dict(calls=0, tokens=0, wall_seconds=0), tool_trace=[]))
                    write_json(self.output / "rows.json", self.rows)
                    continue
                self.budget.check("research")
                spec = manual if arm == "manual_control" else generated
                runtime_arm = "augmented" if arm in {"manual_control", "outer_generated"} else arm
                ledger = Ledger()
                before = len(self.budget.entries)
                def decide(stage, context):
                    schema = PLAN_SCHEMA if stage == "plan" else FINAL_SCHEMA
                    prompt = RESEARCH_TRANSPORT + "\nStage: " + stage + "\n" + json.dumps(context, sort_keys=True)
                    result, tokens = self.model("research", task["id"] + "-" + arm + "-" + stage, prompt, schema)
                    response = result["response"] if result["status"] == "ok" else None
                    if stage == "plan" and response is not None:
                        response = response["requests"]
                    return dict(response=response, calls=1, tokens=tokens, wall_seconds=result["duration_seconds"])
                row = episode(task["payload"], spec, runtime_arm, decide, ledger)
                entries = self.budget.entries[before:]
                row["workflow_ledger"] = ledger.entries
                # Persist actual reservations, not frozen workflow's conservative exception estimate.
                row["cost"] = dict(calls=len(entries), tokens=None if any(e["tokens"] is None for e in entries)
                                   else sum(e["tokens"] for e in entries),
                                   wall_seconds=sum(e["wall_seconds"] or 0 for e in entries))
                row.update(arm=arm, task_id=task["id"], source_family=task["source_family"], split=task["split"],
                           eligible=True, unit_condition=task["unit_condition"])
                self.rows.append(row)
                write_json(self.output / "rows.json", self.rows)
                print(json.dumps(dict(task=task["id"], arm=arm, passed=row["passed"], cost=row["cost"])), flush=True)

    def run(self):
        started = time.perf_counter()
        status, error = "completed", None
        tasks = []
        try:
            tasks = self.science()
            candidate = self.author(tasks)
            self.research(tasks, candidate)
            self.verify_frozen()
        except Exception as exc:
            status, error = "stopped", str(exc)
        for task in tasks:
            for arm in ARMS:
                if not any(r["task_id"] == task["id"] and r["arm"] == arm for r in self.rows):
                    self.rows.append(dict(task_id=task["id"], source_family=task["source_family"], split=task["split"],
                                          arm=arm, eligible=task["eligible"], completed=False, passed=False,
                                          reason=task["reason"] or "Not run: " + str(error),
                                          cost=dict(calls=0, tokens=0, wall_seconds=0), tool_trace=[]))
        write_json(self.output / "rows.json", self.rows)
        model_rows = [e for e in self.budget.entries if e["kind"] == "model"]
        science_rows = [e for e in self.budget.entries if e["kind"] == "science"]
        report = dict(status=status, error=error, visibility="public_engineering_validation",
                      summary=summarize(self.rows), total_model_calls=len(model_rows),
                      tokens=None if any(e.get("tokens") is None for e in model_rows)
                      else sum(e["tokens"] for e in model_rows),
                      science_trajectories=len(science_rows), requested_uff_steps=sum(e["requested_steps"] for e in science_rows),
                      wall_seconds=time.perf_counter() - started, author_validation=self.overhead,
                      requested_model="gpt-6-astra", observed_model_snapshots=sorted({e["actual_model"] for e in model_rows if e.get("actual_model")}),
                      monetary_cost=None, infrastructure_authoring_cost=None,
                      note="All experimental calls and failures charged. Cached input included once in input tokens. Infrastructure authoring and monetary cost unavailable.",
                      root_git=self.manifest["root_git"], crystal_builder_git=self.manifest["crystal_builder_git"])
        write_json(self.output / "summary.json", report)
        print(json.dumps(report, indent=2), flush=True)
        return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cli")
    args = parser.parse_args()
    result = Pilot(args.output, args.cli).run()
    if result["status"] != "completed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()

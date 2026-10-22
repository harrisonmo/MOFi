"""A minimal agentic loop: a natural-language task, model-chosen hub tool calls, and a final report.

The model never executes anything. Each step is one stateless structured decision through any backend
with the ``run(prompt, schema, output_dir)`` contract (Codex CLI, OpenAI or Anthropic API): either one
tool call, given as a tool name and a JSON argument object, or the final report. The controller parses
the arguments, forwards them to ``ScienceToolHub.call`` (which owns bounds, budget and evidence) and
returns the result envelope to the model on the next step. Step, token and observation limits are
controller settings; the final report may cite only call IDs of successful tool calls from this run.

Model calls run in a scratch directory outside any repository: the Codex CLI reads ``AGENTS.md`` files
from its working directory up to the repository root, so a call inside this workspace would carry the
project's own agent instructions into the decision. Each call's records are copied into the run
directory afterwards and the scratch directory is removed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import time

from rsi.hpc.core import read_json, sha256, write_json
from rsi.science.hub import ROOT, ScienceToolHub


ACTIONS = ("call_tool", "final_report")
AGENT_KEYS = {"schema_version", "backend", "model", "reasoning", "cli_timeout_seconds", "cli_executable", "api",
              "model_scratch",
              "max_steps", "max_total_tokens", "max_observation_chars"}
INSTRUCTIONS = """You are a computational materials research agent. Answer the TASK below using the
scientific tools in the CATALOG. You cannot run anything yourself: each reply is one action that the
controller executes, and the controller shows you every result on the next step.

Reply with exactly one action:
- action "call_tool": set "tool" to a tool name from the CATALOG and "arguments_json" to a JSON object
  string with every parameter of that tool (use null for unused nullable parameters; respect enums and
  the "Allowed" ranges). Set "answer", "evidence_call_ids" and "caveats" to null.
- action "final_report": set "tool" and "arguments_json" to null. "answer" states the result with
  numbers and units, "evidence_call_ids" lists the call_id values of successful tool results that
  support it, and "caveats" lists what the calculations do not establish.

Choose the tool and settings that fit the task, spend budget only where it matters, read convergence
fields and errors, and fix and retry a rejected call when that is worthwhile. Never invent numbers that
no tool result reported."""


def step_schema(tool_names):
    nullable = {"type": ["string", "null"]}
    return {"type": "object", "additionalProperties": False,
            "required": ["reasoning", "action", "tool", "arguments_json", "answer", "evidence_call_ids",
                         "caveats"],
            "properties": {"reasoning": {"type": "string"},
                           "action": {"type": "string", "enum": list(ACTIONS)},
                           "tool": {"type": ["string", "null"], "enum": sorted(tool_names) + [None]},
                           "arguments_json": nullable, "answer": nullable,
                           "evidence_call_ids": {"type": ["array", "null"], "items": {"type": "string"}},
                           "caveats": {"type": ["array", "null"], "items": {"type": "string"}}}}


def validate_agent_config(config):
    if not isinstance(config, dict) or set(config) - AGENT_KEYS or {"schema_version", "backend", "model",
                                                                    "max_steps"} - set(config):
        raise ValueError(f"Agent config keys must be within {sorted(AGENT_KEYS)}")
    if config["schema_version"] != 1:
        raise ValueError("Unsupported agent config schema")
    for key in ("max_steps", "max_total_tokens", "max_observation_chars"):
        value = config.get(key)
        if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 1):
            raise ValueError(f"{key} must be a positive integer or null")


def repository(path):
    path = Path(path).resolve()

    def git(*args):
        return subprocess.check_output(["git", "-c", f"safe.directory={path.as_posix()}", "-C", str(path), *args],
                                       text=True, stderr=subprocess.DEVNULL).strip()
    try:
        return {"commit": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--porcelain"))}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}


def isolated_scratch(path=None):
    """Resolve a scratch root that neither lies in a repository nor sees an ``AGENTS.md`` above it."""
    root = Path(path or tempfile.gettempdir()).resolve()
    for folder in (root, *root.parents):
        found = [name for name in ("AGENTS.md", ".git") if (folder / name).exists()]
        if found:
            raise ValueError(f"Model scratch {root} is not isolated: {folder / found[0]} exists")
    root.mkdir(parents=True, exist_ok=True)
    return root


def _tokens(usage):
    usage = usage or {}
    return (usage.get("input_tokens") or 0) + (usage.get("output_tokens") or 0)


def _clip(value, limit):
    text = json.dumps(value, ensure_ascii=False, allow_nan=False, default=str)
    return text if len(text) <= limit else text[:limit] + f" ... [truncated {len(text) - limit} characters]"


class ScienceAgent:
    """Owned by a controller: the backend, hub and limits are never model inputs."""

    def __init__(self, backend, hub, run_root, max_steps=8, max_total_tokens=None, max_observation_chars=6000,
                 model_scratch=None):
        self.backend = backend
        self.hub = hub
        self.run_root = Path(run_root).resolve()
        self.max_steps = max_steps
        self.max_total_tokens = max_total_tokens
        self.max_observation_chars = max_observation_chars
        self.model_scratch = isolated_scratch(model_scratch)

    def prompt(self, task, history, steps_left):
        catalog = self.hub.catalog()
        visible = {key: catalog[key] for key in ("structures", "force_fields", "openmm_systems",
                                                 "unavailable_tools", "guidance", "budget")}
        visible["tools"] = [t["function"] for t in catalog["tools"]]
        lines = [INSTRUCTIONS, "", "TASK:", task.strip(), "", "CATALOG:", json.dumps(visible, indent=1), "",
                 "HISTORY:"]
        if not history:
            lines.append("(no actions yet)")
        for item in history:
            lines.append(f"Step {item['step']}: {item['action']} {item.get('tool') or ''} "
                         f"arguments={item.get('arguments_json')}")
            lines.append(f"Result: {_clip(item['observation'], self.max_observation_chars)}")
        lines += ["", f"Steps left including this one: {steps_left}."]
        if steps_left == 1:
            lines.append("This is the last step: return action final_report now.")
        return "\n".join(lines)

    def run(self, task):
        self.run_root.mkdir(parents=True, exist_ok=True)
        if (self.run_root / "agent-run.json").exists():
            raise ValueError(f"Agent run already exists: {self.run_root}")
        schema = step_schema(t["function"]["name"] for t in self.hub.catalog()["tools"])
        record = {"schema_version": 1, "task": task, "status": "running", "started_at": time.time(),
                  "limits": {"max_steps": self.max_steps, "max_total_tokens": self.max_total_tokens,
                             "max_observation_chars": self.max_observation_chars},
                  "hub_session": str(self.hub.session_root), "model_scratch": str(self.model_scratch), "steps": [], "final": None,
                  "usage": {"model_calls": 0, "tokens": 0}}
        history = []
        for index in range(1, self.max_steps + 1):
            prompt = self.prompt(task, history, self.max_steps - index + 1)
            result = self._model_call(prompt, schema, self.run_root / "model" / f"step-{index:02d}")
            record["usage"]["model_calls"] += 1
            record["usage"]["tokens"] += _tokens(result.get("usage"))
            step = {"step": index, "model_status": result["status"], "usage": result.get("usage"),
                    "actual_model": result.get("actual_model"),
                    "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest()}
            record["steps"].append(step)
            if result["status"] != "ok":
                record["status"] = "model_error"
                break
            reply = result["response"]
            step.update(reasoning=reply["reasoning"], action=reply["action"])
            if reply["action"] == "final_report":
                successful = {s["observation"].get("call_id") for s in record["steps"]
                              if isinstance(s.get("observation"), dict) and s["observation"].get("ok")} - {None}
                cited = reply["evidence_call_ids"] or []
                record["final"] = {"answer": reply["answer"], "evidence_call_ids": cited,
                                   "caveats": reply["caveats"] or [],
                                   "unsupported_citations": sorted(set(cited) - successful)}
                complete = bool(reply["answer"]) and bool(cited) and not record["final"]["unsupported_citations"]
                record["status"] = "completed" if complete else "invalid_report"
                break
            step.update(tool=reply["tool"], arguments_json=reply["arguments_json"])
            step["observation"] = self._execute(reply["tool"], reply["arguments_json"])
            history.append(step)
            if self.max_total_tokens and record["usage"]["tokens"] >= self.max_total_tokens:
                record["status"] = "token_limit"
                break
            write_json(self.run_root / "agent-run.json", record)
        else:
            record["status"] = "step_limit"
        record["finished_at"] = time.time()
        record["hub_budget"] = self.hub.budget_status()
        write_json(self.run_root / "agent-run.json", record)
        return record

    def _model_call(self, prompt, schema, target):
        """Run one decision in a fresh isolated scratch directory and keep its records in ``target``."""
        scratch = Path(tempfile.mkdtemp(prefix="science-agent-", dir=self.model_scratch))
        try:
            result = self.backend.run(prompt, schema, scratch / "call")
            if (scratch / "call").exists():
                shutil.copytree(scratch / "call", target)
            return result
        finally:
            shutil.rmtree(scratch, ignore_errors=True)

    def _execute(self, tool, arguments_json):
        """Turn one model action into a hub envelope; malformed actions never reach the hub."""
        if tool is None or arguments_json is None:
            return {"ok": False, "call_id": None, "error": {"type": "ActionError",
                    "message": "call_tool needs both tool and arguments_json"}}
        try:
            arguments = json.loads(arguments_json)
        except ValueError as exc:
            return {"ok": False, "call_id": None, "error": {"type": "ActionError",
                    "message": f"arguments_json is not valid JSON: {exc}"}}
        if not isinstance(arguments, dict):
            return {"ok": False, "call_id": None, "error": {"type": "ActionError",
                    "message": "arguments_json must encode a JSON object"}}
        return self.hub.call(tool, arguments)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Answer a natural-language task with model-chosen hub tools.")
    parser.add_argument("--task", required=True, help="Task text, or @path to read it from a file")
    parser.add_argument("--run", required=True, help="New run directory")
    parser.add_argument("--agent-config", default=str(ROOT / "configs/science-agent.json"))
    parser.add_argument("--hub-config", default=str(ROOT / "configs/science-hub.json"))
    parser.add_argument("--hub-runner", choices=("local", "wsl"), default="local",
                        help="Run the hub in this process, or inside WSL where GPAW, CP2K and RASPA3 live")
    parser.add_argument("--wsl-distro", default="Ubuntu")
    args = parser.parse_args(argv)
    from rsi.backends.factory import backend_identity, create_backend

    task = Path(args.task[1:]).read_text(encoding="utf-8") if args.task.startswith("@") else args.task
    run = Path(args.run).resolve()
    if run.exists() and any(run.iterdir()):
        parser.exit(1, f"Run directory is not empty: {run}\n")
    try:
        config = read_json(args.agent_config)
        validate_agent_config(config)
        backend = create_backend(config)
        if args.hub_runner == "wsl":
            from rsi.science.wsl_hub import LAUNCHER, WslHub

            hub = WslHub(args.hub_config, run / "hub", distro=args.wsl_distro)
            runner = {"kind": "wsl", "distro": args.wsl_distro, "launcher_sha256": sha256(LAUNCHER),
                      "bridge_sha256": sha256(ROOT / "rsi/science/hub_bridge.py"),
                      "wsl_hub_sha256": sha256(ROOT / "rsi/science/wsl_hub.py"), "environment": hub.environment}
        else:
            hub = ScienceToolHub(args.hub_config, run / "hub")
            runner = {"kind": "local"}
        write_json(run / "manifest.json", {
            "schema_version": 1, "task": task, "agent_config": config,
            "agent_config_sha256": sha256(args.agent_config), "hub_config_sha256": sha256(args.hub_config),
            "backend": backend_identity(backend), "hub_runner": runner, "hub_binding": hub.binding,
            "repositories": {"workspace": repository(ROOT), "crystal_builder": repository(ROOT / "Crystal-Builder")},
            "sources": {"agent": sha256(__file__)}})
        agent = ScienceAgent(backend, hub, run, max_steps=config["max_steps"],
                             max_total_tokens=config.get("max_total_tokens"),
                             max_observation_chars=config.get("max_observation_chars") or 6000,
                             model_scratch=config.get("model_scratch"))
        record = agent.run(task)
    except (ValueError, OSError, ImportError) as exc:
        parser.exit(1, f"Science agent stopped: {exc}\n")
    summary = {"status": record["status"], "final": record["final"], "usage": record["usage"],
               "tool_calls": [{k: s.get(k) for k in ("step", "tool", "arguments_json")} | {
                   "call_id": s["observation"].get("call_id"), "ok": s["observation"].get("ok"),
                   "error": s["observation"].get("error")} for s in record["steps"] if "observation" in s]}
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if record["status"] != "completed":
        parser.exit(1)


if __name__ == "__main__":
    main()

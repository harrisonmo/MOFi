"""Controller-owned scientific tools with immutable tasks and persistent budgets."""

from __future__ import annotations

from contextlib import contextmanager
import copy
import os
from pathlib import Path
import re
import time

from rsi.gcmc import core as gcmc


ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "tools/science/methane-gcmc-v1.json"
TOOL = "methane_gcmc_capacity"
CONTRACT = "methane-gcmc-capacity-v1"


class ScientificToolRegistry:
    """Only the controller chooses config paths, session root and allocation."""

    def __init__(self, config_path, session_root):
        self.config_path = Path(config_path).resolve()
        self.session_root = Path(session_root).resolve()
        config = gcmc.read_json(self.config_path)
        gcmc.exact_keys(config, {"schema_version", "tasks", "session_budget"},
                        {"schema_version", "tasks", "session_budget"})
        if config["schema_version"] != 1 or not isinstance(config["tasks"], dict) or not config["tasks"]:
            raise ValueError("A versioned controller-owned task catalog is required")
        self.limits = copy.deepcopy(config["session_budget"])
        keys = {"max_calls", "max_total_cycles", "max_reserved_wall_seconds"}
        gcmc.exact_keys(self.limits, keys, keys)
        for key in ("max_calls", "max_total_cycles"):
            gcmc.integer(self.limits[key], key)
        gcmc.positive(self.limits["max_reserved_wall_seconds"], "max_reserved_wall_seconds")
        self.spec = gcmc.read_json(MANIFEST)
        if (self.spec["id"], self.spec["version"], self.spec["contract"], self.spec["handler"]) != (
                TOOL, 1, CONTRACT, "rsi.gcmc"):
            raise ValueError("Unsupported scientific tool implementation")
        self.tasks = {}
        for task_id, relative in config["tasks"].items():
            if not isinstance(task_id, str) or not re.fullmatch(r"[a-z][a-z0-9_-]{0,63}", task_id):
                raise ValueError("Invalid task ID")
            if not isinstance(relative, str):
                raise ValueError("Controller task configuration must be a path")
            path = (self.config_path.parent / relative).resolve()
            recipe = gcmc.validate_config(gcmc.read_json(path))
            structure = (path.parent / recipe["structure"]).resolve()
            potential = (path.parent / recipe["force_field"]).resolve()
            ff = gcmc.validate_force_field(gcmc.read_json(potential))
            self.tasks[task_id] = {"path": path, "recipe": recipe,
                "inputs": {str(p): gcmc.sha256(p) for p in (path, structure, potential)},
                "public": {"task_id": task_id, "gas": "methane", "temperature_k": recipe["temperature_k"],
                    "pressures_bar": recipe["pressures_bar"], "seeds": recipe["seeds"],
                    "force_field": ff["name"], "force_field_source": ff["source"],
                    "reserved_cycles": len(gcmc.job_plan(recipe)) *
                        (recipe["initialization_cycles"] + recipe["production_cycles"]),
                    "reserved_wall_seconds": recipe["budget"]["max_wall_seconds"]}}
        self.binding = {"registration_config_sha256": gcmc.sha256(self.config_path),
                        "tool_manifest_sha256": gcmc.sha256(MANIFEST),
                        "registry_source_sha256": gcmc.sha256(__file__),
                        "gcmc_source_hashes": gcmc.code_hashes(), "limits": self.limits,
                        "task_inputs": {key: value["inputs"] for key, value in self.tasks.items()}}

    def catalog(self):
        definition = {"type": "function", "function": {"name": TOOL,
            "description": self.spec["description"], "strict": True,
            "parameters": copy.deepcopy(self.spec["parameters"])}}
        definition["function"]["parameters"]["properties"]["task_id"]["enum"] = sorted(self.tasks)
        return {"tools": [definition], "tasks": [copy.deepcopy(t["public"]) for t in self.tasks.values()],
                "session_budget": copy.deepcopy(self.limits), "contract": CONTRACT}

    def _validate_binding(self):
        if (gcmc.sha256(self.config_path) != self.binding["registration_config_sha256"]
                or gcmc.sha256(MANIFEST) != self.binding["tool_manifest_sha256"]
                or gcmc.sha256(__file__) != self.binding["registry_source_sha256"]
                or gcmc.code_hashes() != self.binding["gcmc_source_hashes"]):
            raise ValueError("Tool registration or scientific source changed after binding")
        for task in self.tasks.values():
            for path, checksum in task["inputs"].items():
                if gcmc.sha256(path) != checksum:
                    raise ValueError("An approved task input changed after binding")

    @contextmanager
    def _locked(self):
        self.session_root.mkdir(parents=True, exist_ok=True)
        lock = self.session_root / "session.lock"
        # Hold this lock throughout native execution; a stale lock blocks reuse.
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        try:
            with os.fdopen(fd, "w") as handle:
                handle.write(str(os.getpid()))
            yield
        finally:
            lock.unlink()

    def _cost(self, run):
        manifest_path = run / "manifest.json"
        if not manifest_path.exists():
            return {"known_completed_cycles": 0, "incomplete_jobs_with_unknown_actual_cycles": 0}
        jobs = gcmc.read_json(manifest_path)["jobs"]
        return {"known_completed_cycles": sum(j.get("actual_cycles") or 0 for j in jobs),
                "incomplete_jobs_with_unknown_actual_cycles": sum(
                    j["status"] in {"reserved", "failed", "timeout", "interrupted"} for j in jobs)}

    def call(self, name, arguments):
        envelope = {"tool": name, "contract": CONTRACT, "ok": False, "task_id": None,
                    "data": None, "evidence": None, "cost": None, "error": None}
        try:
            if name != TOOL:
                raise ValueError("Unknown scientific tool")
            gcmc.exact_keys(arguments, {"task_id"}, {"task_id"})
            task_id = arguments["task_id"]
            if not isinstance(task_id, str) or task_id not in self.tasks:
                raise ValueError("Unknown or unauthorized task ID")
            envelope["task_id"] = task_id
            self._validate_binding()
            with self._locked():
                return self._invoke(task_id, envelope)
        except Exception as exc:
            envelope["error"] = {"type": type(exc).__name__, "message": str(exc)}
            return envelope

    def _invoke(self, task_id, envelope):
        path = self.session_root / "ledger.json"
        ledger = gcmc.read_json(path) if path.exists() else {"schema_version": 1,
                    "binding": self.binding, "calls": []}
        if ledger["binding"] != self.binding:
            raise ValueError("Session registration changed; an existing budget cannot be reset")
        if any(c["status"] in {"reserved", "interrupted"} for c in ledger["calls"]):
            raise ValueError("Prior execution is unresolved; preserve this session for audit")
        task = self.tasks[task_id]
        cycles, wall = task["public"]["reserved_cycles"], task["public"]["reserved_wall_seconds"]
        if (len(ledger["calls"]) >= self.limits["max_calls"]
                or sum(c["reserved_cycles"] for c in ledger["calls"]) + cycles > self.limits["max_total_cycles"]
                or sum(c["reserved_wall_seconds"] for c in ledger["calls"]) + wall > self.limits["max_reserved_wall_seconds"]):
            raise ValueError("Scientific tool session budget exhausted")
        call_id = f"call-{len(ledger['calls']) + 1:06d}"
        folder = self.session_root / call_id
        envelope["call_id"] = call_id
        event = {"id": call_id, "tool": TOOL, "task_id": task_id, "status": "reserved",
                 "reserved_cycles": cycles, "reserved_wall_seconds": wall}
        ledger["calls"].append(event)
        gcmc.write_json(path, ledger)
        started = time.monotonic()
        try:
            folder.mkdir(exist_ok=False)
            run = folder / "gcmc"
            frozen = gcmc.prepare(task["path"], run)
            # Recheck original inputs and copied input hashes immediately before execution.
            self._validate_binding()
            approved_structure = (task["path"].parent / task["recipe"]["structure"]).resolve()
            if frozen["input_hashes"]["source.cif"] != task["inputs"][str(approved_structure)]:
                raise ValueError("Prepared CIF differs from the approved task")
            if gcmc.read_json(run / "config.json") != task["recipe"]:
                raise ValueError("Prepared config differs from the approved task")
            approved_ff = (task["path"].parent / task["recipe"]["force_field"]).resolve()
            if gcmc.read_json(run / "force_field.json") != gcmc.read_json(approved_ff):
                raise ValueError("Prepared force field differs from the approved task")
            result = gcmc.execute(run)
            envelope.update(ok=True, data=result,
                evidence={"run_directory": str(run), "manifest_sha256": gcmc.sha256(run / "manifest.json"),
                          "capacity_sha256": gcmc.sha256(run / "capacity.json"),
                          "tool_manifest_sha256": self.binding["tool_manifest_sha256"]})
            event["status"] = "completed"
        except BaseException as exc:
            event["status"] = "interrupted" if isinstance(exc, (KeyboardInterrupt, SystemExit)) else "failed"
            envelope["error"] = {"type": type(exc).__name__, "message": str(exc)}
            if not isinstance(exc, Exception):
                raise
        finally:
            if (folder / "gcmc/manifest.json").exists():
                envelope["evidence"] = {"run_directory": str(folder / "gcmc"),
                    "manifest_sha256": gcmc.sha256(folder / "gcmc/manifest.json"),
                    "capacity_sha256": gcmc.sha256(folder / "gcmc/capacity.json")
                        if (folder / "gcmc/capacity.json").exists() else None,
                    "tool_manifest_sha256": self.binding["tool_manifest_sha256"]}
            envelope["cost"] = {"reserved_cycles": cycles, "reserved_wall_seconds": wall,
                                "elapsed_seconds": time.monotonic() - started, **self._cost(folder / "gcmc")}
            event.update(cost=envelope["cost"], ok=envelope["ok"], error=envelope["error"])
            if folder.is_dir():
                gcmc.write_json(folder / "tool-result.json", envelope)
                event["result_sha256"] = gcmc.sha256(folder / "tool-result.json")
            gcmc.write_json(path, ledger)
        return envelope

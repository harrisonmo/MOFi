"""Controller-owned Zeo++ pore analysis tool with immutable tasks and persistent budgets.

Mirrors ``rsi.science.tool_registry``: the model supplies only ``task_id``;
the controller owns structures, probes, sampling, the binary and the ledger.
"""

from __future__ import annotations

from contextlib import contextmanager
import copy
import os
from pathlib import Path
import re
import time

from . import core


MANIFEST = core.ROOT / "tools/science/zeopp-pores-v1.json"
TOOL = "zeopp_pore_geometry"
CONTRACT = "zeopp-pore-geometry-v1"


class ZeoppToolRegistry:
    """Only the controller chooses config paths, session root and allocation."""

    def __init__(self, config_path, session_root):
        self.config_path = Path(config_path).resolve()
        self.session_root = Path(session_root).resolve()
        config = core.read_json(self.config_path)
        core.exact_keys(config, {"schema_version", "tasks", "session_budget"},
                        {"schema_version", "tasks", "session_budget"})
        if config["schema_version"] != 1 or not isinstance(config["tasks"], dict) or not config["tasks"]:
            raise ValueError("A versioned controller-owned task catalog is required")
        self.limits = copy.deepcopy(config["session_budget"])
        keys = {"max_calls", "max_reserved_wall_seconds"}
        core.exact_keys(self.limits, keys, keys)
        core.integer(self.limits["max_calls"], "max_calls")
        core.positive(self.limits["max_reserved_wall_seconds"], "max_reserved_wall_seconds")
        self.spec = core.read_json(MANIFEST)
        if (self.spec["id"], self.spec["version"], self.spec["contract"], self.spec["handler"]) != (
                TOOL, 1, CONTRACT, "rsi.zeopp"):
            raise ValueError("Unsupported scientific tool implementation")
        self.tasks = {}
        for task_id, relative in config["tasks"].items():
            if not isinstance(task_id, str) or not re.fullmatch(r"[a-z][a-z0-9_-]{0,63}", task_id):
                raise ValueError("Invalid task ID")
            if not isinstance(relative, str):
                raise ValueError("Controller task configuration must be a path")
            path = (self.config_path.parent / relative).resolve()
            recipe = core.validate_recipe(core.read_json(path))
            structure = (path.parent / recipe["structure"]).resolve()
            self.tasks[task_id] = {"path": path, "recipe": recipe, "structure": structure,
                "inputs": {str(p): core.sha256(p) for p in (path, structure)},
                "public": {"task_id": task_id, "structure": structure.stem,
                    "probe": recipe["probe_label"], "probe_radius_angstrom": recipe["probe_radius_angstrom"],
                    "psd_probe_radius_angstrom": recipe["psd_probe_radius_angstrom"],
                    "radii": recipe["radii"], "high_accuracy": recipe["high_accuracy"],
                    "grid_crosscheck": recipe["grid_crosscheck"] is not None,
                    "reserved_wall_seconds": recipe["budget"]["timeout_seconds"]}}
        self.binding = {"registration_config_sha256": core.sha256(self.config_path),
                        "tool_manifest_sha256": core.sha256(MANIFEST),
                        "registry_source_sha256": core.sha256(__file__),
                        "zeopp_source_hashes": core.code_hashes(), "limits": self.limits,
                        "task_inputs": {key: value["inputs"] for key, value in self.tasks.items()}}

    def catalog(self):
        definition = {"type": "function", "function": {"name": TOOL,
            "description": self.spec["description"], "strict": True,
            "parameters": copy.deepcopy(self.spec["parameters"])}}
        definition["function"]["parameters"]["properties"]["task_id"]["enum"] = sorted(self.tasks)
        return {"tools": [definition], "tasks": [copy.deepcopy(t["public"]) for t in self.tasks.values()],
                "session_budget": copy.deepcopy(self.limits), "contract": CONTRACT}

    def _validate_binding(self):
        if (core.sha256(self.config_path) != self.binding["registration_config_sha256"]
                or core.sha256(MANIFEST) != self.binding["tool_manifest_sha256"]
                or core.sha256(__file__) != self.binding["registry_source_sha256"]
                or core.code_hashes() != self.binding["zeopp_source_hashes"]):
            raise ValueError("Tool registration or scientific source changed after binding")
        for task in self.tasks.values():
            for path, checksum in task["inputs"].items():
                if core.sha256(path) != checksum:
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

    def call(self, name, arguments):
        envelope = {"tool": name, "contract": CONTRACT, "ok": False, "task_id": None,
                    "data": None, "evidence": None, "cost": None, "error": None}
        try:
            if name != TOOL:
                raise ValueError("Unknown scientific tool")
            core.exact_keys(arguments, {"task_id"}, {"task_id"})
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
        ledger = core.read_json(path) if path.exists() else {"schema_version": 1,
                    "binding": self.binding, "calls": []}
        if ledger["binding"] != self.binding:
            raise ValueError("Session registration changed; an existing budget cannot be reset")
        if any(c["status"] in {"reserved", "interrupted"} for c in ledger["calls"]):
            raise ValueError("Prior execution is unresolved; preserve this session for audit")
        task = self.tasks[task_id]
        wall = task["public"]["reserved_wall_seconds"]
        if (len(ledger["calls"]) >= self.limits["max_calls"]
                or sum(c["reserved_wall_seconds"] for c in ledger["calls"]) + wall
                > self.limits["max_reserved_wall_seconds"]):
            raise ValueError("Scientific tool session budget exhausted")
        call_id = f"call-{len(ledger['calls']) + 1:06d}"
        folder = self.session_root / call_id
        envelope["call_id"] = call_id
        event = {"id": call_id, "tool": TOOL, "task_id": task_id, "status": "reserved",
                 "reserved_wall_seconds": wall}
        ledger["calls"].append(event)
        core.write_json(path, ledger)
        started = time.monotonic()
        run = folder / "zeopp"
        try:
            folder.mkdir(exist_ok=False)
            frozen = core.prepare(task["path"], run)
            # Recheck original inputs and copied input hashes immediately before execution.
            self._validate_binding()
            if frozen["input_hashes"]["source.cif"] != task["inputs"][str(task["structure"])]:
                raise ValueError("Prepared CIF differs from the approved task")
            if core.read_json(run / "recipe.json") != task["recipe"]:
                raise ValueError("Prepared recipe differs from the approved task")
            result = core.execute(run)
            envelope.update(ok=True, data=result)
            event["status"] = "completed"
        except BaseException as exc:
            event["status"] = "interrupted" if isinstance(exc, (KeyboardInterrupt, SystemExit)) else "failed"
            envelope["error"] = {"type": type(exc).__name__, "message": str(exc)}
            if not isinstance(exc, Exception):
                raise
        finally:
            if (run / "manifest.json").exists():
                envelope["evidence"] = {"run_directory": str(run),
                    "manifest_sha256": core.sha256(run / "manifest.json"),
                    "pores_sha256": core.sha256(run / "pores.json") if (run / "pores.json").exists() else None,
                    "tool_manifest_sha256": self.binding["tool_manifest_sha256"]}
            envelope["cost"] = {"reserved_wall_seconds": wall, "elapsed_seconds": time.monotonic() - started}
            event.update(cost=envelope["cost"], ok=envelope["ok"], error=envelope["error"])
            if folder.is_dir():
                core.write_json(folder / "tool-result.json", envelope)
                event["result_sha256"] = core.sha256(folder / "tool-result.json")
            core.write_json(path, ledger)
        return envelope

"""One model-facing catalog for every scientific tool, executed through the HPC layer.

The model chooses structures, methods and parameters inside controller bounds. The controller owns
the structure, force-field and OpenMM system libraries, the engines, the compute budget (an
``rsi.hpc.Scheduler`` session) and the evidence ledger. Every argument is checked against the tool
schema and the configured bounds before anything is written or launched, and every call records the
exact arguments the model chose beside its results, so an independent evaluator can judge both.

The task-ID registries (``rsi.science.tool_registry``, ``rsi.zeopp.registry``) stay unchanged for
frozen experiments; this hub is the open-method interface.
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import shlex
import shutil
import time

from rsi.hpc import Scheduler
from rsi.hpc.core import locked, pid_alive, read_json, sha256, write_json


ROOT = Path(__file__).resolve().parents[2]
MANIFESTS = {
    "zeopp_pore_geometry": ROOT / "tools/science/zeopp-pores-v2.json",
    "methane_gcmc_capacity": ROOT / "tools/science/methane-gcmc-v2.json",
    "dft_calculation": ROOT / "tools/science/dft-calculation-v1.json",
    "orca_calculation": ROOT / "tools/science/orca-calculation-v1.json",
    "openmm_minimize": ROOT / "tools/science/openmm-minimize-v1.json",
    "science_budget_status": ROOT / "tools/science/science-budget-status-v1.json",
    "mof_database_catalog": ROOT / "tools/science/mof-database-catalog-v1.json",
}
BUDGET_TOOL = "science_budget_status"
MOF_DB_TOOL = "mof_database_catalog"
SAMPLING = {"fast": {"surface_per_atom": 500, "volume": 5000, "psd": 5000, "block": 1000},
            "standard": {"surface_per_atom": 2000, "volume": 50000, "psd": 20000, "block": 5000},
            "fine": {"surface_per_atom": 5000, "volume": 200000, "psd": 100000, "block": 20000}}
GUIDANCE = [
    "Every call reserves cpus x wall time from one shared budget; unused time is refunded when a job finishes.",
    "Screen cheaply first (zeopp_pore_geometry, emt, short GCMC) and spend on the few candidates that matter.",
    "Results from different force fields, functionals or engines are not comparable; compare like with like.",
    "A finished calculation is not a converged one: read the convergence fields and flags in every result.",
    "Call science_budget_status (free) whenever you plan the next expensive step.",
]
MOF_DB_GUIDANCE = ("mof_database_catalog (free) lists external MOF databases and what they cover; it is unverified "
                   "metadata for choosing data sources, never evidence for a scientific claim.")
ID = r"^[a-z][a-z0-9_-]{0,63}$"


def _digest_tree(directory):
    return {p.name: sha256(p) for p in sorted(Path(directory).glob("*.py"))}


def _number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def check_schema(schema, value, where="arguments"):
    """Validate the strict-mode JSON Schema subset used by the manifests."""
    kinds = schema.get("type")
    kinds = kinds if isinstance(kinds, list) else [kinds]
    if value is None:
        if "null" not in kinds:
            raise ValueError(f"{where} may not be null")
        return
    if "enum" in schema and value not in schema["enum"]:
        raise ValueError(f"{where} must be one of {[v for v in schema['enum'] if v is not None]}")
    if "object" in kinds and isinstance(value, dict):
        properties = schema.get("properties", {})
        if set(value) - set(properties) or set(schema.get("required", [])) - set(value):
            raise ValueError(f"{where} must contain exactly {sorted(properties)}")
        for key, item in value.items():
            check_schema(properties[key], item, f"{where}.{key}")
    elif "array" in kinds and isinstance(value, list):
        for index, item in enumerate(value):
            check_schema(schema.get("items", {}), item, f"{where}[{index}]")
    elif "string" in kinds and isinstance(value, str):
        pass
    elif "boolean" in kinds and isinstance(value, bool):
        pass
    elif "integer" in kinds and isinstance(value, int) and not isinstance(value, bool):
        pass
    elif "number" in kinds and _number(value):
        pass
    else:
        raise ValueError(f"{where} must be of type {kinds}")


def _bounded(value, bounds, name):
    low, high = bounds
    if value is not None and not low <= value <= high:
        raise ValueError(f"{name} must lie in [{low}, {high}]")
    return value


class ScienceToolHub:
    """Owned by a controller: config, session root and HPC backend are never model inputs."""

    def __init__(self, config_path, session_root, runner=None):
        self.config_path = Path(config_path).resolve()
        self.session_root = Path(session_root).resolve()
        self.config = read_json(self.config_path)
        self._validate_config()
        base = self.config_path.parent
        hpc = self.config["hpc"]
        self.hpc_config = (base / hpc["config"]).resolve()
        self.backend = hpc["backend"]
        self.capacity_timeout = hpc["capacity_timeout_seconds"]
        kwargs = {"runner": runner} if runner else {}
        self.scheduler = Scheduler(self.hpc_config, self.session_root / "hpc", **kwargs)
        if self.backend not in self.scheduler.config["backends"]:
            raise ValueError(f"Unknown HPC backend {self.backend}")
        self.structures = {key: self._structure(key, value, base) for key, value in self.config["structures"].items()}
        self.force_fields = {key: (base / path).resolve() for key, path in self.config.get("force_fields", {}).items()}
        mof_db = self.config["tools"].get(MOF_DB_TOOL)
        self.mof_db_files = [(base / mof_db[k]).resolve() for k in ("csv", "annotations")] if mof_db else []
        self.systems = {key: {"system_xml": (base / value["system_xml"]).resolve(),
                              "positions_json": (base / value["positions_json"]).resolve(),
                              "description": value["description"], "force_field": value["force_field"]}
                        for key, value in self.config.get("openmm_systems", {}).items()}
        self.manifests = {name: read_json(path) for name, path in MANIFESTS.items()}
        for name, manifest in self.manifests.items():
            if manifest["id"] != name or manifest["handler"] != "rsi.science.hub":
                raise ValueError(f"Manifest {MANIFESTS[name].name} does not belong to this hub")
        self.unavailable = {}
        self.tools = {name: self._definition(name) for name in self._available()}
        self.binding = self._binding()
        self.ledger_path = self.session_root / "hub-ledger.json"
        with locked(self.ledger_path):
            if self.ledger_path.exists():
                ledger = read_json(self.ledger_path)
                if ledger["binding"] != self.binding:
                    raise ValueError("Hub registration changed after the session started; start a new session")
            else:
                ledger = {"schema_version": 1, "binding": self.binding, "calls": []}
            for call in ledger["calls"]:
                if call["status"] == "running" and not pid_alive(call["controller_pid"]):
                    call.update(status="interrupted", error="Controller exited before the call finished")
            write_json(self.ledger_path, ledger)

    # Configuration --------------------------------------------------------------------------------

    def _validate_config(self):
        config = self.config
        allowed = {"schema_version", "hpc", "max_calls", "structures", "force_fields", "openmm_systems", "tools"}
        if not isinstance(config, dict) or set(config) - allowed or {"schema_version", "hpc", "max_calls",
                                                                    "structures", "tools"} - set(config):
            raise ValueError(f"Hub config must contain {sorted(allowed)}")
        if config["schema_version"] != 1:
            raise ValueError("Unsupported hub config schema")
        if set(config["hpc"]) != {"config", "backend", "capacity_timeout_seconds"}:
            raise ValueError("hpc needs config, backend and capacity_timeout_seconds")
        if isinstance(config["max_calls"], bool) or not isinstance(config["max_calls"], int) or config["max_calls"] < 1:
            raise ValueError("max_calls must be a positive integer")
        for group in ("structures", "force_fields", "openmm_systems"):
            for key in config.get(group, {}):
                if not re.fullmatch(ID, key):
                    raise ValueError(f"Invalid {group} ID {key!r}")
        unknown = set(config["tools"]) - (set(MANIFESTS) - {BUDGET_TOOL})
        if unknown:
            raise ValueError(f"Unknown tools in hub config: {sorted(unknown)}")
        if MOF_DB_TOOL in config["tools"] and set(config["tools"][MOF_DB_TOOL]) != {"csv", "annotations"}:
            raise ValueError(f"{MOF_DB_TOOL} needs exactly csv and annotations paths")

    def _structure(self, key, value, base):
        if set(value) != {"path", "description"}:
            raise ValueError(f"Structure {key} needs exactly path and description")
        path = (base / value["path"]).resolve()
        if not path.is_file():
            raise ValueError(f"Structure {key} not found: {path}")
        atoms = elements = periodic = None
        has_cell = False
        if importlib.util.find_spec("ase"):
            from ase.io import read
            try:
                frames = read(str(path), index=":")
                atoms, elements = len(frames[0]), sorted(set(frames[0].get_chemical_symbols()))
                periodic, has_cell = bool(any(frames[0].pbc)), frames[0].cell.rank == 3
            except Exception:  # Engines apply their own readers; the count only guides the model.
                pass
        # A plane-wave or Gaussian-and-plane-wave code needs a cell; ORCA needs no periodicity.
        return {"path": path, "has_cell": has_cell,
                "public": {"structure_id": key, "description": value["description"],
                           "format": path.suffix.lower().lstrip("."), "atoms_in_cell": atoms, "elements": elements,
                           "periodic": periodic}}

    def _available(self):
        tools = self.config["tools"]
        names = []
        if "zeopp_pore_geometry" in tools:
            from rsi.zeopp import core as zeo
            # find_spec("xtal") alone passes when xtal is on the path but its own dependencies are not.
            missing = [m for m in ("xtal", "gemmi", "spglib", "scipy") if not importlib.util.find_spec(m)]
            if missing:
                self.unavailable["zeopp_pore_geometry"] = f"Crystal-Builder (xtal) is not importable; missing {missing}"
            elif zeo.binary() is None:
                self.unavailable["zeopp_pore_geometry"] = f"Zeo++ network binary not found ({zeo.ENV_VAR})"
            else:
                names.append("zeopp_pore_geometry")
        if "methane_gcmc_capacity" in tools:
            missing = [m for m in ("raspalib", "ase", "psutil") if not importlib.util.find_spec(m)]
            if missing:
                self.unavailable["methane_gcmc_capacity"] = f"Missing packages: {missing}"
            elif not self.force_fields:
                self.unavailable["methane_gcmc_capacity"] = "No force fields registered"
            else:
                names.append("methane_gcmc_capacity")
        if "dft_calculation" in tools:
            self.dft_backends = [b for b in tools["dft_calculation"]["backends"] if self._dft_backend_ready(b)]
            if not importlib.util.find_spec("ase") or not self.dft_backends:
                self.unavailable["dft_calculation"] = "ASE or every configured DFT backend is missing"
            else:
                names.append("dft_calculation")
        if "orca_calculation" in tools:
            from rsi.orca import core as orca
            if not importlib.util.find_spec("xtal") or not importlib.util.find_spec("ase"):
                self.unavailable["orca_calculation"] = "Crystal-Builder (xtal) or ASE is not importable"
            elif orca.binary() is None:
                self.unavailable["orca_calculation"] = f"ORCA executable not found ({orca.ENV_VAR})"
            else:
                self._validate_orca_config(tools["orca_calculation"])
                names.append("orca_calculation")
        if "openmm_minimize" in tools:
            if not importlib.util.find_spec("openmm"):
                self.unavailable["openmm_minimize"] = "OpenMM is not installed"
            elif not self.systems:
                self.unavailable["openmm_minimize"] = "No OpenMM systems registered (no MOF force field is frozen yet)"
            else:
                names.append("openmm_minimize")
        if MOF_DB_TOOL in self.config["tools"]:
            from rsi.science.mof_databases import MofDatabaseCatalog
            try:
                self.mof_databases = MofDatabaseCatalog(*self.mof_db_files)
                names.append(MOF_DB_TOOL)
            except (ValueError, OSError, KeyError) as exc:
                self.unavailable[MOF_DB_TOOL] = f"MOF database catalog rejected: {exc}"
        return names + [BUDGET_TOOL]

    def _dft_backend_ready(self, backend):
        if backend == "gpaw":
            return bool(importlib.util.find_spec("gpaw"))
        if backend == "cp2k":
            executable = self.config["tools"]["dft_calculation"]["backends"]["cp2k"]["defaults"].get("executable", "")
            return bool(executable and shutil.which(shlex.split(executable)[0]))
        return backend == "emt"

    def _validate_orca_config(self, tool):
        from xtal.orca import catalogue

        keys = {"functionals", "basis_sets", "max_atoms", "max_cpus", "maxcore_mb", "max_wall_seconds",
                "max_opt_cycles"}
        if set(tool) != keys:
            raise ValueError(f"orca_calculation needs exactly {sorted(keys)}")
        unknown = [f for f in tool["functionals"] if catalogue.functional(f) is None or catalogue.functional(f).key != f]
        unknown += [b for b in tool["basis_sets"] if catalogue.basis(b) is None or catalogue.basis(b).key != b]
        if unknown or not tool["functionals"]:
            raise ValueError(f"orca_calculation lists keywords outside the ORCA 6.1 catalogue: {unknown}")

    def _bounds(self, name):
        tool = self.config["tools"].get(name, {})
        if name == "zeopp_pore_geometry":
            return {"probe_radius_angstrom": tool["probe_radius_angstrom"],
                    "psd_probe_radius_angstrom": tool["probe_radius_angstrom"],
                    "grid_crosscheck_spacing_angstrom": [0.2, 1.0],
                    "timeout_seconds": [1, tool["max_timeout_seconds"]]}
        if name == "methane_gcmc_capacity":
            return {key: tool[key] for key in ("temperature_k", "pressures_bar", "num_seeds",
                                               "initialization_cycles", "production_cycles",
                                               "timeout_seconds_per_job")} | {
                    "loading_pressure_bar": tool["pressures_bar"], "unloading_pressure_bar": tool["pressures_bar"]}
        if name == "dft_calculation":
            return {"fmax_ev_per_angstrom": [0.001, 1.0], "max_steps": [1, tool["max_steps"]],
                    "cpus": [1, tool["max_cpus"]], "wall_seconds": [1, tool["max_wall_seconds"]],
                    "charge": [-10, 10], "spin_multiplicity": [1, 11]}
        if name == "orca_calculation":
            return {"max_opt_cycles": [1, tool["max_opt_cycles"]], "cpus": [1, tool["max_cpus"]],
                    "wall_seconds": [1, tool["max_wall_seconds"]], "charge": [-10, 10], "spin_multiplicity": [1, 11]}
        if name == "openmm_minimize":
            return {"tolerance_kj_mol_nm": [1e-4, 1e4], "max_iterations": [1, tool["max_iterations"]],
                    "cpus": [1, tool["max_cpus"]], "wall_seconds": [1, tool["max_wall_seconds"]]}
        return {}

    def _definition(self, name):
        manifest = self.manifests[name]
        parameters = copy.deepcopy(manifest["parameters"])
        properties = parameters["properties"]
        tool = self.config["tools"].get(name, {})
        cif = sorted(k for k, s in self.structures.items() if s["public"]["format"] == "cif")
        if name in {"zeopp_pore_geometry", "methane_gcmc_capacity"}:
            properties["structure_id"]["enum"] = cif
        if name == "methane_gcmc_capacity":
            properties["force_field_id"]["enum"] = sorted(self.force_fields)
            properties["pressures_bar"]["description"] += f" At most {tool['max_pressure_points']} points."
        if name == "dft_calculation":
            limit = tool["max_atoms"]
            properties["structure_id"]["enum"] = sorted(
                k for k, s in self.structures.items() if s["has_cell"] and (s["public"]["atoms_in_cell"] or 0) <= limit)
            properties["structure_id"]["description"] += f" At most {limit} atoms."
            properties["backend"]["enum"] = self.dft_backends
            properties["xc"]["enum"] = tool["xc"] + [None]
            cutoffs = "; ".join(f"{b} [{tool['backends'][b]['cutoff_ev'][0]}, {tool['backends'][b]['cutoff_ev'][1]}]"
                                for b in self.dft_backends if "cutoff_ev" in tool["backends"][b])
            if cutoffs:
                properties["cutoff_ev"]["description"] += f" Allowed: {cutoffs}."
        if name == "orca_calculation":
            limit = tool["max_atoms"]
            properties["structure_id"]["enum"] = sorted(
                k for k, s in self.structures.items()
                if s["public"]["periodic"] is False and 0 < (s["public"]["atoms_in_cell"] or 0) <= limit)
            properties["structure_id"]["description"] += f" At most {limit} atoms."
            properties["functional"]["enum"] = list(tool["functionals"])
            properties["basis"]["enum"] = list(tool["basis_sets"]) + [None]
            properties["cpus"]["description"] += f" Each process may use {tool['maxcore_mb']} MB."
        if name == "openmm_minimize":
            properties["system_id"]["enum"] = sorted(self.systems)
            properties["platform"]["enum"] = tool["platforms"]
        if name == MOF_DB_TOOL:
            options = self.mof_databases.options()
            properties["method"]["enum"] = options["methods"] + [None]
            properties["identifier"]["enum"] = options["identifiers"] + [None]
            properties["required_fields"]["items"]["enum"] = options["fields"]
        for key, (low, high) in self._bounds(name).items():
            properties[key]["description"] += f" Allowed: {low} to {high}."
        return {"type": "function", "function": {"name": name, "description": manifest["description"],
                                                 "strict": True, "parameters": parameters}}

    def _binding(self):
        files = [self.config_path, self.hpc_config] + list(MANIFESTS.values())
        files += [s["path"] for s in self.structures.values()] + list(self.force_fields.values())
        files += [p for s in self.systems.values() for p in (s["system_xml"], s["positions_json"])]
        files += self.mof_db_files
        sources = {"hub": sha256(__file__), "hpc": _digest_tree(ROOT / "rsi/hpc"),
                   "dft": _digest_tree(ROOT / "rsi/dft"), "zeopp": _digest_tree(ROOT / "rsi/zeopp"),
                   "gcmc": _digest_tree(ROOT / "rsi/gcmc"), "openmm": sha256(ROOT / "rsi/openmm_engine.py")}
        if self.mof_db_files:
            sources["mof_databases"] = sha256(ROOT / "rsi/science/mof_databases.py")
        if "orca_calculation" in self.tools:
            from rsi.orca import core as orca
            # The Crystal-Builder input writer is part of the engine; the binary is hashed per run.
            sources["orca"] = orca.code_hashes()
            sources["orca_executable"] = str(orca.binary())
        return {"files": {str(p): sha256(p) for p in files}, "sources": sources,
                "tools": sorted(self.tools), "backend": self.backend}

    # Public API -----------------------------------------------------------------------------------

    def catalog(self):
        return {"tools": [copy.deepcopy(d) for d in self.tools.values()],
                "structures": [copy.deepcopy(s["public"]) for s in self.structures.values()],
                "force_fields": [{"force_field_id": k, **{f: read_json(p)[f] for f in ("name", "source")}}
                                 for k, p in self.force_fields.items()],
                "openmm_systems": [{"system_id": k, "description": s["description"], "force_field": s["force_field"]}
                                   for k, s in self.systems.items()],
                "unavailable_tools": copy.deepcopy(self.unavailable),
                "guidance": GUIDANCE + ([MOF_DB_GUIDANCE] if MOF_DB_TOOL in self.tools else []),
                "budget": self.budget_status()}

    def budget_status(self):
        status = self.scheduler.status()
        with locked(self.ledger_path):
            calls = read_json(self.ledger_path)["calls"]
        limits = status["limits"]
        remaining = {key: limits[f"max_{key}"] - status["usage"][key] for key in status["usage"]}
        return {"hpc_limits": limits, "hpc_used": status["usage"], "hpc_remaining": remaining,
                "calls_used": len(calls), "max_calls": self.config["max_calls"],
                "calls": [{k: c.get(k) for k in ("id", "tool", "status", "ok", "cost")} for c in calls]}

    def call(self, name, arguments):
        envelope = {"tool": name, "ok": False, "call_id": None, "arguments": arguments, "data": None,
                    "evidence": None, "cost": None, "error": None}
        try:
            if name not in self.tools:
                reason = self.unavailable.get(name, "Unknown scientific tool")
                raise ValueError(f"Tool {name} is unavailable: {reason}")
            check_schema(self.tools[name]["function"]["parameters"], arguments)
            for key, bounds in self._bounds(name).items():
                value = arguments[key]
                for item in value if isinstance(value, list) else [value]:
                    _bounded(item, bounds, key)
            if name == BUDGET_TOOL:
                envelope.update(ok=True, data=self.budget_status())
                return envelope
            if self._binding() != self.binding:
                raise ValueError("Registered inputs or sources changed after the session started")
            if name == MOF_DB_TOOL:
                envelope.update(ok=True, data=self.mof_databases.query(arguments))
                return envelope
            plan = getattr(self, f"_plan_{name}")(arguments)
        except Exception as exc:
            envelope["error"] = {"type": type(exc).__name__, "message": str(exc)}
            return envelope
        return self._run(name, arguments, plan, envelope)

    def _run(self, name, arguments, plan, envelope):
        with locked(self.ledger_path):
            ledger = read_json(self.ledger_path)
            if len(ledger["calls"]) >= self.config["max_calls"]:
                envelope["error"] = {"type": "ValueError", "message": "Tool call budget exhausted"}
                return envelope
            call_id = f"call-{len(ledger['calls']) + 1:04d}"
            folder = self.session_root / call_id
            folder.mkdir(parents=True, exist_ok=False)
            ledger["calls"].append({"id": call_id, "tool": name, "arguments": arguments, "status": "running",
                                    "controller_pid": os.getpid(), "started_at": time.time()})
            write_json(self.ledger_path, ledger)
        envelope["call_id"] = call_id
        write_json(folder / "arguments.json", {"tool": name, "arguments": arguments})
        started = time.monotonic()
        status = "failed"
        try:
            data, evidence = plan(folder)
            envelope.update(ok=True, data=data, evidence=evidence)
            status = "completed"
        except BaseException as exc:
            status = "interrupted" if isinstance(exc, (KeyboardInterrupt, SystemExit)) else "failed"
            envelope["error"] = {"type": type(exc).__name__, "message": str(exc)}
            if not isinstance(exc, Exception):
                raise
        finally:
            envelope["cost"] = self._cost(folder, time.monotonic() - started)
            write_json(folder / "tool-result.json", envelope)
            with locked(self.ledger_path):
                ledger = read_json(self.ledger_path)
                entry = next(c for c in ledger["calls"] if c["id"] == call_id)
                entry.update(status=status, ok=envelope["ok"], cost=envelope["cost"], error=envelope["error"],
                             result_sha256=sha256(folder / "tool-result.json"))
                write_json(self.ledger_path, ledger)
        return envelope

    def _cost(self, folder, elapsed):
        status = self.scheduler.status()
        prefix = str(folder) + os.sep
        jobs = [j for j in status["jobs"] if j.get("cwd") and (j["cwd"] + os.sep).startswith(prefix)]
        charged = {key: sum((j.get("charged") or {}).get(key, 0) for j in jobs)
                   for key in ("jobs", "cpu_seconds", "gpu_seconds", "wall_seconds")}
        remaining = {key: status["limits"][f"max_{key}"] - status["usage"][key] for key in status["usage"]}
        return {"elapsed_seconds": elapsed, "hpc_jobs": [j["id"] for j in jobs], "charged": charged,
                "remaining": remaining}

    def _schedule(self):
        return {"scheduler": self.scheduler, "backend": self.backend, "capacity_timeout": self.capacity_timeout}

    def _copy_structure(self, folder, structure_id):
        source = self.structures[structure_id]["path"]
        inputs = folder / "inputs"
        inputs.mkdir()
        shutil.copyfile(source, inputs / source.name)
        return inputs, source.name

    # Tool plans: validate everything that needs the library, then return the deferred execution ----

    def _plan_zeopp_pore_geometry(self, a):
        from rsi.zeopp import core as zeo

        if self.structures[a["structure_id"]]["public"]["format"] != "cif":
            raise ValueError("Zeo++ needs a CIF structure")
        tool = self.config["tools"]["zeopp_pore_geometry"]
        grid = a["grid_crosscheck_spacing_angstrom"]
        recipe = {"schema_version": 1, "structure": None,
                  "probe_label": f"probe {a['probe_radius_angstrom']:g} A (model-selected)"[:40],
                  "probe_radius_angstrom": a["probe_radius_angstrom"],
                  "channel_radius_angstrom": a["probe_radius_angstrom"],
                  "psd_probe_radius_angstrom": a["psd_probe_radius_angstrom"], "radii": "zeo-default",
                  "high_accuracy": a["high_accuracy"], "samples": SAMPLING[a["sampling"]],
                  "block_pockets": a["block_pockets"],
                  "grid_crosscheck": None if grid is None else {"spacing_angstrom": grid},
                  "budget": {"timeout_seconds": a["timeout_seconds"], "max_cell_atoms": tool["max_cell_atoms"]}}
        zeo.validate_recipe({**recipe, "structure": "check.cif"})

        def run(folder):
            inputs, name = self._copy_structure(folder, a["structure_id"])
            write_json(inputs / "recipe.json", {**recipe, "structure": name})
            output = folder / "zeopp"
            zeo.prepare(inputs / "recipe.json", output)
            result = zeo.execute(output, **self._schedule())
            return result, {"run_directory": str(output), "manifest_sha256": sha256(output / "manifest.json"),
                            "pores_sha256": sha256(output / "pores.json")}
        return run

    def _plan_methane_gcmc_capacity(self, a):
        from rsi.gcmc import core as gcmc

        if self.structures[a["structure_id"]]["public"]["format"] != "cif":
            raise ValueError("GCMC needs a CIF structure")
        tool = self.config["tools"]["methane_gcmc_capacity"]
        pressures = a["pressures_bar"]
        if len(pressures) > tool["max_pressure_points"]:
            raise ValueError(f"At most {tool['max_pressure_points']} pressure points")
        seeds = [101 * (i + 1) for i in range(a["num_seeds"])]
        jobs = len(pressures) * len(seeds)
        config = {"schema_version": 1, "structure": None, "force_field": "force_field.json",
                  "temperature_k": a["temperature_k"], "pressures_bar": pressures,
                  "loading_pressure_bar": a["loading_pressure_bar"],
                  "unloading_pressure_bar": a["unloading_pressure_bar"], "seeds": seeds,
                  "initialization_cycles": a["initialization_cycles"], "production_cycles": a["production_cycles"],
                  "blocks": tool["blocks"], "cutoff_angstrom": tool["cutoff_angstrom"], "fugacity_coefficients": None,
                  "budget": {"max_jobs": jobs,
                             "max_total_cycles": jobs * (a["initialization_cycles"] + a["production_cycles"]),
                             "timeout_seconds_per_job": a["timeout_seconds_per_job"],
                             "max_wall_seconds": jobs * a["timeout_seconds_per_job"],
                             "max_supercell_atoms": tool["max_supercell_atoms"]}}
        gcmc.validate_config({**config, "structure": "check.cif"})
        gcmc.validate_force_field(read_json(self.force_fields[a["force_field_id"]]))

        def run(folder):
            inputs, name = self._copy_structure(folder, a["structure_id"])
            shutil.copyfile(self.force_fields[a["force_field_id"]], inputs / "force_field.json")
            write_json(inputs / "config.json", {**config, "structure": name})
            output = folder / "gcmc"
            gcmc.prepare(inputs / "config.json", output)
            result = gcmc.execute(output, **self._schedule())
            return result, {"run_directory": str(output), "manifest_sha256": sha256(output / "manifest.json"),
                            "capacity_sha256": sha256(output / "capacity.json")}
        return run

    def _plan_dft_calculation(self, a):
        from rsi.dft import core as dft

        tool = self.config["tools"]["dft_calculation"]
        backend = a["backend"]
        settings = tool["backends"][backend]
        if backend == "emt":
            if any(a[k] is not None for k in ("xc", "cutoff_ev", "kpts")):
                raise ValueError("emt takes no xc, cutoff_ev or kpts; pass null")
            calculator = dict(settings["defaults"])
        else:
            if a["xc"] is None or a["cutoff_ev"] is None:
                raise ValueError(f"{backend} needs an explicit xc and cutoff_ev")
            _bounded(a["cutoff_ev"], settings["cutoff_ev"], "cutoff_ev")
            if backend == "gpaw":
                if a["kpts"] is not None and (len(a["kpts"]) != 3 or not all(1 <= k <= tool["max_kpts"] for k in a["kpts"])):
                    raise ValueError(f"kpts must be three integers in [1, {tool['max_kpts']}]")
                calculator = {**settings["defaults"], "mode": {"name": "pw", "ecut": a["cutoff_ev"]},
                              "xc": a["xc"], "kpts": a["kpts"] or [1, 1, 1]}
                if a["spin_multiplicity"] != 1:
                    calculator["spinpol"] = True
            else:
                if a["kpts"] is not None:
                    raise ValueError("The CP2K adapter runs at the Gamma point; pass kpts null")
                calculator = {**settings["defaults"], "xc": a["xc"], "cutoff_ev": a["cutoff_ev"]}
        optimize = a["task"] == "optimize"
        fields = ("optimizer", "fmax_ev_per_angstrom", "max_steps")
        if optimize != all(a[k] is not None for k in fields) or (not optimize and any(a[k] is not None for k in fields)):
            raise ValueError("optimize needs optimizer, fmax_ev_per_angstrom and max_steps; single_point needs them null")
        config = {"schema_version": 1, "backend": backend, "task": a["task"], "structure": None,
                  "charge": a["charge"], "spin_multiplicity": a["spin_multiplicity"], "calculator": calculator,
                  "budget": {"timeout_seconds_per_attempt": a["wall_seconds"], "max_wall_seconds": a["wall_seconds"],
                             "max_atoms": tool["max_atoms"], "max_restarts": 0, "threads": a["cpus"]}}
        if a["fixed_elements"]:
            config["constraints"] = {"fixed_elements": a["fixed_elements"]}
        if optimize:
            config["optimizer"] = {"name": a["optimizer"], "fmax_ev_per_angstrom": a["fmax_ev_per_angstrom"],
                                   "max_steps": a["max_steps"]}
        dft.validate_config({**config, "structure": "check.xyz"})

        def run(folder):
            inputs, name = self._copy_structure(folder, a["structure_id"])
            write_json(inputs / "config.json", {**config, "structure": name})
            output = folder / "dft"
            dft.prepare(inputs / "config.json", output)
            summary = dft.execute(output, **self._schedule())
            # Full geometry and forces stay in the evidence; the model gets the decision-relevant numbers.
            data = {k: v for k, v in summary.items()
                    if k not in {"forces_ev_per_angstrom", "positions_angstrom", "cell_angstrom"}}
            return data, {"run_directory": str(output), "manifest_sha256": sha256(output / "manifest.json"),
                          "summary_sha256": sha256(output / "summary.json")}
        return run

    def _plan_orca_calculation(self, a):
        from rsi.orca import core as orca

        tool = self.config["tools"]["orca_calculation"]
        optimize = a["task"] == "optimize"
        if optimize != (a["opt_level"] is not None and a["max_opt_cycles"] is not None) or (
                not optimize and (a["opt_level"] is not None or a["max_opt_cycles"] is not None)):
            raise ValueError("optimize needs opt_level and max_opt_cycles; energy and gradient need them null")
        memory = a["cpus"] * tool["maxcore_mb"]
        limit = self.scheduler.config["backends"][self.backend].get("max_memory_mb")
        if limit and memory > limit:
            raise ValueError(f"{a['cpus']} processes x {tool['maxcore_mb']} MB exceed the backend's {limit} MB")
        config = {"schema_version": 1, "structure": None, "charge": a["charge"],
                  "spin_multiplicity": a["spin_multiplicity"], "task": a["task"], "frequencies": a["frequencies"],
                  "method": {key: a[key] for key in ("functional", "basis", "dispersion", "ri", "scf_convergence")},
                  "budget": {"cpus": a["cpus"], "maxcore_mb": tool["maxcore_mb"], "wall_seconds": a["wall_seconds"],
                             "max_atoms": tool["max_atoms"]}}
        if optimize:
            config["optimization"] = {"level": a["opt_level"], "max_cycles": a["max_opt_cycles"]}
        source = self.structures[a["structure_id"]]["path"]
        # Spin, catalogue and periodicity refusals happen here, before a call or a job is spent.
        orca.check({**config, "structure": source.name}, source)

        def run(folder):
            inputs, name = self._copy_structure(folder, a["structure_id"])
            write_json(inputs / "config.json", {**config, "structure": name})
            output = folder / "orca"
            orca.prepare(inputs / "config.json", output)
            summary = orca.execute(output, **self._schedule())
            # Forces, all modes and the final geometry stay in the evidence.
            data = {k: v for k, v in summary.items()
                    if k not in {"forces_ev_per_angstrom", "final_positions_angstrom", "frequencies_cm-1"}}
            return data, {"run_directory": str(output), "manifest_sha256": sha256(output / "manifest.json"),
                          "summary_sha256": sha256(output / "summary.json"),
                          "native_output_sha256": summary["checks"]["native_output_sha256"]}
        return run

    def _plan_openmm_minimize(self, a):
        system = self.systems[a["system_id"]]
        options = {"tolerance": a["tolerance_kj_mol_nm"], "max_iterations": a["max_iterations"],
                   "platform": a["platform"], "return_positions": a["return_positions"]}
        if a["platform"] == "CPU":
            options["threads"] = a["cpus"]
        elif a["cpus"] != 1:
            raise ValueError("Only the CPU platform uses more than one CPU; request cpus 1")

        def run(folder):
            from rsi.openmm_engine import from_openmm_event, run_isolated

            output = folder / "openmm" / "event.json"
            event = run_isolated(system["system_xml"], system["positions_json"], output,
                                 timeout_seconds=a["wall_seconds"], **self._schedule(), **options)
            if not event["result"]["ok"]:
                error = event["result"]["error"]
                raise RuntimeError(f"{error['type']}: {error['message']}")
            data = {**event["result"]["data"], "force_field": system["force_field"],
                    "units": {"energy": "kJ/mol", "force": "kJ/mol/nm", "length": "nm"},
                    "audit_payload": from_openmm_event(event)["payload"]}
            return data, {"event_path": str(output), "event_sha256": sha256(output),
                          "system_sha256": event["action"]["system_sha256"]}
        return run


def main(argv=None):
    parser = argparse.ArgumentParser(description="List or call the model-facing scientific tools.")
    parser.add_argument("command", choices=("list", "call", "budget"))
    parser.add_argument("--config", default=str(ROOT / "configs/science-hub.json"))
    parser.add_argument("--session", default=str(ROOT / "runs/science-hub-default"))
    parser.add_argument("--tool")
    parser.add_argument("--arguments", help="JSON object of tool arguments")
    args = parser.parse_args(argv)
    try:
        hub = ScienceToolHub(args.config, args.session)
        if args.command == "list":
            result = hub.catalog()
        elif args.command == "budget":
            result = hub.budget_status()
        else:
            if not args.tool or args.arguments is None:
                parser.error("call requires --tool and --arguments")
            result = hub.call(args.tool, json.loads(args.arguments))
        print(json.dumps(result, indent=2, allow_nan=False))
        if result.get("ok") is False:
            parser.exit(1)
    except (ValueError, OSError, ImportError) as exc:
        parser.exit(1, f"Science hub stopped: {exc}\n")


if __name__ == "__main__":
    main()

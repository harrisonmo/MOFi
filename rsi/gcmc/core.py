"""Prepare, execute and summarize reproducible rigid-framework methane GCMC."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import re
import statistics
import subprocess
import sys
import time
import warnings


ROOT = Path(__file__).resolve().parents[2]
ENGINE_VERSION = "3.1.0"


def read_json(path):
    def reject(value):
        raise ValueError(f"Nonfinite JSON value: {value}")
    return json.loads(Path(path).read_text(encoding="utf-8"), parse_constant=reject)


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def positive(value, name, allow_zero=False):
    if (type(value) not in (int, float) or not math.isfinite(value)
            or value < 0 or (not allow_zero and value == 0)):
        raise ValueError(f"{name} must be finite and {'nonnegative' if allow_zero else 'positive'}")
    return value


def integer(value, name, minimum=1):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def exact_keys(value, allowed, required):
    if not isinstance(value, dict) or set(value) - set(allowed) or set(required) - set(value):
        raise ValueError(f"Expected required keys {sorted(required)}; allowed keys {sorted(allowed)}")


def validate_config(config):
    keys = {"schema_version", "structure", "force_field", "temperature_k", "pressures_bar",
            "seeds", "initialization_cycles", "production_cycles", "blocks", "cutoff_angstrom",
            "fugacity_coefficients", "loading_pressure_bar", "unloading_pressure_bar", "budget"}
    exact_keys(config, keys, keys - {"fugacity_coefficients"})
    if config["schema_version"] != 1:
        raise ValueError("Unsupported GCMC config schema")
    for key in ("structure", "force_field"):
        if not isinstance(config[key], str) or not config[key]:
            raise ValueError(f"{key} must be a file path")
    for key in ("temperature_k", "cutoff_angstrom"):
        positive(config[key], key)
    pressures, seeds = config["pressures_bar"], config["seeds"]
    if not isinstance(pressures, list) or len(pressures) < 2:
        raise ValueError("At least two pressure points are required")
    for value in pressures:
        positive(value, "pressure")
    if len(set(pressures)) != len(pressures):
        raise ValueError("Pressure points must be unique")
    if not isinstance(seeds, list) or not seeds:
        raise ValueError("At least one seed is required")
    for seed in seeds:
        integer(seed, "seed", 0)
    if len(set(seeds)) != len(seeds) or any(s >= 2**32 for s in seeds):
        raise ValueError("Seeds must be unique unsigned 32-bit integers")
    low, high = config["unloading_pressure_bar"], config["loading_pressure_bar"]
    positive(low, "unloading_pressure_bar")
    positive(high, "loading_pressure_bar")
    if low >= high or low not in pressures or high not in pressures:
        raise ValueError("Working-capacity endpoints must be present with unloading < loading")
    phi = config.get("fugacity_coefficients")
    if phi is not None:
        if not isinstance(phi, list) or len(phi) != len(pressures):
            raise ValueError("Fugacity coefficients must match pressure points, or be null for engine EOS")
        for value in phi:
            positive(value, "fugacity_coefficient")
    for key in ("initialization_cycles", "production_cycles", "blocks"):
        integer(config[key], key, 3 if key == "blocks" else 1)
    if config["production_cycles"] < 10 * config["blocks"]:
        raise ValueError("Production must have at least ten cycles per averaging block")
    budget = config["budget"]
    limits = {"max_jobs", "max_total_cycles", "timeout_seconds_per_job", "max_wall_seconds",
              "max_supercell_atoms"}
    exact_keys(budget, limits, limits)
    for key in limits:
        positive(budget[key], key)
    for key in ("max_jobs", "max_total_cycles", "max_supercell_atoms"):
        integer(budget[key], key)
    jobs = len(seeds) * len(pressures)
    cycles = jobs * (config["initialization_cycles"] + config["production_cycles"])
    if jobs > budget["max_jobs"] or cycles > budget["max_total_cycles"]:
        raise ValueError("Requested jobs or cycles exceed the declared budget")
    return config


def validate_force_field(ff):
    exact_keys(ff, {"name", "source", "framework_types", "element_types", "site_types",
                    "methane", "shifted", "tail_corrections"},
               {"name", "source", "framework_types", "methane", "shifted", "tail_corrections"})
    if not all(isinstance(ff[k], str) and ff[k].strip() for k in ("name", "source")):
        raise ValueError("A force-field name and provenance source are required")
    if ("element_types" in ff) == ("site_types" in ff):
        raise ValueError("Provide exactly one of element_types or explicit P1 site_types")
    for key in ("shifted", "tail_corrections"):
        if type(ff[key]) is not bool:
            raise ValueError(f"{key} must be boolean")
    if not isinstance(ff["framework_types"], dict) or not ff["framework_types"]:
        raise ValueError("Explicit framework LJ types are required")
    for name, spec in ff["framework_types"].items():
        if not isinstance(name, str) or not name:
            raise ValueError("Atom type names must be nonempty strings")
        exact_keys(spec, {"element", "epsilon_k", "sigma_angstrom"},
                   {"element", "epsilon_k", "sigma_angstrom"})
        positive(spec["epsilon_k"], "epsilon_k", True)
        positive(spec["sigma_angstrom"], "sigma_angstrom")
    exact_keys(ff["methane"], {"epsilon_k", "sigma_angstrom"}, {"epsilon_k", "sigma_angstrom"})
    positive(ff["methane"]["epsilon_k"], "methane epsilon_k", True)
    positive(ff["methane"]["sigma_angstrom"], "methane sigma_angstrom")
    return ff


def structure_data(path, ff, cutoff, max_atoms):
    import numpy as np
    from ase.data import atomic_masses, atomic_numbers
    from ase.io import read

    if Path(path).suffix.lower() != ".cif":
        raise ValueError("Export an ordered periodic CIF from Crystal-Builder before GCMC")
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        frames = read(str(path), index=":", format="cif", store_tags=True)
    if len(frames) != 1:
        raise ValueError("Exactly one CIF structure is required")
    atoms = frames[0]
    if not len(atoms) or atoms.cell.rank != 3 or not all(atoms.pbc):
        raise ValueError("A nonempty three-dimensional periodic structure is required")
    occupancy = atoms.info.get("occupancy", {})
    if any(len(site) != 1 or any(abs(float(v) - 1) > 1e-8 for v in site.values())
           for site in occupancy.values()):
        raise ValueError("Resolve disorder and partial occupancy before GCMC")
    if not np.isfinite(atoms.positions).all() or not np.isfinite(atoms.cell).all():
        raise ValueError("Nonfinite geometry")
    symbols = atoms.get_chemical_symbols()
    if "site_types" in ff:
        assignments = ff["site_types"]
        if not isinstance(assignments, list) or len(assignments) != len(atoms):
            raise ValueError("site_types must match ASE-expanded P1 atom order exactly")
    else:
        mapping = ff["element_types"]
        if not isinstance(mapping, dict) or set(symbols) - set(mapping):
            raise ValueError("Every framework element requires an explicit LJ type assignment")
        assignments = [mapping[s] for s in symbols]
    for symbol, name in zip(symbols, assignments):
        if not isinstance(name, str) or name not in ff["framework_types"]:
            raise ValueError(f"Missing LJ type {name!r}")
        if ff["framework_types"][name]["element"] != symbol:
            raise ValueError(f"LJ type {name} does not match element {symbol}")
    if any(spec["element"] not in atomic_numbers for spec in ff["framework_types"].values()):
        raise ValueError("Unrecognized element in force-field types")
    # Perpendicular face separations, not edge lengths, also cover skewed cells.
    widths = 1 / np.linalg.norm(np.linalg.inv(atoms.cell.array), axis=0)
    repeats = np.maximum(1, np.ceil(2 * cutoff / widths - 1e-12)).astype(int)
    n_atoms = int(len(atoms) * np.prod(repeats))
    if n_atoms > max_atoms:
        raise ValueError(f"Required supercell has {n_atoms} atoms, exceeding the atom budget")
    # Crystal-Builder input is already prepared. Do not infer repairs or atom types.
    return {"symbols": symbols, "fractional_positions": atoms.get_scaled_positions().tolist(),
            "reader_warnings": [str(w.message) for w in captured],
            "cell_parameters": atoms.cell.cellpar().tolist(), "type_assignments": assignments,
            "unit_cells": repeats.tolist(), "supercell_atoms": n_atoms,
            "supercell_volume_angstrom3": atoms.get_volume() * int(np.prod(repeats)),
            "framework_mass_u": sum(float(atomic_masses[atomic_numbers[s]]) for s in symbols)
                                * int(np.prod(repeats))}


def git_revision(directory):
    try:
        commit = subprocess.check_output(["git", "-C", str(directory), "rev-parse", "HEAD"],
                                         text=True, stderr=subprocess.DEVNULL).strip()
        dirty = bool(subprocess.check_output(["git", "-C", str(directory), "status", "--porcelain"], text=True))
        return {"commit": commit, "dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return None


def code_hashes():
    return {p.name: sha256(p) for p in sorted(Path(__file__).parent.glob("*.py"))}


def job_plan(config):
    jobs = [{"id": f"p{p:03d}-s{s:03d}", "pressure_bar": pressure,
             "replicate_seed": seed, "engine_seed": (seed + p * 1000003) % 2**32,
             "fugacity_coefficient": None if config.get("fugacity_coefficients") is None
                 else config["fugacity_coefficients"][p], "status": "pending"}
            for p, pressure in enumerate(config["pressures_bar"])
            for s, seed in enumerate(config["seeds"])]
    if len({j["engine_seed"] for j in jobs}) != len(jobs):
        raise ValueError("Derived engine seeds collide; select different replicate seeds")
    return jobs


def prepare(config_path, output):
    config_path, output = Path(config_path).resolve(), Path(output).resolve()
    config = validate_config(read_json(config_path))
    source = (config_path.parent / config["structure"]).resolve()
    ff_path = (config_path.parent / config["force_field"]).resolve()
    ff = validate_force_field(read_json(ff_path))
    geometry = structure_data(source, ff, config["cutoff_angstrom"], config["budget"]["max_supercell_atoms"])
    jobs = job_plan(config)
    output.mkdir(parents=True, exist_ok=False)
    (output / "source_snapshot").mkdir()
    for module in Path(__file__).parent.glob("*.py"):
        (output / "source_snapshot" / module.name).write_bytes(module.read_bytes())
    (output / "source.cif").write_bytes(source.read_bytes())
    write_json(output / "config.json", config)
    write_json(output / "force_field.json", ff)
    write_json(output / "structure.json", geometry)
    manifest = {"schema_version": 1, "status": "prepared", "engine": "raspalib",
                "required_engine_version": ENGINE_VERSION, "source_path": str(source),
                "force_field_path": str(ff_path), "repository": git_revision(ROOT),
                "crystal_builder": git_revision(ROOT / "Crystal-Builder"),
                "code_hashes": code_hashes(), "input_hashes":
                    {name: sha256(output / name) for name in
                     ("source.cif", "config.json", "force_field.json", "structure.json")}, "jobs": jobs,
                "reserved_total_cycles": len(jobs) * (config["initialization_cycles"] + config["production_cycles"]),
                "claim": "Engineering simulation output; convergence and scientific validation are separate"}
    write_json(output / "manifest.json", manifest)
    return manifest


def native_statistics(text, cycles, uptake):
    """Require completed native statistics and cross-check the API's mass units."""
    if text.count("Simulation finished!") != 1 or f"Final state after {cycles} cycles" not in text:
        raise ValueError("Native output does not establish completed production")
    tail = text.split("Simulation finished!", 1)[1]
    if "LoadingData" not in tail:
        raise ValueError("Missing final native loading statistics")
    tail = tail.split("LoadingData", 1)[1]
    if "Component 0 (methane)" not in tail:
        raise ValueError("Native output is not the single methane component")
    values = re.findall(r"Abs\. loading average\s+(\S+)\s+\+/-\s+(\S+)\s+\[mol/kg-framework\]", tail)
    if len(values) != 1:
        raise ValueError("Expected exactly one absolute mol/kg-framework loading")
    loading, error = map(float, values[0])
    positive(loading, "native loading", True)
    positive(error, "native block error", True)
    # The native text prints six significant decimal places; 1 mol/kg = 1 mmol/g.
    if not math.isclose(loading, uptake, rel_tol=5e-6, abs_tol=5e-7):
        raise ValueError("Native mol/kg and API molecule-count normalization disagree")
    phi = re.findall(r"Fugacity coefficient:\s+(\S+)\s+\[-\]", text)
    if len(phi) != 1:
        raise ValueError("Expected exactly one native methane fugacity coefficient")
    return {"absolute_uptake_mmol_g": loading, "engine_reported_error_mmol_g": error,
            "fugacity_coefficient": positive(float(phi[0]), "native fugacity coefficient")}


def capacity_summary(rows, config):
    expected = {(p, seed) for p in config["pressures_bar"] for seed in config["seeds"]}
    if len(rows) != len(expected) or {(r["pressure_bar"], r["replicate_seed"]) for r in rows} != expected:
        raise ValueError("A complete, unique pressure-by-seed grid is required")
    for r in rows:
        positive(r["absolute_uptake_mmol_g"], "absolute uptake", True)
        positive(r["engine_reported_error_mmol_g"], "engine reported error", True)

    def moments(values):
        return {"mean": statistics.mean(values), "replicates": len(values),
                "between_seed_standard_error": statistics.stdev(values) / math.sqrt(len(values))
                    if len(values) > 1 else None}

    by_key = {(r["pressure_bar"], r["replicate_seed"]): r for r in rows}
    points = [{"pressure_bar": p, **moments([by_key[p, seed]["absolute_uptake_mmol_g"]
                                          for seed in config["seeds"]])} for p in config["pressures_bar"]]
    low, high = config["unloading_pressure_bar"], config["loading_pressure_bar"]
    delta = [by_key[high, s]["absolute_uptake_mmol_g"] - by_key[low, s]["absolute_uptake_mmol_g"]
             for s in config["seeds"]]
    return {"gas": "methane", "temperature_k": config["temperature_k"],
            "uptake_definition": "absolute", "units": "mmol/g-dry-framework",
            "isotherm": points, "working_capacity": {"loading_pressure_bar": high,
            "unloading_pressure_bar": low, **moments(delta)}, "negative_capacity_observed": any(v < 0 for v in delta),
            "convergence_established": False, "per_job": rows}


def stop_process_tree(process):
    import psutil
    try:
        parent = psutil.Process(process.pid)
        children = parent.children(recursive=True)
        for child in reversed(children):
            try:
                child.kill()
            except psutil.NoSuchProcess:
                pass
        parent.kill()
    except psutil.NoSuchProcess:
        pass
    process.wait(timeout=10)


def _scheduled_job(scheduler, backend, capacity_timeout, output, folder, job, timeout, before):
    """Run one worker through the HPC layer; returns its exit code or raises like the local path."""
    spec = {"name": f"gcmc-{output.parent.name}-{job['id']}"[:120],
            "argv": ["{python}", "-m", "rsi.gcmc.worker", str(output), job["id"]],
            "cwd": str(folder), "log": "engine.log", "resources": {"cpus": 1, "wall_seconds": timeout}}
    try:
        record = scheduler.run(spec, backend, capacity_timeout)
    except (KeyboardInterrupt, SystemExit):
        job.update(status="interrupted", actual_cycles=None, elapsed_seconds=time.monotonic() - before)
        raise
    job["hpc_job"] = {key: record.get(key) for key in ("id", "backend", "native_id", "status", "charged")}
    if record["status"] in {"timeout", "queue_timeout"}:
        job.update(status="timeout", actual_cycles=None, elapsed_seconds=time.monotonic() - before)
        raise TimeoutError(f"GCMC job {record['status']}: {job['id']}")
    if record["status"] not in {"completed", "failed"}:
        job.update(status="interrupted", actual_cycles=None, elapsed_seconds=time.monotonic() - before)
        raise RuntimeError(f"GCMC job ended as {record['status']} on {record['backend']}: {job['id']}")
    # A Slurm failure such as OUT_OF_MEMORY can report exit code 0; it is still a failure.
    return (record.get("exit_code") or -1) if record["status"] == "failed" else 0


def execute(output, scheduler=None, backend=None, capacity_timeout=0.0):
    """Run every reserved pressure/seed job in order and summarize capacity.

    With an ``rsi.hpc.Scheduler`` each job is a one-CPU scheduled worker (local or Slurm) charged to
    that session's budget; per-job verification stays on the controller.
    """
    output = Path(output).resolve()
    manifest = read_json(output / "manifest.json")
    if manifest["status"] != "prepared":
        raise ValueError("This directory has already been attempted; prepare a new run")
    if manifest["code_hashes"] != code_hashes():
        raise ValueError("GCMC source changed after preparation")
    for name, digest in manifest["code_hashes"].items():
        if sha256(output / "source_snapshot" / name) != digest:
            raise ValueError("Archived GCMC source changed after preparation")
    for name, digest in manifest["input_hashes"].items():
        if sha256(output / name) != digest:
            raise ValueError(f"Frozen input changed: {name}")
    config = validate_config(read_json(output / "config.json"))
    if manifest["jobs"] != job_plan(config):
        raise ValueError("Reserved jobs do not match the frozen config")
    import psutil  # Require process-tree containment before starting native code.
    version = importlib.metadata.version("raspalib")
    if version != ENGINE_VERSION:
        raise ValueError(f"This adapter requires raspalib=={ENGINE_VERSION}; found {version}")
    packages = {n: importlib.metadata.version(n) for n in ("raspalib", "ase", "numpy", "psutil")}
    dist = importlib.metadata.distribution("raspalib")
    binaries = {str(p): sha256(dist.locate_file(p)) for p in dist.files or []
                if str(p).endswith((".so", ".pyd", ".dylib"))}
    if not binaries:
        raise ValueError("Cannot fingerprint the installed native RASPA3 binary")
    manifest.update(status="running", packages=packages, engine_binary_hashes=binaries,
                    python_version=sys.version, python_executable=sys.executable)
    started, rows = time.monotonic(), []
    write_json(output / "manifest.json", manifest)
    try:
        for job in manifest["jobs"]:
            remaining = config["budget"]["max_wall_seconds"] - (time.monotonic() - started)
            if remaining <= 0:
                raise TimeoutError("Total GCMC wall budget exhausted")
            folder = output / job["id"]
            folder.mkdir(exist_ok=False)
            write_json(folder / "job.json", job)
            job.update(status="reserved", requested_cycles=config["initialization_cycles"] + config["production_cycles"])
            write_json(output / "manifest.json", manifest)
            env = dict(os.environ, PYTHONPATH=str(ROOT) + os.pathsep + os.environ.get("PYTHONPATH", ""),
                       OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
            # Preserve the import root even when the worker cwd is the run directory.
            before = time.monotonic()
            timeout = min(remaining, config["budget"]["timeout_seconds_per_job"])
            if scheduler is not None:
                code = _scheduled_job(scheduler, backend, capacity_timeout, output, folder, job, timeout, before)
            else:
                with (folder / "engine.log").open("w", encoding="utf-8") as log:
                    process = subprocess.Popen([sys.executable, "-m", "rsi.gcmc.worker", str(output), job["id"]],
                                               cwd=folder, env=env, stdout=log, stderr=subprocess.STDOUT)
                    try:
                        code = process.wait(timeout=timeout)
                    except subprocess.TimeoutExpired:
                        stop_process_tree(process)
                        job.update(status="timeout", actual_cycles=None, elapsed_seconds=time.monotonic() - before)
                        raise TimeoutError(f"GCMC job timed out: {job['id']}") from None
                    except BaseException:
                        stop_process_tree(process)
                        job.update(status="interrupted", actual_cycles=None, elapsed_seconds=time.monotonic() - before)
                        raise
            job.update(elapsed_seconds=time.monotonic() - before, exit_code=code)
            if code != 0:
                job.update(status="failed", actual_cycles=None)
                raise RuntimeError(f"GCMC job failed: {job['id']}; see engine.log")
            result_path = folder / "result.json"
            row = read_json(result_path)
            if (row["engine_version"] != version or row["completed_production_cycles"] != config["production_cycles"]
                    or row["pressure_bar"] != job["pressure_bar"] or row["replicate_seed"] != job["replicate_seed"]
                    or row["engine_seed"] != job["engine_seed"]):
                raise ValueError("Worker output does not match the reserved job")
            positive(row["absolute_uptake_mmol_g"], "absolute uptake", True)
            positive(row["engine_reported_error_mmol_g"], "engine error", True)
            native_path = folder / "output" / row["engine_output_file"]
            if native_path.parent != folder / "output" or sha256(native_path) != row["engine_output_sha256"]:
                raise ValueError("Native engine output identity mismatch")
            native_statistics(native_path.read_text(encoding="utf-8"), config["production_cycles"], row["absolute_uptake_mmol_g"])
            rows.append(row)
            job.update(status="completed", actual_cycles=job["requested_cycles"], result_sha256=sha256(result_path))
            write_json(output / "manifest.json", manifest)
        if time.monotonic() - started > config["budget"]["max_wall_seconds"]:
            raise TimeoutError("Total GCMC wall budget exhausted")
        summary = capacity_summary(rows, config)
        write_json(output / "capacity.json", summary)
        manifest.update(status="completed", capacity_sha256=sha256(output / "capacity.json"))
    except BaseException as exc:
        manifest.update(status="interrupted" if isinstance(exc, (KeyboardInterrupt, SystemExit)) else "failed",
                        error=f"{type(exc).__name__}: {exc}")
        for job in manifest["jobs"]:
            if job["status"] == "reserved":
                job.update(status="failed", actual_cycles=None)
        raise
    finally:
        manifest["elapsed_seconds"] = time.monotonic() - started
        write_json(output / "manifest.json", manifest)
    return summary

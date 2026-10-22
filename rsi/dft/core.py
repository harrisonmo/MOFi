"""Prepare, execute, resume and verify bounded ASE/GPAW/CP2K energy, force and relaxation jobs."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import shlex
import shutil
import signal
import subprocess
import sys
import time

from .parse import parse_native


ROOT = Path(__file__).resolve().parents[2]
BACKENDS = {"gpaw", "cp2k", "emt"}
# EMT is an analytic test surrogate. Its results are engineering evidence only, never DFT.
DFT_BACKENDS = {"gpaw", "cp2k"}
TASKS = {"single_point", "optimize"}
OPTIMIZERS = {"BFGS", "LBFGS", "FIRE"}
RESUMABLE = {"timeout", "interrupted"}
STRUCTURE_SUFFIXES = {".cif", ".xyz", ".extxyz", ".traj", ".json"}
CALCULATOR_KEYS = {
    "gpaw": ({"mode", "xc", "kpts", "h", "basis", "convergence", "occupations", "maxiter",
              "spinpol", "symmetry"}, {"mode", "xc"}),
    "cp2k": ({"executable", "xc", "cutoff_ev", "basis_set", "pseudo_potential", "basis_set_file",
              "potential_file", "max_scf", "poisson_solver", "inp"},
             {"executable", "xc", "cutoff_ev", "basis_set", "pseudo_potential"}),
    "emt": ({"delay_seconds_per_step"}, set()),
}
NATIVE_LOG = {"gpaw": "gpaw.txt", "cp2k": "cp2k.out"}
FIXED_TOLERANCE_ANGSTROM = 1e-6
ENERGY_TOLERANCE_EV = 1e-4
FORCE_PRINT_TOLERANCE = 2e-5


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


def validate_calculator(backend, params, charge, multiplicity):
    allowed, required = CALCULATOR_KEYS[backend]
    exact_keys(params, allowed, required)
    if backend == "emt":
        if charge != 0 or multiplicity != 1:
            raise ValueError("The EMT test surrogate has no charge or spin")
        positive(params.get("delay_seconds_per_step", 0), "delay_seconds_per_step", True)
        return
    if not isinstance(params["xc"], str) or not params["xc"]:
        raise ValueError("An explicit exchange-correlation functional is required")
    if backend == "gpaw":
        mode = params["mode"]
        if not isinstance(mode, dict) or mode.get("name") not in {"pw", "lcao", "fd"}:
            raise ValueError("GPAW mode must be an object with name pw, lcao or fd")
        if mode["name"] == "pw":
            exact_keys(mode, {"name", "ecut"}, {"name", "ecut"})
            positive(mode["ecut"], "GPAW plane-wave ecut")
        else:
            exact_keys(mode, {"name"}, {"name"})
        if mode["name"] == "fd" and "h" not in params:
            raise ValueError("GPAW fd mode requires an explicit grid spacing h")
        if mode["name"] == "lcao" and "basis" not in params:
            raise ValueError("GPAW lcao mode requires an explicit basis")
        if "kpts" in params and (not isinstance(params["kpts"], list) or len(params["kpts"]) != 3
                                 or any(type(k) is not int or k < 1 for k in params["kpts"])):
            raise ValueError("GPAW kpts must be three positive integers")
        if multiplicity != 1 and params.get("spinpol") is not True:
            raise ValueError("Open-shell GPAW jobs require spinpol true")
        if "maxiter" in params:
            integer(params["maxiter"], "GPAW maxiter")
    else:
        positive(params["cutoff_ev"], "CP2K cutoff_ev")
        for key in ("executable", "basis_set", "pseudo_potential"):
            if not isinstance(params[key], str) or not params[key]:
                raise ValueError(f"CP2K {key} must be a nonempty string")
        if "max_scf" in params:
            integer(params["max_scf"], "CP2K max_scf")


def validate_config(config):
    keys = {"schema_version", "backend", "task", "structure", "charge", "spin_multiplicity",
            "calculator", "constraints", "optimizer", "initial_magmoms", "budget"}
    exact_keys(config, keys, keys - {"optimizer", "initial_magmoms", "constraints"})
    if config["schema_version"] != 1:
        raise ValueError("Unsupported DFT config schema")
    if config["backend"] not in BACKENDS or config["task"] not in TASKS:
        raise ValueError(f"backend must be one of {sorted(BACKENDS)}; task one of {sorted(TASKS)}")
    if not isinstance(config["structure"], str) or Path(config["structure"]).suffix.lower() not in STRUCTURE_SUFFIXES:
        raise ValueError(f"structure must be a path ending in one of {sorted(STRUCTURE_SUFFIXES)}")
    integer(config["charge"], "charge", -1000)
    integer(config["spin_multiplicity"], "spin_multiplicity")
    validate_calculator(config["backend"], config["calculator"], config["charge"], config["spin_multiplicity"])
    magmoms = config.get("initial_magmoms")
    if magmoms is not None:
        if not isinstance(magmoms, list) or not all(type(m) in (int, float) and math.isfinite(m) for m in magmoms):
            raise ValueError("initial_magmoms must be finite numbers")
        if not math.isclose(sum(magmoms), config["spin_multiplicity"] - 1, abs_tol=1e-8):
            raise ValueError("initial_magmoms must sum to spin_multiplicity - 1")
    constraints = config.get("constraints", {})
    exact_keys(constraints, {"fixed_atoms", "fixed_elements", "fixed_cartesian"}, set())
    for index in constraints.get("fixed_atoms", []):
        integer(index, "fixed atom index", 0)
    if not all(isinstance(s, str) and s for s in constraints.get("fixed_elements", [])):
        raise ValueError("fixed_elements must be element symbols")
    for item in constraints.get("fixed_cartesian", []):
        exact_keys(item, {"index", "mask"}, {"index", "mask"})
        integer(item["index"], "fixed_cartesian index", 0)
        if not isinstance(item["mask"], list) or len(item["mask"]) != 3 or not all(type(m) is bool for m in item["mask"]):
            raise ValueError("fixed_cartesian mask must be three booleans (true = fixed)")
    if config["task"] == "optimize":
        if "optimizer" not in config:
            raise ValueError("Optimization requires explicit optimizer settings")
        opt = config["optimizer"]
        exact_keys(opt, {"name", "fmax_ev_per_angstrom", "max_steps", "maxstep_angstrom"},
                   {"name", "fmax_ev_per_angstrom", "max_steps"})
        if opt["name"] not in OPTIMIZERS:
            raise ValueError(f"optimizer must be one of {sorted(OPTIMIZERS)}")
        positive(opt["fmax_ev_per_angstrom"], "fmax_ev_per_angstrom")
        integer(opt["max_steps"], "max_steps")
        if "maxstep_angstrom" in opt:
            positive(opt["maxstep_angstrom"], "maxstep_angstrom")
    elif "optimizer" in config:
        raise ValueError("Single points take no optimizer")
    budget = config["budget"]
    limits = {"timeout_seconds_per_attempt", "max_wall_seconds", "max_atoms", "max_restarts", "threads"}
    exact_keys(budget, limits, limits)
    positive(budget["timeout_seconds_per_attempt"], "timeout_seconds_per_attempt")
    positive(budget["max_wall_seconds"], "max_wall_seconds")
    integer(budget["max_atoms"], "max_atoms")
    integer(budget["max_restarts"], "max_restarts", 0)
    integer(budget["threads"], "threads")
    return config


def structure_data(path, config):
    import numpy as np
    from ase.io import read

    frames = read(str(path), index=":")
    if len(frames) != 1:
        raise ValueError("Exactly one input structure is required")
    atoms = frames[0]
    n = len(atoms)
    if not n or n > config["budget"]["max_atoms"]:
        raise ValueError(f"Structure has {n} atoms; the atom budget is {config['budget']['max_atoms']}")
    if not np.isfinite(atoms.positions).all() or not np.isfinite(atoms.cell.array).all():
        raise ValueError("Nonfinite geometry")
    if any(atoms.pbc) and atoms.cell.rank != 3:
        raise ValueError("Periodic directions require a full three-dimensional cell")
    if not any(atoms.pbc) and atoms.cell.rank != 3:
        raise ValueError("Nonperiodic DFT still requires an explicit vacuum box")
    symbols = atoms.get_chemical_symbols()
    constraints = config.get("constraints", {})
    fixed = set(constraints.get("fixed_atoms", []))
    unknown = set(constraints.get("fixed_elements", [])) - set(symbols)
    if unknown:
        raise ValueError(f"fixed_elements not present in structure: {sorted(unknown)}")
    fixed |= {i for i, s in enumerate(symbols) if s in constraints.get("fixed_elements", [])}
    cartesian = {item["index"]: item["mask"] for item in constraints.get("fixed_cartesian", [])}
    if len(cartesian) != len(constraints.get("fixed_cartesian", [])):
        raise ValueError("Duplicate fixed_cartesian index")
    if any(i >= n for i in fixed | set(cartesian)):
        raise ValueError("Constraint index outside the structure")
    if fixed & set(cartesian):
        raise ValueError("An atom cannot be both fully fixed and partially fixed")
    if config["task"] == "optimize" and len(fixed) == n:
        raise ValueError("Every atom is fixed; nothing can be optimized")
    electrons = int(atoms.numbers.sum()) - config["charge"]
    if config["backend"] in DFT_BACKENDS and (electrons < 0 or electrons % 2 != (config["spin_multiplicity"] - 1) % 2):
        raise ValueError(f"{electrons} electrons are incompatible with multiplicity {config['spin_multiplicity']}")
    magmoms = config.get("initial_magmoms")
    if magmoms is not None and len(magmoms) != n:
        raise ValueError("initial_magmoms must match the atom count")
    # The prepared structure is the identity for every attempt; no repair or atom typing occurs.
    return {"symbols": symbols, "positions": atoms.positions.tolist(), "cell": atoms.cell.array.tolist(),
            "pbc": [bool(p) for p in atoms.pbc], "fixed_atoms": sorted(fixed),
            "fixed_cartesian": [{"index": i, "mask": cartesian[i]} for i in sorted(cartesian)]}


def free_mask(structure):
    """Boolean (n, 3) array: True where a Cartesian component is free to move."""
    import numpy as np

    mask = np.ones((len(structure["symbols"]), 3), dtype=bool)
    mask[structure["fixed_atoms"]] = False
    for item in structure["fixed_cartesian"]:
        mask[item["index"]] = [not m for m in item["mask"]]
    return mask


def git_revision(directory):
    try:
        commit = subprocess.check_output(["git", "-c", "safe.directory=*", "-C", str(directory), "rev-parse", "HEAD"],
                                         text=True, stderr=subprocess.DEVNULL).strip()
        dirty = bool(subprocess.check_output(["git", "-c", "safe.directory=*", "-C", str(directory),
                                              "status", "--porcelain"], text=True))
        return {"commit": commit, "dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return None


def code_hashes():
    return {p.name: sha256(p) for p in sorted(Path(__file__).parent.glob("*.py"))}


def prepare(config_path, output):
    config_path, output = Path(config_path).resolve(), Path(output).resolve()
    config = validate_config(read_json(config_path))
    source = (config_path.parent / config["structure"]).resolve()
    structure = structure_data(source, config)
    output.mkdir(parents=True, exist_ok=False)
    (output / "source_snapshot").mkdir()
    for module in Path(__file__).parent.glob("*.py"):
        (output / "source_snapshot" / module.name).write_bytes(module.read_bytes())
    (output / f"source{source.suffix.lower()}").write_bytes(source.read_bytes())
    write_json(output / "config.json", config)
    write_json(output / "structure.json", structure)
    manifest = {"schema_version": 1, "status": "prepared", "backend": config["backend"],
                "is_dft": config["backend"] in DFT_BACKENDS, "task": config["task"],
                "source_path": str(source), "repository": git_revision(ROOT),
                "crystal_builder": git_revision(ROOT / "Crystal-Builder"), "code_hashes": code_hashes(),
                "input_hashes": {name: sha256(output / name) for name in
                                 (f"source{source.suffix.lower()}", "config.json", "structure.json")},
                "attempts": [], "steps_used": 0, "elapsed_seconds": 0.0,
                "claim": "Engineering calculation output; method adequacy and scientific validation are separate"}
    write_json(output / "manifest.json", manifest)
    return manifest


def verify_frozen(output, manifest):
    if manifest["code_hashes"] != code_hashes():
        raise ValueError("DFT source changed after preparation")
    for name, digest in manifest["code_hashes"].items():
        if sha256(output / "source_snapshot" / name) != digest:
            raise ValueError("Archived DFT source changed after preparation")
    for name, digest in manifest["input_hashes"].items():
        if sha256(output / name) != digest:
            raise ValueError(f"Frozen input changed: {name}")
    return validate_config(read_json(output / "config.json"))


def environment_record(config):
    names = ["ase", "numpy"] + (["gpaw"] if config["backend"] == "gpaw" else [])
    packages = {}
    for name in names:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            raise ImportError(f"Backend {config['backend']} requires the {name} package") from None
    record = {"packages": packages, "python_version": sys.version, "python_executable": sys.executable}
    if config["backend"] == "cp2k":
        # Recent CP2K releases replace cp2k_shell with "cp2k.psmp --shell"; hash the binary itself.
        executable = shutil.which(shlex.split(config["calculator"]["executable"])[0])
        if executable is None:
            raise ImportError(f"CP2K executable not found: {config['calculator']['executable']}")
        # Basis and potential files resolve through this variable when the compiled-in path is unusable.
        record.update(cp2k_executable=executable, cp2k_executable_sha256=sha256(executable),
                      cp2k_data_dir=os.environ.get("CP2K_DATA_DIR"))
    return record


def stop_process_tree(process):
    if process.poll() is not None:
        return
    if os.name == "nt":
        subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
    else:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    process.wait(timeout=30)


def _scheduled_attempt(scheduler, backend, capacity_timeout, output, folder, attempt, threads, timeout):
    """Run the worker through the HPC layer; returns its exit code or raises like the local path."""
    spec = {"name": f"dft-{output.name}-{attempt['id']}"[:120],
            "argv": ["{python}", "-m", "rsi.dft.worker", str(output), attempt["id"]],
            "cwd": str(folder), "log": "worker.log",
            "resources": {"cpus": threads, "wall_seconds": timeout}}
    try:
        job = scheduler.run(spec, backend, capacity_timeout)
    except (KeyboardInterrupt, SystemExit):
        attempt["status"] = "interrupted"
        raise
    attempt["hpc_job"] = {key: job.get(key) for key in ("id", "backend", "native_id", "status", "charged")}
    if job["status"] in {"timeout", "queue_timeout"}:
        attempt["status"] = "timeout"
        raise TimeoutError(f"DFT attempt {job['status']} after {timeout:.1f} s: {attempt['id']}")
    if job["status"] not in {"completed", "failed"}:
        attempt["status"] = "interrupted"
        raise RuntimeError(f"DFT attempt ended as {job['status']} on {job['backend']}: {attempt['id']}")
    # A Slurm failure such as OUT_OF_MEMORY can report exit code 0; it is still a failure.
    return (job.get("exit_code") or -1) if job["status"] == "failed" else 0


def latest_checkpoint(output, manifest):
    """Return the newest verified checkpoint from earlier attempts, or None."""
    for attempt in reversed(manifest["attempts"]):
        if attempt.get("checkpoint_sha256"):
            path = output / attempt["id"] / "checkpoint.json"
            if sha256(path) != attempt["checkpoint_sha256"]:
                raise ValueError(f"Checkpoint changed after attempt {attempt['id']}")
            return attempt["id"], read_json(path)
    return None


def verify_attempt(output, folder, config, structure):
    """Recompute convergence from saved forces and native logs, independent of the worker verdict."""
    import numpy as np

    result = read_json(folder / "result.json")
    checkpoint = read_json(folder / "checkpoint.json")
    if result["checkpoint_step"] != checkpoint["global_step"]:
        raise ValueError("Worker result does not describe its final checkpoint")
    positions = np.array(checkpoint["positions"])
    raw = np.array(checkpoint["raw_forces_ev_per_angstrom"])
    n = len(structure["symbols"])
    if positions.shape != (n, 3) or raw.shape != (n, 3) or not np.isfinite(raw).all():
        raise ValueError("Checkpoint geometry or forces have the wrong shape")
    energy = checkpoint["energy_ev"]
    positive(abs(energy), "energy", True)
    mask = free_mask(structure)
    free = np.where(mask, raw, 0.0)
    max_free = float(np.linalg.norm(free, axis=1).max())
    moved = np.abs(positions - np.array(structure["positions"]))
    if (moved[~mask] > FIXED_TOLERANCE_ANGSTROM).any():
        raise ValueError("A constrained coordinate moved")
    checks = {"max_free_force_ev_per_angstrom": max_free,
              "max_raw_force_ev_per_angstrom": float(np.linalg.norm(raw, axis=1).max()),
              "max_constrained_displacement_angstrom": float(moved[~mask].max()) if (~mask).any() else 0.0}
    if config["backend"] in NATIVE_LOG:
        log = folder / NATIVE_LOG[config["backend"]]
        native = parse_native(config["backend"], log.read_text(encoding="utf-8", errors="replace"))
        if not native["energies_ev"]:
            raise ValueError("Native log reports no energy")
        if abs(native["energies_ev"][-1] - energy) > ENERGY_TOLERANCE_EV:
            raise ValueError("Native log energy and ASE energy disagree")
        if native["last_forces_ev_per_angstrom"] is not None:
            # The native table prints five decimals of the unconstrained forces.
            printed = np.array(native["last_forces_ev_per_angstrom"])
            if printed.shape != raw.shape or np.abs(printed - raw).max() > FORCE_PRINT_TOLERANCE:
                raise ValueError("Native log forces and ASE forces disagree")
            checks["native_force_table_checked"] = True
        checks.update(native_log=log.name, native_log_sha256=sha256(log), scf_all_converged=native["scf_all_converged"],
                      scf_not_converged=native["scf_not_converged"], scf_iterations=native["scf_iterations"],
                      native_last_energy_ev=native["energies_ev"][-1])
    else:
        checks.update(scf_all_converged=None, scf_note="EMT test surrogate has no SCF")
    if config["task"] == "optimize":
        fmax = config["optimizer"]["fmax_ev_per_angstrom"]
        checks.update(force_rule="max |F| over free components <= fmax", fmax_ev_per_angstrom=fmax,
                      force_converged=max_free <= fmax)
    return result, checkpoint, checks


def summarize(config, result, checkpoint, checks, manifest):
    scf_ok = checks["scf_all_converged"] is not False
    converged = scf_ok and (config["task"] == "single_point" or checks["force_converged"])
    return {"backend": config["backend"], "is_dft": config["backend"] in DFT_BACKENDS, "task": config["task"],
            "energy_ev": checkpoint["energy_ev"], "forces_ev_per_angstrom": checkpoint["forces_ev_per_angstrom"],
            "positions_angstrom": checkpoint["positions"], "cell_angstrom": checkpoint["cell"],
            "steps": checkpoint["global_step"], "attempts": len(manifest["attempts"]),
            "force_evaluations": sum(a.get("force_evaluations") or 0 for a in manifest["attempts"]),
            "optimizer_reported_converged": result.get("optimizer_converged"),
            "converged": converged, "checks": checks,
            "units": {"energy": "eV", "force": "eV/Angstrom", "length": "Angstrom"}}


def execute(output, resume=False, scheduler=None, backend=None, capacity_timeout=0.0):
    """Run one attempt. With resume=True, continue a timed-out or interrupted run from its checkpoint.

    With an ``rsi.hpc.Scheduler`` the worker runs as a scheduled job (local or Slurm) charged to that
    session's budget; ``budget.threads`` becomes the job's CPU count.
    """
    output = Path(output).resolve()
    manifest = read_json(output / "manifest.json")
    expected = RESUMABLE if resume else {"prepared"}
    if manifest["status"] not in expected:
        raise ValueError(f"Run status {manifest['status']!r} cannot {'resume' if resume else 'start'}; "
                         "prepare a new run instead")
    config = verify_frozen(output, manifest)
    structure = read_json(output / "structure.json")
    budget = config["budget"]
    if resume and len(manifest["attempts"]) > budget["max_restarts"]:
        raise ValueError("Restart budget exhausted")
    remaining_wall = budget["max_wall_seconds"] - manifest["elapsed_seconds"]
    if remaining_wall <= 0:
        raise TimeoutError("Total DFT wall budget exhausted")
    start = latest_checkpoint(output, manifest) if resume else None
    max_steps = config["optimizer"]["max_steps"] if config["task"] == "optimize" else 0
    attempt = {"id": f"attempt-{len(manifest['attempts']) + 1:03d}", "status": "reserved",
               "resumed_from": start[0] if start else None,
               "start_step": start[1]["global_step"] if start else 0}
    if config["task"] == "optimize" and attempt["start_step"] >= max_steps:
        raise ValueError("Optimization step budget exhausted")
    attempt["steps_allowed"] = max_steps - attempt["start_step"]
    manifest.update(environment_record(config))
    folder = output / attempt["id"]
    folder.mkdir(exist_ok=False)
    write_json(folder / "attempt.json", {**attempt, "start_checkpoint": start[1] if start else None})
    manifest["attempts"].append(attempt)
    manifest["status"] = "running"
    write_json(output / "manifest.json", manifest)
    env = dict(os.environ, PYTHONPATH=str(ROOT) + os.pathsep + os.environ.get("PYTHONPATH", ""),
               OMP_NUM_THREADS=str(budget["threads"]), OPENBLAS_NUM_THREADS=str(budget["threads"]),
               MKL_NUM_THREADS=str(budget["threads"]))
    timeout = min(remaining_wall, budget["timeout_seconds_per_attempt"])
    before = time.monotonic()
    try:
        if scheduler is not None:
            code = _scheduled_attempt(scheduler, backend, capacity_timeout, output, folder, attempt,
                                      budget["threads"], timeout)
        else:
            with (folder / "worker.log").open("w", encoding="utf-8") as log:
                process = subprocess.Popen([sys.executable, "-m", "rsi.dft.worker", str(output), attempt["id"]],
                                           cwd=folder, env=env, stdout=log, stderr=subprocess.STDOUT,
                                           start_new_session=os.name != "nt")
                try:
                    code = process.wait(timeout=timeout)
                except subprocess.TimeoutExpired:
                    stop_process_tree(process)
                    attempt["status"] = "timeout"
                    raise TimeoutError(f"DFT attempt timed out after {timeout:.1f} s: {attempt['id']}") from None
                except BaseException:
                    stop_process_tree(process)
                    attempt["status"] = "interrupted"
                    raise
        attempt["exit_code"] = code
        if code != 0:
            attempt["status"] = "failed"
            raise RuntimeError(f"DFT attempt failed: {attempt['id']}; see worker.log")
        result, checkpoint, checks = verify_attempt(output, folder, config, structure)
        attempt.update(status="completed", result_sha256=sha256(folder / "result.json"),
                       force_evaluations=result["force_evaluations"])
        manifest["steps_used"] = checkpoint["global_step"]
        summary = summarize(config, result, checkpoint, checks, manifest)
        write_json(output / "summary.json", summary)
        manifest.update(status="completed", converged=summary["converged"],
                        summary_sha256=sha256(output / "summary.json"))
        # Earlier attempt errors stay on their attempts; the run-level error describes the current state.
        manifest.pop("error", None)
    except BaseException as exc:
        if attempt["status"] == "reserved":
            attempt["status"] = "failed"
        attempt["error"] = f"{type(exc).__name__}: {exc}"
        manifest.update(status=attempt["status"], error=attempt["error"])
        raise
    finally:
        attempt["elapsed_seconds"] = time.monotonic() - before
        manifest["elapsed_seconds"] += attempt["elapsed_seconds"]
        checkpoint_path = folder / "checkpoint.json"
        if checkpoint_path.exists():
            # A checkpoint is written atomically, so even a killed worker leaves a complete one.
            saved = read_json(checkpoint_path)
            attempt.update(checkpoint_sha256=sha256(checkpoint_path), checkpoint_step=saved["global_step"])
            attempt.setdefault("force_evaluations", saved["force_evaluations"])
            manifest["steps_used"] = max(manifest["steps_used"], saved["global_step"])
        write_json(output / "manifest.json", manifest)
    return summary

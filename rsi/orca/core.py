"""Prepare, execute and verify bounded ORCA molecular DFT jobs.

ORCA is molecular: the structure must be nonperiodic (a molecule or a cut cluster). The input is
written by Crystal-Builder's validated writer (``xtal.orca``), so the functional and basis must be in
its ORCA 6.1 catalogue and an impossible spin is refused before anything runs. ORCA itself runs as
one native job (local or through ``rsi.hpc``); the controller then re-reads ORCA's own files and
recomputes every reported number, independent of the exit code.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import time

from .parse import BOHR_ANGSTROM, HARTREE_EV, parse_engrad, parse_output, parse_xyz


ROOT = Path(__file__).resolve().parents[2]
CRYSTAL_BUILDER_ORCA = ROOT / "Crystal-Builder/xtal/orca"
ENV_VAR = "RSI_ORCA"
TASKS = {"energy", "gradient", "optimize"}
OPT_LEVELS = {"LooseOpt", "Opt", "TightOpt", "VeryTightOpt"}
DISPERSIONS = {"D3BJ", "D3ZERO", "D4"}
RI = {"RIJCOSX", "RI", "RIJK", "NORI"}
SCF_THRESHOLDS = {"NormalSCF", "StrongSCF", "TightSCF", "VeryTightSCF"}
STRUCTURE_SUFFIXES = {".xyz", ".extxyz"}
BASENAME = "job"
ENERGY_TOLERANCE_EH = 1e-8


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
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def exact_keys(value, allowed, required):
    if not isinstance(value, dict) or set(value) - set(allowed) or set(required) - set(value):
        raise ValueError(f"Expected required keys {sorted(required)}; allowed keys {sorted(allowed)}")


def integer(value, name, minimum=1):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def positive(value, name):
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return value


def binary():
    """The ORCA executable, or None. Its folder must also hold orca_scf, which tells the quantum
    chemistry program apart from the GNOME screen reader that Ubuntu also installs as ``orca``."""
    value = os.environ.get(ENV_VAR)
    if not value:
        return None
    path = Path(value)
    # ORCA starts its parallel subprograms by the path it was called with, so it must be absolute.
    if not path.is_absolute() or not path.is_file():
        return None
    if not any((path.parent / name).is_file() for name in ("orca_scf", "orca_scf.exe")):
        return None
    return path


def _catalogue():
    from xtal.orca import catalogue
    return catalogue


def validate_config(config):
    keys = {"schema_version", "structure", "charge", "spin_multiplicity", "method", "task",
            "frequencies", "optimization", "budget"}
    exact_keys(config, keys, keys - {"optimization"})
    if config["schema_version"] != 1:
        raise ValueError("Unsupported ORCA config schema")
    if not isinstance(config["structure"], str) or Path(config["structure"]).suffix.lower() not in STRUCTURE_SUFFIXES:
        raise ValueError(f"structure must be a path ending in one of {sorted(STRUCTURE_SUFFIXES)}")
    integer(config["charge"], "charge", -1000)
    integer(config["spin_multiplicity"], "spin_multiplicity")
    if config["task"] not in TASKS:
        raise ValueError(f"task must be one of {sorted(TASKS)}")
    if type(config["frequencies"]) is not bool:
        raise ValueError("frequencies must be a boolean")
    method = config["method"]
    exact_keys(method, {"functional", "basis", "dispersion", "ri", "scf_convergence"},
               {"functional", "basis", "dispersion", "ri", "scf_convergence"})
    catalogue = _catalogue()
    functional = catalogue.functional(method["functional"]) if isinstance(method["functional"], str) else None
    if functional is None:
        raise ValueError(f"{method['functional']!r} is not an ORCA 6.1 functional in the catalogue")
    if functional.own_basis:
        if method["basis"] is not None:
            raise ValueError(f"{functional.key} brings its own basis set; pass basis null")
    elif not isinstance(method["basis"], str) or catalogue.basis(method["basis"]) is None:
        raise ValueError(f"{method['basis']!r} is not an ORCA 6.1 basis set in the catalogue")
    if method["dispersion"] is not None:
        if method["dispersion"] not in DISPERSIONS:
            raise ValueError(f"dispersion must be one of {sorted(DISPERSIONS)} or null")
        if functional.carries_dispersion:
            raise ValueError(f"{functional.key} already includes dispersion; pass dispersion null")
    if method["ri"] is not None and method["ri"] not in RI:
        raise ValueError(f"ri must be one of {sorted(RI)} or null")
    if method["scf_convergence"] is not None and method["scf_convergence"] not in SCF_THRESHOLDS:
        raise ValueError(f"scf_convergence must be one of {sorted(SCF_THRESHOLDS)} or null")
    if config["task"] == "optimize":
        if "optimization" not in config:
            raise ValueError("optimize requires explicit optimization settings")
        exact_keys(config["optimization"], {"level", "max_cycles"}, {"level", "max_cycles"})
        if config["optimization"]["level"] not in OPT_LEVELS:
            raise ValueError(f"optimization level must be one of {sorted(OPT_LEVELS)}")
        integer(config["optimization"]["max_cycles"], "max_cycles")
    elif "optimization" in config:
        raise ValueError("Only optimize takes optimization settings")
    budget = config["budget"]
    exact_keys(budget, {"cpus", "maxcore_mb", "wall_seconds", "max_atoms"},
               {"cpus", "maxcore_mb", "wall_seconds", "max_atoms"})
    integer(budget["cpus"], "cpus")
    integer(budget["maxcore_mb"], "maxcore_mb", 100)
    positive(budget["wall_seconds"], "wall_seconds")
    integer(budget["max_atoms"], "max_atoms")
    return config


def read_molecule(path, max_atoms):
    """Symbols and Cartesian positions of one nonperiodic structure; no repair or reordering."""
    import numpy as np
    from ase.io import read

    frames = read(str(path), index=":")
    if len(frames) != 1:
        raise ValueError("Exactly one input structure is required")
    atoms = frames[0]
    if any(atoms.pbc):
        raise ValueError("ORCA is molecular; a periodic structure must be cut to a cluster first")
    n = len(atoms)
    if not n or n > max_atoms:
        raise ValueError(f"Structure has {n} atoms; the atom budget is {max_atoms}")
    if not np.isfinite(atoms.positions).all():
        raise ValueError("Nonfinite geometry")
    return atoms.get_chemical_symbols(), atoms.positions.copy()


def orca_input(config):
    from xtal.orca import input as orca

    method = config["method"]
    optimize = config["task"] == "optimize"
    # An optimization writes job.engrad on its own; a gradient single point has to ask for it.
    extra = "EnGrad" if config["task"] == "gradient" else ""
    return orca.OrcaInput(
        functional=method["functional"], basis=method["basis"] or "", dispersion=method["dispersion"] or "",
        ri=method["ri"] or "", scf_threshold=method["scf_convergence"] or "",
        run="opt" if optimize else "sp", opt_level=config["optimization"]["level"] if optimize else "Opt",
        freq=config["frequencies"], max_iter=config["optimization"]["max_cycles"] if optimize else None,
        nprocs=config["budget"]["cpus"], maxcore_mb=config["budget"]["maxcore_mb"],
        charge=config["charge"], multiplicity=config["spin_multiplicity"], extra_keywords=extra)


def check(config, structure_path):
    """Everything prepare refuses, without writing: returns (inp, cluster, cautions)."""
    from xtal.orca import input as orca

    config = validate_config(config)
    symbols, positions = read_molecule(structure_path, config["budget"]["max_atoms"])
    cluster = orca.Cluster(elements=list(symbols), cart=positions)
    inp = orca_input(config)
    refusals, cautions = orca.problems(inp, cluster)
    if refusals:
        raise ValueError("; ".join(refusals))
    return inp, cluster, cautions


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
    """This package and the Crystal-Builder writer it renders inputs with."""
    hashes = {p.name: sha256(p) for p in sorted(Path(__file__).parent.glob("*.py"))}
    for path in sorted(CRYSTAL_BUILDER_ORCA.glob("*.py")) + [CRYSTAL_BUILDER_ORCA / "data/tables.json"]:
        hashes[f"xtal/orca/{path.relative_to(CRYSTAL_BUILDER_ORCA).as_posix()}"] = sha256(path)
    return hashes


def prepare(config_path, output):
    from xtal.orca import input as orca

    config_path, output = Path(config_path).resolve(), Path(output).resolve()
    config = validate_config(read_json(config_path))
    source = (config_path.parent / config["structure"]).resolve()
    inp, cluster, cautions = check(config, source)
    output.mkdir(parents=True, exist_ok=False)
    (output / "source_snapshot").mkdir()
    for module in Path(__file__).parent.glob("*.py"):
        (output / "source_snapshot" / module.name).write_bytes(module.read_bytes())
    (output / f"source{source.suffix.lower()}").write_bytes(source.read_bytes())
    write_json(output / "config.json", config)
    xyz = f"{BASENAME}_from_crystal_builder.xyz"
    (output / xyz).write_text(orca.xyz_text(cluster, comment=f"from {source.name}"), encoding="utf-8", newline="\n")
    (output / f"{BASENAME}.inp").write_text(orca.render(inp, xyz, title=f"rsi.orca {config['task']}"),
                                             encoding="utf-8", newline="\n")
    names = (f"source{source.suffix.lower()}", "config.json", xyz, f"{BASENAME}.inp")
    manifest = {"schema_version": 1, "status": "prepared", "engine": "orca", "is_dft": True,
                "task": config["task"], "source_path": str(source), "atoms": cluster.n_atoms,
                "elements": sorted(set(cluster.elements)),
                "electrons": orca.electrons(cluster.elements, config["charge"]),
                "cautions": cautions, "repository": git_revision(ROOT),
                "crystal_builder": git_revision(ROOT / "Crystal-Builder"), "code_hashes": code_hashes(),
                "input_hashes": {name: sha256(output / name) for name in names}, "elapsed_seconds": 0.0,
                "claim": "Engineering calculation output; method adequacy and scientific validation are separate"}
    write_json(output / "manifest.json", manifest)
    return manifest


def verify_frozen(output, manifest):
    if manifest["code_hashes"] != code_hashes():
        raise ValueError("ORCA adapter or Crystal-Builder writer changed after preparation")
    for name, digest in manifest["input_hashes"].items():
        if sha256(output / name) != digest:
            raise ValueError(f"Frozen input changed: {name}")
    return validate_config(read_json(output / "config.json"))


def job_environment(executable):
    """ORCA's shared libraries sit beside its binaries, and its MPI start needs them on PATH."""
    folder = str(executable.parent)
    return {"PATH": folder + os.pathsep + os.environ.get("PATH", ""),
            "LD_LIBRARY_PATH": folder + os.pathsep + os.environ.get("LD_LIBRARY_PATH", "")}


def _run_local(argv, cwd, env, log, timeout):
    with (cwd / log).open("w", encoding="utf-8") as handle:
        process = subprocess.Popen(argv, cwd=cwd, env=dict(os.environ, **env), stdout=handle,
                                   stderr=subprocess.STDOUT, start_new_session=os.name != "nt")
        try:
            return process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            from rsi.dft.core import stop_process_tree
            stop_process_tree(process)
            raise TimeoutError(f"ORCA timed out after {timeout:.1f} s") from None
        except BaseException:
            from rsi.dft.core import stop_process_tree
            stop_process_tree(process)
            raise


def _run_scheduled(scheduler, backend, capacity_timeout, argv, cwd, env, log, config, record):
    budget = config["budget"]
    spec = {"name": f"orca-{cwd.parent.name}"[:120], "argv": argv, "cwd": str(cwd), "env": env, "log": log,
            "resources": {"cpus": budget["cpus"], "wall_seconds": budget["wall_seconds"],
                          "memory_mb": budget["cpus"] * budget["maxcore_mb"]}}
    job = scheduler.run(spec, backend, capacity_timeout)
    record["hpc_job"] = {key: job.get(key) for key in ("id", "backend", "native_id", "status", "charged")}
    if job["status"] in {"timeout", "queue_timeout"}:
        raise TimeoutError(f"ORCA job {job['status']} after {budget['wall_seconds']} s")
    if job["status"] not in {"completed", "failed"}:
        raise RuntimeError(f"ORCA job ended as {job['status']} on {job['backend']}")
    return (job.get("exit_code") or -1) if job["status"] == "failed" else 0


def verify(run, config, manifest):
    """Recompute the result from ORCA's own files; the exit code alone is never trusted."""
    out = run / f"{BASENAME}.out"
    native = parse_output(out.read_text(encoding="utf-8", errors="replace"))
    if not native["terminated_normally"]:
        raise RuntimeError("ORCA did not terminate normally; see job.out")
    if not native["energies_eh"]:
        raise ValueError("ORCA output reports no final energy")
    if native["mpi_processes"] != config["budget"]["cpus"]:
        raise ValueError(f"ORCA ran {native['mpi_processes']} processes; {config['budget']['cpus']} were reserved")
    energy = native["energies_eh"][-1]
    checks = {"native_output": out.name, "native_output_sha256": sha256(out),
              "scf_all_converged": native["scf_all_converged"], "scf_not_converged": native["scf_not_converged"],
              "scf_iterations": native["scf_iterations"]}
    gradient = None
    engrad = run / f"{BASENAME}.engrad"
    if config["task"] in {"gradient", "optimize"}:
        if not engrad.is_file():
            raise ValueError("ORCA wrote no .engrad for a gradient or optimization")
        parsed = parse_engrad(engrad.read_text(encoding="utf-8", errors="replace"))
        if parsed["atoms"] != manifest["atoms"]:
            raise ValueError("ORCA .engrad atom count differs from the prepared structure")
        if abs(parsed["energy_eh"] - energy) > ENERGY_TOLERANCE_EH:
            raise ValueError("ORCA .engrad energy and final single point energy disagree")
        norms = [math.sqrt(sum(c * c for c in row)) for row in parsed["gradient_eh_per_bohr"]]
        gradient = parsed["gradient_eh_per_bohr"]
        checks.update(engrad_sha256=sha256(engrad), max_gradient_eh_per_bohr=max(norms),
                      rms_gradient_eh_per_bohr=math.sqrt(sum(n * n for n in norms) / len(norms)))
    geometry = None
    if config["task"] == "optimize":
        final = run / f"{BASENAME}.xyz"
        if not final.is_file():
            raise ValueError("ORCA wrote no final geometry for an optimization")
        geometry = parse_xyz(final.read_text(encoding="utf-8", errors="replace"))
        if len(geometry["symbols"]) != manifest["atoms"]:
            raise ValueError("ORCA final geometry atom count differs from the prepared structure")
        checks.update(final_geometry_sha256=sha256(final),
                      optimization_converged=native["optimization_converged"] is True,
                      optimization_cycles=native["optimization_cycles"],
                      optimization_cycle_limit=config["optimization"]["max_cycles"])
    if config["frequencies"]:
        if native["frequencies"] is None:
            raise ValueError("Frequencies were requested but ORCA printed none")
        modes = 3 * manifest["atoms"]
        if len(native["frequencies"]["cm-1"]) != modes:
            raise ValueError(f"ORCA printed {len(native['frequencies']['cm-1'])} modes; {modes} expected")
    return native, energy, gradient, geometry, checks


def summarize(config, native, energy, gradient, geometry, checks, manifest):
    converged = checks["scf_all_converged"] and (config["task"] != "optimize" or checks["optimization_converged"])
    forces = None
    if gradient is not None:
        scale = HARTREE_EV / BOHR_ANGSTROM
        forces = [[-g * scale for g in row] for row in gradient]
    frequencies = native["frequencies"]
    return {"engine": "orca", "orca_version": native["orca_version"], "is_dft": True, "task": config["task"],
            "method": config["method"], "charge": config["charge"], "spin_multiplicity": config["spin_multiplicity"],
            "atoms": manifest["atoms"], "electrons": manifest["electrons"],
            "energy_eh": energy, "energy_ev": energy * HARTREE_EV, "converged": converged,
            "max_force_ev_per_angstrom": (checks["max_gradient_eh_per_bohr"] * HARTREE_EV / BOHR_ANGSTROM
                                          if gradient is not None else None),
            "forces_ev_per_angstrom": forces,
            "final_positions_angstrom": geometry["positions_angstrom"] if geometry else None,
            "homo_lumo_gap_ev": native["homo_lumo_gap_ev"],
            "imaginary_frequencies": frequencies["imaginary_count"] if frequencies else None,
            "lowest_frequency_cm-1": frequencies["lowest_nonzero_cm-1"] if frequencies else None,
            "frequencies_cm-1": frequencies["cm-1"] if frequencies else None,
            "zero_point_energy_eh": native["zero_point_energy_eh"], "total_enthalpy_eh": native["total_enthalpy_eh"],
            "final_gibbs_free_energy_eh": native["final_gibbs_free_energy_eh"],
            "cautions": manifest["cautions"], "checks": checks,
            "units": {"energy": "Eh (energy_ev is converted with CODATA 2018)", "force": "eV/Angstrom",
                      "length": "Angstrom", "frequency": "cm-1"}}


def execute(output, scheduler=None, backend=None, capacity_timeout=0.0):
    """Run the prepared input once. With an ``rsi.hpc.Scheduler`` ORCA is a scheduled job charged to
    that session; ``budget.cpus`` MPI processes and ``cpus x maxcore_mb`` memory are reserved."""
    output = Path(output).resolve()
    manifest = read_json(output / "manifest.json")
    if manifest["status"] != "prepared":
        raise ValueError(f"Run status {manifest['status']!r} cannot start; prepare a new run instead")
    config = verify_frozen(output, manifest)
    executable = binary()
    if executable is None:
        raise ImportError(f"ORCA executable not found; set {ENV_VAR} to the absolute path of orca")
    manifest.update(orca_executable=str(executable), orca_executable_sha256=sha256(executable),
                    status="running")
    run = output / "run"
    run.mkdir(exist_ok=False)
    for name in (f"{BASENAME}.inp", f"{BASENAME}_from_crystal_builder.xyz"):
        shutil.copyfile(output / name, run / name)
    write_json(output / "manifest.json", manifest)
    argv, env, log = [str(executable), f"{BASENAME}.inp"], job_environment(executable), f"{BASENAME}.out"
    record = {}
    before = time.monotonic()
    try:
        if scheduler is not None:
            code = _run_scheduled(scheduler, backend, capacity_timeout, argv, run, env, log, config, record)
        else:
            code = _run_local(argv, run, env, log, config["budget"]["wall_seconds"])
        manifest["exit_code"] = code
        native, energy, gradient, geometry, checks = verify(run, config, manifest)
        if code != 0:
            raise RuntimeError(f"ORCA exited with status {code}; see run/job.out")
        summary = summarize(config, native, energy, gradient, geometry, checks, manifest)
        write_json(output / "summary.json", summary)
        manifest.update(status="completed", converged=summary["converged"], orca_version=native["orca_version"],
                        summary_sha256=sha256(output / "summary.json"))
    except BaseException as exc:
        manifest.update(status="timeout" if isinstance(exc, TimeoutError) else
                        "interrupted" if isinstance(exc, (KeyboardInterrupt, SystemExit)) else "failed",
                        error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        manifest["elapsed_seconds"] = time.monotonic() - before
        manifest.update(record)
        write_json(output / "manifest.json", manifest)
    return summary

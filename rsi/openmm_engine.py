"""Controller-side OpenMM minimization broker; candidates never call this.

Kept separate from the frozen Q2 files so their recorded hashes stay valid.
Events use verb "minimize", so from_uff_event rejects them and cannot
mislabel native OpenMM kJ values as Crystal-Builder kcal values.

File interface (controller only): a serialized System XML plus a positions
JSON {"positions_nm": [[x, y, z], ...], "box_vectors_nm": [[...]] * 3 (optional)}.
`run_isolated` executes one task in a child process under a hard wall-clock
limit so a native hang or crash cannot take down the controller.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import openmm
from openmm import unit

from rsi.tool_runtime import digest, validate_input

ENGINE = "openmm"
KJ_MOL = unit.kilojoule_per_mole
KJ_MOL_NM = unit.kilojoule_per_mole / unit.nanometer
PLATFORMS = ("Reference", "CPU", "OpenCL", "CUDA", "HIP")


class _Budget(openmm.MinimizationReporter):
    """Counts L-BFGS iterations and stops at the wall-clock limit."""

    def __init__(self, max_wall_seconds):
        super().__init__()
        self.max_wall_seconds = max_wall_seconds
        self.deadline = None
        self.iterations = 0
        self.timed_out = False

    def start(self):
        # The clock starts at minimization, not at Context creation.
        if self.max_wall_seconds is not None:
            self.deadline = time.perf_counter() + self.max_wall_seconds

    def report(self, iteration, x, grad, args):
        self.iterations += 1
        if self.deadline is not None and time.perf_counter() > self.deadline:
            self.timed_out = True
            return True
        return False


def _positive(value, name, kind=float):
    if isinstance(value, bool) or not isinstance(value, (int, float) if kind is float else int):
        raise ValueError(f"{name} must be a positive {kind.__name__}")
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a positive {kind.__name__}")
    return value


def _nm(values, name):
    if unit.is_quantity(values):
        values = values.value_in_unit(unit.nanometer)
    array = np.array([[float(v) for v in row] for row in values], dtype=float)
    if array.ndim != 2 or array.shape[1] != 3 or not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite N x 3 coordinates in nm")
    return array


def _check_system(system, positions, box_vectors):
    n = system.getNumParticles()
    if n == 0 or positions.shape[0] != n:
        raise ValueError(f"Expected {n} positions, got {positions.shape[0]}")
    if system.getNumConstraints():
        # Constraint forces never vanish, so the max-force rule is undefined.
        raise ValueError("Constrained systems are not supported by the max-force convergence rule")
    if box_vectors is not None and not system.usesPeriodicBoundaryConditions():
        raise ValueError("Box vectors were given for a non-periodic system")
    free = np.array([system.getParticleMass(i).value_in_unit(unit.dalton) > 0
                     and not system.isVirtualSite(i) for i in range(n)])
    if not free.any():
        raise ValueError("No movable particles")
    return free


def system_sha256(system):
    return hashlib.sha256(openmm.XmlSerializer.serialize(system).encode()).hexdigest()


def minimize(system, positions, tolerance=10.0, max_iterations=1000, platform="CPU",
             box_vectors=None, max_wall_seconds=None, threads=None, return_positions=False):
    """Return a Crystal-Builder-shaped minimization event in native OpenMM units.

    Invalid requests raise ValueError before any compute. Numerical failures
    during minimization return an event with result.ok False.
    """
    _positive(tolerance, "tolerance")
    # OpenMM treats max_iterations=0 as unlimited, which would be an unbounded run.
    _positive(max_iterations, "max_iterations", int)
    if max_wall_seconds is not None:
        _positive(max_wall_seconds, "max_wall_seconds")
    if platform not in PLATFORMS:
        raise ValueError(f"Unsupported platform {platform!r}")
    properties = {}
    if threads is not None:
        if platform != "CPU":
            raise ValueError("threads applies only to the CPU platform")
        properties["Threads"] = str(_positive(threads, "threads", int))
    xyz = _nm(positions, "positions")
    box = None if box_vectors is None else _nm(box_vectors, "box_vectors")
    if box is not None and box.shape != (3, 3):
        raise ValueError("box_vectors must be 3 x 3 in nm")
    free = _check_system(system, xyz, box)

    action = {"verb": "minimize", "engine": ENGINE,
              "tolerance_kj_mol_nm": tolerance, "max_iterations": max_iterations,
              "max_wall_seconds": max_wall_seconds,
              "convergence_rule": "max_atom_force <= tolerance",
              "openmm_version": openmm.__version__, "platform": platform,
              "platform_properties": properties, "system_sha256": system_sha256(system),
              "num_particles": int(xyz.shape[0]), "num_free_particles": int(free.sum()),
              "periodic": system.usesPeriodicBoundaryConditions()}
    started = time.perf_counter()
    budget = _Budget(max_wall_seconds)
    integrator = openmm.VerletIntegrator(0.001)
    try:
        context = openmm.Context(system, integrator, openmm.Platform.getPlatformByName(platform),
                                 properties)
    except Exception as exc:
        del integrator
        raise ValueError(f"Platform {platform} unavailable: {exc}") from exc
    try:
        if box is not None:
            context.setPeriodicBoxVectors(*(openmm.Vec3(*row) for row in box))
        context.setPositions(xyz)
        initial = context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(KJ_MOL)
        budget.start()
        openmm.LocalEnergyMinimizer.minimize(context, tolerance, max_iterations, budget)
        state = context.getState(getEnergy=True, getForces=True, getPositions=return_positions)
    except Exception as exc:
        action["elapsed_seconds"] = time.perf_counter() - started
        return {"action": action, "result": {"ok": False, "data": None,
                "error": {"type": type(exc).__name__, "message": str(exc)}}}
    finally:
        del context, integrator
    energy = state.getPotentialEnergy().value_in_unit(KJ_MOL)
    forces = state.getForces(asNumpy=True).value_in_unit(KJ_MOL_NM)
    action["elapsed_seconds"] = time.perf_counter() - started
    if not (math.isfinite(initial) and math.isfinite(energy) and np.isfinite(forces).all()):
        return {"action": action, "result": {"ok": False, "data": None,
                "error": {"type": "NonFiniteResult", "message": "Energy or force is not finite"}}}
    # Massless and virtual-site particles never move, so they are excluded from the rule.
    max_force = float(np.linalg.norm(forces[free], axis=1).max())
    data = {"initial_energy": initial, "energy": energy, "max_force": max_force,
            # Explicit broker criterion on the per-atom max force; OpenMM's own
            # tolerance applies to the RMS force, so it is not reused as evidence.
            "converged": max_force <= tolerance,
            "iterations": budget.iterations,
            "stopped_by_wall_time": budget.timed_out}
    if return_positions:
        data["positions_nm"] = state.getPositions(asNumpy=True).value_in_unit(unit.nanometer).tolist()
    return {"action": action, "result": {"ok": True, "data": data}}


def from_openmm_event(event):
    """Called by the science broker, never by an untrusted candidate."""
    action, result = event["action"], event["result"]
    if action.get("verb") != "minimize" or action.get("engine") != ENGINE or result.get("ok") is not True:
        raise ValueError("A successful OpenMM minimization event is required")
    data = result["data"]
    payload = {key: data[key] for key in ("initial_energy", "energy", "max_force", "converged")}
    payload.update(energy_unit="kJ/mol", force_unit="kJ/mol/nm")
    validate_input(payload)
    return {"payload": payload, "event_sha256": digest(event),
            "unit_source": f"OpenMM {action['openmm_version']} native units"}


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_task(system_xml, positions_json):
    """Read a controller-owned serialized System and positions file."""
    system = openmm.XmlSerializer.deserialize(Path(system_xml).read_text(encoding="utf-8"))
    if not isinstance(system, openmm.System):
        raise ValueError("Serialized object is not an OpenMM System")
    spec = json.loads(Path(positions_json).read_text(encoding="utf-8"))
    if not isinstance(spec, dict) or not {"positions_nm"} <= set(spec) <= {"positions_nm", "box_vectors_nm"}:
        raise ValueError("Positions file needs positions_nm and optional box_vectors_nm")
    return system, spec["positions_nm"], spec.get("box_vectors_nm")


def minimize_files(system_xml, positions_json, **options):
    """Minimize from files and bind the event to the input file hashes."""
    system, positions, box = load_task(system_xml, positions_json)
    event = minimize(system, positions, box_vectors=box, **options)
    event["action"]["inputs"] = {"system_xml_sha256": _sha256(system_xml),
                                 "positions_json_sha256": _sha256(positions_json)}
    return event


def run_isolated(system_xml, positions_json, output, timeout_seconds, python=sys.executable,
                 scheduler=None, backend=None, capacity_timeout=0.0, **options):
    """Run one file task in a child process; the controller survives hangs and crashes.

    The child receives max_wall_seconds below the hard timeout when unset, so
    a slow minimization normally returns a partial, unconverged event. With an
    ``rsi.hpc.Scheduler`` the child is a scheduled job (CPU count from
    ``threads``, default 1) charged to that session's budget.
    """
    _positive(timeout_seconds, "timeout_seconds")
    options.setdefault("max_wall_seconds", 0.8 * timeout_seconds)
    output = Path(output).resolve()
    if output.exists():
        raise ValueError("Output already exists; isolated runs never overwrite evidence")
    arguments = ["-m", "rsi.openmm_engine", "--system", str(Path(system_xml).resolve()),
                 "--positions", str(Path(positions_json).resolve()), "--output", str(output),
                 "--options", json.dumps(options)]
    if scheduler is not None:
        return _scheduled(scheduler, backend, capacity_timeout, arguments, output, timeout_seconds, options)
    command = [python] + arguments
    root = Path(__file__).resolve().parents[1]
    try:
        completed = subprocess.run(command, cwd=root, capture_output=True, text=True,
                                   timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        return _failure("Timeout", f"Child exceeded {timeout_seconds} s and was killed", options)
    if completed.returncode != 0 or not output.exists():
        return _failure("ChildFailed", (completed.stderr or "")[-2000:] or
                        f"Exit code {completed.returncode}", options)
    return json.loads(output.read_text(encoding="utf-8"))


def _scheduled(scheduler, backend, capacity_timeout, arguments, output, timeout_seconds, options):
    output.parent.mkdir(parents=True, exist_ok=True)
    spec = {"name": f"openmm-{output.stem}"[:120], "argv": ["{python}"] + arguments,
            "cwd": str(output.parent), "log": output.name + ".log",
            "resources": {"cpus": options.get("threads") or 1, "wall_seconds": timeout_seconds}}
    job = scheduler.run(spec, backend, capacity_timeout)
    hpc = {key: job.get(key) for key in ("id", "backend", "native_id", "status", "charged")}
    if job["status"] != "completed" or not output.exists():
        kind = "Timeout" if job["status"] in {"timeout", "queue_timeout"} else "ChildFailed"
        log = output.parent / spec["log"]
        tail = log.read_text(encoding="utf-8", errors="replace")[-2000:] if log.exists() else ""
        event = _failure(kind, tail or f"Scheduled job ended as {job['status']}", options)
    else:
        event = json.loads(output.read_text(encoding="utf-8"))
    event["action"]["hpc_job"] = hpc
    return event


def _failure(kind, message, options):
    return {"action": {"verb": "minimize", "engine": ENGINE, "isolated_options": options},
            "result": {"ok": False, "data": None, "error": {"type": kind, "message": message}}}


def main(argv=None):
    parser = argparse.ArgumentParser(description="Minimize one serialized OpenMM System.")
    parser.add_argument("--system", required=True)
    parser.add_argument("--positions", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--options", default="{}", help="JSON keyword arguments for minimize")
    args = parser.parse_args(argv)
    options = json.loads(args.options)
    allowed = {"tolerance", "max_iterations", "platform", "max_wall_seconds", "threads",
               "return_positions"}
    if not isinstance(options, dict) or not set(options) <= allowed:
        parser.exit(2, "Unsupported minimize options\n")
    event = minimize_files(args.system, args.positions, **options)
    output = Path(args.output)
    temporary = output.with_name(output.name + ".partial")
    temporary.write_text(json.dumps(event, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(output)


if __name__ == "__main__":
    main()

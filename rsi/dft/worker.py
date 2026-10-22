"""Run one DFT attempt in a disposable process; the controller verifies its files."""

from __future__ import annotations

from pathlib import Path
import sys
import time

from .core import DFT_BACKENDS, NATIVE_LOG, read_json, write_json


def build_calculator(config, folder):
    backend, params = config["backend"], dict(config["calculator"])
    if backend == "emt":
        from ase.calculators.emt import EMT
        return EMT()
    if backend == "gpaw":
        from gpaw import GPAW
        return GPAW(txt=str(folder / NATIVE_LOG["gpaw"]), charge=config["charge"], **params)
    from ase.calculators.cp2k import CP2K
    multiplicity = config["spin_multiplicity"]
    # ASE's CP2K calculator takes the plane-wave cutoff in eV and writes <label>.out in cwd.
    return CP2K(command=params.pop("executable"), label=NATIVE_LOG["cp2k"].rsplit(".", 1)[0],
                cutoff=params.pop("cutoff_ev"), charge=config["charge"], multiplicity=multiplicity,
                uks=multiplicity != 1, **params)


def build_atoms(structure, positions):
    from ase import Atoms
    from ase.constraints import FixAtoms, FixCartesian

    atoms = Atoms(structure["symbols"], positions=positions, cell=structure["cell"], pbc=structure["pbc"])
    constraints = [FixCartesian(item["index"], mask=item["mask"]) for item in structure["fixed_cartesian"]]
    if structure["fixed_atoms"]:
        constraints.insert(0, FixAtoms(indices=structure["fixed_atoms"]))
    atoms.set_constraint(constraints)
    return atoms


def run(output, attempt_id):
    import ase.optimize

    folder = output / attempt_id
    config = read_json(output / "config.json")
    structure = read_json(output / "structure.json")
    attempt = read_json(folder / "attempt.json")
    start = attempt["start_checkpoint"]
    atoms = build_atoms(structure, start["positions"] if start else structure["positions"])
    if config.get("initial_magmoms") is not None:
        atoms.set_initial_magnetic_moments(config["initial_magmoms"])
    atoms.calc = build_calculator(config, folder)
    delay = config["calculator"].get("delay_seconds_per_step", 0) if config["backend"] == "emt" else 0
    state = {"force_evaluations": 0, "last_step": None}

    def checkpoint(step):
        energy = float(atoms.get_potential_energy())
        if state["last_step"] != step:
            state["force_evaluations"] += 1
            state["last_step"] = step
        write_json(folder / "checkpoint.json", {
            "global_step": step, "energy_ev": energy, "positions": atoms.positions.tolist(),
            "cell": atoms.cell.array.tolist(), "forces_ev_per_angstrom": atoms.get_forces().tolist(),
            "raw_forces_ev_per_angstrom": atoms.get_forces(apply_constraint=False).tolist(),
            "force_evaluations": state["force_evaluations"], "wall_time": time.time()})
        if delay:
            time.sleep(delay)  # Test surrogate only: makes timeout and restart paths observable.

    base = attempt["start_step"]
    converged = None
    if config["task"] == "single_point":
        checkpoint(0)
    else:
        opt_config = config["optimizer"]
        kwargs = {"maxstep": opt_config["maxstep_angstrom"]} if "maxstep_angstrom" in opt_config else {}
        # A restart rebuilds optimizer memory from the checkpoint geometry; no Hessian is carried over.
        optimizer = getattr(ase.optimize, opt_config["name"])(
            atoms, logfile=str(folder / "optimizer.log"), trajectory=str(folder / "optimizer.traj"), **kwargs)
        optimizer.attach(lambda: checkpoint(base + optimizer.nsteps), interval=1)
        converged = bool(optimizer.run(fmax=opt_config["fmax_ev_per_angstrom"], steps=attempt["steps_allowed"]))
        checkpoint(base + optimizer.nsteps)
    versions = {"ase": ase.__version__}
    if config["backend"] == "gpaw":
        import gpaw
        versions["gpaw"] = gpaw.__version__
    write_json(folder / "result.json", {
        "backend": config["backend"], "is_dft": config["backend"] in DFT_BACKENDS, "versions": versions,
        "optimizer_converged": converged, "checkpoint_step": state["last_step"],
        "force_evaluations": state["force_evaluations"]})


if __name__ == "__main__":
    run(Path(sys.argv[1]).resolve(), sys.argv[2])

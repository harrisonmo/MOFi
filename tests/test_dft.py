"""Engineering checks for the DFT adapter using ASE EMT as a non-DFT surrogate, plus native log parsing."""

import pytest

pytest.importorskip("ase")

from rsi.dft import core  # noqa: E402
from rsi.dft.parse import CP2K_HARTREE_EV, parse_cp2k_output, parse_gpaw_text  # noqa: E402

GPAW_LOG = """
     iter     time        total  log10-change:
                         energy   eigst   dens
iscf:   1 12:00:01   -10.123456                 +0.10
iscf:   2 12:00:02   -10.234567   -2.51  -1.20
Converged after 14 iterations.

Free energy:    -10.250000
Extrapolated:   -10.249000

iscf:   1 12:01:01   -10.300000                 +0.10
Converged after 9 iterations.

Free energy:    -10.310000
Extrapolated:   -10.309500
"""

# Excerpt of a real GPAW 26.7.0 txt log (H2O, FD h=0.20, PBE) from the WSL validation run.
GPAW_26_LOG = """
SCF iterations
|iter     |     time |      energy | eigst |  dens |
|iter:  11| 19:04:16 |  -14.467435c| -7.32 | -5.43c|
|iter:  12| 19:04:17 |  -14.467435c| -7.79c| -5.78c|

Converged in 12 steps

Energy contributions relative to reference atoms: (reference = -2065.832626)

  Free energy:     -14.467435
  Extrapolated:    -14.467435

Forces  # eV/Ang
|        x |        y |       z |
|----------|----------|---------|
|  2.96842 |  1.45135 | 0.00000 |  # O    0
| -2.84189 | -0.27932 | 0.00000 |  # H    1
| -0.12695 | -1.17422 | 0.00000 |  # H    2
"""

CP2K_LOG = """
  *** SCF run converged in    12 steps ***
 ENERGY| Total FORCE_EVAL ( QS ) energy [a.u.]:              -34.123456789012
  Leaving inner SCF loop after reaching    50 steps.
  *** SCF run NOT converged ***
 ENERGY| Total FORCE_EVAL ( QS ) energy [hartree]:           -34.100000000000
"""


def make_inputs(tmp_path, task="optimize", **overrides):
    from ase.build import bulk
    from ase.io import write

    atoms = bulk("Cu", "fcc", a=3.6, cubic=True).repeat((2, 2, 1))  # 16 Cu, even electron count
    atoms.rattle(stdev=0.08, seed=7)
    write(tmp_path / "cu.traj", atoms)
    config = {"schema_version": 1, "backend": "emt", "task": task, "structure": "cu.traj",
              "charge": 0, "spin_multiplicity": 1, "calculator": {},
              "constraints": {"fixed_atoms": [0, 1], "fixed_cartesian": [{"index": 2, "mask": [False, False, True]}]},
              "budget": {"timeout_seconds_per_attempt": 120, "max_wall_seconds": 600, "max_atoms": 64,
                         "max_restarts": 0, "threads": 1}}
    if task == "optimize":
        config["optimizer"] = {"name": "BFGS", "fmax_ev_per_angstrom": 0.02, "max_steps": 200}
    for key, value in overrides.items():
        config[key] = value
    core.write_json(tmp_path / "config.json", config)
    return atoms, config


@pytest.mark.parametrize("key,value", [
    ("backend", "vasp"), ("task", "md"), ("charge", 1), ("spin_multiplicity", 0),
    ("calculator", {"xc": "PBE"}), ("optimizer", {"name": "BFGS", "fmax_ev_per_angstrom": 0.05}),
    ("constraints", {"fixed_cartesian": [{"index": 0, "mask": [1, 0, 0]}]}), ("structure", "cu.pdb"),
    ("unexpected_option", True)])
def test_reject_ambiguous_settings(tmp_path, key, value):
    _, config = make_inputs(tmp_path)
    config[key] = value
    with pytest.raises(ValueError):
        core.validate_config(config)


def test_dft_backends_require_explicit_method(tmp_path):
    _, config = make_inputs(tmp_path)
    config["backend"] = "gpaw"
    config["calculator"] = {"mode": {"name": "pw"}, "xc": "PBE"}
    with pytest.raises(ValueError, match="ecut"):
        core.validate_config(config)
    config["calculator"] = {"mode": {"name": "pw", "ecut": 400}, "xc": "PBE", "kpts": [1, 1, 1]}
    core.validate_config(config)
    config["spin_multiplicity"] = 3
    with pytest.raises(ValueError, match="spinpol"):
        core.validate_config(config)
    config["backend"] = "cp2k"
    config["calculator"] = {"executable": "cp2k_shell", "xc": "PBE", "cutoff_ev": 5000,
                            "basis_set": "DZVP-MOLOPT-SR-GTH", "pseudo_potential": "auto"}
    core.validate_config(config)
    del config["calculator"]["basis_set"]
    with pytest.raises(ValueError):
        core.validate_config(config)


def test_constraint_indices_are_checked_against_structure(tmp_path):
    _, config = make_inputs(tmp_path, constraints={"fixed_atoms": [99]})
    with pytest.raises(ValueError, match="outside"):
        core.prepare(tmp_path / "config.json", tmp_path / "run")
    make_inputs(tmp_path, constraints={"fixed_atoms": [2], "fixed_cartesian": [{"index": 2, "mask": [True] * 3}]})
    with pytest.raises(ValueError, match="both"):
        core.prepare(tmp_path / "config.json", tmp_path / "run")


def test_single_point_matches_direct_calculator(tmp_path):
    import numpy as np
    from ase.calculators.emt import EMT

    atoms, _ = make_inputs(tmp_path, task="single_point", constraints={})
    core.prepare(tmp_path / "config.json", tmp_path / "run")
    summary = core.execute(tmp_path / "run")
    atoms.calc = EMT()
    assert summary["energy_ev"] == pytest.approx(atoms.get_potential_energy(), abs=1e-10)
    assert np.abs(np.array(summary["forces_ev_per_angstrom"]) - atoms.get_forces()).max() < 1e-10
    assert summary["converged"] is True and summary["is_dft"] is False
    assert summary["checks"]["scf_all_converged"] is None


def test_constrained_optimization_converges_and_keeps_fixed_coordinates(tmp_path):
    import numpy as np

    atoms, _ = make_inputs(tmp_path)
    core.prepare(tmp_path / "config.json", tmp_path / "run")
    summary = core.execute(tmp_path / "run")
    final = np.array(summary["positions_angstrom"])
    assert summary["converged"] is True and summary["optimizer_reported_converged"] is True
    assert np.abs(final[:2] - atoms.positions[:2]).max() == 0
    assert final[2, 2] == atoms.positions[2, 2] and np.abs(final[2, :2] - atoms.positions[2, :2]).max() > 0
    assert summary["checks"]["max_free_force_ev_per_angstrom"] <= 0.02
    # Fixed atoms still carry physical force; only free components enter the convergence rule.
    assert summary["checks"]["max_raw_force_ev_per_angstrom"] > 0.02
    manifest = core.read_json(tmp_path / "run/manifest.json")
    assert manifest["status"] == "completed" and manifest["steps_used"] == summary["steps"] > 0
    with pytest.raises(ValueError, match="cannot start"):
        core.execute(tmp_path / "run")


def test_step_budget_exhaustion_is_reported_unconverged(tmp_path):
    make_inputs(tmp_path, optimizer={"name": "FIRE", "fmax_ev_per_angstrom": 1e-4, "max_steps": 3})
    core.prepare(tmp_path / "config.json", tmp_path / "run")
    summary = core.execute(tmp_path / "run")
    assert summary["steps"] == 3 and summary["converged"] is False
    assert summary["checks"]["force_converged"] is False


def test_timeout_kills_worker_then_resume_continues_from_checkpoint(tmp_path):
    make_inputs(tmp_path, calculator={"delay_seconds_per_step": 0.6},
                budget={"timeout_seconds_per_attempt": 6, "max_wall_seconds": 600, "max_atoms": 64,
                        "max_restarts": 30, "threads": 1})
    core.prepare(tmp_path / "config.json", tmp_path / "run")
    with pytest.raises(TimeoutError):
        core.execute(tmp_path / "run")
    manifest = core.read_json(tmp_path / "run/manifest.json")
    first = manifest["attempts"][0]
    assert manifest["status"] == "timeout" and first["status"] == "timeout" and first["checkpoint_step"] > 0
    summary = None
    while summary is None:
        try:
            summary = core.execute(tmp_path / "run", resume=True)
        except TimeoutError:
            pass
    manifest = core.read_json(tmp_path / "run/manifest.json")
    starts = [a["start_step"] for a in manifest["attempts"]]
    assert starts[0] == 0 and starts[1] == first["checkpoint_step"] and starts == sorted(starts)
    assert manifest["attempts"][1]["resumed_from"] == "attempt-001"
    assert "error" not in manifest and manifest["attempts"][0]["error"].startswith("TimeoutError")
    assert summary["converged"] is True and summary["attempts"] == len(manifest["attempts"]) >= 2
    with pytest.raises(ValueError, match="cannot resume"):
        core.execute(tmp_path / "run", resume=True)


def test_restart_budget_and_tampering_are_rejected(tmp_path):
    make_inputs(tmp_path, calculator={"delay_seconds_per_step": 0.6},
                budget={"timeout_seconds_per_attempt": 6, "max_wall_seconds": 600, "max_atoms": 64,
                        "max_restarts": 0, "threads": 1})
    core.prepare(tmp_path / "config.json", tmp_path / "run")
    with pytest.raises(TimeoutError):
        core.execute(tmp_path / "run")
    with pytest.raises(ValueError, match="Restart budget"):
        core.execute(tmp_path / "run", resume=True)
    run = tmp_path / "run"
    config = core.read_json(run / "config.json")
    config["budget"]["max_restarts"] = 5
    core.write_json(run / "config.json", config)
    with pytest.raises(ValueError, match="Frozen input changed"):
        core.execute(run, resume=True)


def test_parsers_read_scf_state_and_units():
    gpaw = parse_gpaw_text(GPAW_LOG)
    assert gpaw["energies_ev"] == [-10.249, -10.3095] and gpaw["scf_iterations"] == [14, 9]
    assert gpaw["scf_all_converged"] is True
    assert parse_gpaw_text(GPAW_LOG.replace("Converged after 9 iterations.", ""))["scf_all_converged"] is False
    assert gpaw["last_forces_ev_per_angstrom"] is None
    new = parse_gpaw_text(GPAW_26_LOG)
    assert new["energies_ev"] == [-14.467435] and new["free_energies_ev"] == [-14.467435]
    assert new["scf_iterations"] == [12] and new["scf_all_converged"] is True
    assert new["last_forces_ev_per_angstrom"][1] == [-2.84189, -0.27932, 0.0]
    cp2k = parse_cp2k_output(CP2K_LOG)
    assert cp2k["energies_hartree"] == [-34.123456789012, -34.1]
    assert cp2k["energies_ev"][1] == pytest.approx(-34.1 * CP2K_HARTREE_EV)
    assert cp2k["scf_not_converged"] == 1 and cp2k["scf_all_converged"] is False
    assert parse_cp2k_output(CP2K_LOG.split("  Leaving")[0])["scf_all_converged"] is True
    # CP2K 2024.2 shell energy (eV) and its log energy (Hartree) from the WSL H2O validation run.
    assert -17.213866151831482 * CP2K_HARTREE_EV == pytest.approx(-468.41311951287, abs=1e-6)


def test_unconverged_native_scf_overrides_force_convergence(tmp_path):
    _, config = make_inputs(tmp_path, task="single_point", constraints={})
    config.update(backend="cp2k", calculator={"executable": "cp2k_shell", "xc": "PBE", "cutoff_ev": 5000,
                                              "basis_set": "DZVP-MOLOPT-SR-GTH", "pseudo_potential": "auto"})
    core.write_json(tmp_path / "config.json", config)
    core.prepare(tmp_path / "config.json", tmp_path / "run")
    structure = core.read_json(tmp_path / "run/structure.json")
    folder = tmp_path / "run/attempt-001"
    folder.mkdir()
    n = len(structure["symbols"])
    checkpoint = {"global_step": 0, "energy_ev": -34.1 * CP2K_HARTREE_EV,
                  "positions": [list(row) for row in structure["positions"]],
                  "cell": structure["cell"], "forces_ev_per_angstrom": [[0.0] * 3] * n,
                  "raw_forces_ev_per_angstrom": [[0.0] * 3] * n, "force_evaluations": 1}
    core.write_json(folder / "checkpoint.json", checkpoint)
    core.write_json(folder / "result.json", {"checkpoint_step": 0, "force_evaluations": 1, "optimizer_converged": None})
    (folder / "cp2k.out").write_text(CP2K_LOG, encoding="utf-8")
    result, saved, checks = core.verify_attempt(tmp_path / "run", folder, config, structure)
    assert checks["scf_all_converged"] is False
    assert core.summarize(config, result, saved, checks, {"attempts": []})["converged"] is False
    checkpoint["energy_ev"] += 0.01
    core.write_json(folder / "checkpoint.json", checkpoint)
    with pytest.raises(ValueError, match="disagree"):
        core.verify_attempt(tmp_path / "run", folder, config, structure)
    checkpoint["energy_ev"] -= 0.01
    checkpoint["positions"][0][0] += 0.01
    core.write_json(folder / "checkpoint.json", checkpoint)
    structure["fixed_atoms"] = [0]
    with pytest.raises(ValueError, match="constrained coordinate moved"):
        core.verify_attempt(tmp_path / "run", folder, config, structure)


def test_electron_count_must_match_multiplicity(tmp_path):
    make_inputs(tmp_path, task="single_point", backend="cp2k", spin_multiplicity=2,
                calculator={"executable": "cp2k_shell", "xc": "PBE", "cutoff_ev": 5000,
                            "basis_set": "DZVP-MOLOPT-SR-GTH", "pseudo_potential": "auto"})
    with pytest.raises(ValueError, match="incompatible"):
        core.prepare(tmp_path / "config.json", tmp_path / "run")

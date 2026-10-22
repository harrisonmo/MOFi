"""Scientific unit handling and failure containment for the optional GCMC adapter."""

import subprocess

import pytest

from rsi.gcmc import core


@pytest.fixture
def inputs(tmp_path):
    pytest.importorskip("ase")
    from ase import Atoms
    from ase.io import write

    write(tmp_path / "host.cif", Atoms("Si", scaled_positions=[[0, 0, 0]], cell=[20, 20, 20], pbc=True))
    ff = {"name": "noninteracting engineering control", "source": "analytic ideal-gas check",
          "framework_types": {"Si": {"element": "Si", "epsilon_k": 0., "sigma_angstrom": 1.}},
          "element_types": {"Si": "Si"}, "methane": {"epsilon_k": 0., "sigma_angstrom": 1.},
          "shifted": True, "tail_corrections": False}
    core.write_json(tmp_path / "ff.json", ff)
    config = {"schema_version": 1, "structure": "host.cif", "force_field": "ff.json",
              "temperature_k": 298., "pressures_bar": [1., 5.], "seeds": [101, 202],
              "loading_pressure_bar": 5., "unloading_pressure_bar": 1.,
              "initialization_cycles": 1000, "production_cycles": 3000, "blocks": 5,
              "cutoff_angstrom": 4., "fugacity_coefficients": [1., 1.],
              "budget": {"max_jobs": 4, "max_total_cycles": 16000,
                         "timeout_seconds_per_job": 30, "max_wall_seconds": 120,
                         "max_supercell_atoms": 1000}}
    core.write_json(tmp_path / "config.json", config)
    return tmp_path, config, ff


@pytest.mark.parametrize("key,value", [("temperature_k", float("nan")), ("pressures_bar", [1., 1.]),
    ("seeds", [True]), ("production_cycles", 10), ("loading_pressure_bar", 1.),
    ("fugacity_coefficients", [1.]), ("fugacity_coefficients", [1., -1.]),
    ("unexpected_engine_option", True)])
def test_reject_ambiguous_or_nonphysical_conditions(inputs, key, value):
    _, cfg, _ = inputs
    cfg[key] = value
    with pytest.raises(ValueError):
        core.validate_config(cfg)


def test_cycle_budget_and_seed_collision(inputs):
    _, cfg, _ = inputs
    cfg["budget"]["max_total_cycles"] = 15999
    with pytest.raises(ValueError, match="budget"):
        core.validate_config(cfg)
    cfg["seeds"] = [0, 1000003]
    with pytest.raises(ValueError, match="collide"):
        core.job_plan(cfg)


def test_preparation_freezes_source_and_respects_cutoff(inputs):
    path, _, ff = inputs
    manifest = core.prepare(path / "config.json", path / "run")
    assert manifest["status"] == "prepared"
    assert manifest["reserved_total_cycles"] == 16000
    assert len({j["engine_seed"] for j in manifest["jobs"]}) == 4
    for name, digest in manifest["input_hashes"].items():
        assert core.sha256(path / "run" / name) == digest
    geometry = core.structure_data(path / "host.cif", ff, 12., 100)
    assert geometry["unit_cells"] == [2, 2, 2]
    assert geometry["framework_mass_u"] == pytest.approx(28.085 * 8, rel=1e-3)
    with pytest.raises(FileExistsError):
        core.prepare(path / "config.json", path / "run")


def test_skewed_cell_uses_perpendicular_widths(inputs):
    from ase import Atoms
    from ase.io import write
    import numpy as np

    path, _, ff = inputs
    atoms = Atoms("Si", scaled_positions=[[.2, .3, .4]], cell=[20, 20, 20, 90, 90, 30], pbc=True)
    write(path / "skewed.cif", atoms)
    geometry = core.structure_data(path / "skewed.cif", ff, 6., 100)
    assert geometry["unit_cells"] == [2, 2, 1]
    widths = 1 / np.linalg.norm(np.linalg.inv(atoms.cell.array), axis=0)
    assert all(widths * geometry["unit_cells"] >= 12 - 1e-8)


def test_missing_typing_and_atom_budget_rejected(inputs):
    path, _, ff = inputs
    with pytest.raises(ValueError, match="atom budget"):
        core.structure_data(path / "host.cif", ff, 12., 1)
    ff["element_types"] = {}
    with pytest.raises(ValueError, match="assignment"):
        core.structure_data(path / "host.cif", ff, 4., 100)


def test_disordered_structure_rejected(inputs):
    path, _, ff = inputs
    cif = (path / "host.cif").read_text()
    # ASE CIF output includes a site occupancy of 1.0.
    cif = cif.replace("1.0000", "0.5000").replace("1.0\n", "0.5\n")
    (path / "disorder.cif").write_text(cif)
    with pytest.raises(ValueError, match="occupancy"):
        core.structure_data(path / "disorder.cif", ff, 4., 100)


def test_force_field_requires_provenance_and_explicit_typing(inputs):
    _, _, ff = inputs
    ff["site_types"] = ["Si"]
    with pytest.raises(ValueError, match="exactly one"):
        core.validate_force_field(ff)
    del ff["site_types"]
    ff["source"] = ""
    with pytest.raises(ValueError, match="provenance"):
        core.validate_force_field(ff)


def test_capacity_keeps_units_seed_errors_and_negative_values(inputs):
    _, cfg, _ = inputs
    rows = [{"pressure_bar": p, "replicate_seed": s, "absolute_uptake_mmol_g": v,
             "engine_reported_error_mmol_g": .3}
            for p, s, v in [(1., 101, 2.), (1., 202, 4.), (5., 101, 6.), (5., 202, 2.)]]
    summary = core.capacity_summary(rows, cfg)
    assert summary["working_capacity"]["mean"] == 1.
    assert summary["working_capacity"]["between_seed_standard_error"] == pytest.approx(3.)
    assert summary["negative_capacity_observed"]
    assert not summary["convergence_established"]
    with pytest.raises(ValueError, match="grid"):
        core.capacity_summary(rows[:-1], cfg)


def test_single_seed_does_not_invent_uncertainty(inputs):
    _, cfg, _ = inputs
    cfg["seeds"] = [101]
    rows = [{"pressure_bar": p, "replicate_seed": 101, "absolute_uptake_mmol_g": p,
             "engine_reported_error_mmol_g": .2} for p in cfg["pressures_bar"]]
    assert core.capacity_summary(rows, cfg)["working_capacity"]["between_seed_standard_error"] is None


def test_native_output_requires_completion_absolute_units_and_matching_uptake():
    text = ('Fugacity coefficient: 0.98 [-]\nSimulation finished!\nFinal state after 100 cycles\n'
            'LoadingData\nComponent 0 (methane)\n'
            'Abs. loading average 2.5 +/- 0.1 [mol/kg-framework]\n')
    assert core.native_statistics(text, 100, 2.5)["fugacity_coefficient"] == .98
    for changed in (text.replace("finished!", "not finished"), text.replace("100 cycles", "10 cycles"),
                    text.replace("Abs.", "Excess"), text.replace("2.5", "2500"),
                    text.replace("mol/kg-framework", "mg/g-framework")):
        with pytest.raises(ValueError):
            core.native_statistics(changed, 100, 2.5)


def test_changed_frozen_inputs_and_job_reservations_rejected(inputs):
    path, _, _ = inputs
    run = path / "run"
    manifest = core.prepare(path / "config.json", run)
    (run / "source.cif").write_text("changed")
    with pytest.raises(ValueError, match="Frozen input"):
        core.execute(run)
    manifest["input_hashes"]["source.cif"] = core.sha256(run / "source.cif")
    manifest["jobs"][0]["pressure_bar"] = 1234.
    core.write_json(run / "manifest.json", manifest)
    with pytest.raises(ValueError, match="Reserved jobs"):
        core.execute(run)


def test_failed_native_job_preserves_ledger_and_forbids_resume(inputs, monkeypatch):
    pytest.importorskip("raspalib")
    pytest.importorskip("psutil")
    path, _, _ = inputs
    run = path / "run"
    core.prepare(path / "config.json", run)

    class FailedProcess:
        def __init__(self, *args, **kwargs):
            pass
        def wait(self, **kwargs):
            return 7

    monkeypatch.setattr(core.subprocess, "Popen", FailedProcess)
    with pytest.raises(RuntimeError, match="job failed"):
        core.execute(run)
    ledger = core.read_json(run / "manifest.json")
    assert ledger["status"] == "failed"
    assert ledger["jobs"][0]["actual_cycles"] is None
    assert all(j["status"] == "pending" for j in ledger["jobs"][1:])
    assert not (run / "capacity.json").exists()
    with pytest.raises(ValueError, match="already been attempted"):
        core.execute(run)


def test_timeout_stops_worker_and_records_unknown_cycles(inputs, monkeypatch):
    pytest.importorskip("raspalib")
    pytest.importorskip("psutil")
    path, _, _ = inputs
    run = path / "run"
    core.prepare(path / "config.json", run)
    killed = []

    class TimedOutProcess:
        def __init__(self, *args, **kwargs):
            pass
        def wait(self, **kwargs):
            raise subprocess.TimeoutExpired("raspalib", 1)

    monkeypatch.setattr(core.subprocess, "Popen", TimedOutProcess)
    monkeypatch.setattr(core, "stop_process_tree", lambda p: killed.append(p))
    with pytest.raises(TimeoutError):
        core.execute(run)
    assert len(killed) == 1
    ledger = core.read_json(run / "manifest.json")
    assert ledger["jobs"][0]["status"] == "timeout"
    assert ledger["jobs"][0]["actual_cycles"] is None

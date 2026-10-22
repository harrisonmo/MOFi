"""Unified science hub: model-chosen arguments, bounds, HPC execution, budgets and session binding."""

import json
from pathlib import Path
import shutil

import pytest

pytest.importorskip("ase")

from rsi.gcmc import core as gcmc  # noqa: E402
from rsi.hpc.core import read_json, write_json  # noqa: E402
from rsi.science.hub import ScienceToolHub, check_schema  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
HOST_CIF = ("data_host\n_cell_length_a 12\n_cell_length_b 12\n_cell_length_c 12\n"
            "_cell_angle_alpha 90\n_cell_angle_beta 90\n_cell_angle_gamma 90\n"
            "_symmetry_space_group_name_H-M 'P 1'\nloop_\n_atom_site_label\n_atom_site_type_symbol\n"
            "_atom_site_fract_x\n_atom_site_fract_y\n_atom_site_fract_z\nC1 C 0.5 0.5 0.5\n")


def emt_relax(**changes):
    value = {"structure_id": "cu19", "backend": "emt", "task": "optimize", "xc": None, "cutoff_ev": None,
             "kpts": None, "charge": 0, "spin_multiplicity": 1, "fixed_elements": [], "optimizer": "BFGS",
             "fmax_ev_per_angstrom": 0.05, "max_steps": 200, "cpus": 1, "wall_seconds": 120}
    value.update(changes)
    return value


def make_config(folder, max_cpu_seconds=10000.0, max_calls=10, emt_delay=None, systems=None):
    shutil.copyfile(ROOT / "examples/dft/cu19-cluster.xyz", folder / "cu19.xyz")
    (folder / "host.cif").write_text(HOST_CIF, encoding="utf-8")
    write_json(folder / "hpc.json", {"schema_version": 1,
               "backends": {"local": {"type": "local", "max_cpus": 4}},
               "session_budget": {"max_jobs": 20, "max_cpu_seconds": max_cpu_seconds,
                                  "max_gpu_seconds": 1.0, "max_wall_seconds": 100000.0}})
    shutil.copyfile(ROOT / "examples/gcmc/mfi-force-field.json", folder / "ff.json")
    config = {"schema_version": 1, "hpc": {"config": "hpc.json", "backend": "local", "capacity_timeout_seconds": 30},
              "max_calls": max_calls,
              "structures": {"cu19": {"path": "cu19.xyz", "description": "Cu cluster"},
                             "host": {"path": "host.cif", "description": "One-atom host"}},
              "force_fields": {"mfi": "ff.json"},
              "openmm_systems": systems or {},
              "tools": json.loads((ROOT / "configs/science-hub.json").read_text(encoding="utf-8"))["tools"]}
    config["tools"]["mof_database_catalog"] = {"csv": str(ROOT / "datasets/catalogs/mofdb.csv"),
                                               "annotations": str(ROOT / "datasets/catalogs/mofdb.annotations.json")}
    config["tools"]["dft_calculation"]["backends"]["emt"]["defaults"] = (
        {"delay_seconds_per_step": emt_delay} if emt_delay else {})
    write_json(folder / "hub.json", config)
    return folder / "hub.json"


@pytest.fixture
def hub(tmp_path):
    return ScienceToolHub(make_config(tmp_path), tmp_path / "session")


def tool(hub, name):
    return next(t["function"] for t in hub.catalog()["tools"] if t["function"]["name"] == name)


def test_catalog_exposes_bounded_choices(hub):
    catalog = hub.catalog()
    names = {t["function"]["name"] for t in catalog["tools"]}
    assert {"dft_calculation", "science_budget_status"} <= names
    assert "openmm_minimize" in catalog["unavailable_tools"]
    dft = tool(hub, "dft_calculation")
    properties = dft["parameters"]["properties"]
    assert dft["strict"] is True
    assert properties["structure_id"]["enum"] == ["cu19", "host"]  # both fit the 64-atom limit
    assert properties["backend"]["enum"] == [b for b in ("gpaw", "cp2k", "emt") if b in hub.dft_backends]
    assert "Allowed: 1 to 8" in properties["cpus"]["description"]
    cu19 = next(s for s in catalog["structures"] if s["structure_id"] == "cu19")
    assert (cu19["atoms_in_cell"], cu19["elements"]) == (19, ["Cu"])
    assert catalog["budget"]["hpc_remaining"]["cpu_seconds"] == 10000.0


@pytest.mark.parametrize("arguments", [
    emt_relax(extra=1), emt_relax(cpus=99), emt_relax(cpus="1"), emt_relax(structure_id="missing"),
    emt_relax(xc="PBE"), emt_relax(optimizer=None), emt_relax(task="single_point"),
    emt_relax(wall_seconds=float("inf"))])
def test_invalid_arguments_launch_nothing(hub, arguments):
    result = hub.call("dft_calculation", arguments)
    assert result["ok"] is False and result["call_id"] is None
    assert hub.scheduler.status()["jobs"] == []
    assert hub.budget_status()["calls_used"] == 0


def test_dft_runs_as_scheduled_job_and_is_charged(hub):
    result = hub.call("dft_calculation", emt_relax())
    assert result["ok"], result["error"]
    assert result["data"]["converged"] is True and result["data"]["is_dft"] is False
    assert "positions_angstrom" not in result["data"]
    assert len(result["cost"]["hpc_jobs"]) == 1
    assert 0 < result["cost"]["charged"]["cpu_seconds"] < 120
    run = Path(result["evidence"]["run_directory"])
    manifest = read_json(run / "manifest.json")
    assert manifest["attempts"][0]["hpc_job"]["status"] == "completed"
    assert read_json(run.parent / "arguments.json")["arguments"] == emt_relax()
    status = hub.call("science_budget_status", {})
    assert status["ok"] and status["data"]["calls_used"] == 1
    assert status["data"]["calls"][0]["tool"] == "dft_calculation"


def test_scheduler_timeout_is_reported(tmp_path):
    hub = ScienceToolHub(make_config(tmp_path, emt_delay=0.5), tmp_path / "session")
    result = hub.call("dft_calculation", emt_relax(fmax_ev_per_angstrom=0.001, max_steps=300, wall_seconds=2))
    assert result["ok"] is False and result["error"]["type"] == "TimeoutError"
    job = hub.scheduler.status()["jobs"][0]
    assert job["status"] == "timeout"
    assert read_json(Path(job["cwd"]).parent / "manifest.json")["status"] == "timeout"


def test_compute_and_call_budgets_are_enforced(tmp_path):
    hub = ScienceToolHub(make_config(tmp_path, max_cpu_seconds=150.0, max_calls=2), tmp_path / "session")
    assert hub.call("dft_calculation", emt_relax(wall_seconds=100))["ok"]
    # The full request is reserved before launch; the first job's measured time leaves less than 149.5 s.
    refused = hub.call("dft_calculation", emt_relax(wall_seconds=149.5))
    assert refused["ok"] is False and "budget exhausted" in refused["error"]["message"]
    assert hub.call("dft_calculation", emt_relax(wall_seconds=10))["ok"] is False  # max_calls reached
    assert hub.budget_status()["calls_used"] == 2


def test_session_binding_refuses_changed_registration(tmp_path):
    config = make_config(tmp_path)
    ScienceToolHub(config, tmp_path / "session")
    value = read_json(config)
    value["max_calls"] = 99
    write_json(config, value)
    with pytest.raises(ValueError, match="registration changed"):
        ScienceToolHub(config, tmp_path / "session")


def test_changed_structure_blocks_calls(hub, tmp_path):
    (tmp_path / "cu19.xyz").write_text((tmp_path / "cu19.xyz").read_text() + "\n", encoding="utf-8")
    result = hub.call("dft_calculation", emt_relax())
    assert result["ok"] is False and "changed" in result["error"]["message"]


def test_gcmc_choices_are_validated_before_launch(hub):
    good = {"structure_id": "host", "force_field_id": "mfi", "temperature_k": 298.0, "pressures_bar": [5.8, 65.0],
            "loading_pressure_bar": 65.0, "unloading_pressure_bar": 5.8, "num_seeds": 2,
            "initialization_cycles": 100, "production_cycles": 500, "timeout_seconds_per_job": 60}
    assert callable(hub._plan_methane_gcmc_capacity(good))
    for change in ({"loading_pressure_bar": 30.0}, {"production_cycles": 20}, {"structure_id": "cu19"}):
        with pytest.raises(ValueError):
            hub._plan_methane_gcmc_capacity({**good, **change})
    if "methane_gcmc_capacity" not in hub.tools:
        assert "methane_gcmc_capacity" in hub.catalog()["unavailable_tools"]


class FakeScheduler:
    def __init__(self, record):
        self.record, self.specs = record, []

    def run(self, spec, backend=None, capacity_timeout=0.0):
        self.specs.append(spec)
        return {"id": "job-0001", "backend": "local", **self.record}


def test_gcmc_scheduled_job_maps_final_states(tmp_path):
    job = {"id": "p000-s000"}
    fake = FakeScheduler({"status": "failed", "exit_code": 0})
    assert gcmc._scheduled_job(fake, None, 0, tmp_path / "run", tmp_path, job, 60, 0.0) == -1
    assert fake.specs[0]["argv"][:3] == ["{python}", "-m", "rsi.gcmc.worker"]
    assert fake.specs[0]["resources"] == {"cpus": 1, "wall_seconds": 60}
    with pytest.raises(TimeoutError):
        gcmc._scheduled_job(FakeScheduler({"status": "queue_timeout"}), None, 0, tmp_path / "run", tmp_path,
                            job, 60, 0.0)
    assert job["status"] == "timeout"


def test_zeopp_runs_through_hub(tmp_path, monkeypatch):
    pytest.importorskip("xtal")
    from rsi.zeopp import core as zeo
    from test_zeopp import STANDIN, write_program

    (tmp_path / "bin").mkdir()
    monkeypatch.setenv(zeo.ENV_VAR, str(write_program(tmp_path / "bin", STANDIN)))
    hub = ScienceToolHub(make_config(tmp_path), tmp_path / "session")
    zeopp = tool(hub, "zeopp_pore_geometry")
    assert zeopp["parameters"]["properties"]["structure_id"]["enum"] == ["host"]
    arguments = {"structure_id": "host", "probe_radius_angstrom": 1.86, "psd_probe_radius_angstrom": 1.2,
                 "sampling": "fast", "high_accuracy": True, "block_pockets": True,
                 "grid_crosscheck_spacing_angstrom": None, "timeout_seconds": 60}
    result = hub.call("zeopp_pore_geometry", arguments)
    assert result["ok"], result["error"]
    assert result["data"]["diameters_angstrom"]
    assert len(result["cost"]["hpc_jobs"]) == 1
    manifest = read_json(Path(result["evidence"]["run_directory"]) / "manifest.json")
    assert manifest["hpc_job"]["backend"] == "local"
    assert hub.call("zeopp_pore_geometry", {**arguments, "probe_radius_angstrom": 4.0})["ok"] is False


def test_zeopp_unavailable_when_xtal_dependency_missing(tmp_path, monkeypatch):
    import importlib.util

    real = importlib.util.find_spec
    monkeypatch.setattr(importlib.util, "find_spec", lambda name, *a: None if name == "gemmi" else real(name, *a))
    catalog = ScienceToolHub(make_config(tmp_path), tmp_path / "session").catalog()
    assert "gemmi" in catalog["unavailable_tools"]["zeopp_pore_geometry"]


def orca_energy(**changes):
    value = {"structure_id": "water", "task": "gradient", "functional": "R2SCAN-3C", "basis": None,
             "dispersion": None, "ri": None, "scf_convergence": None, "opt_level": None, "max_opt_cycles": None,
             "frequencies": False, "charge": 0, "spin_multiplicity": 1, "cpus": 1, "wall_seconds": 60}
    value.update(changes)
    return value


def orca_hub(tmp_path):
    config = make_config(tmp_path)
    shutil.copyfile(ROOT / "examples/orca/water.xyz", tmp_path / "water.xyz")
    value = read_json(config)
    value["structures"]["water"] = {"path": "water.xyz", "description": "Water"}
    write_json(config, value)
    return ScienceToolHub(config, tmp_path / "session")


def test_orca_is_listed_unavailable_without_its_binary(tmp_path, monkeypatch):
    pytest.importorskip("xtal")
    monkeypatch.delenv("RSI_ORCA", raising=False)
    catalog = orca_hub(tmp_path).catalog()
    assert "RSI_ORCA" in catalog["unavailable_tools"]["orca_calculation"]


def test_orca_runs_through_hub(tmp_path, monkeypatch):
    pytest.importorskip("xtal")
    from test_orca import write_orca

    monkeypatch.setenv("RSI_ORCA", str(write_orca(tmp_path / "orca-bin")))
    monkeypatch.delenv("STANDIN_MODE", raising=False)
    hub = orca_hub(tmp_path)
    properties = tool(hub, "orca_calculation")["parameters"]["properties"]
    # Molecules only: the periodic CIF host is left out, the vacuum-box Cu cluster is not periodic.
    assert properties["structure_id"]["enum"] == ["cu19", "water"]
    assert properties["functional"]["enum"][0] == "R2SCAN-3C" and properties["basis"]["enum"][-1] is None
    assert "water" not in tool(hub, "dft_calculation")["parameters"]["properties"]["structure_id"]["enum"]
    water = next(s for s in hub.catalog()["structures"] if s["structure_id"] == "water")
    assert water["periodic"] is False
    for bad in (orca_energy(spin_multiplicity=2), orca_energy(basis="def2-SVP"), orca_energy(opt_level="Opt"),
                orca_energy(task="optimize"), orca_energy(cpus=99), orca_energy(structure_id="host"),
                orca_energy(functional="HF")):
        refused = hub.call("orca_calculation", bad)
        assert refused["ok"] is False and refused["call_id"] is None, bad
    assert hub.scheduler.status()["jobs"] == [] and hub.budget_status()["calls_used"] == 0
    result = hub.call("orca_calculation", orca_energy())
    assert result["ok"], result["error"]
    assert result["data"]["converged"] is True and result["data"]["max_force_ev_per_angstrom"] > 0
    assert "forces_ev_per_angstrom" not in result["data"]
    assert len(result["cost"]["hpc_jobs"]) == 1
    run = Path(result["evidence"]["run_directory"])
    assert read_json(run / "summary.json")["forces_ev_per_angstrom"]
    assert read_json(run / "manifest.json")["hpc_job"]["status"] == "completed"
    job = hub.scheduler.job(hub.scheduler.status()["jobs"][0]["id"])
    assert job["resources"]["memory_mb"] == 2000  # 1 process x the controller's maxcore


def test_openmm_runs_through_hub(tmp_path):
    openmm = pytest.importorskip("openmm")
    system = openmm.System()
    system.addParticle(1.0)
    system.addParticle(1.0)
    bond = openmm.HarmonicBondForce()
    bond.addBond(0, 1, 0.1, 1000.0)
    system.addForce(bond)
    (tmp_path / "dimer.xml").write_text(openmm.XmlSerializer.serialize(system), encoding="utf-8")
    write_json(tmp_path / "dimer.json", {"positions_nm": [[0, 0, 0], [0.2, 0, 0]]})
    systems = {"dimer": {"system_xml": "dimer.xml", "positions_json": "dimer.json",
                         "description": "Harmonic dimer", "force_field": "toy harmonic bond"}}
    hub = ScienceToolHub(make_config(tmp_path, systems=systems), tmp_path / "session")
    arguments = {"system_id": "dimer", "tolerance_kj_mol_nm": 1.0, "max_iterations": 1000,
                 "platform": "Reference", "cpus": 1, "wall_seconds": 120, "return_positions": False}
    result = hub.call("openmm_minimize", arguments)
    assert result["ok"], result["error"]
    assert result["data"]["converged"] is True
    assert result["data"]["audit_payload"]["energy_unit"] == "kJ/mol"
    assert len(result["cost"]["hpc_jobs"]) == 1
    assert hub.call("openmm_minimize", {**arguments, "cpus": 2})["ok"] is False  # Reference is single-threaded


def test_schema_checker_matches_strict_subset():
    schema = {"type": "object", "properties": {"x": {"type": ["number", "null"]},
              "k": {"type": ["array", "null"], "items": {"type": "integer"}}},
              "required": ["x", "k"], "additionalProperties": False}
    check_schema(schema, {"x": None, "k": [1, 2, 3]})
    for bad in ({"x": True, "k": None}, {"x": 1.0}, {"x": 1.0, "k": [1.5]}, {"x": 1, "k": None, "y": 0}):
        with pytest.raises(ValueError):
            check_schema(schema, bad)


def mof_query(**changes):
    value = {"name_contains": None, "method": None, "data_type_contains": None, "identifier": None,
             "required_fields": None, "usable_link_only": None}
    value.update(changes)
    return value


def test_mof_database_catalog_is_free_and_bounded(hub):
    definition = tool(hub, "mof_database_catalog")
    properties = definition["parameters"]["properties"]
    assert "gcmc" in properties["method"]["enum"] and None in properties["method"]["enum"]
    assert "lcd_pld" in properties["required_fields"]["items"]["enum"]
    result = hub.call("mof_database_catalog", mof_query(method="gcmc", required_fields=["lcd_pld"],
                                                        usable_link_only=True))
    assert result["ok"], result["error"]
    assert {d["name"] for d in result["data"]["databases"]} == {"hMOF", "MOSAEC-DB", "H2-MOF (HyMARC)"}
    assert result["data"]["source"]["rows"] == 26
    assert hub.budget_status()["calls_used"] == 0 and hub.scheduler.status()["jobs"] == []
    assert hub.call("mof_database_catalog", mof_query(method="alchemy"))["ok"] is False
    assert hub.call("mof_database_catalog", mof_query(required_fields=["color"]))["ok"] is False


def test_mof_database_catalog_rejects_unreviewed_csv(tmp_path):
    config_path = make_config(tmp_path)
    shutil.copyfile(ROOT / "datasets/catalogs/mofdb.annotations.json", tmp_path / "annotations.json")
    (tmp_path / "mofdb.csv").write_bytes((ROOT / "datasets/catalogs/mofdb.csv").read_bytes() + b"\n")
    config = read_json(config_path)
    config["tools"]["mof_database_catalog"] = {"csv": "mofdb.csv", "annotations": "annotations.json"}
    write_json(config_path, config)
    hub = ScienceToolHub(config_path, tmp_path / "session")
    assert "changed" in hub.catalog()["unavailable_tools"]["mof_database_catalog"]

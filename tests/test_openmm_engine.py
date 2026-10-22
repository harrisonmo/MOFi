"""Toy-system checks for the OpenMM broker feeding the frozen audit contract."""

import json
from pathlib import Path
import subprocess

import numpy as np
import pytest

openmm = pytest.importorskip("openmm")
from openmm import unit  # noqa: E402

from rsi.openmm_engine import from_openmm_event, minimize, run_isolated  # noqa: E402
from rsi.tool_evidence import from_uff_event  # noqa: E402
from rsi.tool_runtime import Registry  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def stretched_dimer(r0=0.1, k=1000.0, r=0.2):
    """Two particles joined by a harmonic bond, started away from r0 (nm)."""
    system = openmm.System()
    system.addParticle(1.0)
    system.addParticle(1.0)
    bond = openmm.HarmonicBondForce()
    bond.addBond(0, 1, r0, k)
    system.addForce(bond)
    return system, [openmm.Vec3(0, 0, 0), openmm.Vec3(r, 0, 0)] * unit.nanometer


@pytest.fixture
def event():
    system, positions = stretched_dimer()
    return minimize(system, positions, tolerance=1.0, platform="Reference")


def test_minimization_reports_native_units(event):
    data = event["result"]["data"]
    # E = k/2 (r - r0)^2 = 500 * 0.1^2 kJ/mol before minimization.
    assert data["initial_energy"] == pytest.approx(5.0)
    assert data["energy"] < 1e-3
    assert data["max_force"] <= 1.0
    assert data["converged"] is True


def test_event_flows_through_frozen_audit_tool(event):
    spec = json.loads((ROOT / "tools/q2/manual-audit-v1.json").read_text())
    registry = Registry()
    registry.register(spec)
    evidence = from_openmm_event(event)
    assert evidence["payload"]["energy_unit"] == "kJ/mol"
    result = registry.call(spec["id"], spec["version"], evidence["payload"])
    data = event["result"]["data"]
    assert result["ok"], result["error"]
    assert result["output"]["energy_kj_mol"] == pytest.approx(data["energy"])
    assert result["output"]["delta_kj_mol"] == pytest.approx(data["energy"] - data["initial_energy"])
    assert result["output"]["max_force_kj_mol_nm"] == pytest.approx(data["max_force"])
    assert result["output"]["converged"] is True


def test_unconverged_run_is_reported_as_such():
    system, positions = stretched_dimer(k=1e6)
    event = minimize(system, positions, tolerance=1e-6, max_iterations=1, platform="Reference")
    assert from_openmm_event(event)["payload"]["converged"] is False


def test_engines_cannot_be_crossed(event):
    with pytest.raises(ValueError, match="UFF"):
        from_uff_event(event)
    event["result"]["ok"] = False
    with pytest.raises(ValueError, match="OpenMM"):
        from_openmm_event(event)


def test_invalid_requests_fail_before_compute():
    system, positions = stretched_dimer()
    for kwargs in ({"max_iterations": 0}, {"tolerance": float("nan")}, {"platform": "Bogus"},
                   {"threads": 2, "platform": "Reference"}, {"max_wall_seconds": -1.0}):
        with pytest.raises(ValueError):
            minimize(system, positions, **{"platform": "Reference", **kwargs})
    with pytest.raises(ValueError, match="positions"):
        minimize(system, positions[:1], platform="Reference")
    with pytest.raises(ValueError, match="finite"):
        minimize(system, [[0, 0, 0], [float("nan"), 0, 0]], platform="Reference")
    with pytest.raises(ValueError, match="non-periodic"):
        minimize(system, positions, box_vectors=np.eye(3), platform="Reference")
    system.addConstraint(0, 1, 0.1)
    with pytest.raises(ValueError, match="Constrained"):
        minimize(system, positions, platform="Reference")


def test_event_records_provenance(event):
    action, data = event["action"], event["result"]["data"]
    assert len(action["system_sha256"]) == 64
    assert (action["num_particles"], action["num_free_particles"], action["periodic"]) == (2, 2, False)
    assert data["iterations"] > 0 and data["stopped_by_wall_time"] is False
    assert action["elapsed_seconds"] >= 0


def test_fixed_particles_are_excluded_from_convergence():
    system, positions = stretched_dimer()
    system.setParticleMass(0, 0.0)
    push = openmm.CustomExternalForce("10*x")
    push.addParticle(0, [])
    system.addForce(push)
    event = minimize(system, positions, tolerance=1.0, platform="Reference", return_positions=True)
    data = event["result"]["data"]
    assert event["action"]["num_free_particles"] == 1
    assert data["converged"] is True
    assert data["positions_nm"][0] == pytest.approx([0, 0, 0])
    assert abs(data["positions_nm"][1][0]) == pytest.approx(0.1, abs=1e-3)


def test_periodic_box_vectors_are_applied():
    system, _ = stretched_dimer()
    system.getForce(0).setUsesPeriodicBoundaryConditions(True)
    system.setDefaultPeriodicBoxVectors(*(openmm.Vec3(*row) for row in 3 * np.eye(3)))
    # Minimum-image distance is 0.1 nm = r0 only inside the supplied 1 nm box.
    event = minimize(system, [[0.05, 0, 0], [0.95, 0, 0]], box_vectors=np.eye(3), platform="Reference")
    assert event["action"]["periodic"] is True
    assert event["result"]["data"]["initial_energy"] == pytest.approx(0.0, abs=1e-9)


def test_wall_time_limit_stops_minimization():
    system, positions = stretched_dimer(k=1e6)
    event = minimize(system, positions, tolerance=1e-9, max_iterations=10000,
                     max_wall_seconds=1e-9, platform="Reference")
    data = event["result"]["data"]
    assert data["stopped_by_wall_time"] is True
    assert data["converged"] is False


def test_numerical_failure_is_an_unsuccessful_event():
    system = openmm.System()
    system.addParticle(1.0)
    wall = openmm.CustomExternalForce("1/x")
    wall.addParticle(0, [])
    system.addForce(wall)
    event = minimize(system, [[0, 0, 0]], platform="Reference")
    assert event["result"]["ok"] is False
    with pytest.raises(ValueError, match="OpenMM"):
        from_openmm_event(event)


def write_task(folder):
    system, _ = stretched_dimer()
    (folder / "system.xml").write_text(openmm.XmlSerializer.serialize(system), encoding="utf-8")
    (folder / "positions.json").write_text(json.dumps({"positions_nm": [[0, 0, 0], [0.2, 0, 0]]}),
                                           encoding="utf-8")
    return folder / "system.xml", folder / "positions.json"


def test_isolated_file_task_flows_through_audit(tmp_path):
    system_xml, positions_json = write_task(tmp_path)
    output = tmp_path / "event.json"
    event = run_isolated(system_xml, positions_json, output, timeout_seconds=120,
                         tolerance=1.0, platform="Reference")
    assert event["result"]["ok"], event["result"].get("error")
    assert event["action"]["max_wall_seconds"] == pytest.approx(96.0)
    assert set(event["action"]["inputs"]) == {"system_xml_sha256", "positions_json_sha256"}
    assert from_openmm_event(event)["payload"]["converged"] is True
    with pytest.raises(ValueError, match="overwrite"):
        run_isolated(system_xml, positions_json, output, timeout_seconds=120)


def test_isolated_failures_are_reported(tmp_path, monkeypatch):
    system_xml, positions_json = write_task(tmp_path)
    bad = run_isolated(system_xml, positions_json, tmp_path / "bad.json", timeout_seconds=120,
                       max_iterations=0, platform="Reference")
    assert bad["result"]["error"]["type"] == "ChildFailed"
    assert "max_iterations" in bad["result"]["error"]["message"]

    def hang(command, **kwargs):
        raise subprocess.TimeoutExpired(command, kwargs["timeout"])

    monkeypatch.setattr(subprocess, "run", hang)
    slow = run_isolated(system_xml, positions_json, tmp_path / "slow.json", timeout_seconds=1)
    assert slow["result"]["error"]["type"] == "Timeout"

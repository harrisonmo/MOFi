"""Zeo++ pore tool: recipe guards, argv, parsing, budgets; a stand-in replaces ``network``."""

import json
import os
from pathlib import Path
import sys

import pytest

pytest.importorskip("xtal")

from rsi.zeopp import core  # noqa: E402
from rsi.zeopp.registry import TOOL, ZeoppToolRegistry  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]

# Captured from a real Zeo++ 0.3 run on MFU-4l (Crystal-Builder tests/conftest_zeo.py).
CANNED = {
    "-res": "   structure.res    18.72736 9.18223  18.72243\n",
    "-chan": ("pores.chan   1 channels identified of dimensionality 3 \n"
              "Channel  0  18.7273  9.18228  18.7225\n"
              "pores.chan summary(Max_of_columns_above)   18.7273 9.18228  18.7225  "
              "probe_rad: 1.86  probe_diam: 3.72\n"),
    "-sa": ("@ pores.sa Unitcell_volume: 29955.3   Density: 0.559382   ASA_A^2: 5358.14 "
            "ASA_m^2/cm^3: 1788.71 ASA_m^2/g: 3197.65 NASA_A^2: 0 NASA_m^2/cm^3: 0 NASA_m^2/g: 0\n"
            "Number_of_channels: 1 Channel_surface_area_A^2: 5358.14  \n"
            "Number_of_pockets: 0 Pocket_surface_area_A^2: \n"),
    "-volpo": ("@ pores.vol Unitcell_volume: 29955.3   Density: 0.559382   POAV_A^3: 22202.9 "
               "POAV_Volume_fraction: 0.7412 POAV_cm^3/g: 1.32503 PONAV_A^3: 0 "
               "PONAV_Volume_fraction: 0 PONAV_cm^3/g: 0\n"),
    "-block": "0\n",
}
ARITY = {"-res": 1, "-chan": 2, "-sa": 4, "-volpo": 4, "-psd": 4, "-block": 3}


def psd_text():
    lines = ["Pore size distribution histogram", "Bin size (A): 0.1", "Number of bins: 1000",
             "From: 0", "To: 100", "Total samples: 5000", "Accessible samples: 2910",
             "Fraction of sample points in node spheres: 0.582",
             "Fraction of sample points outside node spheres: 0", "",
             "Bin Count Cumulative_dist Derivative_dist"]
    for index in range(1000):
        count = {115: 40, 187: 900}.get(index, 0)
        lines.append(f"{index / 10:.1f} {count} 0 0")
    return "\n".join(lines) + "\n"


STANDIN = f"""
import json, sys
CANNED = {CANNED!r}
ARITY = {ARITY!r}
args = sys.argv[1:]
json.dump(args, open("argv.json", "w"))
PSD = {psd_text()!r}
i = 0
while i < len(args):
    flag = args[i]
    if flag in ARITY:
        target = args[i + ARITY[flag]]
        open(target, "w").write(PSD if flag == "-psd" else CANNED[flag])
        i += ARITY[flag] + 1
    else:
        i += 1
sys.exit(int(__import__("os").environ.get("STANDIN_EXIT", "0")))
"""


def write_program(directory, body):
    """Executable stand-in; Windows needs a .bat launcher around the script."""
    script = directory / "network.py"
    script.write_text(body, encoding="utf-8")
    if os.name != "nt":
        launcher = directory / "network"
        launcher.write_text(f"#!/bin/sh\nexec \"{sys.executable}\" \"{script}\" \"$@\"\n")
        launcher.chmod(0o755)
    else:
        launcher = directory / "network.bat"
        launcher.write_text(f'@echo off\r\n"{sys.executable}" "{script}" %*\r\n')
    return launcher


def recipe(**changes):
    value = {"schema_version": 1, "structure": "host.cif", "probe_label": "N2",
             "probe_radius_angstrom": 1.86, "channel_radius_angstrom": 1.86,
             "psd_probe_radius_angstrom": 1.2, "radii": "zeo-default", "high_accuracy": True,
             "samples": {"surface_per_atom": 200, "volume": 2000, "psd": 2000, "block": 200},
             "block_pockets": True, "grid_crosscheck": None,
             "budget": {"timeout_seconds": 60, "max_cell_atoms": 100}}
    value.update(changes)
    return value


@pytest.fixture
def inputs(tmp_path, monkeypatch):
    (tmp_path / "host.cif").write_text(
        "data_host\n_cell_length_a 12\n_cell_length_b 12\n_cell_length_c 12\n"
        "_cell_angle_alpha 90\n_cell_angle_beta 90\n_cell_angle_gamma 90\n"
        "_symmetry_space_group_name_H-M 'P 1'\nloop_\n_atom_site_label\n_atom_site_type_symbol\n"
        "_atom_site_fract_x\n_atom_site_fract_y\n_atom_site_fract_z\nC1 C 0.5 0.5 0.5\n",
        encoding="utf-8")
    core.write_json(tmp_path / "task.json", recipe())
    core.write_json(tmp_path / "registry.json", {"schema_version": 1, "tasks": {"approved": "task.json"},
                    "session_budget": {"max_calls": 1, "max_reserved_wall_seconds": 60}})
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    monkeypatch.setenv(core.ENV_VAR, str(write_program(bin_dir, STANDIN)))
    return tmp_path


@pytest.mark.parametrize("change", [
    {"radii": "vdw"}, {"probe_radius_angstrom": 0}, {"probe_radius_angstrom": 6.0},
    {"samples": {"surface_per_atom": 50, "volume": 2000, "psd": 2000, "block": 200}},
    {"budget": {"timeout_seconds": 1e6, "max_cell_atoms": 100}}, {"extra": 1},
    {"grid_crosscheck": {"spacing_angstrom": 0.05}}, {"high_accuracy": 1}])
def test_recipe_rejects_unadmitted_settings(change):
    with pytest.raises(ValueError):
        core.validate_recipe(recipe(**change))


def test_shipped_recipes_are_valid_and_point_at_real_structures():
    config = core.read_json(ROOT / "configs/zeopp-tools.json")
    for relative in config["tasks"].values():
        path = ROOT / "configs" / relative
        value = core.validate_recipe(core.read_json(path))
        assert (path.parent / value["structure"]).resolve().is_file()


def test_one_invocation_shares_the_decomposition(inputs):
    line = core.argv(recipe(), "network")
    assert line[1] == "-ha" and line[-1] == core.INPUT
    for flag in ARITY:
        assert line.count(flag) == 1
    assert line[line.index("-volpo") + 1:line.index("-volpo") + 5] == ["1.86", "1.86", "2000", "pores.vol"]
    assert "-block" not in core.argv(recipe(block_pockets=False), "network")


def test_prepare_and_execute_produce_structured_evidence(inputs):
    run = inputs / "run"
    core.prepare(inputs / "task.json", run)
    result = core.execute(run)
    assert result["diameters_angstrom"]["largest_free_sphere"] == pytest.approx(9.18223)
    assert result["channels"]["count"] == 1 and result["channels"]["dimensionalities"] == [3]
    assert result["surface_area"]["accessible_m2_g"] == pytest.approx(3197.65)
    assert result["pore_volume"]["accessible_fraction"] == pytest.approx(0.7412)
    assert result["pore_size_distribution"]["mode_diameter_angstrom"] == pytest.approx(18.75)
    assert result["blocking"]["sphere_count"] == 0
    assert result["flags"] == []
    manifest = core.read_json(run / "manifest.json")
    assert manifest["status"] == "completed" and manifest["argv"][-1] == core.INPUT
    assert set(manifest["output_hashes"]) >= {"pores.res", "pores.block", "pores.json"}
    assert json.loads((run / "argv.json").read_text())[0] == "-ha"


def test_tampered_frozen_input_is_refused(inputs):
    run = inputs / "run"
    core.prepare(inputs / "task.json", run)
    (run / core.INPUT).write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="changed"):
        core.execute(run)


def test_engine_failure_is_recorded(inputs, monkeypatch):
    monkeypatch.setenv("STANDIN_EXIT", "3")
    run = inputs / "run"
    core.prepare(inputs / "task.json", run)
    with pytest.raises(RuntimeError, match="status 3"):
        core.execute(run)
    assert core.read_json(run / "manifest.json")["status"] == "failed"


def test_partial_occupancy_is_refused_before_any_run(inputs):
    text = (inputs / "host.cif").read_text().replace(
        "_atom_site_fract_z\n", "_atom_site_fract_z\n_atom_site_occupancy\n").replace(
        "0.5 0.5 0.5\n", "0.5 0.5 0.5 0.5\n")
    (inputs / "host.cif").write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match="refused"):
        core.prepare(inputs / "task.json", inputs / "run")


def test_grid_crosscheck_flags_disagreement_with_engine(inputs):
    core.write_json(inputs / "task.json", recipe(grid_crosscheck={"spacing_angstrom": 0.5}))
    run = inputs / "run"
    core.prepare(inputs / "task.json", run)
    result = core.execute(run)
    check = result["grid_crosscheck"]
    # The canned MFU-4l numbers cannot match a lone carbon atom.
    assert check["agrees"] is False and "GRID_CROSSCHECK_DISAGREES" in result["flags"]
    assert check["channel_count"] == 1


def test_flags_name_pockets_and_blocked_probe():
    result = {"channels": {"count": 0}, "diameters_angstrom": {"largest_free_sphere": 3.0},
              "surface_area": {"nonaccessible_m2_g": 10.0}, "pore_volume": {"nonaccessible_cm3_g": 0.1},
              "blocking": None}
    assert core.flags(result, recipe()) == ["NO_PERCOLATING_CHANNEL", "FREE_SPHERE_BELOW_PROBE_DIAMETER",
                                            "INACCESSIBLE_POCKETS_PRESENT", "POCKETS_NOT_BLOCKED"]


def test_catalog_hides_paths_and_runs_nothing(inputs):
    registry = ZeoppToolRegistry(inputs / "registry.json", inputs / "session")
    catalog = registry.catalog()
    function = catalog["tools"][0]["function"]
    assert function["name"] == TOOL and function["parameters"]["additionalProperties"] is False
    assert function["parameters"]["properties"]["task_id"]["enum"] == ["approved"]
    assert "host.cif" not in json.dumps(catalog["tasks"])
    assert not (inputs / "session").exists()


@pytest.mark.parametrize("name,args", [("unknown", {"task_id": "approved"}),
    (TOOL, {"task_id": "../task.json"}), (TOOL, {"task_id": "approved", "probe": 1.0}), (TOOL, {})])
def test_agent_cannot_override_registration(inputs, name, args):
    result = ZeoppToolRegistry(inputs / "registry.json", inputs / "session").call(name, args)
    assert not result["ok"] and result["cost"] is None
    assert not (inputs / "session").exists()


def test_dispatch_records_evidence_and_budget_persists(inputs):
    result = ZeoppToolRegistry(inputs / "registry.json", inputs / "session").call(TOOL, {"task_id": "approved"})
    assert result["ok"], result["error"]
    assert result["data"]["channels"]["count"] == 1
    assert Path(result["evidence"]["run_directory"]).is_dir() and result["evidence"]["pores_sha256"]
    again = ZeoppToolRegistry(inputs / "registry.json", inputs / "session").call(TOOL, {"task_id": "approved"})
    assert not again["ok"] and "exhausted" in again["error"]["message"]
    ledger = core.read_json(inputs / "session/ledger.json")
    assert [c["status"] for c in ledger["calls"]] == ["completed"]


def test_changed_task_input_after_binding_is_refused(inputs):
    registry = ZeoppToolRegistry(inputs / "registry.json", inputs / "session")
    (inputs / "host.cif").write_text((inputs / "host.cif").read_text() + "\n", encoding="utf-8")
    result = registry.call(TOOL, {"task_id": "approved"})
    assert not result["ok"] and "changed" in result["error"]["message"]


def test_both_repository_revisions_are_recorded():
    assert core.git_revision(core.ROOT)["commit"]
    assert core.git_revision(core.CRYSTAL_ROOT)["commit"]

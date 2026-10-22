"""ORCA engine: config refusals, input writing, native parsing and controller verification.

Real ORCA is not needed: a stand-in program writes ORCA 6 formatted job.out, job.engrad and job.xyz,
and STANDIN_MODE makes it fail in the ways the controller must catch.
"""

import json
import os
from pathlib import Path
import sys

import pytest

pytest.importorskip("ase")
pytest.importorskip("xtal")

from rsi.orca import core  # noqa: E402
from rsi.orca.parse import HARTREE_EV, parse_engrad, parse_output  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]

STANDIN = r'''
import os, re, sys
mode = os.environ.get("STANDIN_MODE", "")
inp = open(sys.argv[1]).read()
words = " ".join(l[1:] for l in inp.splitlines() if l.startswith("!")).split()
nprocs = int(re.search(r"nprocs\s+(\d+)", inp).group(1)) if "%pal" in inp else 1
if mode == "wrong_nprocs":
    nprocs += 1
xyz = re.search(r"\*xyzfile\s+(-?\d+)\s+(\d+)\s+(\S+)", inp).group(3)
rows = [l.split() for l in open(xyz).read().splitlines()[2:] if l.strip()]
n = len(rows)
Z = {"H": 1, "C": 6, "N": 7, "O": 8, "Cu": 29}
energy = -1.25 * n - 0.5
out = ["                                 *****************",
       "                                 * O   R   C   A *",
       "                                 *****************",
       "                       Program Version 6.1.0 -  RELEASE  -", ""]
if nprocs > 1:
    out.append(f"           *        Program running with {nprocs} parallel MPI-processes     *")
opt = any(w.endswith("Opt") for w in words)
cycles = 3 if opt else 1
for cycle in range(1, cycles + 1):
    if opt:
        out += ["*" * 60, f"*                GEOMETRY OPTIMIZATION CYCLE   {cycle}            *", "*" * 60]
    if mode == "scf_fail" and cycle == cycles:
        out.append("SCF NOT CONVERGED AFTER 125 CYCLES")
    else:
        out.append("SCF CONVERGED AFTER  11 CYCLES")
    out += ["----------------", "ORBITAL ENERGIES", "----------------", "",
            "  NO   OCC          E(Eh)            E(eV) ",
            "   0   2.0000     -18.937557      -515.3172 ",
            "   1   2.0000      -0.300000        -8.1634 ",
            "   2   0.0000       0.025000         0.6803 ",
            "", "                    ********************************",
            "                    * MULLIKEN POPULATION ANALYSIS *",
            "-------------------------   --------------------",
            f"FINAL SINGLE POINT ENERGY      {energy - 0.001 * (cycles - cycle):.12f}",
            "-------------------------   --------------------"]
if opt:
    out += ["", "                    ***********************HURRAY********************",
            "                    ***        THE OPTIMIZATION HAS CONVERGED     ***",
            "                    *******************************************************"]
    with open("job.xyz", "w") as f:
        f.write(f"{n}\nCoordinates from ORCA-job job\n" + "".join(f"{r[0]} {r[1]} {r[2]} {r[3]}\n" for r in rows))
if "EnGrad" in words or opt:
    lines = ["#", "# Number of atoms", "#", f"{n}", "#", "# The current total energy in Eh", "#", f"{energy:.12f}",
             "#", "# The current gradient in Eh/bohr", "#"]
    lines += [f"{0.001 * (i + 1) * (k == 2):.9f}" for i in range(n) for k in range(3)]
    lines += ["#", "# The atomic numbers and current coordinates in Bohr", "#"]
    lines += [f"{Z[r[0]]} {float(r[1]) / 0.529177:.7f} {float(r[2]) / 0.529177:.7f} {float(r[3]) / 0.529177:.7f}"
              for r in rows]
    open("job.engrad", "w").write("\n".join(lines) + "\n")
if "Freq" in words:
    out += ["-----------------------", "VIBRATIONAL FREQUENCIES", "-----------------------", ""]
    for i in range(3 * n):
        value = 0.0 if i < 6 else 400.0 + 10 * i
        if mode == "imaginary" and i == 6:
            out.append(f"   {i}:     -45.12 cm**-1 ***imaginary mode***")
        else:
            out.append(f"   {i}:      {value:.2f} cm**-1")
    out += ["", "------------", "NORMAL MODES", "------------",
            "Zero point energy                ...      0.02100000 Eh      13.18 kcal/mol",
            "Total Enthalpy                    ...    -76.30000000 Eh",
            "Final Gibbs free energy         ...    -76.32000000 Eh"]
if mode != "abnormal":
    out += ["", "                             ****ORCA TERMINATED NORMALLY****"]
print("\n".join(out))
sys.exit(int(os.environ.get("STANDIN_EXIT", "0")))
'''


def write_orca(directory):
    """Stand-in ORCA folder: the program and an orca_scf beside it, as binary() requires."""
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "orca_scf").write_text("", encoding="utf-8")
    script = directory / "orca.py"
    script.write_text(STANDIN, encoding="utf-8")
    if os.name == "nt":
        launcher = directory / "orca.bat"
        launcher.write_text(f'@echo off\r\n"{sys.executable}" "{script}" %*\r\n')
    else:
        launcher = directory / "orca"
        launcher.write_text(f"#!/bin/sh\nexec \"{sys.executable}\" \"{script}\" \"$@\"\n")
        launcher.chmod(0o755)
    return launcher


def orca_config(**changes):
    value = {"schema_version": 1, "structure": "water.xyz", "charge": 0, "spin_multiplicity": 1,
             "task": "energy", "frequencies": False,
             "method": {"functional": "R2SCAN-3C", "basis": None, "dispersion": None, "ri": None,
                        "scf_convergence": None},
             "budget": {"cpus": 1, "maxcore_mb": 500, "wall_seconds": 60, "max_atoms": 60}}
    for key, item in changes.items():
        if key == "method":
            value["method"] = {**value["method"], **item}
        else:
            value[key] = item
    return value


@pytest.fixture
def standin(tmp_path, monkeypatch):
    monkeypatch.setenv(core.ENV_VAR, str(write_orca(tmp_path / "orca-bin")))
    monkeypatch.delenv("STANDIN_MODE", raising=False)
    (tmp_path / "water.xyz").write_bytes((ROOT / "examples/orca/water.xyz").read_bytes())
    return tmp_path


def run(folder, **changes):
    (folder / "config.json").write_text(json.dumps(orca_config(**changes)), encoding="utf-8")
    output = folder / "run-out"
    core.prepare(folder / "config.json", output)
    return output, core.execute(output)


@pytest.mark.parametrize("changes, message", [
    ({"method": {"functional": "NOT-A-FUNCTIONAL"}}, "not an ORCA 6.1 functional"),
    ({"method": {"basis": "def2-TZVP"}}, "own basis"),
    ({"method": {"functional": "PBE0", "basis": None}}, "not an ORCA 6.1 basis"),
    ({"method": {"functional": "WB97X-D3BJ", "basis": "def2-SVP", "dispersion": "D4"}}, "already includes"),
    ({"method": {"ri": "RIJONX"}}, "ri must be"),
    ({"task": "optimize"}, "optimization settings"),
    ({"optimization": {"level": "Opt", "max_cycles": 10}}, "Only optimize"),
    ({"task": "md"}, "task must be"),
    ({"budget": {"cpus": 0, "maxcore_mb": 500, "wall_seconds": 60, "max_atoms": 60}}, "cpus"),
    ({"extra": 1}, "Expected required keys")])
def test_config_refuses_unadmitted_methods(changes, message):
    with pytest.raises(ValueError, match=message):
        core.validate_config(orca_config(**changes))


def test_impossible_spin_and_periodic_structures_are_refused_before_writing(standin):
    with pytest.raises(ValueError, match="odd multiplicity"):
        core.check(orca_config(spin_multiplicity=2), standin / "water.xyz")
    (standin / "box.extxyz").write_text(
        '1\nLattice="5 0 0 0 5 0 0 0 5" Properties=species:S:1:pos:R:3 pbc="T T T"\nO 0 0 0\n', encoding="utf-8")
    with pytest.raises(ValueError, match="periodic"):
        core.check(orca_config(), standin / "box.extxyz")
    with pytest.raises(ValueError, match="atom budget"):
        core.check(orca_config(budget={"cpus": 1, "maxcore_mb": 500, "wall_seconds": 60, "max_atoms": 2}),
                   standin / "water.xyz")


def test_prepare_writes_a_crystal_builder_input(standin):
    (standin / "config.json").write_text(json.dumps(orca_config(
        task="optimize", frequencies=True, optimization={"level": "TightOpt", "max_cycles": 50},
        method={"functional": "PBE0", "basis": "def2-TZVP", "dispersion": "D4", "scf_convergence": "TightSCF"},
        budget={"cpus": 4, "maxcore_mb": 1500, "wall_seconds": 60, "max_atoms": 60})), encoding="utf-8")
    manifest = core.prepare(standin / "config.json", standin / "out")
    text = (standin / "out/job.inp").read_text(encoding="utf-8")
    assert "! PBE0 D4 def2-TZVP TightSCF TightOpt Freq" in text
    assert "%pal\n  nprocs 4\nend" in text and "%maxcore 1500" in text and "MaxIter 50" in text
    assert text.rstrip().endswith("*xyzfile 0 1 job_from_crystal_builder.xyz")
    assert (manifest["atoms"], manifest["electrons"], manifest["elements"]) == (3, 10, ["H", "O"])
    assert "xtal/orca/input.py" in manifest["code_hashes"]


def test_energy_single_point_is_verified_from_native_output(standin):
    output, summary = run(standin)
    assert summary["energy_eh"] == pytest.approx(-4.25)
    assert summary["energy_ev"] == pytest.approx(-4.25 * HARTREE_EV)
    assert summary["converged"] is True and summary["orca_version"] == "6.1.0"
    assert summary["homo_lumo_gap_ev"] == pytest.approx(0.6803 + 8.1634)
    assert summary["forces_ev_per_angstrom"] is None
    manifest = core.read_json(output / "manifest.json")
    assert manifest["status"] == "completed" and manifest["orca_executable_sha256"]


def test_gradient_returns_forces_checked_against_engrad(standin):
    _, summary = run(standin, task="gradient")
    assert len(summary["forces_ev_per_angstrom"]) == 3
    assert summary["forces_ev_per_angstrom"][2][2] == pytest.approx(-0.003 * HARTREE_EV / 0.529177210903)
    assert summary["checks"]["max_gradient_eh_per_bohr"] == pytest.approx(0.003)


def test_optimization_with_frequencies_reports_modes_and_thermochemistry(standin):
    _, summary = run(standin, task="optimize", frequencies=True, optimization={"level": "Opt", "max_cycles": 20})
    assert summary["converged"] is True and summary["checks"]["optimization_cycles"] == 3
    assert len(summary["final_positions_angstrom"]) == 3
    assert summary["imaginary_frequencies"] == 0 and len(summary["frequencies_cm-1"]) == 9
    assert summary["final_gibbs_free_energy_eh"] == pytest.approx(-76.32)


def test_imaginary_modes_are_counted_not_hidden(standin, monkeypatch):
    monkeypatch.setenv("STANDIN_MODE", "imaginary")
    _, summary = run(standin, frequencies=True)
    assert summary["imaginary_frequencies"] == 1 and summary["lowest_frequency_cm-1"] == pytest.approx(-45.12)


def test_unconverged_scf_is_reported_as_not_converged(standin, monkeypatch):
    monkeypatch.setenv("STANDIN_MODE", "scf_fail")
    _, summary = run(standin)
    assert summary["converged"] is False and summary["checks"]["scf_not_converged"] == 1


@pytest.mark.parametrize("mode, error", [("abnormal", "did not terminate normally"),
                                         ("wrong_nprocs", "processes")])
def test_untrustworthy_runs_fail_and_are_recorded(standin, monkeypatch, mode, error):
    monkeypatch.setenv("STANDIN_MODE", mode)
    with pytest.raises((RuntimeError, ValueError), match=error):
        run(standin)
    assert core.read_json(standin / "run-out/manifest.json")["status"] == "failed"


def test_nonzero_exit_fails_even_with_normal_termination(standin, monkeypatch):
    monkeypatch.setenv("STANDIN_EXIT", "2")
    with pytest.raises(RuntimeError, match="status 2"):
        run(standin)


def test_changed_input_blocks_execution(standin):
    (standin / "config.json").write_text(json.dumps(orca_config()), encoding="utf-8")
    core.prepare(standin / "config.json", standin / "out")
    inp = standin / "out/job.inp"
    inp.write_text(inp.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Frozen input changed"):
        core.execute(standin / "out")


def test_binary_must_be_the_quantum_chemistry_program(tmp_path, monkeypatch):
    screen_reader = tmp_path / "usr-bin" / "orca"
    screen_reader.parent.mkdir()
    screen_reader.write_text("#!/usr/bin/python3\n", encoding="utf-8")
    monkeypatch.setenv(core.ENV_VAR, str(screen_reader))
    assert core.binary() is None
    monkeypatch.setenv(core.ENV_VAR, "orca")
    assert core.binary() is None


def test_parser_reads_unrestricted_orbitals_and_failures():
    text = ("SCF CONVERGED AFTER  9 CYCLES\nORBITAL ENERGIES\n----------------\n"
            "                 SPIN UP ORBITALS\n  NO   OCC          E(Eh)            E(eV) \n"
            "   0   1.0000      -0.500000       -13.6057 \n   1   0.0000       0.100000         2.7211 \n"
            "                 SPIN DOWN ORBITALS\n  NO   OCC          E(Eh)            E(eV) \n"
            "   0   0.0000      -0.200000        -5.4423 \n\n   *****\n"
            "FINAL SINGLE POINT ENERGY        -0.499\n")
    native = parse_output(text)
    assert native["homo_lumo_gap_ev"] == pytest.approx(-5.4423 + 13.6057)
    assert native["terminated_normally"] is False and native["scf_all_converged"] is True
    assert parse_output(text.replace("SCF CONVERGED AFTER  9 CYCLES", ""))["scf_all_converged"] is False


def test_engrad_parser_rejects_truncated_files():
    text = "#\n# Number of atoms\n#\n2\n#\n# The current total energy in Eh\n#\n-1.0\n#\n# The current gradient in Eh/bohr\n#\n0\n0\n0\n"
    with pytest.raises(ValueError):
        parse_engrad(text)

"""Independent parsers for native ORCA output; nothing here imports ORCA, ASE or Crystal-Builder."""

from __future__ import annotations

import math
import re

# CODATA 2018, the constants ORCA 6 uses; Hartree stays the primary unit in every record.
HARTREE_EV = 27.211386245988
BOHR_ANGSTROM = 0.529177210903

NORMAL = "****ORCA TERMINATED NORMALLY****"
VERSION = re.compile(r"Program Version\s+(\S+)")
PROCESSES = re.compile(r"Program running with\s+(\d+)\s+parallel MPI-processes")
FINAL_ENERGY = re.compile(r"^FINAL SINGLE POINT ENERGY\s+(\S+)", re.M)
SCF_CONVERGED = re.compile(r"SCF CONVERGED AFTER\s+(\d+)\s+CYCLES")
SCF_FAILED = re.compile(r"SCF NOT CONVERGED")
OPT_CYCLE = re.compile(r"GEOMETRY OPTIMIZATION CYCLE\s+(\d+)")
OPT_CONVERGED = "THE OPTIMIZATION HAS CONVERGED"
OPT_EXHAUSTED = "The optimization did not converge"
FREQUENCY = re.compile(r"^\s*(\d+):\s+(-?\d+\.\d+)\s+cm\*\*-1(.*)$", re.M)
THERMO = {"total_enthalpy_eh": re.compile(r"Total [Ee]nthalpy\s+\.+\s+(\S+)\s+Eh"),
          "final_gibbs_free_energy_eh": re.compile(r"Final Gibbs free energy\s+\.+\s+(\S+)\s+Eh"),
          "zero_point_energy_eh": re.compile(r"Zero point energy\s+\.+\s+(\S+)\s+Eh")}
ORBITAL_ROW = re.compile(r"^\s*\d+\s+(\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s*$")


def _finite(values, name):
    out = [float(v) for v in values]
    if not all(math.isfinite(v) for v in out):
        raise ValueError(f"Nonfinite {name} in ORCA output")
    return out


def _last_section(text, header, stop):
    start = text.rfind(header)
    if start < 0:
        return None
    end = text.find(stop, start + len(header))
    return text[start:end if end >= 0 else len(text)]


def frequencies(text):
    """The last VIBRATIONAL FREQUENCIES table: every mode, the six or five zero ones included."""
    block = _last_section(text, "VIBRATIONAL FREQUENCIES", "NORMAL MODES")
    if block is None:
        return None
    rows = FREQUENCY.findall(block)
    values = _finite([value for _, value, _ in rows], "frequency")
    imaginary = [v for v, (_, _, note) in zip(values, rows) if v < 0 or "imaginary" in note]
    return {"cm-1": values, "imaginary_count": len(imaginary),
            "lowest_nonzero_cm-1": min((v for v in values if v != 0.0), default=None)}


def orbital_gap(text):
    """HOMO-LUMO gap (eV) from the last ORBITAL ENERGIES section; both spins for an open shell."""
    start = text.rfind("ORBITAL ENERGIES")
    if start < 0:
        return None
    occupied, virtual = [], []
    for line in text[start:].splitlines()[1:]:
        if line.strip().startswith("*") or "MOLECULAR ORBITALS" in line or "MULLIKEN" in line:
            break
        match = ORBITAL_ROW.match(line)
        if match:
            occupation, energy_ev = float(match.group(1)), float(match.group(3))
            (occupied if occupation > 0 else virtual).append(energy_ev)
    if not occupied or not virtual:
        return None
    return min(virtual) - max(occupied)


def parse_output(text):
    """Return termination, SCF, optimization, energy and frequency records from an ORCA .out file."""
    energies = _finite(FINAL_ENERGY.findall(text), "energy")
    iterations = [int(n) for n in SCF_CONVERGED.findall(text)]
    cycles = [int(n) for n in OPT_CYCLE.findall(text)]
    version = VERSION.search(text)
    processes = PROCESSES.search(text)
    thermo = {}
    for key, pattern in THERMO.items():
        found = pattern.findall(text)
        thermo[key] = _finite(found, key)[-1] if found else None
    return {"engine": "orca", "orca_version": version.group(1) if version else None,
            "mpi_processes": int(processes.group(1)) if processes else 1,
            "terminated_normally": NORMAL in text,
            "energies_eh": energies, "scf_iterations": iterations,
            "scf_not_converged": len(SCF_FAILED.findall(text)),
            # ORCA prints one SCF verdict per energy; an energy without one is not trusted as converged.
            "scf_all_converged": bool(energies) and not SCF_FAILED.search(text) and len(iterations) >= len(energies),
            "optimization_cycles": max(cycles) if cycles else 0,
            "optimization_converged": OPT_CONVERGED in text if cycles else None,
            "optimization_exhausted": OPT_EXHAUSTED in text,
            "frequencies": frequencies(text), "homo_lumo_gap_ev": orbital_gap(text), **thermo}


def parse_engrad(text):
    """Read ORCA's .engrad: atom count, energy (Eh), gradient (Eh/bohr) and coordinates (bohr)."""
    sections, current = {}, None
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            label = stripped.lstrip("#").strip()
            if label:
                current = label
                sections.setdefault(current, [])
            continue
        if stripped and current is not None:
            sections[current].append(stripped)
    def find(prefix):
        key = next((k for k in sections if k.startswith(prefix)), None)
        if key is None:
            raise ValueError(f"ORCA .engrad has no '{prefix}' section")
        return sections[key]
    n = int(find("Number of atoms")[0])
    energy = _finite(find("The current total energy")[:1], "energy")[0]
    gradient = _finite(find("The current gradient")[:3 * n], "gradient")
    rows = find("The atomic numbers and current coordinates")[:n]
    if n < 1 or len(gradient) != 3 * n or len(rows) != n:
        raise ValueError("ORCA .engrad sections do not match its atom count")
    numbers = [int(row.split()[0]) for row in rows]
    coordinates = [_finite(row.split()[1:4], "coordinate") for row in rows]
    return {"atoms": n, "energy_eh": energy, "atomic_numbers": numbers,
            "gradient_eh_per_bohr": [gradient[3 * i:3 * i + 3] for i in range(n)],
            "coordinates_bohr": coordinates}


def parse_xyz(text):
    """Plain XYZ as ORCA writes an optimized geometry: count, comment, symbol x y z."""
    lines = text.splitlines()
    n = int(lines[0].split()[0])
    rows = [line.split() for line in lines[2:2 + n]]
    if len(rows) != n or any(len(r) < 4 for r in rows):
        raise ValueError("ORCA geometry file is truncated")
    return {"symbols": [r[0] for r in rows], "positions_angstrom": [_finite(r[1:4], "coordinate") for r in rows]}

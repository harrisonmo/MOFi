"""Independent parsers for native GPAW and CP2K logs; no calculator import needed."""

from __future__ import annotations

import math
import re

# CP2K converts its own energies with CODATA 2006 (physcon.F); ASE receives CP2K-converted eV through
# the shell, so native Hartree values must use the same factor, not ase.units.Hartree.
CP2K_HARTREE_EV = 27.21138386


def _finite(values, name):
    out = [float(v) for v in values]
    if not all(math.isfinite(v) for v in out):
        raise ValueError(f"Nonfinite {name} in native log")
    return out


# Legacy GPAW prints "Converged after N iterations."; GPAW >= 25 prints "Converged in N steps".
GPAW_CONVERGED = re.compile(r"^\s*Converged (?:after|in) (\d+) (?:iterations|steps)", re.M)
GPAW_FORCE_ROW = re.compile(r"^\|\s*(\S+)\s*\|\s*(\S+)\s*\|\s*(\S+)\s*\|\s*#")


def _gpaw_last_forces(text):
    """Return the last printed GPAW >= 25 force table (eV/Angstrom), or None for legacy logs."""
    blocks = text.split("Forces  # eV/Ang")
    if len(blocks) < 2:
        return None
    rows = [GPAW_FORCE_ROW.match(line) for line in blocks[-1].splitlines()]
    rows = [m for m in rows if m]
    return [_finite(m.groups(), "force") for m in rows] or None


def parse_gpaw_text(text):
    """Return SCF and energy records from a GPAW txt log in eV."""
    iterations = [int(n) for n in GPAW_CONVERGED.findall(text)]
    energies = _finite(re.findall(r"^\s*Extrapolated:\s+(\S+)", text, re.M), "energy")
    free = _finite(re.findall(r"^\s*Free energy:\s+(\S+)", text, re.M), "free energy")
    # GPAW raises on SCF failure, so an energy block without a convergence line is a failure.
    return {"engine": "gpaw", "energies_ev": energies, "free_energies_ev": free,
            "scf_iterations": iterations, "scf_not_converged": max(0, len(energies) - len(iterations)),
            "scf_all_converged": bool(energies) and len(iterations) == len(energies),
            "last_forces_ev_per_angstrom": _gpaw_last_forces(text)}


ENERGY_CP2K = re.compile(r"ENERGY\| Total FORCE_EVAL \( QS \) energy \[(?:a\.u\.|hartree)\]:\s+(\S+)")


def parse_cp2k_output(text):
    """Return SCF and energy records from a CP2K output file, converted to eV."""
    hartree = _finite(ENERGY_CP2K.findall(text), "energy")
    iterations = [int(n) for n in re.findall(r"SCF run converged in\s+(\d+) steps", text)]
    failed = len(re.findall(r"SCF run NOT converged", text))
    # CP2K continues after an unconverged SCF unless told otherwise; count it explicitly.
    return {"engine": "cp2k", "energies_ev": [e * CP2K_HARTREE_EV for e in hartree], "energies_hartree": hartree,
            "scf_iterations": iterations, "scf_not_converged": failed,
            "scf_all_converged": bool(hartree) and failed == 0 and len(iterations) >= len(hartree),
            "last_forces_ev_per_angstrom": None}


def parse_native(backend, text):
    if backend == "gpaw":
        return parse_gpaw_text(text)
    if backend == "cp2k":
        return parse_cp2k_output(text)
    raise ValueError(f"No native log parser for backend {backend!r}")

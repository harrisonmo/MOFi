#!/usr/bin/env python3
"""Build an auditable ODAC23 CO2 adsorption-reference manifest.

The manifest deliberately distinguishes scalar reference rows from a strict
DFT host/guest/complex triplet.  ODAC23 supplies the scalar adsorption table
and official complex CIF archive, while the ARC-MOF archive is only a host
structure candidate until its identity and relaxed-reference provenance are
verified.  A row is therefore never silently promoted to a strict benchmark.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

KJ_MOL_PER_EV = 96.48533212331002


@dataclass
class ArchiveMember:
    name: str
    data: bytes

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.data).hexdigest()


def read_binding_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_archive_members(
    path: Path, name_predicate: Callable[[str], bool] | None = None
) -> dict[str, ArchiveMember]:
    members: dict[str, ArchiveMember] = {}
    with tarfile.open(path, "r:gz") as archive:
        for member in archive:
            if not member.isfile() or not member.name.lower().endswith(".cif"):
                continue
            if name_predicate is not None and not name_predicate(member.name):
                continue
            extracted = archive.extractfile(member)
            if extracted is None:
                continue
            members[member.name] = ArchiveMember(member.name, extracted.read())
    return members


def choose_complex_member(members: dict[str, ArchiveMember], mof_id: str) -> ArchiveMember | None:
    prefix = f"pristine_CO2/{mof_id}/{mof_id}_w_CO2"
    candidates = [
        member
        for name, member in members.items()
        if name.startswith(prefix) and name.endswith("_relaxed.cif")
    ]
    if not candidates:
        return None
    candidates.sort(
        key=lambda item: (
            "random" in item.name,
            "_1_relaxed.cif" not in item.name,
            item.name,
        )
    )
    return candidates[0]


def host_id_match(member_name: str, mof_id: str) -> bool:
    filename = Path(member_name).name
    prefix = f"DB12-{mof_id}"
    return filename.startswith(prefix + "_") or filename.startswith(prefix + ".")


def choose_host_member(members: dict[str, ArchiveMember], mof_id: str) -> ArchiveMember | None:
    candidates = [member for name, member in members.items() if host_id_match(name, mof_id)]
    if not candidates:
        return None
    candidates.sort(
        key=lambda item: (
            "clean_repeat" not in item.name,
            "_repeat.cif" not in item.name,
            "freeONLY" in item.name,
            item.name,
        )
    )
    return candidates[0]


def structure_from_cif(data: bytes):
    from pymatgen.core import Structure

    return Structure.from_str(data.decode("utf-8"), fmt="cif")


def minimum_image_vector(structure, first: int, second: int):
    distance, image = structure.lattice.get_distance_and_image(
        structure.frac_coords[first], structure.frac_coords[second]
    )
    frac = structure.frac_coords[second] + image - structure.frac_coords[first]
    return float(distance), structure.lattice.get_cartesian_coords(frac)


def find_co2_indices(structure) -> tuple[list[int], dict[str, Any]]:
    """Identify one linear C-O-O fragment in a single-CO2 complex CIF."""

    carbon = [i for i, site in enumerate(structure) if site.specie.symbol == "C"]
    oxygen = [i for i, site in enumerate(structure) if site.specie.symbol == "O"]
    candidates: list[tuple[float, int, int, int, float, float, float]] = []
    for c_idx in carbon:
        close = []
        for o_idx in oxygen:
            distance, vector = minimum_image_vector(structure, c_idx, o_idx)
            if distance <= 1.45:
                close.append((o_idx, distance, vector))
        for left in range(len(close)):
            for right in range(left + 1, len(close)):
                o1, d1, v1 = close[left]
                o2, d2, v2 = close[right]
                norm = float(math.sqrt(float(v1 @ v1) * float(v2 @ v2)))
                cosine = float((v1 @ v2) / norm) if norm else 1.0
                angle = math.degrees(math.acos(max(-1.0, min(1.0, cosine))))
                if angle < 150.0:
                    continue
                score = abs(d1 - 1.16) + abs(d2 - 1.16) + abs(angle - 180.0) / 100.0
                candidates.append((score, c_idx, o1, o2, d1, d2, angle))
    if not candidates:
        raise ValueError("could not identify a linear CO2 fragment")
    candidates.sort()
    score, c_idx, o1, o2, d1, d2, angle = candidates[0]
    return [c_idx, o1, o2], {
        "method": "linear_C_O_O_geometry",
        "score": score,
        "carbon_index": c_idx,
        "oxygen_indices": [o1, o2],
        "c_o_distances_A": [d1, d2],
        "o_c_o_angle_deg": angle,
    }


def frozen_host_hash(structure, guest_indices: list[int]) -> tuple[str, dict[str, Any]]:
    host = structure.copy()
    host.remove_sites(sorted(guest_indices, reverse=True))
    payload = {
        "lattice_matrix_A": [[round(float(value), 8) for value in row] for row in host.lattice.matrix],
        "species": [site.specie.symbol for site in host],
        "frac_coords": [[round(float(value), 8) for value in site.frac_coords] for site in host],
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return digest, {"structure": host, "payload": payload}


def guest_hash(structure, guest_indices: list[int]) -> str:
    payload = {
        "lattice_matrix_A": [[round(float(value), 8) for value in row] for row in structure.lattice.matrix],
        "species": [structure[i].specie.symbol for i in guest_indices],
        "cart_coords_A": [
            [round(float(value), 8) for value in structure[i].coords] for i in guest_indices
        ],
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def compare_host(candidate, frozen_host) -> tuple[bool, float | None, str | None]:
    from pymatgen.analysis.structure_matcher import StructureMatcher

    matcher = StructureMatcher(
        primitive_cell=False,
        scale=True,
        attempt_supercell=False,
        allow_subset=False,
    )
    try:
        matched = bool(matcher.fit(candidate, frozen_host))
        rms = None
        if matched:
            value = matcher.get_rms_dist(candidate, frozen_host)
            if value is not None:
                rms = float(value[0])
        return matched, rms, None
    except Exception as exc:  # pragma: no cover - version-dependent matcher failures
        return False, None, f"{type(exc).__name__}:{exc}"


def build_manifest(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = read_binding_rows(args.binding_csv)
    scalar_ids = {row["name"] for row in rows}
    complex_members = read_archive_members(
        args.complex_archive,
        lambda name: any(name.startswith(f"pristine_CO2/{mof_id}/") for mof_id in scalar_ids),
    )
    host_members = read_archive_members(
        args.host_archive,
        lambda name: any(host_id_match(name, mof_id) for mof_id in scalar_ids),
    )

    manifest: list[dict[str, Any]] = []
    counts = {
        "scalar_rows": len(rows),
        "complex_members": 0,
        "host_candidates": 0,
        "host_formula_match": 0,
        "structure_match_pass": 0,
        "strict_dft_triplets": 0,
        "strict_blocked_host_relaxation_or_linkage": 0,
        "errors": 0,
    }

    for row in rows:
        mof_id = row["name"]
        complex_member = choose_complex_member(complex_members, mof_id)
        host_member = choose_host_member(host_members, mof_id)
        record: dict[str, Any] = {
            "system_id": f"{mof_id}__co2_reference_candidate",
            "parent_mof": mof_id,
            "family_id": "UNRESOLVED",
            "family_id_status": "NOT_PROVIDED_BY_SOURCE_MANIFEST",
            "guest": "CO2",
            "n_guest": 1,
            "reference_method": "VASP PBE+D3 (ODAC23 dataset-level provenance)",
            "reference_method_row_level_status": "SCALAR_FILE_DOES_NOT_EMBED_METHOD",
            "functional": "PBE",
            "dispersion": "D3",
            "code": "VASP",
            "complex_structure_source": "Open-Catalyst-Project/odac-data pristine_CO2 archive",
            "host_structure_source": "ARC-MOF_20220610 archive candidate",
            "guest_structure_source": "CO2 fragment identified in official relaxed complex CIF",
            "complex_member": complex_member.name if complex_member else None,
            "host_member": host_member.name if host_member else None,
            "complex_structure_hash": complex_member.sha256 if complex_member else None,
            "host_structure_hash": host_member.sha256 if host_member else None,
            "complex_energy_eV": None,
            "host_energy_eV": None,
            "guest_energy_eV": None,
            "adsorption_energy_eV": float(row["E_BD_CO2(kJ/mol)"]) / KJ_MOL_PER_EV,
            "adsorption_energy_kJ_mol": float(row["E_BD_CO2(kJ/mol)"]),
            "complex_energy_au": float(row["E_MOF+CO2_BSSE(a.u.)"]),
            "guest_energy_au": float(row["E_CO2(a.u.)"]),
            "host_energy_au": float(row["E_MOF_MIN(a.u.)"]),
            "cell": "from complex CIF when available",
            "charge": "UNSPECIFIED",
            "spin": "UNSPECIFIED",
            "relaxation_status": "complex_relaxed_CIF; standalone_host_relaxation_not_linked",
            "complex_to_scalar_linkage": "UNVERIFIED_CONFIGURATION_ID",
            "strict_reference_status": "INCOMPLETE",
            "frozen_geometry_status": "NOT_EVALUATED",
            "guest_detection": None,
            "host_match_rms": None,
            "host_match_error": None,
        }
        try:
            if complex_member is None:
                counts["errors"] += 1
                record["strict_reference_status"] = "MISSING_COMPLEX_STRUCTURE"
                manifest.append(record)
                continue
            complex_structure = structure_from_cif(complex_member.data)
            guest_indices, detection = find_co2_indices(complex_structure)
            frozen_hash, frozen_info = frozen_host_hash(complex_structure, guest_indices)
            record["guest_detection"] = detection
            record["guest_structure_hash"] = guest_hash(complex_structure, guest_indices)
            record["frozen_host_structure_hash"] = frozen_hash
            record["frozen_host_atom_count"] = len(frozen_info["structure"])
            counts["complex_members"] += 1
            if host_member is None:
                record["strict_reference_status"] = "MISSING_HOST_STRUCTURE_CANDIDATE"
                record["frozen_geometry_status"] = "COMPLEX_GUEST_FRAGMENT_IDENTIFIED"
                manifest.append(record)
                continue
            counts["host_candidates"] += 1
            host_structure = structure_from_cif(host_member.data)
            formula_match = host_structure.composition.alphabetical_formula == frozen_info["structure"].composition.alphabetical_formula
            record["host_formula_match"] = bool(formula_match)
            if formula_match:
                counts["host_formula_match"] += 1
            matched, rms, error = compare_host(host_structure, frozen_info["structure"])
            record["host_structure_match"] = matched
            record["host_match_rms"] = rms
            record["host_match_error"] = error
            if matched:
                counts["structure_match_pass"] += 1
                record["frozen_geometry_status"] = "ARC_HOST_MATCHES_COMPLEX_HOST"
            else:
                record["frozen_geometry_status"] = "ARC_HOST_DOES_NOT_MATCH_COMPLEX_HOST"
            # The host CIF is not proven to be the separately relaxed ODAC reference,
            # and the scalar row is not linked to a particular complex member.
            record["strict_reference_status"] = "CANDIDATE_NEEDS_RELAXED_HOST_AND_COMPLEX_LINKAGE"
            counts["strict_blocked_host_relaxation_or_linkage"] += 1
        except Exception as exc:
            counts["errors"] += 1
            record["strict_reference_status"] = f"ERROR:{type(exc).__name__}"
            record["error"] = str(exc)
        manifest.append(record)

    summary = {
        "manifest_id": "odac23-co2-adsorption-reference-manifest-20261017",
        "status": "AUDITED_CANDIDATE_MANIFEST_STRICT_BENCHMARK_BLOCKED",
        "source": {
            "binding_csv": str(args.binding_csv),
            "complex_archive": str(args.complex_archive),
            "host_archive": str(args.host_archive),
            "complex_selection": "first non-random relaxed CIF, preferring *_1_relaxed.cif",
            "host_selection": "first clean/repeat ARC-MOF CIF candidate by exact DB12-ID prefix",
        },
        "reference_definition": "E_MOF+CO2 - E_MOF_MIN - E_CO2",
        "strict_gate": {
            "requires_same_protocol_host_guest_complex": True,
            "requires_separately_relaxed_host_identity": True,
            "requires_complex_row_linkage": True,
            "requires_charge_spin_units_metadata": True,
        },
        "counts": counts,
        "interpretation": "Rows provide ODAC23 scalar DFT provenance and structure candidates. They are not promoted to a strict paired DFT benchmark until the standalone relaxed-host identity and scalar-to-complex configuration linkage are verified.",
    }
    return manifest, summary


def write_outputs(manifest: list[dict[str, Any]], summary: dict[str, Any], args: argparse.Namespace) -> None:
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in manifest for key in row})
    with args.output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(manifest)
    payload = dict(summary)
    payload["records"] = manifest
    args.output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binding-csv", type=Path, required=True)
    parser.add_argument("--complex-archive", type=Path, required=True)
    parser.add_argument("--host-archive", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()
    manifest, summary = build_manifest(args)
    write_outputs(manifest, summary, args)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Compare MACE model-derived CO2 adsorption energies with ODAC scalar labels.

The source extxyz frames already carry atom tags: tag 0 denotes the framework
and positive tags denote adsorbates.  This script uses pymatgen's ASE adaptor to
perform the periodic crystal edit, preserving the parent cell and PBC for both
the host-only and guest-only structures.  It intentionally matches only exact
MOF IDs found in the scalar reference table; no Materials Project analogue is
substituted for an unmatched structure.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import tarfile
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from ase import Atoms
from ase.io import iread
from mace.calculators import MACECalculator
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.io.cif import CifParser
from pymatgen.analysis.structure_matcher import StructureMatcher
from scipy.stats import pearsonr, spearmanr


KJ_MOL_PER_EV = 96.48533212331002


def as_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def load_reference(path: Path) -> dict[str, dict[str, float]]:
    rows: dict[str, dict[str, float]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            name = row["name"].strip()
            rows[name] = {
                "E_MOF_CO2_BSSE_au": float(row["E_MOF+CO2_BSSE(a.u.)"].replace(",", "")),
                "E_CO2_au": float(row["E_CO2(a.u.)"].replace(",", "")),
                "E_MOF_MIN_au": float(row["E_MOF_MIN(a.u.)"].replace(",", "")),
                "E_BD_CO2_kJ_mol": float(row["E_BD_CO2(kJ/mol)"].replace(",", "")),
            }
    return rows


def match_reference(frame_name: str, references: dict[str, dict[str, float]]) -> str | None:
    # The scalar table contains pristine MOF IDs.  Require the pristine frame
    # naming convention ``MOF_ID_w_...`` so a defective ``MOF_ID_0.08_0_w_...``
    # frame is never silently assigned the pristine reference.
    for name in sorted(references, key=len, reverse=True):
        if frame_name.startswith(name + "_w_"):
            return name
    return None


def split_with_pymatgen(atoms: Atoms) -> tuple[Atoms, Atoms]:
    tags = np.asarray(atoms.get_tags(), dtype=int)
    if tags.shape[0] != len(atoms) or not np.any(tags > 0) or not np.any(tags == 0):
        raise ValueError("frame does not contain both tag-0 host and positive-tag guest atoms")

    structure = AseAtomsAdaptor.get_structure(atoms)
    host_structure = structure.copy()
    guest_structure = structure.copy()
    host_structure.remove_sites([int(i) for i in np.flatnonzero(tags > 0)])
    guest_structure.remove_sites([int(i) for i in np.flatnonzero(tags == 0)])
    host = AseAtomsAdaptor.get_atoms(host_structure)
    guest = AseAtomsAdaptor.get_atoms(guest_structure)
    return host, guest


def load_host_cifs(archive: Path, reference_ids: set[str]) -> dict[str, Atoms]:
    """Load exact ARC-MOF host-only CIFs when their filenames are unambiguous."""
    hosts: dict[str, Atoms] = {}
    with tarfile.open(archive, "r:gz") as tar:
        for member in tar.getmembers():
            if not member.name.lower().endswith(".cif"):
                continue
            name = Path(member.name).name
            for ref_id in sorted(reference_ids, key=len, reverse=True):
                if name.lower() not in {
                    f"db12-{ref_id.lower()}_clean_repeat.cif",
                    f"db12-{ref_id.lower()}_freeonly_repeat.cif",
                    f"db12-{ref_id.lower()}_repeat.cif",
                    f"{ref_id.lower()}_clean_repeat.cif",
                    f"{ref_id.lower()}_freeonly_repeat.cif",
                    f"{ref_id.lower()}_repeat.cif",
                }:
                    continue
                raw = tar.extractfile(member).read()
                structures = CifParser.from_str(raw.decode(errors="replace"), occupancy_tolerance=1.0).parse_structures(primitive=False)
                if len(structures) != 1:
                    raise ValueError(f"expected one structure in {member.name}")
                hosts[ref_id] = AseAtomsAdaptor.get_atoms(structures[0])
                break
    return hosts


def energy(calc: MACECalculator, atoms: Atoms) -> float:
    atoms = atoms.copy()
    atoms.calc = calc
    value = atoms.get_potential_energy()
    if not math.isfinite(float(value)):
        raise ValueError("non-finite model energy")
    return float(value)


def structure_key(atoms: Atoms) -> str:
    """Return a geometry/elements/cell key for safe host/guest energy caching."""
    digest = hashlib.sha256()
    digest.update(np.asarray(atoms.get_atomic_numbers(), dtype=np.int16).tobytes())
    digest.update(np.asarray(atoms.get_positions(), dtype=np.float64).tobytes())
    digest.update(np.asarray(atoms.get_cell().array, dtype=np.float64).tobytes())
    digest.update(np.asarray(atoms.get_pbc(), dtype=np.uint8).tobytes())
    return digest.hexdigest()


def metrics(pred: np.ndarray, ref: np.ndarray) -> dict[str, float | int | None]:
    err = pred - ref
    out: dict[str, float | int | None] = {
        "count": int(len(pred)),
        "mae_eV": float(np.mean(np.abs(err))) if len(pred) else None,
        "rmse_eV": float(np.sqrt(np.mean(err**2))) if len(pred) else None,
        "bias_eV": float(np.mean(err)) if len(pred) else None,
        "pearson_r": None,
        "spearman_r": None,
    }
    if len(pred) >= 2 and np.std(pred) > 0 and np.std(ref) > 0:
        out["pearson_r"] = float(pearsonr(ref, pred).statistic)
        out["spearman_r"] = float(spearmanr(ref, pred).statistic)
    return out


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "source_file",
        "frame_index",
        "sid",
        "fid",
        "frame_name",
        "reference_mof_id",
        "reference_eads_eV",
        "reference_eads_kJ_mol",
        "scratch_eads_eV",
        "foundation_eads_eV",
        "natoms_host",
        "natoms_guest",
        "host_geometry_source",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def iter_source_frames(
    source_root: Path | None,
    source_tar: Path | None,
    excluded_source_names: set[str],
):
    """Yield ``(source_name, frame_index, atoms)`` without extracting a tar.

    The full validation archive is several gigabytes compressed.  Streaming the
    extxyz members keeps the comparison reproducible while avoiding a second
    multi-gigabyte extracted copy on the remote host.
    """
    if source_root is not None:
        for source_file in sorted(source_root.rglob("*.extxyz")):
            if source_file.name in excluded_source_names or str(source_file) in excluded_source_names:
                continue
            for frame_index, atoms in enumerate(iread(source_file, index=":")):
                yield str(source_file), frame_index, atoms
        return

    assert source_tar is not None
    with tarfile.open(source_tar, "r:gz") as tar:
        for member in tar:
            if not member.isfile() or not member.name.lower().endswith(".extxyz"):
                continue
            member_basename = Path(member.name).name
            if member.name in excluded_source_names or member_basename in excluded_source_names:
                continue
            extracted = tar.extractfile(member)
            if extracted is None:
                continue
            raw = extracted.read()
            text = raw.decode(errors="replace")
            for frame_index, atoms in enumerate(iread(io.StringIO(text), index=":", format="extxyz")):
                yield member.name, frame_index, atoms


def main() -> int:
    parser = argparse.ArgumentParser()
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--source-root", type=Path)
    source_group.add_argument("--source-tar", type=Path)
    parser.add_argument("--exclude-source-name", nargs="*", default=[])
    parser.add_argument("--reference-csv", type=Path, required=True)
    parser.add_argument("--scratch-model", type=Path, required=True)
    parser.add_argument("--foundation-model", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--host-cif-archive", type=Path, default=None)
    parser.add_argument("--host-cif-ids", nargs="*", default=None)
    parser.add_argument("--max-evaluated-frames", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    references = load_reference(args.reference_csv)
    scratch = MACECalculator(model_paths=[str(args.scratch_model)], device=args.device, default_dtype="float32")
    foundation = MACECalculator(model_paths=[str(args.foundation_model)], device=args.device, default_dtype="float32")
    host_ids = set(args.host_cif_ids) if args.host_cif_ids else set(references)
    host_cifs = load_host_cifs(args.host_cif_archive, host_ids) if args.host_cif_archive else {}
    host_cif_fit: dict[str, bool] = {}
    scratch_host_energy_cache: dict[str, float] = {}
    foundation_host_energy_cache: dict[str, float] = {}
    scratch_guest_energy_cache: dict[str, float] = {}
    foundation_guest_energy_cache: dict[str, float] = {}
    matcher = StructureMatcher(primitive_cell=False, scale=True, attempt_supercell=False, allow_subset=False)

    rows: list[dict[str, Any]] = []
    excluded = defaultdict(int)
    excluded_source_names = set(args.exclude_source_name)
    source_file_names: set[str] = set()
    for source_name, frame_index, atoms in iter_source_frames(args.source_root, args.source_tar, excluded_source_names):
        if args.max_evaluated_frames is not None and len(rows) >= args.max_evaluated_frames:
            break
        source_file_names.add(source_name)
        info = dict(atoms.info)
        frame_name = str(info.get("name", ""))
        ref_id = match_reference(frame_name, references)
        nco2 = as_int(info.get("nco2"))
        nh2o = as_int(info.get("nh2o"))
        if ref_id is None:
            excluded["no_exact_scalar_reference_id"] += 1
            continue
        if nco2 != 1 or nh2o != 0:
            excluded["not_single_co2_only_frame"] += 1
            continue
        try:
            host_from_frame, guest = split_with_pymatgen(atoms)
            if ref_id in host_cifs and ref_id not in host_cif_fit:
                frame_structure = AseAtomsAdaptor.get_structure(host_from_frame)
                cif_structure = AseAtomsAdaptor.get_structure(host_cifs[ref_id])
                host_cif_fit[ref_id] = bool(matcher.fit(frame_structure, cif_structure))
            use_cif = bool(host_cif_fit.get(ref_id, False))
            host = host_cifs[ref_id] if use_cif else host_from_frame
            host_source = "ARC-MOF exact host-only CIF (StructureMatcher pass)" if use_cif else "pymatgen deletion from host+guest frame"
            complex_energy_scratch = energy(scratch, atoms)
            host_key = ref_id if use_cif else structure_key(host)
            guest_key = structure_key(guest)
            if host_key in scratch_host_energy_cache:
                host_energy_scratch = scratch_host_energy_cache[host_key]
            else:
                host_energy_scratch = energy(scratch, host)
                scratch_host_energy_cache[host_key] = host_energy_scratch
            if guest_key in scratch_guest_energy_cache:
                guest_energy_scratch = scratch_guest_energy_cache[guest_key]
            else:
                guest_energy_scratch = energy(scratch, guest)
                scratch_guest_energy_cache[guest_key] = guest_energy_scratch
            complex_energy_foundation = energy(foundation, atoms)
            if host_key in foundation_host_energy_cache:
                host_energy_foundation = foundation_host_energy_cache[host_key]
            else:
                host_energy_foundation = energy(foundation, host)
                foundation_host_energy_cache[host_key] = host_energy_foundation
            if guest_key in foundation_guest_energy_cache:
                guest_energy_foundation = foundation_guest_energy_cache[guest_key]
            else:
                guest_energy_foundation = energy(foundation, guest)
                foundation_guest_energy_cache[guest_key] = guest_energy_foundation
        except Exception as exc:  # keep the audit fail-visible but continue other frames
            excluded[f"decomposition_error:{type(exc).__name__}"] += 1
            continue
        ref = references[ref_id]
        rows.append(
            {
                "source_file": source_name,
                "frame_index": frame_index,
                "sid": str(info.get("sid", "")),
                "fid": str(info.get("fid", "")),
                "frame_name": frame_name,
                "reference_mof_id": ref_id,
                "reference_eads_eV": ref["E_BD_CO2_kJ_mol"] / KJ_MOL_PER_EV,
                "reference_eads_kJ_mol": ref["E_BD_CO2_kJ_mol"],
                "scratch_eads_eV": complex_energy_scratch - host_energy_scratch - guest_energy_scratch,
                "foundation_eads_eV": complex_energy_foundation - host_energy_foundation - guest_energy_foundation,
                "natoms_host": len(host),
                "natoms_guest": len(guest),
                "host_geometry_source": host_source,
            }
        )

    if not rows:
        raise RuntimeError("no matched CO2-only frames were evaluated")

    frame_ref = np.asarray([row["reference_eads_eV"] for row in rows], dtype=float)
    scratch_pred = np.asarray([row["scratch_eads_eV"] for row in rows], dtype=float)
    foundation_pred = np.asarray([row["foundation_eads_eV"] for row in rows], dtype=float)

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["reference_mof_id"]].append(row)
    per_mof: list[dict[str, Any]] = []
    for mof_id, group in sorted(grouped.items()):
        ref = float(group[0]["reference_eads_eV"])
        scratch_values = np.asarray([row["scratch_eads_eV"] for row in group], dtype=float)
        foundation_values = np.asarray([row["foundation_eads_eV"] for row in group], dtype=float)
        per_mof.append(
            {
                "mof_id": mof_id,
                "frame_count": len(group),
                "reference_eads_eV": ref,
                "reference_eads_kJ_mol": float(group[0]["reference_eads_kJ_mol"]),
                "scratch_mean_eV": float(np.mean(scratch_values)),
                "scratch_std_eV": float(np.std(scratch_values)),
                "foundation_mean_eV": float(np.mean(foundation_values)),
                "foundation_std_eV": float(np.std(foundation_values)),
                "scratch_mae_eV": float(np.mean(np.abs(scratch_values - ref))),
                "foundation_mae_eV": float(np.mean(np.abs(foundation_values - ref))),
            }
        )

    mof_ref = np.asarray([entry["reference_eads_eV"] for entry in per_mof], dtype=float)
    mof_scratch = np.asarray([entry["scratch_mean_eV"] for entry in per_mof], dtype=float)
    mof_foundation = np.asarray([entry["foundation_mean_eV"] for entry in per_mof], dtype=float)

    result = {
        "experiment_id": "mof-mace-adsorption-reference-compare-20261016",
        "status": "EXECUTED_EXACT_ID_MATCHES_ONLY",
        "source": {
            "source_root": str(args.source_root) if args.source_root else None,
            "source_tar": str(args.source_tar) if args.source_tar else None,
            "reference_csv": str(args.reference_csv),
            "source_frame_filter": "nco2=1, nh2o=0, exact pristine MOF_ID_w_ frame naming",
            "host_guest_edit": "Pymatgen Structure site deletion via AseAtomsAdaptor for guest isolation; exact ARC-MOF host-only CIF used when available; original periodic cell/PBC retained",
            "host_cif_archive": str(args.host_cif_archive) if args.host_cif_archive else None,
            "host_cif_ids_used": sorted(host_cifs),
            "host_cif_structure_match": host_cif_fit,
            "materials_project_policy": "No Materials Project analogue substituted; exact MP lookup was not available without a verified API key/structure identity.",
        },
        "counts": {
            "extxyz_files_seen": len(source_file_names),
            "evaluated_frames": len(rows),
            "matched_mof_ids": len(per_mof),
            "excluded": dict(sorted(excluded.items())),
        },
        "frame_metrics": {
            "scratch": metrics(scratch_pred, frame_ref),
            "foundation_naive_finetune": metrics(foundation_pred, frame_ref),
        },
        "mof_level_metrics": {
            "observation_unit": "one mean model-derived decomposition per exact matched MOF ID",
            "scratch": metrics(mof_scratch, mof_ref),
            "foundation_naive_finetune": metrics(mof_foundation, mof_ref),
        },
        "per_mof": per_mof,
        "reference_units": "CO2 binding-energy scalar from CO2_binding_energy.csv; converted kJ/mol to eV using 96.48533212331002 kJ/mol per eV",
        "interpretation": "The comparison is a model-derived energy decomposition on frames whose MOF ID exactly matches the scalar table. It is not an independent DFT recalculation and remains subject to geometry/protocol mismatch between the scalar reference and the extxyz frame.",
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_csv(args.output_csv, rows)
    print(json.dumps({"evaluated_frames": len(rows), "matched_mof_ids": len(per_mof), "excluded": dict(excluded)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

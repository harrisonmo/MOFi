"""Audit source tables without fitting models or silently repairing labels."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path


LABEL_COLUMNS = [
    "MOF Name",
    "Methane Adsorption Performance under 5.8 Bar (mmol/g)",
    "Methane Adsorption Performance under 65 Bar (mmol/g)",
]
GEOMETRY_COLUMNS = [
    "MOF Name", "UC_volume", "Density", "ASA", "SAv", "SAg", "AVA",
    "Vf", "Vg", "POAVA", "POVf", "POVg", "LCD", "PLD", "DFSP",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_table(path: Path, expected: list[str]) -> tuple[dict, dict]:
    """Retain IDs, count bad rows, and allow only entirely empty trailing fields."""
    ids: Counter = Counter()
    values = {}
    issues = Counter()
    examples = {}
    blank_rows = 0
    rows = 0
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream)
        header = next(reader)
        if header[:len(expected)] != expected or any(header[len(expected):]):
            raise ValueError(f"Unexpected columns in {path.name}: {header}")
        for line, row in enumerate(reader, 2):
            if not row or not any(field.strip() for field in row):
                blank_rows += 1
                continue
            rows += 1
            code = None
            if row[0].strip() and row[0] == row[0].strip():
                ids[row[0]] += 1
            if len(row) < len(expected) or any(field.strip() for field in row[len(expected):]):
                code = "invalid_row_shape"
            elif not row[0].strip() or row[0] != row[0].strip():
                code = "invalid_identity"
            else:
                identity = row[0]
                try:
                    numbers = tuple(float(x) for x in row[1:len(expected)])
                except ValueError:
                    code = "nonnumeric_or_missing"
                else:
                    if not all(math.isfinite(x) for x in numbers):
                        code = "nonfinite"
                    elif identity not in values:
                        values[identity] = numbers
            if code:
                issues[code] += 1
                examples.setdefault(code, [])
                if len(examples[code]) < 10:
                    examples[code].append({"csv_line": line, "id": row[0] if row else None})
    duplicates = {k: v for k, v in ids.items() if v > 1}
    for identity in duplicates:
        values.pop(identity, None)
    return {
        "file": path.name, "bytes": path.stat().st_size, "sha256": sha256(path),
        "rows": rows, "unique_ids": len(ids), "blank_rows": blank_rows,
        "named_columns": expected, "empty_trailing_header_columns": len(header) - len(expected),
        "issues": dict(issues), "issue_examples": examples,
        "duplicate_ids": duplicates, "usable_numeric_unique_rows": len(values),
    }, values


def audit(raw: Path, inventory: Path) -> dict:
    entries = json.loads(inventory.read_text(encoding="utf-8"))
    verified = []
    for entry in entries:
        path = raw / entry["filename"]
        if not path.exists():
            continue
        detail = entry["content_details"]
        digest = sha256(path)
        if digest != detail["sha256_hash"] or path.stat().st_size != detail["size"]:
            raise ValueError(f"Source checksum mismatch: {path.name}")
        verified.append({"filename": path.name, "bytes": path.stat().st_size,
                         "sha256": digest, "source_file_id": entry["id"]})
    required = {"Adsorption Labels Data.csv", "Geometry Descriptors Data.csv"}
    if not required.issubset({x["filename"] for x in verified}):
        raise ValueError("Required tables are missing from the verified inventory")
    labels, y = read_table(raw / "Adsorption Labels Data.csv", LABEL_COLUMNS)
    geometry, x = read_table(raw / "Geometry Descriptors Data.csv", GEOMETRY_COLUMNS)
    negative_uptake = sorted(k for k, v in y.items() if min(v) < 0)
    negative_delta = sorted(k for k, v in y.items() if v[1] < v[0])
    paired = set(x) & set(y)
    quarantined = set(negative_uptake) | set(negative_delta)
    return {
        "schema": "g1-source-audit-v1",
        "source": "https://data.mendeley.com/datasets/2bp6gcsg46/3",
        "inventory_sha256": sha256(inventory), "verified_files": verified,
        "labels": labels, "geometry": geometry,
        "pairing": {"numeric_unique_id_intersection": len(paired),
                    "labels_without_geometry": len(set(y) - set(x)),
                    "geometry_without_labels": len(set(x) - set(y)),
                    "provisional_nonnegative_pairs": len(paired - quarantined)},
        "label_anomalies": {"negative_uptake_ids": negative_uptake,
                            "negative_working_capacity_ids": negative_delta},
        "training_admitted": False,
        "pending": ["CIF-to-ID pairing and cross-source structural deduplication",
                    "Explicit absolute/excess label convention and full simulation settings",
                    "Descriptor definitions, units, and missing-value conventions",
                    "Frozen family split and independent training environment calibration"],
        "interpretation": "Anomalies are quarantined, not corrected. Counts do not establish structure uniqueness.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=Path("datasets/raw/g1-mendeley-v3"))
    parser.add_argument("--inventory", type=Path, default=Path("docs/reports/g1_sources/mendeley_files.json"))
    parser.add_argument("--output", type=Path, default=Path("docs/reports/G1_DATA_AUDIT.json"))
    args = parser.parse_args()
    result = audit(args.raw, args.inventory)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"pairing": result["pairing"], "anomalies": result["label_anomalies"],
                      "training_admitted": result["training_admitted"]}))


if __name__ == "__main__":
    main()

"""Create a bounded, label-blind CIF subset from the verified public archive."""
from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from collections import Counter
from pathlib import Path, PurePosixPath

import rarfile


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "datasets/raw/g1-mendeley-v3"
OUT = ROOT / "datasets/processed/g1"


def rank(identity, prefix):
    return hashlib.sha256((prefix + identity).encode()).hexdigest()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    archive = RAW / "CIF file V3.0.rar"
    expected = "78d17b1b25979e919192bd3133fee71ba84ff07391654f9bc62606ec6ca0b674"
    with archive.open("rb") as stream:
        assert hashlib.file_digest(stream, "sha256").hexdigest() == expected
    with (RAW / "Adsorption Labels Data.csv").open(newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    ids = {row["MOF Name"] for row in rows}
    anomalous = set(json.loads((ROOT / "docs/reports/G1_DATA_AUDIT.json").read_text())["label_anomalies"]["negative_working_capacity_ids"])
    inventory = []
    members = {}
    for entry in rarfile.RarFile(archive).infolist():
        name = entry.filename.replace("\\", "/")
        path = PurePosixPath(name)
        if path.is_absolute() or ".." in path.parts or ":" in name or entry.is_symlink():
            raise ValueError(f"Unsafe archive member: {name}")
        if entry.isdir():
            continue
        if not entry.is_file() or path.suffix.lower() != ".cif":
            raise ValueError(f"Unexpected archive member: {name}")
        if path.stem in members:
            raise ValueError(f"Ambiguous CIF identity: {path.stem}")
        members[path.stem] = {"path": name, "size": entry.file_size}
        inventory.append({"id": path.stem, **members[path.stem]})
    if sum(x["size"] for x in inventory) > 4 * 1024**3:
        raise ValueError("Archive expansion exceeds the recorded budget")
    paired = ids & members.keys()
    selected = sorted(paired - anomalous, key=lambda k: rank(k, "g1-main-v1:"))[:3000]
    pilot = sorted(paired - anomalous, key=lambda k: rank(k, "g1-pilot-v1:"))[:256]
    paths = [members[k]["path"] for k in sorted(set(selected) | set(pilot))]
    (OUT / "members.txt").write_text("\n".join(paths) + "\n", encoding="utf-8")
    cif_root = OUT / "cifs"
    cif_root.mkdir(exist_ok=True)
    subprocess.run(["tar", "-xf", str(archive), "-C", str(cif_root), "-T", str(OUT / "members.txt")],
                   check=True, timeout=900)
    for identity in set(selected) | set(pilot):
        path = cif_root / members[identity]["path"]
        if not path.resolve().is_relative_to(cif_root.resolve()) or path.stat().st_size != members[identity]["size"]:
            raise ValueError(f"Extracted file mismatch: {identity}")
    report = {"archive_sha256": expected, "archive_cif_count": len(members),
              "archive_uncompressed_bytes": sum(x["size"] for x in inventory),
              "label_ids": len(ids), "paired_ids": len(paired),
              "labels_without_cif": sorted(ids - members.keys()), "cifs_without_label": sorted(members.keys() - ids),
              "pilot_ids": pilot, "selected_ids": selected,
              "extracted_count": len(paths), "extracted_bytes": sum(members[k]["size"] for k in set(selected) | set(pilot)),
              "paths": {k: str((cif_root / members[k]["path"]).relative_to(ROOT)).replace("\\", "/")
                        for k in set(selected) | set(pilot)}}
    (OUT / "selection.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if not k.endswith("ids") and k != "paths"}))


if __name__ == "__main__":
    main()

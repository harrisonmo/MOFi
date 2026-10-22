"""Validate, deduplicate, and freeze a family-disjoint development dataset."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
import warnings
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from pymatgen.core import Structure
from pymatgen.analysis.structure_matcher import StructureMatcher, ElementComparator

from g1_features import grouped_split, linker_family, periodic_graph

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "datasets/processed/g1"


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def process(item):
    identity, relative = item
    path = ROOT / relative
    started = time.monotonic()
    try:
        with warnings.catch_warnings(record=True) as notices:
            warnings.simplefilter("always")
            structure = Structure.from_file(path, primitive=False)
            graph = periodic_graph(structure)
            exact_family, signature = linker_family(structure)
            family, scaffold_signature = linker_family(structure, scaffold=True)
        key = hashlib.sha256(identity.encode()).hexdigest()
        graph_path = OUT / "graphs" / f"{key}.npz"
        np.savez_compressed(graph_path, **graph)
        structure_path = OUT / "structures" / f"{key}.json"
        save(structure_path, structure.as_dict())
        return {"id": identity, "cif": relative, "cif_sha256": digest(path),
                "graph": str(graph_path.relative_to(ROOT)).replace("\\", "/"), "graph_sha256": digest(graph_path),
                "structure": str(structure_path.relative_to(ROOT)).replace("\\", "/"),
                "formula": structure.composition.reduced_formula, "atoms": len(structure),
                "edges": len(graph["src"]), "family": family, "linker_signature": signature,
                "exact_linker_family": exact_family, "scaffold_signature": scaffold_signature,
                "warnings": sorted({str(w.message) for w in notices}), "seconds": time.monotonic() - started}
    except Exception as error:
        return {"id": identity, "error": str(error), "error_type": type(error).__name__,
                "seconds": time.monotonic() - started}


class UnionFind:
    def __init__(self, keys):
        self.parent = {k: k for k in keys}

    def find(self, key):
        while self.parent[key] != key:
            self.parent[key] = self.parent[self.parent[key]]
            key = self.parent[key]
        return key

    def union(self, a, b):
        a, b = self.find(a), self.find(b)
        self.parent[max(a, b)] = min(a, b)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["pilot", "main"])
    args = parser.parse_args()
    started = time.monotonic()
    selection = json.loads((OUT / "selection.json").read_text())
    ids = selection["pilot_ids" if args.mode == "pilot" else "selected_ids"]
    for directory in ["graphs", "structures"]:
        (OUT / directory).mkdir(exist_ok=True)
    rows = []
    with ProcessPoolExecutor(max_workers=4) as executor:
        for row in executor.map(process, [(k, selection["paths"][k]) for k in ids], chunksize=4):
            rows.append(row)
            if len(rows) % 64 == 0:
                print(f"{args.mode}: parsed {len(rows)}/{len(ids)}", flush=True)
            if time.monotonic() - started > 5400:
                raise TimeoutError("Structure processing budget exceeded")
    save(OUT / f"{args.mode}-structure-audit.json", rows)
    valid = [x for x in rows if "error" not in x]
    summary = {"mode": args.mode, "requested": len(ids), "valid": len(valid),
               "failures": dict(Counter(x["error"] for x in rows if "error" in x)),
               "warning_count": sum(bool(x.get("warnings")) for x in rows),
               "families_before_dedup": len({x["family"] for x in valid})}
    if args.mode == "pilot":
        summary["seconds"] = time.monotonic() - started
        save(OUT / "pilot-summary.json", summary)
        print(json.dumps(summary), flush=True)
        return
    structures = {x["id"]: Structure.from_dict(json.loads((ROOT / x["structure"]).read_text())) for x in valid}
    matcher = StructureMatcher(ltol=0.2, stol=0.3, angle_tol=5, primitive_cell=True,
                               scale=True, attempt_supercell=True, comparator=ElementComparator())
    union = UnionFind(structures)
    buckets = defaultdict(list)
    for row in valid:
        buckets[row["formula"]].append(row)
    comparisons, matches = 0, []
    for bucket in buckets.values():
        for i, a in enumerate(bucket):
            for b in bucket[:i]:
                comparisons += 1
                match = a["cif_sha256"] == b["cif_sha256"] or matcher.fit(structures[a["id"]], structures[b["id"]])
                if match:
                    union.union(a["id"], b["id"])
                    matches.append([a["id"], b["id"]])
                if time.monotonic() - started > 5400:
                    raise TimeoutError("Structure deduplication budget exceeded")
        if comparisons and comparisons % 100 == 0:
            print(f"Dedup comparisons: {comparisons}, matches: {len(matches)}", flush=True)
    groups = defaultdict(list)
    for row in valid:
        groups[union.find(row["id"])].append(row["id"])
    keep = {min(group, key=lambda k: hashlib.sha256(("g1-main-v1:" + k).encode()).hexdigest()) for group in groups.values()}
    retained = [x for x in valid if x["id"] in keep]
    splits = grouped_split(retained)
    family_sets = {name: {x["family"] for x in retained if x["id"] in values} for name, values in splits.items()}
    assert not (family_sets["train"] & family_sets["validation"] or family_sets["train"] & family_sets["heldout"] or family_sets["validation"] & family_sets["heldout"])
    raw = ROOT / "datasets/raw/g1-mendeley-v3"
    with (raw / "Adsorption Labels Data.csv").open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.reader(stream)
        next(reader)
        labels = {row[0]: float(row[2]) - float(row[1]) for row in reader if row}
    with (raw / "Geometry Descriptors Data.csv").open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.reader(stream)
        header = next(reader)[:15]
        features = {row[0]: [float(v) for v in row[1:15]] for row in reader if row}
    largest = max(Counter(x["family"] for x in retained).values(), default=0)
    admitted = len(retained) >= 2000 and len(splits["train"]) >= 1000 and len(splits["validation"]) >= 200 and largest <= len(retained) * 0.5
    for name, values in splits.items():
        split_rows = []
        for row in retained:
            if row["id"] in values:
                split_rows.append({**row, "geometry": features[row["id"]], "target": labels[row["id"]]})
        save(OUT / f"{name}.json", split_rows)
    manifest = {"schema": "g1-frozen-dataset-v1", "selection_sha256": digest(OUT / "selection.json"),
                "records": [{k: row[k] for k in ["id", "cif_sha256", "graph_sha256", "family", "atoms", "edges"]} for row in retained],
                "split_ids": splits, "split_sha256": {name: digest(OUT / f"{name}.json") for name in splits},
                "geometry_columns": header[1:], "training_admitted": admitted}
    save(OUT / "manifest.json", manifest)
    summary.update({"comparisons": comparisons, "matched_pairs": matches,
                    "duplicate_components": [v for v in groups.values() if len(v) > 1],
                    "duplicate_label_spreads": [{"ids": v, "target_spread": max(labels[k] for k in v) - min(labels[k] for k in v)} for v in groups.values() if len(v) > 1],
                    "retained": len(retained), "families": len({x["family"] for x in retained}),
                    "largest_family": largest, "split_counts": {k: len(v) for k, v in splits.items()},
                    "split_family_counts": {k: len(v) for k, v in family_sets.items()},
                    "training_admitted": admitted, "seconds": time.monotonic() - started,
                    "manifest_sha256": digest(OUT / "manifest.json")})
    save(OUT / "structure-summary.json", summary)
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()

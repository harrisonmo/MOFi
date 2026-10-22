"""Rebuild 8 A graphs for the frozen G1 structures without reading any labels."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from pymatgen.core import Structure

from g3_features import CUTOFF, MAX_EDGES, periodic_graph

ROOT = Path(__file__).resolve().parents[1]
G1 = ROOT / "datasets/processed/g1"
OUT = ROOT / "datasets/processed/g3"


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def process(identity):
    key = hashlib.sha256(identity.encode()).hexdigest()
    started = time.monotonic()
    try:
        structure = Structure.from_dict(json.loads((G1 / "structures" / f"{key}.json").read_text()))
        graph = periodic_graph(structure)
        path = OUT / "graphs" / f"{key}.npz"
        np.savez_compressed(path, **graph)
        return {"id": identity, "graph": str(path.relative_to(ROOT)).replace("\\", "/"),
                "graph_sha256": digest(path), "atoms": len(graph["z"]), "edges": len(graph["src"]),
                "seconds": time.monotonic() - started}
    except Exception as error:
        return {"id": identity, "error": str(error), "error_type": type(error).__name__,
                "seconds": time.monotonic() - started}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit-seconds", type=float, default=5400)
    args = parser.parse_args()
    if (OUT / "manifest.json").exists():
        raise FileExistsError("Refusing to overwrite frozen G3 graphs")
    started = time.monotonic()
    manifest = json.loads((G1 / "manifest.json").read_text())
    ids = [record["id"] for record in manifest["records"]]
    (OUT / "graphs").mkdir(parents=True, exist_ok=True)
    rows = []
    with ProcessPoolExecutor(max_workers=4) as executor:
        for row in executor.map(process, ids, chunksize=4):
            rows.append(row)
            if len(rows) % 256 == 0:
                print(f"built {len(rows)}/{len(ids)}", flush=True)
            if time.monotonic() - started > args.limit_seconds:
                raise TimeoutError("G3 graph construction budget exceeded")
    valid = [row for row in rows if "error" not in row]
    edges = np.array([row["edges"] for row in valid])
    save(OUT / "manifest.json", {
        "schema": "g3-graphs-v1", "g1_manifest_sha256": digest(G1 / "manifest.json"),
        "cutoff": CUTOFF, "max_edges": MAX_EDGES, "records": valid,
        "failures": [row for row in rows if "error" in row],
        "failure_counts": dict(Counter(row["error"] for row in rows if "error" in row)),
        "edge_quantiles": dict(zip(["min", "p50", "p90", "max"], np.quantile(edges, [0, .5, .9, 1]).tolist())) if len(edges) else {},
        "seconds": time.monotonic() - started,
    })
    print(json.dumps({"valid": len(valid), "failed": len(rows) - len(valid)}), flush=True)


if __name__ == "__main__":
    main()

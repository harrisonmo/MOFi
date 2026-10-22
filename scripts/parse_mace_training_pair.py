#!/usr/bin/env python3
"""Extract auditable epoch metrics and artifact status from two MACE logs."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path


METRIC_RE = re.compile(
    r"(?P<label>Initial|Epoch\s+(?P<epoch>\d+)).*?"
    r"RMSE_E_per_atom=\s*(?P<energy>[0-9.]+)\s*meV.*?"
    r"RMSE_F=\s*(?P<force>[0-9.]+)\s*meV"
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def parse_run(label: str, run_dir: Path) -> dict[str, object]:
    log = run_dir / "train.log"
    text = log.read_text(encoding="utf-8", errors="replace")
    metrics: list[dict[str, float | int | str]] = []
    for match in METRIC_RE.finditer(text):
        marker = match.group("label")
        metrics.append(
            {
                "label": marker,
                "epoch": -1 if marker == "Initial" else int(match.group("epoch")),
                "energy_rmse_meV_per_atom": float(match.group("energy")),
                "force_rmse_meV_per_A": float(match.group("force")),
            }
        )
    model_files = sorted((run_dir / "checkpoints").glob("*.model"))
    checkpoint_files = sorted((run_dir / "checkpoints").glob("*.pt"))
    result: dict[str, object] = {
        "label": label,
        "run_dir": str(run_dir),
        "log_sha256": sha256(log),
        "training_complete_marker": "Training complete" in text,
        "log_runtime_error": "Traceback (most recent call last)" in text,
        "model_save_path_error": "Parent directory" in text and "model does not exist" in text,
        "metrics": metrics,
        "model_artifacts": [{"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)} for path in model_files],
        "checkpoint_count": len(checkpoint_files),
    }
    if metrics:
        result["final_validation"] = metrics[-1]
        result["best_combined_validation"] = min(
            metrics,
            key=lambda row: float(row["energy_rmse_meV_per_atom"]) + float(row["force_rmse_meV_per_A"]),
        )
        result["best_energy_validation"] = min(metrics, key=lambda row: float(row["energy_rmse_meV_per_atom"]))
        result["best_force_validation"] = min(metrics, key=lambda row: float(row["force_rmse_meV_per_A"]))
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scratch-dir", type=Path, required=True)
    parser.add_argument("--foundation-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "BOUNDED_5_EPOCH_PAIR_ARTIFACT_AUDIT",
        "runs": [
            parse_run("scratch", args.scratch_dir),
            parse_run("foundation_naive", args.foundation_dir),
        ],
        "interpretation": "S2EF validation metrics only; no strict DFT adsorption-energy claim is supported by this artifact.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

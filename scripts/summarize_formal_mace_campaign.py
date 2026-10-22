#!/usr/bin/env python3
"""Summarize auditable MACE screening logs without inventing missing metrics."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev


METRIC_RE = re.compile(
    r"(?P<label>Initial|Epoch\s+(?P<epoch>\d+)).*?"
    r"RMSE_E_per_atom=\s*(?P<energy>[0-9.]+)\s*meV.*?"
    r"RMSE_F=\s*(?P<force>[0-9.]+)\s*meV"
)


def parse_log(path: Path) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = METRIC_RE.search(line)
        if not match:
            continue
        label = match.group("label")
        rows.append(
            {
                "label": label,
                "epoch": -1 if label == "Initial" else int(match.group("epoch")),
                "energy_rmse_meV_per_atom": float(match.group("energy")),
                "force_rmse_meV_per_A": float(match.group("force")),
            }
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    args = parser.parse_args()

    run_rows: list[dict[str, object]] = []
    for run_dir in sorted(p for p in args.campaign_dir.iterdir() if p.is_dir()):
        config_path = run_dir / "config.json"
        log_path = run_dir / "train.log"
        if not config_path.exists() or not log_path.exists():
            continue
        config = json.loads(config_path.read_text(encoding="utf-8"))
        metrics = parse_log(log_path)
        if metrics:
            best = min(metrics, key=lambda row: (float(row["energy_rmse_meV_per_atom"]) + float(row["force_rmse_meV_per_A"])))
            last = metrics[-1]
        else:
            best = last = None
        result_path = run_dir / "result.json"
        result = json.loads(result_path.read_text(encoding="utf-8")) if result_path.exists() else {}
        run_rows.append(
            {
                "run_id": config["run_id"],
                **config["spec"],
                "status": result.get("exit_code", "RUNNING"),
                "metric_count": len(metrics),
                "best_metric": best,
                "last_metric": last,
                "train_log": str(log_path),
            }
        )

    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in run_rows:
        grouped[str(row["method"])].append(row)
    method_summary: dict[str, object] = {}
    for method, rows in sorted(grouped.items()):
        best_energy = [float(row["best_metric"]["energy_rmse_meV_per_atom"]) for row in rows if row["best_metric"]]
        best_force = [float(row["best_metric"]["force_rmse_meV_per_A"]) for row in rows if row["best_metric"]]
        method_summary[method] = {
            "run_count": len(rows),
            "completed_runs": sum(row["status"] == 0 for row in rows),
            "best_energy_rmse_meV_per_atom_mean": mean(best_energy) if best_energy else None,
            "best_energy_rmse_meV_per_atom_stdev": stdev(best_energy) if len(best_energy) >= 2 else None,
            "best_force_rmse_meV_per_A_mean": mean(best_force) if best_force else None,
            "best_force_rmse_meV_per_A_stdev": stdev(best_force) if len(best_force) >= 2 else None,
        }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "status": "COMPLETE" if run_rows and all(row["status"] == 0 for row in run_rows) else "PARTIAL_OR_RUNNING",
        "campaign_dir": str(args.campaign_dir),
        "runs_observed": len(run_rows),
        "method_summary": method_summary,
        "runs": run_rows,
        "metric_definition": "best logged validation RMSE_E_per_atom + RMSE_F; this is a screening summary, not strict adsorption-energy validation",
        "strict_adsorption_reference": "not inferred by this parser; read the paired reference manifest gate separately",
    }
    args.output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with args.output_csv.open("w", newline="", encoding="utf-8") as handle:
        fields = ["run_id", "method", "seed", "r_max", "status", "metric_count"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in run_rows:
            writer.writerow({field: row.get(field) for field in fields})
    print(json.dumps({"runs_observed": len(run_rows), "status": payload["status"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

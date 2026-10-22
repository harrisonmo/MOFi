"""Render an already-completed comparison; never train models or invent scores."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


LABELS = {
    "supervised_gnn": "Ordinary supervised GNN",
    "feat_gnn_jarzynski": "FEAT-GNN + Jarzynski",
    "feat_gnn_mean_work": "Same GNN + mean work (ablation)",
    "no_transport_jarzynski": "No transport + exact score + Jarzynski",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = json.loads(args.results.read_text(encoding="utf-8"))
    if result.get("status") != "completed":
        raise ValueError("Only completed formal results can be reported")
    if set(result.get("aggregate", {})) != {"test", "ood"}:
        raise ValueError("Both registered evaluation splits are required")
    config = result["config"]
    seeds = result["seeds"]
    if [row["seed"] for row in seeds] != config["seeds"]:
        raise ValueError("Incomplete or reordered seed results")
    # Recompute MAE/RMSE and counts from graph-level evidence, independently of
    # the runner's metric helper. Stop on a missing arm, graph, or nonfinite value.
    for seed in seeds:
        for split, count_key in (("test", "test_graphs"), ("ood", "ood_graphs")):
            rows = [r for r in result["predictions"] if r["seed"] == seed["seed"] and r["split"] == split]
            if sorted(r["graph_id"] for r in rows) != list(range(config[count_key])):
                raise ValueError("Missing or duplicate graph evidence")
            for arm in LABELS:
                errors = [r["predictions"][arm] - r["truth"] for r in rows]
                if not all(math.isfinite(e) for e in errors):
                    raise ValueError("Nonfinite prediction evidence")
                observed = seed["metrics"][split][arm]
                expected = {"mae": sum(abs(e) for e in errors) / len(errors),
                            "rmse": math.sqrt(sum(e * e for e in errors) / len(errors))}
                if any(not math.isclose(observed[k], value, rel_tol=1e-10, abs_tol=1e-12)
                       for k, value in expected.items()):
                    raise ValueError("Stored metrics disagree with graph predictions")
    args.output.mkdir(parents=True, exist_ok=False)
    lines = ["# Qiankun FEAT/Jarzynski comparison", "", config["scope"], "",
             "**Comparability:** " + config["comparability"], "",
             "Errors are in kBT units; three-seed SD is not a confidence interval.", "",
             "| Split | Model / estimator | MAE mean ± seed SD | RMSE mean |",
             "|---|---|---:|---:|"]
    aggregates = {}
    for split in ("test", "ood"):
        aggregates[split] = {}
        for arm, label in LABELS.items():
            maes = [s["metrics"][split][arm]["mae"] for s in seeds]
            mean = sum(maes) / len(maes)
            sd = math.sqrt(sum((m - mean) ** 2 for m in maes) / (len(maes) - 1)) if len(maes) > 1 else 0.0
            rmse = sum(s["metrics"][split][arm]["rmse"] for s in seeds) / len(seeds)
            aggregates[split][arm] = (mean, sd)
            lines.append(f"| {split} | {label} | {mean:.6f} ± {sd:.6f} | {rmse:.6f} |")
    lines += ["", "## Importance-weight reliability", "",
              "| Seed | Split | Estimator | Mean ESS / paths | Minimum ESS | Graphs below 10% ESS |",
              "|---:|---|---|---:|---:|---:|"]
    for seed in seeds:
        for split in ("test", "ood"):
            for arm in ("feat_gnn_jarzynski", "no_transport_jarzynski"):
                m = seed["metrics"][split][arm]
                lines.append(f"| {seed['seed']} | {split} | {LABELS[arm]} | {m['ess_mean']:.1f} / {config['paths_per_graph']} | {m['ess_min']:.1f} | {m['low_ess_graphs']} |")
    lines += ["", "## Cost", "", "| Seed | Training arm | Parameters | Seconds | Delta-F labels | Endpoint samples |",
              "|---:|---|---:|---:|---:|---:|"]
    for seed in seeds:
        for arm, cost in seed["training"].items():
            lines.append(f"| {seed['seed']} | {arm} | {cost['parameters']} | {cost['seconds']:.2f} | {cost['supervised_free_energy_labels']} | {cost['endpoint_training_configurations']} |")
    lines += ["", "| Seed | Split | Inference arm | Seconds |", "|---:|---|---|---:|"]
    for seed in seeds:
        for split, timing in seed["inference_seconds"].items():
            for arm, seconds in timing.items():
                lines.append(f"| {seed['seed']} | {split} | {LABELS[arm]} | {seconds:.3f} |")
    lines += ["", "Mean-work ablation reuses the FEAT trajectories; it is not a separately trained ordinary GNN.",
              "Low ESS indicates unstable finite-sample importance weighting. Even high ESS cannot prove all tails were sampled.",
              "This synthetic benchmark does not establish molecular, MOF, or adsorption performance.", "",
              "![Comparison](comparison.png)", "",
              f"Source JSON SHA256: `{hashlib.sha256(args.results.read_bytes()).hexdigest()}`.",
              "Full per-graph evidence, configuration, source hashes and environment are in the source JSON.", ""]
    (args.output / "comparison.md").write_text("\n".join(lines), encoding="utf-8")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    short_labels = ["GNN", "FEAT + JE", "Mean work", "No transport"]
    for axis, split in zip(axes, ("test", "ood")):
        values = [aggregates[split][arm] for arm in LABELS]
        axis.bar(range(4), [v[0] for v in values], yerr=[v[1] for v in values], capsize=4,
                 color=["#507dbc", "#2a9d8f", "#e9c46a", "#e76f51"])
        axis.set_xticks(range(4), short_labels, rotation=15)
        axis.set_title("4-node unseen graphs" if split == "test" else "6-node size OOD")
        axis.set_ylabel("Free-energy MAE (kBT)")
        axis.spines[["top", "right"]].set_visible(False)
    figure.suptitle("Synthetic harmonic graphs — supervision and inference costs differ")
    figure.tight_layout()
    figure.savefig(args.output / "comparison.png", dpi=180)
    figure.savefig(args.output / "comparison.svg")
    plt.close(figure)


if __name__ == "__main__":
    main()

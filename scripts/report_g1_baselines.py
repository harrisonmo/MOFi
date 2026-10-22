"""Independently replay saved baselines, verify artifacts, and summarize development results."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import spearmanr

from g1_model import PeriodicMPNN, pack

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs/g1-baseline-01"
DATA = ROOT / "datasets/processed/g1"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    result = read(RUN / "results.json")
    assert result["status"] == "completed" and not result["heldout_evaluated"]
    provenance, config, predictions = (read(RUN / name) for name in ["provenance.json", "config.json", "predictions.json"])
    assert sha(DATA / "manifest.json") == provenance["dataset_manifest_sha256"] == config["dataset_manifest_sha256"]
    assert sha(ROOT / "requirements-gnn-lock.txt") == provenance["environment_lock_sha256"]
    assert read(ROOT / "configs/g1-comparison.json") == config
    for name, expected in provenance["source_sha256"].items():
        assert sha(ROOT / "scripts" / name) == expected, name
    manifest = read(DATA / "manifest.json")
    for name, expected in manifest["split_sha256"].items():
        assert sha(DATA / f"{name}.json") == expected
    train, validation = read(DATA / "train.json"), read(DATA / "validation.json")
    families = {name: set() for name in manifest["split_ids"]}
    assignment = {i: name for name, ids in manifest["split_ids"].items() for i in ids}
    assert len(assignment) == len(manifest["records"])
    for row in manifest["records"]:
        families[assignment[row["id"]]].add(row["family"])
    assert all(not a & b for i, a in enumerate(families.values()) for b in list(families.values())[:i])
    ids = [x["id"] for x in validation]
    y = np.array([x["target"] for x in validation])
    assert predictions["ids"] == ids
    np.testing.assert_array_equal(predictions["target"], y)
    train_y = np.array([x["target"] for x in train])
    assert result["training_target_mean"] == train_y.mean()
    assert result["training_target_std"] == train_y.std()
    graph_count = 0
    graphs = []
    for record in train + validation:
        path = ROOT / record["graph"]
        assert sha(path) == record["graph_sha256"]
        with np.load(path, allow_pickle=False) as graph:
            displacement = graph["positions"][graph["dst"]] + graph["shifts"] @ graph["cell"] - graph["positions"][graph["src"]]
            np.testing.assert_allclose(np.linalg.norm(displacement, axis=1), graph["distance"], rtol=1e-6, atol=1e-6)
            if record["id"] in set(ids):
                graphs.append({k: graph[k].copy() for k in ["z", "src", "dst", "distance"]})
        graph_count += 1
    assert len(graphs) == len(validation)
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    xval = np.array([x["geometry"] for x in validation])
    artifact_hashes = {}
    prediction_checks = []
    for row in result["rows"]:
        model, seed = row["model"], row["seed"]
        key = model if seed is None else f"{model}_{seed}"
        stored = np.array(predictions["models"][key])
        if model == "mean":
            replay = np.full(len(y), train_y.mean())
        elif model == "extra_trees":
            path = RUN / f"extra-trees-{seed}.joblib"
            artifact_hashes[path.name] = sha(path)
            tree = joblib.load(path)
            assert tree.n_features_in_ == 14 and tree.n_estimators == 300 and tree.min_samples_leaf == 2
            replay = tree.predict(xval)
        else:
            history = read(RUN / f"history-{seed}.json")
            assert row["best_epoch"] == min(history, key=lambda h: h["validation_mae"])["epoch"]
            assert row["epochs"] <= 50 and len(history) == row["epochs"]
            assert min(h["validation_mae"] for h in history) == row["mae"]
            path = RUN / f"gnn-{seed}.pt"
            assert sha(path) == row["checkpoint_sha256"]
            artifact_hashes[path.name] = sha(path)
            model_object = PeriodicMPNN().cuda().eval()
            model_object.load_state_dict(torch.load(path, weights_only=True, map_location="cuda"))
            values = []
            with torch.no_grad():
                for start in range(0, len(graphs), 16):
                    output = model_object(pack(graphs[start:start + 16], "cuda"))
                    values.extend((output * train_y.std() + train_y.mean()).cpu().tolist())
            replay = np.array(values)
        np.testing.assert_allclose(replay, stored, rtol=1e-6, atol=1e-5)
        mae = float(np.abs(stored - y).mean())
        rmse = float(np.sqrt(np.square(stored - y).mean()))
        assert abs(mae - row["mae"]) < 1e-12 and abs(rmse - row["rmse"]) < 1e-12
        rho = None if np.ptp(stored) == 0 else float(spearmanr(y, stored).statistic)
        assert rho == row["spearman"]
        k = int(np.ceil(.1 * len(y)))
        order_true = sorted(range(len(y)), key=lambda i: (-y[i], ids[i]))[:k]
        order_pred = sorted(range(len(y)), key=lambda i: (-stored[i], ids[i]))[:k]
        assert len(set(order_true) & set(order_pred)) / k == row["top_10_percent_recall"]
        prediction_checks.append({"model": key, "count": len(y), "max_abs_replay_difference": float(abs(replay - stored).max())})
    assert result["total_seconds"] <= config["wall_seconds"]
    aggregates = {}
    for name in ["mean", "extra_trees", "gnn"]:
        rows = [r for r in result["rows"] if r["model"] == name]
        aggregates[name] = {}
        for metric in ["mae", "rmse", "spearman", "top_10_percent_recall", "seconds"]:
            values = [r[metric] for r in rows if r[metric] is not None]
            aggregates[name][metric] = {"mean": float(np.mean(values)) if values else None,
                                       "seed_sd": float(np.std(values, ddof=1)) if len(values) > 1 else None}
    tree_errors = np.mean([np.abs(np.array(predictions["models"][f"extra_trees_{seed}"]) - y) for seed in config["seeds"]], axis=0)
    gnn_errors = np.mean([np.abs(np.array(predictions["models"][f"gnn_{seed}"]) - y) for seed in config["seeds"]], axis=0)
    difference = gnn_errors - tree_errors
    family_labels = np.array(predictions["family"])
    family_groups = [np.flatnonzero(family_labels == f) for f in sorted(set(family_labels))]
    sums = np.array([difference[index].sum() for index in family_groups])
    counts = np.array([len(index) for index in family_groups])
    rng = np.random.default_rng(20261013)
    samples = []
    for _ in range(2000):
        selected = rng.integers(0, len(family_groups), len(family_groups))
        samples.append(sums[selected].sum() / counts[selected].sum())
    audit = {"artifact_hashes": artifact_hashes, "periodic_graphs_checked": graph_count,
             "prediction_replay": prediction_checks, "metrics_recomputed": len(result["rows"]),
             "family_overlap": 0, "heldout_evaluated": False,
             "paired_family_bootstrap": {"statistic": "mean per-seed GNN absolute error minus tree absolute error",
                 "difference_mmol_g": float(difference.mean()), "percentile_95_interval": np.quantile(samples, [.025, .975]).tolist(),
                 "replicates": 2000, "seed": 20261013,
                 "limitation": "Exploratory development-set interval after early stopping; not an independent confirmatory test."}}
    output = {"results": result, "aggregate": aggregates, "audit": audit, "provenance": provenance,
              "predictions_sha256": sha(RUN / "predictions.json"), "config_sha256": sha(RUN / "config.json")}
    report = ROOT / "docs/reports/G1_BASELINE_RESULT.json"
    report.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), constrained_layout=True)
    names = ["mean", "extra_trees", "gnn"]
    means = [aggregates[n]["mae"]["mean"] for n in names]
    errors = [aggregates[n]["mae"]["seed_sd"] or 0 for n in names]
    axes[0].bar(["Mean", "Geometry tree", "Periodic GNN"], means, yerr=errors, capsize=4, color=["#999999", "#247ba0", "#e59500"])
    axes[0].set_ylabel("Development MAE (mmol/g)")
    axes[0].set_title("Lower is better; bars: seed SD")
    axes[0].tick_params(axis="x", labelsize=9)
    for ax, name, color in zip(axes[1:], ["extra_trees", "gnn"], ["#247ba0", "#e59500"]):
        pred = np.mean([predictions["models"][f"{name}_{seed}"] for seed in config["seeds"]], axis=0)
        ax.scatter(y, pred, s=9, alpha=.5, color=color)
        limits = [min(y.min(), pred.min()), max(y.max(), pred.max())]
        ax.plot(limits, limits, color="black", lw=1, linestyle="--")
        ax.set_xlabel("Source target (mmol/g)")
        ax.set_ylabel("Mean prediction across seeds (mmol/g)")
        ax.set_title("Geometry tree" if name == "extra_trees" else "Periodic GNN")
    fig.suptitle("G1: 449 development structures, 131 held-out-from-training scaffold groups")
    fig.savefig(ROOT / "docs/reports/G1_BASELINE_COMPARISON.png", dpi=160)
    plt.close(fig)
    print(json.dumps({"aggregate": aggregates, "bootstrap": audit["paired_family_bootstrap"], "checks": prediction_checks}), flush=True)


if __name__ == "__main__":
    main()

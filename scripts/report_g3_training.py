"""Replay G3 checkpoints and compare them with the G2 arms on the same development split."""
import hashlib
import json
import os
import time
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from g3_model import SmoothPeriodicMPNN, fit_normalizers, pack

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "datasets/processed/g1"
G3 = ROOT / "datasets/processed/g3"
RUN = ROOT / "runs/g3-training-01"
G2 = ROOT / "runs/g2-completed-01"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    started = time.monotonic()
    result, pred, provenance, config, prep = [read(RUN / name) for name in [
        "results.json", "predictions.json", "provenance.json", "config.json", "preprocessing.json"]]
    deadline = started + config["audit_seconds"]
    assert result["status"] == "completed" and len(result["rows"]) == len(config["seeds"])
    assert result["total_seconds"] < config["formal_seconds"]
    assert not result["heldout_evaluated"] and not provenance["heldout_labels_loaded"]
    assert not result["train_excluded"] and not result["validation_excluded"]
    for name, expected in provenance["source_sha256"].items():
        assert sha(ROOT / "scripts" / name) == expected, name
    assert sha(ROOT / "requirements-gnn-lock.txt") == provenance["environment_lock_sha256"]
    assert sha(ROOT / "configs/g3-training.json") == provenance["config_sha256"]
    assert config == read(ROOT / "configs/g3-training.json")
    assert sha(DATA / "manifest.json") == config["dataset_manifest_sha256"]
    assert sha(G3 / "manifest.json") == provenance["g3_graph_manifest_sha256"]
    admission = read(ROOT / "docs/reports/G3_CALIBRATION.json")
    assert admission["admitted"] and admission["config_sha256"] == provenance["config_sha256"]
    manifest = read(DATA / "manifest.json")
    # Hash heldout bytes for integrity without parsing labels or evaluating that split.
    for name, expected in manifest["split_sha256"].items():
        assert sha(DATA / f"{name}.json") == expected
    train, val = read(DATA / "train.json"), read(DATA / "validation.json")
    assert prep["fit_ids"] == [r["id"] for r in train]
    assert pred["ids"] == [r["id"] for r in val]
    y = np.array([r["target"] for r in train]); yv = np.array([r["target"] for r in val])
    np.testing.assert_array_equal(yv, pred["target"])
    assert y.mean() == prep["target_mean"] and y.std() == prep["target_std"]
    index = {r["id"]: r for r in read(G3 / "manifest.json")["records"]}
    graphs = []
    for row in train + val:
        record = index[row["id"]]
        path = ROOT / record["graph"]
        assert sha(path) == record["graph_sha256"]
        with np.load(path, allow_pickle=False) as graph:
            graphs.append({k: graph[k].copy() for k in ["z", "src", "dst", "distance", "cell"]})
    tg, vg = graphs[:len(train)], graphs[len(train):]
    normalizers = fit_normalizers(tg)
    assert all(normalizers[k] == prep[k] for k in normalizers)
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    def replay(model, graphs):
        out = []
        with torch.no_grad():
            for start in range(0, len(graphs), 16):
                if time.monotonic() >= deadline:
                    raise TimeoutError("Audit budget exceeded")
                values = model(pack(graphs[start:start + 16], "cuda"))
                out.extend((values * prep["target_std"] + prep["target_mean"]).cpu().tolist())
        return np.array(out)

    checks = []
    assert [r["seed"] for r in result["rows"]] == config["seeds"]
    for row in result["rows"]:
        seed = row["seed"]
        path = RUN / f"g3-{seed}.pt"
        assert sha(path) == row["checkpoint_sha256"] and row["seconds"] < config["trial_seconds"]
        model = SmoothPeriodicMPNN(**normalizers, cutoff=config["cutoff"]).cuda().eval()
        model.load_state_dict(torch.load(path, weights_only=True, map_location="cuda"))
        assert sum(p.numel() for p in model.parameters()) == row["parameters"]
        vv, tt = replay(model, vg), replay(model, tg)
        history = read(RUN / f"history-g3-{seed}.json")
        assert len(history) == row["epochs"] <= config["max_epochs"]
        assert min(history, key=lambda h: h["validation_mae"])["epoch"] == row["best_epoch"]
        assert min(h["validation_mae"] for h in history) == row["mae"]
        assert row["optimizer_updates"] == len(history) * int(np.ceil(len(train) / 16))
        stored = np.array(pred["models"][f"g3_{seed}"])
        np.testing.assert_allclose(vv, stored, rtol=1e-6, atol=1e-5)
        assert abs(np.abs(tt - y).mean() - row["train_mae"]) < 1e-6
        assert abs(np.abs(stored - yv).mean() - row["mae"]) < 1e-12
        assert float(spearmanr(stored, yv).statistic) == row["spearman"]
        checks.append({"seed": seed, "max_abs_prediction_difference": float(abs(vv - stored).max())})
    g2 = read(G2 / "predictions.json")
    assert g2["ids"] == pred["ids"] and g2["target"] == pred["target"]
    g2_rows = read(G2 / "results.json")["rows"]
    summary = {}
    for arm in ["g3", "gnn", "hybrid", "mlp", "tree"]:
        rows = result["rows"] if arm == "g3" else [r for r in g2_rows if r["arm"] == arm]
        summary[arm] = {metric: {"mean": float(np.mean([r[metric] for r in rows])),
                                 "seed_sd": float(np.std([r[metric] for r in rows], ddof=1))}
                        for metric in ["mae", "rmse", "train_mae", "spearman", "top_10_percent_recall", "seconds"]}
    family = np.array(pred["family"])
    groups = [np.flatnonzero(family == f) for f in sorted(set(family))]
    models = {**g2["models"], **pred["models"]}
    errors = {arm: np.mean([np.abs(np.array(models[f"{arm}_{s}"]) - yv) for s in config["seeds"]], axis=0)
              for arm in summary}
    paired = []
    for a, b in [("g3", "gnn"), ("g3", "hybrid"), ("g3", "mlp")]:
        d = errors[a] - errors[b]
        sums = np.array([d[g].sum() for g in groups]); counts = np.array([len(g) for g in groups])
        rng = np.random.default_rng(20261021)
        estimates = []
        for _ in range(2000):
            sample = rng.integers(0, len(groups), len(groups))
            estimates.append(sums[sample].sum() / counts[sample].sum())
        paired.append({"contrast": f"{a}_minus_{b}", "mae_difference": float(d.mean()),
                       "family_bootstrap_95_percentile": np.quantile(estimates, [.025, .975]).tolist()})
    report = {"result": result, "summary": summary, "paired_development_contrasts": paired,
              "provenance": provenance, "calibration": admission,
              "audit": {"checkpoint_prediction_checks": checks, "training_predictions_per_model": len(train),
                        "validation_predictions_per_model": len(val), "verified_graph_files": len(graphs),
                        "heldout_evaluated": False, "seconds": time.monotonic() - started},
              "limitations": ["Same repeatedly used development split; early stopping uses validation.",
                              "Cutoff, envelope and count/density changes are introduced together and not separated."]}
    assert report["audit"]["seconds"] < config["audit_seconds"]
    (ROOT / "docs/reports/G3_RESULT.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), constrained_layout=True)
    arms = ["gnn", "g3", "hybrid", "mlp", "tree"]
    labels = ["G2 GNN", "G3 GNN", "G2 GNN + geometry", "Geometry MLP", "Geometry tree"]
    colors = ["#e59500", "#c0392b", "#398564", "#8555a5", "#247ba0"]
    axes[0].bar(labels, [summary[a]["mae"]["mean"] for a in arms], color=colors,
                yerr=[summary[a]["mae"]["seed_sd"] for a in arms], capsize=4)
    axes[0].set_ylabel("Development MAE (mmol/g)")
    axes[0].set_title("Mean and SD across three seeds")
    for seed, colour in zip(config["seeds"], ["#c0392b", "#e67e22", "#7f8c8d"]):
        history = read(RUN / f"history-g3-{seed}.json")
        axes[1].plot([h["epoch"] for h in history], [h["validation_mae"] for h in history], color=colour, label=f"G3 seed {seed}")
    axes[1].axhline(summary["gnn"]["mae"]["mean"], color="#e59500", linestyle="--", label="G2 GNN mean")
    axes[1].axhline(summary["hybrid"]["mae"]["mean"], color="#398564", linestyle="--", label="G2 GNN + geometry mean")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Development MAE (mmol/g)")
    axes[1].set_title("G3 development curves")
    axes[1].legend(fontsize=8)
    axes[0].tick_params(axis="x", rotation=15, labelsize=9)
    fig.suptitle("G3: fixed G1 data and scaffold split; final heldout set not evaluated")
    fig.savefig(ROOT / "docs/reports/G3_COMPARISON.png", dpi=160)
    plt.close(fig)
    print(json.dumps({"summary": {a: summary[a]["mae"] for a in summary}, "paired": paired,
                      "train_mae": summary["g3"]["train_mae"], "audit_seconds": report["audit"]["seconds"]}), flush=True)


if __name__ == "__main__":
    main()

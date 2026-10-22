"""Replay four-arm artifacts and describe paired development-set differences."""
import hashlib
import json
import os
import time
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import joblib
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from g1_model import pack
from g2_models import make_model

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "datasets/processed/g1"
RUN = ROOT / "runs/g2-completed-01"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    started = time.monotonic()
    result, pred, provenance, config, prep = [read(RUN / name) for name in [
        "results.json", "predictions.json", "provenance.json", "config.json", "preprocessing.json"]]
    deadline = started + config["audit_seconds"]
    assert result["status"] == "completed" and len(result["rows"]) == 12
    assert result["total_seconds"] < config["formal_seconds"]
    assert not result["heldout_evaluated"] and not provenance["heldout_labels_loaded"]
    original_dir = ROOT / "runs/g2-comparison-01"
    recovery_dir = ROOT / "runs/g2-recovery-01"
    original = read(original_dir / "results.json")
    recovery = read(recovery_dir / "recovery.json")
    assert original["status"] == "running" and len(original["rows"]) == 11
    assert result["rows"][:11] == original["rows"]
    assert result["rows"][-1] == read(recovery_dir / "row.json")
    assert result["recovery"] == recovery
    assert recovery["source_sha256"] == sha(ROOT / "scripts/recover_g2_interruption.py")
    partial = read(original_dir / "history-hybrid-43.json")
    assert len(partial) == recovery["original_partial_epochs"] == 42
    assert read(RUN / "history-hybrid-43.json")[:len(partial)] == partial
    for name in ["history-hybrid-43.json", "hybrid-43.pt"]:
        assert sha(RUN / name) == sha(recovery_dir / name)
    old_predictions = read(original_dir / "predictions.json")
    for key, values in old_predictions["models"].items():
        assert pred["models"][key] == values
    for row in original["rows"]:
        name = f"{row['arm']}-{row['seed']}.{'joblib' if row['arm'] == 'tree' else 'pt'}"
        assert sha(RUN / name) == sha(original_dir / name)
    assert recovery["attempts"] == 13 and recovery["completed_trials"] == 12
    assert recovery["recovery_seconds"] < config["neural_trial_seconds"]
    accounted = sum(r["seconds"] for r in original["rows"]) + 150 + 30 + recovery["recovery_seconds"]
    assert abs(accounted - result["total_seconds"]) < 1e-9
    assert accounted == recovery["budget_accounted_seconds"]
    for name, expected in provenance["source_sha256"].items():
        assert sha(ROOT / "scripts" / name) == expected, name
    assert sha(ROOT / "requirements-gnn-lock.txt") == provenance["environment_lock_sha256"]
    assert sha(ROOT / "configs/g2-comparison.json") == provenance["config_sha256"]
    assert config == read(ROOT / "configs/g2-comparison.json")
    assert sha(DATA / "manifest.json") == config["dataset_manifest_sha256"]
    manifest = read(DATA / "manifest.json")
    # Hash heldout bytes for integrity without parsing labels or evaluating that split.
    for name, expected in manifest["split_sha256"].items():
        assert sha(DATA / f"{name}.json") == expected
    families = {key: set() for key in manifest["split_ids"]}
    split = {i: key for key, ids in manifest["split_ids"].items() for i in ids}
    assert len(split) == len(manifest["records"])
    for row in manifest["records"]:
        families[split[row["id"]]].add(row["family"])
    assert all(not a & b for i, a in enumerate(families.values()) for b in list(families.values())[:i])
    train, val = read(DATA / "train.json"), read(DATA / "validation.json")
    assert prep["fit_ids"] == [r["id"] for r in train]
    assert pred["ids"] == [r["id"] for r in val]
    x = np.array([r["geometry"] for r in train]); xv = np.array([r["geometry"] for r in val])
    y = np.array([r["target"] for r in train]); yv = np.array([r["target"] for r in val])
    np.testing.assert_array_equal(yv, pred["target"])
    mu, scale = x.mean(axis=0), x.std(axis=0); scale[scale == 0] = 1
    np.testing.assert_array_equal(mu, prep["geometry_mean"])
    np.testing.assert_array_equal(scale, prep["geometry_scale"])
    assert y.mean() == prep["target_mean"] and y.std() == prep["target_std"]
    xs, xvs = ((x - mu) / scale).astype(np.float32), ((xv - mu) / scale).astype(np.float32)
    graphs = []
    for row in train + val:
        path = ROOT / row["graph"]
        assert sha(path) == row["graph_sha256"]
        with np.load(path, allow_pickle=False) as graph:
            graphs.append({k: graph[k].copy() for k in ["z", "src", "dst", "distance"]})
    tg, vg = graphs[:len(train)], graphs[len(train):]
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    def replay(model, arm, graphs, geometry):
        out = []
        with torch.no_grad():
            for start in range(0, len(geometry), 16):
                if time.monotonic() >= deadline:
                    raise TimeoutError("Audit budget exceeded")
                xx = torch.as_tensor(geometry[start:start + 16], device="cuda")
                if arm == "mlp":
                    values = model(xx)
                else:
                    batch = pack(graphs[start:start + 16], "cuda")
                    values = model(batch) if arm == "gnn" else model(batch, xx)
                out.extend((values * prep["target_std"] + prep["target_mean"]).cpu().tolist())
        return np.array(out)

    checks = []
    order = []
    for turn, seed in enumerate(config["seeds"]):
        order.extend((arm, seed) for arm in config["arms"][turn:] + config["arms"][:turn])
    assert [(r["arm"], r["seed"]) for r in result["rows"]] == order
    for row in result["rows"]:
        arm, seed = row["arm"], row["seed"]
        path = RUN / f"{arm}-{seed}.{'joblib' if arm == 'tree' else 'pt'}"
        assert sha(path) == row["checkpoint_sha256"]
        assert row["seconds"] < config["tree_trial_seconds" if arm == "tree" else "neural_trial_seconds"]
        if arm == "tree":
            model = joblib.load(path)
            assert model.n_features_in_ == 14 and model.n_estimators == 300 and model.random_state == seed
            vv, tt = model.predict(xv), model.predict(x)
        else:
            model = make_model(arm).cuda().eval()
            model.load_state_dict(torch.load(path, weights_only=True, map_location="cuda"))
            assert sum(p.numel() for p in model.parameters()) == row["parameters"]
            vv, tt = replay(model, arm, vg, xvs), replay(model, arm, tg, xs)
            history = read(RUN / f"history-{arm}-{seed}.json")
            assert len(history) == row["epochs"] <= 50
            assert min(history, key=lambda h: h["validation_mae"])["epoch"] == row["best_epoch"]
            assert min(h["validation_mae"] for h in history) == row["mae"]
            assert row["optimizer_updates"] == len(history) * int(np.ceil(len(train) / 16))
        stored = np.array(pred["models"][f"{arm}_{seed}"])
        np.testing.assert_allclose(vv, stored, rtol=1e-6, atol=1e-5)
        assert abs(np.abs(tt - y).mean() - row["train_mae"]) < 1e-6
        assert abs(np.abs(stored - yv).mean() - row["mae"]) < 1e-12
        assert abs(np.sqrt(np.square(stored - yv).mean()) - row["rmse"]) < 1e-12
        assert float(spearmanr(stored, yv).statistic) == row["spearman"]
        k = int(np.ceil(.1 * len(yv)))
        true_ids = sorted(range(len(yv)), key=lambda i: (-yv[i], pred["ids"][i]))[:k]
        selected = sorted(range(len(yv)), key=lambda i: (-stored[i], pred["ids"][i]))[:k]
        assert len(set(true_ids) & set(selected)) / k == row["top_10_percent_recall"]
        checks.append({"arm": arm, "seed": seed, "max_abs_prediction_difference": float(abs(vv - stored).max())})
    summary = {}
    for arm in config["arms"]:
        rows = [r for r in result["rows"] if r["arm"] == arm]
        summary[arm] = {metric: {"mean": float(np.mean([r[metric] for r in rows])),
                                 "seed_sd": float(np.std([r[metric] for r in rows], ddof=1))}
                        for metric in ["mae", "rmse", "train_mae", "spearman", "top_10_percent_recall", "seconds"]}
    family = np.array(pred["family"])
    groups = [np.flatnonzero(family == f) for f in sorted(set(family))]
    errors = {arm: np.mean([np.abs(np.array(pred["models"][f"{arm}_{s}"]) - yv) for s in config["seeds"]], axis=0) for arm in config["arms"]}
    paired = []
    for a, b in [("hybrid", "gnn"), ("mlp", "tree"), ("hybrid", "mlp"), ("hybrid", "tree")]:
        d = errors[a] - errors[b]
        sums = np.array([d[g].sum() for g in groups]); counts = np.array([len(g) for g in groups])
        rng = np.random.default_rng(20261013)
        estimates = []
        for _ in range(2000):
            sample = rng.integers(0, len(groups), len(groups))
            estimates.append(sums[sample].sum() / counts[sample].sum())
        paired.append({"contrast": f"{a}_minus_{b}", "mae_difference": float(d.mean()),
                       "family_bootstrap_95_percentile": np.quantile(estimates, [.025, .975]).tolist()})
    old = read(ROOT / "runs/g1-baseline-01/predictions.json")
    assert old["ids"] == pred["ids"]
    anchors = {f"{arm}_{s}": float(np.max(np.abs(np.array(pred["models"][f"{arm}_{s}"]) - np.array(old["models"][f"{old_arm}_{s}"]))))
               for arm, old_arm in [("gnn", "gnn"), ("tree", "extra_trees")] for s in config["seeds"]}
    report = {"result": result, "summary": summary, "paired_development_contrasts": paired,
              "provenance": provenance, "audit": {"checkpoint_prediction_checks": checks,
                "recovery_verified": True, "original_partial_history_sha256": sha(original_dir / "history-hybrid-43.json"),
                "training_predictions_per_model": len(train), "validation_predictions_per_model": len(val),
                "verified_graph_files": len(graphs), "family_overlap": 0, "heldout_evaluated": False,
                "g1_anchor_max_prediction_differences": anchors, "seconds": time.monotonic() - started},
              "limitations": ["Same repeatedly used development split; early stopping uses validation.",
                              "Input representation, capacity, optimization and inductive bias are not fully separated."]}
    assert report["audit"]["seconds"] < config["audit_seconds"]
    (ROOT / "docs/reports/G2_RESULT.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), constrained_layout=True)
    arms = config["arms"]; labels = ["GNN", "GNN + geometry", "Geometry MLP", "Geometry tree"]
    colors = ["#e59500", "#398564", "#8555a5", "#247ba0"]
    axes[0].bar(labels, [summary[a]["mae"]["mean"] for a in arms], color=colors,
                yerr=[summary[a]["mae"]["seed_sd"] for a in arms], capsize=4)
    axes[0].set_ylabel("Development MAE (mmol/g)")
    axes[0].set_title("Mean and SD across three seeds")
    xaxis = np.arange(4)
    axes[1].bar(xaxis - .18, [summary[a]["train_mae"]["mean"] for a in arms], width=.36, label="Training")
    axes[1].bar(xaxis + .18, [summary[a]["mae"]["mean"] for a in arms], width=.36, label="Development")
    axes[1].set_xticks(xaxis, labels)
    axes[1].set_ylabel("MAE at selected checkpoint (mmol/g)")
    axes[1].set_title("Training fit versus development error")
    axes[1].legend()
    for ax in axes: ax.tick_params(axis="x", rotation=15, labelsize=9)
    fig.suptitle("G2: fixed G1 data and scaffold split; final heldout set not evaluated")
    fig.savefig(ROOT / "docs/reports/G2_COMPARISON.png", dpi=160)
    plt.close(fig)
    print(json.dumps({"summary": summary, "paired": paired, "anchor_differences": anchors}), flush=True)


if __name__ == "__main__":
    main()

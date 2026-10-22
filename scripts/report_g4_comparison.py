"""Replay G4 checkpoints and describe the preregistered development-set contrasts."""
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
from g2_models import GeometryScaler
from g4_models import fit_normalizers, forward, make_model, pack

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "datasets/processed/g1"
G3 = ROOT / "datasets/processed/g3"
RUN = ROOT / "runs/g4-comparison-01"
REFERENCES = {"g3_50": (ROOT / "runs/g3-training-01", "g3", "g3"),
              "g3_hard_50": (ROOT / "runs/g3-ablation-01", "c8-hard-sum", "c8-hard-sum"),
              "g2_hybrid_50": (ROOT / "runs/g2-completed-01", "hybrid", "hybrid")}


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    started = time.monotonic()
    result, pred, provenance, config, prep = [read(RUN / name) for name in [
        "results.json", "predictions.json", "provenance.json", "config.json", "preprocessing.json"]]
    deadline = started + config["audit_seconds"]
    arms, seeds = config["arms"], config["seeds"]
    assert result["status"] == "completed" and len(result["rows"]) == len(arms) * len(seeds)
    assert result["total_seconds"] < config["formal_seconds"]
    assert not result["heldout_evaluated"] and not provenance["heldout_labels_loaded"]
    for name, expected in provenance["source_sha256"].items():
        assert sha(ROOT / "scripts" / name) == expected, name
    assert sha(ROOT / "requirements-gnn-lock.txt") == provenance["environment_lock_sha256"]
    assert sha(ROOT / "configs/g4-comparison.json") == provenance["config_sha256"]
    assert config == read(ROOT / "configs/g4-comparison.json")
    assert sha(DATA / "manifest.json") == config["dataset_manifest_sha256"]
    assert sha(G3 / "manifest.json") == provenance["g3_graph_manifest_sha256"]
    admission = read(ROOT / "docs/reports/G4_CALIBRATION.json")
    assert admission["admitted"] and admission["config_sha256"] == provenance["config_sha256"]
    manifest = read(DATA / "manifest.json")
    # Hash heldout bytes for integrity without parsing labels or evaluating that split.
    for name, expected in manifest["split_sha256"].items():
        assert sha(DATA / f"{name}.json") == expected
    train, val = read(DATA / "train.json"), read(DATA / "validation.json")
    assert prep["fit_ids"] == [r["id"] for r in train] and pred["ids"] == [r["id"] for r in val]
    x = np.array([r["geometry"] for r in train]); xv = np.array([r["geometry"] for r in val])
    y = np.array([r["target"] for r in train]); yv = np.array([r["target"] for r in val])
    np.testing.assert_array_equal(yv, pred["target"])
    assert y.mean() == prep["target_mean"] and y.std() == prep["target_std"]
    scaler = GeometryScaler().fit(x)
    np.testing.assert_array_equal(scaler.mean, prep["geometry_mean"])
    np.testing.assert_array_equal(scaler.scale, prep["geometry_scale"])
    xs, xvs = scaler.transform(x), scaler.transform(xv)
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

    def replay(model, arm, graphs, geometry):
        out = []
        with torch.no_grad():
            for start in range(0, len(graphs), 16):
                if time.monotonic() >= deadline:
                    raise TimeoutError("Audit budget exceeded")
                xx = torch.as_tensor(geometry[start:start + 16], device="cuda")
                values = forward(model, arm, pack(graphs[start:start + 16], "cuda"), xx)
                out.extend((values * prep["target_std"] + prep["target_mean"]).cpu().tolist())
        return np.array(out)

    order = [(arm, seed) for turn, seed in enumerate(seeds) for arm in arms[turn:] + arms[:turn]]
    assert [(r["arm"], r["seed"]) for r in result["rows"]] == order
    checks = []
    for row in result["rows"]:
        arm, seed = row["arm"], row["seed"]
        path = RUN / f"{arm}-{seed}.pt"
        assert sha(path) == row["checkpoint_sha256"] and row["seconds"] < config["trial_seconds"]
        model = make_model(arm, normalizers, config["g4_graph_envelope"]).cuda().eval()
        model.load_state_dict(torch.load(path, weights_only=True, map_location="cuda"))
        assert sum(p.numel() for p in model.parameters()) == row["parameters"]
        vv, tt = replay(model, arm, vg, xvs), replay(model, arm, tg, xs)
        history = read(RUN / f"history-{arm}-{seed}.json")
        assert len(history) == row["epochs"] <= config["max_epochs"]
        assert min(history, key=lambda h: h["validation_mae"])["epoch"] == row["best_epoch"]
        assert min(h["validation_mae"] for h in history) == row["mae"]
        assert row["optimizer_updates"] == len(history) * int(np.ceil(len(train) / 16))
        stored = np.array(pred["models"][f"{arm}_{seed}"])
        np.testing.assert_allclose(vv, stored, rtol=1e-6, atol=1e-5)
        assert abs(np.abs(tt - y).mean() - row["train_mae"]) < 1e-6
        assert abs(np.abs(stored - yv).mean() - row["mae"]) < 1e-12
        assert float(spearmanr(stored, yv).statistic) == row["spearman"]
        checks.append({"arm": arm, "seed": seed, "max_abs_prediction_difference": float(abs(vv - stored).max())})
    # The two controls should retrace their 50-epoch runs while both histories exist.
    retrace = {}
    for arm, (run, name, _) in [("g3", REFERENCES["g3_50"]), ("g3_hard", REFERENCES["g3_hard_50"])]:
        for seed in seeds:
            new, old = read(RUN / f"history-{arm}-{seed}.json"), read(run / f"history-{name}-{seed}.json")
            shared = min(len(new), len(old))
            retrace[f"{arm}_{seed}"] = {"shared_epochs": shared, "max_validation_mae_difference": float(max(
                abs(a["validation_mae"] - b["validation_mae"]) for a, b in zip(new[:shared], old[:shared])))}
    models = dict(pred["models"])
    summary = {}
    for arm in arms:
        rows = [r for r in result["rows"] if r["arm"] == arm]
        summary[arm] = {metric: {"mean": float(np.mean([r[metric] for r in rows])),
                                 "seed_sd": float(np.std([r[metric] for r in rows], ddof=1))}
                        for metric in ["mae", "rmse", "train_mae", "spearman", "top_10_percent_recall",
                                       "epochs", "best_epoch", "seconds", "parameters"]}
        summary[arm]["epoch_cap_reached"] = sum(r["epochs"] == config["max_epochs"] for r in rows)
    for label, (run, name, key) in REFERENCES.items():
        old = read(run / "predictions.json")
        assert old["ids"] == pred["ids"]
        rows = [r for r in read(run / "results.json")["rows"] if r.get("arm", r.get("cell")) == name]
        summary[label] = {"mae": {"mean": float(np.mean([r["mae"] for r in rows])),
                                  "seed_sd": float(np.std([r["mae"] for r in rows], ddof=1))}}
        models.update({f"{label}_{s}": old["models"][f"{key}_{s}"] for s in seeds})
    errors = {label: np.mean([np.abs(np.array(models[f"{label}_{s}"]) - yv) for s in seeds], axis=0) for label in summary}
    family = np.array(pred["family"])
    groups = [np.flatnonzero(family == f) for f in sorted(set(family))]
    counts = np.array([len(g) for g in groups])
    rng = np.random.default_rng(20261021)
    samples = [rng.integers(0, len(groups), len(groups)) for _ in range(2000)]
    contrasts = []
    for kind, a, b in [("primary", "g4_hybrid", "g3"), ("primary", "g4_graph", "g3_hard"),
                       ("secondary", "g3_hard", "g3"), ("secondary", "g3", "g3_50"),
                       ("secondary", "g3_hard", "g3_hard_50"), ("secondary", "g4_graph", "g4_hybrid"),
                       ("context", "g4_hybrid", "g2_hybrid_50"), ("context", "g4_graph", "g2_hybrid_50")]:
        d = errors[a] - errors[b]
        sums = np.array([d[g].sum() for g in groups])
        interval = np.quantile([sums[s].sum() / counts[s].sum() for s in samples], [.025, .975]).tolist()
        contrasts.append({"kind": kind, "contrast": f"{a}_minus_{b}", "mae_difference": float(d.mean()),
                          "family_bootstrap_95_percentile": interval, "resolved": bool(interval[0] > 0 or interval[1] < 0)})
    report = {"result": result, "summary": summary, "contrasts": contrasts, "provenance": provenance,
              "calibration": admission,
              "audit": {"checkpoint_prediction_checks": checks, "control_history_retrace": retrace,
                        "training_predictions_per_model": len(train), "validation_predictions_per_model": len(val),
                        "verified_graph_files": len(graphs), "heldout_evaluated": False,
                        "seconds": time.monotonic() - started},
              "limitations": ["Same repeatedly used development split; early stopping uses validation.",
                              "g4_graph has 5.2% more parameters than its control.",
                              "Contrasts with 50-epoch references mix epoch budget and model."]}
    assert report["audit"]["seconds"] < config["audit_seconds"]
    (ROOT / "docs/reports/G4_RESULT.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), constrained_layout=True)
    labels = ["g2_hybrid_50", "g3_50", "g3", "g3_hard", "g4_hybrid", "g4_graph"]
    names = ["G2 hybrid\n(50 ep)", "G3\n(50 ep)", "G3\n(150 ep)", "G3 hard\n(150 ep)", "G4 hybrid", "G4 graph"]
    colors = ["#9aa5a0", "#d4a59a", "#c0392b", "#e67e22", "#398564", "#247ba0"]
    axes[0].bar(names, [summary[k]["mae"]["mean"] for k in labels], color=colors,
                yerr=[summary[k]["mae"]["seed_sd"] for k in labels], capsize=4)
    axes[0].set_ylabel("Development MAE (mmol/g)")
    axes[0].set_title("Mean and SD across three seeds")
    axes[0].tick_params(axis="x", labelsize=8)
    for arm, colour in zip(arms, colors[2:]):
        for i, seed in enumerate(seeds):
            history = read(RUN / f"history-{arm}-{seed}.json")
            axes[1].plot([h["epoch"] for h in history], [h["validation_mae"] for h in history], color=colour,
                         alpha=.8, linewidth=1, label=arm if i == 0 else None)
    axes[1].set_ylim(0.3, 1.2)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Development MAE (mmol/g)")
    axes[1].set_title("Development curves (all seeds)")
    axes[1].legend(fontsize=8)
    fig.suptitle("G4: fixed G1 data and scaffold split; final heldout set not evaluated")
    fig.savefig(ROOT / "docs/reports/G4_COMPARISON.png", dpi=160)
    plt.close(fig)
    print(json.dumps({"summary": {k: summary[k]["mae"] for k in summary}, "contrasts": contrasts,
                      "retrace": retrace, "audit_seconds": report["audit"]["seconds"]}, indent=1), flush=True)


if __name__ == "__main__":
    main()

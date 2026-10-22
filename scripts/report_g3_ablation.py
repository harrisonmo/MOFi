"""Replay G3 ablation checkpoints and estimate factorial effects on the development split."""
import hashlib
import itertools
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
from g3_ablation_model import CELLS, AblationMPNN
from g3_model import fit_normalizers, pack

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "datasets/processed/g1"
G3 = ROOT / "datasets/processed/g3"
RUN = ROOT / "runs/g3-ablation-01"
FACTORS = {"A_cutoff_8A": lambda c: CELLS[c]["cutoff"] == 8.0,
           "B_envelope": lambda c: CELLS[c]["envelope"],
           "C_sum_density": lambda c: CELLS[c]["aggregation"] == "sum_density"}


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    started = time.monotonic()
    result, pred, provenance, config, prep = [read(RUN / name) for name in [
        "results.json", "predictions.json", "provenance.json", "config.json", "preprocessing.json"]]
    deadline = started + config["audit_seconds"]
    cells, seeds = config["cells"], config["seeds"]
    assert result["status"] == "completed" and len(result["rows"]) == len(cells) * len(seeds)
    assert result["total_seconds"] < config["formal_seconds"]
    assert not result["heldout_evaluated"] and not provenance["heldout_labels_loaded"]
    for name, expected in provenance["source_sha256"].items():
        assert sha(ROOT / "scripts" / name) == expected, name
    assert sha(ROOT / "requirements-gnn-lock.txt") == provenance["environment_lock_sha256"]
    assert sha(ROOT / "configs/g3-ablation.json") == provenance["config_sha256"]
    assert config == read(ROOT / "configs/g3-ablation.json")
    assert sha(DATA / "manifest.json") == config["dataset_manifest_sha256"]
    assert sha(G3 / "manifest.json") == provenance["g3_graph_manifest_sha256"]
    admission = read(ROOT / "docs/reports/G3_ABLATION_CALIBRATION.json")
    assert admission["admitted"] and admission["config_sha256"] == provenance["config_sha256"]
    manifest = read(DATA / "manifest.json")
    # Hash heldout bytes for integrity without parsing labels or evaluating that split.
    for name, expected in manifest["split_sha256"].items():
        assert sha(DATA / f"{name}.json") == expected
    train, val = read(DATA / "train.json"), read(DATA / "validation.json")
    assert prep["fit_ids"] == [r["id"] for r in train] and pred["ids"] == [r["id"] for r in val]
    y = np.array([r["target"] for r in train]); yv = np.array([r["target"] for r in val])
    np.testing.assert_array_equal(yv, pred["target"])
    assert y.mean() == prep["target_mean"] and y.std() == prep["target_std"]
    g3_index = {r["id"]: r for r in read(G3 / "manifest.json")["records"]}
    graphs = {}
    for cutoff in (5.0, 8.0):
        loaded = []
        for row in train + val:
            record = row if cutoff == 5.0 else g3_index[row["id"]]
            path = ROOT / record["graph"]
            assert sha(path) == record["graph_sha256"]
            with np.load(path, allow_pickle=False) as graph:
                loaded.append({k: graph[k].copy() for k in ["z", "src", "dst", "distance", "cell"]})
        graphs[cutoff] = (loaded[:len(train)], loaded[len(train):])
        normalizers = fit_normalizers(graphs[cutoff][0])
        assert normalizers == prep["normalizers"][str(cutoff)]
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

    order = [(cell, seed) for turn, seed in enumerate(seeds) for cell in cells[turn:] + cells[:turn]]
    assert [(r["cell"], r["seed"]) for r in result["rows"]] == order
    checks = []
    for row in result["rows"]:
        cell, seed = row["cell"], row["seed"]
        cutoff = CELLS[cell]["cutoff"]
        path = RUN / f"{cell}-{seed}.pt"
        assert sha(path) == row["checkpoint_sha256"] and row["seconds"] < config["trial_seconds"]
        model = AblationMPNN(**CELLS[cell], **prep["normalizers"][str(cutoff)]).cuda().eval()
        model.load_state_dict(torch.load(path, weights_only=True, map_location="cuda"))
        assert sum(p.numel() for p in model.parameters()) == row["parameters"]
        vv, tt = replay(model, graphs[cutoff][1]), replay(model, graphs[cutoff][0])
        history = read(RUN / f"history-{cell}-{seed}.json")
        assert len(history) == row["epochs"] <= config["max_epochs"]
        assert min(history, key=lambda h: h["validation_mae"])["epoch"] == row["best_epoch"]
        assert min(h["validation_mae"] for h in history) == row["mae"]
        assert row["optimizer_updates"] == len(history) * int(np.ceil(len(train) / 16))
        stored = np.array(pred["models"][f"{cell}_{seed}"])
        np.testing.assert_allclose(vv, stored, rtol=1e-6, atol=1e-5)
        assert abs(np.abs(tt - y).mean() - row["train_mae"]) < 1e-6
        assert abs(np.abs(stored - yv).mean() - row["mae"]) < 1e-12
        assert float(spearmanr(stored, yv).statistic) == row["spearman"]
        checks.append({"cell": cell, "seed": seed, "max_abs_prediction_difference": float(abs(vv - stored).max())})
    g2, g3 = read(ROOT / "runs/g2-completed-01/predictions.json"), read(ROOT / "runs/g3-training-01/predictions.json")
    assert g2["ids"] == pred["ids"] == g3["ids"]
    anchors = {f"c5-hard-mean_{s}_vs_g2_gnn": float(np.max(np.abs(np.array(pred["models"][f"c5-hard-mean_{s}"]) - g2["models"][f"gnn_{s}"])))
               for s in seeds}
    anchors.update({f"c8-env-sum_{s}_vs_g3": float(np.max(np.abs(np.array(pred["models"][f"c8-env-sum_{s}"]) - g3["models"][f"g3_{s}"])))
                    for s in seeds})
    summary = {}
    for cell in cells:
        rows = [r for r in result["rows"] if r["cell"] == cell]
        summary[cell] = {metric: {"mean": float(np.mean([r[metric] for r in rows])),
                                  "seed_sd": float(np.std([r[metric] for r in rows], ddof=1))}
                         for metric in ["mae", "rmse", "train_mae", "spearman", "top_10_percent_recall",
                                        "epochs", "best_epoch", "seconds"]}
        summary[cell]["epoch_cap_reached"] = sum(r["epochs"] == config["max_epochs"] for r in rows)
    errors = {cell: np.mean([np.abs(np.array(pred["models"][f"{cell}_{s}"]) - yv) for s in seeds], axis=0) for cell in cells}
    family = np.array(pred["family"])
    groups = [np.flatnonzero(family == f) for f in sorted(set(family))]
    counts = np.array([len(g) for g in groups])
    rng = np.random.default_rng(20261021)
    samples = [rng.integers(0, len(groups), len(groups)) for _ in range(config["bootstrap_resamples"])]

    def describe(name, d):
        sums = np.array([d[g].sum() for g in groups])
        estimates = [sums[s].sum() / counts[s].sum() for s in samples]
        interval = np.quantile(estimates, [.025, .975]).tolist()
        return {"effect": name, "mae_difference": float(d.mean()), "family_bootstrap_95_percentile": interval,
                "resolved": bool(interval[0] > 0 or interval[1] < 0)}

    sign = {name: {c: 1 if test(c) else -1 for c in cells} for name, test in FACTORS.items()}
    effects = []
    for size in (1, 2, 3):
        for combo in itertools.combinations(FACTORS, size):
            d = sum(np.prod([sign[f][c] for f in combo]) * errors[c] for c in cells) / 4
            effects.append(describe(" x ".join(combo), d))
    paths = []
    for start, flips in [("c5-hard-mean", ["c8-hard-mean", "c5-env-mean", "c5-hard-sum"]),
                         ("c8-env-sum", ["c5-env-sum", "c8-hard-sum", "c8-env-mean"])]:
        for other in flips:
            paths.append(describe(f"{other}_minus_{start}", errors[other] - errors[start]))
    report = {"result": result, "summary": summary, "factorial_effects": effects, "single_change_paths": paths,
              "provenance": provenance, "calibration": admission,
              "audit": {"checkpoint_prediction_checks": checks, "anchor_max_prediction_differences": anchors,
                        "training_predictions_per_model": len(train), "validation_predictions_per_model": len(val),
                        "verified_graph_files": 2 * (len(train) + len(val)), "heldout_evaluated": False,
                        "seconds": time.monotonic() - started},
              "limitations": ["Same repeatedly used development split; early stopping uses validation.",
                              "Effects are conditional on the fixed 50-epoch budget.",
                              "Sum aggregation and the density readout are one factor and are not separated."]}
    assert report["audit"]["seconds"] < config["audit_seconds"]
    (ROOT / "docs/reports/G3_ABLATION_RESULT.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), constrained_layout=True)
    colors = ["#e59500" if c.startswith("c5") else "#c0392b" for c in cells]
    axes[0].bar(cells, [summary[c]["mae"]["mean"] for c in cells], color=colors,
                yerr=[summary[c]["mae"]["seed_sd"] for c in cells], capsize=4)
    axes[0].set_ylabel("Development MAE (mmol/g)")
    axes[0].set_title("Cell means and SD across three seeds (orange 5 A, red 8 A)")
    axes[0].tick_params(axis="x", rotation=30, labelsize=8)
    names = [e["effect"].replace("_cutoff_8A", "").replace("_envelope", "").replace("_sum_density", "") for e in effects]
    values = [e["mae_difference"] for e in effects]
    low = [e["mae_difference"] - e["family_bootstrap_95_percentile"][0] for e in effects]
    high = [e["family_bootstrap_95_percentile"][1] - e["mae_difference"] for e in effects]
    axes[1].barh(names[::-1], values[::-1], xerr=[low[::-1], high[::-1]], color="#247ba0", capsize=3)
    axes[1].axvline(0, color="black", linewidth=.8)
    axes[1].set_xlabel("Effect on development MAE (mmol/g); negative is better")
    axes[1].set_title("Factorial effects with family bootstrap 95% intervals")
    fig.suptitle("G3 ablation: A = 8 A cutoff, B = cosine envelope, C = sum aggregation + density")
    fig.savefig(ROOT / "docs/reports/G3_ABLATION.png", dpi=160)
    plt.close(fig)
    print(json.dumps({"mae": {c: summary[c]["mae"] for c in cells}, "effects": effects, "paths": paths,
                      "anchors": anchors, "audit_seconds": report["audit"]["seconds"]}, indent=1), flush=True)


if __name__ == "__main__":
    main()

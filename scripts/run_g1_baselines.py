"""Run bounded calibration or frozen development comparisons without heldout labels."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import subprocess
import time
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.ensemble import ExtraTreesRegressor

from g1_model import PeriodicMPNN, pack

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "datasets/processed/g1"


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_split(name, manifest):
    if name not in {"train", "validation"}:
        raise ValueError("Heldout labels are unavailable to the baseline runner")
    path = DATA / f"{name}.json"
    assert digest(path) == manifest["split_sha256"][name]
    rows = json.loads(path.read_text())
    assert set(x["id"] for x in rows) == set(manifest["split_ids"][name])
    graphs = []
    for row in rows:
        path = ROOT / row["graph"]
        assert digest(path) == row["graph_sha256"]
        with np.load(path, allow_pickle=False) as graph:
            graphs.append({key: graph[key].copy() for key in ["z", "src", "dst", "distance"]})
    return rows, graphs


def configure(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def predict(model, graphs, mean, std, deadline):
    model.eval()
    output = []
    with torch.no_grad():
        for start in range(0, len(graphs), 16):
            if time.monotonic() > deadline:
                raise TimeoutError("Prediction budget exceeded")
            y = model(pack(graphs[start:start + 16], "cuda")) * std + mean
            output.extend(y.cpu().tolist())
    return np.asarray(output)


def epoch(model, graphs, y, optimizer, rng, deadline):
    model.train()
    total = 0.0
    order = rng.permutation(len(graphs))
    for start in range(0, len(graphs), 16):
        if time.monotonic() > deadline:
            raise TimeoutError("Training budget exceeded")
        indices = order[start:start + 16]
        batch = pack([graphs[i] for i in indices], "cuda")
        target = torch.as_tensor(y[indices], dtype=torch.float32, device="cuda")
        optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.functional.mse_loss(model(batch), target)
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite training loss")
        loss.backward()
        optimizer.step()
        total += loss.item() * len(indices)
    torch.cuda.synchronize()
    return total / len(graphs)


def metrics(y, prediction, ids):
    if not np.isfinite(prediction).all():
        raise ValueError("Nonfinite predictions")
    n = max(1, int(np.ceil(len(y) * .1)))
    true_top = set(sorted(range(len(y)), key=lambda i: (-y[i], ids[i]))[:n])
    pred_top = set(sorted(range(len(y)), key=lambda i: (-prediction[i], ids[i]))[:n])
    rho = None if np.ptp(prediction) == 0 or np.ptp(y) == 0 else float(spearmanr(y, prediction).statistic)
    return {"mae": float(np.mean(np.abs(prediction - y))),
            "rmse": float(np.sqrt(np.mean((prediction - y) ** 2))),
            "spearman": rho, "top_10_percent_recall": len(true_top & pred_top) / n}


def run(args):
    if args.output.exists():
        raise FileExistsError("Refusing to overwrite an existing run")
    args.output.mkdir(parents=True)
    started = time.monotonic()
    manifest = json.loads((DATA / "manifest.json").read_text())
    if not manifest["training_admitted"]:
        raise ValueError("Dataset did not pass admission")
    if not torch.cuda.is_available():
        raise RuntimeError("The protocol requires the calibrated CUDA backend")
    train, graphs = load_split("train", manifest)
    y = np.array([x["target"] for x in train])
    provenance = {
        "dataset_manifest_sha256": digest(DATA / "manifest.json"),
        "root_commit": subprocess.check_output(["git", "-c", f"safe.directory={ROOT.as_posix()}", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "crystal_builder_commit": "df014447f0812dd4f55ed60359b71a981c3eeb60",
        "torch": torch.__version__, "cuda": torch.version.cuda, "device": torch.cuda.get_device_name(0),
        "source_sha256": {p.name: digest(p) for p in (ROOT / "scripts").glob("*g1*.py")},
        "environment_lock_sha256": digest(ROOT / "requirements-gnn-lock.txt"),
        "heldout_labels_read": False, "deterministic_algorithms": True,
    }
    save(args.output / "provenance.json", provenance)
    configure(17)
    torch.cuda.reset_peak_memory_stats()
    if args.mode == "calibrate":
        ordered = sorted(range(len(train)), key=lambda i: (train[i]["atoms"], train[i]["id"]))
        chosen = [ordered[i] for i in np.linspace(0, len(ordered) - 1, 128, dtype=int)]
        g = [graphs[i] for i in chosen]
        target = y[chosen]
        mean, std = float(target.mean()), float(target.std())
        if std <= 0:
            raise ValueError("Constant calibration target")
        model = PeriodicMPNN().cuda()
        optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.00001)
        rng = np.random.default_rng(17)
        times, losses = [], []
        deadline = started + 900
        for _ in range(3):
            begin = time.monotonic()
            losses.append(epoch(model, g, (target - mean) / std, optimizer, rng, deadline))
            times.append(time.monotonic() - begin)
        prediction = predict(model, g[:32], mean, std, deadline)
        result = {"status": "completed", "structures": 128, "epochs": 3, "seed": 17,
                  "ids": [train[i]["id"] for i in chosen], "epoch_seconds": times,
                  "training_mse_normalized": losses, "prediction_finite": bool(np.isfinite(prediction).all()),
                  "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated(),
                  "total_seconds": time.monotonic() - started,
                  "estimated_full_3_seed_50_epoch_seconds": max(times) * len(train) / 128 * 150}
        save(args.output / "calibration.json", result)
        print(json.dumps(result), flush=True)
        return
    config = json.loads(args.config.read_text())
    assert config["dataset_manifest_sha256"] == digest(DATA / "manifest.json")
    assert config["seeds"] == [17, 29, 43] and config["max_epochs"] == 50
    assert 0 < config["wall_seconds"] <= 10800
    save(args.output / "config.json", config)
    deadline = started + config["wall_seconds"]
    validation, val_graphs = load_split("validation", manifest)
    yval = np.array([x["target"] for x in validation])
    ids = [x["id"] for x in validation]
    x = np.array([r["geometry"] for r in train])
    xv = np.array([r["geometry"] for r in validation])
    if not np.isfinite(x).all() or not np.isfinite(xv).all():
        raise ValueError("Nonfinite geometry input")
    mean, std = float(y.mean()), float(y.std())
    if std <= 0:
        raise ValueError("Constant training target")
    result = {"status": "running", "unit": "mmol/g", "train_count": len(train),
              "validation_count": len(validation), "training_target_mean": mean,
              "training_target_std": std, "heldout_evaluated": False, "rows": []}
    predictions = {"ids": ids, "family": [r["family"] for r in validation], "target": yval.tolist(), "models": {}}
    constant = np.full(len(yval), mean)
    result["rows"].append({"model": "mean", "seed": None, "seconds": 0.0, **metrics(yval, constant, ids)})
    predictions["models"]["mean"] = constant.tolist()
    for seed in config["seeds"]:
        begin = time.monotonic()
        if begin >= deadline:
            raise TimeoutError("Stage budget exceeded")
        tree = ExtraTreesRegressor(n_estimators=300, min_samples_leaf=2, max_features=1.0, n_jobs=4, random_state=seed)
        tree.fit(x, y)
        pred = tree.predict(xv)
        if time.monotonic() >= deadline:
            raise TimeoutError("Stage budget exceeded during tree fit")
        result["rows"].append({"model": "extra_trees", "seed": seed, "seconds": time.monotonic() - begin, **metrics(yval, pred, ids)})
        predictions["models"][f"extra_trees_{seed}"] = pred.tolist()
        # The fitted tree is a local artifact and must never be loaded from an untrusted source.
        import joblib
        joblib.dump(tree, args.output / f"extra-trees-{seed}.joblib")
        save(args.output / "results.json", result)
        save(args.output / "predictions.json", predictions)
    for seed in config["seeds"]:
        configure(seed)
        torch.cuda.reset_peak_memory_stats()
        begin = time.monotonic()
        model = PeriodicMPNN().cuda()
        optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.00001)
        rng = np.random.default_rng(seed)
        best, best_epoch, stale, history = float("inf"), 0, 0, []
        checkpoint = args.output / f"gnn-{seed}.pt"
        for ep in range(1, config["max_epochs"] + 1):
            loss = epoch(model, graphs, (y - mean) / std, optimizer, rng, deadline)
            pred = predict(model, val_graphs, mean, std, deadline)
            score = float(np.abs(pred - yval).mean())
            history.append({"epoch": ep, "train_mse_normalized": loss, "validation_mae": score})
            if score < best:
                best, best_epoch, stale = score, ep, 0
                torch.save(model.state_dict(), checkpoint)
            else:
                stale += 1
            save(args.output / f"history-{seed}.json", history)
            print(f"seed={seed} epoch={ep} val_mae={score:.6f} elapsed={time.monotonic()-begin:.1f}s", flush=True)
            if stale >= 10:
                break
        model.load_state_dict(torch.load(checkpoint, weights_only=True, map_location="cuda"))
        pred = predict(model, val_graphs, mean, std, deadline)
        result["rows"].append({"model": "gnn", "seed": seed, "seconds": time.monotonic() - begin,
                               "epochs": len(history), "best_epoch": best_epoch,
                               "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated(),
                               "checkpoint_sha256": digest(checkpoint), **metrics(yval, pred, ids)})
        predictions["models"][f"gnn_{seed}"] = pred.tolist()
        save(args.output / "results.json", result)
        save(args.output / "predictions.json", predictions)
    result["status"] = "completed"
    result["total_seconds"] = time.monotonic() - started
    save(args.output / "results.json", result)
    print(json.dumps(result), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["calibrate", "compare"])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    args = parser.parse_args()
    if args.mode == "compare" and args.config is None:
        parser.error("compare requires --config")
    try:
        run(args)
    except Exception as error:
        if args.output.exists() and not isinstance(error, FileExistsError):
            save(args.output / "failure.json", {"error": str(error), "type": type(error).__name__})
        raise


if __name__ == "__main__":
    main()

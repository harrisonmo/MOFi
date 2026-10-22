"""Bounded four-arm representation comparison on the frozen G1 split."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import joblib
import numpy as np
import torch
from sklearn.ensemble import ExtraTreesRegressor

from g1_model import pack
from g2_models import GeometryScaler, make_model
from run_g1_baselines import ROOT, DATA, configure, digest, load_split, metrics, save


def forward(model, arm, graphs, geometry, indices):
    x = torch.as_tensor(geometry[indices], dtype=torch.float32, device="cuda")
    if arm == "mlp":
        return model(x)
    graph = pack([graphs[i] for i in indices], "cuda")
    return model(graph) if arm == "gnn" else model(graph, x)


def evaluate(model, arm, graphs, x, mean, std, deadline):
    model.eval()
    values = []
    with torch.no_grad():
        for start in range(0, len(x), 16):
            if time.monotonic() >= deadline:
                raise TimeoutError("Inference deadline exceeded")
            indices = np.arange(start, min(start + 16, len(x)))
            values.extend((forward(model, arm, graphs, x, indices) * std + mean).cpu().tolist())
    return np.asarray(values)


def train_epoch(model, arm, graphs, x, y, optimizer, rng, deadline):
    model.train()
    total = 0.
    order = rng.permutation(len(y))
    for start in range(0, len(y), 16):
        if time.monotonic() >= deadline:
            raise TimeoutError("Training deadline exceeded")
        ids = order[start:start + 16]
        target = torch.as_tensor(y[ids], dtype=torch.float32, device="cuda")
        optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.functional.mse_loss(forward(model, arm, graphs, x, ids), target)
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite loss")
        loss.backward()
        optimizer.step()
        total += loss.item() * len(ids)
    torch.cuda.synchronize()
    return total / len(y)


def validate_config(config):
    expected = {"seeds": [17, 29, 43], "arms": ["gnn", "hybrid", "mlp", "tree"],
                "calibration_seconds": 900, "formal_seconds": 1800, "neural_trial_seconds": 150,
                "tree_trial_seconds": 30, "audit_seconds": 300, "max_epochs": 50, "patience": 10,
                "batch_size": 16, "lr": .001, "weight_decay": .00001,
                "tree_estimators": 300, "tree_min_samples_leaf": 2,
                "heldout_evaluation": False, "hyperparameter_search_trials": 0}
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(f"Unregistered configuration: {key}")


def run(args):
    if args.output.exists():
        raise FileExistsError("Refusing to overwrite a run")
    args.output.mkdir(parents=True)
    started = time.monotonic()
    config = json.loads(args.config.read_text())
    validate_config(config)
    manifest = json.loads((DATA / "manifest.json").read_text())
    assert digest(DATA / "manifest.json") == config["dataset_manifest_sha256"]
    assert manifest["training_admitted"] and torch.cuda.is_available()
    if args.mode == "compare":
        admission = json.loads((ROOT / "docs/reports/G2_CALIBRATION.json").read_text())
        assert admission["admitted"] and admission["config_sha256"] == digest(args.config)
    sources = list((ROOT / "scripts").glob("*g2*.py")) + [ROOT / "scripts/g1_model.py", ROOT / "scripts/run_g1_baselines.py"]
    science = ROOT / "Crystal-Builder"
    rev = lambda path: subprocess.check_output(["git", "-c", f"safe.directory={path.as_posix()}", "rev-parse", "HEAD"], cwd=path, text=True).strip()
    provenance = {"root_commit": rev(ROOT), "crystal_builder_commit": rev(science),
                  "config_sha256": digest(args.config), "dataset_manifest_sha256": digest(DATA / "manifest.json"),
                  "environment_lock_sha256": digest(ROOT / "requirements-gnn-lock.txt"),
                  "source_sha256": {p.name: digest(p) for p in sources},
                  "torch": torch.__version__, "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0),
                  "heldout_labels_loaded": False}
    save(args.output / "provenance.json", provenance)
    save(args.output / "config.json", config)
    rows, graphs = load_split("train", manifest)
    if args.mode == "calibrate":
        ordered = sorted(range(len(rows)), key=lambda i: (rows[i]["atoms"], rows[i]["id"]))
        chosen = [ordered[i] for i in np.linspace(0, len(ordered) - 1, 128, dtype=int)]
        rows, graphs = [rows[i] for i in chosen], [graphs[i] for i in chosen]
    x = np.array([r["geometry"] for r in rows])
    y = np.array([r["target"] for r in rows])
    mean, std = float(y.mean()), float(y.std())
    assert std > 0
    scaler = GeometryScaler().fit(x)
    xs = scaler.transform(x)
    save(args.output / "preprocessing.json", {"geometry_mean": scaler.mean.tolist(), "geometry_scale": scaler.scale.tolist(),
                                             "target_mean": mean, "target_std": std, "fit_ids": [r["id"] for r in rows]})
    if args.mode == "calibrate":
        deadline = started + config["calibration_seconds"]
        results = []
        for arm in config["arms"]:
            configure(17)
            begin = time.monotonic()
            if arm == "tree":
                tree = ExtraTreesRegressor(n_estimators=300, min_samples_leaf=2, max_features=1., n_jobs=4, random_state=17).fit(x, y)
                assert np.isfinite(tree.predict(x)).all()
                results.append({"arm": arm, "seconds": time.monotonic() - begin})
                continue
            torch.cuda.reset_peak_memory_stats()
            model = make_model(arm).cuda()
            optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.00001)
            rng = np.random.default_rng(17)
            elapsed, losses = [], []
            for _ in range(3):
                ep = time.monotonic()
                losses.append(train_epoch(model, arm, graphs, xs, (y - mean) / std, optimizer, rng, deadline))
                elapsed.append(time.monotonic() - ep)
            assert np.isfinite(evaluate(model, arm, graphs, xs, mean, std, deadline)).all()
            results.append({"arm": arm, "epoch_seconds": elapsed, "losses": losses,
                            "parameters": sum(p.numel() for p in model.parameters()),
                            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                            "seconds": time.monotonic() - begin})
        duration = time.monotonic() - started
        estimate = sum(max(r["epoch_seconds"]) * 2092 / 128 * 150 for r in results if "epoch_seconds" in r)
        admitted = duration < 900 and estimate < 1800 and all(
            min(r["epoch_seconds"]) * 2092 / 128 * 50 < 150 for r in results if "epoch_seconds" in r)
        save(args.output / "calibration.json", {"admitted": admitted, "config_sha256": digest(args.config),
            "structures": 128, "seed": 17, "rows": results, "total_seconds": duration,
            "warmup_inclusive_formal_estimate_seconds": estimate, "heldout_evaluated": False})
        print(json.dumps({"admitted": admitted, "seconds": duration, "formal_estimate": estimate}), flush=True)
        return
    val, vg = load_split("validation", manifest)
    xv = np.array([r["geometry"] for r in val]); xvs = scaler.transform(xv)
    yv = np.array([r["target"] for r in val]); ids = [r["id"] for r in val]
    result = {"status": "running", "rows": [], "heldout_evaluated": False}
    predictions = {"ids": ids, "family": [r["family"] for r in val], "target": yv.tolist(), "models": {}}
    stage_deadline = started + config["formal_seconds"]
    for turn, seed in enumerate(config["seeds"]):
        arms = config["arms"][turn:] + config["arms"][:turn]
        for arm in arms:
            configure(seed)
            begin = time.monotonic()
            deadline = min(stage_deadline, begin + config["tree_trial_seconds" if arm == "tree" else "neural_trial_seconds"])
            if begin >= deadline:
                raise TimeoutError("Stage deadline exceeded")
            if arm == "tree":
                model = ExtraTreesRegressor(n_estimators=300, min_samples_leaf=2, max_features=1., n_jobs=4, random_state=seed).fit(x, y)
                pred, training_pred = model.predict(xv), model.predict(x)
                path = args.output / f"{arm}-{seed}.joblib"
                joblib.dump(model, path)
                details = {"epochs": None, "parameters": None}
            else:
                torch.cuda.reset_peak_memory_stats()
                model = make_model(arm).cuda()
                optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.00001)
                rng = np.random.default_rng(seed)
                best, stale, history, best_epoch = float("inf"), 0, [], 0
                path = args.output / f"{arm}-{seed}.pt"
                for ep in range(1, 51):
                    loss = train_epoch(model, arm, graphs, xs, (y - mean) / std, optimizer, rng, deadline)
                    pred = evaluate(model, arm, vg, xvs, mean, std, deadline)
                    score = float(np.abs(pred - yv).mean())
                    history.append({"epoch": ep, "train_mse_normalized": loss, "validation_mae": score})
                    if score < best:
                        best, stale, best_epoch = score, 0, ep
                        torch.save(model.state_dict(), path)
                    else:
                        stale += 1
                    save(args.output / f"history-{arm}-{seed}.json", history)
                    if ep % 10 == 0:
                        print(f"{arm} seed={seed} epoch={ep} mae={score:.6f}", flush=True)
                    if stale >= 10:
                        break
                model.load_state_dict(torch.load(path, map_location="cuda", weights_only=True))
                pred = evaluate(model, arm, vg, xvs, mean, std, deadline)
                training_pred = evaluate(model, arm, graphs, xs, mean, std, deadline)
                details = {"epochs": len(history), "best_epoch": best_epoch,
                           "optimizer_updates": len(history) * int(np.ceil(len(rows) / 16)),
                           "parameters": sum(p.numel() for p in model.parameters()),
                           "peak_allocated_bytes": torch.cuda.max_memory_allocated()}
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Trial deadline exceeded: {arm}-{seed}")
            result["rows"].append({"arm": arm, "seed": seed, "seconds": time.monotonic() - begin,
                                   "train_mae": float(np.abs(training_pred - y).mean()),
                                   "checkpoint_sha256": digest(path), **details, **metrics(yv, pred, ids)})
            predictions["models"][f"{arm}_{seed}"] = pred.tolist()
            save(args.output / "predictions.json", predictions)
            save(args.output / "results.json", result)
            print(f"Completed {arm}-{seed}: {result['rows'][-1]['mae']:.6f}", flush=True)
    result.update(status="completed", total_seconds=time.monotonic() - started)
    assert result["total_seconds"] < config["formal_seconds"]
    save(args.output / "results.json", result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["calibrate", "compare"])
    parser.add_argument("--config", type=Path, default=ROOT / "configs/g2-comparison.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        run(args)
    except Exception as error:
        if args.output.exists() and not isinstance(error, FileExistsError):
            save(args.output / "failure.json", {"error": str(error), "type": type(error).__name__})
        raise

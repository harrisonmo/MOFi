"""Bounded four-arm G4 comparison (two G3 controls, G3 + geometry, improved graph) on the frozen G1 split."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import torch

from g2_models import GeometryScaler
from g4_models import fit_normalizers, forward, make_model, pack
from run_g1_baselines import ROOT, DATA, configure, digest, metrics, save
from run_g3_training import G3, TRAINING_SIZE, load_split


def validate_config(config):
    expected = {"arms": ["g3", "g3_hard", "g4_hybrid", "g4_graph"], "cutoff": 8.0, "seeds": [17, 29, 43],
                "calibration_structures": 128, "calibration_epochs": 3, "calibration_seconds": 900,
                "formal_seconds": 7200, "trial_seconds": 1200, "audit_seconds": 600,
                "admission_overhead_factor": 1.25, "admission_warmup_epochs_excluded": 1,
                "max_peak_allocated_gib": 20, "max_epochs": 150, "patience": 10, "batch_size": 16,
                "lr": .001, "weight_decay": .00001, "heldout_evaluation": False,
                "hyperparameter_search_trials": 0}
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(f"Unregistered configuration: {key}")
    if not isinstance(config.get("g4_graph_envelope"), bool):
        raise ValueError("Unregistered configuration: g4_graph_envelope")


def evaluate(model, arm, graphs, geometry, mean, std, deadline):
    model.eval()
    values = []
    with torch.no_grad():
        for start in range(0, len(graphs), 16):
            if time.monotonic() >= deadline:
                raise TimeoutError("Inference deadline exceeded")
            x = torch.as_tensor(geometry[start:start + 16], dtype=torch.float32, device="cuda")
            values.extend((forward(model, arm, pack(graphs[start:start + 16], "cuda"), x) * std + mean).cpu().tolist())
    return np.asarray(values)


def train_epoch(model, arm, graphs, geometry, y, optimizer, rng, deadline):
    model.train()
    total = 0.
    order = rng.permutation(len(y))
    for start in range(0, len(y), 16):
        if time.monotonic() >= deadline:
            raise TimeoutError("Training deadline exceeded")
        ids = order[start:start + 16]
        target = torch.as_tensor(y[ids], dtype=torch.float32, device="cuda")
        x = torch.as_tensor(geometry[ids], dtype=torch.float32, device="cuda")
        optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.functional.mse_loss(forward(model, arm, pack([graphs[i] for i in ids], "cuda"), x), target)
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite loss")
        loss.backward()
        optimizer.step()
        total += loss.item() * len(ids)
    torch.cuda.synchronize()
    return total / len(y)


def build(arm, config, normalizers):
    return make_model(arm, normalizers, config["g4_graph_envelope"]).cuda()


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
    g3_manifest = json.loads((G3 / "manifest.json").read_text())
    assert g3_manifest["g1_manifest_sha256"] == config["dataset_manifest_sha256"] and g3_manifest["cutoff"] == config["cutoff"]
    index = {record["id"]: record for record in g3_manifest["records"]}
    if args.mode == "train":
        admission = json.loads((ROOT / "docs/reports/G4_CALIBRATION.json").read_text())
        assert admission["admitted"] and admission["config_sha256"] == digest(args.config)
    sources = list((ROOT / "scripts").glob("*g4*.py")) + [ROOT / "scripts/g3_model.py", ROOT / "scripts/g3_ablation_model.py",
                                                         ROOT / "scripts/g2_models.py",
                                                         ROOT / "scripts/run_g3_training.py", ROOT / "scripts/run_g1_baselines.py"]
    science = ROOT / "Crystal-Builder"
    rev = lambda path: subprocess.check_output(["git", "-c", f"safe.directory={path.as_posix()}", "rev-parse", "HEAD"], cwd=path, text=True).strip()
    save(args.output / "provenance.json", {
        "root_commit": rev(ROOT), "crystal_builder_commit": rev(science),
        "config_sha256": digest(args.config), "dataset_manifest_sha256": digest(DATA / "manifest.json"),
        "g3_graph_manifest_sha256": digest(G3 / "manifest.json"),
        "environment_lock_sha256": digest(ROOT / "requirements-gnn-lock.txt"),
        "source_sha256": {p.name: digest(p) for p in sources},
        "torch": torch.__version__, "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0),
        "heldout_labels_loaded": False})
    save(args.output / "config.json", config)
    rows, graphs, excluded = load_split("train", manifest, index)
    assert not excluded
    if args.mode == "calibrate":
        ordered = sorted(range(len(rows)), key=lambda i: (rows[i]["atoms"], rows[i]["id"]))
        chosen = [ordered[i] for i in np.linspace(0, len(ordered) - 1, config["calibration_structures"], dtype=int)]
        rows, graphs = [rows[i] for i in chosen], [graphs[i] for i in chosen]
    x = np.array([r["geometry"] for r in rows])
    y = np.array([r["target"] for r in rows])
    mean, std = float(y.mean()), float(y.std())
    assert std > 0
    scaler = GeometryScaler().fit(x)
    xs = scaler.transform(x)
    normalizers = fit_normalizers(graphs)
    save(args.output / "preprocessing.json", {**normalizers, "geometry_mean": scaler.mean.tolist(),
                                             "geometry_scale": scaler.scale.tolist(), "target_mean": mean,
                                             "target_std": std, "fit_ids": [r["id"] for r in rows]})
    if args.mode == "calibrate":
        deadline = started + config["calibration_seconds"]
        results = []
        for arm in config["arms"]:
            configure(17)
            torch.cuda.reset_peak_memory_stats()
            model = build(arm, config, normalizers)
            optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
            rng = np.random.default_rng(17)
            elapsed, losses = [], []
            for _ in range(config["calibration_epochs"]):
                begin = time.monotonic()
                losses.append(train_epoch(model, arm, graphs, xs, (y - mean) / std, optimizer, rng, deadline))
                elapsed.append(time.monotonic() - begin)
            assert np.isfinite(evaluate(model, arm, graphs, xs, mean, std, deadline)).all()
            steady = elapsed[config["admission_warmup_epochs_excluded"]:]
            results.append({"arm": arm, "epoch_seconds": elapsed, "losses": losses,
                            "parameters": sum(p.numel() for p in model.parameters()),
                            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                            "trial_estimate_seconds": max(steady) * TRAINING_SIZE / len(rows) * config["max_epochs"]
                            * config["admission_overhead_factor"]})
        duration = time.monotonic() - started
        formal_estimate = sum(r["trial_estimate_seconds"] for r in results) * len(config["seeds"])
        admitted = (duration < config["calibration_seconds"] and formal_estimate < config["formal_seconds"]
                    and all(r["trial_estimate_seconds"] < config["trial_seconds"]
                            and r["peak_allocated_bytes"] < config["max_peak_allocated_gib"] * 2 ** 30 for r in results))
        save(args.output / "calibration.json", {
            "admitted": admitted, "config_sha256": digest(args.config), "structures": len(rows), "seed": 17,
            "rows": results, "formal_estimate_seconds": formal_estimate, "total_seconds": duration,
            "heldout_evaluated": False})
        print(json.dumps({"admitted": admitted, "seconds": duration, "formal_estimate": formal_estimate}), flush=True)
        return
    val, vg, excluded = load_split("validation", manifest, index)
    assert not excluded
    xvs = scaler.transform(np.array([r["geometry"] for r in val]))
    yv = np.array([r["target"] for r in val]); ids = [r["id"] for r in val]
    result = {"status": "running", "rows": [], "heldout_evaluated": False}
    predictions = {"ids": ids, "family": [r["family"] for r in val], "target": yv.tolist(), "models": {}}
    stage_deadline = started + config["formal_seconds"]
    for turn, seed in enumerate(config["seeds"]):
        for arm in config["arms"][turn:] + config["arms"][:turn]:
            configure(seed)
            begin = time.monotonic()
            deadline = min(stage_deadline, begin + config["trial_seconds"])
            if begin >= deadline:
                raise TimeoutError("Stage deadline exceeded")
            torch.cuda.reset_peak_memory_stats()
            model = build(arm, config, normalizers)
            optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
            rng = np.random.default_rng(seed)
            best, stale, history, best_epoch = float("inf"), 0, [], 0
            path = args.output / f"{arm}-{seed}.pt"
            for ep in range(1, config["max_epochs"] + 1):
                loss = train_epoch(model, arm, graphs, xs, (y - mean) / std, optimizer, rng, deadline)
                pred = evaluate(model, arm, vg, xvs, mean, std, deadline)
                score = float(np.abs(pred - yv).mean())
                history.append({"epoch": ep, "train_mse_normalized": loss, "validation_mae": score,
                                "elapsed_seconds": time.monotonic() - begin})
                if score < best:
                    best, stale, best_epoch = score, 0, ep
                    torch.save(model.state_dict(), path)
                else:
                    stale += 1
                save(args.output / f"history-{arm}-{seed}.json", history)
                if stale >= config["patience"]:
                    break
            model.load_state_dict(torch.load(path, map_location="cuda", weights_only=True))
            pred = evaluate(model, arm, vg, xvs, mean, std, deadline)
            training_pred = evaluate(model, arm, graphs, xs, mean, std, deadline)
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Trial deadline exceeded: {arm}-{seed}")
            result["rows"].append({"arm": arm, "seed": seed, "seconds": time.monotonic() - begin,
                                   "epochs": len(history), "best_epoch": best_epoch,
                                   "optimizer_updates": len(history) * int(np.ceil(len(rows) / 16)),
                                   "parameters": sum(p.numel() for p in model.parameters()),
                                   "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                                   "train_mae": float(np.abs(training_pred - y).mean()),
                                   "checkpoint_sha256": digest(path), **metrics(yv, pred, ids)})
            predictions["models"][f"{arm}_{seed}"] = pred.tolist()
            save(args.output / "predictions.json", predictions)
            save(args.output / "results.json", result)
            print(f"Completed {arm}-{seed}: mae={result['rows'][-1]['mae']:.6f} epochs={len(history)} "
                  f"t={result['rows'][-1]['seconds']:.1f}s total={time.monotonic() - started:.0f}s", flush=True)
    result.update(status="completed", total_seconds=time.monotonic() - started)
    assert result["total_seconds"] < config["formal_seconds"]
    save(args.output / "results.json", result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["calibrate", "train"])
    parser.add_argument("--config", type=Path, default=ROOT / "configs/g4-comparison.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        run(args)
    except Exception as error:
        if args.output.exists() and not isinstance(error, FileExistsError):
            save(args.output / "failure.json", {"error": str(error), "type": type(error).__name__})
        raise

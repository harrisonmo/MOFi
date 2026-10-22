"""Bounded 2x2x2 factorial ablation of the G3 representation changes on the frozen G1 split."""
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

from g3_ablation_model import CELLS, AblationMPNN
from g3_model import fit_normalizers
from run_g1_baselines import ROOT, DATA, configure, digest, metrics, save
from run_g3_training import G3, TRAINING_SIZE, evaluate, load_split, train_epoch


def validate_config(config):
    expected = {"cells": list(CELLS), "width": 64, "layers": 3, "radial": 32, "seeds": [17, 29, 43],
                "calibration_structures": 128, "calibration_epochs": 3, "calibration_seconds": 900,
                "formal_seconds": 5400, "trial_seconds": 900, "audit_seconds": 600,
                "admission_overhead_factor": 1.25, "admission_warmup_epochs_excluded": 1,
                "max_peak_allocated_gib": 20, "max_epochs": 50, "patience": 10, "batch_size": 16,
                "lr": .001, "weight_decay": .00001, "bootstrap_resamples": 2000,
                "heldout_evaluation": False, "hyperparameter_search_trials": 0}
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(f"Unregistered configuration: {key}")


def graph_indices(manifest):
    """Map each cutoff to {id: graph record}; 5 A uses frozen G1 graphs, 8 A uses G3 graphs."""
    g1 = {}
    for name in ["train", "validation"]:
        path = DATA / f"{name}.json"
        assert digest(path) == manifest["split_sha256"][name]
        g1.update({r["id"]: {"graph": r["graph"], "graph_sha256": r["graph_sha256"]} for r in json.loads(path.read_text())})
    g3 = {r["id"]: r for r in json.loads((G3 / "manifest.json").read_text())["records"]}
    return {5.0: g1, 8.0: g3}


def make_model(cell, config, normalizers):
    return AblationMPNN(**CELLS[cell], **normalizers, width=config["width"], layers=config["layers"],
                        radial=config["radial"]).cuda()


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
    assert g3_manifest["g1_manifest_sha256"] == config["dataset_manifest_sha256"] and g3_manifest["cutoff"] == 8.0
    if args.mode == "train":
        admission = json.loads((ROOT / "docs/reports/G3_ABLATION_CALIBRATION.json").read_text())
        assert admission["admitted"] and admission["config_sha256"] == digest(args.config)
    sources = list((ROOT / "scripts").glob("*g3*.py")) + [ROOT / "scripts/run_g1_baselines.py"]
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
    indices = graph_indices(manifest)
    train, val = {}, {}
    for cutoff, index in indices.items():
        rows, graphs, excluded = load_split("train", manifest, index)
        assert not excluded
        train[cutoff] = graphs
        if args.mode == "train":
            val_rows, vg, excluded = load_split("validation", manifest, index)
            assert not excluded
            val[cutoff] = vg
    if args.mode == "calibrate":
        ordered = sorted(range(len(rows)), key=lambda i: (rows[i]["atoms"], rows[i]["id"]))
        chosen = [ordered[i] for i in np.linspace(0, len(ordered) - 1, config["calibration_structures"], dtype=int)]
        rows = [rows[i] for i in chosen]
        train = {cutoff: [graphs[i] for i in chosen] for cutoff, graphs in train.items()}
    y = np.array([r["target"] for r in rows])
    mean, std = float(y.mean()), float(y.std())
    assert std > 0
    normalizers = {cutoff: fit_normalizers(graphs) for cutoff, graphs in train.items()}
    save(args.output / "preprocessing.json", {"normalizers": {str(k): v for k, v in normalizers.items()},
                                             "target_mean": mean, "target_std": std,
                                             "fit_ids": [r["id"] for r in rows]})
    if args.mode == "calibrate":
        deadline = started + config["calibration_seconds"]
        results = []
        for cell in config["cells"]:
            cutoff = CELLS[cell]["cutoff"]
            configure(17)
            torch.cuda.reset_peak_memory_stats()
            model = make_model(cell, config, normalizers[cutoff])
            optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
            rng = np.random.default_rng(17)
            elapsed, losses = [], []
            for _ in range(config["calibration_epochs"]):
                begin = time.monotonic()
                losses.append(train_epoch(model, train[cutoff], (y - mean) / std, optimizer, rng, deadline))
                elapsed.append(time.monotonic() - begin)
            assert np.isfinite(evaluate(model, train[cutoff], mean, std, deadline)).all()
            steady = elapsed[config["admission_warmup_epochs_excluded"]:]
            results.append({"cell": cell, "epoch_seconds": elapsed, "losses": losses,
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
    yv = np.array([r["target"] for r in val_rows]); ids = [r["id"] for r in val_rows]
    result = {"status": "running", "rows": [], "heldout_evaluated": False}
    predictions = {"ids": ids, "family": [r["family"] for r in val_rows], "target": yv.tolist(), "models": {}}
    stage_deadline = started + config["formal_seconds"]
    for turn, seed in enumerate(config["seeds"]):
        for cell in config["cells"][turn:] + config["cells"][:turn]:
            cutoff = CELLS[cell]["cutoff"]
            configure(seed)
            begin = time.monotonic()
            deadline = min(stage_deadline, begin + config["trial_seconds"])
            if begin >= deadline:
                raise TimeoutError("Stage deadline exceeded")
            torch.cuda.reset_peak_memory_stats()
            model = make_model(cell, config, normalizers[cutoff])
            optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
            rng = np.random.default_rng(seed)
            best, stale, history, best_epoch = float("inf"), 0, [], 0
            path = args.output / f"{cell}-{seed}.pt"
            for ep in range(1, config["max_epochs"] + 1):
                loss = train_epoch(model, train[cutoff], (y - mean) / std, optimizer, rng, deadline)
                pred = evaluate(model, val[cutoff], mean, std, deadline)
                score = float(np.abs(pred - yv).mean())
                history.append({"epoch": ep, "train_mse_normalized": loss, "validation_mae": score,
                                "elapsed_seconds": time.monotonic() - begin})
                if score < best:
                    best, stale, best_epoch = score, 0, ep
                    torch.save(model.state_dict(), path)
                else:
                    stale += 1
                save(args.output / f"history-{cell}-{seed}.json", history)
                if stale >= config["patience"]:
                    break
            model.load_state_dict(torch.load(path, map_location="cuda", weights_only=True))
            pred = evaluate(model, val[cutoff], mean, std, deadline)
            training_pred = evaluate(model, train[cutoff], mean, std, deadline)
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Trial deadline exceeded: {cell}-{seed}")
            result["rows"].append({"cell": cell, "seed": seed, **CELLS[cell], "seconds": time.monotonic() - begin,
                                   "epochs": len(history), "best_epoch": best_epoch,
                                   "optimizer_updates": len(history) * int(np.ceil(len(rows) / 16)),
                                   "parameters": sum(p.numel() for p in model.parameters()),
                                   "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                                   "train_mae": float(np.abs(training_pred - y).mean()),
                                   "checkpoint_sha256": digest(path), **metrics(yv, pred, ids)})
            predictions["models"][f"{cell}_{seed}"] = pred.tolist()
            save(args.output / "predictions.json", predictions)
            save(args.output / "results.json", result)
            print(f"Completed {cell}-{seed}: mae={result['rows'][-1]['mae']:.6f} epochs={len(history)} "
                  f"t={result['rows'][-1]['seconds']:.1f}s total={time.monotonic() - started:.0f}s", flush=True)
    result.update(status="completed", total_seconds=time.monotonic() - started)
    assert result["total_seconds"] < config["formal_seconds"]
    save(args.output / "results.json", result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["calibrate", "train"])
    parser.add_argument("--config", type=Path, default=ROOT / "configs/g3-ablation.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        run(args)
    except Exception as error:
        if args.output.exists() and not isinstance(error, FileExistsError):
            save(args.output / "failure.json", {"error": str(error), "type": type(error).__name__})
        raise

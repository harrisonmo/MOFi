"""Bounded three-seed training of the G3 smooth long-range encoder on the frozen G1 split."""
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

from g3_model import SmoothPeriodicMPNN, fit_normalizers, pack
from run_g1_baselines import ROOT, DATA, configure, digest, metrics, save

G3 = ROOT / "datasets/processed/g3"
TRAINING_SIZE = 2092


def validate_config(config):
    expected = {"model": "SmoothPeriodicMPNN", "cutoff": 8.0, "width": 64, "layers": 3, "radial": 32,
                "seeds": [17, 29, 43], "calibration_structures": 128, "calibration_epochs": 3,
                "calibration_seconds": 900, "formal_seconds": 3600, "trial_seconds": 900,
                "audit_seconds": 300, "admission_overhead_factor": 1.25, "admission_warmup_epochs_excluded": 1,
                "max_peak_allocated_gib": 20,
                "max_epochs": 50, "patience": 10, "batch_size": 16, "lr": .001, "weight_decay": .00001,
                "heldout_evaluation": False, "hyperparameter_search_trials": 0}
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(f"Unregistered configuration: {key}")


def load_split(name, manifest, graph_index):
    """Return G1 label rows with G3 graphs, plus IDs that G3 admission excluded."""
    if name not in {"train", "validation"}:
        raise ValueError("Heldout labels are unavailable to the G3 runner")
    path = DATA / f"{name}.json"
    assert digest(path) == manifest["split_sha256"][name]
    rows = json.loads(path.read_text())
    assert set(x["id"] for x in rows) == set(manifest["split_ids"][name])
    kept, graphs, excluded = [], [], []
    for row in rows:
        record = graph_index.get(row["id"])
        if record is None:
            excluded.append(row["id"])
            continue
        path = ROOT / record["graph"]
        assert digest(path) == record["graph_sha256"]
        with np.load(path, allow_pickle=False) as graph:
            graphs.append({key: graph[key].copy() for key in ["z", "src", "dst", "distance", "cell"]})
        kept.append(row)
    return kept, graphs, excluded


def make_model(config, normalizers):
    return SmoothPeriodicMPNN(**normalizers, cutoff=config["cutoff"], width=config["width"],
                              layers=config["layers"], radial=config["radial"]).cuda()


def evaluate(model, graphs, mean, std, deadline):
    model.eval()
    values = []
    with torch.no_grad():
        for start in range(0, len(graphs), 16):
            if time.monotonic() >= deadline:
                raise TimeoutError("Inference deadline exceeded")
            values.extend((model(pack(graphs[start:start + 16], "cuda")) * std + mean).cpu().tolist())
    return np.asarray(values)


def train_epoch(model, graphs, y, optimizer, rng, deadline):
    model.train()
    total = 0.
    order = rng.permutation(len(y))
    for start in range(0, len(y), 16):
        if time.monotonic() >= deadline:
            raise TimeoutError("Training deadline exceeded")
        ids = order[start:start + 16]
        target = torch.as_tensor(y[ids], dtype=torch.float32, device="cuda")
        optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.functional.mse_loss(model(pack([graphs[i] for i in ids], "cuda")), target)
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite loss")
        loss.backward()
        optimizer.step()
        total += loss.item() * len(ids)
    torch.cuda.synchronize()
    return total / len(y)


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
    assert g3_manifest["g1_manifest_sha256"] == config["dataset_manifest_sha256"]
    assert g3_manifest["cutoff"] == config["cutoff"]
    graph_index = {record["id"]: record for record in g3_manifest["records"]}
    if args.mode == "train":
        admission = json.loads((ROOT / "docs/reports/G3_CALIBRATION.json").read_text())
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
    rows, graphs, train_excluded = load_split("train", manifest, graph_index)
    if args.mode == "calibrate":
        ordered = sorted(range(len(rows)), key=lambda i: (rows[i]["atoms"], rows[i]["id"]))
        chosen = [ordered[i] for i in np.linspace(0, len(ordered) - 1, config["calibration_structures"], dtype=int)]
        rows, graphs = [rows[i] for i in chosen], [graphs[i] for i in chosen]
    y = np.array([r["target"] for r in rows])
    mean, std = float(y.mean()), float(y.std())
    assert std > 0
    normalizers = fit_normalizers(graphs)
    save(args.output / "preprocessing.json", {**normalizers, "target_mean": mean, "target_std": std,
                                             "fit_ids": [r["id"] for r in rows], "train_excluded": train_excluded})
    if args.mode == "calibrate":
        deadline = started + config["calibration_seconds"]
        configure(17)
        torch.cuda.reset_peak_memory_stats()
        model = make_model(config, normalizers)
        optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
        rng = np.random.default_rng(17)
        elapsed, losses = [], []
        for _ in range(config["calibration_epochs"]):
            begin = time.monotonic()
            losses.append(train_epoch(model, graphs, (y - mean) / std, optimizer, rng, deadline))
            elapsed.append(time.monotonic() - begin)
        assert np.isfinite(evaluate(model, graphs, mean, std, deadline)).all()
        peak = torch.cuda.max_memory_allocated()
        # Amendment 1: the first epoch carries one-off CUDA initialization and is not extrapolated.
        steady = elapsed[config["admission_warmup_epochs_excluded"]:]
        trial_estimate = max(steady) * TRAINING_SIZE / len(rows) * config["max_epochs"] * config["admission_overhead_factor"]
        duration = time.monotonic() - started
        admitted = (duration < config["calibration_seconds"] and trial_estimate < config["trial_seconds"]
                    and trial_estimate * len(config["seeds"]) < config["formal_seconds"]
                    and peak < config["max_peak_allocated_gib"] * 2 ** 30)
        save(args.output / "calibration.json", {
            "admitted": admitted, "config_sha256": digest(args.config), "structures": len(rows), "seed": 17,
            "epoch_seconds": elapsed, "losses": losses, "parameters": sum(p.numel() for p in model.parameters()),
            "peak_allocated_bytes": peak, "trial_estimate_seconds": trial_estimate,
            "formal_estimate_seconds": trial_estimate * len(config["seeds"]), "total_seconds": duration,
            "train_excluded": train_excluded, "heldout_evaluated": False})
        print(json.dumps({"admitted": admitted, "seconds": duration, "trial_estimate": trial_estimate,
                          "peak_gib": peak / 2 ** 30}), flush=True)
        return
    val, vg, validation_excluded = load_split("validation", manifest, graph_index)
    yv = np.array([r["target"] for r in val]); ids = [r["id"] for r in val]
    result = {"status": "running", "rows": [], "train_excluded": train_excluded,
              "validation_excluded": validation_excluded, "heldout_evaluated": False}
    predictions = {"ids": ids, "family": [r["family"] for r in val], "target": yv.tolist(), "models": {}}
    stage_deadline = started + config["formal_seconds"]
    for seed in config["seeds"]:
        configure(seed)
        begin = time.monotonic()
        deadline = min(stage_deadline, begin + config["trial_seconds"])
        if begin >= deadline:
            raise TimeoutError("Stage deadline exceeded")
        torch.cuda.reset_peak_memory_stats()
        model = make_model(config, normalizers)
        optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
        rng = np.random.default_rng(seed)
        best, stale, history, best_epoch = float("inf"), 0, [], 0
        path = args.output / f"g3-{seed}.pt"
        for ep in range(1, config["max_epochs"] + 1):
            loss = train_epoch(model, graphs, (y - mean) / std, optimizer, rng, deadline)
            pred = evaluate(model, vg, mean, std, deadline)
            score = float(np.abs(pred - yv).mean())
            history.append({"epoch": ep, "train_mse_normalized": loss, "validation_mae": score,
                            "elapsed_seconds": time.monotonic() - begin})
            if score < best:
                best, stale, best_epoch = score, 0, ep
                torch.save(model.state_dict(), path)
            else:
                stale += 1
            save(args.output / f"history-g3-{seed}.json", history)
            print(f"g3 seed={seed} epoch={ep} mae={score:.6f} t={time.monotonic() - begin:.1f}s", flush=True)
            if stale >= config["patience"]:
                break
        model.load_state_dict(torch.load(path, map_location="cuda", weights_only=True))
        pred = evaluate(model, vg, mean, std, deadline)
        training_pred = evaluate(model, graphs, mean, std, deadline)
        if time.monotonic() >= deadline:
            raise TimeoutError(f"Trial deadline exceeded: g3-{seed}")
        result["rows"].append({"arm": "g3", "seed": seed, "seconds": time.monotonic() - begin,
                               "epochs": len(history), "best_epoch": best_epoch,
                               "optimizer_updates": len(history) * int(np.ceil(len(rows) / 16)),
                               "parameters": sum(p.numel() for p in model.parameters()),
                               "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                               "train_mae": float(np.abs(training_pred - y).mean()),
                               "checkpoint_sha256": digest(path), **metrics(yv, pred, ids)})
        predictions["models"][f"g3_{seed}"] = pred.tolist()
        save(args.output / "predictions.json", predictions)
        save(args.output / "results.json", result)
        print(f"Completed g3-{seed}: {result['rows'][-1]['mae']:.6f}", flush=True)
    result.update(status="completed", total_seconds=time.monotonic() - started)
    assert result["total_seconds"] < config["formal_seconds"]
    save(args.output / "results.json", result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["calibrate", "train"])
    parser.add_argument("--config", type=Path, default=ROOT / "configs/g3-training.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        run(args)
    except Exception as error:
        if args.output.exists() and not isinstance(error, FileExistsError):
            save(args.output / "failure.json", {"error": str(error), "type": type(error).__name__})
        raise

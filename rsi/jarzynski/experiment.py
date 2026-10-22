"""GPU-only, bounded comparison of supervised GNN and FEAT free energies.

Usage (on a CUDA machine, after running the targeted tests):
  python -m rsi.jarzynski.experiment --phase calibrate --output runs/qiankun-calibration
  python -m rsi.jarzynski.experiment --phase run --output runs/qiankun-run \
      --admission runs/qiankun-calibration/calibration.json

There is intentionally no automatic CPU fallback. Calibration and formal runs
are separate, never overwrite previous evidence, and record failed attempts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import subprocess
import time
from pathlib import Path

import torch

from .estimators import effective_sample_size, jarzynski, simulate_forward
from .models import FreeEnergyGNN, TransportGNN, interpolant_loss
from .systems import exact_delta_f, graph_hash, make_graphs, precision, sample_endpoint

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs" / "qiankun-jarzynski.json"
ARM_NAMES = ("supervised_gnn", "feat_gnn_jarzynski", "feat_gnn_mean_work", "no_transport_jarzynski")


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def source_manifest():
    paths = sorted((ROOT / "rsi" / "jarzynski").glob("*.py"))
    paths += sorted((ROOT / "tests").glob("test_jarzynski*.py"))
    paths.append(ROOT / "scripts" / "report_qiankun_jarzynski.py")
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def git_state():
    def read(*args):
        return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()
    return {"commit": read("rev-parse", "HEAD"), "branch": read("branch", "--show-current"),
            "status": read("status", "--porcelain"), "crystal_builder": "not used"}


def validate_config(config):
    if config.get("required_device") != "cuda":
        raise ValueError("This experiment requires CUDA; CPU fallback is disabled")
    if type(config.get("data_seed")) is not int or config["data_seed"] < 0:
        raise ValueError("data_seed must be a nonnegative integer")
    positive_ints = ("train_graphs", "validation_graphs", "test_graphs", "ood_graphs", "nodes", "ood_nodes",
                     "endpoint_samples_per_state_per_training_graph", "width", "layers", "updates", "batch_size",
                     "inference_steps", "paths_per_graph", "inference_chunk_graphs", "threads")
    for key in positive_ints:
        if type(config.get(key)) is not int or config[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    for key in ("beta", "learning_rate", "gradient_clip", "gamma_scale", "diffusion", "calibration_seconds_limit",
                "training_seconds_per_arm_limit", "inference_seconds_per_seed_limit", "total_seconds_limit"):
        value = config.get(key)
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"{key} must be finite and positive")
    if config["spatial_dim"] != 3 or config["beta"] != 1.0:
        raise ValueError("This registered benchmark uses three dimensions and beta=1")
    if config["nodes"] < 2 or config["ood_nodes"] <= config["nodes"]:
        raise ValueError("OOD must use more nodes than the training graphs")
    if not 0 < config["training_time_min"] < 0.5:
        raise ValueError("training_time_min must be between zero and one half")
    seeds = config.get("seeds")
    if not isinstance(seeds, list) or not seeds or any(type(s) is not int or s < 0 for s in seeds) or len(set(seeds)) != len(seeds):
        raise ValueError("seeds must be distinct nonnegative integers")
    if tuple(config.get("arms", ())) != ARM_NAMES:
        raise ValueError("The registered comparison requires all four arms")
    # A modified budget must be explicitly re-registered in code/protocol.
    if config["updates"] > 5000 or config["paths_per_graph"] > 4096 or config["inference_steps"] > 512:
        raise ValueError("Requested computation exceeds this bounded runner's admitted size")
    if config["total_seconds_limit"] > 7200:
        raise ValueError("Total run budget cannot exceed 7200 seconds")
    limits = {"train_graphs": 4096, "validation_graphs": 1024, "test_graphs": 1024,
              "ood_graphs": 1024, "nodes": 16, "ood_nodes": 32, "width": 128,
              "layers": 6, "batch_size": 1024, "threads": 32,
              "endpoint_samples_per_state_per_training_graph": 512, "inference_chunk_graphs": 16}
    if any(config[key] > limit for key, limit in limits.items()) or len(seeds) > 10:
        raise ValueError("Configuration exceeds this runner's bounded model/data size")
    pool_size = config["train_graphs"] * config["endpoint_samples_per_state_per_training_graph"] * config["nodes"] ** 2
    inference_size = config["inference_chunk_graphs"] * config["paths_per_graph"] * config["ood_nodes"] ** 2 * config["width"]
    if pool_size > 8_000_000 or inference_size > 16_000_000:
        raise ValueError("Configuration exceeds the bounded tensor allocation limit")


def require_cuda():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required. No CPU experiment was started; run later on a GPU machine.")
    device = torch.device("cuda:0")
    return device


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def check_deadline(deadline, context):
    if time.monotonic() > deadline:
        raise TimeoutError(f"Budget exhausted during {context}; preserve this attempt and use a new output directory")


def move_graphs(graphs, device, dtype):
    return {key: value.to(device=device, dtype=dtype) for key, value in graphs.items()}


def index_graphs(graphs, indices):
    # These tensors came from a validated generated split; avoid GPU host-sync
    # validation at every optimization step.
    return {key: value[indices] for key, value in graphs.items()}


def tensor_hash(tensor):
    array = tensor.detach().cpu().contiguous().numpy()
    return hashlib.sha256(array.tobytes()).hexdigest()


def training_pool(graphs, samples, seed, beta):
    """Generate the finite endpoint pool on CPU before either model is trained."""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    expanded = {k: v.repeat_interleave(samples, dim=0) for k, v in graphs.items()}
    count, nodes = graphs["a0"].shape
    x0 = sample_endpoint(expanded, 0, generator, beta).reshape(count, samples, nodes, 3)
    x1 = sample_endpoint(expanded, 1, generator, beta).reshape(count, samples, nodes, 3)
    return x0, x1


def train_arm(arm, train_cpu, pool_cpu, config, seed, output, device, total_deadline, updates=None):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    generator = torch.Generator(device=device).manual_seed(seed + 104729)
    graph_generator = torch.Generator(device=device).manual_seed(seed + 65537)
    model_cls = FreeEnergyGNN if arm == "supervised_gnn" else TransportGNN
    model = model_cls(config["width"], config["layers"]).to(device)
    graph = move_graphs(train_cpu, device, torch.float32)
    if arm == "supervised_gnn":
        labels = exact_delta_f(train_cpu, beta=config["beta"]).to(device=device, dtype=torch.float32)
        pool = None
    else:
        labels = None
        pool = tuple(x.to(device=device, dtype=torch.float32) for x in pool_cpu)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    updates = config["updates"] if updates is None else updates
    started = time.monotonic()
    deadline = min(total_deadline, started + config["training_seconds_per_arm_limit"])
    trace = []
    model.train()
    for step in range(updates):
        check_deadline(deadline, f"{arm} training")
        indices = torch.randint(len(train_cpu["a0"]), (config["batch_size"],), device=device, generator=graph_generator)
        batch = index_graphs(graph, indices)
        optimizer.zero_grad(set_to_none=True)
        if labels is not None:
            loss = (model(batch) - labels[indices]).square().mean()
            components = {}
        else:
            sample_count = pool[0].shape[1]
            left = torch.randint(sample_count, (len(indices),), device=device, generator=generator)
            right = torch.randint(sample_count, (len(indices),), device=device, generator=generator)
            x0, x1 = pool[0][indices, left], pool[1][indices, right]
            t = torch.rand(len(indices), device=device, generator=generator)
            t = config["training_time_min"] + (1 - 2 * config["training_time_min"]) * t
            noise = torch.randn(x0.shape, device=device, dtype=x0.dtype, generator=generator)
            loss, components = interpolant_loss(model, batch, x0, x1, t, noise, config["gamma_scale"])
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError(f"Nonfinite {arm} loss at step {step}")
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config["gradient_clip"], error_if_nonfinite=True)
        optimizer.step()
        if step == 0 or (step + 1) % 100 == 0 or step + 1 == updates:
            row = {"step": step + 1, "loss": float(loss.detach()), "gradient_norm": float(norm),
                   **{key: float(value) for key, value in components.items()}}
            trace.append(row)
            write_json(output / f"{arm}-trace.json", trace)
    synchronize(device)
    check_deadline(deadline, f"{arm} training completion")
    seconds = time.monotonic() - started
    checkpoint = output / f"{arm}.pt"
    torch.save({"state_dict": model.state_dict(), "config": config, "seed": seed, "arm": arm,
                "updates": updates}, checkpoint)
    write_json(output / f"{arm}-trace.json", trace)
    return model.eval(), {"seconds": seconds, "updates": updates,
                          "parameters": sum(p.numel() for p in model.parameters()),
                          "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                          "supervised_free_energy_labels": len(train_cpu["a0"]) if labels is not None else 0,
                          "endpoint_training_configurations": 0 if labels is not None else 2 * pool_cpu[0].shape[0] * pool_cpu[0].shape[1]}


@torch.no_grad()
def evaluate_split(supervised, transport, graphs_cpu, split, config, seed, output, device, deadline):
    """Evaluate both networks after training; references never enter FEAT inference."""
    synchronize(device)
    started = time.monotonic()
    graph32 = move_graphs(graphs_cpu, device, torch.float32)
    predictions = supervised(graph32).to(torch.float64).cpu()
    synchronize(device)
    times = {"supervised_gnn": time.monotonic() - started, "feat_gnn_jarzynski": 0.0,
             "feat_gnn_mean_work": 0.0, "no_transport_jarzynski": 0.0}
    reference = exact_delta_f(graphs_cpu, beta=config["beta"])
    records, saved_work = [], []
    paths = config["paths_per_graph"]
    for start in range(0, len(graphs_cpu["a0"]), config["inference_chunk_graphs"]):
        check_deadline(deadline, f"{split} inference")
        synchronize(device)
        preparation_started = time.monotonic()
        stop = min(start + config["inference_chunk_graphs"], len(graphs_cpu["a0"]))
        chunk_cpu = {k: v[start:stop].repeat_interleave(paths, dim=0) for k, v in graphs_cpu.items()}
        endpoint_seed = seed * 1000003 + start + (10000 if split == "ood" else 0)
        initial = sample_endpoint(chunk_cpu, 0, torch.Generator().manual_seed(endpoint_seed), config["beta"])
        graph64 = move_graphs(chunk_cpu, device, torch.float64)
        x0 = initial.to(device)
        synchronize(device)
        preparation_seconds = time.monotonic() - preparation_started

        def learned_drift(x, t, unused_graph):
            velocity, score = transport(x.float(), t.float(), graph_model)
            return velocity.to(x.dtype), score.to(x.dtype)

        def physical_drift(x, t, unused_graph):
            time_axis = t[:, None, None]
            score = -config["beta"] * (((1 - time_axis) * k0 + time_axis * k1) @ x)
            return torch.zeros_like(x), score

        work, estimates, ess_values = {}, {}, {}
        for arm, drift in (("feat_gnn_jarzynski", learned_drift), ("no_transport_jarzynski", physical_drift)):
            # Reset the noise generator: both stochastic arms share all normals.
            generator = torch.Generator(device=device).manual_seed(endpoint_seed + 99991)
            synchronize(device)
            arm_started = time.monotonic()
            if arm == "feat_gnn_jarzynski":
                graph_model = move_graphs(chunk_cpu, device, torch.float32)
            else:
                k0, k1 = precision(graph64, 0), precision(graph64, 1)
            result = simulate_forward(graph64, x0, drift, config["inference_steps"], config["diffusion"],
                                      generator, beta=config["beta"])
            synchronize(device)
            work[arm] = result["work"].cpu().reshape(stop - start, paths)
            estimates[arm] = jarzynski(work[arm], config["beta"])
            ess_values[arm] = effective_sample_size(work[arm], config["beta"])
            if arm == "feat_gnn_jarzynski":
                estimates["feat_gnn_mean_work"] = work[arm].mean(-1)
            # Charge the shared endpoint sample/transfer cost to each
            # stochastic arm's standalone inference cost.
            times[arm] += preparation_seconds + time.monotonic() - arm_started
            check_deadline(deadline, f"{split}/{arm} inference")
        saved_work.append(work)
        values = estimates
        feat_ess = ess_values["feat_gnn_jarzynski"]
        plain_ess = ess_values["no_transport_jarzynski"]
        for local, graph_id in enumerate(range(start, stop)):
            records.append({"seed": seed, "split": split, "graph_id": graph_id,
                            "paths": paths,
                            "truth": float(reference[graph_id]),
                            "predictions": {"supervised_gnn": float(predictions[graph_id]),
                                            **{arm: float(value[local]) for arm, value in values.items()}},
                            "ess": {"feat_gnn_jarzynski": float(feat_ess[local]),
                                    "no_transport_jarzynski": float(plain_ess[local])}})
        write_json(output / f"{split}-predictions-partial.json", records)
    # Mean-work reuses every FEAT path; it is charged the same standalone
    # inference cost and must not be represented as a free separate model.
    times["feat_gnn_mean_work"] = times["feat_gnn_jarzynski"]
    torch.save(saved_work, output / f"{split}-work.pt")
    return records, times


def metrics(records):
    result = {}
    for split in sorted({r["split"] for r in records}):
        rows = [r for r in records if r["split"] == split]
        result[split] = {}
        for arm in ARM_NAMES:
            errors = torch.tensor([r["predictions"][arm] - r["truth"] for r in rows], dtype=torch.float64)
            result[split][arm] = {"mae": float(errors.abs().mean()), "rmse": float(errors.square().mean().sqrt()),
                                  "signed_bias": float(errors.mean()), "graphs": len(rows)}
            if arm in rows[0]["ess"]:
                ess = torch.tensor([r["ess"][arm] for r in rows], dtype=torch.float64)
                result[split][arm].update(ess_mean=float(ess.mean()), ess_min=float(ess.min()),
                                         ess_fraction_mean=sum(r["ess"][arm] / r["paths"] for r in rows) / len(rows),
                                         low_ess_graphs=sum(r["ess"][arm] < 0.1 * r["paths"] for r in rows))
    return result


def run_calibration(config, output, device, deadline):
    train = make_graphs(8, config["nodes"], 991)
    pool = training_pool(train, 16, 992, config["beta"])
    supervised, sup_info = train_arm("supervised_gnn", train, pool, config, 997, output, device, deadline, updates=20)
    transport, feat_info = train_arm("feat_gnn", train, pool, config, 997, output, device, deadline, updates=20)
    small_config = {**config, "paths_per_graph": 32, "inference_steps": 4, "inference_chunk_graphs": 2}
    graphs = make_graphs(2, config["nodes"], 998)
    records, timings = evaluate_split(supervised, transport, graphs, "calibration", small_config, 997, output, device, deadline)
    # Admission checks execution/numerics only, never a performance threshold.
    return {"admitted": True, "training": {"supervised": sup_info, "feat": feat_info},
            "inference_seconds": timings, "finite_records": len(records)}


def run_formal(config, output, device, deadline):
    graphs = {
        "train": make_graphs(config["train_graphs"], config["nodes"], config["data_seed"]),
        "validation": make_graphs(config["validation_graphs"], config["nodes"], config["data_seed"] + 1),
        "test": make_graphs(config["test_graphs"], config["nodes"], config["data_seed"] + 2),
        "ood": make_graphs(config["ood_graphs"], config["ood_nodes"], config["data_seed"] + 3, ood=True),
    }
    pool = training_pool(graphs["train"], config["endpoint_samples_per_state_per_training_graph"], config["data_seed"] + 4, config["beta"])
    manifest = {"splits": {k: {"sha256": graph_hash(g), "graphs": len(g["a0"]), "nodes": g["a0"].shape[1]} for k, g in graphs.items()},
                "training_endpoint_hashes": [tensor_hash(x) for x in pool], "validation_use": "Reserved; no selection on it in this fixed-checkpoint protocol"}
    write_json(output / "data-manifest.json", manifest)
    torch.save({"graphs": graphs, "training_endpoints": pool}, output / "data.pt")
    all_records, seeds = [], []
    for seed in config["seeds"]:
        check_deadline(deadline, "seed start")
        seed_dir = output / f"seed-{seed}"
        seed_dir.mkdir()
        supervised, sup_info = train_arm("supervised_gnn", graphs["train"], pool, config, seed, seed_dir, device, deadline)
        transport, feat_info = train_arm("feat_gnn", graphs["train"], pool, config, seed, seed_dir, device, deadline)
        inference_deadline = min(deadline, time.monotonic() + config["inference_seconds_per_seed_limit"])
        records, timings = [], {}
        for split in ("test", "ood"):
            rows, seconds = evaluate_split(supervised, transport, graphs[split], split, config, seed, seed_dir, device, inference_deadline)
            records.extend(rows)
            timings[split] = seconds
        seed_result = {"seed": seed, "training": {"supervised_gnn": sup_info, "feat_gnn": feat_info},
                       "inference_seconds": timings, "metrics": metrics(records)}
        seeds.append(seed_result)
        all_records.extend(records)
        write_json(seed_dir / "result.json", seed_result)
        write_json(output / "partial-results.json", {"seeds": seeds, "predictions": all_records})
    aggregate = {}
    for split in ("test", "ood"):
        aggregate[split] = {}
        for arm in ARM_NAMES:
            values = torch.tensor([s["metrics"][split][arm]["mae"] for s in seeds], dtype=torch.float64)
            aggregate[split][arm] = {"mae_mean": float(values.mean()),
                                     "mae_seed_sd": float(values.std(correction=1)) if len(values) > 1 else None}
    return {"status": "completed", "config": config, "manifest": manifest, "seeds": seeds,
            "aggregate": aggregate, "predictions": all_records, "comparability": config["comparability"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("calibrate", "run"), required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--admission", type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    validate_config(config)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    if os.environ["CUBLAS_WORKSPACE_CONFIG"] not in (":4096:8", ":16:8"):
        raise RuntimeError("Deterministic CUDA requires CUBLAS_WORKSPACE_CONFIG=:4096:8 or :16:8")
    device = require_cuda()  # Do not create a run or silently use CPU without CUDA.
    torch.set_num_threads(config["threads"])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    sources = source_manifest()
    environment = {"python": platform.python_version(), "torch": torch.__version__, "cuda": torch.version.cuda,
                   "gpu": torch.cuda.get_device_name(device), "threads": config["threads"],
                   "float32_network": True, "float64_path_weights": True,
                   "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"]}
    if args.phase == "run":
        if args.admission is None:
            raise ValueError("A successful --admission calibration.json is required")
        admission = json.loads(args.admission.read_text(encoding="utf-8"))
        if (admission.get("admitted") is not True or admission.get("config_hash") != digest(config)
                or admission.get("source_hashes") != sources or admission.get("environment") != environment):
            raise ValueError("Calibration does not match this config, source or GPU environment; calibrate again")
    args.output.mkdir(parents=True, exist_ok=False)
    evidence = {"phase": args.phase, "config_hash": digest(config), "source_hashes": sources,
                "environment": environment, "git": git_state(), "config": config}
    write_json(args.output / "provenance.json", evidence)
    started = time.monotonic()
    limit = config["calibration_seconds_limit"] if args.phase == "calibrate" else config["total_seconds_limit"]
    try:
        if args.phase == "calibrate":
            result = run_calibration(config, args.output, device, started + limit)
            result.update(evidence)
            check_deadline(started + limit, "calibration completion")
            write_json(args.output / "calibration.json", result)
        else:
            result = run_formal(config, args.output, device, started + limit)
            result["provenance"] = evidence
            result["total_seconds"] = time.monotonic() - started
            check_deadline(started + limit, "formal completion")
            write_json(args.output / "results.json", result)
        print(json.dumps({"status": "completed", "phase": args.phase, "output": str(args.output)}, ensure_ascii=False))
    except Exception as exc:
        write_json(args.output / "failure.json", {"status": "failed", "exception": type(exc).__name__,
                                                "reason": str(exc), "elapsed_seconds": time.monotonic() - started})
        raise


if __name__ == "__main__":
    main()

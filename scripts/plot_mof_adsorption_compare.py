#!/usr/bin/env python3
"""Render publication-style parity and per-MOF MACE adsorption comparisons."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_plot_atlas(root: Path):
    sys.path.insert(0, str(root / "src"))
    from plot_atlas import apply_publication_style, get_palette, joint_scatter, paired_dumbbell, panel_label, save_figure

    return apply_publication_style, get_palette, joint_scatter, paired_dumbbell, panel_label, save_figure


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_result(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def render(args: argparse.Namespace) -> None:
    apply_publication_style, get_palette, joint_scatter, paired_dumbbell, panel_label, save_figure = load_plot_atlas(args.plot_atlas_root)
    apply_publication_style(9.2)
    rows = read_rows(args.csv)
    result = read_result(args.json)
    ref = np.asarray([float(row["reference_eads_eV"]) for row in rows])
    scratch = np.asarray([float(row["scratch_eads_eV"]) for row in rows])
    foundation = np.asarray([float(row["foundation_eads_eV"]) for row in rows])
    groups = np.asarray([row["reference_mof_id"] for row in rows])

    palette_groups = get_palette("isotope_sites")
    palette_pair = get_palette("paired_slate_orange")
    low = min(float(ref.min()), float(scratch.min()), float(foundation.min()))
    high = max(float(ref.max()), float(scratch.max()), float(foundation.max()))
    pad = 0.05 * (high - low)
    limits = (low - pad, high + pad)

    # Panel 1: frame-level parity. The repeated scalar reference is explicit in
    # the caption; the companion per-MOF panel prevents pseudo-replication.
    fig = plt.figure(figsize=(10.3, 4.8))
    gs = fig.add_gridspec(1, 2, wspace=0.32)
    frame_metrics = result["frame_metrics"]
    for index, (label, values, metric_key, color) in enumerate(
        [
            ("From scratch", scratch, "scratch", palette_groups["categorical"][0]),
            ("Foundation naive fine-tune", foundation, "foundation_naive_finetune", palette_groups["categorical"][2]),
        ]
    ):
        panel = joint_scatter(
            fig,
            gs[0, index],
            ref,
            values,
            palette_groups,
            groups=groups,
            # The dataset reference is one constant per MOF, so a within-group
            # KDE is singular; histogram marginals preserve the repeated-value
            # structure without inventing a continuous density.
            marginal="hist",
            regression=False,
            xlabel="Dataset reference $E_{\\mathrm{ads}}$ (eV)",
            ylabel="Model-derived $E_{\\mathrm{ads}}$ (eV)",
            marginal_axes=False,
            axis_linewidth=1.45,
            tick_width=1.25,
            tick_length=5.0,
            point_size=11,
            alpha=0.62,
            point_edgecolor="white",
            point_linewidth=0.25,
            legend_frame=True,
            marginal_ratio=0.16,
            marginal_gap=0.04,
        )
        ax = panel["main"]
        ax.plot(limits, limits, color="#222222", lw=1.15, ls=(0, (3, 2)), zorder=2)
        ax.set_xlim(*limits)
        ax.set_ylim(*limits)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(False)
        metric = frame_metrics[metric_key]
        ax.text(
            0.04,
            0.96,
            f"{label}\n$n={metric['count']}$\nMAE = {metric['mae_eV']:.3f} eV\n$r$ = {metric['pearson_r']:.2f}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=7.7,
            color="#222222",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.84, pad=2.5),
        )
        panel_label(panel["top"], "(a)" if index == 0 else "(b)", -0.27, 1.35)
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.15, top=0.94)
    parity_path = args.output_dir / "MOF_MACE_ADSORPTION_PARITY.png"
    save_figure(fig, parity_path)
    plt.close(fig)

    # Panel 2: per-MOF paired means with frame-to-frame spread as text. The
    # scalar reference is one value per MOF, while each model point is the mean
    # over the matching frames.
    per_mof = result["per_mof"]
    categories = [entry["mof_id"] for entry in per_mof]
    reference = np.asarray([entry["reference_eads_eV"] for entry in per_mof])
    model_values = {
        "From scratch": np.asarray([entry["scratch_mean_eV"] for entry in per_mof]),
        "Foundation naive fine-tune": np.asarray([entry["foundation_mean_eV"] for entry in per_mof]),
    }
    model_std = {
        "From scratch": [entry["scratch_std_eV"] for entry in per_mof],
        "Foundation naive fine-tune": [entry["foundation_std_eV"] for entry in per_mof],
    }
    fig, axes = plt.subplots(1, 2, figsize=(10.3, 4.0), sharex=True)
    for index, (ax, (label, values)) in enumerate(zip(axes, model_values.items())):
        labels = [f"{v:.3f} ± {s:.3f}" for v, s in zip(values, model_std[label])]
        paired_dumbbell(
            ax,
            categories,
            reference,
            values,
            palette_pair,
            end_labels=labels,
            xlabel="$E_{\\mathrm{ads}}$ (eV)",
            xlim=limits,
            gradient=True,
            marker_size=115,
        )
        ax.set_title(label, fontsize=9.2, fontweight="normal", pad=7)
        ax.tick_params(axis="y", labelsize=7.2)
        panel_label(ax, "(c)" if index == 0 else "(d)", -0.17, 1.14)
    fig.subplots_adjust(left=0.22, right=0.98, bottom=0.14, top=0.88, wspace=0.22)
    per_mof_path = args.output_dir / "MOF_MACE_ADSORPTION_PER_MOF.png"
    save_figure(fig, per_mof_path)
    plt.close(fig)

    qa = {
        "source_csv": str(args.csv),
        "source_json": str(args.json),
        "observation_unit": "frame-level parity and MOF-level mean decomposition",
        "figures": [],
    }
    for path in [parity_path, parity_path.with_suffix(".svg"), per_mof_path, per_mof_path.with_suffix(".svg")]:
        from PIL import Image

        entry = {"path": str(path), "sha256": file_sha256(path), "exists": path.exists()}
        if path.suffix.lower() == ".png":
            with Image.open(path) as image:
                entry["width_px"], entry["height_px"] = image.size
                entry["readable"] = image.format == "PNG"
        else:
            entry["readable"] = path.read_text(encoding="utf-8").lstrip().startswith("<?xml")
        qa["figures"].append(entry)
    (args.output_dir / "MOF_MACE_ADSORPTION_FIGURE_QA.json").write_text(json.dumps(qa, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--plot-atlas-root", type=Path, default=Path(os.environ.get("PLOT_ATLAS_ROOT", ".")))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    render(args)
    print(f"Rendered figures under {args.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

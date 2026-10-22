#!/usr/bin/env python3
"""Render scope-bounded figures for the matched model-method extension.

The renderer reads only the completed post-release model-method analysis.  It
creates raster/vector figures and a pending visual-QA ledger; it never uses a
figure to choose a model or alter an experiment.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


METHODS = (
    "foundation_naive",
    "lora_rank4",
    "layer_freeze5",
    "multihead_pseudolabel_replay",
)
METHOD_LABELS = {
    "scratch_micro": "Scratch",
    "foundation_naive": "Naive FT",
    "lora_rank4": "LoRA r=4",
    "layer_freeze5": "Freeze 5",
    "multihead_pseudolabel_replay": "Multihead replay",
}
ARM_COLORS = {
    "fixed": "#6B7280",
    "outer_generated": "#0066CC",
    "manual_control": "#D97706",
    "removed": "#B91C1C",
}


def load_json(path: Path) -> Dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Empty required input table: {path}")
    return rows


def number(row: Mapping[str, str], field: str) -> float:
    try:
        return float(row[field])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Missing numeric field {field}") from exc


def mean(values: Iterable[float]) -> float:
    copied = list(values)
    if not copied:
        raise ValueError("Cannot average an empty series")
    return sum(copied) / len(copied)


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": ["Arial", "Liberation Sans", "DejaVu Sans"],
            "font.size": 10.5,
            "axes.labelsize": 11.5,
            "axes.linewidth": 1.25,
            "axes.grid": False,
            "legend.fontsize": 9.0,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )


def style_axis(axis: plt.Axes) -> None:
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.tick_params(width=1.15, length=4)
    for label in [*axis.get_xticklabels(), *axis.get_yticklabels()]:
        label.set_fontweight("bold")


def save_figure(figure: plt.Figure, output_dir: Path, basename: str, dpi: int) -> List[str]:
    names = []
    for suffix in ("png", "pdf", "svg"):
        path = output_dir / f"{basename}.{suffix}"
        figure.savefig(path, dpi=dpi, bbox_inches=None)
        names.append(path.name)
    plt.close(figure)
    return names


def plot_method_effects(
    effects: Sequence[Mapping[str, str]],
    arm_rows: Sequence[Mapping[str, str]],
    output_dir: Path,
    dpi: int,
) -> tuple[List[str], str]:
    response = "final_test_delta_force_component_mae_meV_per_A"
    selected = [
        row
        for row in effects
        if row["comparison"] == "outer_generated_minus_removed" and row["response"] == response
    ]
    if {row["method"] for row in selected} != set(METHODS):
        raise ValueError("Model-effect table lacks one or more registered foundation methods")
    by_method = {method: [row for row in selected if row["method"] == method] for method in METHODS}
    force_rows = [row for row in arm_rows if row["metric"] == "force_component_mae_meV_per_A"]
    if {row["method"] for row in force_rows} != set(METHODS):
        raise ValueError("Method-arm table lacks force rows for one or more foundation methods")
    figure, axes = plt.subplots(1, 2, figsize=(12.0, 4.9), gridspec_kw={"width_ratios": [1.0, 1.35]})
    left, right = axes
    effect_values = [mean(number(row, "method_minus_scratch") for row in by_method[method]) for method in METHODS]
    left.scatter(range(len(METHODS)), effect_values, s=58, color="#0066CC", zorder=3)
    left.plot(range(len(METHODS)), effect_values, color="#0066CC", linewidth=1.45, alpha=0.8)
    left.axhline(0.0, color="#333333", linestyle="--", linewidth=1.0, zorder=0)
    left.set_xticks(range(len(METHODS)), [METHOD_LABELS[method] for method in METHODS], rotation=28, ha="right")
    left.set_ylabel("Method minus scratch outer−removed force effect (meV Å$^{-1}$)")
    left.set_xlabel("Foundation adaptation method")
    left.set_title("A", loc="left", fontweight="bold")
    style_axis(left)

    arms = ("fixed", "outer_generated", "manual_control", "removed")
    width = 0.18
    positions = list(range(len(METHODS)))
    for arm_index, arm in enumerate(arms):
        values = []
        for method in METHODS:
            members = [row for row in force_rows if row["method"] == method and row["arm"] == arm]
            if not members:
                raise ValueError(f"Missing method/arm force comparison for {method}/{arm}")
            values.append(mean(number(row, "method_minus_scratch") for row in members))
        right.bar(
            [position + (arm_index - 1.5) * width for position in positions],
            values,
            width=width,
            color=ARM_COLORS[arm],
            label=arm.replace("_", " "),
        )
    right.axhline(0.0, color="#333333", linestyle="--", linewidth=1.0, zorder=0)
    right.set_xticks(positions, [METHOD_LABELS[method] for method in METHODS], rotation=28, ha="right")
    right.set_ylabel("Method minus scratch final force MAE (meV Å$^{-1}$)")
    right.set_xlabel("Foundation adaptation method")
    right.legend(title="Frozen policy arm", frameon=True, facecolor="white", edgecolor="#BBBBBB", loc="upper right")
    right.set_title("B", loc="left", fontweight="bold")
    style_axis(right)
    figure.subplots_adjust(left=0.08, right=0.985, bottom=0.28, top=0.93, wspace=0.35)
    caption = (
        "Figure 5. Matched foundation-method extension on the one frozen toluene policy holdout. "
        "A, change in the outer-generated-minus-information-removed final force-effect relative to the scratch reference; negative values favor the foundation method's paired agent-control contrast. "
        "B, final independent-test force-MAE change relative to scratch within each frozen policy arm. "
        "Every point/bar is descriptive for this matched holdout, not a general model-ranking claim."
    )
    return save_figure(figure, output_dir, "fig5_model_method_effects", dpi), caption


def plot_forgetting(rows: Sequence[Mapping[str, str]], output_dir: Path, dpi: int) -> tuple[List[str], str]:
    target = [row for row in rows if row["method"] == "multihead_pseudolabel_replay"]
    if len(target) != 1:
        raise ValueError("Expected exactly one multihead replay forgetting row")
    row = target[0]
    force_fields = ("mae_f_final_minus_initial", "rmse_f_final_minus_initial")
    energy_fields = ("mae_e_per_atom_final_minus_initial", "rmse_e_per_atom_final_minus_initial")
    figure, axes = plt.subplots(1, 2, figsize=(9.1, 4.4))
    for axis, fields, title, ylabel in (
        (axes[0], energy_fields, "A", "Pretraining-head energy error final − initial"),
        (axes[1], force_fields, "B", "Pretraining-head force error final − initial"),
    ):
        values = [number(row, field) for field in fields]
        axis.bar(range(len(fields)), values, color=["#6B7280", "#0066CC"], width=0.58)
        axis.axhline(0.0, color="#333333", linestyle="--", linewidth=1.0, zorder=0)
        axis.set_xticks(range(len(fields)), [field.split("_")[0].upper() for field in fields])
        axis.set_ylabel(ylabel)
        axis.set_xlabel("Metric")
        axis.set_title(title, loc="left", fontweight="bold")
        style_axis(axis)
    figure.subplots_adjust(left=0.10, right=0.98, bottom=0.16, top=0.93, wspace=0.34)
    caption = (
        "Figure 6. Multihead replay-domain forgetting diagnostic from the validation-only foundation-method admission. "
        "Bars are final minus initial pretraining-head errors on the distinct immutable replay-validation file; positive values indicate an increase in that error. "
        "This diagnostic is not a downstream toluene performance metric."
    )
    return save_figure(figure, output_dir, "fig6_multihead_forgetting", dpi), caption


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError("Figure output directory is immutable; choose a new versioned path")
    analysis = load_json(args.analysis_dir / "analysis.json")
    if analysis.get("schema") != "micro-mlip-model-method-extension-analysis-v1" or analysis.get("status") != "COMPLETED":
        raise ValueError("Figures require a completed model-method extension analysis")
    configure_matplotlib()
    args.output_dir.mkdir(parents=True)
    effect_rows = read_csv(args.analysis_dir / "method_effect_vs_scratch.csv")
    arm_rows = read_csv(args.analysis_dir / "method_arm_vs_scratch.csv")
    forgetting_rows = read_csv(args.analysis_dir / "method_forgetting.csv")
    figures = []
    captions = []
    for names, caption in (
        plot_method_effects(effect_rows, arm_rows, args.output_dir, args.dpi),
        plot_forgetting(forgetting_rows, args.output_dir, args.dpi),
    ):
        figures.append(names)
        captions.append(caption)
    (args.output_dir / "captions.md").write_text("\n\n".join(captions) + "\n", encoding="utf-8")
    inventory = {
        "schema": "micro-mlip-model-method-figure-inventory-v1",
        "analysis_dir": str(args.analysis_dir),
        "analysis_sha256": hashlib.sha256((args.analysis_dir / "analysis.json").read_bytes()).hexdigest(),
        "figures": figures,
        "captions": "captions.md",
    }
    (args.output_dir / "figure_inventory.json").write_text(json.dumps(inventory, indent=2) + "\n", encoding="utf-8")
    qa = {
        "schema": "micro-mlip-model-method-figure-qa-v1",
        "status": "PENDING_MANUAL_VISUAL_AUDIT",
        "reason": "The renderer validates files but cannot truthfully certify final-size visual layout without raster inspection.",
        "figures": [
            {
                "files": names,
                "checks_required": [
                    "axis title versus tick labels",
                    "panel labels versus axes",
                    "legends versus bars/points",
                    "zero lines versus data",
                    "long labels across panel gutters",
                    "PNG/PDF/SVG readability",
                ],
            }
            for names in figures
        ],
    }
    (args.output_dir / "figure_qa.json").write_text(json.dumps(qa, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "RENDERED_PENDING_VISUAL_QA", "output_dir": str(args.output_dir), "figure_count": len(figures)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

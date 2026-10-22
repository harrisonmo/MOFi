#!/usr/bin/env python3
"""Render publication-oriented figures from a completed full P/D/T/W analysis.

The renderer accepts only a machine-verified post-release analysis directory.
It never opens raw training runs or participates in selection.  It creates
PNG/PDF/SVG outputs, captions, an inventory, and a *pending* visual-QA ledger.
The ledger deliberately requires a later human/agent raster inspection before
the figures are presented as final publication artifacts.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ARM_COLORS = {
    "fixed": "#6B7280",
    "outer_generated": "#0066CC",
    "manual_control": "#D97706",
    "removed": "#B91C1C",
}
ARM_LABELS = {
    "fixed": "Fixed",
    "outer_generated": "Outer-generated",
    "manual_control": "Manual",
    "removed": "Information-removed",
}
ACTION_COLORS = {"random": "#6B7280", "geometry_coverage": "#0066CC"}


def load_json(path: Path) -> Dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Empty figure input table: {path}")
    return rows


def number(row: Mapping[str, str], field: str) -> float:
    try:
        return float(row[field])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Missing numeric field {field}") from exc


def mean(values: Iterable[float]) -> float:
    copied = list(values)
    if not copied:
        raise ValueError("Cannot average an empty sequence")
    return sum(copied) / len(copied)


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": ["Arial", "Liberation Sans", "DejaVu Sans"],
            "font.size": 10.5,
            "axes.labelsize": 11.5,
            "axes.linewidth": 1.25,
            "axes.grid": False,
            "legend.fontsize": 9.5,
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
    paths = []
    for suffix in ("png", "pdf", "svg"):
        path = output_dir / f"{basename}.{suffix}"
        figure.savefig(path, dpi=dpi, bbox_inches=None)
        paths.append(path.name)
    plt.close(figure)
    return paths


def grouped(rows: Sequence[Mapping[str, str]], fields: Sequence[str]) -> Dict[tuple[str, ...], List[Mapping[str, str]]]:
    output: Dict[tuple[str, ...], List[Mapping[str, str]]] = {}
    for row in rows:
        key = tuple(str(row[field]) for field in fields)
        output.setdefault(key, []).append(row)
    return output


def plot_learning_curves(rows: Sequence[Mapping[str, str]], output_dir: Path, dpi: int) -> tuple[List[str], str]:
    highlighted = ("P0D0T0W0M0", "P1D1T1W1M0")
    generations = ("g0", "g1")
    figure, axes = plt.subplots(2, 2, figsize=(10.4, 7.4), sharex=True, sharey=False)
    traces = grouped(rows, ("generation", "factorial_cell_id", "molecule", "seed", "arm"))
    for row_index, generation in enumerate(generations):
        for column_index, cell_id in enumerate(highlighted):
            axis = axes[row_index, column_index]
            for arm in ARM_COLORS:
                arm_traces = [
                    values
                    for (trace_generation, trace_cell, _, _, trace_arm), values in traces.items()
                    if trace_generation == generation and trace_cell == cell_id and trace_arm == arm
                ]
                for values in arm_traces:
                    ordered = sorted(values, key=lambda item: int(item["round_index"]))
                    axis.plot(
                        [number(item, "label_budget") for item in ordered],
                        [number(item, "validation_force_component_mae_meV_per_A") for item in ordered],
                        color=ARM_COLORS[arm],
                        linewidth=0.9,
                        alpha=0.35,
                    )
                by_budget = grouped(
                    [item for values in arm_traces for item in values],
                    ("label_budget",),
                )
                if by_budget:
                    budgets = sorted(int(budget) for (budget,) in by_budget)
                    axis.plot(
                        budgets,
                        [
                            mean(
                                number(item, "validation_force_component_mae_meV_per_A")
                                for item in by_budget[(str(budget),)]
                            )
                            for budget in budgets
                        ],
                        color=ARM_COLORS[arm],
                        linewidth=2.15,
                        marker="o",
                        markersize=4.5,
                        label=ARM_LABELS[arm],
                    )
            axis.set_title(f"{generation}: {'All-low' if cell_id == highlighted[0] else 'Full loop'}", loc="left", fontweight="normal")
            style_axis(axis)
            if row_index == 1:
                axis.set_xlabel("Cumulative labelled frames")
            if column_index == 0:
                axis.set_ylabel("Validation force MAE (meV Å$^{-1}$)")
            if row_index == 0 and column_index == 1:
                axis.legend(frameon=True, facecolor="white", edgecolor="#BBBBBB", loc="upper right")
    figure.subplots_adjust(left=0.10, right=0.98, bottom=0.10, top=0.94, wspace=0.25, hspace=0.28)
    caption = (
        "Figure 1. Five-round validation-force learning curves for the all-low and full P/D/T/W cells in g0 and g1. "
        "Thin trajectories are molecule × seed paired traces and heavy lines are their within-panel means; every arm uses the same five label budgets. "
        "These are development validation curves, not inputs retroactively used to change the frozen policy."
    )
    return save_figure(figure, output_dir, "fig1_learning_curves", dpi), caption


def plot_factorial_effects(
    summary_rows: Sequence[Mapping[str, str]],
    individual_rows: Sequence[Mapping[str, str]],
    output_dir: Path,
    dpi: int,
) -> tuple[List[str], str]:
    response = "final_test_delta_force_component_mae_meV_per_A"
    selected = [row for row in summary_rows if row["response"] == response]
    if not selected:
        raise ValueError("No final-force factorial effect summary is available")
    term_order = sorted(
        {row["factor_term"] for row in selected},
        key=lambda term: (term.count("×") + 1, term),
    )
    figure, axis = plt.subplots(figsize=(9.7, max(5.8, 0.34 * len(term_order) + 1.8)))
    y_positions = {term: len(term_order) - index - 1 for index, term in enumerate(term_order)}
    offsets = {"g0": -0.14, "g1": 0.14}
    colors = {"g0": "#6B7280", "g1": "#0066CC"}
    for generation in ("g0", "g1"):
        for row in selected:
            if row["generation"] != generation:
                continue
            term = row["factor_term"]
            members = [
                item
                for item in individual_rows
                if item["generation"] == generation and item["response"] == response and item["factor_term"] == term
            ]
            values = [number(item, "high_minus_low_contrast") for item in members]
            center = number(row, "mean_high_minus_low_contrast")
            low = min(values)
            high = max(values)
            axis.errorbar(
                center,
                y_positions[term] + offsets[generation],
                xerr=[[center - low], [high - center]],
                fmt="o",
                color=colors[generation],
                markersize=5.5,
                capsize=3,
                linewidth=1.1,
                label=generation if term == term_order[0] else None,
            )
    axis.axvline(0.0, color="#333333", linestyle="--", linewidth=1.0, zorder=0)
    axis.set_yticks([y_positions[term] for term in term_order], term_order)
    axis.set_xlabel("High-minus-low contrast in paired outer-generated − removed force error (meV Å$^{-1}$)")
    axis.set_ylabel("P/D/T/W main effect or interaction")
    axis.legend(title="Generation", frameon=True, facecolor="white", edgecolor="#BBBBBB", loc="lower right")
    style_axis(axis)
    figure.subplots_adjust(left=0.20, right=0.98, bottom=0.12, top=0.96)
    caption = (
        "Figure 2. Orthogonal P/D/T/W factorial contrasts for the paired final independent-test force-error response. "
        "Points summarize registered molecule × seed strata within g0 or g1 and horizontal ranges show their observed minimum-to-maximum span. "
        "Because the response is outer-generated minus information-removed, negative contrasts favor the high factor level for the outer policy. "
        "The model recipe is held at scratch in this figure."
    )
    return save_figure(figure, output_dir, "fig2_factorial_effects", dpi), caption


def plot_feedback_transitions(rows: Sequence[Mapping[str, str]], output_dir: Path, dpi: int) -> tuple[List[str], str]:
    selected = [
        row
        for row in rows
        if row["arm"] == "outer_generated" and row["tool_level"] == "high" and row["residual_class"] != "RAW_OR_STATIC_ROUTE"
    ]
    if not selected:
        raise ValueError("No typed residual-feedback transitions are available")
    residuals = sorted({row["residual_class"] for row in selected})
    figure, axis = plt.subplots(figsize=(10.0, 5.8))
    for action_index, action in enumerate(sorted({row["decision_selection"] for row in selected})):
        action_rows = [row for row in selected if row["decision_selection"] == action]
        by_residual = {row["residual_class"]: row for row in action_rows}
        xs, ys, sizes = [], [], []
        for index, residual in enumerate(residuals):
            row = by_residual.get(residual)
            if row is None:
                continue
            xs.append(index + (-0.12 if action_index == 0 else 0.12))
            ys.append(number(row, "mean_next_minus_current_validation_force_meV_per_A"))
            sizes.append(35.0 + 18.0 * number(row, "transition_count"))
        axis.scatter(xs, ys, s=sizes, color=ACTION_COLORS.get(action, "#111827"), alpha=0.85, label=action.replace("_", " "))
    axis.axhline(0.0, color="#333333", linestyle="--", linewidth=1.0, zorder=0)
    axis.set_xticks(range(len(residuals)), [item.replace("_", "\n") for item in residuals])
    axis.set_ylabel("Next minus current validation force MAE (meV Å$^{-1}$)")
    axis.set_xlabel("Registered typed residual class")
    axis.legend(title="Next acquisition", frameon=True, facecolor="white", edgecolor="#BBBBBB", loc="upper right")
    style_axis(axis)
    figure.subplots_adjust(left=0.11, right=0.98, bottom=0.25, top=0.96)
    caption = (
        "Figure 3. Operational feedback trace for the typed-tool outer-generated arms. Each point links a registered residual class, the selected acquisition action, and the following-round change in validation force MAE; point area denotes the number of transitions. "
        "This is a conditional descriptive trace, not evidence that a residual label identifies a physical molecular failure mechanism."
    )
    return save_figure(figure, output_dir, "fig3_feedback_transitions", dpi), caption


def plot_evolution_and_holdout(
    evolution_rows: Sequence[Mapping[str, str]],
    holdout_rows: Sequence[Mapping[str, str]],
    output_dir: Path,
    dpi: int,
) -> tuple[List[str], str]:
    response = "final_test_delta_force_component_mae_meV_per_A"
    evolution = [row for row in evolution_rows if row["response"] == response]
    if not evolution:
        raise ValueError("No g0→g1 force evolution rows are available")
    comparisons = [row for row in holdout_rows if "final_test_delta_force_component_mae_meV_per_A" in row]
    if not comparisons:
        raise ValueError("No toluene holdout paired effects are available")
    figure, axes = plt.subplots(1, 2, figsize=(11.0, 4.9), gridspec_kw={"width_ratios": [1.45, 1.0]})
    left, right = axes
    ordered_cells = sorted({row["factorial_cell_id"] for row in evolution})
    for index, cell in enumerate(ordered_cells):
        values = [number(row, "g1_minus_g0") for row in evolution if row["factorial_cell_id"] == cell]
        left.scatter([index] * len(values), values, color="#0066CC", s=30, alpha=0.75)
        left.plot([index - 0.16, index + 0.16], [mean(values), mean(values)], color="#111827", linewidth=2.0)
    left.axhline(0.0, color="#333333", linestyle="--", linewidth=1.0, zorder=0)
    left.set_xticks(range(len(ordered_cells)), ordered_cells, rotation=60, ha="right")
    left.set_ylabel("g1 minus g0 paired force effect (meV Å$^{-1}$)")
    left.set_xlabel("P/D/T/W cell")
    style_axis(left)

    labels = [row["comparison"].replace("_minus_", " − ").replace("_", " ") for row in comparisons]
    values = [number(row, "final_test_delta_force_component_mae_meV_per_A") for row in comparisons]
    colors = ["#0066CC" if "outer_generated" in row["comparison"] else "#6B7280" for row in comparisons]
    right.bar(range(len(values)), values, color=colors, width=0.64)
    right.axhline(0.0, color="#333333", linestyle="--", linewidth=1.0, zorder=0)
    right.set_xticks(range(len(labels)), labels, rotation=30, ha="right")
    right.set_ylabel("Toluene paired force difference (meV Å$^{-1}$)")
    right.set_xlabel("Four-arm comparison")
    style_axis(right)
    figure.subplots_adjust(left=0.09, right=0.99, bottom=0.31, top=0.95, wspace=0.36)
    caption = (
        "Figure 4. Iterative evolution and frozen-policy holdout. Left, g1 minus g0 paired force effects by P/D/T/W cell and development molecule; g0 and g1 use different registered seeds, so these are not seed-paired confidence intervals. "
        "Right, the four-arm paired final force differences on the single toluene policy holdout. Toluene was excluded from policy selection."
    )
    return save_figure(figure, output_dir, "fig4_evolution_holdout", dpi), caption


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError("Figure output directory is immutable; choose a new versioned path")
    analysis = load_json(args.analysis_dir / "analysis.json")
    if analysis.get("schema") != "micro-mlip-full-pdtw-postrelease-analysis-v1" or analysis.get("status") != "COMPLETED":
        raise ValueError("Figures require a completed, verified full P/D/T/W analysis")
    configure_matplotlib()
    args.output_dir.mkdir(parents=True)
    round_rows = read_csv(args.analysis_dir / "round_rows.csv")
    factorial_summary = read_csv(args.analysis_dir / "factorial_effect_summary.csv")
    factorial_rows = read_csv(args.analysis_dir / "factorial_effects.csv")
    transition_summary = read_csv(args.analysis_dir / "residual_action_transition_summary.csv")
    evolution_rows = read_csv(args.analysis_dir / "generation_evolution.csv")
    holdout_rows = read_csv(args.analysis_dir / "holdout_paired_effects.csv")
    figures = []
    captions = []
    for names, caption in (
        plot_learning_curves(round_rows, args.output_dir, args.dpi),
        plot_factorial_effects(factorial_summary, factorial_rows, args.output_dir, args.dpi),
        plot_feedback_transitions(transition_summary, args.output_dir, args.dpi),
        plot_evolution_and_holdout(evolution_rows, holdout_rows, args.output_dir, args.dpi),
    ):
        figures.append(names)
        captions.append(caption)
    (args.output_dir / "captions.md").write_text(
        "\n\n".join(captions) + "\n",
        encoding="utf-8",
    )
    inventory = {
        "schema": "micro-mlip-full-pdtw-figure-inventory-v1",
        "analysis_dir": str(args.analysis_dir),
        "analysis_sha256": __import__("hashlib").sha256((args.analysis_dir / "analysis.json").read_bytes()).hexdigest(),
        "figures": figures,
        "captions": "captions.md",
    }
    (args.output_dir / "figure_inventory.json").write_text(json.dumps(inventory, indent=2) + "\n", encoding="utf-8")
    qa = {
        "schema": "micro-mlip-full-pdtw-figure-qa-v1",
        "status": "PENDING_MANUAL_VISUAL_AUDIT",
        "reason": "The renderer validates files but cannot truthfully certify final-size visual layout without raster inspection.",
        "figures": [
            {
                "files": names,
                "checks_required": [
                    "axis title versus tick labels",
                    "panel labels versus axes",
                    "legends versus curves/points",
                    "annotations/zero lines versus data",
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

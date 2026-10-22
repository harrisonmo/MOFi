"""Render compact evidence figures for the MOF adsorption admission decision."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "benchmarks" / "mof_adsorption" / "prior_s2ef_metrics.csv"
OUT = ROOT / "docs" / "reports"


def load_rows() -> list[dict[str, str]]:
    with DATA.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def base_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 11,
            "axes.labelsize": 12,
            "axes.linewidth": 1.2,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "svg.fonttype": "none",
            "savefig.bbox": "tight",
        }
    )


def render_prior_s2ef(rows: list[dict[str, str]]) -> None:
    labels = [row["arm"] for row in rows]
    energy = np.array([float(row["energy_mae_eV_per_atom"]) for row in rows])
    force = np.array([float(row["force_component_mae_eV_per_A"]) for row in rows])
    colors = ["#4C78A8", "#F58518", "#54A24B", "#E45756"]

    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.0), constrained_layout=True)
    x = np.arange(len(labels))
    for axis, values, ylabel, scale in [
        (axes[0], energy, "Energy MAE (eV atom$^{-1}$)", "log"),
        (axes[1], force, "Force-component MAE (eV Å$^{-1}$)", "linear"),
    ]:
        axis.bar(x, values, color=colors, width=0.68, edgecolor="#222222", linewidth=0.7)
        axis.set_xticks(x, labels)
        axis.set_yscale(scale)
        axis.set_ylabel(ylabel)
        axis.tick_params(axis="x", rotation=25, width=1.0, length=4)
        axis.tick_params(axis="y", width=1.0, length=4)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.text(
            0.02,
            0.97,
            "Inherited PR #1\nvalidation-derived S2EF; not E$_{ads}$",
            transform=axis.transAxes,
            va="top",
            ha="left",
            fontsize=9,
            bbox={"facecolor": "white", "edgecolor": "#BBBBBB", "pad": 3.0},
        )
    fig.savefig(OUT / "MOF_ADSORPTION_PRIOR_S2EF.svg")
    fig.savefig(OUT / "MOF_ADSORPTION_PRIOR_S2EF.png", dpi=220)
    plt.close(fig)


def render_gate_status() -> None:
    labels = [
        "Paired host/guest/complex\nreference structures",
        "Independent DFT or PFP\noracle admitted",
        "Parent/family split\nmaterialized",
        "Three-parent short MD\nexecuted",
    ]
    statuses = ["BLOCKED", "BLOCKED", "NOT RUN", "NOT RUN"]
    colors = ["#C83E4D", "#C83E4D", "#8C8C8C", "#8C8C8C"]
    fig, axis = plt.subplots(figsize=(9.0, 3.5))
    y = np.arange(len(labels))
    axis.hlines(y, 0, 1, color="#D9D9D9", linewidth=2.0)
    axis.scatter(np.zeros_like(y), y, s=190, c=colors, edgecolors="#222222", linewidths=0.8, zorder=3)
    for yi, label, status in zip(y, labels, statuses):
        axis.text(0.06, yi, label, va="center", ha="left", fontsize=11)
        axis.text(0.98, yi, status, va="center", ha="right", fontsize=11, fontweight="bold", color="#333333")
    axis.set_xlim(-0.03, 1.03)
    axis.set_ylim(-0.7, len(labels) - 0.3)
    axis.invert_yaxis()
    axis.axis("off")
    axis.text(
        0.0,
        1.08,
        "Admission boundary for the 2026-10-16 run",
        transform=axis.transAxes,
        ha="left",
        va="bottom",
        fontsize=12,
        fontweight="bold",
    )
    axis.text(
        0.0,
        -0.08,
        "S2EF comparison completed; paired adsorption references, oracle calls, validated adsorption energy, and MD remain blocked.",
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=9.5,
        color="#444444",
    )
    fig.savefig(OUT / "MOF_ADSORPTION_ADMISSION_GATE.svg")
    fig.savefig(OUT / "MOF_ADSORPTION_ADMISSION_GATE.png", dpi=220)
    plt.close(fig)


def main() -> None:
    base_style()
    rows = load_rows()
    render_prior_s2ef(rows)
    render_gate_status()


if __name__ == "__main__":
    main()

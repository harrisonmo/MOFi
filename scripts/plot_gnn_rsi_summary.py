"""Plot the GNN self-improvement summary figures.

All numbers are copied from tracked reports (G1-G4 results, Blind-G1 campaign
reports and the v2 lineage confirmation record). No training, evaluation or
model calls are performed. Run with the GNN environment:

    & .venv-gnn/Scripts/python.exe scripts/plot_gnn_rsi_summary.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "docs" / "reports"

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#8a8985"
GRID = "#e6e5e1"
BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
GRAY = "#9a9994"

plt.rcParams.update({
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID,
    "axes.labelcolor": INK2,
    "xtick.color": INK2,
    "ytick.color": INK2,
    "text.color": INK,
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
})


def _save(fig, name):
    fig.savefig(OUT / name, dpi=180, bbox_inches="tight")
    plt.close(fig)


def human_ladder():
    """Figure 1: human-designed iterations G1 -> G4 (development MAE, 3 seeds)."""
    # (label, mean, sd, family) from G1/G2/G3/G3-ablation/G4 result reports.
    rows = [
        ("G1 periodic GNN (5 Å, mean)", 2.0202, 0.0459, "graph"),
        ("G1 geometry ExtraTrees", 0.6934, 0.0032, "descriptor"),
        ("G2 geometry MLP", 0.6965, 0.0098, "descriptor"),
        ("G2 GNN + 14 geometry", 0.4681, 0.0129, "hybrid"),
        ("G3 encoder (8 Å, env, sum+density)", 0.5355, 0.0230, "graph"),
        ("G3 ablation best: c8-hard-sum", 0.4952, 0.0167, "graph"),
        ("G4 g3 (150 epochs)", 0.4765, 0.0179, "graph"),
        ("G4 g3_hard (150 epochs)", 0.4579, 0.0175, "graph"),
        ("G4 g4_hybrid (G3 + geometry)", 0.4627, 0.0150, "hybrid"),
        ("G4 g4_graph (volume/mass readout)", 0.5088, 0.0469, "graph"),
    ]
    colors = {"graph": BLUE, "descriptor": ORANGE, "hybrid": AQUA}
    names = {"graph": "Pure graph GNN", "descriptor": "Geometry descriptors only",
             "hybrid": "Graph + geometry descriptors"}
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    ys = list(range(len(rows)))[::-1]
    for y, (label, m, sd, fam) in zip(ys, rows):
        ax.errorbar(m, y, xerr=sd, fmt="o", ms=7, color=colors[fam], ecolor=colors[fam],
                    elinewidth=2, capsize=0, mec=SURFACE, mew=1.5)
        ax.text(m + 0.06, y, f"{m:.3f}", va="center", fontsize=9, color=INK2)
    ax.set_yticks(ys, [r[0] for r in rows])
    ax.set_xlim(0, 2.35)
    ax.set_xlabel("Development MAE (mmol/g), mean ± seed SD, lower is better")
    ax.grid(axis="y", visible=False)
    ax.axvline(0.4579, color=MUTED, lw=1, ls=(0, (3, 3)))
    ax.set_ylim(-1.0, len(rows) - 0.5)
    ax.text(0.4579, -0.75, " best human: 0.458", fontsize=8.5, color=INK2, va="center")
    for fam in ("graph", "descriptor", "hybrid"):
        ax.plot([], [], "o", color=colors[fam], label=names[fam], ms=7)
    ax.legend(loc="lower right", frameon=False, fontsize=9)
    ax.set_title("Human-designed GNN iterations G1 → G4 (seeds 17/29/43, same split)",
                 loc="left", fontsize=11.5)
    _save(fig, "GNN_RSI_SUMMARY_HUMAN_LADDER.png")


def ablation():
    """Figure 2: G3 2x2x2 factorial ablation (50-epoch budget)."""
    cells = {  # (cutoff, aggregation, envelope) -> (mean, sd)
        ("5 Å", "mean", "hard"): (2.0202, 0.0459),
        ("5 Å", "mean", "env"): (2.2540, 0.0985),
        ("5 Å", "sum+density", "hard"): (0.5476, 0.0135),
        ("5 Å", "sum+density", "env"): (0.6334, 0.0484),
        ("8 Å", "mean", "hard"): (1.4015, 0.1735),
        ("8 Å", "mean", "env"): (1.2985, 0.0194),
        ("8 Å", "sum+density", "hard"): (0.4952, 0.0167),
        ("8 Å", "sum+density", "env"): (0.5355, 0.0230),
    }
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.8), sharey=True)
    for ax, agg in zip(axes, ("mean", "sum+density")):
        width = 0.36
        for i, (env, color) in enumerate((("hard", BLUE), ("env", ORANGE))):
            xs = [0 + (i - 0.5) * (width + 0.02), 1 + (i - 0.5) * (width + 0.02)]
            vals = [cells[(c, agg, env)] for c in ("5 Å", "8 Å")]
            ax.bar(xs, [v[0] for v in vals], width, yerr=[v[1] for v in vals], color=color,
                   ecolor=INK2, error_kw={"elinewidth": 1}, label=f"{env} cutoff" if env == "hard"
                   else "cosine envelope", zorder=2)
            for x, (m, _) in zip(xs, vals):
                ax.text(x, m + 0.08, f"{m:.2f}", ha="center", fontsize=8.5, color=INK2)
        ax.set_xticks([0, 1], ["5 Å cutoff", "8 Å cutoff"])
        ax.set_title(f"{agg} aggregation", fontsize=10.5)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("Development MAE (mmol/g)")
    axes[0].set_ylim(0, 2.6)
    axes[1].legend(frameon=False, fontsize=9, loc="upper right")
    fig.suptitle("G3 factorial ablation: sum aggregation + density is the dominant fix",
                 x=0.02, ha="left", fontsize=11.5)
    fig.tight_layout()
    _save(fig, "GNN_RSI_SUMMARY_ABLATION.png")


def agent_confirmation():
    """Figure 3: blind agent campaigns, fresh-seed confirmation (seeds 101/202/303)."""
    # Per-seed confirmation MAE from campaign reports / confirmation records.
    rows = [
        ("Parent G1-F50 (shared ancestor)", [2.3066, 1.9937, 2.1838], GRAY),
        ("Claude v1 · RSI best (rsi-1)", [0.4906, 0.4963, 0.4935], BLUE),
        ("Claude v1 · independent best", [0.4795, 0.4708, 0.4514], ORANGE),
        ("Claude v2 gen 1 · RSI", [0.4512, 0.4452, 0.4426], BLUE),
        ("Claude v2 gen 1 · independent", [0.4029, 0.3926, 0.4271], ORANGE),
        ("GPT-6.1 Sol v3 · gen 1 (action-8)", [0.4691, 0.4684, 0.4678], AQUA),
        ("GPT-6.1 Sol v3 · gen 3 (final)", [0.4772, 0.4870, 0.4731], AQUA),
    ]
    fig, (ax0, ax) = plt.subplots(1, 2, figsize=(8.8, 4.2), sharey=True,
                                  gridspec_kw={"width_ratios": [1, 3.2], "wspace": 0.05})
    ys = list(range(len(rows)))[::-1]
    for y, (label, seeds, color) in zip(ys, rows):
        mean = sum(seeds) / len(seeds)
        for a in (ax0, ax):
            a.scatter(seeds, [y] * 3, s=22, color=color, alpha=0.45, lw=0, zorder=2)
            a.scatter([mean], [y], s=70, color=color, edgecolor=SURFACE, lw=1.5, zorder=3)
        target = ax0 if mean > 1 else ax
        target.text(mean, y + 0.28, f"{mean:.3f}", ha="center", fontsize=8.5, color=INK2)
    ax0.set_xlim(1.9, 2.4)
    ax0.set_xticks([2.0, 2.2])
    ax.set_xlim(0.37, 0.56)
    ax0.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax0.set_yticks(ys, [r[0] for r in rows])
    for a in (ax0, ax):
        a.grid(axis="y", visible=False)
    for x, txt in ((0.4579, "human best\ng3_hard 0.458*"), (0.5355, "human G3\n50 ep 0.536*")):
        ax.axvline(x, color=MUTED, lw=1, ls=(0, (3, 3)), zorder=1)
        ax.text(x, len(rows) - 0.55, txt, fontsize=7.8, color=INK2, ha="center", va="bottom")
    fig.text(0.5, 0.0, "Development MAE (mmol/g) on fresh seeds; small dots = seeds, large = mean. "
             "*human references use early stopping, not the fixed-50-epoch protocol",
             ha="center", fontsize=8.5, color=INK2)
    ax.scatter([], [], color=BLUE, label="RSI arm (inherits own history)")
    ax.scatter([], [], color=ORANGE, label="Independent arm (fresh reset)")
    ax.scatter([], [], color=AQUA, label="Single recursive lineage")
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    fig.suptitle("Blind agent campaigns: one proposal removes ~80% of G1 error",
                 x=0.02, ha="left", fontsize=11.5, y=1.0)
    _save(fig, "GNN_RSI_SUMMARY_AGENT_CONFIRMATION.png")


def sol_lineage():
    """Figure 4: GPT-6.1 Sol v3 three-generation lineage (search MAE, seeds 17/29)."""
    # (label, search mean MAE, promoted) from BLIND_G1_V3_GENERATION01/02 and FINAL.
    gens = [
        [("a3 global descriptors", 0.513001, True), ("a4 cosine LR", 0.476283, True),
         ("a5 Smooth L1", 0.476248, False), ("a6 joint readout", 0.472123, False),
         ("a7 pair radial", 0.511558, False), ("a8 dropout on a6", 0.470689, True)],
        [("radial counts", 0.470455, False), ("std readout", 0.470545, False),
         ("param averaging", 0.470489, False), ("cosine msg agg", 0.470358, False),
         ("LR 3e-3", 0.488858, False), ("LR 3e-4", 0.498049, False)],
        [("mass readout", 0.470623, False), ("2-layer desc MLP", 0.465968, True),
         ("low-rank quad", 0.486859, False), ("joint 2-layer", 0.471372, False),
         ("bilinear", 0.471109, False), ("gated desc", 0.467959, False)],
    ]
    fig, ax = plt.subplots(figsize=(9.2, 4.4))
    x = 0
    inc_x, inc_y = [], []
    incumbent = None
    labels, ticks = [], []
    for g, cands in enumerate(gens, start=1):
        start = x
        for label, mae, promoted in cands:
            color = AQUA if promoted else GRAY
            ax.scatter(x, mae, s=60 if promoted else 34, color=color, edgecolor=SURFACE, lw=1.2,
                       zorder=3)
            if promoted:
                incumbent = mae
            if incumbent is not None:
                inc_x.append(x)
                inc_y.append(incumbent)
            labels.append(label)
            ticks.append(x)
            x += 1
        ax.axvspan(start - 0.5, x - 0.5, color=GRID if g % 2 else SURFACE, alpha=0.35, lw=0,
                   zorder=0)
        ax.text((start + x - 1) / 2, 0.521, f"Generation {g}", ha="center", fontsize=9.5,
                color=INK2)
    ax.step(inc_x, inc_y, where="post", color=AQUA, lw=2, zorder=2, label="Incumbent (search MAE)")
    ax.scatter([], [], color=AQUA, s=60, label="Promoted (≥1% gain)")
    ax.scatter([], [], color=GRAY, s=34, label="Evaluated, not promoted")
    ax.scatter([5, 17], [0.468442, 0.479093], marker="D", s=46, color=SURFACE, edgecolor=INK,
               lw=1.4, zorder=4, label="Fresh-seed confirmation")
    ax.annotate("gen-3 search gain\nfails confirmation", xy=(17, 0.479093), xytext=(14.6, 0.503),
                fontsize=8.5, color=INK2, arrowprops={"arrowstyle": "-", "color": MUTED})
    ax.set_xticks(ticks, labels, rotation=55, ha="right", fontsize=8)
    ax.set_xlim(-0.6, x - 0.4)
    ax.set_ylim(0.455, 0.525)
    ax.set_ylabel("Development MAE (mmol/g)")
    ax.grid(axis="x", visible=False)
    ax.legend(frameon=False, fontsize=8.5, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=4)
    ax.set_title("GPT-6.1 Sol v3 recursive lineage (ancestor G1-F50 = 2.182, off-scale)",
                 loc="left", fontsize=11.5, pad=24)
    _save(fig, "GNN_RSI_SUMMARY_SOL_LINEAGE.png")


def outcomes():
    """Figure 5: what happened to each scored proposal slot per campaign."""
    campaigns = ["Claude v1\n(6 slots)", "Sol v1 pilot\n(3 slots)", "Claude v2 gen 1\n(6 slots)",
                 "Sol v3 lineage\n(18 candidates)"]
    data = {
        "Accepted / promoted": ([2, 2, 3, 4], AQUA),
        "Ran, not promoted": ([0, 0, 2, 14], BLUE),
        "Void: format gate": ([4, 0, 0, 0], ORANGE),
        "Runtime failure": ([0, 1, 1, 0], GRAY),
    }
    fig, ax = plt.subplots(figsize=(8.2, 3.6))
    left = [0] * len(campaigns)
    ys = list(range(len(campaigns)))[::-1]
    for name, (vals, color) in data.items():
        ax.barh(ys, vals, left=left, color=color, height=0.55, edgecolor=SURFACE, lw=2,
                label=name)
        for y, l, v in zip(ys, left, vals):
            if v:
                ax.text(l + v / 2, y, str(v), ha="center", va="center", fontsize=9,
                        color=SURFACE if color in (BLUE, GRAY) else INK)
        left = [a + b for a, b in zip(left, vals)]
    ax.set_yticks(ys, campaigns)
    ax.set_axisbelow(True)
    ax.set_xlabel("Proposals")
    ax.grid(axis="y", visible=False)
    ax.legend(frameon=False, fontsize=8.5, ncol=4, loc="upper left", bbox_to_anchor=(0, 1.13))
    ax.set_title("Proposal outcomes: v1 losses were format, later losses were no gain",
                 loc="left", fontsize=11.5, pad=26)
    _save(fig, "GNN_RSI_SUMMARY_OUTCOMES.png")


if __name__ == "__main__":
    human_ladder()
    ablation()
    agent_confirmation()
    sol_lineage()
    outcomes()
    print("wrote 5 figures to", OUT)

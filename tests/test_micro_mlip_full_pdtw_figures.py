"""Smoke contract for rendering post-release micro-MLIP figures."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).parents[1]
RENDERER = ROOT / "scripts" / "render_micro_mlip_full_pdtw_figures.py"


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def test_renderer_emits_vector_raster_inventory_and_pending_visual_qa(tmp_path: Path):
    analysis = tmp_path / "analysis"
    analysis.mkdir()
    (analysis / "analysis.json").write_text(
        json.dumps(
            {
                "schema": "micro-mlip-full-pdtw-postrelease-analysis-v1",
                "status": "COMPLETED",
                "synthetic_test_data": True,
            }
        ),
        encoding="utf-8",
    )
    arms = ("fixed", "outer_generated", "manual_control", "removed")
    round_rows = []
    for generation, generation_offset in (("g0", 0.0), ("g1", -5.0)):
        for cell_index, cell in enumerate(("P0D0T0W0M0", "P1D1T1W1M0")):
            for arm_index, arm in enumerate(arms):
                for round_index, budget in enumerate((128, 192, 256, 320, 384), start=1):
                    round_rows.append(
                        {
                            "generation": generation,
                            "factorial_cell_id": cell,
                            "molecule": "synthetic",
                            "seed": 1 if generation == "g0" else 2,
                            "arm": arm,
                            "round_index": round_index,
                            "label_budget": budget,
                            "validation_force_component_mae_meV_per_A": 180.0 + generation_offset + 4 * cell_index + arm_index - 8 * round_index,
                        }
                    )
    write_csv(analysis / "round_rows.csv", round_rows)
    factor_rows = []
    factor_summary = []
    for generation, offset in (("g0", 0.0), ("g1", -1.0)):
        for term, value in (("P", -2.0 + offset), ("D", 1.0 + offset), ("P×D", 0.5 + offset)):
            factor_summary.append(
                {
                    "generation": generation,
                    "response": "final_test_delta_force_component_mae_meV_per_A",
                    "factor_term": term,
                    "mean_high_minus_low_contrast": value,
                }
            )
            for molecule, perturbation in (("a", -0.2), ("b", 0.2)):
                factor_rows.append(
                    {
                        "generation": generation,
                        "molecule": molecule,
                        "seed": 1,
                        "response": "final_test_delta_force_component_mae_meV_per_A",
                        "factor_term": term,
                        "high_minus_low_contrast": value + perturbation,
                    }
                )
    write_csv(analysis / "factorial_effect_summary.csv", factor_summary)
    write_csv(analysis / "factorial_effects.csv", factor_rows)
    write_csv(
        analysis / "residual_action_transition_summary.csv",
        [
            {"generation": "g0", "arm": "outer_generated", "tool_level": "high", "residual_class": "force_tail_or_undercoverage", "decision_selection": "geometry_coverage", "transition_count": 2, "mean_next_minus_current_validation_force_meV_per_A": -12.0},
            {"generation": "g1", "arm": "outer_generated", "tool_level": "high", "residual_class": "balanced_or_improving", "decision_selection": "random", "transition_count": 2, "mean_next_minus_current_validation_force_meV_per_A": -8.0},
        ],
    )
    write_csv(
        analysis / "generation_evolution.csv",
        [
            {"factorial_cell_id": "P0D0T0W0M0", "molecule": "a", "response": "final_test_delta_force_component_mae_meV_per_A", "g1_minus_g0": -2.0},
            {"factorial_cell_id": "P1D1T1W1M0", "molecule": "a", "response": "final_test_delta_force_component_mae_meV_per_A", "g1_minus_g0": 1.0},
        ],
    )
    write_csv(
        analysis / "holdout_paired_effects.csv",
        [
            {"comparison": "outer_generated_minus_removed", "final_test_delta_force_component_mae_meV_per_A": -3.0},
            {"comparison": "outer_generated_minus_fixed", "final_test_delta_force_component_mae_meV_per_A": -1.0},
        ],
    )
    output = tmp_path / "figures"
    result = subprocess.run(
        [sys.executable, str(RENDERER), "--analysis-dir", str(analysis), "--output-dir", str(output), "--dpi", "72"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    for figure in ("fig1_learning_curves", "fig2_factorial_effects", "fig3_feedback_transitions", "fig4_evolution_holdout"):
        assert (output / f"{figure}.png").is_file()
        assert (output / f"{figure}.pdf").is_file()
        assert (output / f"{figure}.svg").is_file()
    qa = json.loads((output / "figure_qa.json").read_text(encoding="utf-8"))
    assert qa["status"] == "PENDING_MANUAL_VISUAL_AUDIT"

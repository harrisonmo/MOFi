"""Contract for cautious text generation from closed post-release tables."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).parents[1]
WRITER = ROOT / "scripts" / "write_micro_mlip_full_pdtw_results_report.py"


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def test_results_writer_emits_scope_bounded_report_and_claim_ledger(tmp_path: Path):
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
    delta_fields = {
        "final_test_delta_force_component_mae_meV_per_A": -2.0,
        "final_test_delta_energy_mae_meV_per_atom": -0.4,
        "validation_normalized_auc_delta_force_component_mae_meV_per_A": -1.1,
        "cumulative_training_seconds_delta": 3.0,
    }
    paired = [
        {"generation": "g0", "comparison": "outer_generated_minus_removed", **delta_fields},
        {
            "generation": "g1",
            "comparison": "outer_generated_minus_removed",
            **{key: value - 0.5 for key, value in delta_fields.items()},
        },
    ]
    write_csv(analysis / "paired_effects.csv", paired)
    write_csv(
        analysis / "factorial_effect_summary.csv",
        [
            {
                "generation": "g0",
                "response": "final_test_delta_force_component_mae_meV_per_A",
                "factor_term": "P",
                "mean_high_minus_low_contrast": -1.0,
            },
            {
                "generation": "g1",
                "response": "final_test_delta_force_component_mae_meV_per_A",
                "factor_term": "D",
                "mean_high_minus_low_contrast": 1.5,
            },
        ],
    )
    write_csv(
        analysis / "residual_action_transition_summary.csv",
        [
            {
                "generation": "g0",
                "arm": "outer_generated",
                "tool_level": "high",
                "residual_class": "force_tail_or_undercoverage",
                "decision_selection": "geometry_coverage",
                "transition_count": 4,
                "mean_next_minus_current_validation_force_meV_per_A": -2.0,
            }
        ],
    )
    write_csv(
        analysis / "generation_evolution.csv",
        [
            {
                "response": "final_test_delta_force_component_mae_meV_per_A",
                "g1_minus_g0": -0.8,
            }
        ],
    )
    write_csv(
        analysis / "holdout_paired_effects.csv",
        [
            {
                "comparison": "outer_generated_minus_removed",
                "final_test_delta_force_component_mae_meV_per_A": -3.2,
            }
        ],
    )

    output = tmp_path / "report"
    result = subprocess.run(
        [sys.executable, str(WRITER), "--analysis-dir", str(analysis), "--output-dir", str(output)],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    manuscript = (output / "manuscript_results.md").read_text(encoding="utf-8")
    normalized_manuscript = " ".join(manuscript.split())
    assert "not an adsorption-energy or MOF accuracy claim" in normalized_manuscript
    assert "not a causal mechanism claim" in normalized_manuscript
    assert "-3.200" in manuscript
    summary = (output / "research_summary.md").read_text(encoding="utf-8")
    assert "Research summary" in summary
    assert summary.isascii()
    ledger = json.loads((output / "claim_ledger.json").read_text(encoding="utf-8"))
    assert ledger["claims"][1]["status"] == "UNSUPPORTED"
    assert ledger["claims"][3]["status"] == "PENDING_SEPARATE_MATCHED_MODEL_METHOD_EXTENSION"

    model_analysis = tmp_path / "model_analysis"
    model_analysis.mkdir()
    (model_analysis / "analysis.json").write_text(
        json.dumps(
            {
                "schema": "micro-mlip-model-method-extension-analysis-v1",
                "status": "COMPLETED",
                "synthetic_test_data": True,
            }
        ),
        encoding="utf-8",
    )
    write_csv(
        model_analysis / "method_effect_vs_scratch.csv",
        [
            {
                "method": method,
                "comparison": "outer_generated_minus_removed",
                "response": "final_test_delta_force_component_mae_meV_per_A",
                "method_minus_scratch": -0.2 + index * 0.1,
            }
            for index, method in enumerate(
                ("foundation_naive", "lora_rank4", "layer_freeze5", "multihead_pseudolabel_replay")
            )
        ],
    )
    write_csv(
        model_analysis / "method_forgetting.csv",
        [
            {
                "method": "multihead_pseudolabel_replay",
                "mae_f_final_minus_initial": 0.35,
            }
        ],
    )
    output_with_model = tmp_path / "report_with_model"
    result = subprocess.run(
        [
            sys.executable,
            str(WRITER),
            "--analysis-dir",
            str(analysis),
            "--model-analysis-dir",
            str(model_analysis),
            "--output-dir",
            str(output_with_model),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    model_manuscript = (output_with_model / "manuscript_results.md").read_text(encoding="utf-8")
    assert "matched model-method table" in model_manuscript
    model_ledger = json.loads((output_with_model / "claim_ledger.json").read_text(encoding="utf-8"))
    assert model_ledger["claims"][3]["status"] == "SUPPORTED_ONLY_AS_MATCHED_SINGLE_TOLUENE_HOLDOUT_EXTENSION"

"""Smoke contract for model-method extension figure rendering."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).parents[1]
RENDERER = ROOT / "scripts" / "render_micro_mlip_model_method_figures.py"
METHODS = (
    "foundation_naive",
    "lora_rank4",
    "layer_freeze5",
    "multihead_pseudolabel_replay",
)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def test_model_method_renderer_emits_vector_raster_and_pending_qa(tmp_path: Path):
    analysis = tmp_path / "analysis"
    analysis.mkdir()
    (analysis / "analysis.json").write_text(
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
        analysis / "method_effect_vs_scratch.csv",
        [
            {
                "method": method,
                "comparison": "outer_generated_minus_removed",
                "response": "final_test_delta_force_component_mae_meV_per_A",
                "method_minus_scratch": -1.5 + index,
            }
            for index, method in enumerate(METHODS)
        ],
    )
    arms = ("fixed", "outer_generated", "manual_control", "removed")
    write_csv(
        analysis / "method_arm_vs_scratch.csv",
        [
            {
                "method": method,
                "arm": arm,
                "metric": "force_component_mae_meV_per_A",
                "method_minus_scratch": -1.0 + method_index * 0.4 + arm_index * 0.2,
            }
            for method_index, method in enumerate(METHODS)
            for arm_index, arm in enumerate(arms)
        ],
    )
    write_csv(
        analysis / "method_forgetting.csv",
        [
            {"method": method, "admission_status": "ADMITTED"}
            if method != "multihead_pseudolabel_replay"
            else {
                "method": method,
                "admission_status": "ADMITTED",
                "mae_e_per_atom_final_minus_initial": 0.1,
                "rmse_e_per_atom_final_minus_initial": 0.2,
                "mae_f_final_minus_initial": 0.3,
                "rmse_f_final_minus_initial": 0.4,
            }
            for method in METHODS
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
    for figure in ("fig5_model_method_effects", "fig6_multihead_forgetting"):
        assert (output / f"{figure}.png").is_file()
        assert (output / f"{figure}.pdf").is_file()
        assert (output / f"{figure}.svg").is_file()
    qa = json.loads((output / "figure_qa.json").read_text(encoding="utf-8"))
    assert qa["status"] == "PENDING_MANUAL_VISUAL_AUDIT"

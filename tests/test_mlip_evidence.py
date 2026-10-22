"""Validate evidence contracts using synthetic metrics, never campaign results."""

import json

import pytest

from rsi.mlip_evidence import load_summary, validate_summary


@pytest.fixture
def summary():
    fields = {"energy_mae_eV_per_atom": 1.0, "force_component_mae_eV_per_A": 1.0,
              "force_component_rmse_eV_per_A": 1.0}
    metrics = {arm: dict(fields) for arm in (
        "foundation", "A_random_frozen", "B_random_finetune",
        "C_diagnostic_frozen", "D_diagnostic_finetune")}
    for arm in ("B_random_finetune", "D_diagnostic_finetune"):
        metrics[arm].update(energy_mae_eV_per_atom=2.0, force_component_mae_eV_per_A=0.5)
    return {"metrics": metrics, "provenance": {name: "a" * 64 for name in (
        "dataset_manifest_sha256", "acquisition_sha256", "foundation_model_sha256")},
        "scope": "Synthetic fixture; not a prospective DFT result",
        "decision": {"status": "REJECT_FINE_TUNED_CANDIDATES"}}


def test_synthetic_evidence_is_self_consistent(summary, tmp_path):
    path = tmp_path / "summary.json"
    path.write_text(json.dumps(summary), encoding="utf-8")
    assert validate_summary(load_summary(path))["valid"]


def test_tampered_provenance_or_scope_is_rejected(summary):
    summary["provenance"]["acquisition_sha256"] = "not-a-hash"
    summary["scope"] = "An adsorption benchmark."
    result = validate_summary(summary)
    assert not result["valid"]
    assert "invalid_provenance_hash" in result["errors"]
    assert "scope_missing_nonprospective_boundary" in result["errors"]


def test_energy_regression_cannot_be_promoted_as_a_force_only_win(summary):
    summary["decision"]["status"] = "PROMOTE_FINE_TUNE"
    result = validate_summary(summary)
    assert not result["valid"]
    assert "unexpected_promotion_decision" in result["errors"]

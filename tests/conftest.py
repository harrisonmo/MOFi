"""Synthetic evidence for historical-controller implementation regressions."""

import pytest


@pytest.fixture(autouse=True)
def synthetic_m11_history(request, tmp_path, monkeypatch):
    if request.path.stem not in {"test_m11", "test_report_m11"}:
        return
    from rsi import m11
    from rsi.experiment import write_json

    report = tmp_path / "synthetic-history" / "comparison.json"
    mechanisms = report.parent / "mechanisms.json"
    write_json(report, {
        "summary": {"metrics": {"candidate": {"Y": 9}}},
        "selection": {"promoted": False},
        "candidate": {"workflow": m11.baseline_policy("m8")},
    })
    write_json(mechanisms, {"arms": {
        arm: {"action_counts": {}, "control_counts": {}, "repair_outcomes": {},
              "refusal_cost": 0, "unused_science_units": 0,
              "planning_calls": 0, "feedback_plans": 0}
        for arm in m11.ARMS
    }})
    monkeypatch.setattr(m11, "HISTORY_FILES", (str(report), str(mechanisms)))

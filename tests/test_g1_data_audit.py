"""Check failure modes that could silently contaminate scientific training data."""

import csv
import importlib.util
import json
from pathlib import Path

import pytest


SPEC = importlib.util.spec_from_file_location(
    "g1_audit", Path(__file__).parents[1] / "scripts" / "audit_g1_data.py"
)
AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)


def table(tmp_path, rows, header=None):
    path = tmp_path / "labels.csv"
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(header or AUDIT.LABEL_COLUMNS)
        writer.writerows(rows)
    return path


def test_duplicate_identity_is_not_silently_overwritten(tmp_path):
    summary, values = AUDIT.read_table(table(tmp_path, [["a", 1, 2], ["a", 3, 4]]), AUDIT.LABEL_COLUMNS)
    assert summary["duplicate_ids"] == {"a": 2}
    assert not values


def test_empty_padding_is_allowed_but_hidden_data_is_rejected(tmp_path):
    path = table(tmp_path, [["a", 1, 2, ""], ["b", 1, 2, "hidden"]], AUDIT.LABEL_COLUMNS + [""])
    summary, values = AUDIT.read_table(path, AUDIT.LABEL_COLUMNS)
    assert values == {"a": (1.0, 2.0)}
    assert summary["issues"] == {"invalid_row_shape": 1}


def test_nonfinite_missing_and_whitespace_ids_are_rejected(tmp_path):
    rows = [["a", "nan", 2], ["b", "inf", 3], ["c", "", 4], [" d", 1, 2]]
    summary, values = AUDIT.read_table(table(tmp_path, rows), AUDIT.LABEL_COLUMNS)
    assert not values
    assert summary["issues"] == {"nonfinite": 2, "nonnumeric_or_missing": 1, "invalid_identity": 1}


def test_wrong_pressure_order_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="Unexpected columns"):
        AUDIT.read_table(table(tmp_path, [], list(reversed(AUDIT.LABEL_COLUMNS))), AUDIT.LABEL_COLUMNS)


def test_malformed_duplicate_cannot_leave_a_valid_identity(tmp_path):
    summary, values = AUDIT.read_table(table(tmp_path, [["a", 1, 2], ["a", 3]]), AUDIT.LABEL_COLUMNS)
    assert summary["duplicate_ids"] == {"a": 2}
    assert not values


def test_source_corruption_fails_before_parsing(tmp_path):
    path = table(tmp_path, [["a", 1, 2]])
    inventory = tmp_path / "inventory.json"
    inventory.write_text(json.dumps([{"filename": path.name, "content_details": {
        "size": path.stat().st_size, "sha256_hash": "0" * 64}}]))
    with pytest.raises(ValueError, match="checksum mismatch"):
        AUDIT.audit(tmp_path, inventory)


def test_negative_delta_is_reported_and_pairing_uses_identity(tmp_path):
    label = table(tmp_path, [["a", 3, 1], ["b", 1, 2]])
    label.rename(tmp_path / "Adsorption Labels Data.csv")
    geometry = tmp_path / "Geometry Descriptors Data.csv"
    with geometry.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(AUDIT.GEOMETRY_COLUMNS)
        writer.writerows([["b"] + [1] * 14, ["a"] + [1] * 14])
    entries = [{"filename": p.name, "id": p.name, "content_details": {
        "size": p.stat().st_size, "sha256_hash": AUDIT.sha256(p)}} for p in tmp_path.glob("*.csv")]
    inventory = tmp_path / "inventory.json"
    inventory.write_text(json.dumps(entries))
    result = AUDIT.audit(tmp_path, inventory)
    assert result["pairing"]["numeric_unique_id_intersection"] == 2
    assert result["pairing"]["provisional_nonnegative_pairs"] == 1
    assert result["label_anomalies"]["negative_working_capacity_ids"] == ["a"]
    assert result["training_admitted"] is False

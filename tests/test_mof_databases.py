"""External MOF database catalog: byte-pinned CSV, parsed fields and pure filters."""

import json
from pathlib import Path

import pytest

from rsi.science.mof_databases import MofDatabaseCatalog

ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT / "datasets/catalogs/mofdb.csv"
ANNOTATIONS = ROOT / "datasets/catalogs/mofdb.annotations.json"


@pytest.fixture(scope="module")
def catalog():
    return MofDatabaseCatalog(CSV, ANNOTATIONS)


def by_name(catalog, name):
    return next(row for row in catalog.rows if row["name"] == name)


def test_rows_parse_and_normalize(catalog):
    assert len(catalog.rows) == 26
    core = by_name(catalog, "CoRE MOF 2025")
    assert core["identifiers"] == ["MOFid", "MOFkey"] and core["methods"] == ["experimental", "dft"]
    assert core["volume"] == {"raw": "40000+", "min_count": 40000, "lower_bound": True, "unit": "records"}
    assert core["has"]["topology"] and not core["has"]["negative_data"]
    assert by_name(catalog, "ARC-MOF")["methods"] == ["dft"]  # "(DDEC6)" qualifier dropped
    assert by_name(catalog, "CURATED MOF")["methods"] == ["experimental"]  # "Experimental/Exp" deduplicated
    assert by_name(catalog, "L2M3 Dataset")["volume"]["min_count"] == 2_000_000
    assert by_name(catalog, "L2M3 Dataset")["volume"]["unit"] == "tokens"


def test_link_status_marks_unusable_locators(catalog):
    assert by_name(catalog, "MOFh6")["link_status"] == "doi_placeholder"
    assert by_name(catalog, "MOFCat")["link_status"] == "doi_incomplete"
    usable = catalog.query({"usable_link_only": True})["databases"]
    assert all(row["link_status"] in {"doi", "website"} for row in usable) and len(usable) == 11


def test_filters_combine(catalog):
    result = catalog.query({"identifier": "Refcode", "required_fields": ["synthesis_activation"],
                            "data_type_contains": "synth"})
    assert [row["name"] for row in result["databases"]] == ["FAIR-MOFs", "DigiMOF"]
    assert catalog.query({"name_contains": "qmof"})["matches"] == 1
    with pytest.raises(ValueError):
        catalog.query({"required_fields": ["unknown"]})


def test_changed_csv_or_annotations_are_refused(tmp_path):
    changed = tmp_path / "mofdb.csv"
    changed.write_bytes(CSV.read_bytes().replace(b"40000+", b"50000+"))
    with pytest.raises(ValueError, match="changed"):
        MofDatabaseCatalog(changed, ANNOTATIONS)
    annotations = json.loads(ANNOTATIONS.read_text(encoding="utf-8"))
    del annotations["link_status"]["QMOF"]
    (tmp_path / "annotations.json").write_text(json.dumps(annotations), encoding="utf-8")
    with pytest.raises(ValueError, match="QMOF"):
        MofDatabaseCatalog(CSV, tmp_path / "annotations.json")

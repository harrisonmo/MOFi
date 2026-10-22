"""Read-only catalog of external MOF databases (``datasets/catalogs/mofdb.csv``).

The CSV is a user-supplied survey table: one row per database with its identifier scheme, source
method, data type, size, Y/N coverage flags, formats and license. It is stored byte-for-byte; a
sidecar annotation file pins its SHA-256 and records a link status per row, so an edited CSV is
refused until the annotations are reviewed again.

The catalog is metadata for planning (which database could supply structures, labels or negative
data for a task). It contains no structures or property values, and none of its entries has been
verified against the source publications.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parent
DEFAULT_CSV = ROOT / "data/mofdb.csv"
DEFAULT_ANNOTATIONS = ROOT / "data/mofdb.annotations.json"
COLUMNS = ["Name", "Standard ID", "Update", "Link", "Method (Exp/DFT/GCMC)", "Data Type (Ads/Cat/Elec)",
           "Volume", "Negative Data (Y/N)", "Topology", "LCD/PLD", "ASA/VF/Density", "Stability",
           "Synthesis/Activation", "Format", "License", "API"]
# Y/N columns and the keys the model filters on.
FLAGS = {"negative_data": "Negative Data (Y/N)", "topology": "Topology", "lcd_pld": "LCD/PLD",
         "asa_vf_density": "ASA/VF/Density", "stability": "Stability",
         "synthesis_activation": "Synthesis/Activation", "api": "API"}
METHOD_ALIASES = {"exp": "experimental", "ai": "ml"}
LINK_STATUS = {"doi", "doi_incomplete", "doi_placeholder", "arxiv_placeholder", "website", "label_only"}
USABLE_LINKS = {"doi", "website"}
VOLUME = re.compile(r"^(\d+(?:\.\d+)?)([KkMm])?(\+)?(?:\s+([A-Za-z]+))?$")
CAVEAT = ("Unverified survey metadata supplied by the user: names, links, sizes, dates, licenses and Y/N "
          "flags were not checked against the sources. Use it to choose where to look, not as evidence; "
          "the hub cannot download any of these databases.")


def _split(value):
    return [part.strip() for part in value.split("/") if part.strip()]


def _methods(value):
    tokens = []
    for part in _split(re.sub(r"\([^)]*\)", "", value)):
        token = METHOD_ALIASES.get(part.lower(), part.lower())
        if token not in tokens:
            tokens.append(token)
    return tokens


def _volume(value):
    match = VOLUME.fullmatch(value.strip())
    if not match:
        raise ValueError(f"Unparseable Volume {value!r}")
    number, scale, plus, unit = match.groups()
    count = float(number) * {"k": 1e3, "m": 1e6}.get((scale or "").lower(), 1)
    return {"raw": value, "min_count": int(count), "lower_bound": bool(plus),
            "unit": (unit or "records").lower()}


def _flag(value, name, column):
    if value not in {"Y", "N"}:
        raise ValueError(f"{name}: {column} must be Y or N, got {value!r}")
    return value == "Y"


class MofDatabaseCatalog:
    """Parsed, validated rows; every query is a pure filter over them."""

    def __init__(self, csv_path=DEFAULT_CSV, annotations_path=DEFAULT_ANNOTATIONS):
        self.csv_path = Path(csv_path).resolve()
        self.annotations_path = Path(annotations_path).resolve()
        raw = self.csv_path.read_bytes()
        self.sha256 = hashlib.sha256(raw).hexdigest()
        annotations = json.loads(self.annotations_path.read_text(encoding="utf-8"))
        if annotations.get("schema_version") != 1:
            raise ValueError("Unsupported MOF database annotation schema")
        if annotations["csv_sha256"] != self.sha256:
            raise ValueError(f"{self.csv_path.name} changed (SHA-256 {self.sha256}); review "
                             f"{self.annotations_path.name} before using it")
        reader = csv.DictReader(io.StringIO(raw.decode("utf-8-sig")))
        if reader.fieldnames != COLUMNS:
            raise ValueError(f"Unexpected columns {reader.fieldnames}")
        links = annotations["link_status"]
        self.rows = [self._row(record, links) for record in reader]
        names = [row["name"] for row in self.rows]
        if len(set(names)) != len(names):
            raise ValueError("Database names must be unique")
        if set(names) != set(links):
            raise ValueError(f"Annotations and CSV rows differ: {sorted(set(names) ^ set(links))}")
        self.provenance = annotations["provenance"]
        self.verification = annotations["verification"]

    @staticmethod
    def _row(record, links):
        if None in record or any(value is None for value in record.values()):
            raise ValueError(f"Row {record.get('Name')!r} does not have {len(COLUMNS)} fields")
        record = {key: value.strip() for key, value in record.items()}
        name = record["Name"]
        status = links.get(name)
        if status not in LINK_STATUS:
            raise ValueError(f"{name}: link status {status!r} is not one of {sorted(LINK_STATUS)}")
        return {"name": name, "identifiers": _split(record["Standard ID"]), "updated": record["Update"],
                "link": record["Link"], "link_status": status,
                "methods": _methods(record["Method (Exp/DFT/GCMC)"]),
                "method_raw": record["Method (Exp/DFT/GCMC)"],
                "data_type": record["Data Type (Ads/Cat/Elec)"], "volume": _volume(record["Volume"]),
                "has": {key: _flag(record[column], name, column) for key, column in FLAGS.items()},
                "formats": _split(record["Format"]), "license": record["License"]}

    def options(self):
        """Values the tool schema offers as enums."""
        def ordered(key):
            return sorted({value for row in self.rows for value in row[key]})
        return {"methods": ordered("methods"), "identifiers": ordered("identifiers"), "fields": list(FLAGS)}

    def source(self):
        return {"path": self.csv_path.relative_to(ROOT).as_posix() if self.csv_path.is_relative_to(ROOT)
                else str(self.csv_path), "sha256": self.sha256, "rows": len(self.rows),
                "provenance": self.provenance, "verification": self.verification, "caveat": CAVEAT}

    def query(self, arguments):
        """Filter rows; every argument is optional (null) and all given filters must hold."""
        name = (arguments.get("name_contains") or "").lower()
        data_type = (arguments.get("data_type_contains") or "").lower()
        method, identifier = arguments.get("method"), arguments.get("identifier")
        fields = arguments.get("required_fields") or []
        unknown = set(fields) - set(FLAGS)
        if unknown:
            raise ValueError(f"Unknown fields {sorted(unknown)}; use {list(FLAGS)}")
        matches = [row for row in self.rows
                   if name in row["name"].lower() and data_type in row["data_type"].lower()
                   and (method is None or method in row["methods"])
                   and (identifier is None or identifier in row["identifiers"])
                   and all(row["has"][field] for field in fields)
                   and (not arguments.get("usable_link_only") or row["link_status"] in USABLE_LINKS)]
        return {"source": self.source(), "filters": arguments, "matches": len(matches), "databases": matches}

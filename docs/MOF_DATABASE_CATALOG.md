# External MOF database catalog


## What it is

[`datasets/catalogs/mofdb.csv`](../datasets/catalogs/mofdb.csv) is a survey table of 26 external MOF databases supplied by the user (CoRE MOF 2025, QMOF, ARC-MOF, hMOF, CSD MOF, FAIR-MOFs, DigiMOF, NIST ISODB and others). Each row lists the database's identifier scheme, update date, link, source method, data type, size, seven Y/N coverage flags, file formats, license and API availability.

It is **planning metadata**: it tells an agent which database could supply structures, labels or negative data for a task and how databases cross-link (MOFid, Refcode). It contains no structures or property values, and the harness cannot download any of the listed databases.

## Provenance and verification

- The CSV is stored byte-for-byte as received (SHA-256 `2fe064e6…2a08cf9`, 26 rows, ASCII, CRLF line endings, no trailing newline; `.gitattributes` stores it as binary-exact `-text` so every checkout has the same hash). No value was edited.
- [`mofdb.annotations.json`](../datasets/catalogs/mofdb.annotations.json) pins that hash and assigns every row a `link_status` by format inspection: 9 `doi`, 8 `doi_incomplete` (journal prefix only, e.g. `10.1021/acscatal`), 1 `doi_placeholder` (`…4cxxxxx`), 1 `arxiv_placeholder`, 2 `website`, 5 `label_only` (e.g. `CCDC Website`, `GitHub Repo`).
- **Nothing was verified against the sources.** No DOI or URL was resolved, and record counts, dates, licenses and flags are as supplied. Some names (for example USMOF DB, BW-DB) have no locator at all. Treat every field as a lead to check, never as evidence for a scientific claim.

## How the harness uses it

### Through the science hub (agents)

`configs/science-hub.json` registers the catalog under `tools.mof_database_catalog` with `csv` and `annotations` paths (relative to the config). When the hub starts it:

1. Parses and validates the CSV (exact columns, Y/N flags, parseable sizes) and refuses it if its SHA-256 differs from the annotations or any row lacks a link status. A refused catalog is listed in `unavailable_tools` with the reason; the other tools are unaffected.
2. Exposes `mof_database_catalog` in `hub.catalog()["tools"]` (OpenAI function format, `strict: true`), filling the enums from the data: normalized methods, identifier schemes and coverage-flag names. A guidance line states that the table is unverified metadata.
3. Adds the CSV, the annotations and `rsi/science/mof_databases.py` to the session binding, so changing any of them refuses to reopen an existing session and blocks calls in a running one.

A call is **free**: it does not count against `max_calls`, launches no HPC job, creates no call folder and is not written to the hub ledger. The result envelope carries the CSV hash, so a transcript records which table the agent saw.

Arguments (all required; pass `null` for an unused filter; all given filters must hold):

| Argument | Type | Meaning |
|---|---|---|
| `name_contains` | string / null | Case-insensitive substring of the name |
| `method` | enum / null | `calculated`, `dft`, `experimental`, `gcmc`, `geometric`, `md`, `ml`, `multi-modal`, `simulation`, `text` |
| `data_type_contains` | string / null | Case-insensitive substring of the data type (`Ads`, `Synth`, `Stability`, …) |
| `identifier` | enum / null | Identifier scheme such as `MOFid`, `Refcode`, `SMILES`, `NIST-ID` |
| `required_fields` | array of enum / null | Flags that must all be Y: `negative_data`, `topology`, `lcd_pld`, `asa_vf_density`, `stability`, `synthesis_activation`, `api` |
| `usable_link_only` | boolean / null | `true` keeps only `doi` and `website` rows |

Each returned database has `name`, `identifiers`, `updated`, `link`, `link_status`, `methods`, `method_raw`, `data_type`, `volume` (`raw`, `min_count`, `lower_bound` for "N+", `unit` = `records` or `tokens`), `has` (the seven flags), `formats` and `license`. The `source` block repeats the path, hash, row count, provenance and caveat.

Normalization rules: methods are split on `/`, parenthetical qualifiers dropped (`DFT (DDEC6)` → `dft`), `Exp` → `experimental`, `AI` → `ml`, duplicates removed; identifiers and formats are split on `/`; `40000+` → `min_count` 40000 with `lower_bound` true; `2M Tokens` → 2,000,000 `tokens`.

```bash
python -m rsi.science.hub call --session runs/science-hub-default --tool mof_database_catalog --arguments '{"name_contains": null, "method": null, "data_type_contains": null, "identifier": "MOFid", "required_fields": ["topology", "lcd_pld"], "usable_link_only": true}'
```

returns CoRE MOF 2025, ARC-MOF, QMOF and MOSAEC-DB.

In an agent controller, nothing extra is needed: pass `hub.catalog()["tools"]` as the tool list and forward each call to `hub.call(name, arguments)`, as for every other hub tool.

### Directly (controllers, scripts, notebooks)

```python
from rsi.science.mof_databases import MofDatabaseCatalog

catalog = MofDatabaseCatalog()  # defaults to datasets/catalogs/mofdb.csv and its annotations
catalog.query({"method": "experimental", "required_fields": ["synthesis_activation"]})
```

```bash
python -m rsi.science.mof_databases summary
python -m rsi.science.mof_databases query --arguments '{"data_type_contains": "stability"}'
```

The standalone module needs only the standard library.

## Rules for experiments

- A protocol that lets an agent use the catalog must say so and record the CSV hash, as for any other registered input.
- An agent choosing a data source from the catalog must still obtain and verify the actual data through an admitted route (the project's data admission gates); the catalog row is not admission.
- Do not report catalog sizes, licenses or coverage flags as facts without checking the source.

## Updating the table

1. Replace `datasets/catalogs/mofdb.csv` (keep the column header exactly).
2. Review every row's link, update `link_status` in the annotations for added or renamed rows, and set `csv_sha256` to the new hash. Until both match, the loader refuses the table.
3. Run `tests/test_mof_databases.py` and `tests/test_science_hub.py`; adjust the expectations that name specific rows.
4. Existing hub sessions bound to the old table cannot be reopened; start a new session.

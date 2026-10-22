"""Run one admitted M8 successor comparison on the frozen public M7 pools."""

import argparse
import hashlib
import importlib.metadata
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from rsi.experiment import ROOT, cli_identity, git_state, read_json, write_json
from rsi.m6 import ThreeArmExperiment
from rsi.m10 import ARMS, CONFIG, Budget, selection, summarize
from rsi.m11_evidence import check_saved_call
from rsi.m11_workflow import (
    POLICY_SCHEMA, PROPOSAL_SCHEMA, SEMANTICS, WorkflowScreeningExperiment,
    baseline_policy, policy_hash,
)

CRYSTAL_COMMIT = "df014447f0812dd4f55ed60359b71a981c3eeb60"
ADMISSION_PROMPT = 'Return {"ready":true}. This is a transport/schema admission probe only; do not use tools.'
ADMISSION_SCHEMA = {"type": "object", "additionalProperties": False,
                    "properties": {"ready": {"type": "boolean"}}, "required": ["ready"]}
ADMISSION_LIMITS = {"original": 10000, "local-v2": 30000}
SCIENCE_PACKAGES = ("numpy", "scipy", "ase", "spglib", "gemmi", "crystal-builder")
SCIENCE_CHECKS = {
    "worker_completed_expected_contract", "exactly_one_paid_action", "no_calculation_or_model_call",
    "checkpoint_sequence", "checkpoint_integrity", "diagnostics_cleared",
    "framework_and_cell_preserved", "expected_atom_count", "noop_preserves_clean_structure",
}
HISTORY_FILES = ("docs/reports/M10_RESULT.json", "docs/reports/M10_MECHANISMS.json")
RUNNER_FILES = (
    "rsi/m8.py", "rsi/m8_workflow.py", "rsi/m10.py", "rsi/m10_workflow.py",
    "rsi/m11.py", "rsi/m11_workflow.py", "rsi/m11_evidence.py", "tests/test_m11.py",
    "docs/M11_WORKFLOW_PROTOCOL.md", "docs/reports/M7_RESULT.json",
    "workflows/m5-accepted.json", "workflows/m8-screening-accepted.json",
    "requirements-controller.txt",
)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def admission_limit(manifest):
    """Resolve a versioned allocation without changing historical probe limits."""
    protocol = manifest.get("admission_protocol", "original")
    if protocol not in ADMISSION_LIMITS:
        raise ValueError("Unknown admission protocol")
    limit = ADMISSION_LIMITS[protocol]
    if manifest.get("max_observed_tokens") != limit:
        raise ValueError("Admission allocation differs from its versioned protocol")
    return limit


def validate_admission(path):
    """Require a successful, fully accounted prior CLI admission probe."""
    record = read_json(path)
    directory = Path(path).parent
    manifest = read_json(directory / "manifest.json")
    limit = admission_limit(manifest)
    if (record.get("admitted") is not True or record.get("usage_known") is not True
            or record.get("status") != "ok" or type(record.get("tokens")) is not int
            or not 0 <= record["tokens"] <= limit):
        raise ValueError("M11 admission failed or usage is unknown; no model calls authorized")
    result = read_json(directory / "probe/result.json")
    command = read_json(directory / "probe/command.json")
    usage = result.get("usage") or {}
    if (result.get("status") != "ok" or result.get("exit_code") != 0
            or result.get("unexpected_tool_types") != []
            or result.get("requested_model") != CONFIG["model"]
            or result.get("actual_model") not in {None, CONFIG["model"]}
            or command.get("requested_model") != CONFIG["model"]
            or command.get("reasoning") != CONFIG["reasoning"]
            or not all(type(usage.get(key)) is int and usage[key] >= 0
                       for key in ["input_tokens", "output_tokens"])
            or usage["input_tokens"] + usage["output_tokens"] != record["tokens"]
            or not all(isinstance(manifest.get("cli", {}).get(key), str) and manifest["cli"][key]
                       for key in ["path", "sha256", "version"])):
        raise ValueError("M11 admission raw evidence does not establish a known successful Astra/high call")
    # Reject inconsistent evidence before constructing any model or scientific action.
    verified, tokens = check_saved_call(directory / "probe", {
        "model": CONFIG["model"], "reasoning": CONFIG["reasoning"], "cli_timeout_seconds": 120,
    }, ADMISSION_PROMPT, ADMISSION_SCHEMA)
    if verified["response"] != {"ready": True} or tokens != record["tokens"]:
        raise ValueError("M11 admission probe did not return ready=true with matching usage")
    ledger = read_json(directory / "ledger.json")
    if (ledger != [{"label": "admission", "status": "known", "usage": usage, "result_status": "ok"}]
            or manifest.get("max_calls") != 1
            or manifest.get("science_units") != 0):
        raise ValueError("M11 admission allocation or ledger differs from the probe")
    return record


def frozen_hashes():
    """Verify historical scientific bytes, allowing only documented CRLF migration."""
    crystal = git_state(ROOT / "Crystal-Builder")
    if crystal["commit"] != CRYSTAL_COMMIT or crystal["status"]:
        raise ValueError("Crystal-Builder must be clean at the frozen scientific commit")
    prior = read_json(ROOT / "docs/reports/M7_RESULT.json")["manifest"]
    hashes, newline_matches = {}, []
    for name, expected in prior["source_hashes"].items():
        content = (ROOT / name).read_bytes()
        actual = hashlib.sha256(content).hexdigest()
        if actual != expected:
            legacy = content.replace(b"\r\n", b"\n").replace(b"\n", b"\r\n")
            if hashlib.sha256(legacy).hexdigest() != expected:
                raise ValueError("Frozen M7 source changed: " + name)
            newline_matches.append(name)
        hashes[name] = actual
    for name, expected in prior["fixture_hashes"].items():
        relative = "benchmarks/m7/fixtures/" + name
        if digest(ROOT / relative) != expected:
            raise ValueError("Frozen M7 fixture changed: " + name)
        hashes[relative] = expected
    pools = read_json(ROOT / "benchmarks/m7/pools.json")["pools"]
    if hashlib.sha256(json.dumps(pools, sort_keys=True).encode()).hexdigest() != prior["pools_sha256"]:
        raise ValueError("Frozen M7 pools changed")
    files = set(RUNNER_FILES) | set(HISTORY_FILES)
    files.update(p.relative_to(ROOT).as_posix() for p in (ROOT / "benchmarks/m7").rglob("*") if p.is_file())
    # Optional report helpers are frozen when present, without requiring old runs.
    files.update(p.relative_to(ROOT).as_posix() for p in (ROOT / "scripts").glob("*m11*.py"))
    files.update(p.relative_to(ROOT).as_posix() for p in (ROOT / "tests").glob("*m11*.py"))
    for name in files:
        hashes[name] = digest(ROOT / name)
    return hashes, {
        "legacy_newline_matches": sorted(newline_matches),
        "legacy_source_count": len(prior["source_hashes"]),
        "normalization": "Only LF/CRLF source line endings; fixtures remain byte-exact",
        "historical_raw_runs_available": False,
        "historical_evidence_scope": "Versioned M10 aggregate reports; original historical runs are unavailable and not re-audited",
    }


def verify_frozen(hashes):
    for name, expected in hashes.items():
        if digest(ROOT / name) != expected:
            raise RuntimeError("Frozen experiment content changed: " + name)


def feedback():
    """Use versioned development summaries without hidden material identities."""
    report = read_json(ROOT / HISTORY_FILES[0])
    mechanisms = read_json(ROOT / HISTORY_FILES[1])
    fields = ("action_counts", "control_counts", "repair_outcomes", "refusal_cost",
              "unused_science_units", "planning_calls", "feedback_plans")
    return {
        "scope": "Public development feedback from versioned aggregate reports only",
        "historical_raw_runs_available": False,
        "provenance": {name: digest(ROOT / name) for name in HISTORY_FILES},
        "historical_summary": report["summary"],
        "historical_selection": report["selection"],
        "rejected_m10_workflow": report["candidate"]["workflow"],
        "mechanisms": {arm: {key: mechanisms["arms"][arm][key] for key in fields} for arm in ARMS},
        "limitations": "Saved reports describe historical outcomes, not newly replayed evidence or counterfactual repair success. No held-out evidence is supplied.",
    }


def proposal_prompt(data):
    return (
        "You are the Evolution Loop. Propose exactly one changed screening workflow and an English hypothesis. "
        "The accepted parent is M8; M10 was rejected. Only closed numeric, boolean and enum policy fields may change. "
        "No material identity or candidate-ID rules, free-form executable instructions, new tools, model changes, "
        "scientific prompt changes, grading changes or budget changes. The hypothesis is recorded but never sent "
        "to the Research Loop. Use the versioned aggregate M10 development reports below; original raw historical "
        "runs are unavailable and have not been re-audited. Investigate selective paid inspection, affordable "
        "duplicate preparation and preserving useful queued completion actions. Repairs need not succeed. "
        "All scientific actions remain charged; failed optimization costs 3 units per requested step. "
        "One proposal is frozen before all 18 fresh batch/M8/candidate episodes across six public pools, "
        "with 12 science units, 13 model calls and 325000 observed tokens per episode. No tuning during comparison. "
        "Promotion requires every report reliable, known usage, no per-pool yield regression against either control, "
        "and total yield strictly greater than both with tokens at most 1.5 times M8; alternatively equal yield "
        "to both with at least 10% fewer tokens and no more calls than either. Public development is not held-out "
        "generalization.\n" + SEMANTICS + "\n" + json.dumps({"parent": baseline_policy("m8"),
            "schema": POLICY_SCHEMA, "development": data}, ensure_ascii=False))


def admission_sources(path):
    sources = {"summary.json": Path(path)}
    sources.update({name: Path(path).parent / name for name in
                    ["manifest.json", "probe/result.json", "probe/command.json"]})
    for name in ["ledger.json", "probe/schema.json", "probe/prompt.txt", "probe/events.jsonl", "probe/final.json"]:
        sources[name] = Path(path).parent / name
    return {name: {"path": str(source), "sha256": digest(source)} for name, source in sources.items()}


def verify_admission_sources(sources):
    if any(digest(record["path"]) != record["sha256"] for record in sources.values()):
        raise RuntimeError("Frozen admission evidence changed")


def science_source_hashes(hashes):
    """Select runtime science dependencies, excluding evolving reports and tests."""
    return {name: sha for name, sha in hashes.items() if
            (name.startswith("rsi/") and name.endswith(".py"))
            or name.startswith("Crystal-Builder/xtal/")
            or name.startswith("benchmarks/m7/fixtures/")
            or name == "scripts/check_m11_science.py"}


def preparation_contract_ok(result, variant):
    """Accept completed preparation or an explicit unchanged clean-base no-op."""
    if result.get("ok") is True:
        return True
    diagnostics = result.get("diagnostics", [])
    before, after = result.get("atoms_before"), result.get("atoms_after")
    return (variant == "base" and result.get("ok") is False
            and type(before) is int and type(after) is int and before > 0 and before == after
            and isinstance(diagnostics, list) and bool(diagnostics)
            and all(isinstance(item, dict) and isinstance(item.get("level"), str)
                    and item["level"].lower() != "error" for item in diagnostics)
            and any(item.get("code") == "NOTHING_TO_DO" for item in diagnostics))


def validate_science_admission(path, current_hashes=None, packages=None):
    """Verify completed preparation admission and its saved worker evidence."""
    path = Path(path)
    directory = path.parent
    summary = read_json(path)
    expected_totals = {"prepare_actions": 3, "science_units": 4, "requested_uff_steps": 0,
                       "model_calls": 0, "checks_passed": 3 * len(SCIENCE_CHECKS)}
    if (summary.get("admitted") is not True or any(type(summary.get(key)) is not int
            or summary[key] != value for key, value in expected_totals.items())):
        raise ValueError("M11 scientific admission is missing, incomplete, or over budget")
    manifest = read_json(directory / "manifest.json")
    rows = read_json(directory / "rows.json")
    ledger = read_json(directory / "ledger.json")
    if current_hashes is None:
        current_hashes = frozen_hashes()[0]
    if packages is None:
        packages = {name: importlib.metadata.version(name) for name in SCIENCE_PACKAGES}
    expected_hashes = science_source_hashes(current_hashes)
    if (not expected_hashes or science_source_hashes(manifest.get("source_and_input_hashes", {})) != expected_hashes
            or any(manifest.get("packages", {}).get(name) != packages[name] for name in SCIENCE_PACKAGES)
            or manifest.get("crystal_builder") != {"commit": CRYSTAL_COMMIT, "status": ""}
            or manifest.get("config") != {"science_timeout_seconds": CONFIG["science_timeout_seconds"],
                                           "science_seed": CONFIG["science_seed"]}
            or manifest.get("limits") != {key: expected_totals[key] for key in
                                           ["prepare_actions", "science_units", "requested_uff_steps", "model_calls"]}):
        raise ValueError("M11 scientific admission source, packages, or environment changed")
    verify_frozen(expected_hashes)
    family = sorted((ROOT / "benchmarks/m7/fixtures").glob("*-base.xtalproj"))[0].name.removesuffix("-base.xtalproj")
    if (manifest.get("family") != family or len(rows) != 3 or len(ledger) != 3
            or summary.get("rows") != rows):
        raise ValueError("M11 scientific admission requires all three declared cases")
    for index, (variant, cost) in enumerate((("base", 1), ("duplicate", 1), ("compound", 2))):
        row, entry = rows[index], ledger[index]
        filename = f"{family}-{variant}.xtalproj"
        action = entry.get("action", {})
        steps = ["duplicates", "deuterium"] if variant == "compound" else ["duplicates"]
        if (row.get("fixture") != filename or row.get("passed") is not True
                or set(row.get("checks", {})) != SCIENCE_CHECKS
                or not all(value is True for value in row.get("checks", {}).values())
                or entry.get("fixture") != filename or entry.get("status") != "completed"
                or entry.get("passed") is not True or action != {"verb": "prepare", "steps": steps}
                or any(type(record.get("science_units")) is not int or record["science_units"] != cost
                       for record in [row, entry])):
            raise ValueError("M11 scientific admission ledger or case checks failed")
        case = directory / variant
        task = {"id": "m11-preparation-" + variant,
                "initialization": {"kind": "open", "root": "m7_fixture", "path": filename}}
        initial = read_json(case / "initial/result.json")
        result = read_json(case / "prepare/result.json")
        for stage, actions, raw in [("initial", [], initial), ("prepare", [action], result)]:
            if (read_json(case / stage / "task.json") != task
                    or read_json(case / stage / "actions.json") != actions
                    or raw.get("task_id") != task["id"] or raw.get("errors") != []
                    or raw.get("attempted_actions") != actions
                    or raw.get("stopped") is not False
                    or raw.get("artifact", {}).get("sha256") != digest(case / stage / "structure.xtalproj")):
                raise ValueError("M11 scientific admission raw worker evidence differs")
        if (initial.get("events") != [] or read_json(case / "prepare/previous.json") != initial
                or len(result.get("events", [])) != 1 or result["events"][0].get("action") != action
                or not preparation_contract_ok(result["events"][0].get("result", {}), variant)
                or [item.get("action_count") for item in initial.get("checkpoint_history", [])] != [0]
                or [item.get("action_count") for item in result.get("checkpoint_history", [])] != [0, 1]
                or result["checkpoint_history"][0] != initial["checkpoint_history"][0]
                or any(raw["checkpoint_history"][-1].get(key) != raw["artifact"].get(key)
                       for raw in [initial, result] for key in ["path", "sha256"])
                or result.get("resumed_from") != initial.get("artifact")):
            raise ValueError("M11 scientific admission checkpoint or action sequence differs")
        codes = sorted({item["code"] for item in result.get("final", {}).get("diagnostics", [])})
        if (set(codes) & {"COINCIDENT_ATOMS", "DEUTERIUM"} or row.get("current_codes") != codes
                or ((case / "ledger.json").exists() and read_json(case / "ledger.json") != [])):
            raise ValueError("M11 scientific admission has unresolved diagnostics or model calls")
    return summary


def science_admission_sources(path):
    directory = Path(path).parent
    sources = {p.relative_to(directory).as_posix(): {"path": str(p), "sha256": digest(p)}
               for p in directory.rglob("*") if p.is_file()}
    sources["summary.json"] = {"path": str(path), "sha256": digest(path)}
    return sources


def run(output, admission_summary, science_admission_summary):
    admission_summary = Path(admission_summary).resolve()
    admission = validate_admission(admission_summary)
    admission_files = admission_sources(admission_summary)
    admission_cli = read_json(admission_summary.parent / "manifest.json")["cli"]
    root_state = git_state(ROOT)
    if root_state["status"]:
        raise ValueError("Commit a clean M11 implementation before dependent model calls")
    hashes, freeze_audit = frozen_hashes()
    packages = {name: importlib.metadata.version(name) for name in
                ["numpy", "scipy", "ase", "spglib", "gemmi", "crystal-builder", "jsonschema", "psutil"]}
    science_admission_summary = Path(science_admission_summary).resolve()
    science_admission = validate_science_admission(science_admission_summary, hashes, packages)
    science_admission_files = science_admission_sources(science_admission_summary)
    pools = read_json(ROOT / "benchmarks/m7/pools.json")["pools"]
    if (len(pools) != 6 or {p["id"] for p in pools} != {f"m7-dev-{i:02d}" for i in range(1, 7)}
            or any(p["split"] != "development" for p in pools)):
        raise ValueError("Only the six frozen development pools may run")
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    budget = Budget(output / "stage-ledger.json")
    proposer = ThreeArmExperiment(dict(CONFIG, max_llm_calls=1, max_observed_tokens=100000), output / "evolution", stage=budget)
    data = feedback()
    write_json(output / "feedback.json", data)
    write_json(output / "admission-summary.json", admission)
    for name, record in admission_files.items():
        destination = output / "admission-evidence" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(Path(record["path"]).read_bytes())
    for name, record in science_admission_files.items():
        destination = output / "science-admission-evidence" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(Path(record["path"]).read_bytes())
    current_cli = cli_identity(proposer.backend.executable)
    if current_cli != admission_cli:
        raise RuntimeError("CLI differs from the admitted executable")
    schedule = [(pool, arm) for i, pool in enumerate(pools) for arm in ARMS[i % 3:] + ARMS[:i % 3]]
    manifest = {
        "started_utc": datetime.now(timezone.utc).isoformat(), "root": root_state,
        "crystal_builder": git_state(ROOT / "Crystal-Builder"),
        "cli": current_cli, "config": CONFIG,
        "python": sys.version, "platform": platform.platform(), "packages": packages,
        "source_and_input_hashes": hashes, "freeze_audit": freeze_audit,
        "baseline_policies": {arm: baseline_policy(arm) for arm in ["batch", "m8"]},
        "historical_feedback_sources": data["provenance"],
        "feedback_sha256": digest(output / "feedback.json"),
        "admission": {"path": str(admission_summary), "sha256": digest(admission_summary), "summary": admission,
                      "cli": admission_cli, "sources": admission_files},
        "science_admission": {"path": str(science_admission_summary), "sha256": digest(science_admission_summary),
                              "summary": science_admission, "sources": science_admission_files},
        "schedule": [{"pool": pool["id"], "arm": arm} for pool, arm in schedule],
        "scope": "One frozen autonomous proposal; fresh public-development comparison; not held-out generalization",
    }
    write_json(output / "manifest.json", manifest)
    write_json(output / "pools.json", pools)
    verify_frozen(hashes)
    verify_admission_sources(admission_files)
    verify_admission_sources(science_admission_files)
    if git_state(ROOT) != root_state:
        raise RuntimeError("Root repository changed before proposal")
    proposal = proposer.model(proposal_prompt(data), PROPOSAL_SCHEMA, output / "evolution/proposal")
    sha = policy_hash(proposal["workflow"])
    write_json(output / "candidate.json", {**proposal, "sha256": sha})
    if proposer.observed_tokens > 100000 or any(e["status"] != "known" for e in budget.entries):
        raise RuntimeError("Outer proposal exceeded budget or usage is unknown; comparison stopped")
    rows = []
    unchanged = sha == policy_hash(baseline_policy("m8"))
    for pool, arm in ([] if unchanged else schedule):
        verify_frozen(hashes)
        verify_admission_sources(admission_files)
        verify_admission_sources(science_admission_files)
        if (policy_hash(read_json(output / "candidate.json")["workflow"]) != sha
                or digest(output / "feedback.json") != manifest["feedback_sha256"]):
            raise RuntimeError("Frozen candidate or feedback changed")
        exp = WorkflowScreeningExperiment(CONFIG, output / pool["id"] / arm, stage=budget)
        if cli_identity(exp.backend.executable) != manifest["cli"]:
            raise RuntimeError("CLI changed")
        policy = proposal["workflow"] if arm == "candidate" else baseline_policy(arm)
        started = time.monotonic()
        try:
            row = exp.episode(pool, arm, policy)
        except Exception as exc:
            row = {"pool": pool["id"], "arm": arm, "error": f"{type(exc).__name__}: {exc}"}
        path = exp.output / "pool-state.json"
        state = read_json(path) if path.exists() else {"spent": 0, "events": []}
        row["cost"] = {
            "calls": exp.calls, "tokens": exp.observed_tokens, "units": state["spent"],
            "requested_steps": sum(event["action"].get("max_steps", 0) for event in state["events"]),
            "unknown_usage": sum(not all(type(entry.get("usage", {}).get(key)) is int
                and entry["usage"][key] >= 0 for key in ["input_tokens", "output_tokens"]) for entry in exp.ledger),
            "wall_seconds": time.monotonic() - started,
        }
        rows.append(row)
        write_json(exp.output / "graded.json", row)
        write_json(output / "rows.json", rows)
        write_json(output / "summary.json", summarize(rows))
        print(pool["id"], arm, row.get("score", row.get("error")), row["cost"]["tokens"], flush=True)
        if row["cost"]["unknown_usage"] or any(entry["status"] != "known" for entry in budget.entries):
            break
    verify_frozen(hashes)
    decision = {"promoted": False, "reason": "unchanged_policy"} if unchanged else selection(rows)
    write_json(output / "rows.json", rows)
    write_json(output / "summary.json", summarize(rows))
    write_json(output / "selection.json", decision)
    selected = proposal["workflow"] if decision["promoted"] else baseline_policy("m8")
    write_json(output / "accepted.json", {"workflow": selected, "sha256": policy_hash(selected),
               "source": "candidate" if decision["promoted"] else "m8", "scope": "M7 public development"})
    print(json.dumps(decision), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--admission-summary", type=Path, required=True)
    parser.add_argument("--science-admission-summary", type=Path, required=True)
    args = parser.parse_args()
    preexisting = args.output.exists()
    try:
        run(args.output, args.admission_summary, args.science_admission_summary)
    except Exception as exc:
        if not preexisting and args.output.exists():
            write_json(args.output / "failure.json", {"error": f"{type(exc).__name__}: {exc}"})
        raise

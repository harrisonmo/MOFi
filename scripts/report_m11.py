"""Audit a completed M11 comparison using saved decisions and artifacts only."""

import argparse
import json
import sys
from pathlib import Path

from jsonschema import validate

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rsi import m11
from rsi.backends.codex_cli import DECISION_INSTRUCTIONS, FORBIDDEN_ITEMS
from rsi.experiment import read_json
from rsi.m11_workflow import PROPOSAL_SCHEMA, WorkflowScreeningExperiment, baseline_policy, policy_hash


from rsi.m11_evidence import AuditError, require, known_tokens, check_saved_call


class ReplayBackend:
    def __init__(self, source, destination, config, records):
        self.source, self.destination, self.config, self.records = source, destination, config, records

    def run(self, prompt, schema, directory):
        original = self.source / directory.relative_to(self.destination)
        for key in ['"qualified"', '"canonical"', '"framework_id"', '"reference_actions"']:
            require(key not in prompt, "Private evaluator field exposed")
        result, tokens = check_saved_call(original, self.config, prompt, schema)
        self.records.append({"label": str(original), "status": "known", "tokens": tokens})
        return result


def check_comparison(rows, schedule):
    expected = {(f"m7-dev-{i:02d}", arm) for i in range(1, 7) for arm in m11.ARMS}
    keys = [(row.get("pool"), row.get("arm")) for row in rows]
    require(len(keys) == 18 and len(set(keys)) == 18 and set(keys) == expected, "Incomplete or wrong comparison pools")
    require([{"pool": pool, "arm": arm} for pool, arm in keys] == schedule, "Comparison schedule differs")
    require(not any(row.get("error") for row in rows), "Comparison contains an execution failure")
    require(all(row.get("score", {}).get("reliable") is True for row in rows), "Comparison contains an unreliable report")
    require(all(type(row.get("cost", {}).get("unknown_usage")) is int and row["cost"]["unknown_usage"] == 0
                for row in rows), "Comparison contains unknown usage")


def check_stage(stage, records):
    require(stage == records, "Stage ledger differs from the ordered saved CLI calls")
    require(len(records) <= 235, "Stage call ceiling exceeded")
    tokens = 0
    for record in records:
        require(record.get("status") == "known" and type(record.get("tokens")) is int and record["tokens"] >= 0,
                "Unknown stage usage")
        require(tokens < 5950000, "Call started after the stage token ceiling")
        tokens += record["tokens"]
    require(tokens <= 5950000, "Stage token ceiling exceeded")
    return tokens


def check_evidence_copy(run, manifest, kind):
    record = manifest[kind]
    evidence = run / (kind.replace("_", "-") + "-evidence")
    require("summary.json" in record["sources"], "Admission summary source is missing")
    for name, source in record["sources"].items():
        path = (evidence / name).resolve()
        require(path.is_relative_to(evidence.resolve()), "Admission evidence path escapes its archive")
        require(m11.digest(path) == source["sha256"], "Admission evidence copy hash differs")
    require(m11.digest(evidence / "summary.json") == record["sha256"], "Admission summary hash differs")
    return evidence


def check_admission(run, manifest):
    admission = manifest["admission"]
    evidence = check_evidence_copy(run, manifest, "admission")
    source = evidence / "summary.json"
    record = m11.validate_admission(source)
    require(read_json(run / "admission-summary.json") == record == admission["summary"], "Admission summary differs")
    admission_manifest = read_json(source.parent / "manifest.json")
    require(admission_manifest["cli"] == admission["cli"] == manifest["cli"], "Admission CLI identity differs from comparison")
    limit = m11.admission_limit(admission_manifest)
    require(admission_manifest.get("max_calls") == 1
            and admission_manifest.get("science_units") == 0, "Admission allocation differs")
    command = read_json(source.parent / "probe/command.json")
    result, tokens = check_saved_call(source.parent / "probe", {
        "model": m11.CONFIG["model"], "reasoning": m11.CONFIG["reasoning"],
        "cli_timeout_seconds": command["timeout_seconds"],
    })
    require(record["tokens"] == tokens <= limit and record["status"] == result["status"], "Admission usage differs")
    require(record.get("actual_model") == result.get("actual_model"), "Admission model evidence differs")
    ledger = read_json(source.parent / "ledger.json")
    require(len(ledger) == 1 and ledger[0].get("status") == "known"
            and ledger[0].get("usage") == result["usage"] and ledger[0].get("result_status") == result["status"],
            "Admission ledger differs from raw call")
    return {"calls": 1, "tokens": tokens, "usage_known": True,
            "summary_sha256": admission["sha256"], "raw_result_sha256": m11.digest(source.parent / "probe/result.json"),
            "preceding_failed_admission": admission_manifest.get("preceding_failed_admission"),
            "scope": "Successful admission probe only; preceding failed admission costs are not counted as zero or included in comparison costs"}


def check_science_admission(run, manifest):
    evidence = check_evidence_copy(run, manifest, "science_admission")
    record = m11.validate_science_admission(evidence / "summary.json", manifest["source_and_input_hashes"], manifest["packages"])
    require(record == manifest["science_admission"]["summary"], "Science admission summary differs")
    return {"summary": record, "summary_sha256": manifest["science_admission"]["sha256"],
            "scope": "Separately accounted scientific admission; frozen saved evidence only"}


def check_provenance(run, manifest):
    hashes, freeze_audit = m11.frozen_hashes()
    require(manifest["source_and_input_hashes"] == hashes, "Frozen raw source/input hashes differ")
    require(manifest["freeze_audit"] == freeze_audit, "Frozen legacy-source audit differs")
    require(manifest["config"] == m11.CONFIG, "Comparison configuration differs from fixed protocol")
    require(manifest["crystal_builder"] == {"commit": m11.CRYSTAL_COMMIT, "status": ""}, "Scientific checkout identity differs")
    require(manifest["root"].get("status") == "", "Comparison started with an uncommitted implementation")
    require(manifest["baseline_policies"] == {arm: baseline_policy(arm) for arm in ["batch", "m8"]},
            "Baseline policies differ")
    data = read_json(run / "feedback.json")
    require(m11.digest(run / "feedback.json") == manifest["feedback_sha256"], "Feedback file hash differs")
    require(data == m11.feedback(), "Feedback differs from versioned aggregate evidence")
    require(data["provenance"] == manifest["historical_feedback_sources"], "Historical feedback provenance differs")
    require(data["historical_raw_runs_available"] is False, "Unavailable historical raw runs cannot be re-audited")
    return data


def audit(run, destination):
    run, destination = Path(run).resolve(), Path(destination).resolve()
    require(not destination.exists(), "Replay output already exists")
    require(not destination.is_relative_to(run) and not run.is_relative_to(destination), "Replay output overlaps source run")
    manifest, rows = read_json(run / "manifest.json"), read_json(run / "rows.json")
    check_comparison(rows, manifest["schedule"])
    data = check_provenance(run, manifest)
    admission = check_admission(run, manifest)
    science_admission = check_science_admission(run, manifest)
    summary = m11.summarize(rows)
    require(summary == read_json(run / "summary.json"), "Saved summary differs")
    candidate = read_json(run / "candidate.json")
    require(policy_hash(candidate["workflow"]) == candidate["sha256"], "Candidate policy hash differs")
    require(candidate["sha256"] != policy_hash(baseline_policy("m8")), "Unchanged candidate cannot form a completed M11 comparison")
    saved_pools = read_json(run / "pools.json")
    require(saved_pools == read_json(ROOT / "benchmarks/m7/pools.json")["pools"], "Saved pools differ from frozen M7")
    schedule = [{"pool": pool["id"], "arm": arm} for i, pool in enumerate(saved_pools)
                for arm in m11.ARMS[i % 3:] + m11.ARMS[:i % 3]]
    require(manifest["schedule"] == schedule, "Arm rotation differs from protocol")
    pools = {pool["id"]: pool for pool in saved_pools}
    records = []
    outer_config = dict(m11.CONFIG, max_llm_calls=1, max_observed_tokens=100000)
    outer_source, outer_target = run / "evolution", destination / "evolution"
    outer = m11.ThreeArmExperiment(outer_config, outer_target,
                                  ReplayBackend(outer_source, outer_target, outer_config, records))
    proposal = outer.model(m11.proposal_prompt(data), PROPOSAL_SCHEMA, outer_target / "proposal")
    require(proposal == {key: candidate[key] for key in ("workflow", "hypothesis")}, "Outer proposal differs from frozen candidate")
    require(outer.observed_tokens <= 100000 and outer.calls == 1, "Outer proposal budget exceeded")
    require(read_json(outer_source / "config.json") == outer_config, "Outer configuration differs")
    require(read_json(outer_source / "ledger.json") == outer.ledger, "Outer call ledger differs")
    counts = {"episodes_replayed": 0, "action_artifacts": 0, "candidate_artifacts": 0,
              "decision_prompts_checked": 0, "outer_prompts_checked": 1}
    episodes = []
    for row in rows:
        source, target = run / row["pool"] / row["arm"], destination / row["pool"] / row["arm"]
        exp = WorkflowScreeningExperiment(manifest["config"], target,
                                          ReplayBackend(source, target, manifest["config"], records))

        def science(task, actions, directory, previous=None):
            original = source / directory.relative_to(target)
            require(read_json(original / "task.json") == task, "Scientific task replay differs")
            require(read_json(original / "actions.json") == actions, "Scientific action replay differs")
            if previous is not None:
                require(read_json(original / "previous.json") == previous, "Checkpoint replay differs")
            else:
                require(not (original / "previous.json").exists(), "Unexpected prior checkpoint")
            result = read_json(original / "result.json")
            require(result["attempted_actions"] == (previous or {}).get("attempted_actions", []) + actions,
                    "Worker action history differs")
            require([event["action"] for event in result["events"]]
                    == [event["action"] for event in (previous or {}).get("events", [])] + actions,
                    "Worker executed unexpected actions")
            return result

        exp.science = science
        expected_policy = candidate["workflow"] if row["arm"] == "candidate" else manifest["baseline_policies"][row["arm"]]
        require(row["workflow"] == expected_policy, "Episode policy differs")
        replay = exp.episode(pools[row["pool"]], row["arm"], expected_policy)
        require(all(row.get(key) == value for key, value in replay.items()), "Decision, control, report or grading replay differs")
        saved = read_json(source / "pool-state.json")
        require(saved == read_json(target / "pool-state.json"), "Scientific state replay differs")
        require(read_json(source / "control.json") == replay["control"], "Saved controller events differ")
        require(read_json(source / "graded.json") == row, "Episode result differs from comparison row")
        require(read_json(source / "config.json") == manifest["config"], "Episode configuration differs")
        require(read_json(source / "ledger.json") == exp.ledger, "Episode call ledger differs")
        require(exp.calls == row["cost"]["calls"] <= 13, "Episode call budget differs")
        require(exp.observed_tokens == row["cost"]["tokens"] <= 325000, "Episode token budget differs")
        require(saved["spent"] == row["cost"]["units"] <= 12 and len(saved["events"]) <= 12, "Episode science budget differs")
        require(sum(event["action"].get("max_steps", 0) for event in saved["events"])
                == row["cost"]["requested_steps"] <= 4, "Episode requested UFF budget differs")
        for state in saved["states"].values():
            endpoint = state["checkpoint_history"][-1]
            require(all(state["artifact"][key] == endpoint[key] for key in ["path", "sha256"]), "Final checkpoint differs")
            require(sum(event["action"].get("max_steps", 0) for event in state["events"]) <= 3, "Candidate UFF budget exceeded")
            for checkpoint in state["checkpoint_history"]:
                artifact = Path(checkpoint["path"]).resolve()
                require(artifact.is_relative_to(run), "Checkpoint lies outside the source run")
                require(m11.digest(artifact) == checkpoint["sha256"], "Checkpoint artifact hash differs")
        candidates = {item["id"]: item for item in pools[row["pool"]]["candidates"]}
        for claim in row["report"]["candidates"]:
            if claim["status"] == "rejected":
                cid = claim["candidate_id"]
                state = saved["states"][cid]
                current = {diagnostic["code"] for diagnostic in state["final"]["diagnostics"]}
                require(candidates[cid]["catalog"]["incomplete_record"]
                        or current & {"COINCIDENT_ATOMS", "DEUTERIUM", "OVERCOORDINATED", "MISSING_HYDROGENS", "CALCULATION_REFUSED"}
                        or "BONDING_WOULD_CHANGE" in state["diagnostic_codes"], "Stale rejection evidence")
        counts["episodes_replayed"] += 1
        counts["action_artifacts"] += len(saved["events"])
        counts["candidate_artifacts"] += len(saved["states"])
        counts["decision_prompts_checked"] += exp.calls
        episodes.append({**row, "actions": saved["events"], "final_qualifications": saved["grades"]})
    total_tokens = check_stage(read_json(run / "stage-ledger.json"), records)
    require(total_tokens == outer.observed_tokens + sum(row["cost"]["tokens"] for row in rows), "Aggregate token accounting differs")
    science_units = sum(row["cost"]["units"] for row in rows)
    requested_steps = sum(row["cost"]["requested_steps"] for row in rows)
    require(science_units <= 216 and requested_steps <= 72, "Stage scientific budget exceeded")
    decision = m11.selection(rows)
    require(read_json(run / "selection.json") == decision, "Saved promotion decision differs")
    accepted = read_json(run / "accepted.json")
    expected = candidate["workflow"] if decision["promoted"] else manifest["baseline_policies"]["m8"]
    require(accepted == {"workflow": expected, "sha256": policy_hash(expected),
                        "source": "candidate" if decision["promoted"] else "m8", "scope": "M7 public development"},
            "Accepted policy differs from frozen gate")
    return {
        "manifest": manifest, "candidate": candidate, "summary": summary, "selection": decision,
        "accepted": accepted, "admission": admission, "science_admission": science_admission,
        "audit": {**counts, "all_checks_passed": True,
                  "scope": "Saved prompt/schema/CLI/action/artifact replay only; no inference or scientific worker calls. Historical M10 aggregate provenance is checked without re-auditing unavailable historical runs."},
        "total_cost": {"calls": len(records), "tokens": total_tokens, "outer_tokens": outer.observed_tokens,
                       "science_units": science_units, "requested_steps": requested_steps,
                       "scope": "M11 outer proposal and comparison only; admission recorded separately"},
        "episodes": episodes,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--replay-output", type=Path, required=True)
    parser.add_argument("--report-output", type=Path, required=True)
    args = parser.parse_args(argv)
    run, replay, output = args.run.resolve(), args.replay_output.resolve(), args.report_output.resolve()
    require(not output.exists(), "Report output already exists; refusing to overwrite")
    require(not output.is_relative_to(run), "Report output must be separate from source run")
    require(output != replay and not replay.is_relative_to(output), "Report output overlaps replay directory")
    report = audit(run, replay)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    print(json.dumps({key: report[key] for key in ["audit", "summary", "selection", "total_cost", "admission", "science_admission"]}, indent=2))


if __name__ == "__main__":
    main()

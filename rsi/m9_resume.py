"""Complete the frozen M9 sixth pool in a separately budgeted recovery segment."""

import argparse
import time
from datetime import datetime, timezone
from pathlib import Path

from rsi.experiment import ROOT, read_json, write_json, git_state, cli_identity
from rsi.m9 import CONFIG, Budget, digest, selection, summarize
from rsi.m9_workflow import WorkflowScreeningExperiment, policy_hash


class RecoveryBudget(Budget):
    def reserve(self, label):
        if len(self.entries) >= 39 or sum(e.get("tokens", 0) for e in self.entries) >= 975000:
            raise RuntimeError("Recovery budget exhausted")
        return super().reserve(label)


def run(output):
    source = ROOT / "runs/m9-workflow-01"
    manifest = read_json(source / "manifest.json")
    original_rows = read_json(source / "rows.json")
    prior_rows = [r for r in original_rows if not r.get("error")]
    if len(prior_rows) != 15 or len(original_rows) != 16:
        raise ValueError("Unexpected interrupted run")
    failed = original_rows[-1]
    if (failed["pool"], failed["arm"], failed["cost"]["units"], failed["cost"]["unknown_usage"]) != ("m7-dev-06", "candidate", 0, 1):
        raise ValueError("Recovery requires a pre-science failure")
    hashes = dict(manifest["source_and_input_hashes"])
    for p in source.rglob("*"):
        if p.is_file():
            hashes[p.relative_to(ROOT).as_posix()] = digest(p)
    for name in ["rsi/m9_resume.py", "docs/M9_RECOVERY_PROTOCOL.md", "scripts/report_m9_recovery.py"]:
        hashes[name] = digest(ROOT / name)
    if any(digest(ROOT / p) != h for p, h in hashes.items()):
        raise ValueError("Frozen input changed")
    pools = read_json(source / "pools.json")
    pool = next(p for p in pools if p["id"] == "m7-dev-06")
    schedule = [(pool, a) for a in ["candidate", "batch", "m8"]]
    proposal = read_json(source / "candidate.json")
    sha = policy_hash(proposal["workflow"])
    assert sha == proposal["sha256"]
    output.mkdir(parents=True, exist_ok=False)
    budget = RecoveryBudget(output / "stage-ledger.json")
    recovery = {"root": git_state(ROOT), "crystal_builder": git_state(ROOT / "Crystal-Builder"),
                "started_utc": datetime.now(timezone.utc).isoformat(), "original_manifest": manifest,
                "source_and_input_hashes": hashes, "prior_failed_call_usage": "unknown; retained, not zero",
                "schedule": [{"pool": p["id"], "arm": a} for p, a in schedule]}
    write_json(output / "manifest.json", recovery)
    write_json(output / "candidate.json", proposal)
    rows = []
    for pool, arm in schedule:
        if any(digest(ROOT / p) != h for p, h in hashes.items()) or policy_hash(read_json(output / "candidate.json")["workflow"]) != sha:
            raise RuntimeError("Frozen experiment content changed")
        exp = WorkflowScreeningExperiment(CONFIG, output / pool["id"] / arm, stage=budget)
        if cli_identity(exp.backend.executable) != manifest["cli"]:
            raise RuntimeError("CLI changed")
        policy = proposal["workflow"] if arm == "candidate" else manifest["baseline_policies"][arm]
        started = time.monotonic()
        try:
            row = exp.episode(pool, arm, policy)
        except Exception as exc:
            row = {"pool": pool["id"], "arm": arm, "error": f"{type(exc).__name__}: {exc}"}
        path = exp.output / "pool-state.json"
        state = read_json(path) if path.exists() else {"spent": 0, "events": []}
        row["cost"] = {"calls": exp.calls, "tokens": exp.observed_tokens, "units": state["spent"],
                       "requested_steps": sum(e["action"].get("max_steps", 0) for e in state["events"]),
                       "unknown_usage": sum(not all(k in e.get("usage", {}) for k in ["input_tokens", "output_tokens"]) for e in exp.ledger),
                       "wall_seconds": time.monotonic() - started}
        rows.append(row)
        write_json(exp.output / "graded.json", row)
        write_json(output / "rows.json", rows)
        write_json(output / "summary.json", summarize(prior_rows + rows))
        print(pool["id"], arm, row.get("score", row.get("error")), row["cost"]["tokens"], flush=True)
        if row["cost"]["unknown_usage"] or any(e["status"] != "known" for e in budget.entries):
            break
    decision = selection(prior_rows + rows)
    write_json(output / "selection.json", decision)
    selected = proposal["workflow"] if decision["promoted"] else manifest["baseline_policies"]["m8"]
    write_json(output / "accepted.json", {"workflow": selected, "sha256": policy_hash(selected),
               "source": "candidate" if decision["promoted"] else "m8", "scope": "M7 public development"})
    print(decision, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.output.resolve())

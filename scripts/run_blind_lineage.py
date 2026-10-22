"""Run successive Blind-G1 v2 campaigns for Claude; each confirmed champion becomes the next parent.

A generation is one full v2 campaign (three RSI and three independent slots,
then nine confirmation fits). rsi.blind_lineage selects the configuration with
the lowest fresh-seed confirmation MAE; the lineage continues only if it beats
that generation's parent by at least 1%, and never beyond --max-generations.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

import run_blind_claude_cli_v2 as runner

ROOT = Path(__file__).resolve().parents[1]
PROBE = "runs/blind-g1-runtime-probe-05/report.json"
MODEL_RECORD = "claude-opus-5-5 (effort high; Claude Code CLI 2.1.305; isolated, no tools)"
LEDGER = ROOT / "runs/blind-g1-lineage.json"


def lineage(*args):
    command = ["wsl", "-d", "Ubuntu", "--", "python3", "-m", "rsi.blind_lineage", *args]
    done = subprocess.run(command, cwd=ROOT, capture_output=True, timeout=3600)
    if done.returncode != 0:
        raise RuntimeError(f"Lineage {args[0]} failed: {done.stderr.decode('utf-8', 'replace')[-2000:]}")
    return json.loads(done.stdout.decode("utf-8"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--first", default="runs/blind-g1-campaign-02")
    parser.add_argument("--max-generations", type=int, default=10)
    parser.add_argument("--wait-for", help="Start only after this file exists (the previous experiment's final report)")
    args = parser.parse_args()
    if args.wait_for:
        while not (ROOT / args.wait_for).exists():
            time.sleep(60)
        print(f"[lineage] {args.wait_for} found; starting", flush=True)
    history = json.loads(LEDGER.read_text(encoding="utf-8")) if LEDGER.exists() else []
    campaign = args.first
    first_index = int(Path(args.first).name.rsplit("-", 1)[1])
    for generation in range(1, args.max_generations + 1):
        print(f"[lineage] generation {generation}: {campaign}", flush=True)
        runner.use_campaign(campaign)
        summary = runner.run_campaign()
        if not summary or summary.get("status") != "completed":
            print(f"[lineage] stopping: confirmation of {campaign} did not complete", flush=True)
            break
        following = f"runs/blind-g1-campaign-{first_index + generation:02d}"
        if generation == args.max_generations:
            decision = {**lineage("champion", "--previous", campaign), "continued": False,
                        "stop_reason": "maximum generations reached"}
        else:
            decision = lineage("successor", "--previous", campaign, "--output", following, "--probe", PROBE,
                               "--model", MODEL_RECORD)
            if not decision["continued"]:
                decision["stop_reason"] = "confirmed champion did not beat its parent by 1%"
        history.append({"generation": generation, "campaign": campaign, **decision})
        LEDGER.write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")
        print(f"[lineage] {campaign} champion={decision['configuration']} "
              f"confirmation={decision['confirmation_aggregate']} improvement={decision['relative_improvement']:.3%} "
              f"continued={decision['continued']}", flush=True)
        if not decision["continued"]:
            print(f"[lineage] stopping: {decision['stop_reason']}", flush=True)
            break
        campaign = following


if __name__ == "__main__":
    main()

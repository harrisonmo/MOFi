"""Operator-side driver for the second Blind-G1 campaign: isolated Claude CLI calls with one format repair.

Same isolation as scripts/run_blind_claude_cli.py (empty temporary directory,
no tools, MCP servers, claude.ai connectors, session persistence or inherited
model overrides). After the first reply the controller's free format check
runs; if it fails, a second isolated call receives the original prompt, the
previous reply and only the check's error list, and its reply is submitted.
Candidate code is never imported or run here; the controller runs it in Docker.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

from run_blind_claude_cli import CLI_ARGS, EFFORT, MODEL, STRIPPED_ENV, SYSTEM_PROMPT, clean_env, save, unwrap

ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN = "runs/blind-g1-campaign-02"
OUT = ROOT / "runs/blind-claude-cli-v2/blind-g1-campaign-02"
START_HERE = ROOT / "templates/blind_g1_v2/START_HERE.md"
# Leave margin inside the 2700 s proposal window for the format check and submission.
PROPOSAL_MARGIN_SECONDS = 180
# A usage-limited or unreachable model must never cost a slot: wait before exporting instead.
PREFLIGHT_RETRY_SECONDS = 1800
PREFLIGHT_MAX_WAIT_SECONDS = 12 * 3600


def use_campaign(campaign):
    global CAMPAIGN, OUT
    CAMPAIGN = campaign
    OUT = ROOT / "runs/blind-claude-cli-v2" / Path(campaign).name


def preflight():
    """Wait until a tiny synthetic call succeeds; nothing is exported while the model is unavailable."""
    waited = 0
    while True:
        workdir = Path(tempfile.mkdtemp(prefix="blind-preflight-"))
        try:
            done = subprocess.run([shutil.which("claude"), *CLI_ARGS], input=b'Reply with only: {"ok": true}',
                                  cwd=workdir, env=clean_env(), capture_output=True, timeout=300)
            result = json.loads(done.stdout.decode("utf-8"))
            if not result.get("is_error") and MODEL in (result.get("modelUsage") or {}):
                return waited
            reason = result.get("result")
        except (subprocess.TimeoutExpired, ValueError, OSError) as error:
            reason = str(error)[:200]
        finally:
            shutil.rmtree(workdir, ignore_errors=True)
        if waited >= PREFLIGHT_MAX_WAIT_SECONDS:
            raise RuntimeError(f"Model unavailable for {waited} s; stopping without exporting: {reason}")
        print(f"[preflight] model unavailable ({reason}); retrying in {PREFLIGHT_RETRY_SECONDS} s", flush=True)
        time.sleep(PREFLIGHT_RETRY_SECONDS)
        waited += PREFLIGHT_RETRY_SECONDS


def controller(*args):
    command = ["wsl", "-d", "Ubuntu", "--", "python3", "-m", "rsi.blind_campaign_v2", args[0],
               "--campaign", CAMPAIGN, *args[1:]]
    done = subprocess.run(command, cwd=ROOT, capture_output=True, timeout=14400)
    if done.returncode != 0:
        raise RuntimeError(f"Controller {args[0]} failed: {done.stderr.decode('utf-8', 'replace')[-2000:]}")
    return json.loads(done.stdout.decode("utf-8"))


def format_check(arm, response):
    """Run the controller's free check; a controller refusal (for example a missed deadline) leaves submit to record it."""
    try:
        return controller("check", "--provider", "claude", "--arm", arm, "--submission", response.relative_to(ROOT).as_posix())
    except RuntimeError as error:
        return {"valid": False, "errors": [str(error)[-300:]], "warnings": [], "checks_remaining": 0}


def call(prompt, files, timeout, out, label):
    """One isolated CLI call; returns the reply text and a call record."""
    workdir = Path(tempfile.mkdtemp(prefix="blind-claude-"))
    for name, path in files.items():
        shutil.copy(path, workdir / name)
    before = sorted(p.name for p in workdir.iterdir())
    (out / f"{label}-prompt.txt").write_text(prompt, encoding="utf-8")
    record = {"prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(), "timeout_seconds": timeout}
    started = time.time()
    text = ""
    try:
        done = subprocess.run([shutil.which("claude"), *CLI_ARGS], input=prompt.encode("utf-8"), cwd=workdir,
                              env=clean_env(), capture_output=True, timeout=max(1, timeout))
        (out / f"{label}-cli-stdout.json").write_bytes(done.stdout)
        (out / f"{label}-cli-stderr.txt").write_bytes(done.stderr)
        result = json.loads(done.stdout.decode("utf-8"))
        text, unwrapped = unwrap(result.get("result") or "")
        record.update(cli_exit=done.returncode, is_error=result.get("is_error"), terminal_reason=result.get("terminal_reason"),
                      num_turns=result.get("num_turns"), usage=result.get("usage"), model_usage=result.get("modelUsage"),
                      cost_usd=result.get("total_cost_usd"), permission_denials=result.get("permission_denials"),
                      fence_unwrapped=unwrapped)
    except subprocess.TimeoutExpired:
        record["timed_out"] = True
    except (ValueError, OSError) as error:
        record["cli_error"] = str(error)[:500]
    record["cli_seconds"] = time.time() - started
    record["workdir_unchanged"] = sorted(p.name for p in workdir.iterdir()) == before
    shutil.rmtree(workdir, ignore_errors=True)
    return text, record


def run_slot(arm, proposal_seconds):
    state = controller("status")["providers"]["claude"]["arms"][arm]
    generation = len(state["slots"]) + 1
    out = OUT / f"{arm}-{generation}"
    waited = preflight()
    out.mkdir(parents=True, exist_ok=False)
    exported_at = time.time()
    controller("export", "--provider", "claude", "--arm", arm)
    deadline = exported_at + proposal_seconds - PROPOSAL_MARGIN_SECONDS
    packet_path = ROOT / CAMPAIGN / "outbox/claude" / f"{arm}-{generation}.json"
    packet = packet_path.read_text(encoding="utf-8")
    prompt = "START_HERE.md:\n\n" + START_HERE.read_text(encoding="utf-8") + "\n\npacket.json:\n\n" + packet + "\n"
    files = {"packet.json": packet_path, "START_HERE.md": START_HERE}
    record = {"arm": arm, "generation": generation, "model": MODEL, "effort": EFFORT, "cli_args": CLI_ARGS,
              "system_prompt": SYSTEM_PROMPT, "packet_sha256": hashlib.sha256(packet_path.read_bytes()).hexdigest(),
              "stripped_env": sorted(k for k in __import__("os").environ if STRIPPED_ENV.match(k)),
              "exported_at": exported_at, "preflight_waited_seconds": waited, "calls": []}
    print(f"[{arm}-{generation}] exported; calling {MODEL} (effort {EFFORT})", flush=True)
    text, first = call(prompt, files, deadline - time.time(), out, "first")
    record["calls"].append(first)
    response = out / "response.json"
    response.write_text(text, encoding="utf-8")
    check = format_check(arm, response)
    record["checks"] = [check]
    print(f"[{arm}-{generation}] reply after {first['cli_seconds']:.0f}s; format check valid={check['valid']}", flush=True)
    if not check["valid"] and check["checks_remaining"] > 0 and deadline - time.time() > 60:
        repair = (prompt + "\n\nYour previous reply was:\n\n" + text + "\n\nThe controller's format check failed with these "
                  "errors (no code was run and no score is revealed):\n\n" + "\n".join(f"- {e}" for e in check["errors"]) +
                  "\n\nReturn the complete corrected JSON object only.\n")
        text, second = call(repair, files, deadline - time.time(), out, "repair")
        record["calls"].append(second)
        response.write_text(text, encoding="utf-8")
        record["checks"].append(format_check(arm, response))
        print(f"[{arm}-{generation}] repair after {second['cli_seconds']:.0f}s; format check valid={record['checks'][-1]['valid']}", flush=True)
    record["response_sha256"] = hashlib.sha256(response.read_bytes()).hexdigest()
    save(out / "call.json", record)
    feedback = controller("submit", "--provider", "claude", "--arm", arm, "--submission", response.relative_to(ROOT).as_posix())
    record.update(feedback=feedback, submitted_seconds_after_export=time.time() - exported_at)
    save(out / "call.json", record)
    metrics = feedback.get("metrics", {})
    print(f"[{arm}-{generation}] {feedback['status']} mean_mae={metrics.get('mean_mae')} "
          f"accepted={feedback.get('accepted')} reason={feedback.get('reason')}", flush=True)
    return feedback


def run_campaign(arms=("rsi", "independent"), confirm=True):
    """Fill every remaining slot of the current campaign, then run its confirmation once."""
    policy = json.loads((ROOT / CAMPAIGN / "policy.json").read_text(encoding="utf-8"))
    for arm in arms:
        while len(controller("status")["providers"]["claude"]["arms"][arm]["slots"]) < policy["rounds"]:
            run_slot(arm, policy["proposal_wall_seconds"])
    if confirm and "confirmation" not in controller("status")["providers"]["claude"]:
        print(f"[confirm] {Path(CAMPAIGN).name}: running nine confirmation fits", flush=True)
        summary = controller("confirm", "--provider", "claude")
        save(OUT / "confirmation.json", summary)
        print(f"[confirm] {summary['status']} aggregate={summary.get('aggregate')}", flush=True)
        return summary
    return controller("status")["providers"]["claude"].get("confirmation")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", default=CAMPAIGN)
    # Counterbalanced as in the runbook: Claude runs RSI first, ChatGPT independent first.
    parser.add_argument("--arms", default="rsi,independent")
    parser.add_argument("--confirm", action="store_true")
    args = parser.parse_args()
    use_campaign(args.campaign)
    run_campaign(tuple(args.arms.split(",")), args.confirm)


if __name__ == "__main__":
    main()

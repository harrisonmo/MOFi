"""Operator-side Blind-G1 driver: one isolated Claude Code CLI call per Claude slot.

Each slot exports a packet through the frozen WSL controller, gives the model
only START_HERE.md and that packet in an empty temporary directory, with no
tools, MCP servers, claude.ai connectors, session persistence or inherited
model overrides, then submits the raw reply through the controller. Candidate
code is never imported or run here; the controller runs it in Docker.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN = "runs/blind-g1-campaign-01"
OUT = ROOT / "runs/blind-claude-cli"
START_HERE = ROOT / "templates/blind_g1/START_HERE.md"
MODEL = "claude-opus-5-5"
EFFORT = "high"
# The controller rejects submissions more than 1800 s after export; leave margin for submission.
CLI_TIMEOUT_SECONDS = 1680
SYSTEM_PROMPT = ("You are the candidate author in a blind machine-learning experiment. You have no tools and "
                 "no access to files beyond the text in this message. Follow START_HERE.md and the packet exactly, "
                 "and reply with a single JSON object matching the packet's response_contract.")
CLI_ARGS = ["-p", "--model", MODEL, "--effort", EFFORT, "--tools", "", "--strict-mcp-config",
            "--disallowedTools", "mcp__claude_ai_Claude_Docs", "--no-session-persistence",
            "--system-prompt", SYSTEM_PROMPT, "--output-format", "json"]
# Inherited desktop-session and third-party model variables that must not reach the candidate process.
STRIPPED_ENV = re.compile(r"^(ANTHROPIC_MODEL|ANTHROPIC_DEFAULT_.*|ANTHROPIC_SMALL_FAST_MODEL|DEEPSEEK_.*|OLLAMA_.*|"
                          r"CLAUDECODE|CLAUDE_CODE_.*|CLAUDE_AGENT_SDK_.*|CLAUDE_PID|CLAUDE_PREVIEW_.*)$", re.I)


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def controller(*args):
    command = ["wsl", "-d", "Ubuntu", "--", "python3", "-m", "rsi.blind_campaign", args[0],
               "--campaign", CAMPAIGN, *args[1:]]
    done = subprocess.run(command, cwd=ROOT, capture_output=True, timeout=14400)
    if done.returncode != 0:
        raise RuntimeError(f"Controller {args[0]} failed: {done.stderr.decode('utf-8', 'replace')[-2000:]}")
    return json.loads(done.stdout.decode("utf-8"))


def clean_env():
    env = {k: v for k, v in os.environ.items() if not STRIPPED_ENV.match(k)}
    env["ENABLE_CLAUDEAI_MCP_SERVERS"] = "false"
    return env


def unwrap(text):
    """Remove only a surrounding Markdown fence; never edit the JSON itself."""
    stripped = text.strip()
    match = re.fullmatch(r"```[A-Za-z]*\n(.*)\n```", stripped, re.S)
    return (match.group(1), True) if match else (stripped, False)


def run_slot(arm):
    state = controller("status")["providers"]["claude"]["arms"][arm]
    generation = len(state["slots"]) + 1
    out = OUT / f"{arm}-{generation}"
    out.mkdir(parents=True, exist_ok=False)
    exported_at = time.time()
    controller("export", "--provider", "claude", "--arm", arm)
    packet_path = ROOT / CAMPAIGN / "outbox/claude" / f"{arm}-{generation}.json"
    packet = packet_path.read_text(encoding="utf-8")
    prompt = ("START_HERE.md:\n\n" + START_HERE.read_text(encoding="utf-8") +
              "\n\npacket.json:\n\n" + packet + "\n")
    (out / "prompt.txt").write_text(prompt, encoding="utf-8")
    workdir = Path(tempfile.mkdtemp(prefix="blind-claude-"))
    shutil.copy(packet_path, workdir / "packet.json")
    shutil.copy(START_HERE, workdir / "START_HERE.md")
    before = sorted(p.name for p in workdir.iterdir())
    record = {"arm": arm, "generation": generation, "model": MODEL, "effort": EFFORT, "cli_args": CLI_ARGS,
              "system_prompt": SYSTEM_PROMPT, "packet_sha256": hashlib.sha256(packet_path.read_bytes()).hexdigest(),
              "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
              "stripped_env": sorted(k for k in os.environ if STRIPPED_ENV.match(k)), "exported_at": exported_at}
    print(f"[{arm}-{generation}] exported; calling {MODEL} (effort {EFFORT})", flush=True)
    started = time.time()
    try:
        done = subprocess.run([shutil.which("claude"), *CLI_ARGS], input=prompt.encode("utf-8"), cwd=workdir,
                              env=clean_env(), capture_output=True, timeout=CLI_TIMEOUT_SECONDS)
        (out / "cli-stdout.json").write_bytes(done.stdout)
        (out / "cli-stderr.txt").write_bytes(done.stderr)
        result = json.loads(done.stdout.decode("utf-8"))
        text, unwrapped = unwrap(result.get("result") or "")
        record.update(cli_exit=done.returncode, is_error=result.get("is_error"), terminal_reason=result.get("terminal_reason"),
                      num_turns=result.get("num_turns"), usage=result.get("usage"), model_usage=result.get("modelUsage"),
                      cost_usd=result.get("total_cost_usd"), permission_denials=result.get("permission_denials"),
                      fence_unwrapped=unwrapped)
    except subprocess.TimeoutExpired:
        text, record["timed_out"] = "", True
    except (ValueError, OSError) as error:
        text, record["cli_error"] = "", str(error)[:500]
    record["cli_seconds"] = time.time() - started
    record["workdir_unchanged"] = sorted(p.name for p in workdir.iterdir()) == before
    shutil.rmtree(workdir, ignore_errors=True)
    response = out / "response.json"
    response.write_text(text, encoding="utf-8")
    record["response_sha256"] = hashlib.sha256(response.read_bytes()).hexdigest()
    save(out / "call.json", record)
    print(f"[{arm}-{generation}] reply after {record['cli_seconds']:.0f}s; submitting", flush=True)
    relative = response.relative_to(ROOT).as_posix()
    feedback = controller("submit", "--provider", "claude", "--arm", arm, "--submission", relative)
    record.update(feedback=feedback, submitted_seconds_after_export=time.time() - exported_at)
    save(out / "call.json", record)
    metrics = feedback.get("metrics", {})
    print(f"[{arm}-{generation}] {feedback['status']} mean_mae={metrics.get('mean_mae')} "
          f"accepted={feedback.get('accepted')} reason={feedback.get('reason')}", flush=True)
    return feedback


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arms", default="rsi,independent")
    parser.add_argument("--confirm", action="store_true")
    args = parser.parse_args()
    for arm in args.arms.split(","):
        while len(controller("status")["providers"]["claude"]["arms"][arm]["slots"]) < 3:
            run_slot(arm)
    if args.confirm:
        print("[confirm] running nine confirmation fits", flush=True)
        summary = controller("confirm", "--provider", "claude")
        save(OUT / "confirmation.json", summary)
        print(f"[confirm] {summary['status']} aggregate={summary.get('aggregate')}", flush=True)


if __name__ == "__main__":
    main()

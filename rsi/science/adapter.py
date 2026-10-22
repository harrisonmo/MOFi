"""Replay allowlisted actions against an isolated scientific session."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CRYSTAL_ROOT = ROOT / "Crystal-Builder"
PREPARE_STEPS = {"duplicates", "deuterium", "primitive", "disorder", "solvent", "hydrogens"}


def input_path(initialization: dict) -> Path:
    """Resolve public fixtures without granting arbitrary filesystem access."""
    roots = {None: CRYSTAL_ROOT, "m6_fixture": ROOT / "benchmarks/m6/fixtures",
             "m7_fixture": ROOT / "benchmarks/m7/fixtures"}
    if initialization.get("root") not in roots:
        raise ValueError("Unknown input root")
    base = roots[initialization.get("root")]
    source = (base / initialization["path"]).resolve()
    if not source.is_relative_to(base.resolve()):
        raise ValueError("Input must be inside its allowed root")
    return source


def validate_action(action: dict) -> None:
    """Reject unspecified options before any scientific operation."""
    if not isinstance(action, dict):
        raise ValueError("An action must be an object")
    verb = action.get("verb")
    allowed = {"inspect": {"verb"}, "stop": {"verb"},
               "prepare": {"verb", "steps"}, "optimize": {"verb", "max_steps"}}
    if verb not in allowed or set(action) - allowed[verb]:
        raise ValueError("Disallowed action or option")
    if verb == "prepare":
        steps = action.get("steps")
        if not isinstance(steps, list) or not steps or any(s not in PREPARE_STEPS for s in steps):
            raise ValueError("Preparation requires explicit permitted steps; cap is not authorized")
    if verb == "optimize" and (type(action.get("max_steps")) is not int or not 1 <= action["max_steps"] <= 3):
        raise ValueError("Optimization requires an integer max_steps between 1 and 3")


def run_task(task: dict, actions: list[dict], output_dir: Path, previous: dict | None = None) -> dict:
    """Execute new actions once, optionally continuing a verified saved checkpoint."""
    from xtal.agent import Session

    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    result = {"task_id": task["id"], "events": [], "errors": [], "stopped": False,
              "artifact": None, "initial": None, "final": None, "diagnostic_codes": [],
              "attempted_actions": actions, "allowed_prepare_steps": sorted(PREPARE_STEPS)}
    session = None
    prior_actions = []
    try:
        if previous is not None:
            if previous["task_id"] != task["id"] or previous["errors"] or previous["stopped"]:
                raise ValueError("Cannot resume a mismatched, failed, or stopped episode")
            artifact = previous["artifact"]
            checkpoint = Path(artifact["path"])
            if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != artifact["sha256"]:
                raise ValueError("Checkpoint hash mismatch")
            prior_actions = previous["attempted_actions"]
            result["events"] = copy.deepcopy(previous["events"])
            result["initial"] = previous["initial"]
            result["resumed_from"] = dict(artifact)
        result["attempted_actions"] = prior_actions + actions
        if not isinstance(actions, list) or len(prior_actions) + len(actions) > 8:
            raise ValueError("An episode permits at most eight actions")
        for action in actions:
            validate_action(action)
        if sum(a.get("max_steps", 0) for a in prior_actions + actions if a["verb"] == "optimize") > 3:
            raise ValueError("Total optimization budget is three steps per episode")
        init = task["initialization"]
        if previous is not None:
            session = Session.open(checkpoint)
        elif init["kind"] == "open":
            source = input_path(init)
            session = Session.open(source)
        elif init["kind"] == "build" and init["action"] == "mof.build":
            session = Session.build("mof.build", output_dir / "workspace", **init["params"])
        else:
            raise ValueError("Unsupported initialization")
        if previous is None:
            result["initial"] = session.inspect().to_dict()
        if hasattr(session, "built"):
            result["events"].append({"action": {"verb": "build"}, "result": session.built.to_dict()})
        for action in actions:
            if result["stopped"]:
                raise ValueError("Actions after stop are disallowed")
            verb = action["verb"]
            if verb == "stop":
                result["stopped"] = True
                answer = {"ok": True}
            elif verb == "inspect":
                answer = session.inspect().to_dict()
            elif verb == "prepare":
                answer = session.prepare(steps=action["steps"]).to_dict()
            elif (task.get("fault_fixture") == "first_optimize_unavailable"
                  and not any(e["action"]["verb"] == "optimize" for e in result["events"])):
                answer = {"ok": False, "retryable": True, "injected": True,
                          "diagnostics": [{"code": "SIMULATED_ENGINE_UNAVAILABLE",
                                           "message": "Engineering fault fixture: no computation or mutation occurred; retry is permitted."}]}
            else:
                answer = session.optimize(engine="uff", max_steps=action["max_steps"]).to_dict()
            result["events"].append({"action": action, "result": answer})
    except Exception as exc:
        result["errors"].append({"type": type(exc).__name__, "message": str(exc)})
    if session is not None:
        try:
            result["final"] = session.inspect().to_dict()
            path = session.save(output_dir / "structure.xtalproj")
            result["artifact"] = {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            result["checkpoint_history"] = copy.deepcopy(previous.get("checkpoint_history", [])) if previous else []
            result["checkpoint_history"].append({**result["artifact"], "action_count": len(result["attempted_actions"])})
            (output_dir / "session-log.json").write_text(json.dumps(session.log, indent=2), encoding="utf-8")
        except Exception as exc:
            result["errors"].append({"type": type(exc).__name__, "message": str(exc)})
    observations = [result["initial"] or {}, result["final"] or {}]
    observations.extend(event["result"] for event in result["events"])
    result["diagnostic_codes"] = sorted({d["code"] for obs in observations for d in obs.get("diagnostics", [])})
    (output_dir / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result

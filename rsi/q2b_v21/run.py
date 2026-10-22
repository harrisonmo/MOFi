"""Approved Astra outer-generation and four-arm paired Q2B pilot."""
import argparse
from collections import Counter
import hashlib
import importlib.metadata
import json
from pathlib import Path
import random
import sys
import time

from .transport import CodexCLI
from rsi.experiment import cli_identity, git_state, write_json
from .budget import RunBudget, BudgetStop
from .limits import Ledger, ROOT
from .protocol import AUTHOR, AUTHOR_SCHEMA, CATALOG, REPORT_SCHEMA, research_prompt, plan_schema
from .runtime import digest, manual_program, validate_program
from .scoring import summarize

ARMS = ("fixed", "outer_generated", "manual_control", "removed")
LIMITS = dict(total_model_calls=197, probe_calls=2, generation_and_debug_calls=3, research_calls=192,
              observed_token_soft_limit=6_304_000, tokens_per_call_soft_limit=32_000)


def fingerprints():
    paths = list((ROOT / "rsi/q2b_v21").glob("*.py")) + [ROOT / p for p in (
        "rsi/backends/codex_cli.py", "rsi/q2_budget.py", "rsi/experiment.py", "configs/q2b_v21.json",
        "tools/q2b/manual-v2.json")]
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}


class Pilot:
    def __init__(self, output, executable=None):
        self.output = Path(output).resolve()
        self.read = lambda name: json.loads((self.output / name).read_text())
        if (self.output / "model_budget.json").exists():
            raise RuntimeError("Model run already started; never overwrite or silently resume")
        self.gate = self.read("gate.json")
        self.pairs = self.read("pairs.json")
        if not self.gate["passed"] or digest(self.pairs) != self.gate["dataset_hash"]:
            raise RuntimeError("P0 gate or dataset hash failed")
        self.manual = json.loads((ROOT / "tools/q2b/manual-v2.json").read_text())
        if digest(self.manual) != self.gate["manual_hash"] or self.manual != manual_program():
            raise RuntimeError("Manual control differs from calibrated program")
        self.frozen = fingerprints()
        self.backend = CodexCLI("gpt-6-astra", "high", 240, executable)
        self.manifest = dict(root_git=git_state(ROOT), crystal_builder_git=git_state(ROOT / "Crystal-Builder"),
                             files=self.frozen, gate=self.gate, limits=LIMITS, cli=cli_identity(self.backend.executable),
                             python=sys.version, packages={p: importlib.metadata.version(p) for p in ("numpy", "scipy", "ase", "gemmi", "jsonschema")},
                             requested_model="gpt-6-astra", actual_model=None, reasoning="high",
                             experiment_version="q2b-v2.1", visibility="public_engineering_validation", workflow="q2b-one-plan-one-batch-one-report-v2",
                             source_hashes={str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                                            for p in (ROOT / "Crystal-Builder/resources/samples").rglob("*.cif")
                                            if str(p.relative_to(ROOT / "Crystal-Builder/resources/samples")) in
                                            {"prepared\\MOF-5.cif", "Ni2Cl2BTDD.cif", "prepared\\HKUST-1.cif", "prepared\\ZIF-8.cif"}})
        if self.manifest["root_git"]["status"] or self.manifest["crystal_builder_git"]["status"]:
            raise RuntimeError("Start only from clean independent repositories")
        self.budget = RunBudget(self.output / "model_budget.json", LIMITS)
        self.science = Ledger(self.output / "science_budget.json")
        self.rows, self.authors = [], []
        write_json(self.output / "manifest.json", self.manifest)

    def unchanged(self):
        if fingerprints() != self.frozen or digest(self.read("pairs.json")) != self.gate["dataset_hash"]:
            raise BudgetStop("Frozen code or inputs changed")
        for path, expected in self.manifest["source_hashes"].items():
            if hashlib.sha256(Path(path).read_bytes()).hexdigest() != expected:
                raise BudgetStop("Frozen source changed")

    def model(self, stage, label, prompt, schema):
        self.unchanged()
        row = self.budget.reserve_model(stage, label)
        started = time.perf_counter()
        try:
            result = self.backend.run(prompt, schema, self.output / "model" / label)
        except Exception as exc:
            result = dict(status="controller_error", usage=None, response=None,
                          duration_seconds=time.perf_counter() - started, error=str(exc))
        tokens = self.budget.finish_model(row, result)
        if result["status"] in {"unexpected_tool_use", "auth_error", "model_unavailable"}:
            self.budget.blocked = result["status"]
            self.budget.save()
        print(json.dumps(dict(label=label, status=result["status"], tokens=tokens, blocked=self.budget.blocked)), flush=True)
        return result

    def author(self):
        development = [p for p in self.pairs if p["block"] == "development"]
        visible = [dict(id=p["id"], input_metadata={k: dict(atoms=len(p[k]["frac"]),
                         composition=dict(Counter(p[k]["elements"])), lattice=p[k]["lattice"],
                         sample_frac=p[k]["frac"][:3], sample_edges=p[k]["edges"][:3]) for k in ("A", "B")},
                        development_verdict=p["expected"]) for p in development]
        feedback = None
        for attempt in range(1, 4):
            result = self.model("author", f"author-{attempt}", AUTHOR + CATALOG + json.dumps(dict(development=visible, feedback=feedback)), AUTHOR_SCHEMA)
            record = dict(attempt=attempt, result_status=result["status"], validation=[], accepted=False)
            if self.budget.blocked:
                self.authors.append(record)
                write_json(self.output / "authors.json", self.authors)
                raise BudgetStop(self.budget.blocked)
            try:
                if result["status"] != "ok":
                    raise ValueError("Author response invalid")
                program = result["response"]["program"]
                validate_program(program, top=False)
                record.update(program=program, hypothesis=result["response"]["hypothesis"], sha256=digest(program))
                for pair in development:
                    checked = self.science.run("author_validation", f"author-{attempt}-{pair['id']}",
                        dict(action="validate_tool", program=program, pair=pair), seconds=20)
                    record["validation"].append(dict(id=pair["id"], result=checked))
                record["accepted"] = True
                record["identical_to_manual"] = program == self.manual
                self.authors.append(record)
                write_json(self.output / "authors.json", self.authors)
                write_json(self.output / "candidate.json", dict(id="periodic_pair_evidence", version=1,
                           origin="outer_generated", program=program, sha256=digest(program),
                           parent_sha256=self.authors[-2].get("sha256") if len(self.authors) > 1 else None))
                return program
            except Exception as exc:
                record["error"] = str(exc)
                feedback = dict(error=str(exc), validation=record["validation"])
                self.authors.append(record)
                write_json(self.output / "authors.json", self.authors)
        raise BudgetStop("No valid candidate in approved author budget")

    def episode(self, pair, arm, candidate):
        label = f"{pair['id']}-{arm}"
        registered = candidate if arm == "outer_generated" else self.manual if arm == "manual_control" else None
        prompt = research_prompt(pair, registered)
        row = dict(id=pair["id"], source=pair["source"], case=pair["case"], block=pair["block"], arm=arm,
                   completed=False, passed=False, registered_uses=0, operations=0, certificate_valid=False)
        self.rows.append(row)
        try:
            start_index = len(self.budget.entries)
            first = self.model("research", label + "-plan", prompt + "\nReturn the plan program now.", plan_schema(registered is not None))
            if self.budget.blocked:
                raise BudgetStop(self.budget.blocked)
            output = None
            observation = {"error": "No valid plan"}
            runtime_seconds = 0.
            if first["status"] == "ok":
                program = first["response"]["program"]
                row["registered_uses"] = sum(n["op"] == "periodic_pair_evidence" for n in program["nodes"])
                before = len(self.science.data["entries"])
                try:
                    output = self.science.run("research_tools", label, dict(action="execute", program=program,
                        A=pair["A"], B=pair["B"], tool=registered), seconds=20)
                    write_json(self.output / "evidence" / (label + ".json"), output)
                    observation = summarize(output)
                    row["operations"] = observation["operation_count"]
                except Exception as exc:
                    observation = {"error": str(exc)}
                    row["tool_error"] = str(exc)
                runtime_seconds = sum(r.get("wall_seconds", 0) for r in self.science.data["entries"][before:])
            report_prompt = (prompt + "\nExecuted plan: " + json.dumps(first.get("response")) +
                "\nRetained evidence summaries: " + json.dumps(observation) +
                "\nReturn final report. Evidence values must be exact retained handle names, not fabricated data. "
                "A requires alignment and edge_delta; B also site_delta and neighborhood; C also site_delta and components. "
                "Unused evidence fields may be empty strings. Counts are missing/extra atoms and removed/added matched periodic edges. "
                "Use uncertain if you lack required evidence. No further tools or replanning.")
            second = self.model("research", label + "-report", report_prompt, REPORT_SCHEMA)
            row["completed"] = second["status"] == "ok"
            if self.budget.blocked:
                raise BudgetStop(self.budget.blocked)
            if second["status"] == "ok" and output is not None and runtime_seconds < 20:
                row.update(self.science.run("research_score", label, dict(action="score", pair=pair, output=output,
                           report=second["response"]), seconds=20 - runtime_seconds))
            else:
                row["error"] = "Missing evidence, invalid report, or episode computation limit"
        finally:
            calls = self.budget.entries[start_index:]
            row["calls"] = len(calls)
            row["tokens"] = None if any(c.get("tokens") is None for c in calls) else sum(c["tokens"] for c in calls)
            row["model_wall_seconds"] = sum(c.get("wall_seconds") or 0 for c in calls)
            write_json(self.output / "episodes.json", self.rows)
            print(json.dumps({k: row.get(k) for k in ("id", "arm", "passed", "calls", "tokens", "error")}), flush=True)

    def run(self):
        stop = None
        try:
            probe_schema = {"type": "object", "properties": {"ready": {"type": "boolean"}},
                            "required": ["ready"], "additionalProperties": False}
            for index in range(2):
                probe = self.model("probe", f"probe-{index + 1}",
                                   "Return ready=true. This is a transport-only check. Do not use tools.", probe_schema)
                if self.budget.blocked or probe["status"] != "ok" or probe["response"] != {"ready": True}:
                    raise BudgetStop(self.budget.blocked or "Transport admission probe failed")
            candidate = self.author()
            order = list(self.pairs)
            random.Random(91723).shuffle(order)
            schedule = [(p, ARMS[(i + k) % 4]) for i, p in enumerate(order) for k in range(4)]
            write_json(self.output / "schedule.json", [dict(id=p["id"], arm=arm) for p, arm in schedule])
            for pair, arm in schedule:
                self.episode(pair, arm, candidate)
        except Exception as exc:
            stop = f"{type(exc).__name__}: {exc}"
        self.unchanged()
        write_json(self.output / "completion.json", dict(completed=len(self.rows) == 96 and stop is None,
                    stop=stop, episodes=len(self.rows), model_calls=len(self.budget.entries), frozen_unchanged=True))
        print(json.dumps(self.read("completion.json")), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--codex")
    args = parser.parse_args()
    Pilot(args.output, args.codex).run()


if __name__ == "__main__":
    main()

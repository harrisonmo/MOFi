"""Durable, fail-closed model and scientific reservations for one approved run."""

import time

from rsi.experiment import write_json


class BudgetStop(RuntimeError):
    pass


class RunBudget:
    def __init__(self, path, limits):
        self.path, self.limits = path, limits
        self.entries = []
        self.blocked = None
        self.save()

    def save(self):
        write_json(self.path, {"limits": self.limits, "blocked": self.blocked, "entries": self.entries})

    def check(self, stage):
        models = [r for r in self.entries if r["kind"] == "model"]
        if self.blocked:
            raise BudgetStop(self.blocked)
        if any(r["status"] == "reserved" for r in self.entries):
            raise BudgetStop("Unresolved reservation")
        if len(models) >= self.limits["total_model_calls"]:
            raise BudgetStop("Total model call limit")
        cap = self.limits["generation_and_debug_calls"] if stage == "author" else self.limits["research_calls"]
        if sum(r["stage"] == stage for r in models) >= cap:
            raise BudgetStop("Stage model call limit")
        if sum(r["tokens"] for r in models) >= self.limits["observed_token_soft_limit"]:
            raise BudgetStop("Observed token limit")

    def reserve_model(self, stage, label):
        if stage not in {"author", "research"}:
            raise ValueError("Unknown model stage")
        self.check(stage)
        row = {"kind": "model", "stage": stage, "label": label, "status": "reserved", "tokens": None}
        self.entries.append(row)
        self.save()
        return row

    def finish_model(self, row, result):
        usage = result.get("usage")
        known = (type(usage) is dict and all(type(usage.get(k)) is int and usage[k] >= 0
                                           for k in ("input_tokens", "output_tokens")))
        tokens = usage["input_tokens"] + usage["output_tokens"] if known else None
        row.update(status=result.get("status", "error"), usage=usage, tokens=tokens,
                   wall_seconds=result.get("duration_seconds"), actual_model=result.get("actual_model"),
                   requested_model=result.get("requested_model"), model_evidence=result.get("model_evidence", []))
        if tokens is None:
            self.blocked = "Unknown model usage"
        elif tokens > self.limits["tokens_per_call_soft_limit"]:
            self.blocked = "Per-call soft token limit exceeded"
        elif sum(r["tokens"] or 0 for r in self.entries if r["kind"] == "model") >= self.limits["observed_token_soft_limit"]:
            self.blocked = "Total soft token limit reached"
        self.save()
        return tokens

    def reserve_science(self, label, steps):
        rows = [r for r in self.entries if r["kind"] == "science"]
        if self.blocked or any(r["status"] == "reserved" for r in self.entries):
            raise BudgetStop("Blocked or unresolved reservation")
        if (type(steps) is not int or steps < 0 or steps > self.limits["uff_steps_per_trajectory"]
                or len(rows) >= self.limits["new_scientific_trajectories"]
                or sum(r["requested_steps"] for r in rows) + steps > self.limits["total_uff_requested_steps"]):
            raise BudgetStop("Scientific budget exceeded")
        row = dict(kind="science", label=label, requested_steps=steps, status="reserved", started=time.time())
        self.entries.append(row)
        self.save()
        return row

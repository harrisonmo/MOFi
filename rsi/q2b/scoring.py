"""Evidence-bound reporting and independent scoring for frozen Q2B tasks."""
from .reference import verify


def lookup(output, ref):
    evidence = output["evidence"]
    if ref in evidence:
        return evidence[ref]
    parts = ref.split(".")
    current = evidence
    for part in parts:
        current = current[part]
    return current


def score(pair, output, report):
    try:
        needed = ["alignment", "edge_delta"]
        if pair["case"][0] in "BC":
            needed.append("site_delta")
        if pair["case"][0] == "B":
            needed.append("neighborhood")
        if pair["case"][0] == "C":
            needed.append("components")
        evidence = {k: lookup(output, report["evidence"][k]) for k in needed}
        result = verify(pair["A"], pair["B"], evidence, pair["case"][0])
        expected_sites = {"A0": (0, 0), "A1": (0, 0), "B0": (1, 0), "B1": (0, 1), "C0": (5, 0), "C1": (5, 0)}
        if (result["missing"], result["extra"]) != expected_sites[pair["case"]]:
            raise ValueError("Certificate hides correspondences required by the edit provenance")
        passed = report["verdict"] == pair["expected"] == result["verdict"]
        passed &= all(type(report[k]) is int and report[k] == result[k]
                      for k in ("missing", "extra", "removed_edges", "added_edges"))
        return dict(passed=bool(passed), certificate_valid=True, verified=result,
                    error=None if passed else "Verdict or witness-count mismatch")
    except Exception as exc:
        return dict(passed=False, certificate_valid=False, error=f"{type(exc).__name__}: {exc}")


def summarize(output):
    """Identical evidence presentation for base and registered computations."""
    import hashlib
    import json
    result = {}

    def visit(ref, value):
        if isinstance(value, dict) and {"mapping", "shifts", "origin", "basis"} <= set(value):
            result[ref] = {k: value[k] for k in ("origin", "basis", "residual", "tied_maximum_matches") if k in value}
            result[ref].update(matched=sum(i >= 0 for i in value["mapping"]),
                               unmatched=sum(i < 0 for i in value["mapping"]),
                               certificate_sha256=hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest())
        elif isinstance(value, dict) and any(k in value for k in ("alignment", "site_delta", "edge_delta", "components", "neighborhood")):
            for k, v in value.items():
                visit(ref + "." + k, v)
        elif len(json.dumps(value)) <= 24000:
            result[ref] = value
        else:
            result[ref] = dict(large_retained_table=True, bytes=len(json.dumps(value)),
                               sha256=hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest())
    for ref, value in output["evidence"].items():
        visit(ref, value)
    return dict(input_hash=output["input_hash"], program_hash=output["program_hash"], handles=result,
                operation_count=len([x for x in output["operations"] if x["op"] != "registered"]))

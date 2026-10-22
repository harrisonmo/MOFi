"""Full registered entrypoint, serialization, handle binding and frozen scoring."""
from copy import deepcopy
import json

from jsonschema import validate
from .protocol import plan_schema, REPORT_SCHEMA, TOOL_OP
from .runtime import execute, manual_program, validate_program, digest
from .scoring import score, summarize


def check_registered(pair, reference, projected):
    fields = ["alignment", "site_delta", "edge_delta", "components", "neighborhood"]
    program = dict(version=2, nodes=[dict(id="audit", op=TOOL_OP, inputs=["A", "B"], config="{}")],
                   outputs=["audit." + k for k in fields] if projected else ["audit"])
    validate(dict(program=program), plan_schema(True))
    a_hash, b_hash = digest(pair["A"]), digest(pair["B"])
    output = json.loads(json.dumps(execute(program, pair["A"], pair["B"], manual_program())))
    assert digest(pair["A"]) == a_hash and digest(pair["B"]) == b_hash
    observed = summarize(output)
    assert observed["operation_count"] == 9
    assert set(observed["handles"]) == {"audit." + k for k in fields}
    report = dict(verdict=pair["expected"], evidence={k: "audit." + k for k in fields},
                  **{k: reference[k] for k in ("missing", "extra", "removed_edges", "added_edges")})
    validate(report, REPORT_SCHEMA)
    result = score(pair, output, report)
    assert result["passed"], result
    return dict(id=pair["id"], projected=projected, passed=True,
                program_hash=output["program_hash"], input_hash=output["input_hash"], result=result,
                primitive_operations=observed["operation_count"])

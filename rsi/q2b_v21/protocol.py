"""Frozen model transport and catalog, identical across research arms."""


def obj(props):
    return dict(type="object", properties=props, required=list(props), additionalProperties=False)


TEXT = {"type": "string"}
from copy import deepcopy
from .runtime import OPS

TOOL_OP = "periodic_pair_evidence"
PROGRAM = obj(dict(version={"type": "integer", "enum": [2]},
                   nodes={"type": "array", "minItems": 1, "maxItems": 128,
                          "items": obj(dict(id=TEXT, op=TEXT, inputs={"type": "array", "items": TEXT}, config=TEXT))},
                   outputs={"type": "array", "items": TEXT}))
AUTHOR_SCHEMA = obj(dict(hypothesis=TEXT, program=PROGRAM))
PLAN_SCHEMA = obj(dict(program=PROGRAM))


def plan_schema(tool_available):
    schema = deepcopy(PLAN_SCHEMA)
    program = schema["properties"]["program"]
    program["properties"]["nodes"]["maxItems"] = 16
    program["properties"]["nodes"]["items"]["properties"]["op"] = {
        "type": "string", "enum": sorted(OPS) + ([TOOL_OP] if tool_available else [])}
    return schema


AUTHOR_SCHEMA["properties"]["program"]["properties"]["nodes"]["items"]["properties"]["op"] = {
    "type": "string", "enum": sorted(OPS)}
FIELDS = ["alignment", "site_delta", "edge_delta", "components", "neighborhood"]
REPORT_SCHEMA = obj(dict(verdict={"type": "string", "enum": ["equivalent", "different", "localized", "authorized_removal", "unauthorized_change", "uncertain"]},
                         evidence=obj({k: TEXT for k in FIELDS}),
                         **{k: {"type": "integer", "minimum": 0} for k in ("missing", "extra", "removed_edges", "added_edges")}))
CATALOG = """Pure immutable DAG v2. Inputs A,B are bound structure tables, each with
unit=angstrom, lattice row vectors, frac, elements, occupancy, edges [i,j,nx,ny,nz,order].
No file access, Python, model calls, task decisions, retries, repairs or actions.
Each node has id, op, inputs (earlier ids or dot paths), config (a JSON object encoded as a string).
All scientific ops take config="{}". Output list contains refs to retain as evidence.
Scientific ops and argument order:
basis(A,B): integer unimodular matrix M with lattice_B=M*lattice_A, normalized frac_B*M. Refuses unsupported cells.
origins(A,B,basis): origin candidates from the least frequent common element; at most64.
distances(A,B,basis,origins): exact ASE MIC distance matrices, at most4M elements.
assignment(A,B,distances): SciPy one-to-one assignment for each origin, equal element/occupancy,
1e-5 A coordinate tolerance and1e-6 occupancy tolerance; maximizes matches then minimizes residual.
alignment(A,B,basis,origins,assignment): selected mapping B->A (-1 unmatched), integer shifts, origin,basis,residual and tied candidate count.
site_delta(A,B,alignment): missing A ids, extra B ids, nearest A and distance for each extra.
edge_delta(A,B,alignment): canonical periodic edge multiset removed/added (A ids) and unmapped B edges.
components(A): atom ids, composition and periodic-cycle flag, using frozen Crystal-Builder BondGraph.
neighborhood(A,site_delta,edge_delta): affected A seeds (missing, nearest-to-extra, changed-edge endpoints) and union within two graph hops.
Table ops: project(table) config={fields:[...]}; filter(table) config={field:...,equals:...};
sort(table),group(table),join(left,right) config={field:...}; map(table) config={field:...,operation:abs|negate}.
All tables bounded32768 rows, join products bounded32768. No dynamic operations.
Structures <=2048 atoms. Manifest <=32KiB, research <=16 top nodes, expanded <=128,
2GiB process memory,20s complete batch, no retries. The fixed baseline may compose any of these operations per task.
Evidence summaries omit long numeric tensors but retained handles bind full results to exact input/program hashes.
Never infer success solely from composition, degree or atom count. Stored edges are the object of audit; do not reperceive them.
"""
AUTHOR = """You are the outer tool author. Create one reusable pure periodic_pair_evidence(A,B)
program using the catalog. This is outer tool evolution, not inner autonomous tool invention.
The development task is equivalence of periodic structure records under permutation, origin translation,
wrap and supported integer basis changes; periodic edge images must be preserved.
Return hypothesis and a version2 program, no actions or verdict. Return all five generic evidence outputs
named alignment,site_delta,edge_delta,components,neighborhood (use these as the final node ids).
Only A/B inputs and earlier refs are available. A program cannot call periodic_pair_evidence or recurse.
You may receive development validation feedback, at most three author calls, first valid freezes.
You cannot read or modify evaluators, reference tools, other tasks or evaluation inputs.
"""
QUESTIONS = {
    "A": "Audit whether B represents the same atoms, occupancies, lattice and stored periodic edges as A. Report equivalent or different with a correspondence certificate and periodic edge witnesses.",
    "B": "Localize the difference between A and B: missing/extra sites, nearest reference position for extras, affected periodic edges and reference two-hop neighborhood. Report localized only with complete valid witnesses.",
    "C": "Audit a removal: the authorized target is one synthetic finite CH4 component of A. Every periodic framework component and all its edges must be retained. Report authorized_removal or unauthorized_change with correspondence, removed sites, components and edge witnesses. Matching deleted element counts is insufficient.",
}


def research_prompt(pair, registered=None):
    import json
    from collections import Counter
    info = {k: dict(atoms=len(pair[k]["frac"]), elements=dict(Counter(pair[k]["elements"])),
                    recorded_edges=len(pair[k]["edges"])) for k in ("A", "B")}
    tool = ("Additional pure tool periodic_pair_evidence(A,B), config={}; the exact op string is periodic_pair_evidence. Frozen reusable analysis. "
            "Returns fields alignment,site_delta,edge_delta,components,neighborhood. Access as node_id.field.\n") if registered else ""
    return (CATALOG + tool + "One plan, one batch, one final report; no replanning.\n" + QUESTIONS[pair["case"][0]]
            + "\nInput metadata: " + json.dumps(info))

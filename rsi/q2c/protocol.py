"""Identical research workflow and explicit machine-readable ID rules."""
from copy import deepcopy
import json
from .runtime import ID,TOOL
from .primitives import OPS


def obj(props):return dict(type='object',properties=props,required=list(props),additionalProperties=False)
TEXT={'type':'string'}
PROGRAM=obj(dict(version={'type':'integer','enum':[1]},nodes=dict(type='array',minItems=1,maxItems=16,
    items=obj(dict(id=dict(type='string',pattern=ID),op=dict(type='string',enum=sorted(OPS)),
                   inputs=dict(type='array',maxItems=2,items=TEXT)))),
    outputs=dict(type='array',minItems=1,maxItems=8,items=TEXT)))
AUTHOR_SCHEMA=obj(dict(hypothesis=TEXT,program=PROGRAM))
REPORT_SCHEMA=obj(dict(uncertain={'type':'boolean'},evidence=dict(type='array',maxItems=2,items=TEXT),
                       values=dict(type='array',maxItems=32,items={'type':'number'})))


def plan_schema(available):
    p=deepcopy(PROGRAM)
    if available:p['properties']['nodes']['items']['properties']['op']['enum'].append(TOOL)
    return obj(dict(program=p))


CATALOG='''Pure immutable DAG version 1. Inputs A,B are bound records with unit=angstrom,
lattice row vectors, frac, elements, occupancy=1, edges [i,j,tx,ty,tz]. Edges are stored
records, not inferred chemical bonds. IDs must match ^[a-z][a-z0-9_]{0,31}$.
Each node has id,op,inputs. No config or constants. Inputs name earlier nodes or A,B;
dot paths project named dictionary fields. outputs is a list of retained computed refs.
cycle_groups(A): weighted periodic components with atoms and integer cycle vectors.
integer_lattice(cycle_groups_result): list of {atoms,rank,index}; index is saturation
index of the cycle lattice, not full Z3 index for rank<3. Equal descriptors do not prove isomorphism.
mic_probe(A,B): {squared_distance,multiplicity} for the shortest periodic displacement
from A.frac[0] to B.frac[0], using the common cell; units angstrom squared.
translation_maps(A,B): {multiplicity,matches} for exact same-species bijections modulo
a global translation in the same cell. Every symmetric match counts, not just one.
All primitives are shared across arms. No Python, files, models, labels, repair,
retry or research decisions. At most16 top nodes,64 expanded ops,20s,2GiB.
One plan, one execution, one report, no replanning. A tool error requires uncertain=true.
'''
AUTHOR='''You are an outer tool author. Create a reusable pure evidence DAG from the catalog.
Expose exactly graph_a,graph_b,mic,mappings as final node IDs and outputs. graph_a/b
are integer_lattice results for A/B; mic is mic_probe result; mappings is translation_maps.
The registered name will be periodic_evidence(A,B). Do not output decisions or workflow.
Only development G examples follow; do not inspect files, evaluators or evaluation data.
First independently valid candidate freezes. This is outer tool evolution only.\n'''
QUESTIONS={
 'G':'Report [rank,index] for each component sorted lexicographically, flattened for A then B. Retain two lattice-result handles in A,B order. Do not claim full network equivalence.',
 'M':'Report [squared_distance,multiplicity] of the shortest periodic displacement A.frac[0] to B.frac[0]. Retain one mic_probe-result handle.',
 'S':'Report [multiplicity] of exact equal-species bijections modulo global translation. Retain one translation_maps-result handle; count all symmetric matches.'}


def research_prompt(task,available):
    meta={k:dict(atoms=len(task[k]['frac']),species=task[k]['elements'],edges=len(task[k]['edges'])) for k in ('A','B')}
    tool='Additional op periodic_evidence(A,B) returns {graph_a,graph_b,mic,mappings}. Use node.field refs; exact op name periodic_evidence.\n' if available else ''
    return CATALOG+tool+QUESTIONS[task['family']]+'\nInput metadata only: '+json.dumps(meta)

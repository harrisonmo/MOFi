"""Versioned pure-DAG registry and bounded execution; no dynamic code."""
from copy import deepcopy
import hashlib
import json
import re
from .primitives import OPS,primitive,validate

TOOL='periodic_evidence'
ID=r'^[a-z][a-z0-9_]{0,31}$'


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def check(program,tool=None,author=False):
    if set(program)!={'version','nodes','outputs'} or program['version']!=1 or len(json.dumps(program))>32768:
        raise ValueError('Manifest contract')
    if not 1<=len(program['nodes'])<=16:raise ValueError('Top node limit')
    known={'A','B'};count=0
    for node in program['nodes']:
        if set(node)!={'id','op','inputs'} or not re.fullmatch(ID,node['id']) or node['id'] in known:
            raise ValueError('Invalid or duplicate node ID')
        op=node['op'];arity=2 if op==TOOL and tool is not None and not author else OPS.get(op)
        if arity is None or len(node['inputs'])!=arity:raise ValueError('Unavailable operation or arity')
        for ref in node['inputs']:
            if ref in ('A','B'):continue
            if not isinstance(ref,str) or any(not re.fullmatch(ID,p) for p in ref.split('.')) or ref.split('.')[0] not in known:
                raise ValueError('Invalid reference')
        known.add(node['id']);count+=check(tool,author=True) if op==TOOL else 1
    if count>64:raise ValueError('Expanded operation limit')
    if not 1<=len(program['outputs'])<=8:raise ValueError('Output count')
    for ref in program['outputs']:
        if ref.split('.')[0] not in known-{'A','B'}:raise ValueError('Only retained computed outputs allowed')
    if author and set(program['outputs'])!={'graph_a','graph_b','mic','mappings'}:
        raise ValueError('Tool must expose graph_a,graph_b,mic,mappings')
    return count


def manual_program():
    nodes=[dict(id='ca',op='cycle_groups',inputs=['A']),dict(id='cb',op='cycle_groups',inputs=['B']),
           dict(id='graph_a',op='integer_lattice',inputs=['ca']),dict(id='graph_b',op='integer_lattice',inputs=['cb']),
           dict(id='mic',op='mic_probe',inputs=['A','B']),dict(id='mappings',op='translation_maps',inputs=['A','B'])]
    return dict(version=1,nodes=nodes,outputs=['graph_a','graph_b','mic','mappings'])


def resolve(values,ref):
    parts=ref.split('.');value=values[parts[0]]
    for part in parts[1:]:value=value[part]
    return value


def execute(program,a,b,tool=None):
    count=check(program,tool);validate(a);validate(b)
    bound=digest([a,b]);values={'A':deepcopy(a),'B':deepcopy(b)}
    for node in program['nodes']:
        args=[resolve(values,r) for r in node['inputs']]
        values[node['id']]=execute(tool,*args)['outputs'] if node['op']==TOOL else primitive(node['op'],args)
    if digest([a,b])!=bound:raise ValueError('Input mutation')
    return dict(input_hash=bound,program_hash=digest(program),operations=count,
                outputs={r:deepcopy(resolve(values,r)) for r in program['outputs']})

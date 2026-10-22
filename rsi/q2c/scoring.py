"""Frozen independent reference targets and evidence-bound reporting score."""
from fractions import Fraction as F
import math
from validation.q2c_reference import minimum_image,periodic_signature,translation_matchings
from .runtime import digest


def graph_values(rows,reference=False):
    key='saturation_index' if reference else 'index'
    return [x for pair in sorted((r['rank'],r[key]) for r in rows) for x in pair]


def target(task):
    a,b=task['A'],task['B']
    if task['family']=='G':
        return graph_values(periodic_signature(len(a['frac']),a['edges']),True)+graph_values(periodic_signature(len(b['frac']),b['edges']),True)
    if task['family']=='M':
        d=[F(str(y))-F(str(x)) for x,y in zip(a['frac'][0],b['frac'][0])]
        r=minimum_image(d,a['lattice'],a['unit'])
        return [float(F(r['distance_squared'])),len(r['images'])]
    if task['family']=='S':
        return [translation_matchings(a['frac'],b['frac'],a['elements'],b['elements'],a['lattice'],a['occupancy'],a['unit'])['multiplicity']]
    raise ValueError('Unknown task family')


def evidence_values(family,outputs,handles):
    values=[outputs[h] for h in handles]
    if family=='G' and len(values)==2:return graph_values(values[0])+graph_values(values[1])
    if family=='M' and len(values)==1:return [values[0]['squared_distance'],values[0]['multiplicity']]
    if family=='S' and len(values)==1:return [values[0]['multiplicity']]
    raise ValueError('Incomplete evidence handles')


def equivalent(a,b):
    return len(a)==len(b) and all(type(x) in (int,float) and math.isfinite(x) and abs(x-y)<=1e-10 for x,y in zip(a,b))


def score(task,output,report):
    expected=target(task)
    try:
        observed=evidence_values(task['family'],output['outputs'],report['evidence'])
        valid=output['input_hash']==digest([task['A'],task['B']]) and equivalent(observed,expected)
        passed=valid and not report['uncertain'] and equivalent(report['values'],expected)
    except (KeyError,ValueError,TypeError):valid=passed=False
    return dict(passed=bool(passed),certificate_valid=bool(valid),expected=expected)

"""Shared scientific primitives; no reference-oracle or evaluator imports."""
from itertools import combinations, permutations, product
from math import gcd
import numpy as np
from ase.geometry import find_mic


def validate(a):
    if set(a)!={'unit','lattice','frac','elements','occupancy','edges'} or a['unit']!='angstrom':
        raise ValueError('Structure schema or unit')
    n=len(a['frac'])
    if not 1<=n<=8 or len(a['elements'])!=n or len(a['occupancy'])!=n or any(x!=1 for x in a['occupancy']):
        raise ValueError('Only 1..8 fully occupied sites supported')
    lattice,frac=np.asarray(a['lattice'],float),np.asarray(a['frac'],float)
    if lattice.shape!=(3,3) or frac.shape!=(n,3) or not np.isfinite(lattice).all() or not np.isfinite(frac).all():
        raise ValueError('Invalid geometry')
    if abs(np.linalg.det(lattice))<1e-8 or max(abs(frac).flat)>1000 or max(abs(lattice).flat)>1000:
        raise ValueError('Unsupported cell or coordinates')
    if len(a['edges'])>64 or any(not isinstance(e,str) or not e for e in a['elements']):
        raise ValueError('Graph or species limit')
    seen=set()
    for e in a['edges']:
        if len(e)!=5 or any(type(v)is not int for v in e): raise ValueError('Integer edge required')
        i,j,x,y,z=e
        if not 0<=i<n or not 0<=j<n or max(abs(x),abs(y),abs(z))>32: raise ValueError('Edge out of scope')
        key=min(tuple(e),(j,i,-x,-y,-z))
        if key in seen or (i==j and (x,y,z)==(0,0,0)): raise ValueError('Duplicate or zero edge')
        seen.add(key)


def cycle_groups(a):
    """Weighted union-find, distinct from the independent BFS reference."""
    n=len(a['frac']);parent=list(range(n));weight=[[0,0,0] for _ in range(n)];cycles=[]
    def find(i):
        total=[0,0,0]
        while parent[i]!=i:
            total=[x+y for x,y in zip(total,weight[i])];i=parent[i]
        return i,total
    for i,j,*t in a['edges']:
        ri,wi=find(i);rj,wj=find(j)
        offset=[wi[k]+t[k]-wj[k] for k in range(3)]
        if ri==rj:
            if any(offset):cycles.append((i,offset))
        else:
            parent[rj]=ri;weight[rj]=offset
    groups={}
    for i in range(n):groups.setdefault(find(i)[0],dict(atoms=[],cycles=[]))['atoms'].append(i)
    for i,t in cycles:groups[find(i)[0]]['cycles'].append(t)
    return sorted(groups.values(),key=lambda g:g['atoms'][0])


def integer_lattice(groups):
    """Explicit cross products and scalar products, with integer arithmetic."""
    def cross(a,b):return [a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]]
    result=[]
    for group in groups:
        rows=group['cycles'];g1=g2=g3=0
        for row in rows:
            for value in row:g1=gcd(g1,abs(value))
        for a,b in combinations(rows,2):
            for value in cross(a,b):g2=gcd(g2,abs(value))
        for a,b,c in combinations(rows,3):g3=gcd(g3,abs(sum(x*y for x,y in zip(a,cross(b,c)))))
        rank,index=(3,g3) if g3 else (2,g2) if g2 else (1,g1) if g1 else (0,1)
        result.append(dict(atoms=group['atoms'],rank=rank,index=index))
    return result


def mic_probe(a,b):
    lattice=np.asarray(a['lattice'],float)
    if not np.allclose(lattice,b['lattice'],atol=1e-12,rtol=0):raise ValueError('Same cell required')
    delta=(np.asarray(b['frac'][0])-np.asarray(a['frac'][0]))%1
    _,length=find_mic(delta@lattice,lattice,pbc=True)
    squared=float(length)**2
    sigma=float(np.linalg.svd(lattice)[1][-1])
    for radius in range(9):
        if (radius+1-max(abs(delta)))*sigma > float(length)+1e-10:
            shifts=np.array(list(product(range(-radius,radius+1),repeat=3)))
            values=np.sum(((delta-shifts)@lattice)**2,axis=1)
            if abs(float(values.min())-squared)>1e-10:raise ValueError('MIC enumeration disagrees')
            return dict(squared_distance=squared,multiplicity=int(np.sum(abs(values-squared)<1e-10)))
    raise ValueError('MIC radius exceeds 8')


def translation_maps(a,b):
    if len(a['frac'])!=len(b['frac']) or not np.allclose(a['lattice'],b['lattice'],atol=1e-12,rtol=0):
        raise ValueError('Same cell and site count required')
    aa,bb=np.array(a['frac'],float),np.array(b['frac'],float);matches=[]
    for mapping in permutations(range(len(aa))):
        if any(a['elements'][i]!=b['elements'][j] for j,i in enumerate(mapping)):continue
        residual=bb-aa[list(mapping)];origin=residual[0]%1
        if np.max(abs(residual-origin-np.rint(residual-origin)))<1e-10:
            matches.append(dict(mapping=list(mapping),origin=origin.tolist()))
    return dict(multiplicity=len(matches),matches=matches)


OPS={'cycle_groups':1,'integer_lattice':1,'mic_probe':2,'translation_maps':2}


def primitive(op,args):
    return {'cycle_groups':cycle_groups,'integer_lattice':integer_lattice,
            'mic_probe':mic_probe,'translation_maps':translation_maps}[op](*args)

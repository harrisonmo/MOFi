"""Independent analytic witnesses, adversarial inputs, and metamorphic checks."""
from fractions import Fraction as F
import json
from pathlib import Path

import pytest
from validation.q2c_reference import OutOfScope, minimum_image, periodic_signature, translation_matchings

CASES = json.loads((Path(__file__).resolve().parents[1] / 'benchmarks/q2c_reference/cases.json').read_text())
CELL = [[1,0,0],[0,1,0],[0,0,1]]


@pytest.mark.parametrize('case', CASES['graph_cases'], ids=lambda x:x['id'])
def test_analytic_graph(case):
    result = periodic_signature(case['atoms'], case['edges'])
    assert len(result) == 1
    assert (result[0]['rank'], result[0]['saturation_index']) == (case['rank'],case['index'])


@pytest.mark.parametrize('case', CASES['distance_cases'], ids=lambda x:x['id'])
def test_analytic_distance(case):
    result = minimum_image(case['delta'], case['cell'])
    assert result['distance_squared'] == case['squared']
    assert result['images'] == case['images']
    assert F(result['outside_lower_bound_squared']) > F(case['squared'])


def test_same_degree_different_periodic_rank():
    chain = periodic_signature(1, [[0,0,1,0,0],[0,0,2,0,0]])[0]
    layer = periodic_signature(1, [[0,0,1,0,0],[0,0,0,1,0]])[0]
    assert chain['rank'] == 1 and layer['rank'] == 2


def test_equal_signatures_do_not_prove_graph_equivalence():
    from itertools import product
    a = periodic_signature(1, [[0,0,1,0,0],[0,0,2,0,0]])
    b = periodic_signature(1, [[0,0,1,0,0],[0,0,3,0,0]])
    assert a == b
    # The first infinite graph has triangles; the second is bipartite.
    assert any(sum(cycle) == 0 for cycle in product([-2,-1,1,2], repeat=3))
    assert not any(sum(cycle) == 0 for cycle in product([-3,-1,1,3], repeat=3))


@pytest.mark.parametrize('case', CASES['graph_cases'])
def test_graph_basis_and_edge_orientation_invariance(case):
    transformed = [[j,i,-x-y,-y,-z] for i,j,x,y,z in reversed(case['edges'])]
    expected = periodic_signature(case['atoms'],case['edges'])
    assert periodic_signature(case['atoms'],transformed) == expected


def test_site_gauge_permutation_and_components():
    edges = [[0,1,0,0,0],[1,2,0,0,0],[2,0,1,0,0]]
    gauges = [[2,-1,1],[-2,0,1],[1,2,-1]]
    perm = [2,0,1]
    changed = [[perm[i],perm[j],*[v[k]+gauges[i][k]-gauges[j][k] for k in range(3)]] for i,j,*v in edges]
    assert periodic_signature(3,changed) == periodic_signature(3,edges)
    disconnected = periodic_signature(4,edges)
    assert sorted(r['rank'] for r in disconnected) == [0,1]


def test_distance_wrap_scale_and_unimodular_basis():
    case = CASES['distance_cases'][0]
    expected = F(case['squared'])
    assert F(minimum_image(['149/100','-151/100',3],case['cell'])['distance_squared']) == expected
    scaled = [[F(str(x))*2 for x in r] for r in case['cell']]
    assert F(minimum_image(case['delta'],scaled)['distance_squared']) == expected*4
    # L'=U L and d'=d U^-1 for U=[[1,1,0],[0,1,0],[0,0,1]].
    changed = [['19/10','1/5',0],['9/10','1/5',0],[0,0,1]]
    assert F(minimum_image(['49/100',0,0],changed)['distance_squared']) == expected


def test_symmetric_mapping_enumerates_both_valid_answers():
    positions = [[0,0,0],['1/2',0,0]]
    result = translation_matchings(positions,positions,['C','C'],['C','C'],CELL)
    assert result == dict(matches=[dict(mapping=[0,1],origin=['0','0','0']),
                                  dict(mapping=[1,0],origin=['1/2','0','0'])],multiplicity=2,ambiguous=True)
    unique = translation_matchings(positions,positions,['C','O'],['C','O'],CELL)
    assert unique['multiplicity'] == 1
    assert translation_matchings(positions,positions,['C','O'],['N','N'],CELL)['multiplicity'] == 0


@pytest.mark.parametrize('unit', ['nm','bohr',None])
def test_wrong_units_rejected(unit):
    with pytest.raises(OutOfScope): minimum_image([0,0,0],CELL,unit)


@pytest.mark.parametrize('bad', [float('nan'),float('inf'),True,'1/10000000',1001])
def test_invalid_numeric_inputs(bad):
    with pytest.raises(OutOfScope): minimum_image([bad,0,0],CELL)


def test_singular_and_search_scope():
    with pytest.raises(OutOfScope): minimum_image([0,0,0],[[0,0,0]]*3)
    with pytest.raises(OutOfScope): minimum_image(['1/2','1/2','1/2'],[[1,0,0],[0,1,0],[0,0,'1/1000']])


@pytest.mark.parametrize('edges', [[[0,0,0,0,0]], [[0,1,0,0,0]], [[0,0,True,0,0]], [[0,0,1,0,0],[0,0,-1,0,0]]])
def test_invalid_graphs(edges):
    with pytest.raises(OutOfScope): periodic_signature(1,edges)


def test_partial_occupancy_and_supercell_out_of_scope():
    with pytest.raises(OutOfScope): translation_matchings([[0,0,0]],[[0,0,0]],['C'],['C'],CELL,[.5])
    with pytest.raises(OutOfScope): translation_matchings([[0,0,0]],[[0,0,0],[.5,0,0]],['C'],['C','C'],CELL)

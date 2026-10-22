"""Small exact oracles, independent of candidate code and Q2 scoring.

Row-vector cells, angstrom lengths, undirected integer-image edges.
Graph invariants are necessary descriptors, not an isomorphism certificate.
"""
from fractions import Fraction as F
from itertools import combinations, permutations, product
from math import gcd


class OutOfScope(ValueError):
    """The requested input exceeds this oracle's declared domain."""


def rational(x):
    if isinstance(x, bool) or not isinstance(x, (int, float, str, F)):
        raise OutOfScope('Expected a bounded rational number')
    try:
        value = F(str(x))
    except (ValueError, ZeroDivisionError):
        raise OutOfScope('Nonfinite or invalid number') from None
    if abs(value) > 1000 or value.denominator > 1_000_000:
        raise OutOfScope('Rational magnitude or precision limit')
    return value


def vector(x):
    if not isinstance(x, (list, tuple)) or len(x) != 3:
        raise OutOfScope('Expected a three-vector')
    return tuple(map(rational, x))


def determinant(a):
    if len(a) == 1:
        return a[0][0]
    return sum((-1)**i * a[0][i] * determinant([r[:i] + r[i+1:] for r in a[1:]]) for i in range(len(a)))


def cell(a, unit):
    if unit != 'angstrom' or not isinstance(a, (list, tuple)) or len(a) != 3:
        raise OutOfScope('Expected an angstrom 3x3 cell')
    result = [list(vector(r)) for r in a]
    if determinant(result) == 0:
        raise OutOfScope('Singular cell')
    return result


def minimum_image(delta, lattice, unit='angstrom'):
    """Return every nearest image with an exact outside-cube certificate.

For v=(delta-n)L, ||delta-n|| <= ||v|| ||L^-1||_F. Outside
[-R,R]^3 the infinity norm exceeds R+1-max(abs(delta)). Thus a
strict lower bound larger than the incumbent certifies all ties found.
"""
    d = vector(delta)
    a = cell(lattice, unit)
    d = tuple(x - x.numerator // x.denominator for x in d)
    det = determinant(a)
    inverse_norm2 = sum((determinant([r[:j] + r[j+1:] for k, r in enumerate(a) if k != i]) / det)**2
                        for i in range(3) for j in range(3))
    best, images = None, []
    for radius in range(9):
        for n in product(range(-radius, radius+1), repeat=3):
            if radius and max(map(abs, n)) != radius:
                continue
            cart = [sum((d[i]-n[i])*a[i][j] for i in range(3)) for j in range(3)]
            squared = sum(x*x for x in cart)
            if best is None or squared < best:
                best, images = squared, [n]
            elif squared == best:
                images.append(n)
        outside_bound = (F(radius+1)-max(map(abs, d)))**2 / inverse_norm2
        if outside_bound > best:
            return dict(distance_squared=str(best), reduced_delta=list(map(str, d)),
                        images=[list(n) for n in sorted(images)], radius=radius,
                        outside_lower_bound_squared=str(outside_bound))
    raise OutOfScope('Image search did not certify within radius 8')


def periodic_signature(atom_count, edges):
    """Exact cycle-translation rank and saturation index per quotient component.

The gcd of maximal nonzero minors is the index in the saturated cycle
lattice. For rank < 3 this is not a finite index in all of Z^3.
"""
    if type(atom_count) is not int or not 1 <= atom_count <= 8 or len(edges) > 64:
        raise OutOfScope('Graph size limit')
    adjacency = [[] for _ in range(atom_count)]
    seen_edges = set()
    for edge in edges:
        if len(edge) != 5 or any(type(x) is not int for x in edge):
            raise OutOfScope('Expected i,j,tx,ty,tz integer edges')
        i, j, *t = edge
        if not 0 <= i < atom_count or not 0 <= j < atom_count or max(map(abs, t)) > 32:
            raise OutOfScope('Invalid vertex or image')
        reverse = (j, i, *[-v for v in t])
        key = min(tuple(edge), reverse)
        if key in seen_edges or (i == j and not any(t)):
            raise OutOfScope('Duplicate edge or zero-image self loop')
        seen_edges.add(key)
        adjacency[i].append((j, tuple(t)))
        adjacency[j].append((i, tuple(-v for v in t)))
    remaining, result = set(range(atom_count)), []
    while remaining:
        root = min(remaining)
        offsets, queue, cycles = {root: (0, 0, 0)}, [root], set()
        for i in queue:
            for j, t in adjacency[i]:
                proposed = tuple(offsets[i][k] + t[k] for k in range(3))
                if j not in offsets:
                    offsets[j] = proposed
                    queue.append(j)
                else:
                    v = tuple(proposed[k]-offsets[j][k] for k in range(3))
                    if any(v):
                        cycles.add(min(v, tuple(-x for x in v)))
        remaining.difference_update(offsets)
        rank, index = 0, 1
        for size in (3, 2, 1):
            minor_gcd = 0
            for selected in combinations(sorted(cycles), size):
                for cols in combinations(range(3), size):
                    minor_gcd = gcd(minor_gcd, abs(determinant([[r[j] for j in cols] for r in selected])))
            if minor_gcd:
                rank, index = size, minor_gcd
                break
        result.append(dict(vertices=sorted(offsets), rank=rank, saturation_index=index))
    return result


def translation_matchings(a, b, elements_a, elements_b, lattice, occupancy=None, unit='angstrom'):
    """Enumerate exact equal-species bijections modulo a global translation.

Same-cell, exact-coordinate oracle only. No noise, partial occupancy,
supercell equivalence, rotations, or inference of bonds is supported.
"""
    n = len(a)
    if not 1 <= n <= 8 or len(b) != n or len(elements_a) != n or len(elements_b) != n:
        raise OutOfScope('Same-size 1..8-site inputs required')
    cell(lattice, unit)
    if any(not isinstance(e, str) or not e for e in elements_a + elements_b):
        raise OutOfScope('Explicit species labels required')
    if occupancy is not None and (len(occupancy) != n or any(rational(x) != 1 for x in occupancy)):
        raise OutOfScope('Partial occupancy is unsupported')
    aa, bb = list(map(vector, a)), list(map(vector, b))
    results = []
    for mapping in permutations(range(n)):
        if any(elements_b[j] != elements_a[i] for j, i in enumerate(mapping)):
            continue
        origin = tuple((bb[0][k]-aa[mapping[0]][k]) % 1 for k in range(3))
        if all(all((bb[j][k]-aa[i][k]-origin[k]).denominator == 1 for k in range(3))
               for j, i in enumerate(mapping)):
            results.append(dict(mapping=list(mapping), origin=list(map(str, origin))))
    return dict(matches=results, multiplicity=len(results), ambiguous=len(results)>1)

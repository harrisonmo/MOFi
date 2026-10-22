"""Independent certificate verifier; never imports the candidate runtime."""
from collections import Counter, deque
from itertools import product
import numpy as np


def periodic_components(a):
    adjacency = [[] for _ in a["frac"]]
    for i, j, x, y, z, _ in a["edges"]:
        adjacency[i].append((j, (x, y, z)))
        adjacency[j].append((i, (-x, -y, -z)))
    seen, result = set(), []
    for root in range(len(adjacency)):
        if root in seen:
            continue
        offsets, queue, periodic = {root: (0, 0, 0)}, deque([root]), False
        while queue:
            i = queue.popleft()
            seen.add(i)
            for j, t in adjacency[i]:
                proposed = tuple(offsets[i][k] + t[k] for k in range(3))
                if j in offsets:
                    periodic |= proposed != offsets[j]
                else:
                    offsets[j] = proposed
                    queue.append(j)
        result.append(dict(atoms=sorted(offsets), periodic=bool(periodic),
                           composition=dict(Counter(a["elements"][i] for i in offsets))))
    return result


def edge_key(i, j, image, order):
    forward = (i, j, *image, order)
    reverse = (j, i, *[-v for v in image], order)
    return forward if forward < reverse else reverse


def nearest(a, position):
    """Exhaustive finite lattice search with a singular-value bound."""
    lattice = np.asarray(a["lattice"], float)
    delta = position - np.asarray(a["frac"])
    delta -= np.rint(delta)
    initial = np.linalg.norm(delta @ lattice, axis=1)
    radius = int(np.ceil(float(max(initial)) / np.linalg.svd(lattice)[1][-1] + .5))
    if radius > 12:
        raise ValueError("Reference lattice search out of scope")
    shifts = np.array(list(product(range(-radius, radius + 1), repeat=3)))
    distances = np.linalg.norm((delta[:, None, :] - shifts[None, :, :]) @ lattice, axis=2).min(axis=1)
    best = float(min(distances))
    return {i for i, d in enumerate(distances) if abs(d - best) <= 1e-8}, best


def verify(a, b, evidence, family="A", build=False):
    """Verify raw witnesses, including alternative symmetric alignments."""
    c = evidence["alignment"]
    matrix, origin, shifts = (np.asarray(c[k], float) for k in ("basis", "origin", "shifts"))
    mapping = c["mapping"]
    n, m = len(a["frac"]), len(b["frac"])
    if matrix.shape != (3, 3) or origin.shape != (3,) or shifts.shape != (m, 3) or len(mapping) != m:
        raise ValueError("Certificate shape")
    if not all(np.isfinite(v).all() for v in (matrix, origin, shifts)):
        raise ValueError("Nonfinite certificate")
    if not np.array_equal(matrix, np.rint(matrix)) or not np.array_equal(shifts, np.rint(shifts)) or abs(round(np.linalg.det(matrix))) != 1:
        raise ValueError("Noninteger image or invalid basis")
    if not np.allclose(matrix @ a["lattice"], b["lattice"], rtol=0, atol=1e-7):
        raise ValueError("Lattice mismatch")
    assigned = [i for i in mapping if i != -1]
    if any(type(i) is not int or i < -1 or i >= n for i in mapping) or len(set(assigned)) != len(assigned):
        raise ValueError("Noninjective atom mapping")
    normalized = np.asarray(b["frac"]) @ matrix - origin
    residuals = []
    for j, i in enumerate(mapping):
        if i < 0:
            continue
        residual = float(np.linalg.norm((normalized[j] - a["frac"][i] - shifts[j]) @ a["lattice"]))
        if residual > 1e-5 or b["elements"][j] != a["elements"][i] or abs(b["occupancy"][j] - a["occupancy"][i]) > 1e-6:
            raise ValueError("False atom correspondence")
        residuals.append(residual)
    missing, extra = sorted(set(range(n)) - set(assigned)), [j for j, i in enumerate(mapping) if i < 0]
    # Equal-position compatibility classes must be maximally matched, not hidden.
    for i in missing:
        for j in extra:
            d = normalized[j] - a["frac"][i]
            if (np.linalg.norm((d - np.rint(d)) @ a["lattice"]) <= 1e-5
                    and a["elements"][i] == b["elements"][j]
                    and abs(a["occupancy"][i] - b["occupancy"][j]) <= 1e-6):
                raise ValueError("Incomplete atom mapping")
    left = Counter(edge_key(e[0], e[1], e[2:5], e[5]) for e in a["edges"])
    right, unmapped = Counter(), []
    for e in b["edges"]:
        i, j = e[:2]
        if min(mapping[i], mapping[j]) < 0:
            unmapped.append(e)
        else:
            image = [int(sum(e[2 + k] * matrix[k, d] for k in range(3)) + shifts[j, d] - shifts[i, d]) for d in range(3)]
            right[edge_key(mapping[i], mapping[j], image, e[5])] += 1
    edge_delta = dict(removed=[list(e) for e in sorted((left - right).elements())],
                      added=[list(e) for e in sorted((right - left).elements())], unmapped=unmapped)
    if build:
        evidence["edge_delta"] = edge_delta
    if evidence["edge_delta"] != edge_delta:
        raise ValueError("Incorrect periodic-edge witness")
    components = periodic_components(a)
    if build:
        rows = []
        for j in extra:
            choices, distance = nearest(a, normalized[j])
            rows.append(dict(b=j, a=min(choices), distance=distance))
        evidence["site_delta"] = dict(missing=missing, extra=extra, nearest=rows)
        evidence["components"] = components
    if family in {"B", "C"}:
        sites = evidence["site_delta"]
        if sites["missing"] != missing or sites["extra"] != extra or len(sites["nearest"]) != len(extra):
            raise ValueError("Incorrect site witness")
        if sorted(r["b"] for r in sites["nearest"]) != extra:
            raise ValueError("Repeated nearest witness")
        for row in sites["nearest"]:
            allowed, distance = nearest(a, normalized[row["b"]])
            if row["a"] not in allowed or abs(row["distance"] - distance) > 1e-7:
                raise ValueError("Incorrect nearest witness")
    if family == "B" or build:
        seeds = set(missing) | {r["a"] for r in evidence["site_delta"]["nearest"]}
        seeds |= {i for e in edge_delta["removed"] + edge_delta["added"] for i in e[:2]}
        graph = [set() for _ in range(n)]
        for e in a["edges"]:
            graph[e[0]].add(e[1])
            graph[e[1]].add(e[0])
        reached, frontier = set(seeds), set(seeds)
        for _ in range(2):
            frontier = {j for i in frontier for j in graph[i]} - reached
            reached.update(frontier)
        expected_neighborhood = dict(seeds=sorted(seeds), atoms=sorted(reached), depth=2)
        if build:
            evidence["neighborhood"] = expected_neighborhood
        if evidence["neighborhood"] != expected_neighborhood:
            raise ValueError("Incorrect two-hop witness")
    if family == "C" and evidence["components"] != components:
        raise ValueError("Incorrect component witness")
    unchanged = not missing and not extra and not any(edge_delta.values())
    allowed_guest = [set(f["atoms"]) for f in components if not f["periodic"] and f["composition"] == {"C": 1, "H": 4}]
    framework = {i for f in components if f["periodic"] for i in f["atoms"]}
    authorized = (set(missing) in allowed_guest and not extra and not set(missing) & framework
                  and not edge_delta["added"] and not edge_delta["unmapped"]
                  and all(e[0] in missing or e[1] in missing for e in edge_delta["removed"]))
    verdict = ("equivalent" if unchanged else "different") if family == "A" else (
        "localized" if family == "B" else "authorized_removal" if authorized else "unauthorized_change")
    return dict(verdict=verdict, missing=len(missing), extra=len(extra),
                removed_edges=len(edge_delta["removed"]), added_edges=len(edge_delta["added"]),
                max_coordinate_residual=max(residuals, default=0.), valid=True)


def provenance_certificate(pair):
    a, b, p = pair["A"], pair["B"], pair["provenance"]
    shifts = [[0, 0, 0] if i < 0 else np.rint(np.asarray(b["frac"][j]) @ p["basis"] - p["origin"] - a["frac"][i]).astype(int).tolist()
              for j, i in enumerate(p["mapping"])]
    return dict(mapping=p["mapping"], basis=p["basis"], origin=p["origin"], shifts=shifts)

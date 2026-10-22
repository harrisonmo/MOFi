"""Pure-data, bounded periodic geometry DAG; no research actions or labels."""
from collections import Counter
from copy import deepcopy
import hashlib
import json
import re
import time

import numpy as np
from ase.geometry import find_mic
from scipy.optimize import linear_sum_assignment

TOL = 1e-5
OPS = {"basis": 2, "origins": 3, "distances": 4, "assignment": 3, "alignment": 5,
       "site_delta": 3, "edge_delta": 3, "components": 1, "neighborhood": 3,
       "project": 1, "filter": 1, "sort": 1, "group": 1, "join": 2, "map": 1}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def validate_structure(s):
    if set(s) != {"unit", "lattice", "frac", "elements", "occupancy", "edges"} or s["unit"] != "angstrom":
        raise ValueError("Invalid structure fields or length unit")
    n = len(s["elements"])
    if not 1 <= n <= 2048 or any(not isinstance(e, str) or not re.fullmatch(r"[A-Z][a-z]?", e) for e in s["elements"]):
        raise ValueError("Invalid atom count or element")
    lattice, frac, occupancy = (np.asarray(s[k], dtype=float) for k in ("lattice", "frac", "occupancy"))
    if lattice.shape != (3, 3) or frac.shape != (n, 3) or occupancy.shape != (n,):
        raise ValueError("Invalid array shape")
    if not all(np.isfinite(x).all() for x in (lattice, frac, occupancy)):
        raise ValueError("Nonfinite data")
    if abs(np.linalg.det(lattice)) < 1e-6 or np.linalg.cond(lattice) > 100 or np.max(abs(frac)) > 100:
        raise ValueError("Unsupported lattice or coordinates")
    if np.min(occupancy) <= 0 or np.max(occupancy) > 1:
        raise ValueError("Invalid occupancy")
    if len(s["edges"]) > 32768:
        raise ValueError("Edge shape limit")
    for edge in s["edges"]:
        if len(edge) != 6 or any(type(x) is not int for x in edge[:5]):
            raise ValueError("Invalid integer edge record")
        if not all(0 <= i < n for i in edge[:2]) or max(abs(x) for x in edge[2:5]) > 100:
            raise ValueError("Invalid edge index or image")
        if not isinstance(edge[5], (int, float)) or not np.isfinite(edge[5]) or edge[5] <= 0:
            raise ValueError("Invalid bond order")


def validate_program(program, registered=None, top=True):
    if len(json.dumps(program).encode()) > 32768 or set(program) != {"version", "nodes", "outputs"} or program["version"] != 2:
        raise ValueError("Invalid v2 manifest")
    nodes = program["nodes"]
    if not 1 <= len(nodes) <= (16 if top else 128):
        raise ValueError("Node limit")
    known = {"A", "B"}
    expanded = 0
    for node in nodes:
        if set(node) != {"id", "op", "inputs", "config"} or not re.fullmatch(r"[a-z][a-z0-9_]{0,31}", node["id"]) or node["id"] in known:
            raise ValueError("Invalid or duplicate node")
        arity = 2 if node["op"] == "registered" and registered else OPS.get(node["op"])
        if arity is None or len(node["inputs"]) != arity:
            raise ValueError("Unknown primitive or arity")
        for ref in node["inputs"]:
            if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*(\.[a-z][a-z0-9_]*)*", ref) or ref.split(".")[0] not in known:
                raise ValueError("Forward, cyclic or invalid reference")
        config = json.loads(node["config"])
        if not isinstance(config, dict) or len(node["config"]) > 512:
            raise ValueError("Invalid config")
        allowed = {"project": {"fields"}, "filter": {"field", "equals"}, "sort": {"field"},
                   "group": {"field"}, "join": {"field"}, "map": {"field", "operation"}}.get(node["op"], set())
        if set(config) != allowed:
            raise ValueError("Unexpected config fields")
        expanded += validate_program(registered, top=False) if node["op"] == "registered" else 1
        known.add(node["id"])
    if expanded > 128 or not program["outputs"] or len(program["outputs"]) > 32:
        raise ValueError("Expanded node or output limit")
    for ref in program["outputs"]:
        if ref.split(".")[0] not in known or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*(\.[a-z][a-z0-9_]*)*", ref):
            raise ValueError("Invalid output reference")
    return expanded


def canonical(edge):
    i, j, x, y, z, order = edge
    direct = (i, j, x, y, z, order)
    reverse = (j, i, -x, -y, -z, order)
    return min(direct, reverse)


def primitive(op, args, cfg):
    if op == "basis":
        a, b = args
        raw = np.asarray(b["lattice"]) @ np.linalg.inv(a["lattice"])
        matrix = np.rint(raw).astype(int)
        if not np.allclose(raw, matrix, atol=1e-8, rtol=0) or abs(round(np.linalg.det(matrix))) != 1:
            raise ValueError("Cells are outside integer-unimodular scope")
        return dict(matrix=matrix.tolist(), normalized=(np.asarray(b["frac"]) @ matrix).tolist())
    if op == "origins":
        a, b, basis = args
        common = set(a["elements"]) & set(b["elements"])
        if not common:
            raise ValueError("No common anchor element")
        element = min(common, key=lambda e: (b["elements"].count(e), e))
        i = a["elements"].index(element)
        positions = [j for j, e in enumerate(b["elements"]) if e == element]
        if len(positions) > 64:
            raise ValueError("Origin candidate limit")
        return [(np.asarray(basis["normalized"][j]) - a["frac"][i]).tolist() for j in positions]
    if op == "distances":
        a, b, basis, origins = args
        n, m = len(a["frac"]), len(b["frac"])
        if n * m * len(origins) > 4_000_000:
            raise ValueError("Distance tensor shape limit")
        result = []
        for origin in origins:
            vectors = (np.asarray(basis["normalized"])[None, :, :] - origin - np.asarray(a["frac"])[:, None, :]) @ a["lattice"]
            _, lengths = find_mic(vectors.reshape(-1, 3), a["lattice"], pbc=True)
            result.append(lengths.reshape(n, m))
        return np.asarray(result)
    if op == "assignment":
        a, b, distances = args
        compatible = np.array(a["elements"])[:, None] == np.array(b["elements"])[None, :]
        compatible &= abs(np.asarray(a["occupancy"])[:, None] - b["occupancy"]) <= 1e-6
        candidates = []
        for k, distance in enumerate(distances):
            valid = compatible & (distance <= TOL)
            cost = np.where(valid, distance / TOL, 1e6)
            ii, jj = linear_sum_assignment(cost)
            pairs = [(int(i), int(j)) for i, j in zip(ii, jj) if valid[i, j]]
            residual = sum(float(distance[i, j]) for i, j in pairs)
            candidates.append(dict(origin_index=k, pairs=pairs, residual=residual))
        return sorted(candidates, key=lambda c: (-len(c["pairs"]), c["residual"]))
    if op == "alignment":
        a, b, basis, origins, assignments = args
        chosen = assignments[0]
        origin = np.asarray(origins[chosen["origin_index"]])
        mapping, shifts = [-1] * len(b["frac"]), [[0, 0, 0] for _ in b["frac"]]
        for i, j in chosen["pairs"]:
            mapping[j] = i
            shifts[j] = np.rint(np.asarray(basis["normalized"][j]) - origin - a["frac"][i]).astype(int).tolist()
        return dict(mapping=mapping, shifts=shifts, origin=origin.tolist(), basis=basis["matrix"],
                    residual=chosen["residual"], tied_maximum_matches=sum(len(c["pairs"]) == len(chosen["pairs"]) for c in assignments))
    if op == "site_delta":
        a, b, alignment = args
        mapping = alignment["mapping"]
        missing = sorted(set(range(len(a["frac"]))) - set(mapping))
        extra = [j for j, i in enumerate(mapping) if i < 0]
        nearest = []
        for j in extra:
            pos = np.asarray(b["frac"][j]) @ alignment["basis"] - alignment["origin"]
            _, lengths = find_mic((pos - a["frac"]) @ np.asarray(a["lattice"]), a["lattice"], pbc=True)
            nearest.append(dict(b=j, a=int(np.argmin(lengths)), distance=float(min(lengths))))
        return dict(missing=missing, extra=extra, nearest=nearest)
    if op == "edge_delta":
        a, b, alignment = args
        left = Counter(canonical(e) for e in a["edges"])
        right = Counter()
        unmapped = []
        for e in b["edges"]:
            i, j = e[:2]
            ai, aj = alignment["mapping"][i], alignment["mapping"][j]
            if min(ai, aj) < 0:
                unmapped.append(e)
                continue
            image = np.asarray(e[2:5]) @ alignment["basis"] + np.asarray(alignment["shifts"][j]) - alignment["shifts"][i]
            right[canonical([ai, aj, *image.tolist(), e[5]])] += 1
        return dict(removed=[list(e) for e in sorted((left - right).elements())],
                    added=[list(e) for e in sorted((right - left).elements())], unmapped=unmapped)
    if op == "components":
        from xtal.core.bonding import BondGraph
        from xtal.core.structure import CellBond
        a = args[0]
        bonds = [CellBond(e[0], e[1], tuple(e[2:5]), 0., order=e[5]) for e in a["edges"]]
        return [dict(atoms=list(f.atoms), periodic=bool(f.periodic),
                     composition=dict(Counter(a["elements"][i] for i in f.atoms)))
                for f in BondGraph(len(a["frac"]), bonds).fragments()]
    if op == "neighborhood":
        a, sites, edges = args
        seeds = set(sites["missing"]) | {r["a"] for r in sites["nearest"]}
        seeds |= {i for e in edges["removed"] + edges["added"] for i in e[:2]}
        reached = set(seeds)
        for _ in range(2):
            reached |= {j for e in a["edges"] for i, j in [e[:2], e[1::-1]] if i in reached}
        return dict(seeds=sorted(seeds), atoms=sorted(reached), depth=2)
    table = args[0]
    if not isinstance(table, list) or len(table) > 32768:
        raise ValueError("Table shape limit")
    if op == "project":
        return [{k: row[k] for k in cfg["fields"]} for row in table]
    if op == "filter":
        return [row for row in table if row[cfg["field"]] == cfg["equals"]]
    if op == "sort":
        return sorted(table, key=lambda row: row[cfg["field"]])
    if op == "group":
        return [dict(key=k, count=v) for k, v in Counter(row[cfg["field"]] for row in table).items()]
    if op == "join":
        if len(table) * len(args[1]) > 32768:
            raise ValueError("Join shape limit")
        return [dict(left=l, right=r) for l in table for r in args[1] if l[cfg["field"]] == r[cfg["field"]]]
    if op == "map":
        fn = {"abs": abs, "negate": lambda x: -x}.get(cfg["operation"])
        if fn is None:
            raise ValueError("Unknown map operation")
        return [fn(row[cfg["field"]]) for row in table]
    raise ValueError("Unknown primitive")


def resolve(values, ref):
    parts = ref.split(".")
    value = values[parts[0]]
    for part in parts[1:]:
        value = value[part]
    return value


def jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {k: jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return value


def execute(program, a, b, registered=None, _nested=False):
    validate_structure(a)
    validate_structure(b)
    validate_program(program, registered, top=not _nested)
    values, timings = {"A": deepcopy(a), "B": deepcopy(b)}, []
    for node in program["nodes"]:
        start = time.perf_counter()
        args = [resolve(values, ref) for ref in node["inputs"]]
        if node["op"] == "registered":
            nested = execute(registered, *args, _nested=True)
            value = nested["evidence"]
            timings.extend(nested["operations"])
        else:
            value = primitive(node["op"], args, json.loads(node["config"]))
        values[node["id"]] = value
        timings.append(dict(id=node["id"], op=node["op"], seconds=time.perf_counter() - start))
    evidence = {ref: jsonable(resolve(values, ref)) for ref in program["outputs"]}
    if len(json.dumps(evidence)) > 16_000_000:
        raise ValueError("Output shape limit")
    return dict(evidence=evidence, input_hash=digest([a, b]), program_hash=digest(program), operations=timings)


def manual_program():
    specs = [("basis", ["A", "B"]), ("origins", ["A", "B", "basis"]),
             ("distances", ["A", "B", "basis", "origins"]), ("assignment", ["A", "B", "distances"]),
             ("alignment", ["A", "B", "basis", "origins", "assignment"]),
             ("site_delta", ["A", "B", "alignment"]), ("edge_delta", ["A", "B", "alignment"]),
             ("components", ["A"]), ("neighborhood", ["A", "site_delta", "edge_delta"])]
    return dict(version=2, nodes=[dict(id=op, op=op, inputs=refs, config="{}") for op, refs in specs],
                outputs=["alignment", "site_delta", "edge_delta", "components", "neighborhood"])

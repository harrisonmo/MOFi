"""Periodic graph construction and structure-derived organic-linker families."""
from __future__ import annotations

import hashlib
import json
from collections import Counter

import networkx as nx
import numpy as np
from ase import Atoms
from ase.data import covalent_radii
from ase.neighborlist import neighbor_list


def periodic_graph(structure):
    if not structure.is_ordered:
        raise ValueError("partial_occupancy")
    z = np.array([site.specie.Z for site in structure], dtype=np.int64)
    if not 8 <= len(z) <= 768:
        raise ValueError("atom_count_out_of_bounds")
    if 6 not in z or not any(site.specie.is_metal for site in structure):
        raise ValueError("not_carbon_metal_framework")
    if not np.isfinite(structure.cart_coords).all() or not np.isfinite(structure.lattice.matrix).all() or structure.volume <= 0:
        raise ValueError("invalid_cell_or_coordinates")
    atoms = Atoms(numbers=z, positions=structure.cart_coords, cell=structure.lattice.matrix, pbc=True)
    src, dst, distance, shifts = neighbor_list("ijdS", atoms, 5.0, self_interaction=False)
    if len(src) == 0 or distance.min() < 0.5:
        raise ValueError("empty_graph_or_coincident_atoms")
    if len(src) > 50000:
        raise ValueError("edge_budget_exceeded")
    if len(set(src)) != len(z):
        raise ValueError("isolated_radius_graph_atom")
    order = np.lexsort((shifts[:, 2], shifts[:, 1], shifts[:, 0], dst, src))
    return {"z": z, "src": src[order], "dst": dst[order], "distance": distance[order].astype(np.float32),
            "shifts": shifts[order], "positions": atoms.positions, "cell": atoms.cell.array}


def linker_family(structure, scaffold=False):
    """Hash finite nonmetal organic components, ignoring hydrogen and multiplicity."""
    z = np.array([site.specie.Z for site in structure])
    organic = {i for i, site in enumerate(structure) if not site.specie.is_metal and site.specie.Z != 1}
    atoms = Atoms(numbers=z, positions=structure.cart_coords, cell=structure.lattice.matrix, pbc=True)
    src, dst, shifts = neighbor_list("ijS", atoms, covalent_radii[z] * 1.2, self_interaction=False)
    graph = nx.Graph()
    graph.add_nodes_from((i, {"element": str(z[i])}) for i in organic)
    adjacency = {i: [] for i in organic}
    for i, j, shift in zip(src, dst, shifts):
        if i in organic and j in organic:
            graph.add_edge(int(i), int(j))
            adjacency[i].append((j, shift))
    signatures = []
    for component in nx.connected_components(graph):
        if not any(z[i] == 6 for i in component):
            continue
        root = min(component)
        images = {root: np.zeros(3, dtype=int)}
        stack = [root]
        while stack:
            i = stack.pop()
            for j, shift in adjacency[i]:
                image = images[i] + shift
                if j in images and not np.array_equal(images[j], image):
                    raise ValueError("periodic_organic_component")
                if j not in images:
                    images[j] = image
                    stack.append(j)
        sub = graph.subgraph(component)
        if scaffold:
            core = nx.k_core(sub, k=2)
            # Retain the full acyclic backbone when no cyclic core exists.
            if core.number_of_nodes() == 0:
                core = sub.copy()
            signatures.append({"nodes": core.number_of_nodes(), "edges": core.number_of_edges(),
                               "wl": nx.weisfeiler_lehman_graph_hash(core, iterations=4)})
            continue
        formula = sorted(Counter(str(z[i]) for i in component).items())
        signatures.append({"formula": formula, "nodes": len(component), "edges": sub.number_of_edges(),
                           "wl": nx.weisfeiler_lehman_graph_hash(sub, node_attr="element", iterations=4)})
    if not signatures:
        raise ValueError("no_organic_linker")
    if scaffold:
        # Group by the dominant organic scaffold, regardless of terminal chemistry.
        signatures = [max(signatures, key=lambda x: (x["nodes"], x["edges"], x["wl"]))]
    canonical = sorted({json.dumps(x, sort_keys=True) for x in signatures})
    return hashlib.sha256(json.dumps(canonical).encode()).hexdigest(), canonical


def grouped_split(records):
    groups = {}
    for row in records:
        groups.setdefault(row["family"], []).append(row["id"])
    count = len(records)
    names = ["train", "validation", "heldout"]
    fractions = dict(zip(names, [0.7, 0.15, 0.15]))
    result = {name: [] for name in names}
    for family in sorted(groups, key=lambda k: (-len(groups[k]), hashlib.sha256(("g1-split-v1:" + k).encode()).hexdigest())):
        name = max(names, key=lambda n: fractions[n] * count - len(result[n]))
        result[name].extend(sorted(groups[family]))
    return result

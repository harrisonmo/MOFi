"""Long-range periodic graphs for the G3 smooth-cutoff encoder.

G1 graphs use a 5 A radius, which cannot span typical MOF pores. G3 uses an
8 A radius: three message-passing layers then reach 24 A, while the edge
count grows by (8/5)^3 ~ 4.1x. The G1 50,000-edge admission limit is scaled
by the same volume ratio so that the admission rule stays equivalent.
"""
from __future__ import annotations

import numpy as np
from ase import Atoms
from ase.neighborlist import neighbor_list

CUTOFF = 8.0
MAX_EDGES = int(50_000 * (CUTOFF / 5.0) ** 3)


def periodic_graph(structure, cutoff=CUTOFF, max_edges=MAX_EDGES):
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
    src, dst, distance, shifts = neighbor_list("ijdS", atoms, cutoff, self_interaction=False)
    if len(src) == 0 or distance.min() < 0.5:
        raise ValueError("empty_graph_or_coincident_atoms")
    if len(src) > max_edges:
        raise ValueError("edge_budget_exceeded")
    if len(set(src)) != len(z):
        raise ValueError("isolated_radius_graph_atom")
    order = np.lexsort((shifts[:, 2], shifts[:, 1], shifts[:, 0], dst, src))
    return {"z": z, "src": src[order], "dst": dst[order], "distance": distance[order].astype(np.float32),
            "shifts": shifts[order], "positions": atoms.positions, "cell": atoms.cell.array,
            "cutoff": np.float64(cutoff)}

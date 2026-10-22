"""Controller-only material preparation and provenance-backed synthetic edits."""
from collections import Counter
from copy import deepcopy
import hashlib
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SOURCES = [
    ("D1", "prepared/MOF-5.cif", "c904d0481bd30faea22b13623eceae0f75db81cb4997f82f5746da9b6f164c4a"),
    ("D2", "Ni2Cl2BTDD.cif", "b7b375c3ad0a24dd8e2dad4083ba1107df730c8e48bf25ce86e688cf2d03de9b"),
    ("E1", "prepared/HKUST-1.cif", "7d290dd0653f7b73236010d494d7df79482705ca1bf45f2a248ac4fef3b1d38f"),
    ("E2", "prepared/ZIF-8.cif", "71b13aec4a701e997fc59d6477023bba7933a1b87f442d0a0d02ae279489a2f8"),
]
BASES = [np.eye(3, dtype=int), np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]]),
         np.array([[1, 1, 0], [0, 1, 0], [0, 0, 1]])]


def source(index):
    from ase.geometry import find_mic
    from xtal.agent.session import Session
    from xtal.io import FORMATS
    from xtal.core.p1 import expand
    from xtal.core.bonding import perceive, BondGraph

    role, name, expected = SOURCES[index]
    path = ROOT / "Crystal-Builder/resources/samples" / name
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != expected:
        raise ValueError("Source hash mismatch")
    session = Session(FORMATS.read(path))
    preparation = None
    if role == "D2":
        preparation = session.prepare(steps=["duplicates", "deuterium", "primitive", "disorder", "solvent", "hydrogens"]).to_dict()
        if not preparation["ok"]:
            raise ValueError(f"Preparation refused: {preparation}")
    cell = expand(session.structure)
    bonds = perceive(session.structure)
    fragments = BondGraph(cell.n_atoms, bonds).fragments()
    framework = sorted({i for f in fragments if f.periodic for i in f.atoms})
    counts = Counter(cell.elements[i] for i in framework)
    if not framework or counts["C"] < 1 or counts["H"] < 4:
        raise ValueError(f"Missing periodic framework or C/H capacity: {counts}")
    lattice = np.asarray(cell.lattice.matrix)
    result = dict(unit="angstrom", lattice=lattice.tolist(), frac=cell.frac.tolist(),
                  elements=list(cell.elements), occupancy=cell.occupancy.tolist(),
                  edges=[[b.i, b.j, *map(int, b.image), float(b.order)] for b in bonds])
    # Fixed placement grid and threshold, specified before any model observations.
    positions = [(np.array([k % 4, (k // 4) % 4, k // 16]) + .5) / [4, 4, 2] for k in range(32)]
    vectors = np.array([[0, 0, 0], [1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]], float)
    vectors[1:] *= 1.09 / np.sqrt(3)
    guest = None
    for attempt, center in enumerate(positions, 1):
        cart = center @ lattice + vectors
        _, lengths = find_mic((cart[:, None, :] - cell.cart[None, :, :]).reshape(-1, 3), lattice, pbc=True)
        if float(min(lengths)) >= 1.50:
            guest = dict(frac=(cart @ np.linalg.inv(lattice)).tolist(), attempt=attempt,
                         minimum_separation=float(min(lengths)))
            break
    return dict(role=role, path=name, sha256=digest, preparation=preparation, structure=result,
                framework=framework, counts=dict(counts), guest=guest,
                fragments=[dict(atoms=list(f.atoms), periodic=bool(f.periodic)) for f in fragments])


def transform(a, keep, seed, extra=None, fault=False, boundary=None):
    rng = np.random.default_rng(seed)
    matrix = BASES[seed % len(BASES)]
    inv = np.rint(np.linalg.inv(matrix)).astype(int)
    origin = np.array([.193, .317, .419])
    if boundary is not None:
        origin = -np.asarray(a["frac"])[boundary] + np.array([.9998, .4, .6]) @ matrix
    raw = (np.asarray(a["frac"]) + origin) @ inv
    frac = raw - np.floor(raw) + rng.integers(-1, 2, raw.shape)
    shifts = np.rint(frac @ matrix - np.asarray(a["frac"]) - origin).astype(int)
    order = list(rng.permutation(keep))
    old_new = {int(old): i for i, old in enumerate(order)}
    b = dict(unit="angstrom", lattice=(matrix @ np.asarray(a["lattice"])).tolist(),
             frac=[frac[i].tolist() for i in order], elements=[a["elements"][i] for i in order],
             occupancy=[a["occupancy"][i] for i in order], edges=[])
    for i, j, x, y, z, bond_order in a["edges"]:
        if i in old_new and j in old_new:
            image = (np.array([x, y, z]) - shifts[j] + shifts[i]) @ inv
            b["edges"].append([old_new[i], old_new[j], *image.tolist(), bond_order])
    if fault:
        b["edges"][0][2] += 1
    if extra is not None:
        b["frac"].append((frac[extra] + np.array([.02, 0, 0]) @ np.linalg.inv(np.asarray(b["lattice"]))).tolist())
        b["elements"].append(a["elements"][extra])
        b["occupancy"].append(a["occupancy"][extra])
        order.append(-1)
    return b, dict(mapping=[int(i) for i in order], origin=origin.tolist(), basis=matrix.tolist())


def pairs(record, index):
    if record["guest"] is None:
        raise ValueError(f"No valid CH4 position for {record['role']} in 32 attempts")
    base = record["structure"]
    carbon = next(i for i in record["framework"] if base["elements"][i] == "C")
    hydrogens = [i for i in record["framework"] if base["elements"][i] == "H"][:4]
    out = []
    for case in ("A0", "A1", "B0", "B1", "C0", "C1"):
        a = deepcopy(base)
        n = len(a["elements"])
        guest_ids = []
        if case[0] == "C":
            guest_ids = list(range(n, n + 5))
            a["frac"] += record["guest"]["frac"]
            a["elements"] += ["C", "H", "H", "H", "H"]
            a["occupancy"] += [1.] * 5
            a["edges"] += [[n, n + i, 0, 0, 0, 1.] for i in range(1, 5)]
        removed = ([carbon] if case == "B0" else guest_ids if case == "C0"
                   else [carbon] + hydrogens if case == "C1" else [])
        keep = [i for i in range(len(a["elements"])) if i not in removed]
        seed = 7300 + index * 23 + len(out)
        b, provenance = transform(a, keep, seed, extra=carbon if case == "B1" else None,
                                  fault=case == "A1", boundary=carbon if case == "B1" else None)
        block = ("development" if case[0] == "A" else "task_transfer") if index < 2 else (
            "material_transfer" if case[0] == "A" else "joint_transfer")
        opaque = hashlib.sha256(f"q2b:{seed}".encode()).hexdigest()[:12]
        out.append(dict(id=opaque, source=record["role"], case=case, block=block, A=a, B=b,
                        provenance=provenance, removed=removed, guest=guest_ids,
                        expected={"A0": "equivalent", "A1": "different", "B0": "localized",
                                  "B1": "localized", "C0": "authorized_removal", "C1": "unauthorized_change"}[case]))
    return out

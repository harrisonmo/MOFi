"""Prepare explicit, charge-balanced PFAS-study starting models from raw CIFs.

This paper-specific workflow produces geometry-relaxation candidates, not a
validated aqueous force field or adsorption calculation. See prepared/README.md.
Dependencies are recorded in prepared/requirements.txt. No source CIF is edited.
"""
from __future__ import annotations

import hashlib
import itertools
import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import gemmi
import numpy as np
from rdkit import Chem, rdBase
from rdkit.Chem import AllChem, rdMolDescriptors
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/pfas/anie.1567631"
OUT = SOURCE / "prepared"
EXPECTED = {
    "Zr-PTA1": "4749e07f196b008516d2e1ca6fd93bbed2a4db06be62b22e6621da48657a3634",
    "Zr-PTA2": "d2199d09660c56a259a9746a6552933c0fa8fcf8c59055fb32087ac965ec5561",
}


def dump(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def delta(a, b, cell):
    v = np.asarray(b) - np.asarray(a)
    return (v - np.rint(v)) @ cell


def norm(v):
    return v / np.linalg.norm(v)


def read_raw(name):
    path = SOURCE / f"{name}.cif"
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != EXPECTED[name]:
        raise ValueError(f"Unexpected source hash: {path}")
    # The first published block has a bare, non-CIF 'End' after all data items.
    # Remove that terminator in memory only; preserve every raw source byte.
    text = "\n".join(l for l in raw.decode().splitlines() if l.strip() != "End")
    block = gemmi.cif.read_string(text).sole_block()
    cell = np.diag([gemmi.cif.as_number(block.find_value("_cell_length_" + a)) for a in "abc"])
    angles = [gemmi.cif.as_number(block.find_value("_cell_angle_" + a)) for a in ("alpha", "beta", "gamma")]
    if not np.allclose(angles, 90):
        raise ValueError("This paper-specific preparation requires orthogonal cells")
    operations = [gemmi.Op(gemmi.cif.as_string(x)) for x in block.find_values("_space_group_symop_operation_xyz")]
    keys = ["label", "type_symbol", "fract_x", "fract_y", "fract_z", "occupancy", "disorder_assembly", "disorder_group"]
    atoms, seen = [], set()
    for row in block.find(["_atom_site_" + k for k in keys]):
        label, element, x, y, z, occupancy, assembly, group = list(row)
        # Reconstruct hydrogens from chemistry instead of propagating partially
        # occupied H positions or both core-oxygen protonation assignments.
        if element == "H" or (assembly == "B" and group == "2"):
            continue
        frac = np.array([gemmi.cif.as_number(v) for v in (x, y, z)])
        for op_id, operation in enumerate(operations):
            q = np.asarray(operation.apply_to_xyz(frac)) % 1
            key = (element, *np.round(q, 7))
            if key in seen:
                continue
            seen.add(key)
            atoms.append({"element": element, "frac": q, "source_label": label,
                          "source_operation": op_id, "source_occupancy": gemmi.cif.as_number(occupancy),
                          "assembly": assembly, "disorder_group": group})
    return atoms, cell


def organic_graph(atoms, cell):
    xyz = np.array([a["frac"] for a in atoms]) @ cell
    tree = cKDTree(xyz % np.diag(cell), boxsize=np.diag(cell))
    adjacency = [set() for _ in atoms]
    for i, j in sorted(tree.query_pairs(1.85)):
        a, b = atoms[i], atoms[j]
        if "Zr" in (a["element"], b["element"]) or a["element"] == b["element"] == "O":
            continue
        cutoff = 1.65 if "O" in (a["element"], b["element"]) else 1.85
        if np.linalg.norm(delta(a["frac"], b["frac"], cell)) < cutoff:
            adjacency[i].add(j)
            adjacency[j].add(i)
    seen, components = set(), []
    for i in range(len(atoms)):
        if i in seen:
            continue
        todo, component = [i], []
        seen.add(i)
        while todo:
            j = todo.pop()
            component.append(j)
            for k in sorted(adjacency[j] - seen):
                seen.add(k)
                todo.append(k)
        components.append(sorted(component))
    return adjacency, components


def composition(atoms, ids):
    return Counter(atoms[i]["element"] for i in ids)


def prepare_fragment(atoms, cell, adj, ids, component_id, fixed_labels):
    """Rebuild valence/H and regularize organics with fixed metal-bound O atoms.

    MMFF94s is used only for isolated organic-fragment geometry repair. Its
    energies/charges are NOT a Zr-MOF or PFAS-in-water interaction model.
    """
    index = {v: k for k, v in enumerate(ids)}
    rw = Chem.RWMol()
    for i in ids:
        rw.AddAtom(Chem.Atom(atoms[i]["element"]))
    for i in ids:
        for j in sorted(adj[i]):
            if j > i:
                rw.AddBond(index[i], index[j], Chem.BondType.SINGLE)
    mol = rw.GetMol()
    for ring in Chem.GetSymmSSSR(mol):
        if len(ring) == 6:
            for i in ring:
                mol.GetAtomWithIdx(i).SetIsAromatic(True)
            for k, i in enumerate(ring):
                mol.GetBondBetweenAtoms(i, ring[(k + 1) % 6]).SetBondType(Chem.BondType.AROMATIC)
    for i in ids:
        oxygens = sorted(j for j in adj[i] if atoms[j]["element"] == "O")
        if atoms[i]["element"] == "C" and len(oxygens) == 2:
            # Localized Lewis form only; the two O atoms form a resonance pair.
            mol.GetBondBetweenAtoms(index[i], index[oxygens[0]]).SetBondType(Chem.BondType.DOUBLE)
            mol.GetAtomWithIdx(index[oxygens[1]]).SetFormalCharge(-1)
    Chem.SanitizeMol(mol)
    positions = {ids[0]: atoms[ids[0]]["frac"] @ cell}
    todo = [ids[0]]
    while todo:
        i = todo.pop()
        for j in sorted(adj[i]):
            if j not in positions:
                positions[j] = positions[i] + delta(atoms[i]["frac"], atoms[j]["frac"], cell)
                todo.append(j)
    conformer = Chem.Conformer(len(ids))
    for k, i in enumerate(ids):
        conformer.SetAtomPosition(k, positions[i])
    mol.AddConformer(conformer)
    mol = Chem.AddHs(mol, addCoords=True)
    if not AllChem.MMFFHasAllMoleculeParams(mol):
        raise ValueError(f"Organic fragment lacks MMFF94s parameters: {component_id}")
    props = AllChem.MMFFGetMoleculeProperties(mol, mmffVariant="MMFF94s")
    ff = AllChem.MMFFGetMoleculeForceField(mol, props)
    fixed = []
    for k, i in enumerate(ids):
        if atoms[i]["source_label"] in fixed_labels:
            ff.AddFixedPoint(k)
            fixed.append(k)
        else:
            ff.MMFFAddPositionConstraint(k, 0.35, 10.0)
    status = ff.Minimize(maxIts=5000)
    if status != 0:
        raise ValueError(f"Organic geometry regularization did not converge: {component_id}")
    xyz = np.asarray(mol.GetConformer().GetPositions())
    inv = np.linalg.inv(cell)
    result = []
    for k, atom in enumerate(mol.GetAtoms()):
        source = dict(atoms[ids[k]]) if k < len(ids) else {"source_label": "generated-H"}
        source.update(element=atom.GetSymbol(), frac=(xyz[k] @ inv) % 1,
                      component=component_id, role="linker" if len(ids) == 54 else "acetate")
        result.append(source)
    bonds = [(b.GetBeginAtomIdx(), b.GetEndAtomIdx(), str(b.GetBondType())) for b in mol.GetBonds()]
    report = {"component": component_id, "formula": rdMolDescriptors.CalcMolFormula(mol),
              "formal_charge": Chem.GetFormalCharge(mol), "fixed_metal_bound_oxygens": len(fixed),
              "max_heavy_displacement_angstrom": float(max(np.linalg.norm(xyz[k] - positions[i]) for k, i in enumerate(ids))),
              "organic_mmff_converged": True}
    return result, bonds, report


def add_hydrogens_on_oxygen(atoms, cell, oxygen, count):
    """Place seed H directions away from Zr; rotate water to avoid close contacts."""
    o = atoms[oxygen]["frac"]
    zrs = [a for a in atoms if a["element"] == "Zr" and np.linalg.norm(delta(o, a["frac"], cell)) < 2.7]
    if not zrs:
        raise ValueError("Node oxygen has no nearby Zr")
    outward = norm(-sum(norm(delta(o, a["frac"], cell)) for a in zrs))
    reference = np.eye(3)[int(np.argmin(np.abs(outward)))]
    v = norm(np.cross(outward, reference))
    w = np.cross(outward, v)
    if count == 1:
        candidates = [np.array([outward])]
    else:
        angle = np.deg2rad(104.52 / 2)
        candidates = [np.array([np.cos(angle)*outward + s*np.sin(angle)*(np.cos(t)*v+np.sin(t)*w)
                                for s in (-1, 1)]) for t in np.linspace(0, 2*np.pi, 24, endpoint=False)]
    others = [i for i in range(len(atoms)) if i != oxygen]
    other_xyz = np.array([delta(o, atoms[i]["frac"], cell) for i in others])
    # Distance normalized by a conservative seed-contact threshold.
    thresholds = np.array([1.05 if atoms[i]["element"] == "H" else 1.55 for i in others])
    def score(directions):
        d = np.linalg.norm(0.97*directions[:, None, :] - other_xyz[None, :, :], axis=2)
        return float(np.min(d / thresholds))
    directions = max(candidates, key=score)
    bonds = []
    for direction in directions:
        h = len(atoms)
        atoms.append({"element":"H", "frac": (o + 0.97*direction @ np.linalg.inv(cell)) % 1,
                      "source_label":"generated-H", "role":atoms[oxygen]["role"],
                      "component":atoms[oxygen]["component"]})
        bonds.append((oxygen, h, "SINGLE"))
    return bonds


def audit(atoms, bonds, cell, ledger):
    positions = np.array([a["frac"] for a in atoms])
    if not np.isfinite(positions).all():
        raise ValueError("Nonfinite coordinates")
    edges = {tuple(sorted((i,j))) for i,j,_ in bonds}
    tree = cKDTree((positions @ cell) % np.diag(cell), boxsize=np.diag(cell))
    contacts = []
    for i,j in sorted(tree.query_pairs(1.65)):
        if (i,j) in edges:
            continue
        distance = float(np.linalg.norm(delta(positions[i],positions[j],cell)))
        pair = (atoms[i]["element"], atoms[j]["element"])
        cutoff = 0.75 if pair == ("H","H") else 1.15 if "H" in pair else 1.65
        if distance < cutoff:
            contacts.append({"atoms":[i,j],"elements":pair,"distance_angstrom":distance})
    if contacts:
        raise ValueError(f"Severe nonbonded contacts: {contacts[:8]}")
    zrs = [i for i,a in enumerate(atoms) if a["element"] == "Zr"]
    coordination = []
    for i in zrs:
        coordination.append(int(sum(a["element"] == "O" and np.linalg.norm(delta(positions[i],a["frac"],cell)) < 2.65 for a in atoms)))
    if min(coordination) < 7 or max(coordination) > 9:
        raise ValueError(f"Unexpected Zr-O coordination: {Counter(coordination)}")
    h_degree = Counter(i for b in bonds for i in b[:2])
    if any(h_degree[i] != 1 for i,a in enumerate(atoms) if a["element"] == "H"):
        raise ValueError("Unbonded/multiply bonded H")
    bond_ranges = {}
    for i,j,order in bonds:
        pair = "-".join(sorted((atoms[i]["element"], atoms[j]["element"])))
        d = float(np.linalg.norm(delta(positions[i],positions[j],cell)))
        key = pair + ":" + order
        bond_ranges.setdefault(key, []).append(d)
        lo,hi = (0.85,1.25) if "H" in pair.split("-") else (1.15,1.7)
        if not lo <= d <= hi:
            raise ValueError(f"Bad covalent bond {i}-{j}: {d}")
    charge = sum(e["count"]*e["charge_each"] for e in ledger)
    if charge != 0:
        raise ValueError(f"Nonneutral framework formal-charge ledger: {charge}")
    actual = Counter({"Zr(IV)": len(zrs)})
    role_species = {"core_oxide":"mu3-oxide", "core_hydroxide":"mu3-hydroxide",
                    "terminal_hydroxide":"terminal hydroxide", "coordinated_water":"coordinated water"}
    for a in atoms:
        if a["element"] == "O" and a["role"] in role_species:
            actual[role_species[a["role"]]] += 1
    for role,species in (("linker","PTA4-"),("acetate","acetate")):
        actual[species] = len({a["component"] for a in atoms if a["role"] == role})
    if any(actual[e["species"]] != e["count"] for e in ledger):
        raise ValueError("Charge-ledger species counts disagree with the output structure")
    # Audit periodic connectivity with integer image shifts. Winding cycles must
    # span all three lattice directions, not merely connect one finite cluster.
    adjacency = [[] for _ in atoms]
    links = [(i,j) for i,j,_ in bonds]
    for i in zrs:
        links += [(i,j) for j,a in enumerate(atoms) if a["element"] == "O"
                  and np.linalg.norm(delta(positions[i],a["frac"],cell)) < 2.65]
    for i,j in links:
        shift = -np.rint(positions[j]-positions[i]).astype(int)
        adjacency[i].append((j,shift)); adjacency[j].append((i,-shift))
    images={0:np.zeros(3,dtype=int)};todo=[0];cycles=[]
    while todo:
        i=todo.pop()
        for j,shift in adjacency[i]:
            image=images[i]+shift
            if j in images:
                cycles.append(image-images[j])
            else:
                images[j]=image;todo.append(j)
    rank=int(np.linalg.matrix_rank(np.asarray(cycles))) if cycles else 0
    if len(images)!=len(atoms) or rank!=3:
        raise ValueError(f"Framework not one three-dimensional periodic network: {len(images)}/{len(atoms)}, rank={rank}")
    return {"atom_count":len(atoms), "composition":dict(sorted(Counter(a["element"] for a in atoms).items())),
            "formal_charge_e":charge,"severe_nonbonded_contacts":0,
            "zr_o_coordination_cutoff_angstrom":2.65,"zr_o_coordination_histogram":dict(sorted(Counter(coordination).items())),
            "hydrogens_have_one_covalent_bond":True,
            "formal_charge_ledger_matches_structure":True,"periodic_connectivity_rank":rank,
            "bond_length_ranges_angstrom":{k:[min(v),max(v)] for k,v in sorted(bond_ranges.items())}}


def write_model(name, atoms, bonds, cell, ledger, fragments, assumptions):
    report = audit(atoms,bonds,cell,ledger)
    doc = gemmi.cif.Document()
    b = doc.add_new_block(name.replace("-","_"))
    for axis,length in zip("abc",np.diag(cell)):
        b.set_pair("_cell_length_"+axis, f"{length:.8f}")
    for angle in ("alpha","beta","gamma"):
        b.set_pair("_cell_angle_"+angle,"90")
    b.set_pair("_space_group_name_H-M_alt", "'P 1'")
    b.set_pair("_space_group_IT_number","1")
    sym=b.init_loop("_space_group_symop_",["operation_xyz"])
    sym.add_row(["'x,y,z'"])
    loop=b.init_loop("_atom_site_",["label","type_symbol","fract_x","fract_y","fract_z","occupancy"])
    for i,a in enumerate(atoms):
        loop.add_row([a["element"]+str(i+1),a["element"],*[f"{v:.10f}" for v in a["frac"]],"1"])
    path=OUT/(name+".cif")
    doc.write_file(str(path))
    # Independent reader verifies P1 count/composition and explicit occupancy.
    from ase.io import read
    reread=read(str(path),format="cif")
    if len(reread)!=len(atoms) or Counter(reread.get_chemical_symbols()) != Counter(a["element"] for a in atoms):
        raise ValueError("ASE round-trip changed composition")
    sites=[]
    for i,a in enumerate(atoms):
        sites.append({"index":i,"cif_label":a["element"]+str(i+1),**{k:v for k,v in a.items() if k!="frac"}})
    dump(OUT/(name+".sites.json"), {"indexing":"zero-based; matches CIF atom order", "partial_charges_assigned":False,
         "sites":sites,"covalent_bonds":[list(x) for x in bonds],"formal_charge_ledger":ledger})
    report.update(name=name,file=path.name,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                  organic_fragments=fragments,formal_charge_ledger=ledger,assumptions=assumptions,
                  ready_for_geometry_relaxation=True,ready_for_capacity_simulation=False,
                  pending=["periodic Zr-framework relaxation","aqueous node-speciation validation",
                           "validated force field and partial charges","bulk water/counterion packing and equilibration"])
    return report


def assemble(raw,cell,prepared_fragments,core_h_label,water_ids=(),hydrolyze=False):
    atoms,bonds,fragment_reports=[],[],[]
    core_ids=[i for i,a in enumerate(raw) if a["element"]=="Zr" or a["source_label"] in ("O5","O6","O003","O004")]
    for i in core_ids:
        a=dict(raw[i]);a.update(component="inorganic_nodes",role="zirconium" if a["element"]=="Zr" else "core_hydroxide" if a["source_label"]==core_h_label else "core_oxide")
        atoms.append(a)
    for fragment,edges,report in prepared_fragments:
        if hydrolyze and fragment[0]["role"]=="acetate":
            oxygens=[dict(a) for a in fragment if a["element"]=="O"]
            assert len(oxygens)==2
            for a,role in zip(oxygens,("terminal_hydroxide","coordinated_water")):
                a.update(role=role,component=report["component"]+"_replacement")
                atoms.append(a)
            continue
        offset=len(atoms);atoms.extend(dict(a) for a in fragment)
        bonds.extend((i+offset,j+offset,k) for i,j,k in edges);fragment_reports.append(report)
    for i in water_ids:
        a=dict(raw[i]);a.update(role="coordinated_water",component="terminal_water")
        atoms.append(a)
    # Limit iteration to O atoms existing before appending hydrogens.
    for i in range(len(atoms)):
        role=atoms[i]["role"]
        if role in ("core_hydroxide","terminal_hydroxide","coordinated_water"):
            bonds.extend(add_hydrogens_on_oxygen(atoms,cell,i,2 if role=="coordinated_water" else 1))
    return atoms,bonds,fragment_reports


def main():
    OUT.mkdir(exist_ok=True)
    reports=[]
    common=["Zr(IV), four mu3-oxide and four mu3-hydroxide per Zr6 node",
            "Core hydroxides assigned to the four longer-Zr-O faces (O5 in PTA1; O004 in PTA2)",
            "Pyridyl N remains neutral; framework PTA is fully deprotonated (4-)",
            "Organic geometry repaired with constrained isolated-fragment MMFF94s; metal-bound O fixed",
            "No metal potential, partial charges, bulk solvent or pH-dependent equilibrium assigned"]
    for source_name in ("Zr-PTA1","Zr-PTA2"):
        raw,cell=read_raw(source_name)
        # A2 waters overlap A1 linker O3; keep them separate during graph construction.
        water_ids=[i for i,a in enumerate(raw) if a["source_label"]=="O3A"]
        graph_atoms=[dict(a) for a in raw if a["source_label"]!="O3A"]
        adj,components=organic_graph(graph_atoms,cell)
        organic=[c for c in components if any(graph_atoms[i]["element"]=="C" for i in c)]
        for comp in organic:
            if composition(graph_atoms,comp) not in (Counter(C=44,N=2,O=8),Counter(C=2,O=2)):
                raise ValueError("Unexpected organic component")
        labels={"O1","O2","O3"} if source_name=="Zr-PTA1" else {"O005","O00F","O1"}
        jobs=[(graph_atoms,cell,adj,c,f"organic_{i:02}",labels) for i,c in enumerate(organic)]
        print(source_name,"regularizing",len(jobs),"organic fragments",flush=True)
        with ThreadPoolExecutor(max_workers=4) as pool:
            fragments=list(pool.map(lambda args:prepare_fragment(*args),jobs))
        if source_name=="Zr-PTA1":
            extra=[i for i,c in enumerate(organic) if any(graph_atoms[j]["source_label"]=="C13" for j in c)]
            fixed=[i for i in range(len(organic)) if i not in extra]
            zrs=[j for j,a in enumerate(graph_atoms) if a["source_label"]=="Zr2"]
            contacts=np.zeros((len(extra),len(zrs)),dtype=int)
            for k,i in enumerate(extra):
                for j in organic[i]:
                    if graph_atoms[j]["source_label"]=="O3":
                        for q,zr in enumerate(zrs):
                            if np.linalg.norm(delta(graph_atoms[j]["frac"],graph_atoms[zr]["frac"],cell))<2.6:
                                contacts[k,q]+=1
            valid=[c for c in itertools.combinations(range(len(extra)),4) if np.all(contacts[list(c)].sum(axis=0)==1)]
            if len(valid)!=16:
                raise ValueError("Unexpected linker-ordering enumeration")
            first=valid[0];second=tuple(i for i in range(len(extra)) if i not in first)
            if second not in valid:
                raise ValueError("No complementary ordering")
            for suffix,chosen in zip(("A","B"),(first,second)):
                selected=fixed+[extra[i] for i in chosen]
                selected_o=[a for i in selected for a in fragments[i][0] if a["source_label"]=="O3"]
                # Complementary A2 water occupies only the unselected A1 positions.
                waters=[i for i in water_ids if all(np.linalg.norm(delta(raw[i]["frac"],a["frac"],cell))>1 for a in selected_o)]
                if len(waters)!=16:
                    raise ValueError("Expected four terminal waters per Zr6")
                atoms,bonds,fr=assemble(raw,cell,[fragments[i] for i in selected],"O5",waters)
                ledger=[{"species":"Zr(IV)","count":24,"charge_each":4},
                        {"species":"mu3-oxide","count":16,"charge_each":-2},
                        {"species":"mu3-hydroxide","count":16,"charge_each":-1},
                        {"species":"PTA4-","count":12,"charge_each":-4},
                        {"species":"coordinated water","count":16,"charge_each":0}]
                assumptions=common+[f"Paper-stoichiometry hypothesis; whole extra-linker choices {chosen} of 8 expanded candidates",
                    "16 unit-cell orderings satisfy local site constraints; A/B are two complementary representatives, not an exhaustive ensemble",
                    "N2 is coupled to its selected half-occupied whole linker despite source N2 occupancy=1",
                    "This yields N6 per Zr6, whereas the published empirical formula implies N8; source composition needs clarification",
                    "Monodentate extra PTA is assumed fully deprotonated to match the paper formula; neutral H4PTA/hydroxide alternatives require a separate model"]
                reports.append(write_model(f"Zr-PTA1-paper-neutral-{suffix}",atoms,bonds,cell,ledger,fr,assumptions))
        else:
            for hydrolyze in (False,True):
                atoms,bonds,fr=assemble(raw,cell,fragments,"O004",hydrolyze=hydrolyze)
                ledger=[{"species":"Zr(IV)","count":24,"charge_each":4},
                        {"species":"mu3-oxide","count":16,"charge_each":-2},
                        {"species":"mu3-hydroxide","count":16,"charge_each":-1},
                        {"species":"PTA4-","count":8,"charge_each":-4},
                        {"species":"terminal hydroxide" if hydrolyze else "acetate","count":16,"charge_each":-1}]
                if hydrolyze:
                    ledger.append({"species":"coordinated water","count":16,"charge_each":0})
                assumptions=common+["Deposited acetates retained; no naked open Zr site implied" if not hydrolyze else
                    "Sensitivity hypothesis: each acetate is replaced by one OH- and one H2O at its original two O positions; not experimentally established speciation"]
                reports.append(write_model("Zr-PTA2-"+("hydrolyzed" if hydrolyze else "acetate")+"-neutral",atoms,bonds,cell,ledger,fr,assumptions))
    species={"PFOA":"O=C([O-])C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)F",
             "PFOS":"O=S(=O)([O-])C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)F",
             "PFBA":"O=C([O-])C(F)(F)C(F)(F)C(F)(F)F",
             "PFBS":"O=S(=O)([O-])C(F)(F)C(F)(F)C(F)(F)C(F)(F)F"}
    molecules=[]
    for name,smiles in species.items():
        m=Chem.AddHs(Chem.MolFromSmiles(smiles))
        params=AllChem.ETKDGv3();params.randomSeed=20261030
        if AllChem.EmbedMolecule(m,params)!=0:
            raise ValueError("PFAS embedding failed")
        if AllChem.MMFFOptimizeMolecule(m,maxIters=3000)!=0:
            raise ValueError("PFAS seed geometry failed to converge")
        if Chem.GetFormalCharge(m)!=-1:
            raise ValueError("Expected singly charged PFAS anion")
        m.SetProp("_Name",name+" anion; unparameterized seed")
        with Chem.SDWriter(str(OUT/(name+"-anion.sdf"))) as writer:
            writer.write(m)
        molecules.append({"name":name,"file":name+"-anion.sdf","smiles":smiles,
            "formula":rdMolDescriptors.CalcMolFormula(m),"formal_charge_e":-1,
            "counterion_rule":"one Na+ per PFAS anion; user-selected pH-7 target with no added background salt",
            "partial_charges_assigned":False})
    dump(OUT/"audit.json",{"schema_version":1,"models":reports,"pfas_templates":molecules,
        "versions":{"gemmi":gemmi.__version__,"numpy":np.__version__,"rdkit":rdBase.rdkitVersion},
        "source_hashes":EXPECTED,"capacity_simulations_run":False})
    print(json.dumps([{k:r[k] for k in ("name","atom_count","composition","formal_charge_e","zr_o_coordination_histogram")} for r in reports],indent=2),flush=True)


def solvate_seeds():
    """Pack bounded, reproducible one-anion cells for subsequent equilibration.

    Bulk water packing is geometric, not an adsorption prediction. A nominal
    one-anion concentration in these small cells is deliberately reported.
    """
    from scipy.spatial.transform import Rotation
    from ase import Atoms
    from ase.io import write, read
    data=json.loads((OUT/"audit.json").read_text())
    summary=OUT/"aqueous-seeds.json"
    if summary.exists():
        summary.unlink()  # A failed new preparation must not leave a stale success manifest.
    all_reports=[]
    radii={"C":1.70,"N":1.55,"O":1.52,"F":1.47,"S":1.80,"Zr":2.10,"H":1.20}
    for model_id,model in enumerate(data["models"]):
        framework=read(str(OUT/model["file"]))
        cell=np.asarray(framework.cell);lengths=np.diag(cell);inv=np.linalg.inv(cell)
        fxyz=framework.get_positions()%lengths
        fels=framework.get_chemical_symbols()
        ftree=cKDTree(fxyz,boxsize=lengths)
        pairs=("PFOA","PFOS") if model["name"].startswith("Zr-PTA1") else ("PFBA","PFBS")
        site_data=json.loads((OUT/(model["name"]+".sites.json")).read_text())
        for species_id,pfas_name in enumerate(pairs):
            seed=20261030+10*model_id+species_id
            rng=np.random.default_rng(seed)
            mol=Chem.SDMolSupplier(str(OUT/(pfas_name+"-anion.sdf")),removeHs=False)[0]
            pxyz=np.array(mol.GetConformer().GetPositions());pxyz-=pxyz.mean(axis=0)
            pels=[a.GetSymbol() for a in mol.GetAtoms()]
            # Find one pore position without assuming that it is an adsorption site.
            for attempt in range(20000):
                candidate=(pxyz@Rotation.random(random_state=rng).as_matrix().T+rng.random(3)*lengths)%lengths
                ok=True
                for q,el in zip(candidate,pels):
                    for j in ftree.query_ball_point(q,3.6):
                        dist=np.linalg.norm(delta(q@inv,fxyz[j]@inv,cell))
                        cutoff=1.8 if fels[j]=="H" else .86*(radii[el]+radii[fels[j]])
                        if dist<cutoff:
                            ok=False;break
                    if not ok:break
                if ok:break
            else:raise ValueError("Could not place PFAS without framework overlaps")
            pxyz=candidate
            fixed_xyz=np.vstack((fxyz,pxyz));fixed_els=fels+pels
            solute_tree=cKDTree(fixed_xyz,boxsize=lengths)
            water_cutoffs={"C":3.0,"N":2.8,"O":2.6,"F":2.6,"S":3.0,"Zr":3.1,"H":1.7,"Na":2.3}
            def water_site_clear(q):
                return all(np.linalg.norm(delta(q@inv,fixed_xyz[j]@inv,cell))>=water_cutoffs[fixed_els[j]]
                           for j in solute_tree.query_ball_point(q,3.6))
            # No added electrolyte; one Na+ compensates one singly charged PFAS.
            for ion_attempt in range(20000):
                na=rng.random(3)*lengths
                ds=np.linalg.norm((fixed_xyz-na-lengths*np.rint((fixed_xyz-na)/lengths)),axis=1)
                thresholds=np.array([1.9 if e=="H" else 2.5 if e=="O" else 3.0 for e in fixed_els])
                if not (np.all(ds>thresholds) and np.min(ds[len(fels):])>4.0):continue
                # Seed six first-shell O atoms at 2.45 A. This geometric shell is
                # not a prediction of equilibrium coordination or residence time.
                shell=(na+2.45*np.vstack((np.eye(3),-np.eye(3)))@Rotation.random(random_state=rng).as_matrix().T)%lengths
                if all(water_site_clear(q) for q in shell):break
            else:raise ValueError("Could not place Na+ without overlaps")
            fixed_xyz=np.vstack((fixed_xyz,na));fixed_els+= ["Na"]
            fixed_tree=cKDTree(fixed_xyz,boxsize=lengths)
            # Choose a commensurate oxygen grid close to 0.0334 molecules/A^3
            # before excluded-volume filtering. This is a seed, not equilibrated density.
            target=.0334*np.linalg.det(cell)
            choices=[]
            for counts in itertools.product(*(range(max(1,int(l/3.1)-1),int(l/3.1)+3) for l in lengths)):
                spacing=lengths/np.array(counts)
                if min(spacing)<2.8:continue
                score=abs(np.prod(counts)/target-1)+.001*np.std(spacing)
                choices.append((score,counts))
            counts=np.array(min(choices)[1]);offset=rng.random(3)
            water_o=list(shell)
            for grid in itertools.product(*(range(int(n)) for n in counts)):
                q=((np.array(grid)+offset)/counts*lengths)%lengths
                neighbors=fixed_tree.query_ball_point(q,3.6)
                near_shell=min(np.linalg.norm(delta(q@inv,o@inv,cell)) for o in shell)
                if near_shell>=2.8 and all(np.linalg.norm(delta(q@inv,fixed_xyz[j]@inv,cell))>=water_cutoffs[fixed_els[j]] for j in neighbors):
                    water_o.append(q)
            if not water_o:raise ValueError("No solvent waters packed")
            water_o=np.array(water_o)
            environment=np.vstack((fixed_xyz,water_o))
            env_els=fixed_els+["O"]*len(water_o)
            env_tree=cKDTree(environment,boxsize=lengths)
            hprevious=[];waters=[]
            theta=np.deg2rad(104.52/2)
            template=.9572*np.array([[np.cos(theta),np.sin(theta),0],[np.cos(theta),-np.sin(theta),0]])
            for wi,o in enumerate(water_o):
                neighbors=[j for j in env_tree.query_ball_point(o,3.4) if j!=len(fixed_xyz)+wi]
                nearby=environment[neighbors]-o
                nearby-=lengths*np.rint(nearby/lengths)
                threshold=np.array([1.0 if env_els[j]=="H" else 1.85 if env_els[j] in ("Zr","Na") else 1.4 if env_els[j] in ("O","F","N") else 1.65 for j in neighbors])
                best_score,best_h=-1,None
                for rotation in Rotation.random(48,random_state=rng).as_matrix():
                    h=template@rotation.T
                    score=float(np.min(np.linalg.norm(h[:,None,:]-nearby[None,:,:],axis=2)/threshold)) if neighbors else 10.0
                    if hprevious:
                        ph=np.asarray(hprevious)-o;ph-=lengths*np.rint(ph/lengths)
                        score=min(score,float(np.min(np.linalg.norm(h[:,None,:]-ph[None,:,:],axis=2)/.85)))
                    if score>best_score:best_score,best_h=score,h
                if best_score<1:
                    # Discard the water when no orientation meets the seed-contact limits.
                    continue
                hs=(o+best_h)%lengths;hprevious.extend(hs);waters.append((o,hs))
            symbols=list(fixed_els);xyz=list(fixed_xyz)
            bonds=[list(b) for b in site_data["covalent_bonds"]]
            offset_pfas=len(fels)
            bonds.extend([b.GetBeginAtomIdx()+offset_pfas,b.GetEndAtomIdx()+offset_pfas,str(b.GetBondType())] for b in mol.GetBonds())
            for o,hs in waters:
                i=len(symbols);symbols.extend(("O","H","H"));xyz.extend((o,*hs));bonds.extend(([i,i+1,"SINGLE"],[i,i+2,"SINGLE"]))
            hydration_count=int(sum(np.linalg.norm(delta(na@inv,o@inv,cell))<3.4 for o,hs in waters))
            if hydration_count<6:
                raise ValueError("Na hydration shell lost water during orientation")
            xyz=np.array(xyz)%lengths
            # Check actual positions, including periodic boundary contacts and covalent bonds.
            bonded={tuple(sorted(b[:2])) for b in bonds}
            tree=cKDTree(xyz,boxsize=lengths);minimum=100.0
            for i,j in sorted(tree.query_pairs(1.65)):
                if (i,j) in bonded:continue
                d=float(np.linalg.norm(delta(xyz[i]@inv,xyz[j]@inv,cell)))
                minimum=min(minimum,d)
                cutoff=.75 if symbols[i]==symbols[j]=="H" else 1.15 if "H" in (symbols[i],symbols[j]) else 1.65
                if d<cutoff:raise ValueError(f"Packed seed contains a severe overlap: {i},{j},{d}")
            for i,j,_ in bonds:
                distance=np.linalg.norm(delta(xyz[i]@inv,xyz[j]@inv,cell))
                if not (.85 if "H" in (symbols[i],symbols[j]) else 1.15)<=distance<=(1.25 if "H" in (symbols[i],symbols[j]) else 1.85):
                    raise ValueError("Packed seed has a broken covalent bond")
            name=model["name"]+"-"+pfas_name+"-aqueous-seed"
            path=OUT/(name+".cif")
            write(str(path),Atoms(symbols,positions=xyz,cell=cell,pbc=True),format="cif")
            back=read(str(path))
            if len(back)!=len(symbols) or back.get_chemical_symbols()!=symbols:
                raise ValueError("Solvated CIF round-trip mismatch")
            shift=back.positions-xyz;shift-=lengths*np.rint(shift/lengths)
            if np.max(np.abs(shift))>1e-5:
                raise ValueError("Solvated CIF round-trip changed atom ordering/positions")
            # Group ranges and bond connectivity are explicit; CIF alone has no topology/charge model.
            groups={"framework":[0,len(fels)],"PFAS_anion":[len(fels),len(fels)+len(pels)],
                    "sodium_counterion":[len(fels)+len(pels),len(fixed_els)],"bulk_water":[len(fixed_els),len(symbols)]}
            topology={"ranges_are":"zero-based, half-open; match CIF order","groups":groups,
                      "framework_topology_file":model["name"]+".sites.json","covalent_bonds":bonds,
                      "formal_charge_e":{"framework":0,"PFAS":-1,"Na":1,"water":0,"total":0},
                      "force_field_assigned":False,"partial_charges_assigned":False}
            dump(OUT/(name+".topology.json"),topology)
            report={"name":name,"file":path.name,"sha256":hashlib.sha256(path.read_bytes()).hexdigest(),
                    "framework_file":model["file"],"pfas":pfas_name,"pfas_count":1,"na_count":1,
                    "bulk_water_count":len(waters),"atom_count":len(symbols),"formal_charge_e":0,
                    "initial_na_water_oxygen_neighbors_within_3p4_angstrom":hydration_count,
                    "target_temperature_k":298.15,"target_pH":7,"added_salt":None,
                    "pH_implemented_as":"assumed fixed protonation; no constant-pH equilibration",
                    "packing_seed":seed,"water_oxygen_grid_counts":counts.tolist(),
                    "nominal_total_pfas_cell_concentration_molar":float(1/(6.02214076e23*np.linalg.det(cell)*1e-27)),
                    "concentration_note":"Counts divided by whole cell volume; NOT free aqueous concentration or matched experimental loading",
                    "severe_overlap_count":0,"covalent_bonds_intact":True,"cif_round_trip_passed":True,
                    "ready_for_equilibrated_production":False,
                    "pending":["select and validate MOF/PFAS/water/Na interaction parameters and partial charges",
                               "relax the periodic framework and solvent","equilibrate solvent density and node hydration",
                               "establish chemical potential/concentration and finite-size convergence"]}
            all_reports.append(report)
            print(name,"waters",len(waters),"atoms",len(symbols),flush=True)
    dump(summary,{"schema_version":1,"models":all_reports,
        "salt_free_means":"no background electrolyte; charge-compensating Na+ is present",
        "capacity_simulations_run":False})


if __name__=="__main__":
    import sys
    if sys.argv[1:]==["--solvate"]:
        solvate_seeds()
    elif not sys.argv[1:]:
        main()
    else:
        raise SystemExit("usage: prepare_pfas_structures.py [--solvate]")

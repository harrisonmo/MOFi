# Ordered framework and aqueous PFAS starting models

These files are **charge-balanced geometry-relaxation candidates**, derived from
the experimental CIFs for DOI [10.1002/anie.1567631](https://doi.org/10.1002/anie.1567631).
They are not equilibrated aqueous structures or capacity predictions.

## User-selected solution scenario

- Target temperature: 298.15 K (25 C).
- Target pH: 7, represented by assumed fixed protonation states. No constant-pH
  method or acid-base equilibrium has been run.
- PFAS are singly charged anions; each starting cell contains one PFAS and one
  Na+ counterion, plus explicitly packed water.
- No background electrolyte is added. "Salt-free" still includes the Na+
  needed for charge compensation.

## Framework candidates

All files are fully expanded P1 CIFs with occupancy 1. Each cell contains four
Zr6 nodes. Companion `.sites.json` files retain source labels, symmetry-operation
indices, molecule identities, covalent bonds and a formal-charge ledger.

| File stem | Atoms | Composition per Zr6 node | Role |
| --- | ---: | --- | --- |
| [Zr-PTA1-paper-neutral-A](Zr-PTA1-paper-neutral-A.cif) | 1056 | C132H84N6O36Zr6 | Extra-linker ordering A |
| [Zr-PTA1-paper-neutral-B](Zr-PTA1-paper-neutral-B.cif) | 1056 | C132H84N6O36Zr6 | Complementary ordering B |
| [Zr-PTA2-acetate-neutral](Zr-PTA2-acetate-neutral.cif) | 808 | C96H64N4O32Zr6 | Deposited acetate ligands retained |
| [Zr-PTA2-hydrolyzed-neutral](Zr-PTA2-hydrolyzed-neutral.cif) | 776 | C88H64N4O32Zr6 | Acetate-to-OH/H2O replacement hypothesis |

The first two implement Zr6(O)4(OH)4(PTA)3(H2O)4, with PTA4-.
The retained-acetate model implements Zr6(O)4(OH)4(PTA)2(OAc)4.
The hydrolyzed model implements Zr6(O)4(OH)4(PTA)2(OH)4(H2O)4.
Each formula has net formal charge zero. Formal charge bookkeeping is not a
force-field partial-charge assignment.

## Preparation and checks

1. Verify the SHA-256 hashes of the original CIFs.
2. Parse crystallography with Gemmi. The bare `End` terminator in the first raw
   block is omitted in memory; the published source file stays unchanged.
3. Expand the published symmetry operations, deduplicate special positions,
   and choose the major B1 linker conformation for Zr-PTA1.
4. Reconstruct complete organic molecules through their covalent connectivity.
   Zr-PTA1 has eight candidate extra-linker molecules at half occupancy. Exactly
   four must be selected, coupled to the complementary water positions. Sixteen
   cell orderings satisfy one extra-linker O plus one water O at each equatorial
   Zr. A/B are two complementary representatives, not an exhaustive disorder
   ensemble or a thermodynamic ranking.
5. Rebuild organic hydrogens and valence using RDKit. Regularize each organic
   fragment with MMFF94s, holding its metal-bound oxygens fixed and weakly
   restraining other heavy atoms. This repairs distorted covalent geometries;
   it is **not a periodic Zr-MOF force field** or aqueous binding calculation.
6. Assign four core hydroxides to the longer-Zr-O faces per node (O5 in PTA1,
   O004 in PTA2). Remaining core O atoms are oxide. Construct O-H directions
   away from Zr; rotate bound-water hydrogens to reduce close contacts.
7. Require neutral charge ledgers that match actual species counts, intact
   covalent bonds, one covalent neighbor per H, no severe nonbonded overlaps,
   and one connected periodic network spanning all three lattice directions.
8. Read every generated framework CIF independently with ASE. The generated
   models have eight O neighbors per Zr using a 2.65 A cutoff. `audit.json`
   records counts, geometry ranges, displacements, checksums and assumptions.

## Chemical uncertainties carried forward

**PTA1 composition:** complete PTA molecules have C44N2O8 heavy atoms.
Twelve linkers per four-node cell give N24, or N6 per Zr6. The raw empirical
formula C66H42N4O18Zr3 instead implies N8 per node. The A-site N2 atom has
occupancy 1 while its surrounding linker atoms have occupancy 0.5. Our ordered
model couples N2 to the selected whole molecule, rather than retaining isolated
nitrogen atoms. This discrepancy needs clarification before comparison using
an experimental dry-mass denominator.

**PTA1 protonation:** the paper-formula candidate treats every extra linker as
PTA4-, with complementary sites occupied by water. A proton-transfer alternative
with neutral H4PTA and terminal OH- can also be charge-balanced. Related MFM-600
work discusses neutral disordered H4L linkers and hydroxide caps, but does not
establish the speciation of this PFAS sample:
[Carter et al., JACS 2018](https://doi.org/10.1021/jacs.8b08433).
The current candidate must therefore be checked against aqueous protonation
energetics; matching pH 7 is not proven by a neutral charge ledger.

**PTA2 sites:** the deposited model has four acetates per node and eight O
neighbors at every Zr. It does not contain naked open Zr sites. The hydrolyzed
candidate replaces each acetate with OH-/H2O at its original two oxygen sites;
this is an explicit sensitivity hypothesis, not measured acetate removal or an
equilibrium prediction. No released acetate species is included in that cell.

## Explicit aqueous seeds

`*-aqueous-seed.cif` contains a framework, one PFAS anion, one Na+ and water.
PTA1 A/B are paired with PFOA and PFOS; both PTA2 candidates are paired with
PFBA and PFBS. `aqueous-seeds.json` lists water counts, packing seeds,
checksums and initial cell concentrations. Companion `.topology.json` files
define molecule ranges and covalent bonds. Atom indexing is zero-based there.

PFAS/Na+ positions are chosen to avoid overlaps, without assuming an adsorption
site or bound ion pair. Each Na+ is seeded with six water oxygens at 2.45 A
in a rotated octahedral arrangement; this is an initial hydration geometry,
not a calculated equilibrium coordination number. The placement must fit
without solute overlaps, and all six waters must survive hydrogen orientation.
Additional water oxygen positions use a commensurate grid near
bulk-water number density **before excluded-volume filtering**. Hydrogens are
oriented to avoid contacts. The accepted water count is a packing outcome, not
an equilibrium hydration number or validated pore-water density.

One PFAS per crystallographic cell corresponds to about **0.059 M based on
whole-cell volume**. This is a finite starting configuration, not the free
aqueous concentration or a matched experimental feed. It must not be used to
infer capacity or fit an adsorption isotherm.

Every aqueous seed is net neutral and checked for intact covalent bonds,
periodic severe overlaps and CIF round-trip atom order/coordinates. No
interaction parameters, partial charges, timestep or production run are set.

## Reproduce

Use an isolated Python environment; the tested interpreter was Python 3.13.5.
The main MOFi application has its own Python/environment constraints.

```sh
python3 -m venv /tmp/mofi-pfas-prep
/tmp/mofi-pfas-prep/bin/python -m pip install -r data/pfas/anie.1567631/prepared/requirements.txt
/tmp/mofi-pfas-prep/bin/python scripts/prepare_pfas_structures.py
/tmp/mofi-pfas-prep/bin/python scripts/prepare_pfas_structures.py --solvate
```

Run from the MOFi root. Geometry checks execute during preparation and fail on
bad topology, charge counts, broken bonds or severe overlaps. Raw CIFs are never
rewritten. Gemmi, ASE, RDKit and dependency versions are pinned in the supplied
requirements file. Framework preparation and packing are separate commands.

## Before production

Select and validate Zr-node, PFAS, water and Na+ interaction parameters and
partial charges; relax the periodic models and check competing protonation/
ligand-exchange states; equilibrate solvent density and hydration; then establish
concentration/chemical-potential, loading and finite-size convergence. A fixed
nonreactive model will not itself describe carboxylate ligand exchange.

No PFAS capacity calculation has been run.

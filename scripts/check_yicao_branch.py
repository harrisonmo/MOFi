"""Independently verify saved adsorption results and non-MACE helpers."""
import argparse
import ast
import csv
import hashlib
import io
import json
import tarfile
import time
from collections import defaultdict
from pathlib import Path
import numpy as np
from scipy.stats import pearsonr, spearmanr
from ase import Atoms
from ase.io import write

parser = argparse.ArgumentParser()
parser.add_argument('--source', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=False)
started = time.perf_counter()
package = args.source/'benchmarks/mof_adsorption'
reports = []
for stem in ['matched_reference_compare','matched_reference_compare_host_cif_sensitivity','full_archive_unseen_last_frame_compare']:
    expected = json.loads((package/f'{stem}.json').read_text())
    with (package/f'{stem}.csv').open(encoding='utf-8',newline='') as f:
        rows = list(csv.DictReader(f))
    assert len(rows)==expected['counts']['evaluated_frames']
    groups = defaultdict(list)
    for row in rows:
        assert row['frame_name'].startswith(row['reference_mof_id']+'_w_')
        assert int(row['natoms_guest'])==3
        np.testing.assert_allclose(float(row['reference_eads_eV'])*96.48533212331002,float(row['reference_eads_kJ_mol']),atol=1e-10)
        groups[row['reference_mof_id']].append(row)
    assert len(groups)==expected['counts']['matched_mof_ids']
    metrics = {}
    for level in ['frame_metrics','mof_level_metrics']:
        metrics[level] = {}
        for arm, column in [('scratch','scratch_eads_eV'),('foundation_naive_finetune','foundation_eads_eV')]:
            if level=='frame_metrics':
                pred = np.array([float(x[column]) for x in rows])
                ref = np.array([float(x['reference_eads_eV']) for x in rows])
            else:
                for group in groups.values():
                    assert len({x['reference_eads_eV'] for x in group})==1
                pred = np.array([np.mean([float(x[column]) for x in group]) for group in groups.values()])
                ref = np.array([float(group[0]['reference_eads_eV']) for group in groups.values()])
            error = pred-ref
            values = {'count':len(pred),'mae_eV':float(abs(error).mean()),'rmse_eV':float(np.sqrt((error**2).mean())),
                      'bias_eV':float(error.mean()),'pearson_r':float(pearsonr(ref,pred).statistic),
                      'spearman_r':float(spearmanr(ref,pred).statistic)}
            for key, value in values.items():
                np.testing.assert_allclose(value,expected[level][arm][key],rtol=1e-10,atol=1e-10)
            metrics[level][arm]=values
    reports.append({'result':stem,'frames':len(rows),'mofs':len(groups),'metrics':metrics,
                    'csv_sha256':hashlib.sha256((package/f'{stem}.csv').read_bytes()).hexdigest()})

source_path = args.source/'scripts/mof_adsorption_model_decomposition.py'
tree = ast.parse(source_path.read_text(encoding='utf-8'))
# Preserve helper bodies; do not import MACE or execute the command-line entry.
tree.body = [node for node in tree.body if not (isinstance(node,ast.ImportFrom) and node.module=='mace.calculators')
             and not isinstance(node,ast.If)]
scope = {'__name__':'isolated_yicao_helpers'}
exec(compile(tree,str(source_path),'exec'),scope)
atoms = Atoms('ZnOCO2',positions=[[1,1,1],[2,1,1],[6,6,6],[5,6,6],[7,6,6]],cell=[12,12,12],pbc=True,tags=[0,0,1,1,1])
host, guest = scope['split_with_pymatgen'](atoms)
assert len(host)==2 and len(guest)==3
for part, mask in [(host,np.array([True,True,False,False,False])),(guest,np.array([False,False,True,True,True]))]:
    np.testing.assert_allclose(part.positions,atoms.positions[mask])
    np.testing.assert_array_equal(part.numbers,atoms.numbers[mask])
    np.testing.assert_allclose(part.cell.array,atoms.cell.array)
    np.testing.assert_array_equal(part.pbc,atoms.pbc)
key = scope['structure_key']
assert key(atoms)==key(atoms.copy())
for change in ['positions','cell','pbc','elements']:
    changed = atoms.copy()
    if change=='positions': changed.positions[0,0]+=.01
    elif change=='cell': changed.cell[0,0]+=.01
    elif change=='pbc': changed.pbc[0]=False
    else: changed.numbers[0]=6
    assert key(changed)!=key(atoms)
assert scope['match_reference']('ABC_w_CO2_1',{'ABC':{}})=='ABC'
assert scope['match_reference']('ABC_0.08_0_w_CO2_1',{'ABC':{}}) is None
atoms.info.update(name='ABC_w_CO2_1',nco2=1,nh2o=0)
buffer = io.StringIO()
write(buffer,[atoms,atoms],format='extxyz')
payload = buffer.getvalue().encode()
archive = args.output/'synthetic.tar.gz'
with tarfile.open(archive,'w:gz') as tar:
    member = tarfile.TarInfo('frames/synthetic.extxyz')
    member.size = len(payload)
    tar.addfile(member,io.BytesIO(payload))
frames = list(scope['iter_source_frames'](None,archive,set()))
assert len(frames)==2 and [x[1] for x in frames]==[0,1]
assert list(scope['iter_source_frames'](None,archive,{'synthetic.extxyz'}))==[]
seconds = time.perf_counter()-started
assert seconds<120
result = {'source_commit':'6a7dd43eeefac993df200fe278a2484b8395b49f','status':'offline_checks_passed_gpu_not_run',
          'seconds':seconds,'admission_tests':2,'saved_result_audits':reports,
          'helper_checks':['periodic_geometry_split','cache_copy_stability','cache_positions_cell_pbc_elements',
                           'exact_pristine_id','streamed_extxyz','source_exclusion'],
          'source_sha256':hashlib.sha256(source_path.read_bytes()).hexdigest(),
          'gpu_blockers':['Original ODAC frames/reference CSV and trained checkpoints unavailable locally',
                          'MACE not installed in local GNN environment','SSH alias kozo-dev-yi cannot be resolved'],
          'limits':['Saved prediction replay is not independent MACE inference',
                    'Scalar reference is not a matched DFT triplet for each frame',
                    'Frame-level non-overlap is not parent/family-disjoint admission']}
(args.output/'result.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))

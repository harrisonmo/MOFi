"""Verify public source downloads and reconstruct the recorded evaluated frames."""
import argparse
import csv
import hashlib
import io
import json
import tarfile
import time
from pathlib import Path, PurePosixPath
from ase.io import read

parser = argparse.ArgumentParser()
parser.add_argument('--source',type=Path,required=True)
args = parser.parse_args()
root = Path(__file__).resolve().parents[1]
raw = root/'datasets/raw/yicao'
out = root/'datasets/processed/yicao/evaluated-129'
out.mkdir(parents=True,exist_ok=False)
started = time.monotonic()
def guard():
    if time.monotonic()-started>600:
        raise TimeoutError('Source verification and reconstruction exceeded 600 seconds')
def hashes(path):
    md5, sha = hashlib.md5(),hashlib.sha256()
    with path.open('rb') as f:
        while chunk:=f.read(8*1024*1024):
            md5.update(chunk);sha.update(chunk);guard()
    return {'md5':md5.hexdigest(),'sha256':sha.hexdigest(),'bytes':path.stat().st_size}
manifest = json.loads((args.source/'benchmarks/mof_adsorption/source_manifest.json').read_text())
expected = next(x['files'] for x in manifest['sources'] if x['id']=='odac_ml_ip_scalar_corrections')
metadata = {}
for name, item in expected.items():
    metadata[name] = hashes(raw/'metadata'/name)
    assert metadata[name]['sha256']==item['sha256']
archive = raw/'extxyz_val.tar.gz'
archive_hash = hashes(archive)
assert archive_hash['md5']=='09913759c6e0f8d649f7ec9dff9e0e8b'
print('Verified public archive and all four reference files',flush=True)
with (args.source/'benchmarks/mof_adsorption/full_archive_unseen_last_frame_compare.csv').open(newline='',encoding='utf-8') as f:
    rows=list(csv.DictReader(f))
targets={PurePosixPath(row['source_file']).name.replace('__','/'):row for row in rows}
assert len(targets)==129
records=[]
with tarfile.open(archive,'r|gz') as tar:
    for member in tar:
        guard()
        name=member.name.removeprefix('./')
        if not member.isfile() or name not in targets:
            continue
        data=tar.extractfile(member).read().decode('utf-8')
        lines=data.splitlines(keepends=True)
        index=0
        last=None
        count=0
        while index<len(lines):
            if not lines[index].strip():
                index+=1
                continue
            n=int(lines[index]);stop=index+n+2
            assert stop<=len(lines)
            last=''.join(lines[index:stop]);index=stop;count+=1
        assert last is not None
        atoms=read(io.StringIO(last),format='extxyz')
        row=targets[name]
        assert str(atoms.info['name'])==row['frame_name']
        assert int(atoms.info['sid'])==int(row['sid'])
        assert int(atoms.info['fid'])==int(row['fid'])
        assert int(atoms.info['nco2'])==1 and int(atoms.info['nh2o'])==0
        assert len(atoms)==int(row['natoms_host'])+int(row['natoms_guest'])
        assert int((atoms.get_tags()>0).sum())==3
        destination=out/PurePosixPath(row['source_file']).name
        destination.write_text(last,encoding='utf-8',newline='')
        records.append({'member':name,'source_frames':count,'sid':row['sid'],'fid':row['fid'],
                        'output':str(destination.relative_to(root)),**hashes(destination)})
assert len(records)==129
result={'date':'2026-10-19','source_commit':'6a7dd43eeefac993df200fe278a2484b8395b49f',
        'archive_url':'https://dl.fbaipublicfiles.com/dac/datasets/extxyz_val.tar.gz',
        'archive':archive_hash,'metadata_record':'https://zenodo.org/records/15076798','metadata':metadata,
        'reconstructed_evaluated_frames':records,'seconds':time.monotonic()-started,
        'scope':'Reconstructs the 129 successful historical evaluation frames, not the 16 excluded inputs or training split',
        'gpu_run':False,'remaining_blocker':'Two original trained checkpoints have no published download in repository releases, Actions artifacts or PR1',
        'archive_hash_resolution':'The previously observed f7f2f58669a30abae8cb9ba1b7f2bcd2 is the official IS2RS archive hash, not the extxyz validation archive hash. This identifies a source-record mismatch, not corrupt bytes.'}
(root/'docs/reports/YICAO_DOWNLOAD.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({'archive':archive_hash,'reconstructed':len(records),'seconds':result['seconds']},indent=2))

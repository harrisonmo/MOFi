"""Bounded live transport and GPU preflight for the v3 launch."""
import json
from pathlib import Path
from rsi.blind_g1 import ROOT, read, write
from rsi.blind_v3_runner import diagnostic_input, gpu, health
from rsi.blind_campaign_v3 import health_errors
from scripts.run_blind_codex_cli import invoke, smoke_admitted


def main():
    output = ROOT / 'runs/blind-v3-preflight-01'
    output.mkdir(exist_ok=False)
    source = output / 'candidate'
    source.mkdir()
    (source / 'train.py').write_text('''import argparse, json
from pathlib import Path
import torch
p=argparse.ArgumentParser()
for name in ('input','output','seed','epochs'): p.add_argument('--'+name)
a=p.parse_args()
assert torch.cuda.is_available()
x=torch.ones(4,device='cuda').sum().item()
rows=json.loads((Path(a.input)/'validation.json').read_text())
out=Path(a.output)
(out/'predictions.json').write_text(json.dumps({r['id']:x for r in rows}))
(out/'metadata.json').write_text(json.dumps({'diagnostic_features':{r['id']:[x] for r in rows}}))
''', encoding='utf-8')
    rows = diagnostic_input(ROOT / 'runs/blind-g1-preparation-01', output / 'input')
    image = read(ROOT / 'runs/blind-g1-calibration-03/report.json')['image_id']
    result = gpu(output / 'input', source, image, 17, 60, epochs=1)
    write(output / 'gpu.json', result)
    assert not health_errors(health(result, rows)), 'GPU diagnostic admission failed'
    prompt = ('Synthetic transport check only. Do not use tools or access files. Return JSON '
              'with ready=true and tools_listing containing the names of tools available to you. '
              'Do not invent tool names; an empty list is valid.')
    invoke(prompt, output / 'cli', 120)
    assert smoke_admitted(output / 'cli'), 'CLI admission failed'
    write(output / 'admission.json', {'passed': True, 'model': 'gpt-6.1-sol', 'effort': 'high'})
    print(json.dumps({'passed': True}), flush=True)


if __name__ == '__main__':
    main()

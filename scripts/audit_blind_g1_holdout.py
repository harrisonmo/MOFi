"""Audit recorded test exposure without opening dataset label files."""
import argparse
from pathlib import Path
from rsi.blind_g1 import ROOT, read, sha, write

FIELDS = {'heldout_evaluated', 'heldout_labels_read', 'heldout_labels_loaded'}

def walk(value, path=''):
    if isinstance(value, dict):
        for key, item in value.items():
            location = path + '/' + key
            if key in FIELDS:
                yield {'location': location, 'value': item}
            yield from walk(item, location)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from walk(item, path + '/' + str(index))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    evidence = []
    for path in sorted((ROOT / 'docs/reports').glob('G[1234]*.json')):
        evidence.append({'path': path.relative_to(ROOT).as_posix(), 'sha256': sha(path),
                         'declarations': list(walk(read(path)))})
    flags = [item for report in evidence for item in report['declarations']]
    result = {'scope': 'recorded G1-G4 reports, not all historical human or agent access',
              'reports': evidence, 'declarations_checked': len(flags),
              'declared_exposure_detected': any(item['value'] is not False for item in flags),
              'historical_strong_access_isolation_proven': False,
              'heldout_label_files_opened_by_audit': 0,
              'final_unseen_claim_admitted': False,
              'decision': 'Development search can proceed. Final unseen claims need a sealed evaluation and an exposure decision; historical files were not strongly access-isolated.'}
    write(args.output, result)
    print(f'Reports: {len(evidence)}; declarations: {len(flags)}; exposure flag: {result["declared_exposure_detected"]}')

if __name__ == '__main__':
    main()

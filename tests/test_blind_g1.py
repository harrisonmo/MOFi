"""Synthetic tests for the blind package and external evaluator."""
import pytest
from rsi.blind_g1 import evaluate, read, sha, verify, write

@pytest.fixture
def package(tmp_path):
    public = tmp_path / 'public'
    write(public / 'train.json', [{'id': 'a', 'graph': 'a.npz', 'cif': 'a.cif', 'target': 1.0}])
    write(public / 'validation.json', [{'id': 'b', 'graph': 'b.npz', 'cif': 'b.cif'}])
    write(tmp_path / 'private/development.json', [{'id': 'b', 'target': 2.0, 'family': 'f'}])
    write(tmp_path / 'manifest.json', {'public_files': {p.name: sha(p) for p in public.iterdir()},
          'development_sha256': sha(tmp_path / 'private/development.json')})
    return tmp_path

def test_scoring_and_persistent_slots(package):
    predictions = {'17': {'b': 1.0}, '29': {'b': 3.0}}
    assert evaluate(package, predictions, 'rsi', 1)['mean_mae'] == 1.0
    assert read(package / 'private/predictions/rsi-1.json') == predictions
    with pytest.raises(ValueError, match='Slot'):
        evaluate(package, predictions, 'rsi', 1)
    with pytest.raises(ValueError, match='Slot'):
        evaluate(package, predictions, 'rsi', 3)
    assert evaluate(package, predictions, 'independent', 1)['mean_mae'] == 1.0
    assert evaluate(package, predictions, 'rsi', 2)['mean_mae'] == 1.0

@pytest.mark.parametrize('value', [float('nan'), float('inf'), True, '2'])
def test_non_numeric_rejected(package, value):
    with pytest.raises(ValueError):
        evaluate(package, {'17': {'b': value}, '29': {'b': 2}}, 'rsi', 1)
    assert not (package / 'private/evaluation-ledger.json').exists()

@pytest.mark.parametrize('predictions', [{'17': {'b': 2}}, {'17': {'wrong': 2}, '29': {'b': 2}}])
def test_incomplete_submission(package, predictions):
    with pytest.raises(ValueError):
        evaluate(package, predictions, 'rsi', 1)

@pytest.mark.parametrize('target', ['public/train.json', 'private/development.json', 'public/extra.txt'])
def test_tamper_rejected(package, target):
    (package / target).write_text('[]')
    with pytest.raises(ValueError):
        verify(package)

def test_labels_cannot_be_added_to_public(package):
    path = package / 'public/validation.json'
    rows = read(path)
    rows[0]['target'] = 2
    write(path, rows)
    manifest = read(package / 'manifest.json')
    manifest['public_files']['validation.json'] = sha(path)
    write(package / 'manifest.json', manifest)
    with pytest.raises(ValueError, match='fields'):
        verify(package)

def test_build_scrubs_metadata_and_never_opens_heldout(tmp_path):
    from rsi.blind_g1 import build, SOURCES
    root = tmp_path / 'source'
    data = root / 'datasets/processed/g1'
    data.mkdir(parents=True)
    source_hashes = {}
    for name in SOURCES:
        path = root / 'scripts' / name
        path.parent.mkdir(exist_ok=True)
        path.write_text('# Synthetic parent source\n')
        source_hashes[name] = sha(path)
    config = root / 'configs/g1-comparison.json'
    write(config, {'gnn': {}, 'batch_size': 16, 'max_epochs': 50,
                   'early_stopping_patience': 10, 'historical_hint': 'exclude'})
    split_hashes, split_ids = {}, {}
    for split, identifier in [('train', 'a'), ('validation', 'b')]:
        row = {'id': identifier, 'family': identifier, 'target': 2,
               'geometry': [42], 'historical_hint': 'exclude'}
        for field, extension in [('graph', '.npz'), ('cif', '.cif')]:
            path = data / (identifier + extension)
            path.write_bytes(b'synthetic')
            row[field] = path.relative_to(root).as_posix()
            row[field + '_sha256'] = sha(path)
        write(data / (split + '.json'), [row])
        split_hashes[split] = sha(data / (split + '.json'))
        split_ids[split] = [identifier]
    write(data / 'manifest.json', {'training_admitted': True,
          'split_sha256': split_hashes, 'split_ids': split_ids})
    write(root / 'docs/reports/G1_BASELINE_RESULT.json', {
          'config_sha256': sha(config), 'provenance': {'root_commit': 'synthetic',
          'source_sha256': source_hashes, 'dataset_manifest_sha256': sha(data / 'manifest.json')}})
    output = tmp_path / 'package'
    result = build(output, root)
    assert result['heldout_labels_read'] is False
    assert result['execution_admitted'] is False
    assert set(read(output / 'public/train.json')[0]) == {'id', 'graph', 'cif', 'target'}
    assert set(read(output / 'public/validation.json')[0]) == {'id', 'graph', 'cif'}
    assert 'historical_hint' not in read(output / 'public/baseline.json')
    with pytest.raises(FileExistsError):
        build(output, root)
    config.write_text('{}')
    with pytest.raises(ValueError, match='configuration'):
        build(tmp_path / 'changed', root)

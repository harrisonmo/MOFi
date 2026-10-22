"""Check report calculations on synthetic confirmed predictions, without training."""
from scripts import report_blind_codex as report
from rsi.blind_g1 import read, write


def test_paired_confirmation_summary(tmp_path, monkeypatch):
    monkeypatch.setattr(report, 'ROOT', tmp_path)
    campaign = tmp_path / 'runs/blind-g1-campaign-01'
    seeds = [101, 202, 303]
    confirmation = {'status': 'completed', 'aggregate': {'parent': 2, 'independent': 1, 'rsi': .8},
        'rows': [{'configuration': name, 'seed': s, 'mae': value} for name,value in [('parent',2),('independent',1),('rsi',.8)] for s in seeds]}
    write(campaign / 'policy.json', {'baseline': {'mean_mae': 2}, 'fit_wall_seconds':270, 'confirmation_seeds':seeds})
    state = {'model':'gpt-6.1-sol', 'confirmation':confirmation, 'arms': {
        arm:{'slots':[], 'incumbent':'parent', 'metrics':{'mean_mae':1}} for arm in ('independent','rsi')}}
    write(campaign / 'ledger.json', {'providers':{'chatgpt':state}})
    report.main()
    result = read(tmp_path / 'docs/reports/BLIND_G1_SOL_RESULT.json')
    assert abs(result['paired_rsi_minus_independent_mae']['mean'] + .2) < 1e-10
    assert result['paired_rsi_minus_independent_mae']['sample_std'] == 0
    assert result['heldout_evaluated'] is False
    assert 'Single evolutionary run' in result['scope']

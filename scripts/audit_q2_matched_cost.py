"""Read-only matched-task cost analysis; never reruns a candidate or scorer."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    paths = [ROOT/'benchmarks/q2b_v21/astra-summary.json',ROOT/'benchmarks/q2b_v21/http-recovery-summary.json']
    segments = [json.loads(p.read_text(encoding='utf-8')) for p in paths]
    groups = []
    for name, s in zip(['initial','http'],segments):
        assert s['audit']['passed']
        assert sum(c['tokens'] or 0 for c in s['calls']) == s['known_token_lower_bound']
        assert sum(c['tokens'] is None for c in s['calls']) == s['unknown_usage']
        for episode in s['episodes']:
            prefix = episode['id']+'-'+episode['arm']+'-'
            calls = [c for c in s['calls'] if c['stage']=='research' and c['label'].startswith(prefix)]
            assert len(calls) == episode['calls'] == 2
            tokens = None if any(c['tokens'] is None for c in calls) else sum(c['tokens'] for c in calls)
            assert tokens == episode['tokens']
        index = {(r['id'],r['arm']):r for r in s['episodes']}
        assert len(index)==len(s['episodes'])
        for control in ['fixed','manual_control','removed']:
            pairs = []
            for (task,arm),g in index.items():
                if arm!='outer_generated' or (task,control) not in index:
                    continue
                c = index[task,control]
                assert g['block']==c['block'] and g['source']==c['source']
                pairs.append(dict(id=task,block=g['block'],source=g['source'],
                    generated_pass=g['passed'],control_pass=c['passed'],
                    generated_tokens=g['tokens'],control_tokens=c['tokens'],
                    generated_tool_error=g.get('tool_error'),control_tool_error=c.get('tool_error')))
            known = [r for r in pairs if r['generated_tokens'] is not None and r['control_tokens'] is not None]
            g_tokens = sum(r['generated_tokens'] for r in known)
            c_tokens = sum(r['control_tokens'] for r in known)
            both = [r for r in known if r['generated_pass'] and r['control_pass']]
            groups.append(dict(segment=name,control=control,matched=len(pairs),known_cost_pairs=len(known),
                wins=sum(r['generated_pass'] and not r['control_pass'] for r in pairs),
                losses=sum(r['control_pass'] and not r['generated_pass'] for r in pairs),
                generated_tokens=g_tokens,control_tokens=c_tokens,saved_tokens=c_tokens-g_tokens,
                both_successful=dict(pairs=len(both),generated_tokens=sum(r['generated_tokens'] for r in both),
                                     control_tokens=sum(r['control_tokens'] for r in both)),
                use_only_saving_fraction=(c_tokens-g_tokens)/c_tokens if c_tokens else None,pairs=pairs))
    author = segments[0]['author_tokens']
    aggregate = []
    for control in ['fixed','manual_control','removed']:
        subset = [g for g in groups if g['control']==control]
        g_tokens,c_tokens = (sum(g[k] for g in subset) for k in ['generated_tokens','control_tokens'])
        aggregate.append(dict(control=control,matched=sum(g['matched'] for g in subset),
            generated_use_tokens=g_tokens,control_use_tokens=c_tokens,
            generated_plus_full_author_tokens=g_tokens+author,
            saving_after_author_fraction=(c_tokens-g_tokens-author)/c_tokens,
            warning='Descriptive sum of within-segment matches; not a causal or all-cost estimate.'))
    author_validation = segments[0]['science_stages']['author_validation']['charged_seconds']
    result = dict(visibility='public_engineering_validation',sources={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        groups=groups,descriptive_aggregate=aggregate,author_tokens=author,author_validation_seconds=author_validation,
        total_calls=sum(len(s['calls']) for s in segments),known_total_tokens=sum(s['known_token_lower_bound'] for s in segments),
        unknown_usage=sum(s['unknown_usage'] for s in segments),
        transport_probe_tokens=sum(s['probe_tokens'] for s in segments),
        science_seconds_with_parent_counted_once=segments[1]['science_charged_seconds'],
        exclusions=['Cross-transport task pairs are excluded from matched comparisons.',
                    'All failed and unknown-usage calls remain in total accounting.',
                    'Author cost is charged once in each standalone hypothetical comparison, not per segment.',
                    'Shared calibration and probes are listed separately, not assigned a fictitious per-arm price.',
                    'Engineering-reference authoring, development-session tokens and labor are unmetered, not zero.',
                    'Subsets differ across controls; never compare unmatched totals as cost effectiveness.',
                    'No money conversion, statistical significance, or hidden-test claim.'])
    out = ROOT/'benchmarks/q2c_reference/q2-matched-cost.json'
    out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ['descriptive_aggregate','author_tokens','transport_probe_tokens','known_total_tokens','unknown_usage']},indent=2))


if __name__=='__main__':
    main()

"""Complete one externally interrupted trial within the original active-execution budget."""
import json
import subprocess
import time
from pathlib import Path
import shutil

import numpy as np
import torch
from run_g2_comparison import ROOT, DATA, configure, digest, load_split, save, train_epoch, evaluate
from run_g1_baselines import metrics
from g2_models import GeometryScaler, make_model


def main():
    source = ROOT / "runs/g2-comparison-01"
    output = ROOT / "runs/g2-recovery-01"
    merged = ROOT / "runs/g2-completed-01"
    if output.exists() or merged.exists():
        raise FileExistsError("Refusing to overwrite recovery artifacts")
    config = json.loads((source / "config.json").read_text())
    original = json.loads((source / "results.json").read_text())
    provenance = json.loads((source / "provenance.json").read_text())
    assert len(original["rows"]) == 11 and original["status"] == "running"
    assert {(r['arm'], r['seed']) for r in original['rows']} == {
        (a, s) for a in config['arms'] for s in config['seeds']} - {('hybrid', 43)}
    for name, expected in provenance["source_sha256"].items():
        assert digest(ROOT / "scripts" / name) == expected
    assert digest(DATA / "manifest.json") == config["dataset_manifest_sha256"]
    assert digest(ROOT / "requirements-gnn-lock.txt") == provenance["environment_lock_sha256"]
    # Charge the interrupted attempt its full per-trial allowance; do not use an
    # unrecorded partial duration or charge the inactive time between turns.
    completed_seconds = sum(r["seconds"] for r in original["rows"])
    overhead_allowance = 30.
    original_accounted = completed_seconds + 150. + overhead_allowance
    assert original_accounted + 150. < config["formal_seconds"]
    output.mkdir()
    begin = time.monotonic(); deadline = begin + 150
    manifest = json.loads((DATA / "manifest.json").read_text())
    train, graphs = load_split("train", manifest)
    val, vg = load_split("validation", manifest)
    x = np.array([r["geometry"] for r in train]); xv = np.array([r["geometry"] for r in val])
    y = np.array([r["target"] for r in train]); yv = np.array([r["target"] for r in val])
    scaler = GeometryScaler().fit(x); xs, xvs = scaler.transform(x), scaler.transform(xv)
    mean, std = float(y.mean()), float(y.std())
    configure(43)
    model = make_model("hybrid").cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.00001)
    rng = np.random.default_rng(43)
    checkpoint = output / 'hybrid-43.pt'
    best, stale, history, best_epoch = float('inf'), 0, [], 0
    torch.cuda.reset_peak_memory_stats()
    for ep in range(1, 51):
        loss = train_epoch(model, 'hybrid', graphs, xs, (y-mean)/std, optimizer, rng, deadline)
        pred = evaluate(model, 'hybrid', vg, xvs, mean, std, deadline)
        score = float(abs(pred-yv).mean())
        history.append({'epoch':ep,'train_mse_normalized':loss,'validation_mae':score})
        if score < best:
            best, stale, best_epoch = score, 0, ep
            torch.save(model.state_dict(), checkpoint)
        else:
            stale += 1
        save(output/'history-hybrid-43.json', history)
        if ep % 10 == 0: print(f'Recovery hybrid-43 epoch={ep} mae={score:.6f}',flush=True)
        if stale >= 10: break
    old_history = json.loads((source/'history-hybrid-43.json').read_text())
    assert history[:len(old_history)] == old_history, 'Interrupted prefix did not reproduce exactly'
    model.load_state_dict(torch.load(checkpoint,map_location='cuda',weights_only=True))
    pred = evaluate(model,'hybrid',vg,xvs,mean,std,deadline)
    training_pred = evaluate(model,'hybrid',graphs,xs,mean,std,deadline)
    elapsed = time.monotonic()-begin
    assert elapsed < 150
    row = {'arm':'hybrid','seed':43,'seconds':elapsed,'train_mae':float(abs(training_pred-y).mean()),
           'checkpoint_sha256':digest(checkpoint),'epochs':len(history),'best_epoch':best_epoch,
           'optimizer_updates':len(history)*int(np.ceil(len(train)/16)),
           'parameters':sum(p.numel() for p in model.parameters()),
           'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
           **metrics(yv,pred,[r['id'] for r in val])}
    recovery = {'reason':'External turn interruption; one fixed-configuration replacement',
                'root_commit':subprocess.check_output(['git','-c',f'safe.directory={ROOT.as_posix()}','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                'source_sha256':digest(Path(__file__)), 'completed_original_trial_seconds':completed_seconds,
                'interrupted_attempt_charge_seconds':150.,'startup_io_allowance_seconds':overhead_allowance,
                'recovery_seconds':elapsed,'budget_accounted_seconds':original_accounted+elapsed,
                'original_partial_epochs':len(old_history),'prefix_exactly_reproduced':True,
                'attempts':13,'completed_trials':12,'inactive_time_excluded':True,'heldout_evaluated':False}
    save(output/'recovery.json',recovery);save(output/'row.json',row)
    shutil.copytree(source,merged)
    for name in ['hybrid-43.pt','history-hybrid-43.json']:
        shutil.copyfile(output/name,merged/name)
    result = {**original, 'status':'completed','rows':original['rows']+[row],
              'total_seconds':original_accounted+elapsed,
              'time_interpretation':'Conservative active-execution budget accounting, not measured uninterrupted wall time',
              'recovery':recovery}
    predictions = json.loads((source/'predictions.json').read_text())
    predictions['models']['hybrid_43'] = pred.tolist()
    save(merged/'results.json',result);save(merged/'predictions.json',predictions)
    print(json.dumps(recovery),flush=True)


if __name__=='__main__':
    main()

"""Probability-averaging ensemble of Phase 4 runs, computed locally (CPU).

    python -I phase4_cjpe/tools/local_ensemble.py <data_dir_with_Phase_4_CJPE> <probs_dir> <out_dir> [runs=1,2,3,4]

<probs_dir> holds run<k>_dev.json / run<k>_test.json ({case id: P(label=1)}), exported by each run notebook's cell 6.
Dev labels come from dev.json (dev only). test.json is read for its ids only; its labels are never touched."""
import itertools, json, sys, zipfile
from pathlib import Path
import numpy as np
REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO / 'phase4_cjpe'), str(REPO / 'phase3_rr'), str(REPO / 'common')]
import cjpe_lib as J

data_dir, probs_dir, out_dir = Path(sys.argv[1]) / 'Phase_4_CJPE', Path(sys.argv[2]), Path(sys.argv[3])
runs = [int(x) for x in (sys.argv[4] if len(sys.argv) > 4 else '1,2,3,4').split(',')]
raw = json.load(open(data_dir / 'Train/dev.json')); y_by_id = {str(d['id']): int(d['label']) for d in raw}
test_ids = [str(d['id']) for d in json.load(open(data_dir / 'Test/test.json'))]          # ids only
P = {k: {s: json.load(open(probs_dir / f'run{k}_{s}.json')) for s in ('dev', 'test')} for k in runs}
dev_ids = list(P[runs[0]]['dev'])
assert all(set(P[k]['dev']) == set(y_by_id) == set(dev_ids) for k in runs), 'dev ids differ between runs / dev.json'
assert all(list(P[k]['test']) == test_ids for k in runs), 'test ids differ from test.json (order or content)'
y = np.array([y_by_id[i] for i in dev_ids])
dev = {k: np.array([P[k]['dev'][i] for i in dev_ids]) for k in runs}
test = {k: np.array([P[k]['test'][i] for i in test_ids]) for k in runs}

def score(p):
    s = J.macro_scores(y, (p >= 0.5).astype(int)); thr, ft = J.best_threshold(y, p)
    return s['macro_f1'], ft, thr
print('== single runs (dev macro-F1 at 0.5 | tuned @ threshold) ==')
for k in runs:
    f, ft, t = score(dev[k]); print(f'run {k}: {f:.4f} | {ft:.4f} @ {t:.2f}')
print('\n== agreement between runs (Pearson r of dev probabilities) ==')
print('      ' + ' '.join(f'run{k:>2}' for k in runs))
for a in runs: print(f'run {a}  ' + ' '.join(f'{np.corrcoef(dev[a], dev[b])[0,1]:5.2f}' for b in runs))
rows = []
for n in range(2, len(runs) + 1):
    for sub in itertools.combinations(runs, n):
        f, ft, t = score(np.mean([dev[k] for k in sub], 0)); rows.append((ft, f, t, sub))
print('\n== equal-weight probability averages, sorted by tuned dev F1 ==')
for ft, f, t, sub in sorted(rows, reverse=True): print(f'runs {sub}: {f:.4f} | tuned {ft:.4f} @ {t:.2f}')

# is "all runs" better than the best single run by more than dev noise? paired bootstrap over dev cases at fixed thresholds
best1 = max(runs, key=lambda k: score(dev[k])[1]); thr1 = score(dev[best1])[2]
pe = np.mean([dev[k] for k in runs], 0); thre = score(pe)[2]
rng = np.random.RandomState(0); diffs = []
for _ in range(2000):
    i = rng.randint(0, len(y), len(y))
    diffs.append(J.macro_scores(y[i], (pe[i] >= thre).astype(int))['macro_f1'] - J.macro_scores(y[i], (dev[best1][i] >= thr1).astype(int))['macro_f1'])
diffs = np.array(diffs)
print(f'\nbootstrap (2000 resamples of dev): all-runs ensemble minus best single run {best1}, each at its own tuned threshold: '
      f'mean {diffs.mean():+.4f}, 95% interval [{np.percentile(diffs,2.5):+.4f}, {np.percentile(diffs,97.5):+.4f}], ensemble better in {100*np.mean(diffs>0):.0f}% of resamples')

def write(p_test, thr, name):
    d = out_dir / name; d.mkdir(parents=True, exist_ok=True)
    preds = [{'id': i, 'pred': int(p >= thr)} for i, p in zip(test_ids, p_test)]
    (d / 'prediction.json').write_text(json.dumps(preds))
    with zipfile.ZipFile(d / 'submission.zip', 'w', zipfile.ZIP_DEFLATED) as z: z.write(d / 'prediction.json', arcname='prediction.json')
    return d / 'submission.zip', np.mean([r['pred'] for r in preds])
f_all, ft_all, t_all = score(pe)
print(f'\nENSEMBLE of all runs {runs}: dev macro-F1 {f_all:.4f} at 0.5, {ft_all:.4f} tuned @ {t_all:.2f}')
pt = np.mean([test[k] for k in runs], 0)
for tag, t in (('thr0.5', 0.5), ('tuned', t_all)):
    path, share = write(pt, t, f'ens_all_runs{"".join(map(str, runs))}__{tag}'); print(f'wrote {path.parent.name}/submission.zip (threshold {t:.2f}, predicts label 1 for {100*share:.1f}% of test cases)')

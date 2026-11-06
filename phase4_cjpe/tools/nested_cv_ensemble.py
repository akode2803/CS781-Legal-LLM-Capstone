"""Does FITTING the ensemble on dev help? Nested cross-validation on the dev set.

Each recipe makes all its choices (which runs, weights, threshold) on the training part of a split of dev and is scored on the
held-out part; pooled held-out predictions give an honest macro-F1 for the recipe. Repeated over many random splits.

    python -I phase4_cjpe/tools/nested_cv_ensemble.py <data_dir_with_Phase_4_CJPE> <probs_dir> [runs=1,2,3,4]"""
import itertools, json, sys
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO / 'phase4_cjpe'), str(REPO / 'phase3_rr'), str(REPO / 'common')]
import cjpe_lib as J
data_dir, probs_dir = Path(sys.argv[1]) / 'Phase_4_CJPE', Path(sys.argv[2])
runs = [int(x) for x in (sys.argv[3] if len(sys.argv) > 3 else '1,2,3,4').split(',')]
y_by_id = {str(d['id']): int(d['label']) for d in json.load(open(data_dir / 'Train/dev.json'))}   # dev labels only
P = {k: json.load(open(probs_dir / f'run{k}_dev.json')) for k in runs}
ids = list(P[runs[0]]); y = np.array([y_by_id[i] for i in ids]); X = np.stack([[P[k][i] for i in ids] for k in runs], 1)
eps = 1e-4; L = np.log(np.clip(X, eps, 1 - eps) / (1 - np.clip(X, eps, 1 - eps)))
def f1(yy, pp, t): return J.macro_scores(yy, (pp >= t).astype(int))['macro_f1']
def tune(yy, pp): return J.best_threshold(yy, pp)          # (threshold, F1)
subsets = [s for n in range(1, len(runs) + 1) for s in itertools.combinations(range(len(runs)), n)]

def recipe_single_fixed(tr, te, k=0):            # always run 1 with its threshold tuned on the training part
    t, _ = tune(y[tr], X[tr, k]); return (X[te, k] >= t).astype(int)
def recipe_best_single(tr, te):                  # pick the best run AND threshold on the training part
    best = max(range(len(runs)), key=lambda k: tune(y[tr], X[tr, k])[1]); t, _ = tune(y[tr], X[tr, best]); return (X[te, best] >= t).astype(int)
def recipe_avg_all(tr, te):                      # equal-weight average of all runs, tuned threshold
    p = X.mean(1); t, _ = tune(y[tr], p[tr]); return (p[te] >= t).astype(int)
def recipe_avg_all_05(tr, te): return (X[te].mean(1) >= 0.5).astype(int)
def recipe_best_subset(tr, te):                  # pick the best subset of runs (15 candidates) AND threshold on the training part
    best = max(subsets, key=lambda s: tune(y[tr], X[tr][:, list(s)].mean(1))[1]); p = X[:, list(best)].mean(1); t, _ = tune(y[tr], p[tr]); return (p[te] >= t).astype(int)
def recipe_stack(tr, te, C=1.0):                 # logistic regression on logit(p): learns one weight per run + intercept
    m = LogisticRegression(C=C, max_iter=500).fit(L[tr], y[tr]); p = m.predict_proba(L)[:, 1]; t, _ = tune(y[tr], p[tr]); return (p[te] >= t).astype(int)
recipes = {'run 1 alone, tuned threshold (no run selection)': recipe_single_fixed, 'best single run + threshold (picked on train part)': recipe_best_single,
           'average of all runs, threshold 0.5 (nothing fitted)': recipe_avg_all_05, 'average of all runs, tuned threshold (1 fitted number)': recipe_avg_all,
           'best subset of runs + threshold (picked on train part)': recipe_best_subset, 'logistic-regression stacking + threshold (5 weights + threshold)': recipe_stack}
REPEATS, K = 30, 5; res = {n: [] for n in recipes}
for rep in range(REPEATS):
    rng = np.random.RandomState(rep); order = rng.permutation(len(y)); folds = np.array_split(order, K); pred = {n: np.zeros(len(y), int) for n in recipes}
    for f in range(K):
        te = folds[f]; tr = np.concatenate([folds[g] for g in range(K) if g != f])
        for n, fn in recipes.items(): pred[n][te] = fn(tr, te)
    for n in recipes: res[n].append(J.macro_scores(y, pred[n])['macro_f1'])
print(f'nested CV on dev ({len(y)} cases, {K}-fold, {REPEATS} repeats): held-out macro-F1, mean +- sd over repeats')
for n in recipes: print(f'  {np.mean(res[n]):.4f} +- {np.std(res[n]):.4f}   {n}')
print('\nfor reference, the in-sample (optimistic) dev scores from the earlier table: run 1 tuned 0.6243 | all-four avg tuned 0.6245 | best subset 0.6388')
m = LogisticRegression(C=1.0, max_iter=500).fit(L, y)
print('stacking weights fitted on ALL of dev (runs', runs, '):', np.round(m.coef_[0], 3), 'intercept', np.round(m.intercept_, 3))

"""One-factor controls on dev: neural-only F1, AUC inside the lexical top 50, paired bootstrap against each run's baseline, and the seed spread for scale.
usage: python report_controls.py <ctx features pickle> <out json>"""
import json, pickle, sys
import numpy as np
from ltr_neural import RUNS, load, z

PAIRS = [('p7x-enc-pairwise', 'p7-enc-full', 'loss: pairwise, not listwise'), ('p7x-enc-randneg', 'p7-enc-full', 'negatives: random, not lexical top 40'),
         ('p7x-enc-last', 'p7-enc-full', 'encoder pooling: last, not mean'), ('p7x-gqa-mean', 'p7-gqa-full', 'GQA pooling: mean, not last'),
         ('p7x-enc-frozen', 'p7-enc-full', 'transfer: frozen, head only'), ('p7x-enc-random', 'p7-enc-full', 'transfer: random start, no Phase 1')]
SEEDS = ('p7c-enc-2ep', 'p7c-enc-s101', 'p7c-enc-s102', 'p7c-enc-s103', 'p7c-enc-s104')
K = 7
F = pickle.load(open(sys.argv[1], 'rb')); X, Y, G, I, gold, qi = F['dev']
rows = {k: np.where((G == k) & (X[:, 15] < 50))[0] for k in qi}
NG = np.array([len(gold[q]) for q in qi])


def score(run):
    sc = load(f'{RUNS}/{run}/dev_scores.json'); return np.array([sc.get((a, b), -1e9) for a, b in zip(G, I)])


def hits(s):
    return np.array([len(set(I[r][np.argsort(-s[r], kind='stable')][:K]) & set(gold[q])) for q, r in rows.items()])


def f1(h, i=slice(None)):
    t = h[i].sum(); p = t / (K * len(h[i])); r = t / NG[i].sum(); return 2 * p * r / (p + r) if t else 0.


def auc(s):
    v = []
    for r in rows.values():
        y = Y[r].astype(bool)
        if 0 < y.sum() < len(y):
            o = s[r].argsort().argsort() + 1; v.append((o[y].sum() - y.sum() * (y.sum() + 1) / 2) / (y.sum() * (~y).sum()))
    return float(np.mean(v))


def paired(a, b, n=5000):
    rng = np.random.default_rng(0); d = []
    for _ in range(n):
        i = rng.integers(0, len(qi), len(qi)); d.append(f1(a, i) - f1(b, i))
    return [round(float(np.percentile(d, 2.5)), 4), round(float(np.percentile(d, 97.5)), 4)]


out = {}
for run, base, what in PAIRS:
    s, b = score(run), score(base); h, hb = hits(s), hits(b); r = json.load(open(f'{RUNS}/{run}/results.json'))
    out[run] = dict(what=what, baseline=base, f1=round(f1(h), 4), base_f1=round(f1(hb), 4), diff=round(f1(h) - f1(hb), 4), ci=paired(h, hb), auc=round(auc(s), 4), base_auc=round(auc(b), 4), trainable_pct=round(r['trainable_pct'], 3), minutes=round(r['minutes'], 1), peak_gb=round(r['peak_gpu_gb'], 1), top1=[round(e['top1'], 3) for e in r['history']], neural_by_epoch=[round(e['neural_only'], 4) for e in r['history']])
sf = [f1(hits(score(r))) for r in SEEDS]; sa = [auc(score(r)) for r in SEEDS]
out['seed_spread'] = dict(runs=SEEDS, f1=[round(x, 4) for x in sf], f1_sd=round(float(np.std(sf, ddof=1)), 4), auc=[round(x, 4) for x in sa], auc_sd=round(float(np.std(sa, ddof=1)), 4))
json.dump(out, open(sys.argv[2], 'w'), indent=1); print(json.dumps(out, indent=0))

"""Do the one-factor controls change the final system? Each control's score is blended with the citing-context ranker on dev, alone and added to the final three.
usage: python report_controls_blend.py <ctx features pickle> <out json>"""
import json, pickle, sys
import numpy as np
from ltr_neural import RUNS, fit, load, z

FINAL = ('p7-enc-full', 'p7c-enc', 'p7c-lorma')
CTRL = ('p7x-enc-pairwise', 'p7x-enc-randneg', 'p7x-enc-last', 'p7x-gqa-mean', 'p7x-enc-frozen', 'p7x-enc-random')
K = 7
F = pickle.load(open(sys.argv[1], 'rb')); Xt, Yt = F['train'][:2]; Xd, Yd, Gd, Id, gold, qi = F['dev']
rows = {k: np.where(Gd == k)[0] for k in qi}; NG = np.array([len(gold[q]) for q in qi])


def neural(run):
    sc = load(f'{RUNS}/{run}/dev_scores.json'); o = np.array([sc.get((a, b), np.nan) for a, b in zip(Gd, Id)])
    for r in rows.values():
        v = o[r]; ok = ~np.isnan(v); v[ok] = z(v[ok]); v[~ok] = -2.; o[r] = v
    return o


def hits(s):
    return np.array([len(set(Id[r][np.argsort(-s[r], kind='stable')][:K]) & set(gold[q])) for q, r in rows.items()])


def f1(h, i=slice(None)):
    t = h[i].sum(); p = t / (K * len(h[i])); r = t / NG[i].sum(); return 2 * p * r / (p + r) if t else 0.


def paired(a, b, n=5000):
    rng = np.random.default_rng(0); d = []
    for _ in range(n):
        i = rng.integers(0, len(qi), len(qi)); d.append(f1(a, i) - f1(b, i))
    return [round(float(np.percentile(d, 2.5)), 4), round(float(np.percentile(d, 97.5)), 4)]


p = fit(Xt[:, :27], Yt, [Xd[:, :27]])[0].clip(1e-6, 1 - 1e-6); rank = np.log(p / (1 - p))
for r in rows.values(): rank[r] = z(rank[r])
N = {r: neural(r) for r in FINAL + CTRL}
base = hits(rank + .2 * np.mean([N[r] for r in FINAL], 0)); out = dict(ranker=[round(f1(hits(rank)), 4), int(hits(rank).sum())], final=[round(f1(base), 4), int(base.sum())])
for c in CTRL:
    e = {}
    for w in (.1, .2, .3, .5):
        h = hits(rank + w * N[c]); e[f'alone_w{w}'] = [round(f1(h), 4), int(h.sum())]
    for w in (.2, .3):
        h = hits(rank + w * np.mean([N[r] for r in FINAL + (c,)], 0)); e[f'with_final_w{w}'] = [round(f1(h), 4), int(h.sum())]
    h = hits(rank + .2 * N[c]); e['alone_w0.2_vs_final_ci'] = paired(h, base); out[c] = e
from scipy.stats import spearmanr
m50 = {k: r[Xd[r, 15] < 50] for k, r in rows.items()}
rho = lambda c: round(float(np.nanmean([spearmanr(N[c][r], rank[r])[0] for r in m50.values()])), 3)
N['p7-enc-full'] = neural('p7-enc-full'); one = hits(rank + .2 * N['p7-enc-full'])
out['enc_full_alone_w0.2'] = [round(f1(one), 4), int(one.sum())]; out['rho_with_ranker_top50'] = {c: rho(c) for c in ('p7-enc-full',) + CTRL}
out['vs_enc_full_single_ci'] = {c: paired(hits(rank + .2 * N[c]), one) for c in CTRL}
g = sum(len(set(Id[r]) & set(gold[k])) for k, r in m50.items()); hc = 7 / 50 * g; out['chance_top50'] = [int(g), round(hc, 1), round(2 * hc / (7 * len(qi) + NG.sum()), 4)]
json.dump(out, open(sys.argv[2], 'w'), indent=1, default=float)
for k, v in out.items(): print(k, v)

"""Variable-length lists for Micro-F1@7: keep a top-7 prediction only when its estimated hit probability is above tau.
Micro-F1 rises when a prediction with hit probability below about F/2 is dropped, because it removes one predicted slot
for less than F/2 expected hits. Dev only; tau is chosen by 2-fold cross-fit over dev queries (repeated splits).
usage: python vark_eval.py <ctx features pickle> [<run>,<run>,... (default: final three)]"""
import pickle, sys
import numpy as np, pcr_lib as P
from ltr_neural import RUNS, fit, load, z

F = pickle.load(open(sys.argv[1], 'rb'))
runs = (sys.argv[2] if len(sys.argv) > 2 else 'p7-enc-full,p7c-enc,p7c-lorma').split(',')
Xt, Yt = F['train'][:2]; Xd, _, Gd, Id, gold, qi = F['dev']

def neural(run):
    sc = load(f'{RUNS}/{run}/dev_scores.json'); o = np.array([sc.get((a, b), np.nan) for a, b in zip(Gd, Id)])
    for k in qi:
        m = Gd == k; v = o[m]; ok = ~np.isnan(v); v[ok] = z(v[ok]); v[~ok] = -2.; o[m] = v
    return o

p, = fit(Xt, Yt, [Xd]); p = p.clip(1e-6, 1 - 1e-6)
s = np.log(p / (1 - p))
for k in qi: s[Gd == k] = z(s[Gd == k])
f = s + 0.2 * np.mean([neural(r) for r in runs], 0)

top = {}   # query -> [(case, ranker prob, rank)] for the top 7 by the final blend
for k in qi:
    m = np.where(Gd == k)[0]; o = m[np.argsort(-f[m], kind='stable')][:7]
    top[k] = [(Id[i], p[i], r) for r, i in enumerate(o)]

def score(qs, tau, min_k=1):
    pred = {k: [c for c, pr, r in top[k] if r < min_k or pr >= tau] for k in qs}
    return P.micro_prf(pred, {k: gold[k] for k in qs})

full = score(qi, -1); print('fixed 7: F1 %.4f P %.4f R %.4f hits %d preds %d' % (full['f1'], full['precision'], full['recall'], full['tp'], 7 * len(qi)))
# calibration of the ranker probability on dev top-7 slots
pr = np.array([x[1] for k in qi for x in top[k]]); hit = np.array([x[0] in set(gold[k]) for k in qi for x in top[k]])
for lo, hi in ((0, .05), (.05, .1), (.1, .2), (.2, .3), (.3, .5), (.5, 1.01)):
    m = (pr >= lo) & (pr < hi); print(f'  ranker p in [{lo}, {hi}): {m.sum()} slots, hit rate {hit[m].mean() if m.any() else float("nan"):.3f}')
taus = [0, .02, .04, .06, .08, .1, .12, .15, .2, .25, .3]
for t in taus:
    r = score(qi, t); print(f'tau {t:.2f}: F1 {r["f1"]:.4f} P {r["precision"]:.4f} R {r["recall"]:.4f} hits {r["tp"]}  (in-sample)')
# honest estimate: pick tau on one half of the queries, score the other half; 20 random splits
rng = np.random.default_rng(0); gains = []; picked = []
for rep in range(20):
    q = np.array(qi); rng.shuffle(q); A, B = list(q[:len(q) // 2]), list(q[len(q) // 2:])
    for fit_q, ev_q in ((A, B), (B, A)):
        t = max(taus, key=lambda t: score(fit_q, t)['f1']); picked.append(t)
        gains.append(score(ev_q, t)['f1'] - score(ev_q, -1)['f1'])
g = np.array(gains); print(f'cross-fit gain over fixed 7: mean {g.mean():+.4f}, sd {g.std():.4f}, positive in {np.mean(g > 0):.0%} of 40 halves; taus picked {sorted(set(picked))}')

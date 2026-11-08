"""Test zip for the final blend (ranker + 0.2 mean z of three rerankers) with variable-length lists:
a top-7 prediction is kept only if the ranker's hit probability is >= tau (the first one is always kept). See vark_eval.py.
usage: python vark_submit.py <ctx features pickle> <tau> <out zip>"""
import json, pickle, sys, zipfile
import numpy as np, pcr_lib as P
from ltr_neural import RUNS, fit, load, z

F = pickle.load(open(sys.argv[1], 'rb')); tau = float(sys.argv[2]); out = sys.argv[3]
runs = ['p7-enc-full', 'p7c-enc', 'p7c-lorma']
Xt, Yt = F['train'][:2]; Xs, _, Gs, Is, _, ti = F['test']
pool = set(P.load_split(__import__('ltr_features').R, 'test')[1])

def neural(run):
    sc = load(f'{RUNS}/{run}/test_scores.json'); o = np.array([sc.get((a, b), np.nan) for a, b in zip(Gs, Is)])
    for k in ti:
        m = Gs == k; v = o[m]; ok = ~np.isnan(v); v[ok] = z(v[ok]); v[~ok] = -2.; o[m] = v
    return o

p, = fit(Xt, Yt, [Xs]); p = p.clip(1e-6, 1 - 1e-6)
s = np.log(p / (1 - p))
for k in ti: s[Gs == k] = z(s[Gs == k])
f = s + 0.2 * np.mean([neural(r) for r in runs], 0)

pred, kept = {}, []
for k in ti:
    m = np.where(Gs == k)[0]; o = m[np.argsort(-f[m], kind='stable')][:7]
    pred[k] = [str(Is[i]) for r, i in enumerate(o) if r == 0 or p[i] >= tau]; kept.append(len(pred[k]))
    assert 1 <= len(pred[k]) <= 7 and len(set(pred[k])) == len(pred[k]) and all(c in pool for c in pred[k])
with zipfile.ZipFile(out, 'w', zipfile.ZIP_DEFLATED) as zf:
    zf.writestr('prediction.json', json.dumps({k: pred[k] for k in ti}, indent=1))
full7 = {k: [str(Is[i]) for i in np.where(Gs == k)[0][np.argsort(-f[Gs == k], kind='stable')][:7]] for k in ti}
print(out, 'queries', len(ti), 'predictions', sum(kept), f'({sum(kept) / (7 * len(ti)):.0%} of 7 per query)',
      'mean per query', round(np.mean(kept), 2), '| same top order as the 0.4938 entry:', all(pred[k] == full7[k][:len(pred[k])] for k in ti))

"""Dev comparison of chunk scores against the final blend (ranker + 0.2 mean z of the three excerpt rerankers, 449 hits).
A small fixed set of options, declared before looking at dev: how window scores are pooled per candidate, and whether they replace or join the excerpt score.
usage: python chunk_eval.py <ctx features pickle> [test <option> <out zip>]"""
import json, pickle, sys
import numpy as np, pcr_lib as P
from ltr_neural import RUNS, fit, load, z

F = pickle.load(open(sys.argv[1], 'rb')); split = 'test' if len(sys.argv) > 2 and sys.argv[2] == 'test' else 'dev'
Xt, Yt = F['train'][:2]; X, _, G, I, gold, qi = F[split]
NAMES = ['p7-enc-full', 'p7c-enc', 'p7c-lorma']

def zcol(o):
    o = o.copy()
    for k in qi:
        m = G == k; v = o[m]; ok = ~np.isnan(v); v[ok] = z(v[ok]) if ok.sum() > 1 else 0; v[~ok] = -2.; o[m] = v
    return o

def excerpt(run): sc = load(f'{RUNS}/{run}/{split}_scores.json'); return zcol(np.array([sc.get((a, b), np.nan) for a, b in zip(G, I)]))

def chunks(run, pool):
    raw = json.load(open(f'{RUNS}/{run}/chunks_{split}.json'))
    f = {'max': np.max, 'top2': lambda v: np.mean(sorted(v)[-2:]), 'mean': np.mean}[pool]
    return zcol(np.array([f(raw[f'{a}|{b}']) if f'{a}|{b}' in raw else np.nan for a, b in zip(G, I)]))

p, = fit(Xt, Yt, [X]); p = p.clip(1e-6, 1 - 1e-6); s = np.log(p / (1 - p))
for k in qi: s[G == k] = z(s[G == k])
EX = np.mean([excerpt(r) for r in NAMES], 0)
CH = {pool: np.mean([chunks(r, pool) for r in NAMES], 0) for pool in ('max', 'top2', 'mean')}
OPTIONS = {'final (excerpt only)': s + .2 * EX}
for pool in CH:
    OPTIONS[f'windows {pool} only'] = s + .2 * CH[pool]
    OPTIONS[f'excerpt + windows {pool}'] = s + .2 * (EX + CH[pool]) / 2
    OPTIONS[f'excerpt + windows {pool}, w .3'] = s + .3 * (EX + CH[pool]) / 2

def ranked(f): return {k: [str(x) for x in I[G == k][np.argsort(-f[G == k], kind='stable')][:7]] for k in qi}

if split == 'dev':
    base = ranked(OPTIONS['final (excerpt only)']); bt = {k: len(set(base[k]) & set(gold[k])) for k in qi}
    for name, f in OPTIONS.items():
        pr = ranked(f); r = P.micro_prf(pr, gold); d = np.array([len(set(pr[k]) & set(gold[k])) - bt[k] for k in qi])
        bs = np.random.default_rng(0).choice(len(qi), (2000, len(qi))); lo, hi = np.percentile(d[bs].sum(1), [2.5, 97.5])
        print(f'{name:34s} F1 {r["f1"]:.4f} hits {r["tp"]}  vs final {d.sum():+d} hits [95% {lo:+.0f}, {hi:+.0f}]')
    # per-model check: does each model's window max beat its own excerpt score alone?
    for r in NAMES:
        for nm, v in (('excerpt', excerpt(r)), ('windows max', chunks(r, 'max'))):
            print(f'  {r} {nm} alone: F1 {P.micro_prf(ranked(v), gold)["f1"]:.4f}')
else:
    opt, out = sys.argv[3], sys.argv[4]; pred = ranked(OPTIONS[opt]); pool = set(P.load_split(__import__('ltr_features').R, 'test')[1])
    print(P.write_submission(pred, qi, pool, out), opt)

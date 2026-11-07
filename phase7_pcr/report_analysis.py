"""Dev analysis for the Phase 7 report: rebuilds the final blend, then breaks errors down and compares models. CPU only, reads cached features and saved scores.
usage: python report_analysis.py <ctx features pickle> <out json>"""
import json, pickle, sys
import numpy as np, pcr_lib as P
from scipy.stats import spearmanr
from ltr_features import R
from ltr_neural import RUNS, fit, load, z

FINAL = ('p7-enc-full', 'p7c-enc', 'p7c-lorma')
SEVEN = ('p7-enc-full', 'p7-gqa-full', 'p7-gqa-lorma', 'p7-gqa-kd', 'p7-gqa-ia3', 'p7-gqa-partial', 'p7-gqa-lora')
K = 7
F = pickle.load(open(sys.argv[1], 'rb')); Xt, Yt = F['train'][:2]; Xd, Yd, Gd, Id, gold, qi = F['dev']
rows = {k: np.where(Gd == k)[0] for k in qi}
LEX = -Xd[:, 15]                        # lexical order: column 15 is the rank in the date-filtered 4-gram list


def neural(run):
    sc = load(f'{RUNS}/{run}/dev_scores.json'); o = np.array([sc.get((a, b), np.nan) for a, b in zip(Gd, Id)])
    for r in rows.values():
        v = o[r]; ok = ~np.isnan(v); v[ok] = z(v[ok]); v[~ok] = -2.; o[r] = v
    return o


def top(s, k=K):
    return {q: list(Id[r][np.argsort(-s[r], kind='stable')][:k]) for q, r in rows.items()}


def hits(pred):
    """Hits per query, in qi order."""
    return np.array([len(set(pred[q]) & set(gold[q])) for q in qi])


NG = np.array([len(gold[q]) for q in qi]); NP = np.array([K] * len(qi))


def f1(h, idx=slice(None)):
    t = h[idx].sum(); p = t / NP[idx].sum(); r = t / NG[idx].sum()
    return 2 * p * r / (p + r) if t else 0.


def paired(a, b, n=5000, seed=0):
    """Bootstrap over queries of F1(a) - F1(b): mean difference, 95% interval, share of resamples above 0."""
    rng = np.random.default_rng(seed); d = []
    for _ in range(n):
        i = rng.integers(0, len(qi), len(qi)); d.append(f1(a, i) - f1(b, i))
    d = np.array(d); return dict(diff=round(f1(a) - f1(b), 4), lo=round(float(np.percentile(d, 2.5)), 4), hi=round(float(np.percentile(d, 97.5)), 4), p_gt0=round(float((d > 0).mean()), 3))


p = fit(Xt[:, :27], Yt, [Xd[:, :27]])[0].clip(1e-6, 1 - 1e-6); rank_s = np.log(p / (1 - p))
for r in rows.values(): rank_s[r] = z(rank_s[r])
pold = fit(Xt[:, :19], Yt, [Xd[:, :19]])[0].clip(1e-6, 1 - 1e-6); old_s = np.log(pold / (1 - pold))
for r in rows.values(): old_s[r] = z(old_s[r])
blend = rank_s + .2 * np.mean([neural(r) for r in FINAL], 0)
S = dict(lexical=LEX, ranker=old_s, ctx_ranker=rank_s, final=blend)
H = {k: hits(top(v)) for k, v in S.items()}
out = dict(f1={k: round(f1(v), 4) for k, v in H.items()}, tp={k: int(v.sum()) for k, v in H.items()})
out['paired'] = {'ranker-lexical': paired(H['ranker'], H['lexical']), 'ctx-ranker': paired(H['ctx_ranker'], H['ranker']), 'final-ctx': paired(H['final'], H['ctx_ranker'])}

# where the gold goes under the final ranking
order = {q: list(Id[r][np.argsort(-blend[r], kind='stable')]) for q, r in rows.items()}
tot = int(NG.sum()); tp = int(H['final'].sum()); cap = int(np.maximum(NG - K, 0).sum())
in100 = sum(len(set(order[q]) & set(gold[q])) for q in qi); in50 = sum(len(set(order[q][:50]) & set(gold[q])) for q in qi); in20 = sum(len(set(order[q][:20]) & set(gold[q])) for q in qi)
out['gold'] = dict(total=tot, hit=tp, over_cap=cap, top20=in20, top50=in50, top100=in100, outside100=tot - in100)
out['precision_at_rank'] = [round(float(np.mean([order[q][i] in gold[q] for q in qi])), 3) for i in range(K)]
out['zero_hit_queries'] = int((H['final'] == 0).sum()); out['no_gold_queries'] = int((NG == 0).sum())


def bucket(vals, edges, idx_hits=H['final']):
    res = []
    for a, b in zip(edges[:-1], edges[1:]):
        m = (vals >= a) & (vals < b)
        if m.sum(): res.append(dict(range=[float(a), float(b)], n=int(m.sum()), gold=int(NG[m].sum()), hits=int(idx_hits[m].sum()), precision=round(float(idx_hits[m].sum() / (K * m.sum())), 3), recall=round(float(idx_hits[m].sum() / max(NG[m].sum(), 1)), 3), f1=round(float(f1(idx_hits, m)), 3)))
    return res


out['by_gold_size'] = bucket(NG, [0, 1, 3, 6, 11, 21, 1000])
q, c = P.load_split(R, 'dev')
qlen = np.array([len(P.tokens(q[k]['text'])) for k in qi]); clen = {k: len(P.tokens(v)) for k, v in c.items()}
out['query_words'] = dict(median=int(np.median(qlen)), p90=int(np.percentile(qlen, 90)))
cl = np.array(list(clen.values())); out['cand_words'] = dict(median=int(np.median(cl)), p90=int(np.percentile(cl, 90)))
out['by_query_words'] = bucket(qlen, [0, 2000, 4000, 8000, 16000, 10**7])
cites = np.array([q[k]['text'].count('<CITATION>') for k in qi]); out['by_placeholders'] = bucket(cites, [0, 1, 5, 11, 21, 10**5])
out['placeholders_vs_gold_spearman'] = round(float(spearmanr(cites, NG)[0]), 3)
# candidate length: share of gold candidates retrieved, per length band
g_all = [(k, x) for k in qi for x in gold[k] if x in clen]; hit = {(k, x) for k in qi for x in top(blend)[k] if x in gold[k]} if 0 else None
T = top(blend); g_len = np.array([clen[x] for _, x in g_all]); g_hit = np.array([x in T[k] for k, x in g_all])
out['gold_recall_by_cand_words'] = [dict(range=[a, b], n=int(((g_len >= a) & (g_len < b)).sum()), recall=round(float(g_hit[(g_len >= a) & (g_len < b)].mean()), 3)) for a, b in zip([0, 1000, 2000, 4000, 8000], [1000, 2000, 4000, 8000, 10**7]) if ((g_len >= a) & (g_len < b)).any()]
# popularity: how many dev queries cite a candidate; do we favour landmark cases
cited = {}
for k in qi:
    for x in gold[k]: cited[x] = cited.get(x, 0) + 1
g_pop = np.array([cited[x] for _, x in g_all])
out['gold_recall_by_times_cited'] = [dict(range=[a, b], n=int(((g_pop >= a) & (g_pop < b)).sum()), recall=round(float(g_hit[(g_pop >= a) & (g_pop < b)].mean()), 3)) for a, b in zip([1, 2, 3, 6], [2, 3, 6, 10**5]) if ((g_pop >= a) & (g_pop < b)).any()]
fp = [x for k in qi for x in T[k] if x not in gold[k]]
out['false_positives'] = dict(n=len(fp), gold_for_another_query=round(float(np.mean([x in cited for x in fp])), 3), unique=len(set(fp)))
pc = Xd[:, 11]; idx = {(a, b): i for i, (a, b) in enumerate(zip(Gd, Id))}          # column 11: pool-level mean 4-gram overlap
tpv = [pc[idx[k, x]] for k in qi for x in T[k] if x in gold[k]]; fpv = [pc[idx[k, x]] for k in qi for x in T[k] if x not in gold[k]]; allv = pc
out['pool_overlap_mean'] = dict(hits=round(float(np.mean(tpv)), 3), false_positives=round(float(np.mean(fpv)), 3), shortlist=round(float(allv.mean()), 3))
lz = Xd[:, 6]                           # column 6: per-query z of 4-gram overlap
tpz = [lz[idx[k, x]] for k in qi for x in T[k] if x in gold[k]]; fpz = [lz[idx[k, x]] for k in qi for x in T[k] if x not in gold[k]]
missed = [lz[idx[k, x]] for k in qi for x in gold[k] if (k, x) in idx and x not in T[k]]
out['overlap_z_mean'] = dict(hits=round(float(np.mean(tpz)), 3), false_positives=round(float(np.mean(fpz)), 3), missed_in_top100=round(float(np.mean(missed)), 3))
gap = Xd[:, 18]; gg = np.array([gap[idx[k, x]] for k in qi for x in gold[k] if (k, x) in idx]); out['date_gap_known_share'] = round(float((~np.isnan(gg)).mean()), 3)

# what the neural term changes
a, b = top(rank_s), T
gain = sum(len((set(b[k]) - set(a[k])) & set(gold[k])) for k in qi); loss = sum(len((set(a[k]) - set(b[k])) & set(gold[k])) for k in qi); swapped = sum(len(set(b[k]) - set(a[k])) for k in qi)
out['neural_term'] = dict(swapped=swapped, gold_gained=gain, gold_lost=loss)

# models: finer than F1@7, measured inside the lexical top 50 where every run has scores
m50 = Xd[:, 15] < 50
def auc(s, r):
    y = Yd[r].astype(bool)
    if y.sum() in (0, len(y)): return np.nan
    o = s[r].argsort().argsort() + 1
    return (o[y].sum() - y.sum() * (y.sum() + 1) / 2) / (y.sum() * (~y).sum())
res = {}; Z = {}
for run in SEVEN + ('p7c-enc-2ep', 'p7c-enc', 'p7c-lorma'):
    s = neural(run); Z[run] = s; h = hits(top(np.where(m50, s, -9.)))
    au = [auc(s, r[m50[r]]) for r in rows.values()]
    res[run] = dict(neural_f1=round(f1(h), 4), tp=int(h.sum()), auc_top50=round(float(np.nanmean(au)), 4), blend_ctx_w2=round(f1(hits(top(rank_s + .2 * s))), 4), blend_tp=int(hits(top(rank_s + .2 * s)).sum()))
res['lexical'] = dict(auc_top50=round(float(np.nanmean([auc(LEX, r[m50[r]]) for r in rows.values()])), 4)); res['ctx_ranker'] = dict(auc_top50=round(float(np.nanmean([auc(rank_s, r[m50[r]]) for r in rows.values()])), 4))
out['models'] = res
sp = lambda u, v: round(float(np.nanmean([spearmanr(u[r[m50[r]]], v[r[m50[r]]])[0] for r in rows.values()])), 3)
out['spearman'] = {f'{u}|{v}': sp(Z[u], Z[v]) for u, v in (('p7-gqa-full', 'p7-enc-full'), ('p7-gqa-full', 'p7-gqa-lorma'), ('p7-gqa-full', 'p7-gqa-kd'), ('p7-gqa-full', 'p7-gqa-partial'), ('p7-gqa-lorma', 'p7-gqa-lora'), ('p7-gqa-lorma', 'p7-gqa-ia3'))}
out['spearman_with_lexical'] = {r: sp(Z[r], LEX) for r in SEVEN}
hs = {r: hits(top(np.where(m50, Z[r], -9.))) for r in SEVEN}
out['paired_neural'] = {f'{r} - p7-gqa-full': paired(hs[r], hs['p7-gqa-full']) for r in SEVEN if r != 'p7-gqa-full'}
json.dump(out, open(sys.argv[2], 'w'), indent=1, default=float); print(json.dumps(out, indent=0, default=float))

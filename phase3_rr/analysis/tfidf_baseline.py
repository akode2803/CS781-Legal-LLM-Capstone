"""TF-IDF + logistic regression baselines for Phase 3, then Viterbi decoding. Usage: python tfidf_baseline.py <data_dir>"""
import json, sys, numpy as np, scipy.sparse as sp, warnings
from pathlib import Path
warnings.filterwarnings('ignore')
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support
root = Path(sys.argv[1])
labels = ["Fact","Issue","ArgumentPetitioner","ArgumentRespondent","Statute","Dissent","PrecedentReliedUpon","PrecedentNotReliedUpon",
          "PrecedentOverruled","RulingByLowerCourt","RatioOfTheDecision","RulingByPresentCourt","None"]
l2i = {l: i for i, l in enumerate(labels)}
def load(split):
    return [(dom, d['text'], [l2i[x] for x in d['labels']]) for dom in ('CL', 'IT') for d in json.load(open(root / f'{dom}_{split}.json'))]
tr, dv = load('train'), load('dev')
print('train docs', len(tr), 'sents', sum(len(d[1]) for d in tr), '| dev docs', len(dv), 'sents', sum(len(d[1]) for d in dv))
def flat(docs):
    T, Y, P, DOM = [], [], [], []
    for dom, txt, y in docs:
        n = len(txt)
        for i, s in enumerate(txt):
            T.append(s); Y.append(y[i]); DOM.append(dom == 'CL')
            P.append([i / max(n - 1, 1), min(i, 60) / 60, min(n - 1 - i, 60) / 60] + [float(b / 10 <= i / max(n - 1, 1) < (b + 1) / 10) for b in range(10)])
    return T, np.array(Y), np.array(P), np.array(DOM, float)
def shift(M, docs, k):
    out, o = [], 0
    for _, txt, _ in docs:
        n = len(txt); idx = np.arange(n) + k; ok = (idx >= 0) & (idx < n)
        out.append(M[o:o + n][np.clip(idx, 0, n - 1)].multiply(ok[:, None])); o += n
    return sp.vstack(out).tocsr()
Ttr, ytr, Ptr, Dtr = flat(tr); Tdv, ydv, Pdv, Ddv = flat(dv)
vw = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, max_features=20000)
Xtr = vw.fit_transform(Ttr); Xdv = vw.transform(Tdv)
def score(y, p):
    _, _, f, s = precision_recall_fscore_support(y, p, labels=range(13), zero_division=0)
    pres = [i for i in range(13) if s[i] > 0 or (p == i).any()]
    return round(float(f[pres].mean()), 4), round(float(f.mean()), 4)
def fit(Atr, C=1.0):
    return LogisticRegression(C=C, max_iter=200, class_weight='balanced').fit(Atr, ytr)
print('majority class (Fact), (present, all13):', score(ydv, np.zeros_like(ydv)))
print('position + domain only:', score(ydv, fit(np.hstack([Ptr, Dtr[:, None]])).predict(np.hstack([Pdv, Ddv[:, None]]))))
print('TF-IDF sentence only:', score(ydv, fit(Xtr).predict(Xdv)))
S = lambda X, P_, D_: sp.hstack([X, sp.csr_matrix(P_), sp.csr_matrix(D_[:, None])]).tocsr()
print('TF-IDF + position + domain:', score(ydv, fit(S(Xtr, Ptr, Dtr)).predict(S(Xdv, Pdv, Ddv))))
ks = (-1, 1)
A = sp.hstack([Xtr] + [shift(Xtr, tr, k) for k in ks] + [sp.csr_matrix(Ptr), sp.csr_matrix(Dtr[:, None])]).tocsr()
B = sp.hstack([Xdv] + [shift(Xdv, dv, k) for k in ks] + [sp.csr_matrix(Pdv), sp.csr_matrix(Ddv[:, None])]).tocsr()
m = fit(A, 0.3); print('TF-IDF + neighbours (-1,1) + position, C=0.3:', score(ydv, m.predict(B)))
# count-based Viterbi on top (this made results worse in our run)
P = np.full((B.shape[0], 13), 1e-9); P[:, m.classes_] = m.predict_proba(B)
prior = np.bincount(ytr, minlength=13) + 1.0; prior /= prior.sum()
T = np.ones((13, 13)) * 0.5
for _, _, y in tr:
    for a, b in zip(y[:-1], y[1:]): T[a, b] += 1
T /= T.sum(1, keepdims=True); first = np.ones(13)
for _, _, y in tr: first[y[0]] += 1
first /= first.sum()
def viterbi(logE, lam):
    n = len(logE); V = np.zeros((n, 13)); bp = np.zeros((n, 13), int); V[0] = logE[0] + lam * np.log(first)
    for t in range(1, n):
        sc = V[t - 1][:, None] + lam * np.log(T); bp[t] = sc.argmax(0); V[t] = sc.max(0) + logE[t]
    path = [V[-1].argmax()]
    for t in range(n - 1, 0, -1): path.append(bp[t][path[-1]])
    return path[::-1]
logE = np.log(P) - 0.5 * np.log(prior)
for lam in (0.3, 0.5, 1.0, 2.0):
    o, pred = 0, []
    for _, txt, _ in dv:
        pred += viterbi(logE[o:o + len(txt)], lam); o += len(txt)
    print(f'Viterbi weight={lam}:', score(ydv, np.array(pred)))

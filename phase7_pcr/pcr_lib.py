"""Phase 7 (prior case retrieval): data loading, Micro-F1@7, lexical retrieval. CPU only; the neural runs build on this."""
import json, re, zipfile
from pathlib import Path
import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer


def course_dir():
    """MyDrive/legal-llm-course as synced by Google Drive for Desktop (first account found); set COURSE_DRIVE to override."""
    import glob, os
    if os.environ.get('COURSE_DRIVE'): return os.environ['COURSE_DRIVE']
    hits = sorted(glob.glob(os.path.expanduser('~/Library/CloudStorage/GoogleDrive-*/My Drive/legal-llm-course')))
    if not hits: raise FileNotFoundError('set COURSE_DRIVE to the legal-llm-course folder')
    return hits[0]

K = 7
SPLITS = {'train': ('Train/train.json', 'Train/train_candidates.json'), 'dev': ('Dev/dev.json', 'Dev/dev_candidates.json'), 'test': ('Test/test.json', 'Test/test_candidates.json')}


def load_split(root, split):
    q, c = (json.load(open(Path(root) / f, encoding='utf-8')) for f in SPLITS[split])
    return ({k: dict(text=' '.join(v['text']) if isinstance(v['text'], list) else v['text'], gold=v.get('relevant_candidates')) for k, v in q.items()},
            {k: ' '.join(v['text']) if isinstance(v['text'], list) else v['text'] for k, v in c.items()})


def micro_prf(pred, gold, k=K):
    """pred, gold: {query id: list of case ids}. Micro over all queries; only the first k predictions count, the gold sets stay whole."""
    tp = sum(len(set(pred[q][:k]) & set(g)) for q, g in gold.items()); n_pred = sum(len(pred[q][:k]) for q in gold); n_gold = sum(len(g) for g in gold.values())
    p, r = tp / max(n_pred, 1), tp / max(n_gold, 1)
    return dict(f1=2 * p * r / max(p + r, 1e-9), precision=p, recall=r, tp=tp)


DATE = re.compile(r'DATE OF JUDGMENT:?\s*(\d{1,2})[/.-](\d{1,2})[/.-](\d{4})')


def judgment_year(text):
    m = DATE.search(text[:1500])
    return (int(m[3]) + (int(m[2]) - 1) / 12 + (int(m[1]) - 1) / 372) if m else None


class BM25:
    """Okapi BM25 over the candidate pool (sparse), on word n-grams."""

    def __init__(self, docs, ngram=(1, 1), k1=1.2, b=0.75, min_df=1, max_df=1.0, stop=None):
        self.vec = CountVectorizer(lowercase=True, token_pattern=r"[a-z][a-z]+|\d+", ngram_range=ngram, min_df=min_df, max_df=max_df, stop_words=stop)
        X = self.vec.fit_transform(docs).astype(np.float32).tocsr(); n = X.shape[0]
        df = np.asarray((X > 0).sum(0)).ravel(); self.idf = np.log(1 + (n - df + .5) / (df + .5)).astype(np.float32)
        dl = np.asarray(X.sum(1)).ravel(); norm = k1 * (1 - b + b * dl / dl.mean())
        Xc = X.tocoo(); Xc.data = Xc.data * (k1 + 1) / (Xc.data + norm[Xc.row]); self.X = Xc.tocsr()

    def scores(self, queries, binary=True):
        """(n queries, n docs). binary: a query term counts once, the usual choice when the query is a whole document."""
        Q = self.vec.transform(queries).astype(np.float32)
        if binary: Q.data[:] = 1
        return np.asarray((Q.multiply(self.idf) @ self.X.T).todense())


def top_k(scores, q_ids, c_ids, k=K, exclude_self=True):
    out = {}; c_ids = np.array(c_ids)
    for i, q in enumerate(q_ids):
        s = scores[i].copy()
        if exclude_self: s[c_ids == q] = -np.inf
        out[q] = c_ids[np.argsort(-s, kind='stable')[:k]].tolist()
    return out


def write_submission(pred, q_ids, pool, path):
    for q in q_ids:
        assert len(pred[q]) == K == len(set(pred[q])) and all(c in pool for c in pred[q]), q
    with zipfile.ZipFile(path, 'w', zipfile.ZIP_DEFLATED) as z:
        z.writestr('prediction.json', json.dumps({q: pred[q] for q in q_ids}, indent=1))
    return path


# ------------------------------------------------------------------------------ fast word n-gram overlap (hashed shingles)
TOKEN = re.compile(r"[a-z][a-z]+|\d+")


def tokens(text):
    """Lower-cased words and numbers; the <NAME> / <ORG> / <CITATION> masks are dropped."""
    return TOKEN.findall(re.sub(r'<[A-Z]+>', ' ', text).lower())


def shingles(toks, n):
    """Distinct 64-bit hashes of the word n-grams of one document."""
    ids = np.array([hash(t) for t in toks], dtype=np.int64)
    if len(ids) < n: return np.zeros(0, np.int64)
    h = np.zeros(len(ids) - n + 1, np.int64)
    with np.errstate(over='ignore'):
        for i in range(n): h = h * np.int64(1000003) + ids[i:len(ids) - n + 1 + i]
    return np.unique(h)


class Shingles:
    """Pool index: score(query, doc) = sum of idf over n-grams they share, optionally divided by doc_len ** b."""

    def __init__(self, docs_tok, n, max_df=0.05):
        sh = [shingles(t, n) for t in docs_tok]; self.n = n; self.N = len(sh)
        allh = np.concatenate(sh); self.vocab, df = np.unique(allh, return_counts=True)
        keep = (df >= 1) & (df <= max(2, max_df * self.N)); self.vocab, df = self.vocab[keep], df[keep]
        self.idf = np.log(1 + self.N / df).astype(np.float32)
        rows, cols = [], []
        for i, s in enumerate(sh):
            j = np.searchsorted(self.vocab, s); j[j == len(self.vocab)] = 0; m = self.vocab[j] == s
            rows.append(np.full(m.sum(), i)); cols.append(j[m])
        self.X = sparse.csr_matrix((np.ones(sum(map(len, rows)), np.float32), (np.concatenate(rows), np.concatenate(cols))), shape=(self.N, len(self.vocab)))
        self.size = np.asarray(self.X.sum(1)).ravel()

    def scores(self, queries_tok, b=0.0):
        rows, cols = [], []
        for i, t in enumerate(queries_tok):
            s = shingles(t, self.n); j = np.searchsorted(self.vocab, s); j[j == len(self.vocab)] = 0; m = self.vocab[j] == s
            rows.append(np.full(m.sum(), i)); cols.append(j[m])
        Q = sparse.csr_matrix((np.ones(sum(map(len, rows)), np.float32), (np.concatenate(rows), np.concatenate(cols))), shape=(len(queries_tok), len(self.vocab)))
        S = np.asarray((Q.multiply(self.idf) @ self.X.T).todense())
        return S / np.maximum(self.size, 1)[None] ** b

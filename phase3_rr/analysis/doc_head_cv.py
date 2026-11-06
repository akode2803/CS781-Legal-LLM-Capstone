"""Grouped k-fold CV of document-level heads on TF-IDF-SVD sentence vectors (no language model).
Usage: python doc_head_cv.py <data_dir> [epochs]"""
import json, sys, time, numpy as np, torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import doclib as DL
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
torch.set_num_threads(8)
root = Path(sys.argv[1]); EP = int(sys.argv[2]) if len(sys.argv) > 2 else 20
labels = ["Fact","Issue","ArgumentPetitioner","ArgumentRespondent","Statute","Dissent","PrecedentReliedUpon","PrecedentNotReliedUpon",
          "PrecedentOverruled","RulingByLowerCourt","RatioOfTheDecision","RulingByPresentCourt","None"]
l2i = {l: i for i, l in enumerate(labels)}
raw = [(d['text'], [l2i[x] for x in d['labels']], di) for dom, di in (('CL', 1), ('IT', 0)) for sp_ in ('train', 'dev')
       for d in json.load(open(root / f'{dom}_{sp_}.json'))]
texts = [s for t, _, _ in raw for s in t]
X = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, max_features=30000).fit_transform(texts)
Z = TruncatedSVD(256, random_state=0).fit_transform(X); Z /= (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-8)
docs, o = [], 0
for t, y, di in raw:
    docs.append(DL.make_doc(Z[o:o + len(t)], y, di)); o += len(t)
print('docs', len(docs), 'sentences', len(texts), flush=True)
M = lambda **k: dict(hidden=128, layers=1, dropout=0.3, in_dropout=0.3, **k)
cfgs = {'sentence MLP (+pos)': dict(model=M(bilstm=False)), 'BiLSTM doc': dict(model=M()), 'BiLSTM + CRF': dict(model=M(crf=True)),
        'BiLSTM + shift aux': dict(model=M(shift=True)), 'BiLSTM + CRF + shift aux': dict(model=M(crf=True, shift=True))}
for name, cfg in cfgs.items():
    cfg['eval_every'] = 5; t0 = time.time()
    c = DL.cross_validate(docs, cfg, EP, torch.device('cpu'), k=5, seeds=(0,))[0]
    print(f'{name:28s}', ' '.join(f'ep{e}:{f:.3f}' for e, f, _ in c), f'| {time.time()-t0:.0f}s', flush=True)

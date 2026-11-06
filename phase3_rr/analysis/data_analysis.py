"""Phase 3 data facts: label share, lengths, truncation, label persistence. Usage: python data_analysis.py <data_dir> [tokenizer.json]"""
import json, sys, collections, numpy as np
from pathlib import Path
from sklearn.metrics import precision_recall_fscore_support
root = Path(sys.argv[1]); tokp = sys.argv[2] if len(sys.argv) > 2 else str(Path(__file__).resolve().parents[2] / 'common/tokenizer/tokenizer.json')
labels = ["Fact","Issue","ArgumentPetitioner","ArgumentRespondent","Statute","Dissent","PrecedentReliedUpon","PrecedentNotReliedUpon",
          "PrecedentOverruled","RulingByLowerCourt","RatioOfTheDecision","RulingByPresentCourt","None"]
l2i = {l: i for i, l in enumerate(labels)}
D = {s: {dom: json.load(open(root / f'{dom}_{s}.json')) for dom in ('CL', 'IT')} for s in ('train', 'dev')}
from tokenizers import Tokenizer
tok = Tokenizer.from_file(tokp)
def f1(y, p):
    _, _, f, s = precision_recall_fscore_support(y, p, labels=range(13), zero_division=0)
    pres = [i for i in range(13) if s[i] > 0 or (np.array(p) == i).any()]; return round(float(f[pres].mean()), 3)
print('== label share (%) train / dev, by domain ==')
for dom in ('CL', 'IT'):
    for s in ('train', 'dev'):
        c = collections.Counter(l for d in D[s][dom] for l in d['labels']); n = sum(c.values())
        print(dom, s, n, 'sents |', ' '.join(f'{l[:8]}={100*c[l]/n:.1f}' for l in labels))
print('\n== document length (sentences) ==')
for dom in ('CL', 'IT'):
    for s in ('train', 'dev'):
        L = [len(d['text']) for d in D[s][dom]]; print(dom, s, 'docs', len(L), 'min/median/max', min(L), int(np.median(L)), max(L))
print('\n== sentence token length (runs cap sentences at 63 tokens + <eos>) ==')
for dom in ('CL', 'IT'):
    T = [len(e.ids) for d in D['train'][dom] for e in tok.encode_batch(d['text'])]
    print(dom, 'mean', round(np.mean(T), 1), 'median', int(np.median(T)), 'p90', int(np.percentile(T, 90)), '%>63 tokens:', round(100 * np.mean(np.array(T) > 63), 1))
print('\n== label persistence and sequence upper bounds (dev) ==')
for dom in ('CL', 'IT', 'ALL'):
    ds = ('CL', 'IT') if dom == 'ALL' else (dom,)
    docs = [d for x in ds for d in D['dev'][x]]; tr = [d for x in ds for d in D['train'][x]]
    same = np.mean([a == b for d in docs for a, b in zip(d['labels'][:-1], d['labels'][1:])])
    y = [l2i[l] for d in docs for l in d['labels']]
    prev = [l2i[d['labels'][i - 1]] if i else 0 for d in docs for i in range(len(d['labels']))]
    print(dom, 'P(same label as previous)=%.3f' % same, '| copy-GOLD-previous macro-F1 (an oracle, not reachable)', f1(y, prev))

import sys, json, time, numpy as np
from pathlib import Path as _P
REPO = _P(__file__).resolve().parents[2]
import sys
sys.path[:0] = [str(REPO/'phase4_cjpe'), str(REPO/'phase3_rr'), str(REPO/'common')]
import cjpe_lib as J
from tokenizers import Tokenizer
root=sys.argv[1]; t0=time.time()
DATA=J.load_all(root+'/Phase_4_CJPE')
print('loaded in',round(time.time()-t0),'s |',{k:len(v) for k,v in DATA.items()},'| test labels kept:',any(d['label'] is not None for d in DATA['test']))
tok=Tokenizer.from_file(str(REPO/'common/tokenizer/tokenizer.json'))
t0=time.time()
for k in ('dev','test'): J.tokenize_docs(DATA[k],tok,40000)
sub=DATA['train'][:3000]; J.tokenize_docs(sub,tok,40000)
print('tokenised dev+test+3000 train in',round(time.time()-t0),'s')
for k,v in (('dev',DATA['dev']),('test',DATA['test']),('train[:3000]',sub)):
    n=np.array([len(d['tok']) for d in v]); ch=np.array([len(J.chunk_spans(len(d['tok']),512,512)) for d in v])
    chars=np.array([len(d['text']) for d in v])
    print(f'{k:13s} tokens mean/median/p90/max {int(n.mean())}/{int(np.median(n))}/{int(np.percentile(n,90))}/{int(n.max())} | chars per token {chars.sum()/n.sum():.2f} | 512-chunks mean/p90 {ch.mean():.1f}/{int(np.percentile(ch,90))} | cases fully covered by 8 chunks: {100*np.mean(ch<=8):.0f}%')
print('label-1 share: train',round(np.mean([d['label'] for d in DATA['train']]),3),'| dev',round(np.mean([d['label'] for d in DATA['dev']]),3))
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
TR=[DATA['train'][i] for i in np.random.RandomState(0).permutation(len(DATA['train']))[:8000]]
ytr=np.array([d['label'] for d in TR]); ydv=np.array([d['label'] for d in DATA['dev']])
def part(t,w):
    n=len(t); a,b={'whole':(0,1),'first third':(0,1/3),'middle third':(1/3,2/3),'last third':(2/3,1)}[w]; return t[int(a*n):int(b*n)]
print('majority-class macro-F1 on dev:',round(J.macro_scores(ydv,np.zeros_like(ydv))['macro_f1'],4))
for w in ('whole','first third','middle third','last third'):
    t0=time.time(); vec=TfidfVectorizer(ngram_range=(1,2),min_df=3,max_features=100000,sublinear_tf=True)
    Xtr=vec.fit_transform([part(d['text'],w) for d in TR]); Xdv=vec.transform([part(d['text'],w) for d in DATA['dev']])
    best=None
    for C in (0.3,1,3):
        m=LogisticRegression(C=C,max_iter=400,class_weight='balanced').fit(Xtr,ytr); p=m.predict_proba(Xdv)[:,1]
        s=J.macro_scores(ydv,(p>=.5).astype(int)); thr,ft=J.best_threshold(ydv,p)
        if best is None or s['macro_f1']>best[1]: best=(C,s['macro_f1'],s['macro_precision'],s['macro_recall'],ft,thr)
    print(f'{w:13s} best C={best[0]} dev macro-F1 {best[1]:.4f} (P {best[2]:.3f} R {best[3]:.3f}) | tuned {best[4]:.4f} @ {best[5]:.2f} | {time.time()-t0:.0f}s',flush=True)

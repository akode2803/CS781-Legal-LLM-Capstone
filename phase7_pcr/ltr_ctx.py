"""Adds citing-context features to the cached ranker features: overlap between candidates and only the query sentences near a <CITATION> placeholder. The placeholder marks where a citation was, not which case it was.
usage: python ltr_ctx.py <features pickle> <out pickle>"""
import pickle, re, sys
import numpy as np, pcr_lib as P
from ltr_features import R
CFG=[(3,1),(4,1),(3,2),(4,2)]
def ctx(t,w):
    s=re.split(r'(?<=[.?!])\s+',t); h=[i for i,x in enumerate(s) if '<CITATION>' in x]
    return ' '.join(s[j] for j in sorted({j for i in h for j in range(max(0,i-w),min(len(s),i+w+1))}))
def cols(split,G,I):
    q,c=P.load_split(R,split); qi,ci=list(q),list(c); pos={k:j for j,k in enumerate(ci)}; qp={k:i for i,k in enumerate(qi)}
    rows=np.array([qp[k] for k in G]); cc=np.array([pos[k] for k in I]); out=[]; docs=[P.tokens(c[k]) for k in ci]
    for n in (3,4):
        sh=P.Shingles(docs,n,.05)
        for m,w in CFG:
            if m!=n: continue
            v=np.log1p(np.maximum(sh.scores([P.tokens(ctx(q[k]['text'],w)) for k in qi],.6)[rows,cc],0)).astype(np.float64); zc=v.copy()
            for k in np.unique(G):
                b=G==k; zc[b]=(v[b]-v[b].mean())/(v[b].std()+1e-9)
            out+=[v,zc]
    return np.c_[tuple(out)]
if __name__=='__main__':
    F=pickle.load(open(sys.argv[1],'rb')); O={}
    for s,(X,Y,G,I,g,qi) in F.items():
        O[s]=(np.c_[X,cols(s,G,I)],Y,G,I,g,qi); print(s,O[s][0].shape,flush=True)
    pickle.dump(O,open(sys.argv[2],'wb'))

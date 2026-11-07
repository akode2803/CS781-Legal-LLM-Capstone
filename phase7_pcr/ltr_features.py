"""Learning to rank on content features only: overlap scores at several n-gram sizes, document lengths, date gap. Feature builder."""
import os, numpy as np, pcr_lib as P
from sklearn.ensemble import HistGradientBoostingClassifier as GB
R=os.path.expanduser('~/Library/CloudStorage/GoogleDrive-ACCOUNT/My Drive/legal-llm-course/Phase_7_PCR')
CFG=[(3,.6),(4,.6),(5,.4),(6,.6),(8,.5)]; TOP=100
def feats(split):
    q,c=P.load_split(R,split); qi,ci=list(q),list(c); pos={k:j for j,k in enumerate(ci)}
    qt=[P.tokens(q[k]['text']) for k in qi]; dt=[P.tokens(c[k]) for k in ci]
    M=[]
    for n,b in CFG:
        sh=P.Shingles(dt,n,.05); M.append(sh.scores(qt,b).astype(np.float32)); raw=None
    qh=np.array([P.judgment_year(q[k]['text']) or np.nan for k in qi]); ch=np.array([P.judgment_year(c[k]) or np.nan for k in ci])
    base=M[1].copy(); base[ch[None]>qh[:,None]]=-1e9
    for i,k in enumerate(qi):
        if k in pos: base[i,pos[k]]=-2e9
    L=[np.log1p(m) for m in M]; pop=[l.mean(0) for l in L]; ql=np.log1p([len(t) for t in qt]); cl=np.log1p([len(t) for t in dt])
    X,Y,G,ids=[],[],[],[]
    for i,k in enumerate(qi):
        o=np.argsort(-base[i],kind='stable')[:TOP]; g=set(q[k]['gold'] or [])
        zs=[(l[i,o]-l[i,o].mean())/(l[i,o].std()+1e-9) for l in L]
        gap=qh[i]-ch[o]
        X.append(np.c_[tuple(l[i,o] for l in L)+tuple(zs)+tuple(p[o] for p in pop)+(np.arange(TOP),np.full(TOP,ql[i]),cl[o],gap)])
        Y.append([ci[j] in g for j in o]); G+= [k]*TOP; ids+=[ci[j] for j in o]
    return np.vstack(X),np.concatenate(Y).astype(int),np.array(G),np.array(ids),{k:list(q[k]['gold'] or []) for k in qi},qi

if __name__ == '__main__':
    import pickle, sys
    pickle.dump({s: feats(s) for s in ('train', 'dev', 'test')}, open(sys.argv[1], 'wb'))

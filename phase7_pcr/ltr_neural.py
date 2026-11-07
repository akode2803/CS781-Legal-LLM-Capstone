"""Learned ranker with out-of-fold reranker scores as input features. Train rows: lexical top 40 with an out-of-fold score. Dev / test rows: lexical top 50, scored by rerankers trained on train only.
usage: python ltr_neural.py <features pickle> <run> [<run> ...]"""
import json, os, pickle, sys
import numpy as np, pcr_lib as P
from sklearn.ensemble import HistGradientBoostingClassifier as GB
RUNS=os.path.expanduser('~/Library/CloudStorage/GoogleDrive-ACCOUNT/My Drive/legal-llm-course/runs/phase7_pcr')
load=lambda p:{tuple(k.split('|')):v for k,v in json.load(open(p)).items()}
z=lambda v:(v-v.mean())/(v.std()+1e-9)
def col(G,I,sc):
    o=np.full(len(G),np.nan)
    for i,(a,b) in enumerate(zip(G,I)):
        o[i]=sc.get((a,b),np.nan)
    for k in np.unique(G):
        m=(G==k)&~np.isnan(o); o[m]=z(o[m]) if m.sum()>1 else 0
    return o
def fit(X,Y,Xs,seeds=5):
    m=[GB(max_iter=150,learning_rate=.05,max_leaf_nodes=15,min_samples_leaf=50,l2_regularization=1.,random_state=s).fit(X,Y) for s in range(seeds)]
    return [np.mean([x.predict_proba(e)[:,1] for x in m],0) for e in Xs]
def rank(G,I,s): return {k:list(I[G==k][np.argsort(-s[G==k],kind='stable')]) for k in dict.fromkeys(G)}
if __name__=='__main__':
    F=pickle.load(open(sys.argv[1],'rb')); runs=sys.argv[2:]
    Xt,Yt,Gt,It,_,_=F['train']; Xd,Yd,Gd,Id,gold,qi=F['dev']
    tn=[col(Gt,It,{k:v for f in range(3) for k,v in load(f'{RUNS}/crossfit/{r}/fold{f}.json').items()}) for r in runs]
    dn=[col(Gd,Id,load(f'{RUNS}/{r}/dev_scores.json')) for r in runs]
    keep=(Xt[:,15]<40)&~np.isnan(np.c_[tuple(tn)]).any(1); kd=Xd[:,15]<50
    print('train rows',keep.sum(),'pos',Yt[keep].sum())
    ev=lambda s:P.micro_prf(rank(Gd[kd],Id[kd],s),gold)
    a,=fit(Xt[keep],Yt[keep],[Xd[kd]]); r=ev(a); print('ranker, same rows, no neural',round(r['f1'],4),r['tp'],flush=True)
    b,=fit(np.c_[(Xt[keep],)+tuple(t[keep] for t in tn)],Yt[keep],[np.c_[(Xd[kd],)+tuple(d[kd] for d in dn)]]); r=ev(b); print('ranker + neural features',runs,round(r['f1'],4),r['tp'],flush=True)
    lo=lambda p:np.log(p.clip(1e-6,1-1e-6)/(1-p.clip(1e-6,1-1e-6)))
    def zq(s):
        o=s.copy()
        for k in qi: m=Gd[kd]==k; o[m]=z(s[m])
        return o
    for w in (.1,.2,.3):
        r=ev(zq(lo(a))+w*sum(d[kd] for d in dn)/len(dn)); print('additive w',w,round(r['f1'],4),r['tp'])

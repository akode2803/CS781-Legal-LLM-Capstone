"""Final entries: ranker fit on train + dev (lexical features only), plus 0.2 z(neural) from rerankers retrained on train + dev. Dev cannot score these, so the dev line uses the train-only ranker and rerankers.
usage: python ltr_final.py <features pickle> <pickle from ltr_probe.py>"""
import json, pickle, sys
import numpy as np, pcr_lib as P
from ltr_neural import RUNS, GB, z, load
F=pickle.load(open(sys.argv[1],'rb')); d=pickle.load(open(sys.argv[2],'rb'))
Xt,Yt=F['train'][:2]; Xd,Yd=F['dev'][:2]; Xs,_,Gs,Is,_,ti=F['test']
X,Y=np.r_[Xt,Xd],np.r_[Yt,Yd]
m=[GB(max_iter=150,learning_rate=.05,max_leaf_nodes=15,min_samples_leaf=50,l2_regularization=1.,random_state=s).fit(X,Y) for s in range(5)]
T=np.mean([np.log(p/(1-p)) for p in (x.predict_proba(Xs)[:,1].clip(1e-6,1-1e-6) for x in m)],0)
def neural(G,I,qs,paths):
    """Mean of per-query z scores; candidates without a score get -2."""
    tot=np.zeros(len(G))
    for p in paths:
        sc=load(p); o=np.array([sc.get((a,b),np.nan) for a,b in zip(G,I)])
        for k in qs:
            q=G==k; v=o[q]; ok=~np.isnan(v); v[ok]=z(v[ok]); v[~ok]=-2.; o[q]=v
        tot+=o
    return tot/len(paths)
def order(G,I,qs,s,n,w=.2):
    f=s.copy()
    for k in qs: f[G==k]=z(s[G==k])
    f=f+w*n
    return {k:[str(x) for x in I[G==k][np.argsort(-f[G==k],kind='stable')][:7]] for k in qs}
fin=lambda r:f'{RUNS}/final/{r}/test_scores.json'; old=lambda r:f'{RUNS}/{r}/test_scores.json'; dev=lambda r:f'{RUNS}/{r}/dev_scores.json'
for r in (['p7-enc-full'],['p7-gqa-lorma'],['p7-enc-full','p7-gqa-lorma']):
    v=P.micro_prf(order(d['Gd'],d['Id'],d['qi'],d['D'],neural(d['Gd'],d['Id'],d['qi'],[dev(x) for x in r])),d['gold']); print('dev, train-only',r,round(v['f1'],4),v['tp'])
ENTRIES={'p7-final-ranker':[],'p7-final-ranker-oldlorma':[old('p7-gqa-lorma')],'p7-final-enc':[fin('p7-enc-full')],'p7-final-lorma':[fin('p7-gqa-lorma')],'p7-final-enc-lorma':[fin('p7-enc-full'),fin('p7-gqa-lorma')]}
prev=json.load(__import__('zipfile').ZipFile('submissions/p7-ltr-gqa-lorma.zip').open('prediction.json'))
for name,paths in ENTRIES.items():
    p=order(Gs,Is,ti,T,neural(Gs,Is,ti,paths) if paths else np.zeros(len(Gs)))
    same=np.mean([len(set(p[k])&set(prev[k]))/7 for k in ti])
    print(name,P.write_submission(p,ti,set(d['pool']),f'submissions/{name}.zip'),'overlap with best entry',round(same,3),flush=True)

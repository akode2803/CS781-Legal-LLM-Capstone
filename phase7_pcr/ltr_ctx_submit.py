"""Test zips: train-only ranker with citing-context features, alone and with z(neural) from train-only rerankers.
usage: python ltr_ctx_submit.py <ctx features pickle> <name>=<run>[,<run>]:<w> ...   (a bare <name> means ranker alone)"""
import pickle, sys
import numpy as np, pcr_lib as P
from ltr_neural import RUNS, fit, load, z
F=pickle.load(open(sys.argv[1],'rb')); Xt,Yt=F['train'][:2]; Xs,_,Gs,Is,_,ti=F['test']; pool=set(P.load_split(__import__('ltr_features').R,'test')[1])
p=fit(Xt,Yt,[Xs])[0].clip(1e-6,1-1e-6); s=np.log(p/(1-p))
for k in ti: s[Gs==k]=z(s[Gs==k])
def neural(run):
    sc=load(f'{RUNS}/{run}/test_scores.json'); o=np.array([sc.get((a,b),np.nan) for a,b in zip(Gs,Is)])
    for k in ti:
        m=Gs==k; v=o[m]; ok=~np.isnan(v); v[ok]=z(v[ok]); v[~ok]=-2.; o[m]=v
    return o
for e in sys.argv[2:]:
    name,_,spec=e.partition('='); f=s
    if spec:
        runs,w=spec.split(':'); f=s+float(w)*np.mean([neural(r) for r in runs.split(',')],0)
    pred={k:[str(x) for x in Is[Gs==k][np.argsort(-f[Gs==k],kind='stable')][:7]] for k in ti}
    print(P.write_submission(pred,ti,pool,f'submissions/{name}.zip'),flush=True)

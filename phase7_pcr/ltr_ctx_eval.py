"""Dev comparison: ranker with and without citing-context features, alone and with 0.2 z(neural). usage: python ltr_ctx_eval.py <ctx features pickle> <run> [<run> ...]"""
import pickle, sys
import numpy as np, pcr_lib as P
from ltr_neural import RUNS, fit, load, z
F=pickle.load(open(sys.argv[1],'rb')); Xt,Yt=F['train'][:2]; Xd,_,Gd,Id,gold,qi=F['dev']
def neural(run):
    t=0
    for r in run.split(','):
        sc=load(f'{RUNS}/{r}/dev_scores.json'); o=np.array([sc.get((a,b),np.nan) for a,b in zip(Gd,Id)])
        for k in qi:
            m=Gd==k; v=o[m]; ok=~np.isnan(v); v[ok]=z(v[ok]); v[~ok]=-2.; o[m]=v
        t=t+o
    return t/len(run.split(','))
def ev(s):
    r=P.micro_prf({k:list(Id[Gd==k][np.argsort(-s[Gd==k],kind='stable')][:7]) for k in qi},gold); return round(r['f1'],4),r['tp']
def zq(p):
    s=np.log(p.clip(1e-6,1-1e-6)/(1-p.clip(1e-6,1-1e-6)))
    for k in qi: s[Gd==k]=z(s[Gd==k])
    return s
for name,c in (('citing context',slice(0,27)),):
    s=zq(fit(Xt[:,c],Yt,[Xd[:,c]])[0]); print(name,'alone',ev(s),flush=True)
    for r in sys.argv[2:]: print(' ',r,*[(w,ev(s+w*neural(r))) for w in (.1,.2,.3)],flush=True)

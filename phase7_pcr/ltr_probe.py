"""Learning to rank on content features only: overlap scores at several n-gram sizes, document lengths, date gap. Fit on train, scored on dev."""
import os, numpy as np, pcr_lib as P
from sklearn.ensemble import HistGradientBoostingClassifier as GB
R=P.course_dir()+'/Phase_7_PCR'
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
import json, sys
B=os.path.dirname(R); RUNS=B+'/runs/phase7_pcr'
Xt,Yt,Gt,_,_,_=feats('train'); Xd,Yd,Gd,Id,gold,qi=feats('dev'); Xs,_,Gs,Is,_,ti=feats('test')
pool=list(P.load_split(R,'test')[1])
POP=list(range(10,15)); keep=[j for j in range(Xt.shape[1]) if j not in POP]
def fit(cols):
    m=[GB(max_iter=150,learning_rate=.05,max_leaf_nodes=15,min_samples_leaf=50,l2_regularization=1.,random_state=s).fit(Xt[:,cols],Yt) for s in range(5)]
    logit=lambda X:np.mean([np.log(p/(1-p)) for p in (x.predict_proba(X[:,cols])[:,1].clip(1e-6,1-1e-6) for x in m)],0)
    return logit(Xd),logit(Xs)
def z(v): return (v-v.mean())/(v.std()+1e-9)
def neural(G,I,qs,run,split):
    sc={tuple(k.split('|')):v for k,v in json.load(open(f'{RUNS}/{run}/{split}_scores.json')).items()}; out=np.full(len(G),np.nan)
    for i,(a,b) in enumerate(zip(G,I)):
        if (a,b) in sc: out[i]=sc[a,b]
    for k in qs:
        m=(G==k); v=out[m]; ok=~np.isnan(v); v[ok]=z(v[ok]); v[~ok]=-2.; out[m]=v
    return out
def rank(G,I,qs,s): return {k:list(I[G==k][np.argsort(-s[G==k],kind='stable')]) for k in qs}
def ev(s): return P.micro_prf(rank(Gd,Id,qi,s),gold)
def zq(G,qs,s):
    o=s.copy()
    for k in qs: o[G==k]=z(s[G==k])
    return o
allc=list(range(Xt.shape[1]))
for name,cols in (('all features',allc),('no pool-level feature',keep)):
    d,t=fit(cols); r=ev(d); print(name,round(r['f1'],4),r['tp'],flush=True)
    if name=='all features': D,T=d,t
    else: D2,T2=d,t
for name,(d,t) in (('all',(D,T)),('nopop',(D2,T2))):
    for run in ('p7-enc-full','p7-gqa-full','p7-gqa-lorma','p7-gqa-kd'):
        nd=neural(Gd,Id,qi,run,'dev'); print(name,run,*[f"{w}:{ev(zq(Gd,qi,d)+w*nd)['f1']:.4f}" for w in (0,.1,.2,.3,.5,.75,1)],flush=True)
np.savez('/tmp/ltr_cols.npz',x=0) if False else None
import pickle; pickle.dump(dict(D=D,T=T,D2=D2,T2=T2,Gd=Gd,Id=Id,Gs=Gs,Is=Is,qi=qi,ti=ti,gold=gold,pool=pool),open(sys.argv[1],'wb'))

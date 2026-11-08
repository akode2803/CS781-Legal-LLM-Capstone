"""Writes test zips: z(learned ranker) + 0.2 z(neural). usage: python ltr_submit.py <pickle from ltr_probe.py>"""
import json, os, pickle, sys
import numpy as np, pcr_lib as P
d=pickle.load(open(sys.argv[1],'rb')); R=P.course_dir()+'/runs/phase7_pcr'
z=lambda v:(v-v.mean())/(v.std()+1e-9)
def neural(G,I,qs,run,split):
    sc={tuple(k.split('|')):v for k,v in json.load(open(f'{R}/{run}/{split}_scores.json')).items()}; out=np.full(len(G),np.nan)
    for i,(a,b) in enumerate(zip(G,I)):
        if (a,b) in sc: out[i]=sc[a,b]
    for k in qs:
        m=G==k; v=out[m]; ok=~np.isnan(v); v[ok]=z(v[ok]); v[~ok]=-2.; out[m]=v
    return out
def blend(G,I,qs,s,run,split,w=.2):
    f=s.copy()
    for k in qs: f[G==k]=z(s[G==k])
    if run: f=f+w*neural(G,I,qs,run,split)
    return {k:list(I[G==k][np.argsort(-f[G==k],kind='stable')]) for k in qs}
for name,run in (('p7-ltr-lexical',None),('p7-ltr-enc-full','p7-enc-full'),('p7-ltr-gqa-full','p7-gqa-full'),('p7-ltr-gqa-lorma','p7-gqa-lorma'),('p7-ltr-gqa-kd','p7-gqa-kd')):
    dv=P.micro_prf(blend(d['Gd'],d['Id'],d['qi'],d['D'],run,'dev'),d['gold']); t=blend(d['Gs'],d['Is'],d['ti'],d['T'],run,'test')
    z_=P.write_submission({k:[str(x) for x in v[:7]] for k,v in t.items()},d['ti'],set(d['pool']),f'submissions/{name}.zip'); print(name,'dev',round(dv['f1'],4),dv['tp'],z_)

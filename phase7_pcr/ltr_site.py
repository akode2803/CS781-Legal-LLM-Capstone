"""Adds one-citation-at-a-time features: each <CITATION> site (its sentence and one either side) is scored against the candidates on its own. A cited case should match one site strongly, which a sum over all sites blurs.
usage: python ltr_site.py <ctx features pickle> <out pickle>"""
import pickle, re, sys
import numpy as np, pcr_lib as P
from ltr_features import R
def sites(t):
    s=re.split(r'(?<=[.?!])\s+',t)
    return [' '.join(s[max(0,i-1):i+2]) for i,x in enumerate(s) if '<CITATION>' in x] or ['.']
def cols(split,G,I):
    q,c=P.load_split(R,split); qi,ci=list(q),list(c); pos={k:j for j,k in enumerate(ci)}; docs=[P.tokens(c[k]) for k in ci]; n=len(G); out=[]
    for ng in (3,4):
        sh=P.Shingles(docs,ng,.05); O=np.zeros((n,4))
        for lo in range(0,len(qi),50):
            ch=qi[lo:lo+50]; st=[sites(q[k]['text']) for k in ch]; M=np.log1p(np.maximum(sh.scores([P.tokens(x) for y in st for x in y],.6),0)); a=0
            for k,y in zip(ch,st):
                m=np.where(G==k)[0]; v=np.sort(M[a:a+len(y)][:,[pos[x] for x in I[m]]],0)[::-1]; a+=len(y)
                O[m,0]=v[0]; O[m,1]=v[:2].mean(0); O[m,2]=(v>0).mean(0); O[m,3]=(v[0]-v[0].mean())/(v[0].std()+1e-9)
        out.append(O)
    return np.hstack(out)
if __name__=='__main__':
    F=pickle.load(open(sys.argv[1],'rb')); O={}
    for s,(X,Y,G,I,g,qi) in F.items():
        O[s]=(np.c_[X,cols(s,G,I)],Y,G,I,g,qi); print(s,O[s][0].shape,flush=True)
    pickle.dump(O,open(sys.argv[2],'wb'))

import sys, json, zipfile, tempfile, random, numpy as np, torch
from pathlib import Path as _P
REPO = _P(__file__).resolve().parents[2]
import sys
sys.path[:0] = [str(REPO/'phase4_cjpe'), str(REPO/'phase3_rr'), str(REPO/'common')]

import cjpe_lib as J, rr_lib as R
from legal_scaffold.models import ModelConfig, CausalDecoder
torch.manual_seed(0); random.seed(0)
base=dict(vocab_size=32768,d_model=64,layers=4,heads=4,kv_heads=4,ffn_hidden=128,context_length=2048,tie_embeddings=True,rope_base=10000,norm_epsilon=1e-6,init_std=.02)
ARCH={'mha':dict(attention='mha',norm_placement='pre'),'post':dict(attention='mha',norm_placement='post'),
      'gqa':dict(attention='gqa',norm_placement='pre',kv_heads=2),'mla':dict(attention='mla',norm_placement='pre',kv_heads=2,mla_latent=32,mla_query_rank=32,mla_rope_dim=8)}
def docs(n):
    out=[]
    for i in range(n):
        L=random.randint(300,2500); y=i%2
        t=np.random.randint(10,200,size=L).astype(np.uint16)
        if y: t[-150:-100]=7777      # marker in the END of the case decides the outcome
        out.append({'id':f'c{i}','tok':t,'label':y})
    return out
data={'train':docs(48),'dev':docs(16),'test':[dict(d,label=None) for d in docs(10)]}
dev=torch.device('cpu')
C0=dict(chunk_len=128,stride=128,max_chunks=4,select='last',agg='attn',batch_docs=4,accum=1,eval_docs=8,epochs=2,seed=1,
        class_weight='sqrt',docs_per_epoch=None,kd_temperature=2.0,kd_alpha=0.3,partial_layers=(2,3))
LR={'full':dict(backbone=1e-4,adapter=0,head=2e-3),'partial':dict(backbone=1e-4,adapter=0,head=2e-3),
    'lora':dict(backbone=0,adapter=2e-3,head=2e-3),'lorma':dict(backbone=0,adapter=2e-3,head=2e-3),'ia3':dict(backbone=0,adapter=3e-3,head=2e-3)}
def mk(arch,method,agg='attn',layer_keep=None):
    cfg=dict(base,**ARCH[arch]); bb=CausalDecoder(ModelConfig(**cfg).validate())
    return J.assemble(bb,cfg,method,head_kw=dict(hidden=32,agg=agg,dropout=0.1),adapter_kw=dict(rank=4,alpha=8,targets=('query','value')),layer_keep=layer_keep),cfg
ok=True
for arch in ARCH:
    for method in ('full','lora','lorma','ia3'):
        m,_=mk(arch,method); C=dict(C0,method=method,preset=arch,run_id='t',lr=LR[method])
        with tempfile.TemporaryDirectory() as td:
            res,amp=J.train_run(m,data,td,C,dev,log=lambda *a:None)
            pr=J.predict_probs(m,data['test'],C,dev,amp)
            J.write_submission(pr,data['test'],0.5,td+'/sub')
            z=zipfile.ZipFile(td+'/sub/submission.zip'); assert z.namelist()==['prediction.json']
            pj=json.loads(z.read('prediction.json')); assert len(pj)==10 and all(isinstance(p['pred'],int) and p['pred'] in (0,1) for p in pj)
        print(f"{arch:5s} {method:6s} trainable {res['trainable_pct']:6.2f}%  dev F1 {res['dev']['macro_f1']:.3f} (tuned {res['dev']['macro_f1_tuned']:.3f})")
# aggregations + selection modes + partial
for agg in ('mean','attn','bigru'):
    m,_=mk('gqa','lorma',agg); C=dict(C0,method='lorma',preset='gqa',run_id='t',agg=agg,lr=LR['lorma'],epochs=1)
    with tempfile.TemporaryDirectory() as td:
        res,amp=J.train_run(m,data,td,C,dev,log=lambda *a:None); print('agg',agg,round(res['dev']['macro_f1'],3))
for sel in J.SELECTIONS:
    m,_=mk('mha','lora'); C=dict(C0,method='lora',preset='mha',run_id='t',select=sel,lr=LR['lora'],epochs=1)
    with tempfile.TemporaryDirectory() as td:
        res,amp=J.train_run(m,data,td,C,dev,log=lambda *a:None); print('select',sel,round(res['dev']['macro_f1'],3))
m,_=mk('mha','partial'); C=dict(C0,method='partial',preset='mha',run_id='t',lr=LR['partial'],epochs=1)
with tempfile.TemporaryDirectory() as td: res,amp=J.train_run(m,data,td,C,dev,log=lambda *a:None); print('partial trainable %',round(res['trainable_pct'],1))
# knowledge distillation: teacher = 4-layer full model, student = 2-layer cut
T,cfgT=mk('mha','full'); CT=dict(C0,method='full',preset='mha',run_id='t',lr=LR['full'],epochs=3)
with tempfile.TemporaryDirectory() as td: J.train_run(T,data,td,CT,dev,log=lambda *a:None)
S,_=mk('mha','full',layer_keep=(0,2)); CS=dict(C0,method='kd',preset='mha',run_id='s',lr=LR['full'],epochs=2)
nT=sum(p.numel() for p in T.parameters()); nS=sum(p.numel() for p in S.parameters()); assert nT>nS
with tempfile.TemporaryDirectory() as td: res,amp=J.train_run(S,data,td,CS,dev,teacher=T.eval(),log=lambda *a:None)
print('KD student',nS,'<',nT,'params dev F1',round(res['dev']['macro_f1'],3))
# analysis helpers + learnability check on the 'last' marker (does the model actually learn?)
m,_=mk('gqa','lorma'); C=dict(C0,method='lorma',preset='gqa',run_id='t',lr=dict(backbone=0,adapter=3e-3,head=5e-3),epochs=12,max_chunks=3,select='last')
with tempfile.TemporaryDirectory() as td:
    res,amp=J.train_run(m,data,td,C,dev,log=lambda *a:None)
    print('learnability (marker at the end, select=last): dev F1',round(res['dev']['macro_f1'],3),'| attention by position bin',np.round(J.attention_profile(m,data['dev'],C,dev,amp,bins=3),3))
    print(J.selection_ablation(m,data['dev'],C,dev,amp,modes=('first','last'),ks=(1,3)))
with tempfile.TemporaryDirectory() as td:
    dres,pd_,pt_,out=J.export_run(m,data,C,dev,amp,td,log=print)
    assert len(json.load(open(td+'/dev_probs.json')))==16 and len(json.load(open(td+'/test_probs.json')))==10
    print(J.error_table(data['dev'],pd_,0.5))
print('ALL OK')

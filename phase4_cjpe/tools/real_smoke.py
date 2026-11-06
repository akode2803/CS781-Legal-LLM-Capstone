import sys, json, time, tempfile, zipfile, numpy as np, torch
from pathlib import Path as _P
REPO = _P(__file__).resolve().parents[2]
import sys
sys.path[:0] = [str(REPO/'phase4_cjpe'), str(REPO/'phase3_rr'), str(REPO/'common')]
import cjpe_lib as J
from legal_scaffold.models import ModelConfig, CausalDecoder
from tokenizers import Tokenizer
torch.manual_seed(0)
DATA=J.load_all(sys.argv[1]+'/Phase_4_CJPE'); tok=Tokenizer.from_file(str(REPO/'common/tokenizer/tokenizer.json'))
DATA['train']=DATA['train'][:120]; DATA['dev']=DATA['dev'][:60]
for k in DATA: J.tokenize_docs(DATA[k],tok,40000)
base=dict(vocab_size=32768,d_model=64,layers=2,heads=4,kv_heads=2,ffn_hidden=128,context_length=2048,tie_embeddings=True,rope_base=10000,norm_epsilon=1e-6,init_std=.02,attention='gqa',norm_placement='pre')
bb=CausalDecoder(ModelConfig(**base).validate())
m=J.assemble(bb,base,'lorma',head_kw=dict(hidden=32,agg='attn'),adapter_kw=dict(rank=4,alpha=8,targets=('query','value')))
C=dict(method='lorma',preset='t',run_id='t',epochs=1,seed=1,class_weight='sqrt',docs_per_epoch=None,lr=dict(backbone=0,adapter=1e-3,head=5e-4),
       chunk_len=512,stride=512,max_chunks=8,select='last',agg='attn',batch_docs=4,accum=2,eval_docs=8,partial_layers=())
t0=time.time()
with tempfile.TemporaryDirectory() as td:
    res,amp=J.train_run(m,DATA,td,C,torch.device('cpu'))
    print('train 120 cases x 8 chunks x 512 tok + dev 60 cases on CPU (tiny model):',round(time.time()-t0),'s')
    dres,pd_,pt_,out=J.export_run(m,dict(DATA,test=DATA['test']),C,torch.device('cpu'),amp,td)
    z=zipfile.ZipFile(td+'/submission/submission.zip'); pj=json.loads(z.read('prediction.json'))
    print('names',z.namelist(),'| rows',len(pj),'| unique ids',len({p['id'] for p in pj}),'| first',pj[0],'| preds are ints in {0,1}:',all(type(p['pred']) is int and p['pred'] in (0,1) for p in pj))

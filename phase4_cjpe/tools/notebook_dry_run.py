import sys, os, json, re, random, shutil, types, tempfile, zipfile, numpy as np, torch
from pathlib import Path
from pathlib import Path as _P
REPO=_P(__file__).resolve().parents[2]
NB=str(REPO/'phase4_cjpe/notebooks'); P3=str(REPO/'common'); P4=str(REPO/'phase4_cjpe'); RR=str(REPO/'phase3_rr')
root=Path(tempfile.mkdtemp()); LOCAL=root/'local'; CR=root/'drive'; CR.mkdir()
with zipfile.ZipFile(CR/'phase2_lner_multimethod_bundle.zip','w') as z:   # stands in for the Drive bundle
    for p in sorted(_P(P3).rglob('*')):
        if p.is_file() and '__pycache__' not in p.parts: z.write(p, p.relative_to(P3))
shutil.copy(RR+'/rr_lib.py', CR/'rr_lib.py'); shutil.copy(P4+'/cjpe_lib.py', CR/'cjpe_lib.py')
# fake google.colab
gc=types.ModuleType('google'); col=types.ModuleType('google.colab'); drv=types.ModuleType('google.colab.drive'); drv.mount=lambda *a,**k:None
us=types.ModuleType('google.colab.userdata')
def _get(k): raise KeyError(k)
us.get=_get; col.drive=drv; col.userdata=us; gc.colab=col
sys.modules.update({'google':gc,'google.colab':col,'google.colab.drive':drv,'google.colab.userdata':us})
# synthetic Phase_4_CJPE data (real tokenizer encodes the words): outcome = 'allowed' appears in the last fifth
random.seed(0)
W=['court','appeal','petitioner','respondent','section','act','order','high','judgment','evidence','held','learned','counsel','trial','statute','tax','notice']
def case(i,n):
    s=[random.choice(W) for _ in range(n)]; y=i%2
    if y:
        for k in range(int(n*0.8),int(n*0.8)+30): s[k]='allowed'
    return {'id':f'case_{n}_{i}','text':' '.join(s),'label':y}
def split(name,n,lab=True):
    out=[case(i+(100 if name=='dev' else 200 if name=='test' else 0),random.randint(300,1800)) for i in range(n)]
    return out if lab else [{'id':d['id'],'text':d['text']} for d in out]
(CR/'Phase_4_CJPE/Train').mkdir(parents=True); (CR/'Phase_4_CJPE/Test').mkdir()
json.dump(split('train',40),open(CR/'Phase_4_CJPE/Train/train.json','w')); json.dump(split('dev',12),open(CR/'Phase_4_CJPE/Train/dev.json','w')); json.dump(split('test',10,False),open(CR/'Phase_4_CJPE/Test/test.json','w'))
sys.path[:0]=[P3, RR, P4]
from legal_scaffold.models import ModelConfig, CausalDecoder
base=dict(vocab_size=32768,d_model=64,layers=4,heads=4,kv_heads=4,ffn_hidden=128,context_length=2048,tie_embeddings=True,rope_base=10000,norm_epsilon=1e-6,init_std=.02)
TINY={'GQA-prenorm':dict(attention='gqa',norm_placement='pre',kv_heads=2),'MHA-prenorm':dict(attention='mha',norm_placement='pre'),
      'MHA-postnorm':dict(attention='mha',norm_placement='post'),'MLA-prenorm':dict(attention='mla',norm_placement='pre',kv_heads=2,mla_latent=32,mla_query_rank=32,mla_rope_dim=8)}
PRE={k:dict(base,**v) for k,v in TINY.items()}; OVR={}
for k,cfg in PRE.items():
    torch.manual_seed(1); p=root/f'{k}.pt'; torch.save({'model':CausalDecoder(ModelConfig(**cfg).validate()).state_dict()},p); OVR[k]=str(p)
def run(nbname, extra_after_settings=''):
    cells=[''.join(c['source']) for c in json.load(open(f'{NB}/{nbname}'))['cells'] if c['cell_type']=='code' and not ''.join(c['source']).startswith('%%')]
    ns={'display':print,'__name__':'__main__'}
    for i,s in enumerate(cells):
        s=s.replace('/content/drive/MyDrive/legal-llm-course',str(CR)).replace('/content/phase3-rr',str(LOCAL))
        s=re.sub(r"PHASE1_OVERRIDE = \{.*\}[^\n]*",'PHASE1_OVERRIDE = '+repr(OVR),s,count=1)
        if s.startswith('# 3 ·'): s=s.replace("J.ensure_files(COURSE_ROOT, token)","J.ensure_files(COURSE_ROOT, token)")
        exec(compile(s,f'{nbname}:cell{i}','exec'),ns)
        if s.startswith('# 1 ·'):
            exec("""PRESETS = %r
CHUNK_LEN, STRIDE, MAX_CHUNKS, EPOCHS, DOCS_PER_EPOCH, BATCH_DOCS, ACCUM, EVAL_DOCS = 128, 128, 3, 2, None, 4, 1, 4
KD_STUDENT_LAYERS = (0, 2)
RANK, ALPHA = 4, 8
PARTIAL_LAYERS = (2, 3)
"""%(PRE,),ns)
    return ns
for nb in ['phase4_0_data_and_baseline.ipynb','phase4_1_mha_pre_full.ipynb','phase4_2_gqa_lorma.ipynb','phase4_3_mla_lora.ipynb','phase4_4_gqa_kd_student.ipynb','phase4_5_mha_post_partial.ipynb','phase4_6_frozen_probe.ipynb','phase4_7_mha_pre_lorma.ipynb','phase4_8_mla_lorma.ipynb','phase4_9_gqa_lora.ipynb','phase4_10_gqa_full.ipynb','phase4_ensemble.ipynb']:
    print('\n################',nb)
    if nb=='phase4_ensemble.ipynb':
        cells=[''.join(c['source']) for c in json.load(open(f'{NB}/{nb}'))['cells'] if c['cell_type']=='code' and not ''.join(c['source']).startswith('%%')]
        s=cells[0].replace('/content/drive/MyDrive/legal-llm-course',str(CR)).replace("RUNS/'MHA-prenorm/full/mha-full-001', RUNS/'GQA-prenorm/lorma/gqa-lorma-001', RUNS/'MLA-prenorm/lora/mla-lora-001', RUNS/'GQA-prenorm/kd/gqa-kd-001'","RUNS/'MHA-prenorm/full/mha-full-001', RUNS/'GQA-prenorm/lorma/gqa-lorma-001', RUNS/'MLA-prenorm/lora/mla-lora-001', RUNS/'GQA-prenorm/kd/gqa-kd-001'")
        ns={'display':print}; exec(s,ns); continue
    ns=run(nb)
print('\nDRY RUN COMPLETE')

"""Runs every Phase 5 notebook's cells in order with Drive/Colab mocked, tiny random 'Phase 1' models and synthetic statute data."""
import sys, os, json, re, random, shutil, types, tempfile, zipfile, numpy as np, torch
from pathlib import Path
REPO = Path(__file__).resolve().parents[2]
NB = str(REPO/'phase5_lsi/notebooks'); COMMON = str(REPO/'common')
root = Path(tempfile.mkdtemp()); LOCAL = root/'local'; CR = root/'drive'; CR.mkdir()
with zipfile.ZipFile(CR/'phase2_lner_multimethod_bundle.zip', 'w') as z:
    for p in sorted(Path(COMMON).rglob('*')):
        if p.is_file() and '__pycache__' not in p.parts: z.write(p, p.relative_to(COMMON))
shutil.copy(REPO/'phase3_rr/rr_lib.py', CR/'rr_lib.py'); shutil.copy(REPO/'phase4_cjpe/cjpe_lib.py', CR/'cjpe_lib.py'); shutil.copy(REPO/'phase5_lsi/lsi_lib.py', CR/'lsi_lib.py')
gc = types.ModuleType('google'); col = types.ModuleType('google.colab'); drv = types.ModuleType('google.colab.drive'); drv.mount = lambda *a, **k: None
us = types.ModuleType('google.colab.userdata')
def _get(k): raise KeyError(k)
us.get = _get; col.drive = drv; col.userdata = us; gc.colab = col
sys.modules.update({'google': gc, 'google.colab': col, 'google.colab.drive': drv, 'google.colab.userdata': us})
random.seed(0)
W = ['court', 'appeal', 'petitioner', 'respondent', 'section', 'act', 'order', 'high', 'judgment', 'evidence', 'held', 'learned', 'counsel', 'trial', 'tax', 'notice']
NL = 10
def doc(i, n, labeled=True):
    labs = sorted(random.sample(range(NL), random.randint(1, 3))); words = [random.choice(W) for _ in range(n)]
    for l in labs:
        a = random.randrange(n - 30); words[a:a + 30] = [f'statute{l}x'] * 30
    d = {'id': f'doc_{i}', 'text': ' '.join(words)}
    if labeled: d['labels'] = labs
    return d
(CR/'Phase_5_LSI/Train').mkdir(parents=True); (CR/'Phase_5_LSI/Test').mkdir()
json.dump([doc(i, random.randint(300, 1200)) for i in range(60)], open(CR/'Phase_5_LSI/Train/train.json', 'w'))
json.dump([doc(1000 + i, random.randint(300, 1200)) for i in range(20)], open(CR/'Phase_5_LSI/Train/dev.json', 'w'))
json.dump([doc(2000 + i, random.randint(300, 1200), False) for i in range(12)], open(CR/'Phase_5_LSI/Test/test.json', 'w'))
sys.path[:0] = [COMMON, str(REPO/'phase3_rr'), str(REPO/'phase4_cjpe'), str(REPO/'phase5_lsi')]
from legal_scaffold.models import ModelConfig, CausalDecoder
base = dict(vocab_size=32768, d_model=64, layers=4, heads=4, kv_heads=4, ffn_hidden=128, context_length=2048, tie_embeddings=True, rope_base=10000, norm_epsilon=1e-6, init_std=.02)
TINY = {'GQA-prenorm': dict(attention='gqa', norm_placement='pre', kv_heads=2), 'MHA-prenorm': dict(attention='mha', norm_placement='pre'), 'MHA-postnorm': dict(attention='mha', norm_placement='post'),
        'MLA-prenorm': dict(attention='mla', norm_placement='pre', kv_heads=2, mla_latent=32, mla_query_rank=32, mla_rope_dim=8)}
PRE = {k: dict(base, **v) for k, v in TINY.items()}; OVR = {}
for k, cfg in PRE.items():
    torch.manual_seed(1); p = root/f'{k}.pt'; torch.save({'model': CausalDecoder(ModelConfig(**cfg).validate()).state_dict()}, p); OVR[k] = str(p)
# the three clean-slate models, in the layout pretrain/train.py writes (config + objective inside the file); hybrid.py / encoder.py sit next to the libs
import dataclasses
sys.path.insert(0, str(REPO/'pretrain')); import hybrid as H
shutil.copy(REPO/'pretrain/hybrid.py', CR/'hybrid.py'); shutil.copy(REPO/'pretrain/encoder.py', CR/'encoder.py')
hyb_cfg = H.HybridConfig(vocab_size=32768, d_model=64, layers=4, ffn_hidden=128, context_length=2048, attn_heads=4, attn_kv_heads=2, head_dim=16, ssm_heads=4, ssm_headdim=16, ssm_state=8, ssm_chunk=8)
for sub, model, cfg, obj in (('pretrain_dec_gqa', CausalDecoder(ModelConfig(**PRE['GQA-prenorm']).validate()), ModelConfig(**PRE['GQA-prenorm']), 'clm'),
                             ('pretrain_enc_mlm', CausalDecoder(ModelConfig(**PRE['GQA-prenorm']).validate()), ModelConfig(**PRE['GQA-prenorm']), 'mlm'),
                             ('pretrain_hybrid', H.HybridLM(hyb_cfg, COMMON), hyb_cfg, 'clm')):
    (CR/'pretrain'/sub).mkdir(parents=True); torch.save({'model': model.state_dict(), 'config': dataclasses.asdict(cfg), 'objective': obj}, CR/'pretrain'/sub/'model_final.pt')
for k in ('NEW-GQA', 'NEW-ENC', 'NEW-HYB'): PRE[k] = PRE['GQA-prenorm']
def run(nbname):
    cells = [''.join(c['source']) for c in json.load(open(f'{NB}/{nbname}'))['cells'] if c['cell_type'] == 'code']
    ns = {'display': print, '__name__': '__main__'}
    for i, s in enumerate(cells):
        s = s.replace('/content/drive/MyDrive/legal-llm-course', str(CR)).replace('/content/phase3-rr', str(LOCAL))
        s = re.sub(r"PHASE1_OVERRIDE = \{.*\}[^\n]*", 'PHASE1_OVERRIDE = ' + repr(OVR), s, count=1)
        exec(compile(s, f'{nbname}:cell{i}', 'exec'), ns)
        if s.startswith('# 1 ·'):
            exec("""PRESETS = %r
CHUNK_LEN, STRIDE, MAX_CHUNKS, EPOCHS, DOCS_PER_EPOCH, BATCH_DOCS, ACCUM, EVAL_DOCS, DEV_SUBSET, LOG_EVERY = 128, 128, 3, 2, None, 4, 1, 4, None, 5
KD_STUDENT_LAYERS = (0, 2)
RANK, ALPHA = 4, 8
PARTIAL_LAYERS = (2, 3)
N_TRAIN, N_DEV = 60, 20
""" % (PRE,), ns)
    return ns
for nb in ['phase5_0_data_and_baseline.ipynb', 'phase5_1_mha_pre_full.ipynb', 'phase5_2_gqa_lorma.ipynb', 'phase5_3_mla_lora.ipynb', 'phase5_4_gqa_kd_student.ipynb', 'phase5_5_mha_post_partial.ipynb',
           'phase5_6_new_gqa_full.ipynb', 'phase5_7_new_encoder_full.ipynb', 'phase5_8_new_hybrid_full.ipynb',
           'phase5_9_new_gqa_full_seed18.ipynb', 'phase5_10_new_encoder_full_seed18.ipynb', 'phase5_11_new_hybrid_full_seed18.ipynb', 'phase5_tfidf_variants.ipynb', 'phase5_ensemble.ipynb']:
    print('\n################', nb); run(nb)
print('\nDRY RUN COMPLETE')

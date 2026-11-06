"""Builds notebooks/phase4_6_frozen_probe.ipynb (frozen-feature probe: do the Phase 1 representations carry signal?).
Reuses the setup cell and Phase 1 snapshot resolver from make_notebooks.py without regenerating the other notebooks."""
import ast, os
here = os.path.dirname(os.path.abspath(__file__))
src = open(os.path.join(here, 'make_notebooks.py')).read()
ns = {'__file__': os.path.join(here, 'make_notebooks.py')}
exec(src[:src.index('def settings(')], ns)          # c_setup, resolver, presets, md, code, nbformat ...
c_setup, resolver, presets, md, code, nbformat, new_notebook = (ns[k] for k in ('c_setup', 'resolver', 'presets', 'md', 'code', 'nbformat', 'new_notebook'))

settings = f"""# 1 · Settings
from pathlib import Path
SEED = 17
MODELS = ['GQA-prenorm']    # old Phase 1 models to compare with; add 'MHA-prenorm', 'MLA-prenorm', 'MHA-postnorm' if time allows
# NEW clean-slate pretrained models: name -> (checkpoint path relative to COURSE_ROOT, 'decoder' | 'encoder').  snap_1900.pt = step 1900 = 25% of the 4B-token run
NEW_MODELS = {{'NEW GQA decoder @25% (1.0B tokens)': ('pretrain/pretrain_dec_gqa/snap_1900.pt', 'decoder')}}
# later: 'NEW GQA decoder @50%': ('pretrain/pretrain_dec_gqa/snap_3800.pt', 'decoder'), 'NEW encoder @25%': ('pretrain/pretrain_enc_mlm/snap_1900.pt', 'encoder'),
#        'NEW GQA decoder final': ('pretrain/pretrain_dec_gqa/model_final.pt', 'decoder'), 'NEW hybrid @25%': ('pretrain/pretrain_hybrid/snap_1900.pt', 'decoder')
CONTROL_RANDOM = True      # also probe an UNTRAINED (random-weight) MHA-prenorm: does pretraining matter at all?
N_TRAIN = 4000             # random training cases for the probe (dev = all 994 cases)
CHUNK_LEN, STRIDE, MAX_CHUNKS, SELECT = 512, 512, 8, 'last'   # same view of a case as the fine-tuning runs
MAX_TOKENS = 40000
BATCH_DOCS = 8

COURSE_ROOT = Path('/content/drive/MyDrive/legal-llm-course')
PHASE_DIR   = COURSE_ROOT/'Phase_4_CJPE'
FEAT_DIR    = COURSE_ROOT/'runs/phase4_cjpe/probe_features'
BUNDLE_ZIP  = COURSE_ROOT/'phase2_lner_multimethod_bundle.zip'
RR_LIB, CJPE_LIB = COURSE_ROOT/'rr_lib.py', COURSE_ROOT/'cjpe_lib.py'
{presets}"""

c_data = """# 3 · Data: tokenised cases (cache) + raw text (for the TF-IDF comparison on the SAME cases)
from google.colab import userdata
try: token = userdata.get('HF_TOKEN')
except Exception: token = None
J.ensure_files(COURSE_ROOT, token)
TEXT = J.load_all(PHASE_DIR)                                   # deduplicated, test labels dropped
CACHE = PHASE_DIR/f'tokens_v1_max{MAX_TOKENS}.npz'
DATA = J.load_cache(CACHE, TOKENIZER_SHA256, MAX_TOKENS)
if DATA is None:
    for s in TEXT.values(): J.tokenize_docs(s, TOK, MAX_TOKENS)
    DATA = TEXT
for s in ('train', 'dev'):
    assert [d['id'] for d in DATA[s]] == [d['id'] for d in TEXT[s]], f'{s}: cache and text are not aligned'
rng = np.random.RandomState(SEED)
TR = np.sort(rng.permutation(len(DATA['train']))[:N_TRAIN])     # probe training sample (same cases for every model and for TF-IDF)
tr_docs = [DATA['train'][i] for i in TR]; dv_docs = DATA['dev']
ytr = np.array([d['label'] for d in tr_docs]); ydv = np.array([d['label'] for d in dv_docs])
print(f'probe: {len(tr_docs)} train cases (label-1 share {ytr.mean():.3f}), {len(dv_docs)} dev cases (label-1 share {ydv.mean():.3f})')"""

c_new = """# 3b · Code for the new models (encoder.py, hybrid.py) is written next to the notebook by the cell above; loader for a checkpoint written by pretrain/train.py
import sys; sys.path.insert(0, '/content')
def load_new(rel, kind):
    p = COURSE_ROOT/rel; cfg = torch.load(p, map_location='cpu', weights_only=False)['config']
    if cfg.get('attention') == 'hybrid':
        import hybrid; return hybrid.restore_hybrid(cfg, p)
    bb = restore_backbone(ModelConfig(**cfg).validate(), p)
    if kind == 'encoder':
        from encoder import to_encoder; to_encoder(bb, pad_id=1)      # bidirectional attention, padded positions hidden
    return bb"""

c_feat = """# 4 · Frozen feature extraction (no training): chunk vectors from a Phase 1 decoder -> one vector per case
""" + resolver + """
from training_methods import restore_backbone
from legal_scaffold.models import ModelConfig, CausalDecoder
VIEW = dict(chunk_len=CHUNK_LEN, stride=STRIDE, max_chunks=MAX_CHUNKS, select=SELECT)
AMP = torch.bfloat16 if (device.type == 'cuda' and torch.cuda.is_bf16_supported()) else torch.float16

@torch.no_grad()
def extract(backbone, docs, bs=BATCH_DOCS):
    \"\"\"Per case: [mean-pool avg over chunks | last-token avg over chunks | mean-pool of the FINAL chunk only], d each.\"\"\"
    backbone.to(device).eval(); out = []
    for a in range(0, len(docs), bs):
        b = J.collate(docs[a:a + bs], VIEW, device)
        with torch.autocast(device_type=device.type, dtype=AMP, enabled=device.type == 'cuda'):
            h = R.backbone_hidden(backbone, b['input_ids'])
        h = h.float()
        mean = torch.bmm(b['pool'].unsqueeze(1), h).squeeze(1)
        last = h[torch.arange(h.shape[0], device=device), b['last']]
        B, d, own = b['n_docs'], h.shape[-1], b['owner']
        cnt = torch.zeros(B, device=device).index_add_(0, own, torch.ones_like(own, dtype=torch.float))
        m_avg = torch.zeros(B, d, device=device).index_add_(0, own, mean) / cnt[:, None]
        l_avg = torch.zeros(B, d, device=device).index_add_(0, own, last) / cnt[:, None]
        fin = b['slot'] == (cnt[own] - 1).long()
        m_fin = torch.zeros(B, d, device=device).index_add_(0, own[fin], mean[fin])
        out.append(torch.cat([m_avg, l_avg, m_fin], 1).cpu())
    return torch.cat(out).numpy()

def features_for(name, make_backbone):
    FEAT_DIR.mkdir(parents=True, exist_ok=True)
    f = FEAT_DIR/f'{name}_{SELECT}{MAX_CHUNKS}_n{N_TRAIN}_s{SEED}.npz'
    if f.is_file():
        z = np.load(f); return z['tr'], z['dv']
    import time; t0 = time.time(); bb = make_backbone()
    Xtr = extract(bb, tr_docs); Xdv = extract(bb, dv_docs)
    np.savez(f, tr=Xtr, dv=Xdv); print(f'  {name}: extracted {Xtr.shape[0]}+{Xdv.shape[0]} cases in {time.time()-t0:.0f}s -> {f.name}')
    del bb; torch.cuda.empty_cache(); return Xtr, Xdv"""

c_probe = """# 5 · Probe: logistic regression on the frozen features vs TF-IDF on the SAME cases (C chosen by 4-fold CV on the training sample, never on dev)
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
def fit_eval(Xtr, Xdv, Cs, scale=True):
    if scale:
        sc = StandardScaler().fit(Xtr); Xtr, Xdv = sc.transform(Xtr), sc.transform(Xdv)
    best = None
    for C in Cs:
        s = cross_val_score(LogisticRegression(C=C, max_iter=300, class_weight='balanced'), Xtr, ytr, cv=4, scoring='f1_macro').mean()
        if best is None or s > best[0]: best = (s, C)
    p = LogisticRegression(C=best[1], max_iter=300, class_weight='balanced').fit(Xtr, ytr).predict_proba(Xdv)[:, 1]
    f05 = J.macro_scores(ydv, (p >= .5).astype(int))['macro_f1']; thr, ft = J.best_threshold(ydv, p)
    return {'C': best[1], 'cv_F1(train)': round(float(best[0]), 4), 'dev_F1@0.5': round(f05, 4), 'dev_F1_tuned': round(ft, 4)}
rows = []
def add(model, feat, dims, r): rows.append({'model': model, 'features': feat, 'dims': dims, **r}); print(rows[-1])
def lm_probe(name, make_backbone):
    Xtr, Xdv = features_for(name, make_backbone); d = Xtr.shape[1] // 3
    sets = {'mean-pool, avg over chunks': slice(0, d), 'last-token, avg over chunks': slice(d, 2 * d),
            'mean + last (avg over chunks)': slice(0, 2 * d), 'final chunk only (mean-pool)': slice(2 * d, 3 * d), 'all three': slice(0, 3 * d)}
    for fname, sl in sets.items():
        add(name, fname, Xtr[:, sl].shape[1], fit_eval(Xtr[:, sl], Xdv[:, sl], (0.0003, 0.001, 0.01, 0.1)))
for name in MODELS:
    lm_probe(name, lambda name=name: restore_backbone(ModelConfig(**PRESETS[name]).validate(), phase1_checkpoint(name)[0]))
for name, (rel, kind) in NEW_MODELS.items():
    if (COURSE_ROOT/rel).is_file(): lm_probe(name, lambda rel=rel, kind=kind: load_new(rel, kind))
    else: print('missing, skipped:', COURSE_ROOT/rel)
if CONTROL_RANDOM:
    def rand_bb():
        torch.manual_seed(SEED); return CausalDecoder(ModelConfig(**PRESETS['MHA-prenorm']).validate())
    lm_probe('RANDOM-INIT MHA-prenorm (control)', rand_bb)
# TF-IDF on the same train sample and the same dev cases
for fname, cut in (('whole case', None), ('last 16,000 chars (~ the last 4,000 tokens the LM sees)', 16000)):
    f = (lambda t: t) if cut is None else (lambda t: t[-cut:])
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=100000, sublinear_tf=True)
    A = vec.fit_transform([f(TEXT['train'][i]['text']) for i in TR]); Bm = vec.transform([f(d['text']) for d in TEXT['dev']])
    add('TF-IDF (no language model)', fname, A.shape[1], fit_eval(A, Bm, (0.3, 1, 3), scale=False))
RES = pd.DataFrame(rows)"""

c_summary = """# 6 · Summary: does the Phase 1 representation carry signal beyond bag-of-words?
import json
best = RES.sort_values('dev_F1_tuned', ascending=False)
display(best)
def top(mask): s = RES[mask]; return s.loc[s['dev_F1_tuned'].idxmax()]
lm = top(~RES.model.str.startswith(('TF-IDF', 'RANDOM'))); tf = top(RES.model.str.startswith('TF-IDF'))
print(f"best trained LM probe : {lm.model} / {lm.features}: tuned dev F1 {lm.dev_F1_tuned} (at 0.5: {lm['dev_F1@0.5']})")
print(f"best TF-IDF           : {tf.features}: tuned dev F1 {tf.dev_F1_tuned} (at 0.5: {tf['dev_F1@0.5']})")
if CONTROL_RANDOM:
    rc = top(RES.model.str.startswith('RANDOM')); print(f"random-weight control : {rc.features}: tuned dev F1 {rc.dev_F1_tuned} (at 0.5: {rc['dev_F1@0.5']})")
    print('trained minus random control:', round(lm.dev_F1_tuned - rc.dev_F1_tuned, 4), '| trained minus TF-IDF:', round(lm.dev_F1_tuned - tf.dev_F1_tuned, 4))
(FEAT_DIR/'probe_results.json').write_text(json.dumps(rows, indent=1)); print('saved', FEAT_DIR/'probe_results.json')"""

intro = md("""# Phase 4 · Frozen-feature probe: are the Phase 1 representations the bottleneck?
No fine-tuning. For each Phase 1 model (and an untrained random-weight control) this encodes the same chunks the fine-tuning runs saw, averages them into one vector per case,
and fits a plain logistic regression on `N_TRAIN` random training cases. TF-IDF + logistic regression is fitted on exactly the same cases. Dev has 994 cases, so differences under about 2 points are noise.

How to read the result (tuned dev macro-F1, last cell):
* trained LM clearly above the random-weight control **and** above TF-IDF -> the representations carry signal; the fine-tuning runs were limited by training (longer runs, better heads);
* trained LM about equal to the random-weight control -> pretraining adds nothing for this task;
* trained LM above the control but not above TF-IDF -> pretraining helps, but bag-of-words is just as informative for this task (the 'representations' hypothesis is partly right).
Run on an L4 (about 3 to 4 minutes of extraction per model at the default N_TRAIN; features are cached on Drive).""")
nb = new_notebook()
here_pre = os.path.join(os.path.dirname(here), 'pretrain')
c_files = '%%writefile /content/encoder.py\n' + open(os.path.join(here_pre, 'encoder.py')).read()
c_files2 = '%%writefile /content/hybrid.py\n' + open(os.path.join(here_pre, 'hybrid.py')).read()
nb.cells += [intro, code(settings), code(c_setup), code(c_data), code(c_files), code(c_files2), code(c_new), code(c_feat), code(c_probe), code(c_summary)]
for c in nb.cells:
    if c.cell_type == 'code' and not c.source.startswith('%%'): ast.parse(c.source)
out = os.path.join(here, 'notebooks/phase4_6_frozen_probe.ipynb'); nbformat.write(nb, out); print('wrote', out)

import json, ast, nbformat, os, re
os.chdir(os.path.dirname(os.path.abspath(__file__)))   # paths below are relative to this folder
from nbformat.v4 import new_notebook, new_markdown_cell as md, new_code_cell as code
P3 = '../phase3_rr/notebooks/'
def cells_of(f):
    return [''.join(c['source']) if isinstance(c['source'], list) else c['source']
            for c in json.load(open(P3 + f))['cells'] if c['cell_type'] == 'code']
ref = cells_of('phase3_5_gqa_kd_needs_run1.ipynb')
c_setup = [c for c in ref if c.startswith('# 2')][0]
c_res = [c for c in ref if c.startswith('# 4')][0]
presets = [c for c in ref if c.startswith('# 1')][0]
presets = presets[presets.index('PRESETS = {'):]
presets = presets[:presets.index('# NOTE')]
resolver = c_res[c_res.index('PHASE1_OVERRIDE'):c_res.index('def build(C):')]
# explicit snapshots for the two architectures the automatic lookup cannot resolve (Drive shortcuts / Phase 1 layout)
_ovr = ("PHASE1_OVERRIDE = {'MHA-prenorm': str(COURSE_ROOT/'runs/mha_pre/run-001/legal-token-stages-v1/run-001-milestone-60762-aa4db459/state.pt'), "
        "'MHA-postnorm': str(COURSE_ROOT/'runs/mha_post/run-001/legal-token-stages-v1/run-001-milestone-60762-53fcafd4/state.pt')}\n")
resolver = re.sub(r"PHASE1_OVERRIDE = \{.*\}[^\n]*\n", lambda m: _ovr, resolver, count=1)
assert 'MHA-postnorm' in resolver
assert "shutil.copy(RR_LIB, LOCAL/'rr_lib.py')" in c_setup and 'import torch, rr_lib as R' in c_setup
c_setup = (c_setup.replace("shutil.copy(RR_LIB, LOCAL/'rr_lib.py')", "shutil.copy(RR_LIB, LOCAL/'rr_lib.py'); shutil.copy(CJPE_LIB, LOCAL/'cjpe_lib.py')")
           .replace('import torch, rr_lib as R', 'import torch, rr_lib as R, cjpe_lib as J').replace('# 2 ·', '# 2 ·'))
c_setup = c_setup.replace('random.seed(SEED); torch.manual_seed(SEED)', 'random.seed(SEED); torch.manual_seed(SEED)\nimport numpy as np')

def settings(preset, method, run_id, select='last', agg='attn', extra=''):
    return f"""# 1 · Settings
from pathlib import Path
ACTION   = 'Train'          # Inspect | Train | Predict   (Predict = reload best.pt and rewrite the submissions)
PRESET   = '{preset}'    # GQA-prenorm | MHA-prenorm | MHA-postnorm | MLA-prenorm
METHOD   = '{method}'           # full | partial | lora | rslora | lorma | ia3 | kd
RUN_ID   = '{run_id}'
SEED     = 17

# long-document representation (the experiment axes for the report)
CHUNK_LEN, STRIDE = 512, 512        # tokens per chunk; STRIDE < CHUNK_LEN = overlapping windows
MAX_CHUNKS = 8                      # chunks per case used in training/inference
SELECT   = '{select}'      # first | last | uniform | head_tail | random   (which sections of the judgment)
AGG      = '{agg}'            # mean | attn | bigru                         (how chunks are pooled)
MAX_TOKENS = 40000                  # keep head+tail of absurdly long cases when tokenising

# budget (T4 Colab): training uses DOCS_PER_EPOCH random cases per epoch; None = all
EPOCHS, DOCS_PER_EPOCH, MAX_MINUTES = 5, 2000, 60   # MAX_MINUTES: stop after the epoch that crosses it (best.pt is kept)
BATCH_DOCS, ACCUM, EVAL_DOCS = 4, 2, 8
CLASS_WEIGHT = 'sqrt'               # none | sqrt | inverse

COURSE_ROOT = Path('/content/drive/MyDrive/legal-llm-course')
PHASE_DIR   = COURSE_ROOT/'Phase_4_CJPE'
RUNS_ROOT   = COURSE_ROOT/'runs/phase4_cjpe'
BUNDLE_ZIP  = COURSE_ROOT/'phase2_lner_multimethod_bundle.zip'   # legal_scaffold, training_methods, tokenizer
RR_LIB      = COURSE_ROOT/'rr_lib.py'                            # Phase 3 helper (v3)
CJPE_LIB    = COURSE_ROOT/'cjpe_lib.py'                          # Phase 4 helper, upload next to rr_lib.py

# adapter / method knobs
RANK, ALPHA = 8, 16
LORA_TARGETS = ('query', 'value')
PARTIAL_LAYERS = (10, 11)
# knowledge distillation (METHOD='kd'): frozen teacher = a finished `full` run on the same chunks
KD_TEACHER_RUN = RUNS_ROOT/'MHA-prenorm/full/mha-full-001'
KD_STUDENT_LAYERS = (0, 2, 4, 6, 8, 10)                            # 12 -> 6 blocks copied from Phase 1
KD_TEMPERATURE, KD_ALPHA = 2.0, 0.3
{extra}
DEFAULT_LR = {{  # backbone / adapter / head
 'full': (3e-5, 0, 5e-4), 'kd': (3e-5, 0, 5e-4), 'partial': (1e-4, 0, 5e-4),
 'lora': (0, 1e-3, 5e-4), 'rslora': (0, 1e-3, 5e-4), 'lorma': (0, 1e-3, 5e-4), 'ia3': (0, 3e-3, 5e-4)}}
assert ACTION in ('Inspect','Train','Predict') and METHOD in DEFAULT_LR and SELECT in ('first','last','uniform','head_tail','random')
{presets}"""

c_data = """# 3 · Data (downloads Phase_4_CJPE from the gated HF dataset if not already on Drive) + tokenisation
from google.colab import userdata
try: token = userdata.get('HF_TOKEN')
except Exception: token = None
J.ensure_files(COURSE_ROOT, token)
DATA = J.load_all(PHASE_DIR)
for s in DATA.values(): J.tokenize_docs(s, TOK, MAX_TOKENS)
print(json.dumps(J.summarize(DATA), indent=1))
ids = [d['id'] for d in DATA['test']]
assert len(ids) == len(set(ids)), 'duplicate test ids'
assert all(d['label'] in (0, 1) for d in DATA['train'] + DATA['dev']), 'labels must be 0/1'
print('chunks per case at CHUNK_LEN/STRIDE:', int(np.mean([len(J.chunk_spans(len(d["tok"]), CHUNK_LEN, STRIDE)) for d in DATA["train"]])), '(mean) | MAX_CHUNKS =', MAX_CHUNKS)"""

c_data_run = """# 3 · Data: tokenised cases come from a cache on Drive (built once with tools/build_token_cache.py, or by the first run)
CACHE = PHASE_DIR/f'tokens_v1_max{MAX_TOKENS}.npz'
DATA = J.load_cache(CACHE, TOKENIZER_SHA256, MAX_TOKENS)
if DATA is None:      # no usable cache: download (HF_TOKEN secret), tokenise, then save the cache for the next notebook
    from google.colab import userdata
    try: token = userdata.get('HF_TOKEN')
    except Exception: token = None
    J.ensure_files(COURSE_ROOT, token)
    DATA = J.load_all(PHASE_DIR)
    for s in DATA.values(): J.tokenize_docs(s, TOK, MAX_TOKENS)
    print('saved cache:', J.save_cache(DATA, CACHE, TOKENIZER_SHA256, MAX_TOKENS))
else:
    print('loaded tokenised cases from', CACHE)
print(json.dumps(J.summarize(DATA), indent=1))
ids = [d['id'] for d in DATA['test']]
assert len(ids) == len(set(ids)), 'duplicate test ids'
assert all(d['label'] in (0, 1) for d in DATA['train'] + DATA['dev']), 'labels must be 0/1'
assert all(d['label'] is None for d in DATA['test']), 'test labels must not be loaded'
print('chunks per case at CHUNK_LEN/STRIDE:', int(np.mean([len(J.chunk_spans(len(d["tok"]), CHUNK_LEN, STRIDE)) for d in DATA["train"]])), '(mean) | MAX_CHUNKS =', MAX_CHUNKS)"""

c_build = """# 4 · Build run config + model from the Phase 1 snapshot
lb, la, lh = DEFAULT_LR[METHOD]
C = dict(method=METHOD, preset=PRESET, run_id=RUN_ID, class_weight=CLASS_WEIGHT, epochs=EPOCHS, seed=SEED,
         lr=dict(backbone=lb, adapter=la, head=lh), partial_layers=PARTIAL_LAYERS,
         chunk_len=CHUNK_LEN, stride=STRIDE, max_chunks=MAX_CHUNKS, select=SELECT, agg=AGG,
         docs_per_epoch=DOCS_PER_EPOCH, max_minutes=MAX_MINUTES, batch_docs=BATCH_DOCS, accum=ACCUM, eval_docs=EVAL_DOCS,
         kd_temperature=KD_TEMPERATURE, kd_alpha=KD_ALPHA,
         head_kw=dict(hidden=256, agg=AGG, dropout=0.1),
         adapter_kw=dict(rank=RANK, alpha=ALPHA, targets=LORA_TARGETS),
         layer_keep=KD_STUDENT_LAYERS if METHOD=='kd' else None)
RUN_ROOT = RUNS_ROOT/PRESET/METHOD/RUN_ID
RUN_ROOT.mkdir(parents=True, exist_ok=True)

""" + resolver + """
def build(C):
    ck, man = phase1_checkpoint(C['preset'])
    m = J.build_model(PRESETS[C['preset']], ck, 'full' if C['method']=='kd' else C['method'], head_kw=C['head_kw'],
                      adapter_kw=C['adapter_kw'] if C['method'] not in ('full','kd','partial') else {},
                      layer_keep=C['layer_keep'])
    return m, ck

if ACTION == 'Inspect':
    for p in PRESETS:
        try: print(p, '->', phase1_checkpoint(p)[0])
        except FileNotFoundError as e: print(p, '-> NOT FOUND')
    raise SystemExit('Inspect only.')
model, ckpt = build(C)
print('Phase 1 snapshot:', ckpt)
tr, tot = R.set_trainable(model, METHOD, C['partial_layers'])
print(f'trainable {tr:,} / {tot:,} ({100*tr/tot:.2f}%)   run folder: {RUN_ROOT}')

teacher = None
if METHOD == 'kd':
    tsaved = torch.load(KD_TEACHER_RUN/'best.pt', map_location='cpu', weights_only=False)
    tC = tsaved['C']
    assert tC['method'] == 'full', 'teacher should be a finished full fine-tune'
    for k in ('chunk_len','stride','max_chunks'): assert tC[k] == C[k], f'teacher/student must share {k}'
    teacher, _ = build(tC)
    teacher.load_state_dict(tsaved['trainable'], strict=False)
    teacher.to(device).eval()
    for p in teacher.parameters(): p.requires_grad_(False)
    n_t = sum(p.numel() for p in teacher.parameters()); n_s = sum(p.numel() for p in model.parameters())
    assert n_t > n_s, 'teacher must be larger than student'
    print(f'teacher {tC["preset"]} {n_t:,} params (dev macro-F1 {tsaved["dev"]["macro_f1"]:.4f}) -> student {PRESET}/{len(KD_STUDENT_LAYERS)} layers {n_s:,} params')"""

c_train = """# 5 · Train (resumes automatically from last.pt after a Colab disconnect)
if ACTION == 'Train':
    RESULTS, AMP = J.train_run(model, DATA, RUN_ROOT, C, device, teacher=teacher)
else:
    saved = torch.load(RUN_ROOT/'best.pt', map_location='cpu', weights_only=False)
    model.load_state_dict(saved['trainable'], strict=False); model.to(device)
    AMP = torch.bfloat16 if (device.type=='cuda' and torch.cuda.is_bf16_supported()) else torch.float16
    RESULTS = json.loads((RUN_ROOT/'results.json').read_text())
d = RESULTS['dev']
print('best epoch', RESULTS['best_epoch'], '| dev macro-F1', round(d['macro_f1'],4), '| tuned', round(d['macro_f1_tuned'],4),
      f"@ {d['threshold_tuned']:.2f} | P {d['macro_precision']:.3f} R {d['macro_recall']:.3f} acc {d['accuracy']:.3f}")"""

c_export = """# 6 · Score dev + test, write BOTH submissions (threshold 0.5 and dev-tuned) and the probability files for ensembling
dev_res, P_DEV, P_TEST, OUT = J.export_run(model, DATA, C, device, AMP, RUN_ROOT)
print('upload one file, as-is:', {k: v['zip'] for k, v in OUT.items()})"""

c_analysis = """# 7 · Analysis for the report: classes, errors, sections, cost
import pandas as pd
dv = dev_res
print(pd.DataFrame(dv['per_class']).T.round(3).rename(index={'0':'label 0','1':'label 1'}))
print('confusion (rows = gold 0/1, cols = predicted 0/1):', dv['confusion'])
print('\\nfalse positives / false negatives vs correct cases (document length in tokens):')
print(pd.DataFrame(J.error_table(DATA['dev'], P_DEV, 0.5)))
print('\\nwhich part of the judgment matters? (this model, evaluated with different chunk selections, no retraining)')
abl = pd.DataFrame(J.selection_ablation(model, DATA['dev'], C, device, AMP)).pivot(index='select', columns='chunks', values='macro_f1')
display(abl)
if AGG in ('attn','bigru'):
    print('mean attention by position bin (start -> end of the selected chunks):', [round(x,3) for x in J.attention_profile(model, DATA['dev'], C, device, AMP)])
print(f"\\ncost: {RESULTS['total_params']/1e6:.1f}M params, {RESULTS['trainable_params']/1e6:.2f}M trainable, "
      f"peak GPU {RESULTS['peak_gpu_gb']} GB, train {RESULTS['train_minutes']:.1f} min, inference {dv['infer_sec_per_doc']*1000:.0f} ms/case")
print(pd.DataFrame(RESULTS['history'])[['epoch','train_loss','dev_macro_f1','dev_macro_f1_tuned','minutes']].round(4))"""

c_compare = """# 8 · Compare every finished run (for the report's tables)
import pandas as pd
rows = []
for f in sorted(RUNS_ROOT.rglob('results.json')):
    r = json.loads(f.read_text())
    rows.append({'run': f.parent.name, 'preset': r['preset'], 'method': r['method'], 'select': r['setup']['select'],
                 'agg': r['setup']['agg'], 'chunks': r['setup']['max_chunks'],
                 'trainable_M': round(r['trainable_params']/1e6,3), 'total_M': round(r['total_params']/1e6,1),
                 'best_ep': r['best_epoch'], 'dev_F1': round(r['dev']['macro_f1'],4), 'tuned_F1': round(r['dev']['macro_f1_tuned'],4),
                 'P': round(r['dev']['macro_precision'],4), 'R': round(r['dev']['macro_recall'],4),
                 'min': round(r['train_minutes'] or 0,1), 'peak_GB': r['peak_gpu_gb'] and round(r['peak_gpu_gb'],1),
                 'ms/case': round(r['dev']['infer_sec_per_doc']*1000)})
if rows: display(pd.DataFrame(rows).sort_values('dev_F1', ascending=False))"""

HEAD = {
 'run': lambda t, body: md(f"# Phase 4 · CJPE · {t}\n{body}\nNeeds `rr_lib.py` (v3) and `cjpe_lib.py` in `MyDrive/legal-llm-course/`. Input: a whole case, cut into {{chunk}}-token pieces, encoded by the frozen/adapted Phase 1 decoder, pooled by a learned head. Output: `submission.zip` with `prediction.json` = list of `{{id, pred}}` at the zip root (upload the file, not the folder)."),
}
def run_nb(fname, title, body, preset, method, run_id, **kw):
    nb = new_notebook()
    nb.cells += [HEAD['run'](title, body), code(settings(preset, method, run_id, **kw)), code(c_setup), code(c_data_run), code(c_build), code(c_train), code(c_export), code(c_analysis), code(c_compare)]
    for c in nb.cells:
        if c.cell_type == 'code': ast.parse(c.source)
    nbformat.write(nb, f'notebooks/{fname}')

run_nb('phase4_1_mha_pre_full.ipynb', 'MHA-prenorm, full fine-tuning (also the distillation teacher)', 'Technique 1: full fine-tuning.', 'MHA-prenorm', 'full', 'mha-full-001')
run_nb('phase4_2_gqa_lorma.ipynb', 'GQA-prenorm + LoRMA (mandatory technique)', 'Technique 2: LoRMA adapters on query/value, backbone frozen.', 'GQA-prenorm', 'lorma', 'gqa-lorma-001')
run_nb('phase4_3_mla_lora.ipynb', 'MLA-prenorm + LoRA', 'Technique 3: LoRA adapters on MLA query_up/value_up, backbone frozen.', 'MLA-prenorm', 'lora', 'mla-lora-001')
run_nb('phase4_4_gqa_kd_student.ipynb', 'GQA-prenorm 6-layer student, knowledge distillation (mandatory)', 'Technique 4: the student is the Phase 1 GQA model cut to 6 blocks and trained with the cross-entropy + teacher-KL loss; the teacher is the finished full fine-tune from notebook 1 (run that first).', 'GQA-prenorm', 'kd', 'gqa-kd-001')
run_nb('phase4_5_mha_post_partial.ipynb', 'MHA-postnorm, partial fine-tuning (4th Phase 1 model)', 'Technique 5: only the last two blocks + head train. Included so all four Phase 1 architectures are covered; Phase 3 showed this model is the least stable.', 'MHA-postnorm', 'partial', 'mhapost-partial-001')
# architecture controls for the report: the SAME method (LoRMA) and settings as the GQA run (notebook 2) on the other two architectures
run_nb('phase4_7_mha_pre_lorma.ipynb', 'MHA-prenorm + LoRMA (architecture control)', 'Control: identical method, settings and seed to the GQA LoRMA run (notebook 2); only the Phase 1 architecture changes (MHA instead of GQA).', 'MHA-prenorm', 'lorma', 'mha-lorma-001')
run_nb('phase4_8_mla_lorma.ipynb', 'MLA-prenorm + LoRMA (architecture control)', 'Control: identical method, settings and seed to the GQA LoRMA run (notebook 2); only the Phase 1 architecture changes (MLA instead of GQA).', 'MLA-prenorm', 'lorma', 'mla-lorma-001')
run_nb('phase4_9_gqa_lora.ipynb', 'GQA-prenorm + LoRA (method control)', 'Control: identical architecture, settings and seed to the GQA LoRMA run (notebook 2); only the adaptation method changes (LoRA instead of LoRMA).', 'GQA-prenorm', 'lora', 'gqa-lora-001')
run_nb('phase4_10_gqa_full.ipynb', 'GQA-prenorm + full fine-tuning (method control)', 'Control: identical architecture, settings and seed to the GQA LoRMA run (notebook 2); only the adaptation method changes (full fine-tuning instead of LoRMA).', 'GQA-prenorm', 'full', 'gqa-full-001')

# ---- data + baseline notebook (CPU is enough) -------------------------------------------------
nb = new_notebook()
nb.cells.append(md("# Phase 4 · CJPE · data, length statistics and a no-language-model baseline\nRun this first (CPU runtime is enough). It tells you the case lengths in tokens (how many 512-token chunks a case really has), the label balance, and what a TF-IDF + logistic-regression model scores on the dev set using the **whole case**, the **first third**, the **middle third** and the **last third**. The language-model runs have to beat that to show the Phase 1 model adds anything, and the thirds are a cheap first answer to 'which part of the judgment carries the signal?'."))
nb.cells.append(code("""# 1 · Settings
from pathlib import Path
COURSE_ROOT = Path('/content/drive/MyDrive/legal-llm-course')
PHASE_DIR   = COURSE_ROOT/'Phase_4_CJPE'
BUNDLE_ZIP  = COURSE_ROOT/'phase2_lner_multimethod_bundle.zip'
RR_LIB, CJPE_LIB = COURSE_ROOT/'rr_lib.py', COURSE_ROOT/'cjpe_lib.py'
SEED, CHUNK_LEN, MAX_TOKENS = 17, 512, 40000"""))
nb.cells.append(code(c_setup))
nb.cells.append(code(c_data.replace("print('chunks per case at CHUNK_LEN/STRIDE:', int(np.mean([len(J.chunk_spans(len(d[\"tok\"]), CHUNK_LEN, STRIDE)) for d in DATA[\"train\"]])), '(mean) | MAX_CHUNKS =', MAX_CHUNKS)",
 "print('chunks of CHUNK_LEN per train case: mean', round(np.mean([len(J.chunk_spans(len(d['tok']), CHUNK_LEN, CHUNK_LEN)) for d in DATA['train']]),1), '| p90', int(np.percentile([len(J.chunk_spans(len(d['tok']), CHUNK_LEN, CHUNK_LEN)) for d in DATA['train']], 90)))")))
nb.cells.append(code("""# 4 · TF-IDF + logistic regression on the whole case and on each third (no language model, CPU)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import numpy as np, pandas as pd
def part(text, which):
    n = len(text); a, b = {'whole': (0, 1), 'first third': (0, 1/3), 'middle third': (1/3, 2/3), 'last third': (2/3, 1)}[which]
    return text[int(a*n):int(b*n)]
TR = [DATA['train'][i] for i in np.random.RandomState(0).permutation(len(DATA['train']))[:8000]]   # subsample: TF-IDF over all ~31k full cases is slow on CPU
ytr = np.array([d['label'] for d in TR]); ydv = np.array([d['label'] for d in DATA['dev']])
rows = []
for which in ('whole', 'first third', 'middle third', 'last third'):
    vec = TfidfVectorizer(ngram_range=(1,2), min_df=3, max_features=100000, sublinear_tf=True)
    Xtr = vec.fit_transform([part(d['text'], which) for d in TR]); Xdv = vec.transform([part(d['text'], which) for d in DATA['dev']])
    for C_ in (0.3, 1, 3):
        m = LogisticRegression(C=C_, max_iter=400, class_weight='balanced').fit(Xtr, ytr)
        p = m.predict_proba(Xdv)[:, 1]; s = J.macro_scores(ydv, (p >= 0.5).astype(int))
        rows.append({'part': which, 'C': C_, 'macro_F1': round(s['macro_f1'],4), 'P': round(s['macro_precision'],3), 'R': round(s['macro_recall'],3), 'acc': round(s['accuracy'],3)})
res = pd.DataFrame(rows); display(res.sort_values('macro_F1', ascending=False))
print('best per part:'); print(res.groupby('part').macro_F1.max().round(4))
print('majority-class macro-F1 on dev:', round(J.macro_scores(ydv, np.full_like(ydv, int(ytr.mean() >= .5)))['macro_f1'], 4))"""))
for c in nb.cells:
    if c.cell_type == 'code': ast.parse(c.source)
nbformat.write(nb, 'notebooks/phase4_0_data_and_baseline.ipynb')

# ---- ensemble notebook (numpy only, reads the probability files each run exported) --------------
nb = new_notebook()
nb.cells.append(md("# Phase 4 · CJPE · probability-averaging ensemble\nReads `dev_probs.json` / `test_probs.json` exported by each run notebook (cell 6), averages the probabilities, picks the threshold on dev, and writes one `submission.zip`. No GPU, no model loading. Edit `RUN_DIRS`."))
nb.cells.append(code("""from pathlib import Path
import json, zipfile, numpy as np
COURSE_ROOT = Path('/content/drive/MyDrive/legal-llm-course'); RUNS = COURSE_ROOT/'runs/phase4_cjpe'
from google.colab import drive; drive.mount('/content/drive')
RUN_DIRS = [RUNS/'MHA-prenorm/full/mha-full-001', RUNS/'GQA-prenorm/lorma/gqa-lorma-001', RUNS/'MLA-prenorm/lora/mla-lora-001', RUNS/'GQA-prenorm/kd/gqa-kd-001']
WEIGHTS = None            # e.g. [2,1,1,1]
OUT = RUNS/'ensemble/ens-001'; OUT.mkdir(parents=True, exist_ok=True)
PHASE_DIR = COURSE_ROOT/'Phase_4_CJPE'
dev = {d['id']: int(d['label']) for d in json.loads((PHASE_DIR/'Train/dev.json').read_text())} if isinstance(json.loads((PHASE_DIR/'Train/dev.json').read_text()), list) else None
if dev is None:
    raw = json.loads((PHASE_DIR/'Train/dev.json').read_text()); dev = {str(k): int(v.get('label', v.get('labels'))) for k, v in raw.items()}
test_ids = [str(d['id']) for d in json.loads((PHASE_DIR/'Test/test.json').read_text())] if isinstance(json.loads((PHASE_DIR/'Test/test.json').read_text()), list) else list(json.loads((PHASE_DIR/'Test/test.json').read_text()))
P = {s: [json.loads((r/f'{s}_probs.json').read_text()) for r in RUN_DIRS] for s in ('dev', 'test')}
dev_ids = list(P['dev'][0]); assert all(list(p) == dev_ids for p in P['dev']) and set(dev_ids) == set(dev), 'dev ids differ between runs'
assert all(set(p) == set(test_ids) for p in P['test']), 'test ids differ between runs'
y = np.array([dev[i] for i in dev_ids])
def macro_f1(y, p):
    f = []
    for c in (0, 1):
        tp = ((p == c) & (y == c)).sum(); fp = ((p == c) & (y != c)).sum(); fn = ((p != c) & (y == c)).sum()
        pr = tp/(tp+fp) if tp+fp else 0; rc = tp/(tp+fn) if tp+fn else 0; f.append(2*pr*rc/(pr+rc) if pr+rc else 0)
    return float(np.mean(f))
w = np.array(WEIGHTS or [1.0]*len(RUN_DIRS)); w = w/w.sum()
def blend(split, ids): return sum(wi*np.array([p[i] for i in ids]) for wi, p in zip(w, P[split]))
for r, p in zip(RUN_DIRS, P['dev']): print(f'{r.name:24s} dev macro-F1 @0.5 =', round(macro_f1(y, (np.array([p[i] for i in dev_ids]) >= .5).astype(int)), 4))
pd_ = blend('dev', dev_ids)
grid = np.linspace(0.05, 0.95, 91); sc = [macro_f1(y, (pd_ >= t).astype(int)) for t in grid]; thr = float(grid[int(np.argmax(sc))])
print('ENSEMBLE dev macro-F1 @0.5 =', round(macro_f1(y, (pd_ >= .5).astype(int)), 4), '| tuned', round(max(sc), 4), f'@ {thr:.2f}')
pt = blend('test', test_ids)
for tag, t in (('submission', 0.5), ('submission_tuned', thr)):
    d = OUT/tag; d.mkdir(exist_ok=True); preds = [{'id': i, 'pred': int(p >= t)} for i, p in zip(test_ids, pt)]
    (d/'prediction.json').write_text(json.dumps(preds))
    with zipfile.ZipFile(d/'submission.zip', 'w', zipfile.ZIP_DEFLATED) as z: z.write(d/'prediction.json', arcname='prediction.json')
    print(tag, 'threshold', round(t, 2), '| share predicted 1:', round(float(np.mean([p['pred'] for p in preds])), 3), '->', d/'submission.zip')"""))
for c in nb.cells:
    if c.cell_type == 'code': ast.parse(c.source)
nbformat.write(nb, 'notebooks/phase4_ensemble.ipynb')
print('generated')

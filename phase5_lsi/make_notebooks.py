"""Generates the Phase 5 (legal statute identification) Colab notebooks into ./notebooks.
Reuses the Colab setup cell and the Phase 1 snapshot resolver from phase4_cjpe/make_notebooks.py (without regenerating Phase 4)."""
import ast, os
here = os.path.dirname(os.path.abspath(__file__))
p4 = os.path.join(here, '..', 'phase4_cjpe', 'make_notebooks.py')
src = open(p4).read()
ns = {'__file__': p4}
exec(src[:src.index('def settings(')], ns)
c_setup, resolver, presets, md, code, nbformat, new_notebook = (ns[k] for k in ('c_setup', 'resolver', 'presets', 'md', 'code', 'nbformat', 'new_notebook'))
os.chdir(here)
a = "shutil.copy(CJPE_LIB, LOCAL/'cjpe_lib.py')"; b = 'import torch, rr_lib as R, cjpe_lib as J'
assert a in c_setup and b in c_setup
c_setup = c_setup.replace(a, a + "; shutil.copy(LSI_LIB, LOCAL/'lsi_lib.py')").replace(b, b + ', lsi_lib as LSI')


def settings(preset, method, run_id, select='uniform', agg='attn', pool='meanlast'):
    return f"""# 1 · Settings
from pathlib import Path
ACTION   = 'Train'          # Inspect | Train | Predict   (Predict = reload best.pt and rewrite the submissions)
PRESET   = '{preset}'    # GQA-prenorm | MHA-prenorm | MHA-postnorm | MLA-prenorm
METHOD   = '{method}'           # full | partial | lora | rslora | lorma | ia3 | kd
RUN_ID   = '{run_id}'
SEED     = 17

# long-document representation
CHUNK_LEN, STRIDE = 512, 512        # tokens per chunk
MAX_CHUNKS = 8                      # chunks per document used in training/inference (8 chunks cover about 90% of documents fully)
SELECT   = '{select}'      # first | last | uniform | head_tail | random
AGG      = '{agg}'            # mean | attn | bigru      (how chunk vectors are pooled into a document vector)
POOL     = '{pool}'    # meanlast | meanmaxlast     (meanmaxlast adds max-pooling over each chunk's tokens: keeps sparse cues)
MAX_TOKENS = 40000

# multi-label loss
CLASS_WEIGHT = 'sqrt'               # none | sqrt | inverse   (positive-class weights in the BCE loss, per statute)
FOCAL_GAMMA  = 0.0                  # > 0 adds a focal term to the BCE

# budget: each epoch trains on DOCS_PER_EPOCH random documents (of ~42,000); per-epoch evaluation uses DEV_SUBSET dev documents
EPOCHS, DOCS_PER_EPOCH, MAX_MINUTES = 4, 12000, 60   # MAX_MINUTES stops mid-epoch when crossed (best.pt is kept)
BATCH_DOCS, ACCUM, EVAL_DOCS, DEV_SUBSET, LOG_EVERY = 4, 4, 16, 3000, 100

COURSE_ROOT = Path('/content/drive/MyDrive/legal-llm-course')
PHASE_DIR   = COURSE_ROOT/'Phase_5_LSI'
RUNS_ROOT   = COURSE_ROOT/'runs/phase5_lsi'
BUNDLE_ZIP  = COURSE_ROOT/'phase2_lner_multimethod_bundle.zip'   # legal_scaffold, training_methods, tokenizer
RR_LIB, CJPE_LIB, LSI_LIB = COURSE_ROOT/'rr_lib.py', COURSE_ROOT/'cjpe_lib.py', COURSE_ROOT/'lsi_lib.py'

# adapter / method knobs
RANK, ALPHA = 8, 16
LORA_TARGETS = ('query', 'value')
PARTIAL_LAYERS = (10, 11)
# knowledge distillation (METHOD='kd'): frozen teacher = a finished `full` run on the same chunks
KD_TEACHER_RUN = RUNS_ROOT/'MHA-prenorm/full/mha-full-001'
KD_STUDENT_LAYERS = (0, 2, 4, 6, 8, 10)
KD_TEMPERATURE, KD_ALPHA = 2.0, 0.3

DEFAULT_LR = {{  # backbone / adapter / head
 'full': (5e-5, 0, 1e-3), 'kd': (5e-5, 0, 1e-3), 'partial': (1e-4, 0, 1e-3),
 'lora': (0, 1e-3, 1e-3), 'rslora': (0, 1e-3, 1e-3), 'lorma': (0, 1e-3, 1e-3), 'ia3': (0, 3e-3, 1e-3)}}
assert ACTION in ('Inspect','Train','Predict') and METHOD in DEFAULT_LR and SELECT in ('first','last','uniform','head_tail','random') and POOL in ('meanlast','meanmaxlast')
{presets}"""


c_data = """# 3 · Data: tokenised documents come from a cache on Drive (build once with tools/build_token_cache.py, or the first run builds it)
CACHE = PHASE_DIR/f'tokens_lsi_v1_max{MAX_TOKENS}.npz'
DATA = LSI.load_cache(CACHE, TOKENIZER_SHA256, MAX_TOKENS)
if DATA is None:
    from google.colab import userdata
    try: token = userdata.get('HF_TOKEN')
    except Exception: token = None
    LSI.ensure_files(COURSE_ROOT, token)
    DATA = LSI.load_all(PHASE_DIR)
    for s in DATA.values(): J.tokenize_docs(s, TOK, MAX_TOKENS)
    print('saved cache:', LSI.save_cache(DATA, CACHE, TOKENIZER_SHA256, MAX_TOKENS))
else:
    print('loaded tokenised documents from', CACHE)
for s, docs in DATA.items():
    n = np.array([len(d['tok']) for d in docs]); k = np.array([len(d['labels']) for d in docs]) if docs[0]['labels'] is not None else None
    print(f"{s:5s}: {len(docs):6d} documents | tokens mean/median/p90 {int(n.mean())}/{int(np.median(n))}/{int(np.percentile(n,90))}" + (f" | statutes per document mean {k.mean():.2f} max {k.max()}" if k is not None else ' | no labels'))
ids = [d['id'] for d in DATA['test']]
assert len(ids) == len(set(ids)), 'duplicate test ids'
assert all(d['labels'] for d in DATA['train'] + DATA['dev']), 'every train/dev document should have at least one statute'
assert all(d['labels'] is None for d in DATA['test']), 'test labels must not be loaded'
print('chunks per train document (mean):', round(np.mean([len(J.chunk_spans(len(d['tok']), CHUNK_LEN, STRIDE)) for d in DATA['train'][:5000]]), 1), '| MAX_CHUNKS =', MAX_CHUNKS)"""

c_build = """# 4 · Build run config + model from the Phase 1 snapshot
lb, la, lh = DEFAULT_LR[METHOD]
C = dict(method=METHOD, preset=PRESET, run_id=RUN_ID, class_weight=CLASS_WEIGHT, focal_gamma=FOCAL_GAMMA, epochs=EPOCHS, seed=SEED,
         lr=dict(backbone=lb, adapter=la, head=lh), partial_layers=PARTIAL_LAYERS,
         chunk_len=CHUNK_LEN, stride=STRIDE, max_chunks=MAX_CHUNKS, select=SELECT, agg=AGG, pool=POOL,
         docs_per_epoch=DOCS_PER_EPOCH, max_minutes=MAX_MINUTES, batch_docs=BATCH_DOCS, accum=ACCUM, eval_docs=EVAL_DOCS, dev_subset=DEV_SUBSET, log_every=LOG_EVERY,
         kd_temperature=KD_TEMPERATURE, kd_alpha=KD_ALPHA,
         head_kw=dict(n_classes=LSI.N_LABELS, hidden=256, agg=AGG, dropout=0.1, pool=POOL),
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
    print(f'teacher {tC["preset"]} {n_t:,} params -> student {PRESET}/{len(KD_STUDENT_LAYERS)} layers {n_s:,} params')"""

c_train = """# 5 · Train (resumes automatically from last.pt after a Colab disconnect)
if ACTION == 'Train':
    RESULTS, AMP, P_DEV = LSI.train_run(model, DATA, RUN_ROOT, C, device, teacher=teacher)
else:
    saved = torch.load(RUN_ROOT/'best.pt', map_location='cpu', weights_only=False)
    model.load_state_dict(saved['trainable'], strict=False); model.to(device)
    AMP = torch.bfloat16 if (device.type=='cuda' and torch.cuda.is_bf16_supported()) else torch.float16
    RESULTS = json.loads((RUN_ROOT/'results.json').read_text()); P_DEV = None
print('best epoch', RESULTS['best_epoch'], '| trainable', f"{RESULTS['trainable_params']:,}", '| train minutes', round(RESULTS['train_minutes'] or 0, 1))"""

c_export = """# 6 · Score dev + test; write both submissions and the probability files (for ensembling)
import pandas as pd
EV, P_DEV, P_TEST, OUT = LSI.export_run(model, DATA, C, device, AMP, RUN_ROOT, P_dev=P_DEV)
rows = [{'threshold strategy': k, **{m: round(v[m], 4) for m in ('macro_f1', 'micro_f1', 'subset_acc', 'macro_precision', 'macro_recall', 'labels_per_doc_pred')}} for k, v in EV.items() if isinstance(v, dict)]
display(pd.DataFrame(rows)); print('global threshold:', EV['global_threshold'])
print('upload one file, as-is:', {k: v['zip'] for k, v in OUT.items()})"""

c_analysis = """# 7 · Analysis for the report: rare statutes, thresholds, confusions, label relations, sections, cost
Y = LSI.multi_hot(DATA['dev']); g = EV['global_threshold']
tab = pd.DataFrame(LSI.per_label_table(Y, P_DEV, g, DATA['train']))
print('five hardest / five easiest statutes (global threshold, dev):'); display(pd.concat([tab.sort_values('F1').head(5), tab.sort_values('F1').tail(5)]))
tab['train-frequency quartile'] = pd.qcut(tab.train_docs.rank(method='first'), 4, labels=['Q1 rarest', 'Q2', 'Q3', 'Q4 most common'])
print('mean F1 by how common the statute is in training:'); print(tab.groupby('train-frequency quartile', observed=True).F1.mean().round(4))
print('rank correlation of per-statute F1 with training frequency:', round(tab.F1.corr(tab.train_docs, method='spearman'), 3))
pred = LSI.decide(P_DEV, g, True)
print('\\nstatute pairs most often confused (missed one, predicted another, same document):'); display(pd.DataFrame(LSI.confusion_pairs(Y, pred, 12)))
card = Y.sum(1); bucket = pd.cut(card, [0, 1, 2, 4, 8, 100], labels=['1', '2', '3-4', '5-8', '9+'])
print('exact-match accuracy and micro-F1 by number of true statutes in the document:')
display(pd.DataFrame([{'true statutes': b, 'documents': int((bucket == b).sum()), 'exact match': round(float((pred[bucket == b] == Y[bucket == b].astype(bool)).all(1).mean()), 3)} for b in bucket.categories]))
S2 = LSI.stack_cooccurrence_cv(Y, P_DEV)
print('\\nDo label relationships help? second-stage model on the 100 probabilities (cross-validated on dev):')
print('  independent predictions, global threshold:', round(LSI.scores(Y, LSI.decide(P_DEV, g, True))['macro_f1'], 4), '| stacked on label probabilities:', round(LSI.scores(Y, LSI.decide(S2, LSI.tune_global(Y, S2)[0], True))['macro_f1'], 4))
print('\\nwhich part of a document matters? (this model, other chunk selections, 2,000 dev documents, no retraining)')
abl = pd.DataFrame(LSI.selection_ablation(model, DATA['dev'][:2000], C, device, AMP)).pivot(index='select', columns='chunks', values='macro_f1_global_tuned'); display(abl)
print(f"\\ncost: {RESULTS['total_params']/1e6:.1f}M params, {RESULTS['trainable_params']/1e6:.2f}M trainable, peak GPU {RESULTS['peak_gpu_gb']} GB, "
      f"train {RESULTS['train_minutes']:.1f} min, inference {RESULTS['infer_sec_per_doc']*1000:.0f} ms/document")
print(pd.DataFrame(RESULTS['history'])[['epoch', 'train_loss', 'dev_macro_f1_tuned', 'dev_micro_f1_tuned', 'minutes']].round(4))"""

c_compare = """# 8 · Compare every finished run (for the report's tables)
rows = []
for f in sorted(RUNS_ROOT.rglob('results.json')):
    r = json.loads(f.read_text()); d = r['dev']
    rows.append({'run': f.parent.name, 'preset': r['preset'], 'method': r['method'], 'select': r['setup']['select'], 'agg': r['setup']['agg'], 'weights': r['setup']['class_weight'],
                 'trainable_M': round(r['trainable_params']/1e6, 3), 'total_M': round(r['total_params']/1e6, 1), 'best_ep': r['best_epoch'],
                 'macro_F1 (global)': round(d['global_tuned+min_one']['macro_f1'], 4), 'macro_F1 (per-label, cross-fit)': round(d['per_label_cross_fit+min_one']['macro_f1'], 4),
                 'micro_F1': round(d['global_tuned+min_one']['micro_f1'], 4), 'subset_acc': round(d['global_tuned+min_one']['subset_acc'], 4),
                 'min': round(r['train_minutes'] or 0, 1), 'peak_GB': r['peak_gpu_gb'] and round(r['peak_gpu_gb'], 1), 'ms/doc': round(r['infer_sec_per_doc']*1000)})
if rows: display(pd.DataFrame(rows).sort_values('macro_F1 (global)', ascending=False))"""


def run_nb(fname, title, body, preset, method, run_id, **kw):
    nb = new_notebook()
    intro = md(f"# Phase 5 · Legal statute identification · {title}\n{body}\nNeeds `rr_lib.py`, `cjpe_lib.py` and `lsi_lib.py` in `MyDrive/legal-llm-course/`. Multi-label: a document gets a SET of statute ids "
               "(100 statutes). Output: `submission.zip` with `prediction.json` = list of `{id, pred: [statute ids]}` at the zip root (upload the file, not the folder).")
    nb.cells += [intro, code(settings(preset, method, run_id, **kw)), code(c_setup), code(c_data), code(c_build), code(c_train), code(c_export), code(c_analysis), code(c_compare)]
    for c in nb.cells:
        if c.cell_type == 'code': ast.parse(c.source)
    nbformat.write(nb, f'notebooks/{fname}')


run_nb('phase5_1_mha_pre_full.ipynb', 'MHA-prenorm, full fine-tuning (also the distillation teacher)', 'Technique 1: full fine-tuning.', 'MHA-prenorm', 'full', 'mha-full-001')
run_nb('phase5_2_gqa_lorma.ipynb', 'GQA-prenorm + LoRMA (mandatory technique)', 'Technique 2: LoRMA adapters on query/value, backbone frozen.', 'GQA-prenorm', 'lorma', 'gqa-lorma-001')
run_nb('phase5_3_mla_lora.ipynb', 'MLA-prenorm + LoRA', 'Technique 3: LoRA adapters on MLA query_up/value_up, backbone frozen.', 'MLA-prenorm', 'lora', 'mla-lora-001')
run_nb('phase5_4_gqa_kd_student.ipynb', 'GQA-prenorm 6-layer student, knowledge distillation (mandatory)',
       'Technique 4: the student is the Phase 1 GQA model cut to 6 blocks, trained with BCE on the gold statutes plus a soft-target BCE on the teacher probabilities; the teacher is the finished full fine-tune from notebook 1 (run that first).', 'GQA-prenorm', 'kd', 'gqa-kd-001')
run_nb('phase5_5_mha_post_partial.ipynb', 'MHA-postnorm, partial fine-tuning (4th Phase 1 model)', 'Technique 5: only the last two blocks + head train. Included so all four Phase 1 architectures are covered; Phase 3 and 4 showed this model is the least stable.', 'MHA-postnorm', 'partial', 'mhapost-partial-001')
print('generated run notebooks')


# ---------------------------------------------------------------- clean-slate pretrained models (pretrain/): ONE recipe for all three, so the models are the only difference
NEW_MODELS = [  # preset key, checkpoint under COURSE_ROOT, notebook, run id, title
    ('NEW-GQA', 'pretrain/pretrain_dec_gqa/model_final.pt', 'phase5_6_new_gqa_full.ipynb', 'new-gqa-full-001', 'new GQA causal decoder (4B tokens)'),
    ('NEW-ENC', 'pretrain/pretrain_enc_mlm/model_final.pt', 'phase5_7_new_encoder_full.ipynb', 'new-enc-full-001', 'new bidirectional MLM encoder (4B tokens)'),
    ('NEW-HYB', 'pretrain/pretrain_hybrid/model_final.pt', 'phase5_8_new_hybrid_full.ipynb', 'new-hyb-full-001', 'new Hymba-style hybrid (4B tokens)')]
c_setup_new = c_setup.replace("shutil.copy(LSI_LIB, LOCAL/'lsi_lib.py')", "shutil.copy(LSI_LIB, LOCAL/'lsi_lib.py')\n"
                              "for _f in ('hybrid.py', 'encoder.py'): shutil.copy(COURSE_ROOT/_f, LOCAL/_f)   # rr_lib.load_backbone imports these for the hybrid / encoder checkpoints")
assert c_setup_new != c_setup
SEED2_NB = {'NEW-GQA': 'phase5_9_new_gqa_full_seed18.ipynb', 'NEW-ENC': 'phase5_10_new_encoder_full_seed18.ipynb', 'NEW-HYB': 'phase5_11_new_hybrid_full_seed18.ipynb'}
RUNS_BY_SEED = [(17, lambda k, f, r: (f, r)), (18, lambda k, f, r: (SEED2_NB[k], r.replace('-001', '-002')))]     # the second seed changes the document draws and the head init, nothing else
for seed, name_of in RUNS_BY_SEED:
  for key, ck, fname0, run_id0, title in NEW_MODELS:
    fname, run_id = name_of(key, fname0, run_id0)
    s = settings('GQA-prenorm', 'full', run_id).replace("PRESET   = 'GQA-prenorm'", f"PRESET   = '{key}'").replace('SEED     = 17', f'SEED     = {seed}')
    s = s.replace('EPOCHS, DOCS_PER_EPOCH, MAX_MINUTES = 4, 12000, 60', 'EPOCHS, DOCS_PER_EPOCH, MAX_MINUTES = 4, 12000, 120')      # same budget for every model: the time cap must not bind on the slower one
    s += ("\n# the new checkpoints carry their own config and objective (rr_lib.load_backbone), so these entries only name the preset\n"
          "for _k in ('NEW-GQA', 'NEW-ENC', 'NEW-HYB'): PRESETS[_k] = PRESETS['GQA-prenorm']")
    build = c_build.replace('def build(C):', "PHASE1_OVERRIDE.update({" + ', '.join(f"'{k}': str(COURSE_ROOT/'{c}')" for k, c, *_ in NEW_MODELS) + "})\n\ndef build(C):", 1)
    assert build != c_build
    nb = new_notebook()
    nb.cells += [md(f"# Phase 5 · Legal statute identification · {title}, full fine-tuning\nIdentical recipe for the three new pretrained models (seed {seed}, same chunks, same schedule, 4 x 12,000 documents). "
                    "Only the checkpoint differs, so the dev scores compare the models. Needs `hybrid.py` and `encoder.py` next to `rr_lib.py` in `MyDrive/legal-llm-course/` "
                    "and the finished `model_final.pt` of the matching pretraining run."),
                 code(s), code(c_setup_new), code(c_data), code(build), code(c_train), code(c_export), code(c_analysis), code(c_compare)]
    for c in nb.cells:
        if c.cell_type == 'code': ast.parse(c.source)
    nbformat.write(nb, f'notebooks/{fname}')
print('generated the six new-model notebooks (seeds 17 and 18)')


# ---------------------------------------------------------------- data + TF-IDF baseline (CPU)
nb = new_notebook()
nb.cells.append(md("# Phase 5 · data statistics and a no-language-model baseline (CPU runtime is enough)\nRun this first. It prints document lengths in tokens, statutes per document and how common each statute is, then scores **TF-IDF + a one-vs-rest linear model** on dev. "
                   "The language-model runs have to beat that to show the Phase 1 model adds anything. It also compares one global threshold with per-statute thresholds, with the per-statute thresholds scored honestly (tuned on one half of dev, scored on the other)."))
nb.cells.append(code("""# 1 · Settings
from pathlib import Path
SEED, N_TRAIN, N_DEV, MAX_TOKENS = 17, 12000, 3000, 40000
COURSE_ROOT = Path('/content/drive/MyDrive/legal-llm-course')
PHASE_DIR   = COURSE_ROOT/'Phase_5_LSI'
BUNDLE_ZIP  = COURSE_ROOT/'phase2_lner_multimethod_bundle.zip'
RR_LIB, CJPE_LIB, LSI_LIB = COURSE_ROOT/'rr_lib.py', COURSE_ROOT/'cjpe_lib.py', COURSE_ROOT/'lsi_lib.py'"""))
nb.cells.append(code(c_setup))
nb.cells.append(code("""# 3 · Data (downloads Phase_5_LSI from the gated HF dataset if not already on Drive) + statistics
import pandas as pd
from google.colab import userdata
try: token = userdata.get('HF_TOKEN')
except Exception: token = None
LSI.ensure_files(COURSE_ROOT, token)
TEXT = LSI.load_all(PHASE_DIR)
for s, docs in TEXT.items(): print(f'{s}: {len(docs)} documents')
sample = TEXT['train'][:3000]; J.tokenize_docs(sample, TOK, MAX_TOKENS); n = np.array([len(d['tok']) for d in sample])
print('tokens per document (first 3,000 train): mean/median/p90/max', int(n.mean()), int(np.median(n)), int(np.percentile(n, 90)), int(n.max()), '| 512-token chunks: mean', round(np.mean([len(J.chunk_spans(len(d['tok']), 512, 512)) for d in sample]), 1), '| documents fully covered by 8 chunks:', f"{100*np.mean([len(J.chunk_spans(len(d['tok']), 512, 512)) <= 8 for d in sample]):.0f}%")
Ytr_all = LSI.multi_hot(TEXT['train']); freq = Ytr_all.sum(0)
print('statutes per document (train): mean', round(Ytr_all.sum(1).mean(), 2), '| documents per statute: min', int(freq.min()), 'median', int(np.median(freq)), 'max', int(freq.max()))"""))
nb.cells.append(code("""# 4 · TF-IDF + one-vs-rest LinearSVC baseline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
rng = np.random.RandomState(SEED); tr = [TEXT['train'][i] for i in rng.permutation(len(TEXT['train']))[:N_TRAIN]]; dv = [TEXT['dev'][i] for i in rng.permutation(len(TEXT['dev']))[:N_DEV]]
ytr, ydv = LSI.multi_hot(tr), LSI.multi_hot(dv)
vec = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=200000, sublinear_tf=True, dtype=np.float32)
A = vec.fit_transform([d['text'] for d in tr]); B = vec.transform([d['text'] for d in dv]); print('tf-idf', A.shape)
rows = []
for C_ in (0.1, 0.5):
    S = OneVsRestClassifier(LinearSVC(C=C_), n_jobs=-1).fit(A, ytr).decision_function(B); P = 1 / (1 + np.exp(-4 * S))     # squash scores to (0,1) so the threshold grid applies
    ev = LSI.evaluate_probs(ydv, P)
    for k in ('fixed_0.5', 'global_tuned', 'global_tuned+min_one', 'per_label_cross_fit+min_one'):
        rows.append({'C': C_, 'threshold strategy': k, **{m: round(ev[k][m], 4) for m in ('macro_f1', 'micro_f1', 'subset_acc')}})
display(pd.DataFrame(rows))
f1 = np.array(LSI.scores(ydv, LSI.decide(P, ev['global_threshold'], True))['per_label_f1']); order = np.argsort(f1)
print('worst 5 statutes (id, F1):', [(int(k), round(float(f1[k]), 2)) for k in order[:5]], '| best 5:', [(int(k), round(float(f1[k]), 2)) for k in order[-5:]])"""))
for c in nb.cells:
    if c.cell_type == 'code': ast.parse(c.source)
nbformat.write(nb, 'notebooks/phase5_0_data_and_baseline.ipynb')

# ---------------------------------------------------------------- TF-IDF variants (CPU): the lexical model that the ensemble mixes with the neural runs
nb = new_notebook()
nb.cells.append(md("# Phase 5 · TF-IDF variants for the ensemble (CPU, High-RAM runtime)\nThe ensemble notebook found that mixing a TF-IDF + linear model into the neural runs adds macro-F1. This notebook trains several versions of that model "
                   "(feature sets x regularisation x class weighting), scores each alone and mixed with the neural ensemble (both with honest cross-fitted thresholds on dev), saves every version's probabilities, "
                   "and copies the best mixed one to `runs/phase5_lsi/tfidf/probs.npz`, which is what the ensemble notebook loads. Every fit prints its time; trim `FEATURES` / `MODELS` if you are short of time."))
nb.cells.append(code("""# 1 · Settings
from pathlib import Path
SEED = 17
COURSE_ROOT = Path('/content/drive/MyDrive/legal-llm-course'); PHASE_DIR = COURSE_ROOT/'Phase_5_LSI'; RUNS_ROOT = COURSE_ROOT/'runs/phase5_lsi'
BUNDLE_ZIP  = COURSE_ROOT/'phase2_lner_multimethod_bundle.zip'
RR_LIB, CJPE_LIB, LSI_LIB = COURSE_ROOT/'rr_lib.py', COURSE_ROOT/'cjpe_lib.py', COURSE_ROOT/'lsi_lib.py'
# feature sets: name -> (ngram range, max features, min document frequency, characters kept per document)
FEATURES = {'w12': ((1, 2), 200000, 3, 60000), 'w13': ((1, 3), 400000, 3, 120000)}
# linear models on top: name -> (LinearSVC C, class weighting)   'balanced' up-weights the rare statutes, which is what macro-F1 rewards
MODELS = {'svm0.1': (0.1, None), 'svm0.5': (0.5, None), 'svm0.1bal': (0.1, 'balanced'), 'svm0.5bal': (0.5, 'balanced')}
MIX_WEIGHT, SHRINK_M = 0.5, 30       # how each version is judged when mixed with the neural ensemble (the best setting found in the ensemble notebook)
CANDIDATES = [RUNS_ROOT/f'NEW-{m}/full/new-{n}-full-00{i}' for i in (1, 2) for m, n in (('GQA', 'gqa'), ('ENC', 'enc'), ('HYB', 'hyb'))]"""))
nb.cells.append(code(c_setup))
nb.cells.append(code("""# 3 · Data, and the neural ensemble's dev probabilities (the thing each TF-IDF version is mixed with)
import time, pandas as pd
tr_docs = LSI.load_split(PHASE_DIR/'Train/train.json', dedup=True); dev_docs = LSI.load_split(PHASE_DIR/'Train/dev.json', dedup=True); test_docs = LSI.load_split(PHASE_DIR/'Test/test.json', keep_labels=False)
ytr, Y = LSI.multi_hot(tr_docs), LSI.multi_hot(dev_docs); dev_ids = [d['id'] for d in dev_docs]; test_ids = [d['id'] for d in test_docs]
RUN_DIRS = [r for r in CANDIDATES if (r/'dev_probs.npy').exists()]
print('neural runs found:', [r.name for r in RUN_DIRS])
P_NN = np.mean([np.load(r/'dev_probs.npy') for r in RUN_DIRS], 0) if RUN_DIRS else None
def cross_fit_mix(P_tf, wt=MIX_WEIGHT, m=SHRINK_M, seeds=5):
    # mean macro-F1 over repeated 2-fold splits of dev: thresholds (shrunk toward the global one) tuned on one half, scored on the other
    P = P_tf if P_NN is None else (1 - wt) * P_NN + wt * P_tf; out = []
    for sd in range(seeds):
        halves = np.array_split(np.random.RandomState(sd).permutation(len(Y)), 2)
        for a, b in ((0, 1), (1, 0)):
            tr, te = halves[a], halves[b]; f = LSI._f1_by_threshold(Y[tr], P[tr]); n = Y[tr].sum(0)
            g = LSI.GRID[int(np.argmax(f.mean(1)))]; t = LSI.GRID[np.argmax(f, 0)]; lam = n / (n + m)
            out.append(LSI.scores(Y[te], LSI.decide(P[te], lam * t + (1 - lam) * g, True))['macro_f1'])
    return float(np.mean(out))"""))
nb.cells.append(code("""# 4 · Train every version (the slow cell: vectorising the 42,000 training documents dominates, then each fit takes a few minutes)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
VAR = RUNS_ROOT/'tfidf/variants'; VAR.mkdir(parents=True, exist_ok=True)
sig = lambda S: (1 / (1 + np.exp(-4 * S))).astype(np.float32)        # squash scores to (0,1), as in the baseline notebook
rows = []
for fname, (ngram, maxf, mindf, chars) in FEATURES.items():
    t0 = time.time(); cut = lambda ds: [d['text'][:chars] for d in ds]
    vec = TfidfVectorizer(ngram_range=ngram, min_df=mindf, max_features=maxf, sublinear_tf=True, dtype=np.float32)
    A = vec.fit_transform(cut(tr_docs)); B = vec.transform(cut(dev_docs)); T = vec.transform(cut(test_docs))
    print(f'{fname}: matrix {A.shape} in {time.time() - t0:.0f} s', flush=True)
    for mname, (C_, cw) in MODELS.items():
        name = f'{fname}_{mname}'; f = VAR/f'{name}.npz'
        if f.exists() and list(np.load(f)['dev_ids']) == dev_ids:
            z = np.load(f); P_dev, P_test = z['dev'], z['test']; how = 'cached'
        else:
            t1 = time.time(); clf = OneVsRestClassifier(LinearSVC(C=C_, class_weight=cw), n_jobs=-1).fit(A, ytr)
            P_dev, P_test = sig(clf.decision_function(B)), sig(clf.decision_function(T)); how = f'{time.time() - t1:.0f} s'
            np.savez(f, dev=P_dev, test=P_test, dev_ids=np.array(dev_ids), test_ids=np.array(test_ids), name=name)
        alone = LSI.evaluate_probs(Y, P_dev)['per_label_cross_fit+min_one']['macro_f1']; mixed = cross_fit_mix(P_dev)
        rows.append({'version': name, 'alone (per-label cross-fit)': round(alone, 4), f'mixed with neural ensemble (w={MIX_WEIGHT})': round(mixed, 4), 'fit': how}); print(rows[-1], flush=True)"""))
nb.cells.append(code("""# 5 · Pick the version that helps the ensemble most and hand it to the ensemble notebook
col = f'mixed with neural ensemble (w={MIX_WEIGHT})'
tab = pd.DataFrame(rows).sort_values(col, ascending=False); display(tab)
best = tab.iloc[0]['version']; z = np.load(VAR/f'{best}.npz')
np.savez(RUNS_ROOT/'tfidf/probs.npz', dev=z['dev'], test=z['test'], dev_ids=z['dev_ids'], test_ids=z['test_ids'], name=best)
print('best version:', best, '-> copied to', RUNS_ROOT/'tfidf/probs.npz', '(the ensemble notebook loads it from there; rerun its cells 5-7)')"""))
for c in nb.cells:
    if c.cell_type == 'code': ast.parse(c.source)
nbformat.write(nb, 'notebooks/phase5_tfidf_variants.ipynb')
print('generated the TF-IDF variants notebook')


# ---------------------------------------------------------------- ensemble (reads each run's saved probabilities)
nb = new_notebook()
nb.cells.append(md("# Phase 5 · probability-averaging ensemble\nReads `dev_probs.npy` / `test_probs.npy` written by each run notebook's cell 6, averages them, tunes thresholds on dev and writes two more submissions. CPU only. "
                   "Scores are reported with per-statute thresholds estimated honestly (tuned on half of dev, scored on the other half), because choosing runs AND thresholds on the same dev data flatters every number."))
nb.cells.append(code("""# 1 · Settings
from pathlib import Path
SEED = 17
COURSE_ROOT = Path('/content/drive/MyDrive/legal-llm-course'); PHASE_DIR = COURSE_ROOT/'Phase_5_LSI'; RUNS_ROOT = COURSE_ROOT/'runs/phase5_lsi'
BUNDLE_ZIP  = COURSE_ROOT/'phase2_lner_multimethod_bundle.zip'
RR_LIB, CJPE_LIB, LSI_LIB = COURSE_ROOT/'rr_lib.py', COURSE_ROOT/'cjpe_lib.py', COURSE_ROOT/'lsi_lib.py'
CANDIDATES = [RUNS_ROOT/f'NEW-{m}/full/new-{n}-full-00{i}' for i in (1, 2) for m, n in (('GQA', 'gqa'), ('ENC', 'enc'), ('HYB', 'hyb'))]
# add the older runs here to include them (every subset of RUN_DIRS is scored, so keep the list short): RUNS_ROOT/'MHA-prenorm/full/mha-full-001', ...
WEIGHTS = None            # e.g. [2, 1, 1]
OUT = RUNS_ROOT/'ensemble/ens-001'"""))
nb.cells.append(code(c_setup))
nb.cells.append(code("""# 3 · Load every run's probabilities and check that ids line up with the official files
RUN_DIRS = [r for r in CANDIDATES if (r/'dev_probs.npy').exists()]      # runs that have finished (checked here, after Drive is mounted in cell 2)
print('ensembling', [r.name for r in RUN_DIRS])
for r in CANDIDATES:
    if r not in RUN_DIRS: print('  not found (no dev_probs.npy):', r, '| folder exists:', r.exists(), '| files:', sorted(x.name for x in r.glob('*'))[:12] if r.exists() else '-')
assert RUN_DIRS, 'no finished runs found under RUNS_ROOT: check the paths printed above (run folders are RUNS_ROOT/<PRESET>/full/<run id>)'
import itertools, pandas as pd
dev_docs = LSI.load_split(PHASE_DIR/'Train/dev.json', dedup=True); test_docs = LSI.load_split(PHASE_DIR/'Test/test.json', keep_labels=False)
Y = LSI.multi_hot(dev_docs); dev_ids = [d['id'] for d in dev_docs]; test_ids = [d['id'] for d in test_docs]
PD = [np.load(r/'dev_probs.npy') for r in RUN_DIRS]; PT = [np.load(r/'test_probs.npy') for r in RUN_DIRS]
for r in RUN_DIRS:
    assert json.loads((r/'dev_ids.json').read_text()) == dev_ids and json.loads((r/'test_ids.json').read_text()) == test_ids, f'{r.name}: ids differ from the official files'
w = np.array(WEIGHTS or [1.0]*len(RUN_DIRS)); w = w / w.sum()
def row(name, P):
    ev = LSI.evaluate_probs(Y, P); return {'members': name, 'macro_F1 (global thr)': round(ev['global_tuned+min_one']['macro_f1'], 4), 'macro_F1 (per-label, cross-fit)': round(ev['per_label_cross_fit+min_one']['macro_f1'], 4), 'micro_F1': round(ev['global_tuned+min_one']['micro_f1'], 4)}
rows = [row(r.name, p) for r, p in zip(RUN_DIRS, PD)]
for n in range(2, len(RUN_DIRS) + 1):
    for sub in itertools.combinations(range(len(RUN_DIRS)), n): rows.append(row('+'.join(RUN_DIRS[i].name for i in sub), np.mean([PD[i] for i in sub], 0)))
display(pd.DataFrame(rows).sort_values('macro_F1 (per-label, cross-fit)', ascending=False))"""))
nb.cells.append(code("""# 4 · Write the ensemble submissions (all runs in RUN_DIRS, weights as set above)
P_dev = sum(wi * p for wi, p in zip(w, PD)); P_test = sum(wi * p for wi, p in zip(w, PT))
g, _ = LSI.tune_global(Y, P_dev); th = LSI.tune_per_label(Y, P_dev)
for tag, thr in (('submission_global', g), ('submission_perlabel', th)):
    pred = LSI.decide(P_test, thr, True); path = LSI.write_submission(P_test, test_ids, pred, OUT/tag)
    print(tag, '| statutes per document', round(float(pred.sum(1).mean()), 2), '->', path)
print('dev macro-F1 of this ensemble (global threshold, in-sample):', round(LSI.scores(Y, LSI.decide(P_dev, g, True))['macro_f1'], 4))"""))
nb.cells.append(code("""# 5 · TF-IDF + linear model as one more ensemble member (CPU; the first run takes a while, then it is cached on Drive)
USE_TFIDF = True
TF_DEV = TF_TEST = None
if USE_TFIDF:
    import time
    TF_CACHE = RUNS_ROOT/'tfidf/probs.npz'
    z = np.load(TF_CACHE) if TF_CACHE.exists() else None
    if z is not None and list(z['dev_ids']) == dev_ids and list(z['test_ids']) == test_ids:
        TF_DEV, TF_TEST = z['dev'], z['test']; print('loaded cached TF-IDF probabilities:', str(z['name']) if 'name' in z.files else 'first baseline version')
    else:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.multiclass import OneVsRestClassifier
        from sklearn.svm import LinearSVC
        t0 = time.time()
        tr_docs = LSI.load_split(PHASE_DIR/'Train/train.json', dedup=True); ytr = LSI.multi_hot(tr_docs)
        cut = lambda ds: [d['text'][:60000] for d in ds]
        vec = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=200000, sublinear_tf=True, dtype=np.float32)
        A = vec.fit_transform(cut(tr_docs)); print('tf-idf matrix', A.shape, round(time.time() - t0), 's')
        clf = OneVsRestClassifier(LinearSVC(C=0.1), n_jobs=-1).fit(A, ytr)
        sig = lambda S: (1 / (1 + np.exp(-4 * S))).astype(np.float32)        # squash scores to (0,1), same as the baseline notebook
        TF_DEV, TF_TEST = sig(clf.decision_function(vec.transform(cut(dev_docs)))), sig(clf.decision_function(vec.transform(cut(test_docs))))
        TF_CACHE.parent.mkdir(parents=True, exist_ok=True)
        np.savez(TF_CACHE, dev=TF_DEV, test=TF_TEST, dev_ids=np.array(dev_ids), test_ids=np.array(test_ids)); print('done in', round(time.time() - t0), 's')
    ev = LSI.evaluate_probs(Y, TF_DEV)
    print('TF-IDF alone on dev: macro-F1 global', round(ev['global_tuned+min_one']['macro_f1'], 4), '| per-label cross-fit', round(ev['per_label_cross_fit+min_one']['macro_f1'], 4))"""))
nb.cells.append(code("""# 6 · Two cheap improvements, both chosen by cross-fitting on dev: (a) how much TF-IDF to mix in, (b) how far to shrink each statute's threshold toward the global one
# threshold for statute k = lam_k * (its own tuned threshold) + (1 - lam_k) * global,  lam_k = n_k / (n_k + m),  n_k = positives of that statute in the tuning half
# m = 0 is plain per-label thresholds, m = inf is one global threshold. Rare statutes have noisy own-thresholds, so they lean on the global value.
WS = (0.0, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0) if TF_DEV is not None else (0.0,)      # 1.0 = TF-IDF alone; if the best weight is at the top of this list, extend it
MS = (0, 30, 100, 300, float('inf'))
def mix(Pn, Pt, wt): return Pn if wt == 0 else (1 - wt) * Pn + wt * Pt
def thresholds(f, n, m):
    g = LSI.GRID[int(np.argmax(f.mean(1)))]; t = LSI.GRID[np.argmax(f, 0)]
    lam = np.ones_like(t) if m == 0 else (np.zeros_like(t) if not np.isfinite(m) else n / (n + m))
    return lam * t + (1 - lam) * g
res = {}
for wt in WS:
    P = mix(P_dev, TF_DEV, wt)
    for sd in range(5):                                                        # five random 2-fold splits of dev
        halves = np.array_split(np.random.RandomState(sd).permutation(len(Y)), 2)
        for a, b in ((0, 1), (1, 0)):
            tr, te = halves[a], halves[b]; f = LSI._f1_by_threshold(Y[tr], P[tr]); n = Y[tr].sum(0)
            for m in MS: res.setdefault((wt, m), []).append(LSI.scores(Y[te], LSI.decide(P[te], thresholds(f, n, m), True))['macro_f1'])
tab = pd.DataFrame({(('inf' if not np.isfinite(m) else m)): {wt: np.mean(res[(wt, m)]) for wt in WS} for m in MS}).round(4)
tab.index.name = 'TF-IDF weight'; tab.columns.name = 'shrinkage m (0 = per-label, inf = global)'
display(tab)
best = max(res, key=lambda k: np.mean(res[k])); base = np.mean(res[(0.0, 0)])
print('best (TF-IDF weight, m):', best, '| cross-fit macro-F1', round(float(np.mean(res[best])), 4), '| plain neural ensemble with per-label thresholds', round(float(base), 4))"""))
nb.cells.append(code("""# 7 · Write the tuned ensemble submission (settings from cell 6; thresholds tuned on all of dev)
wt, m = best
P_d, P_t = mix(P_dev, TF_DEV, wt), mix(P_test, TF_TEST, wt)
thr = thresholds(LSI._f1_by_threshold(Y, P_d), Y.sum(0), m)
pred = LSI.decide(P_t, thr, True); OUT2 = RUNS_ROOT/'ensemble/ens-002'
path = LSI.write_submission(P_t, test_ids, pred, OUT2/'submission_tuned')
print('TF-IDF weight', wt, '| m', m, '| statutes per document', round(float(pred.sum(1).mean()), 2), '->', path)
print('dev macro-F1 with these thresholds (in-sample, flattering):', round(LSI.scores(Y, LSI.decide(P_d, thr, True))['macro_f1'], 4))"""))
for c in nb.cells:
    if c.cell_type == 'code': ast.parse(c.source)
nbformat.write(nb, 'notebooks/phase5_ensemble.ipynb')
print('generated baseline + ensemble notebooks')

"""Generates the Phase 6 (abstractive summarization) Colab notebooks into ./notebooks. One template; the runs differ only in cell 1."""
import os
import nbformat
from nbformat.v4 import new_notebook, new_code_cell as code, new_markdown_cell as md

here = os.path.dirname(os.path.abspath(__file__)); os.chdir(here)

DEFAULT_LR = "{'full': dict(backbone=3e-5, adapter=0), 'kd': dict(backbone=5e-5, adapter=0), 'partial': dict(backbone=5e-5, adapter=0),\n             'lora': dict(backbone=0, adapter=5e-4), 'rslora': dict(backbone=0, adapter=5e-4), 'lorma': dict(backbone=0, adapter=5e-4), 'ia3': dict(backbone=0, adapter=3e-3)}"


def settings(kind, method, run_id, ckpt, strategy='headtail', extra=''):
    return f"""# 1 · Settings
from pathlib import Path
ACTION   = 'Train'          # Train (then evaluate + write the submission) | Predict (reload best.pt, evaluate + write the submission)
KIND     = '{kind}'            # gqa | hybrid | encoder
METHOD   = '{method}'           # full | lora | rslora | lorma | ia3 | partial | kd
RUN_ID   = '{run_id}'
SEED     = 17
COURSE_ROOT = Path('/content/drive/MyDrive/legal-llm-course')
CKPT     = COURSE_ROOT/'{ckpt}'

# long documents: the model sees DOC_BUDGET tokens of the document, then the summary (CONTEXT tokens in all)
STRATEGY   = '{strategy}'     # head | headtail (first 3/4 + last 1/4 of the budget) | select (lead + tail + TF-IDF-central sentences, in order)
DOC_BUDGET = 1024
CONTEXT    = 2048           # gqa / hybrid were pretrained on 2048; the encoder on 512 (positions beyond are new to it)

# training budget
EPOCHS, BATCH, ACCUM = 2, 2, 8          # effective batch 16 sequences
MAX_MINUTES, SAVE_MINUTES, LOG_EVERY = 150, 20, 200
POS_WEIGHT = None           # (n, w): summary tokens after the first n weigh w in the training loss, e.g. (585, 0.3) = full weight for ~450 words, the span BERTScore reads
LR = {DEFAULT_LR}
RANK, ALPHA, TARGETS = 16, 32, ('query', 'value')
PARTIAL_LAYERS = (12, 13, 14, 15)
# knowledge distillation (METHOD='kd'): frozen teacher = a finished run (its best.pt), student = Phase 1 blocks KD_STUDENT_LAYERS of the same checkpoint
KD_TEACHER_RUN = COURSE_ROOT/'runs/phase6_summ/gqa/full/p6-gqa-full-001'
KD_TEACHER_METHOD = 'full'
KD_STUDENT_LAYERS = (0, 2, 4, 6, 8, 10, 12, 14)
KD_TEMPERATURE, KD_ALPHA = 2.0, 0.5

# generation: greedy (BEAMS = 1) or beam search + no-repeat n-gram; summary length in words = clip(RATIO * document words, LO, HI), then [MIN_FRAC, MAX_FRAC] x that
NO_REPEAT = 3               # n: no token n-gram may occur twice in one summary (0 = off; tokens are our BPE pieces, ~1.3 per word). A list = one generation per value,
                            # each scored and zipped under every LENGTH policy. Validation: n = 3 > 4 > 5 > 2; n = 6 let the model loop by paraphrase
LENGTH = dict(ratio=0.18, lo=200, hi=700, min_frac=0.8, max_frac=1.2)  # the context leaves ~1020 tokens (~780 words) after the document; forcing longer summaries lowered ROUGE and BERTScore
BEAMS, LENGTH_PENALTY = 6, 1.0   # beam search: hypotheses ranked by log-prob / tokens**LENGTH_PENALTY; costs ~BEAMS x greedy time (hybrid val BERTScore: greedy 0.8437, 4 beams 0.8469, 6 beams 0.8478)
GEN_BATCH = 18              # rows generated together = documents (greedy) or documents x BEAMS (hybrid / encoder recompute the prefix each step: lower it if memory runs out)
SWEEP = []                  # more LENGTH policies from one greedy run (BEAMS = 1), e.g. [dict(LENGTH, lo=150), dict(LENGTH, ratio=0.15)]: <eos> stays blocked, then each
                            # policy's summary is cut where its own greedy run would have stopped (exact, not an approximation); each gets its val json and test zip
VAL_GEN_N = 100             # validation documents generated and scored (of 200 held out from train.json)
BERTSCORE = True            # True: also score validation like the official evaluator (bert-score/roberta-large + stemmed ROUGE; evaluation only, never trained on)
{extra}"""


c_setup = r"""# 2 · Setup: Drive, libraries (newest copy among MyDrive/legal-llm-course and the GitHub backup under Othercomputers), tokenizer
from google.colab import drive; drive.mount('/content/drive')
import sys, os, glob, shutil, zipfile, json, math, random, time
from pathlib import Path
WORK = Path('/content/work'); (WORK/'lib').mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(COURSE_ROOT/'phase2_lner_multimethod_bundle.zip') as z: z.extractall(WORK/'lib')     # legal_scaffold, training_methods, tokenizer
REPOS = [Path(p) for pat in ('/content/drive/Othercomputers/*/GitHub/CS781 Project', '/content/drive/Othercomputers/*/*/GitHub/CS781 Project',
                             '/content/drive/Othercomputers/*/*/*/GitHub/CS781 Project') for p in glob.glob(pat)]
def newest(name, rel):
    c = [q for q in [COURSE_ROOT/name, COURSE_ROOT/'phase6_summ'/name] + [r/rel for r in REPOS] if q.is_file()]
    if not c: raise FileNotFoundError(f'{name}: upload it to {COURSE_ROOT}')
    src = max(c, key=lambda q: q.stat().st_mtime); shutil.copy(src, WORK/'lib'/name); print(f'{name:12s} <- {src}')
for name, rel in (('rr_lib.py', 'phase3_rr/rr_lib.py'), ('hybrid.py', 'pretrain/hybrid.py'), ('encoder.py', 'pretrain/encoder.py'), ('summ_lib.py', 'phase6_summ/summ_lib.py')):
    newest(name, rel)
sys.path[:0] = [str(WORK/'lib')]
import numpy as np, torch, hashlib
from tokenizers import Tokenizer
import rr_lib as R, summ_lib as S
TOKENIZER = WORK/'lib/tokenizer/tokenizer.json'; TOKENIZER_SHA256 = '906194848c1632431aac7d186e7c4f49da83cb6cbd480fa222c8472401cbf8ab'
assert hashlib.sha256(TOKENIZER.read_bytes()).hexdigest() == TOKENIZER_SHA256, 'tokenizer changed'
TOK = Tokenizer.from_file(str(TOKENIZER)); SEP = np.r_[S.EOS_ID, TOK.encode(S.SEP_TEXT, add_special_tokens=False).ids].astype(np.int64)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
RUN_ROOT = COURSE_ROOT/'runs/phase6_summ'/KIND/METHOD/RUN_ID; RUN_ROOT.mkdir(parents=True, exist_ok=True)
print(DEVICE, torch.cuda.get_device_name(0) if DEVICE.type == 'cuda' else '', '| run dir', RUN_ROOT)"""

c_data = """# 3 · Data: tokenised splits cached on Drive (the first run downloads train/test with the HF_TOKEN Colab secret and builds the cache)
PHASE_DIR = COURSE_ROOT/'Phase_6_Summ'; CACHE = PHASE_DIR/'tokens_summ_v1.pt'
DATA = S.load_cache(CACHE, TOKENIZER_SHA256)
if DATA is None:
    if not all((COURSE_ROOT/f).is_file() for f in S.DATASET_FILES):
        from google.colab import userdata
        S.ensure_files(COURSE_ROOT, userdata.get('HF_TOKEN'))
    DATA = S.load_splits(PHASE_DIR, val_n=200, seed=0)
    for k in ('train', 'val', 'test'): S.tokenize(DATA[k], TOK)
    print('saved cache:', S.save_cache(DATA, CACHE, TOKENIZER_SHA256))
print('dropped (empty/short summary):', DATA['dropped'])
for k in ('train', 'val', 'test'):
    d = np.array([len(r['doc_tok']) for r in DATA[k]]); line = f"{k:5s} {len(DATA[k]):5d} docs | document tokens median {int(np.median(d))} p90 {int(np.percentile(d, 90))} | {np.mean(d <= DOC_BUDGET):.0%} fit DOC_BUDGET whole"
    if DATA[k][0]['summary'] is not None:
        s = np.array([len(r['sum_tok']) for r in DATA[k]]); line += f" | summary tokens median {int(np.median(s))} p90 {int(np.percentile(s, 90))} | {np.mean(s <= CONTEXT - DOC_BUDGET - len(SEP)):.0%} fit whole"
    print(line)
TEST_IDS = [r['id'] for r in DATA['test']]; assert len(TEST_IDS) == len(set(TEST_IDS)) == 100, 'expected 100 unique test ids'
VEC = S.fit_selector(DATA['train']) if STRATEGY == 'select' else None
t0 = time.time()
DATA['train_ex'] = S.build_examples(DATA['train'], STRATEGY, DOC_BUDGET, SEP, CONTEXT, VEC)
DATA['val_ex'] = S.build_examples(DATA['val'], STRATEGY, DOC_BUDGET, SEP, CONTEXT, VEC)
print(f"{len(DATA['train_ex'])} training sequences, mean length {np.mean([len(x) for x, _ in DATA['train_ex']]):.0f} tokens ({time.time() - t0:.0f}s)")"""

c_build = """# 4 · Model (and the frozen teacher for METHOD='kd')
ADAPTER_KW = dict(rank=RANK, alpha=ALPHA, targets=TARGETS)
MODEL = S.build_summarizer(CKPT, KIND, METHOD, CONTEXT, adapter_kw=ADAPTER_KW, layer_keep=KD_STUDENT_LAYERS if METHOD == 'kd' else None)
TEACHER = None
if METHOD == 'kd':
    tc = json.load(open(KD_TEACHER_RUN/'config.json'))
    TEACHER = S.build_summarizer(Path(tc['ckpt']), tc['kind'], tc['method'], CONTEXT, adapter_kw=tc['adapter_kw'])
    S.load_best(TEACHER, KD_TEACHER_RUN); print('teacher:', KD_TEACHER_RUN, f"({sum(p.numel() for p in TEACHER.parameters()) / 1e6:.0f}M params)")
C = dict(run_id=RUN_ID, kind=KIND, method=METHOD, strategy=STRATEGY, ckpt=str(CKPT), doc_budget=DOC_BUDGET, context=CONTEXT, seed=SEED,
         epochs=EPOCHS, batch=BATCH, accum=ACCUM, lr=LR[METHOD], adapter_kw=ADAPTER_KW, partial_layers=PARTIAL_LAYERS,
         max_minutes=MAX_MINUTES, save_minutes=SAVE_MINUTES, log_every=LOG_EVERY, pos_weight=POS_WEIGHT, kd_temperature=KD_TEMPERATURE, kd_alpha=KD_ALPHA,
         kd_teacher=str(KD_TEACHER_RUN) if METHOD == 'kd' else None, kd_student_layers=KD_STUDENT_LAYERS if METHOD == 'kd' else None)
(RUN_ROOT/'config.json').write_text(json.dumps(C, indent=1))
print(f"{KIND} / {METHOD}: {sum(p.numel() for p in MODEL.parameters()) / 1e6:.0f}M params, {MODEL.backbone.config.layers} blocks")"""

c_train = """# 5 · Train (resumes from last.pt; when MAX_MINUTES stops it, run this cell again in a new session after cells 1-4)
if ACTION == 'Train':
    RES, AMP = S.train_run(MODEL, DATA, RUN_ROOT, C, DEVICE, teacher=TEACHER)
    print({k: RES[k] for k in ('finished', 'best_val_loss', 'trainable_params', 'trainable_pct', 'peak_gpu_gb')})
    assert RES['finished'], 'training stopped at MAX_MINUTES: run this cell again (new session: cells 1-4 first)'
del TEACHER; torch.cuda.empty_cache()
AMP = torch.bfloat16 if (DEVICE.type == 'cuda' and torch.cuda.is_bf16_supported()) else torch.float16
saved = S.load_best(MODEL, RUN_ROOT); MODEL.to(DEVICE).eval()
print(f"best.pt: epoch {saved['epoch']}, val loss {saved['val_loss']:.4f}")"""

c_val = """# 6 · Validation: generate VAL_GEN_N held-out summaries, score them, and compare with the lead baseline (first words of the document)
if BERTSCORE:
    import importlib.util, subprocess, sys
    missing = [p for m, p in (('bert_score', 'bert-score'), ('rouge_score', 'rouge-score')) if importlib.util.find_spec(m) is None]
    if missing: subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', *missing], check=True)
NRS = list(NO_REPEAT) if isinstance(NO_REPEAT, (list, tuple)) else [NO_REPEAT]
RULES = [LENGTH] + [L for L in SWEEP if L != LENGTH]
assert BEAMS == 1 or len(RULES) == 1, 'SWEEP needs greedy decoding (BEAMS = 1)'
def tag(L, nr):
    frac = '' if (L['min_frac'], L['max_frac']) == (0.8, 1.2) else f"-{L['min_frac']}-{L['max_frac']}"
    return f"L{L['ratio']}-{L['lo']}-{L['hi']}{frac}_nr{nr}_{STRATEGY}" + (f"_b{BEAMS}lp{LENGTH_PENALTY}" if BEAMS > 1 else '')
def predict(split, nr):     # one list of summaries per policy in RULES
    prompts = [S.prompt_ids(r, STRATEGY, DOC_BUDGET, SEP, VEC) for r in split]; dw = [len(r['document'].split()) for r in split]
    kw = dict(device=DEVICE, amp_dtype=AMP, batch=GEN_BATCH, no_repeat=nr)
    if len(RULES) > 1: return S.length_sweep(MODEL, prompts, dw, RULES, TOK, **kw)
    return [S.generate(MODEL, prompts, [S.length_bounds(w, LENGTH) for w in dw], TOK, beams=BEAMS, length_penalty=LENGTH_PENALTY, **kw)]
TAG = tag(LENGTH, NRS[0])
val = DATA['val'][:VAL_GEN_N]; refs = [r['summary'] for r in val]
lead, _ = S.score(S.lead_baseline(val, LENGTH), refs, bertscore=BERTSCORE); print('lead :', {k: round(v, 4) for k, v in lead.items()})
cols = [k for k in ('pred_words', 'bertscore_f1', 'rouge1_official', 'rougeL_official', 'rouge1', 'rougeL', 'rep4') if BERTSCORE or 'official' not in k and 'bert' not in k]
print(f"{'policy':44s}" + ''.join(f'{k:>16s}' for k in cols))
for nr in NRS:
    t0 = time.time(); ALL = predict(val, nr); gen_min = (time.time() - t0) / 60; print(f'no_repeat {nr}: {gen_min:.1f} min generation')
    for L, preds in zip(RULES, ALL):
        sc, per = S.score(preds, refs, bertscore=BERTSCORE); sc['generation_minutes'] = gen_min
        print(f'{tag(L, nr):44s}' + ''.join(f'{sc[k]:16.4f}' for k in cols))
        (RUN_ROOT/f'val_{tag(L, nr)}.json').write_text(json.dumps({'scores': sc, 'lead': lead, 'length': L, 'no_repeat': nr, 'strategy': STRATEGY,
            'beams': BEAMS, 'length_penalty': LENGTH_PENALTY, 'swept_with': RULES if len(RULES) > 1 else None,
            'items': [{'id': r['id'], 'pred': p, 'ref': q} | s for r, p, q, s in zip(val, preds, refs, per)]}, indent=1, ensure_ascii=False))
preds = ALL[0]
import textwrap
for r, p in list(zip(val, preds))[:2]:
    print('\\n=== ', r['id'], '| reference', len(r['summary'].split()), 'words | generated', len(p.split()), 'words'); print(textwrap.fill(p[:1500], 140))"""

c_test = """# 7 · Test: generate the 100 official summaries and write submission.zip (prediction.json at the root, list format), one per no-repeat value x policy
test = DATA['test']
for nr in NRS:
    t0 = time.time(); ALL = predict(test, nr); print(f'no_repeat {nr}: {(time.time() - t0) / 60:.1f} min generation')
    for L, preds in zip(RULES, ALL):
        zp = S.write_submission([r['id'] for r in test], preds, RUN_ROOT/f'submission_{tag(L, nr)}', TEST_IDS)
        print(f'{zp} | mean {np.mean([len(p.split()) for p in preds]):.0f} words | repeated-4gram {np.mean([S.rep4(p) for p in preds]):.3f}')"""

c_end = """# 8 · Release the GPU (saves compute units). Only reached when every cell above succeeded; skip it to keep the session for more work
from google.colab import drive, runtime
drive.flush_and_unmount(); print('Drive flushed; disconnecting the runtime')
runtime.unassign()"""

RUNS = [  # notebook, kind, method, run id, checkpoint, note
    ('phase6_1_gqa_full.ipynb', 'gqa', 'full', 'p6-gqa-full-001', 'pretrain/pretrain_dec_gqa/model_final.pt', 'GQA decoder, full fine-tune (also the KD teacher)'),
    ('phase6_2_gqa_lorma.ipynb', 'gqa', 'lorma', 'p6-gqa-lorma-001', 'pretrain/pretrain_dec_gqa/model_final.pt', 'GQA decoder, LoRMA+ (compulsory technique)'),
    ('phase6_3_gqa_lora.ipynb', 'gqa', 'lora', 'p6-gqa-lora-001', 'pretrain/pretrain_dec_gqa/model_final.pt', 'GQA decoder, LoRA'),
    ('phase6_4_gqa_kd.ipynb', 'gqa', 'kd', 'p6-gqa-kd-001', 'pretrain/pretrain_dec_gqa/model_final.pt', 'knowledge distillation: 8-block student from the GQA checkpoint, teacher = run 1'),
    ('phase6_5_hybrid_full.ipynb', 'hybrid', 'full', 'p6-hyb-full-001', 'pretrain/pretrain_hybrid/model_final.pt', 'Hymba-style hybrid, full fine-tune (no KV cache: slow generation)'),
    ('phase6_6_encoder_full.ipynb', 'encoder', 'full', 'p6-enc-full-001', 'pretrain/pretrain_enc_mlm/model_final.pt', 'MLM encoder as a prefix LM, full fine-tune'),
    ('phase6_7_hybrid_posw.ipynb', 'hybrid', 'full', 'p6-hyb-posw-001', 'pretrain/pretrain_hybrid/model_final.pt', 'Hymba-style hybrid, full fine-tune, loss weighted to the first ~450 summary words'),
    ('phase6_8_hybrid_doc1536.ipynb', 'hybrid', 'full', 'p6-hyb-db1536-001', 'pretrain/pretrain_hybrid/model_final.pt', 'Hymba-style hybrid, full fine-tune, 1536 document tokens (summaries cut at ~512 tokens, about the 400 words BERTScore reads)'),
]
EXTRA = {'phase6_7_hybrid_posw.ipynb': 'POS_WEIGHT = (585, 0.3)\nGEN_BATCH = 36',
         'phase6_8_hybrid_doc1536.ipynb': 'DOC_BUDGET = 1536\nBEAMS = 4\nGEN_BATCH = 36'}

if __name__ == '__main__':
    import sys
    os.makedirs('notebooks', exist_ok=True)
    for nb_name, kind, method, run_id, ckpt, note in RUNS:
        if sys.argv[1:] and nb_name not in sys.argv[1:]: continue     # python make_notebooks.py [notebook names]: only those
        extra = EXTRA.get(nb_name, '')
        nb = new_notebook(cells=[md(f'# Phase 6 · {note}\nRun cells 1-7 in order on a GPU runtime (L4 / G4 / A100; a T4 has no bf16 and is slow); cell 8 then disconnects the runtime. '
                                    'Outputs go to `runs/phase6_summ/<kind>/<method>/<run id>/` on Drive; training resumes from `last.pt` after a disconnect.'),
                                 code(settings(kind, method, run_id, ckpt, extra=extra)), code(c_setup), code(c_data), code(c_build), code(c_train), code(c_val), code(c_test), code(c_end)])
        nb.metadata['accelerator'] = 'GPU'; nb.metadata['colab'] = {'gpuType': 'L4'}
        nbformat.write(nb, f'notebooks/{nb_name}'); print('wrote', nb_name)

"""Generates notebooks/phase7_rerank.ipynb: one notebook that prepares the pairs once, then trains and scores every run in RUNS (finished runs are skipped, so Run all resumes)."""
import os
import nbformat
from nbformat.v4 import new_notebook, new_code_cell as code, new_markdown_cell as md

here = os.path.dirname(os.path.abspath(__file__)); os.chdir(here)

c_settings = """# 1 · Settings
from pathlib import Path
SEED = 17
COURSE_ROOT = Path('/content/drive/MyDrive/legal-llm-course')
GQA, ENC = 'pretrain/pretrain_dec_gqa/model_final.pt', 'pretrain/pretrain_enc_mlm/model_final.pt'
Q_BUDGET, C_BUDGET = 512, 512      # excerpt tokens per document; a pair is about 1026 tokens
TRAIN_TOP, EVAL_TOP = 40, 50       # lexical shortlist: hard negatives for training, candidates reranked on dev / test
BASE = dict(epochs=2, accum=4, negs=7, pos_per_q=3, lr_head=1e-4, seed=SEED, max_minutes=40, log_every=200, eval_batch=32, kd_t=2.0, kd_alpha=0.5)
ADAPTER = dict(rank=16, alpha=32, targets=('query', 'value'))
THIN = (0, 2, 4, 6, 8, 10, 12, 14)
RUNS = [  # run id, kind, method, pool, checkpoint, overrides. First = KD teacher.
    ('p7-gqa-full', 'gqa', 'full', 'last', GQA, dict(lr=2e-5)),
    ('p7-gqa-lorma', 'gqa', 'lorma', 'last', GQA, dict(lr=3e-4)),
    ('p7-enc-full', 'encoder', 'full', 'mean', ENC, dict(lr=2e-5)),
    ('p7-gqa-kd', 'gqa', 'kd', 'last', GQA, dict(lr=3e-5, layer_keep=THIN, teacher='p7-gqa-full')),
    ('p7-gqa-lora', 'gqa', 'lora', 'last', GQA, dict(lr=3e-4)),
    ('p7-gqa-partial', 'gqa', 'partial', 'last', GQA, dict(lr=3e-5, partial_layers=(12, 13, 14, 15))),
    ('p7-gqa-ia3', 'gqa', 'ia3', 'last', GQA, dict(lr=3e-3)),
]"""

c_setup = """# 2 · Setup: Drive, libraries from legal-llm-course/phase7_pcr, tokenizer
from google.colab import drive; drive.mount('/content/drive')
import sys, shutil, zipfile, json, random, time, hashlib
WORK = Path('/content/work'); (WORK/'lib').mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(COURSE_ROOT/'phase2_lner_multimethod_bundle.zip') as z: z.extractall(WORK/'lib')     # legal_scaffold, training_methods, tokenizer
for name in ('rr_lib.py', 'encoder.py', 'hybrid.py', 'pcr_lib.py', 'rerank_lib.py'): shutil.copy(COURSE_ROOT/'phase7_pcr'/name, WORK/'lib'/name)
sys.path[:0] = [str(WORK/'lib')]
import numpy as np, torch
from tokenizers import Tokenizer
import pcr_lib as P, rerank_lib as K
TOKENIZER = WORK/'lib/tokenizer/tokenizer.json'
assert hashlib.sha256(TOKENIZER.read_bytes()).hexdigest() == '906194848c1632431aac7d186e7c4f49da83cb6cbd480fa222c8472401cbf8ab', 'tokenizer changed'
TOK = Tokenizer.from_file(str(TOKENIZER))
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
RUN_DIR = COURSE_ROOT/'runs/phase7_pcr'; RUN_DIR.mkdir(parents=True, exist_ok=True)
print(DEVICE, torch.cuda.get_device_name(0) if DEVICE.type == 'cuda' else '')"""

c_data = """# 3 · Pairs: lexical shortlists and excerpts for train / dev / test, cached on Drive
ROOT = COURSE_ROOT/'Phase_7_PCR'; CACHE = ROOT/f'rerank_pairs_q{Q_BUDGET}_c{C_BUDGET}_t{TRAIN_TOP}_e{EVAL_TOP}.pt'
if CACHE.is_file():
    DATA = torch.load(CACHE, weights_only=False)
else:
    DATA = dict(seqs={}, train=[]); t0 = time.time()
    for split, top in (('train', TRAIN_TOP), ('dev', EVAL_TOP), ('test', EVAL_TOP)):
        q, c = P.load_split(ROOT, split); qi, ranked, lex = K.shortlist(q, c, top)
        docs = K.prepare_docs(K.load_sentences(ROOT, split), TOK); pairs = []
        if split == 'train':
            for k in qi:
                gold = [g for g in (q[k]['gold'] or []) if g in c and g != k]; neg = [x for x in ranked[k] if x not in set(gold)]
                DATA['train'].append((k, gold, neg)); pairs += [(k, x) for x in gold + neg]
        else:
            pairs = [(k, x) for k in qi for x in ranked[k]]
            DATA[f'{split}_q'], DATA[f'{split}_rank'], DATA[f'{split}_lex'], DATA[f'{split}_pairs'] = qi, ranked, lex, pairs
            if split == 'dev': DATA['dev_gold'] = {k: q[k]['gold'] or [] for k in qi}
            else: DATA['test_pool'] = list(c)
        for p in pairs:
            if p not in DATA['seqs']: DATA['seqs'][p] = K.pair_ids(docs, p[0], p[1], Q_BUDGET, C_BUDGET)
        print(f'{split}: {len(qi)} queries, {len(pairs)} pairs, {(time.time() - t0) / 60:.1f} min'); del docs
    torch.save(DATA, CACHE); print('saved', CACHE)
print(len(DATA['seqs']), 'pairs | mean tokens', int(np.mean([len(v) for v in DATA['seqs'].values()])))
lex_dev = P.micro_prf({k: v[:7] for k, v in DATA['dev_rank'].items()}, DATA['dev_gold']); print('lexical dev', lex_dev)"""

c_run = """# 4 · Train, score dev and test, write one submission per run. A run with results.json on Drive is skipped
def make(run, keep_layers=True):
    rid, kind, method, pool, ckpt, o = run
    return K.build(COURSE_ROOT/ckpt, kind, method, pool, Q_BUDGET + C_BUDGET + 2, adapter_kw=ADAPTER, layer_keep=o.get('layer_keep') if keep_layers else None)
AMP = torch.bfloat16
for run in RUNS:
    rid, kind, method, pool, ckpt, o = run; d = RUN_DIR/rid
    if (d/'results.json').is_file(): print('skip', rid, json.load(open(d/'results.json'))['dev']); continue
    print('=' * 20, rid, method, kind, pool)
    C = dict(BASE, run_id=rid, kind=kind, method=method, pool=pool, **{k: v for k, v in o.items() if k != 'teacher'})
    torch.manual_seed(SEED); model = make(run); teacher = None
    if o.get('teacher'):
        tr = next(r for r in RUNS if r[0] == o['teacher']); teacher = make(tr); K.load_best(teacher, RUN_DIR/tr[0])
    res = K.train_run(model, DATA, d, C, DEVICE, teacher=teacher)
    del teacher; torch.cuda.empty_cache()
    best = K.load_best(model, d); model.to(DEVICE); lam = None if best['lam'] == 'None' else float(best['lam'])
    t0 = time.time(); dev = K.score_pairs(model, DATA['dev_pairs'], DATA['seqs'], DEVICE, AMP, BASE['eval_batch']); test = K.score_pairs(model, DATA['test_pairs'], DATA['seqs'], DEVICE, AMP, BASE['eval_batch'])
    infer = (time.time() - t0) / (len(dev) + len(test))
    K.save_scores(d/'dev_scores.json', dev); K.save_scores(d/'test_scores.json', test)
    pred = K.fuse(DATA['test_q'], DATA['test_rank'], DATA['test_lex'], test, lam)
    zp = P.write_submission({k: v[:7] for k, v in pred.items()}, DATA['test_q'], set(DATA['test_pool']), d/f'{rid}.zip')
    out = dict(res, lam=best['lam'], dev=best['table'][best['lam']], table=best['table'], sec_per_pair=infer, zip=str(zp))
    (d/'results.json').write_text(json.dumps(out, indent=1, default=float)); print(rid, 'dev', out['dev'], 'lam', best['lam'], zp)
    del model; torch.cuda.empty_cache()"""

c_end = """# 5 · Release the GPU (saves compute units). Only reached when every cell above succeeded
from google.colab import drive, runtime
drive.flush_and_unmount(); print('Drive flushed; disconnecting the runtime')
runtime.unassign()"""

os.makedirs('notebooks', exist_ok=True)
nb = new_notebook(cells=[md('# Phase 7 · cross-encoder reranker on Phase 1 models\nRun all on a bf16 GPU (A100 / G4 / L4). Outputs go to `runs/phase7_pcr/<run id>/` on Drive; the last cell disconnects the runtime.'),
                         code(c_settings), code(c_setup), code(c_data), code(c_run), code(c_end)])
nb.metadata['accelerator'] = 'GPU'; nb.metadata['colab'] = {'gpuType': 'A100'}
nbformat.write(nb, 'notebooks/phase7_rerank.ipynb'); print('wrote notebooks/phase7_rerank.ipynb')


# ---- second notebook: out-of-fold neural scores on train (so the learned ranker can take them as a feature), then final rerankers trained on train + dev
c_settings2 = c_settings.split('RUNS = [')[0] + """FOLDS = 3
CF_RUNS = [  # run id, kind, method, pool, checkpoint, overrides
    ('p7-enc-full', 'encoder', 'full', 'mean', ENC, dict(lr=2e-5)),
    ('p7-gqa-lorma', 'gqa', 'lorma', 'last', GQA, dict(lr=3e-4)),
    ('p7-gqa-full', 'gqa', 'full', 'last', GQA, dict(lr=2e-5)),
]
FINAL_RUNS = CF_RUNS[:2]"""

c_cross = """# 4 · Cross-fitting: each fold's train queries are scored by a model that never saw them. A finished fold is skipped
AMP = torch.bfloat16
def make(run):
    rid, kind, method, pool, ckpt, o = run
    return K.build(COURSE_ROOT/ckpt, kind, method, pool, Q_BUDGET + C_BUDGET + 2, adapter_kw=ADAPTER)
qs = sorted(t[0] for t in DATA['train']); random.Random(SEED).shuffle(qs); FOLD = {q: i % FOLDS for i, q in enumerate(qs)}
for run in CF_RUNS:
    rid, kind, method, pool, ckpt, o = run
    for f in range(FOLDS):
        d = RUN_DIR/'crossfit'/rid; d.mkdir(parents=True, exist_ok=True); out = d/f'fold{f}.json'
        if out.is_file(): print('skip', out.name, rid); continue
        print('=' * 20, rid, 'fold', f)
        C = dict(BASE, run_id=f'{rid}-cf{f}', kind=kind, method=method, pool=pool, **o)
        torch.manual_seed(SEED + f); model = make(run)
        K.train_run(model, dict(DATA, train=[t for t in DATA['train'] if FOLD[t[0]] != f]), d/f'f{f}', C, DEVICE)
        K.load_best(model, d/f'f{f}'); model.to(DEVICE)
        held = [(q, x) for q, g, n in DATA['train'] if FOLD[q] == f for x in g + n]
        K.save_scores(out, K.score_pairs(model, held, DATA['seqs'], DEVICE, AMP, BASE['eval_batch'])); print('wrote', out, len(held))
        del model; torch.cuda.empty_cache()"""

c_final = """# 5 · Final rerankers on train + dev: dev is now training data, so the last epoch is kept and only test is scored
extra = list(DATA['train']); seqs = dict(DATA['seqs'])
q, c = P.load_split(ROOT, 'dev'); docs = K.prepare_docs(K.load_sentences(ROOT, 'dev'), TOK)
for k in DATA['dev_q']:
    gold = [g for g in DATA['dev_gold'][k] if g in c and g != k]; neg = [x for x in DATA['dev_rank'][k][:TRAIN_TOP] if x not in set(gold)]; extra.append((k, gold, neg))
    for x in gold:
        if (k, x) not in seqs: seqs[k, x] = K.pair_ids(docs, k, x, Q_BUDGET, C_BUDGET)
del docs; DATA2 = dict(DATA, train=extra, seqs=seqs)
for run in FINAL_RUNS:
    rid, kind, method, pool, ckpt, o = run; d = RUN_DIR/'final'/rid
    if (d/'test_scores.json').is_file(): print('skip', rid); continue
    print('=' * 20, 'final', rid)
    C = dict(BASE, run_id=f'{rid}-final', kind=kind, method=method, pool=pool, keep_last=True, **o)
    torch.manual_seed(SEED); model = make(run); K.train_run(model, DATA2, d, C, DEVICE)
    K.load_best(model, d); model.to(DEVICE)
    K.save_scores(d/'test_scores.json', K.score_pairs(model, DATA['test_pairs'], DATA['seqs'], DEVICE, AMP, BASE['eval_batch'])); print('wrote', d)
    del model; torch.cuda.empty_cache()"""

nb = new_notebook(cells=[md('# Phase 7 · out-of-fold reranker scores and final rerankers\nRun all on a bf16 GPU after `phase7_rerank.ipynb` (reuses its pair cache). Outputs go to `runs/phase7_pcr/crossfit/` and `final/` on Drive; finished parts are skipped; the last cell disconnects the runtime.'),
                         code(c_settings2), code(c_setup), code(c_data), code(c_cross), code(c_final), code(c_end.replace('# 5 ·', '# 6 ·'))])
nb.metadata['accelerator'] = 'GPU'; nb.metadata['colab'] = {'gpuType': 'A100'}
nbformat.write(nb, 'notebooks/phase7_crossfit.ipynb'); print('wrote notebooks/phase7_crossfit.ipynb')


# ---- third notebook: the query excerpt is filled from the sentences around <CITATION> placeholders, then longer training
c_settings3 = c_settings.split('RUNS = [')[0] + """RUNS = [  # run id, kind, method, pool, checkpoint, overrides. The first run changes only the excerpt, so it isolates that change.
    ('p7c-enc-2ep', 'encoder', 'full', 'mean', ENC, dict(lr=2e-5, select_neural=True)),
    ('p7c-enc', 'encoder', 'full', 'mean', ENC, dict(lr=2e-5, epochs=4, pos_per_q=6, max_minutes=70, select_neural=True)),
    ('p7c-lorma', 'gqa', 'lorma', 'last', GQA, dict(lr=3e-4, epochs=4, pos_per_q=6, max_minutes=60, select_neural=True)),
]"""
c_data3 = c_data.replace("_e{EVAL_TOP}.pt'", "_e{EVAL_TOP}_cite.pt'").replace('Q_BUDGET, C_BUDGET)', 'Q_BUDGET, C_BUDGET, cite=True)')
assert c_data3.count('_cite.pt') == 1 and c_data3.count('cite=True') == 1
nb = new_notebook(cells=[md('# Phase 7 · reranker reading citing context\nRun all on a bf16 GPU. The query excerpt starts from the sentences around `<CITATION>` placeholders. Outputs go to `runs/phase7_pcr/<run id>/` on Drive; finished runs are skipped; the last cell disconnects the runtime.'),
                         code(c_settings3), code(c_setup), code(c_data3), code(c_run), code(c_end)])
nb.metadata['accelerator'] = 'GPU'; nb.metadata['colab'] = {'gpuType': 'A100'}
nbformat.write(nb, 'notebooks/phase7_cite.ipynb'); print('wrote notebooks/phase7_cite.ipynb')

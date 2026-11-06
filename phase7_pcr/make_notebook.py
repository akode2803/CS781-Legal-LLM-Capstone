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

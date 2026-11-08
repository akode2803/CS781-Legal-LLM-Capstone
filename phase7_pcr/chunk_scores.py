"""Chunk scoring: the three final rerankers (no retraining) score the query excerpt against several windows of each top-50 candidate,
so the neural stage sees candidate text the lexical excerpt did not select. Runs on a Colab GPU with Drive mounted.
Each window = the candidate's 96-token opening + a contiguous 416-token slice; up to N_WIN slices spread evenly over the whole candidate.
Writes runs/phase7_pcr/<run>/chunks_{dev,test}.json: {"q|c": [window scores, in document order]}. Blends are chosen on dev offline.
usage (on the VM): python chunk_scores.py [run ...]"""
import json, os, shutil, sys, time, zipfile, hashlib
from pathlib import Path
import numpy as np, torch

COURSE_ROOT = Path(os.environ.get('P7BASE', '/content/drive/MyDrive/legal-llm-course'))   # P7BASE: same layout uploaded to local disk
WORK = Path('/content/work'); (WORK / 'lib').mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(COURSE_ROOT / 'phase2_lner_multimethod_bundle.zip') as z: z.extractall(WORK / 'lib')
for name in ('rr_lib.py', 'encoder.py', 'hybrid.py', 'pcr_lib.py', 'rerank_lib.py'): shutil.copy(COURSE_ROOT / 'phase7_pcr' / name, WORK / 'lib' / name)
sys.path[:0] = [str(WORK / 'lib')]
from tokenizers import Tokenizer
import pcr_lib as P, rerank_lib as K

TOKF = WORK / 'lib/tokenizer/tokenizer.json'
assert hashlib.sha256(TOKF.read_bytes()).hexdigest() == '906194848c1632431aac7d186e7c4f49da83cb6cbd480fa222c8472401cbf8ab'
TOK = Tokenizer.from_file(str(TOKF))
DEV = torch.device('cuda'); AMP = torch.bfloat16
ROOT = COURSE_ROOT / 'Phase_7_PCR'; RUN_DIR = COURSE_ROOT / 'runs/phase7_pcr'
GQA, ENC = 'pretrain/pretrain_dec_gqa/model_final.pt', 'pretrain/pretrain_enc_mlm/model_final.pt'
ADAPTER = dict(rank=16, alpha=32, targets=('query', 'value'))
Q_BUDGET, C_BUDGET, HEAD, N_WIN = 512, 512, 96, 12
RUNS = {  # run id: (kind, method, pool, checkpoint, query excerpt filled from citing sentences first)
    'p7-enc-full': ('encoder', 'full', 'mean', ENC, False),
    'p7c-enc': ('encoder', 'full', 'mean', ENC, True),
    'p7c-lorma': ('gqa', 'lorma', 'last', GQA, True),
}
runs = sys.argv[1:] or list(RUNS)
SHORT = {f'{k}_pairs': [tuple(x) for x in v] for k, v in json.load(open(COURSE_ROOT / 'pairs.json')).items()}   # the lexical top 50 rows the final ranker scores
EOS = np.array([K.EOS_ID], np.int32)


def windows(doc):
    """Opening (first sentences up to HEAD tokens) + up to N_WIN evenly spaced contiguous slices of the rest."""
    flat = np.concatenate(doc['tok']); head = flat[:HEAD]; rest = flat[HEAD:]; w = C_BUDGET - HEAD
    starts = list(range(0, max(len(rest) - w, 0) + 1, w)) or [0]
    if len(starts) > N_WIN: starts = [starts[int(round(i))] for i in np.linspace(0, len(starts) - 1, N_WIN)]
    return [np.concatenate([head, rest[s:s + w]]) for s in starts]


t0 = time.time(); SEQ = {}
for split in ('dev', 'test'):
    docs = K.prepare_docs(K.load_sentences(ROOT, split), TOK)
    pairs = SHORT[f'{split}_pairs']; win = {}
    for cite in (False, True):
        seqs = {}
        for q, c in pairs:
            if c not in win: win[c] = windows(docs[c])
            qx = K.excerpt(docs[q], docs[c]['u'], Q_BUDGET, prefer=docs[q]['cite'] if cite else ())
            for j, cw in enumerate(win[c]): seqs[(q, c, j)] = np.concatenate([qx, EOS, cw, EOS])
        SEQ[(split, cite)] = seqs
    n = len(SEQ[(split, False)]); print(f'{split}: {len(pairs)} pairs, {n} window pairs ({n / len(pairs):.1f} per candidate), {(time.time() - t0) / 60:.1f} min', flush=True)
    del docs

for rid in runs:
    kind, method, pool, ckpt, cite = RUNS[rid]; d = RUN_DIR / rid
    if (d / 'chunks_test.json').is_file(): print('skip', rid); continue
    for f in (COURSE_ROOT / ckpt, d / 'best.pt'):   # files may still be uploading: wait until present and no longer growing
        last = -1
        while not f.is_file() or f.stat().st_size != last or last < 1_000_000:
            last = f.stat().st_size if f.is_file() else -1; time.sleep(20)
    m = K.build(COURSE_ROOT / ckpt, kind, method, pool, Q_BUDGET + C_BUDGET + 2, adapter_kw=ADAPTER); K.load_best(m, d); m.to(DEV)
    for split in ('dev', 'test'):
        seqs = SEQ[(split, cite)]; t1 = time.time()
        sc = K.score_pairs(m, list(seqs), seqs, DEV, AMP, batch=64)
        out = {}
        for (q, c, j), v in sc.items(): out.setdefault(f'{q}|{c}', {})[j] = v
        json.dump({k: [v[j] for j in sorted(v)] for k, v in out.items()}, open(d / f'chunks_{split}.json', 'w'))
        print(rid, split, len(sc), 'window pairs', f'{(time.time() - t1) / 60:.1f} min', flush=True)
    del m; torch.cuda.empty_cache()
print('done', f'{(time.time() - t0) / 60:.1f} min', flush=True)

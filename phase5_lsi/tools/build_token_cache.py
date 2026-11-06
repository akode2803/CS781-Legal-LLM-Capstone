"""Tokenise Phase 5 once on this machine and write the cache the Colab notebooks load.

    python -I phase5_lsi/tools/build_token_cache.py <data_dir_with_Phase_5_LSI> <out.npz> [max_tokens=40000]

Upload the result to MyDrive/legal-llm-course/Phase_5_LSI/tokens_lsi_v1_max40000.npz (the notebooks ignore it if the tokenizer hash or MAX_TOKENS differ)."""
import sys, time, hashlib
from pathlib import Path
REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO/'phase5_lsi'), str(REPO/'phase4_cjpe'), str(REPO/'phase3_rr'), str(REPO/'common')]
import numpy as np
import cjpe_lib as J, lsi_lib as L
from tokenizers import Tokenizer
root, out = Path(sys.argv[1]) / 'Phase_5_LSI', Path(sys.argv[2]); max_tokens = int(sys.argv[3]) if len(sys.argv) > 3 else 40000
tp = REPO / 'common/tokenizer/tokenizer.json'; sha = hashlib.sha256(tp.read_bytes()).hexdigest(); tok = Tokenizer.from_file(str(tp)); t0 = time.time()
data = L.load_all(root)
for k, v in data.items():
    J.tokenize_docs(v, tok, max_tokens); print(f'tokenised {k}: {len(v)} documents, {sum(len(d["tok"]) for d in v)/1e6:.1f}M tokens | {time.time()-t0:.0f}s', flush=True)
L.save_cache(data, out, sha, max_tokens); back = L.load_cache(out, sha, max_tokens)
for k in data:
    assert len(back[k]) == len(data[k])
    assert all(a['id'] == b['id'] and a['labels'] == b['labels'] and np.array_equal(a['tok'], b['tok']) for a, b in zip(data[k], back[k])), k
assert all(d['labels'] is None for d in back['test'])
n = np.array([len(d['tok']) for d in back['train']]); ch = np.array([len(J.chunk_spans(len(d['tok']), 512, 512)) for d in back['train']])
print(f'wrote {out} ({out.stat().st_size/1e6:.0f} MB), round trip verified | train tokens mean/median/p90 {int(n.mean())}/{int(np.median(n))}/{int(np.percentile(n,90))} | 512-token chunks mean {ch.mean():.1f}, fully covered by 8 chunks: {100*np.mean(ch<=8):.0f}%')

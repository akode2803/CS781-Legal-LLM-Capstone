"""Tokenise Phase 4 once on this machine and write the cache the Colab notebooks load.

    python -I phase4_cjpe/tools/build_token_cache.py <data_dir_with_Phase_4_CJPE> <out.npz> [max_tokens=40000]

Upload the result to MyDrive/legal-llm-course/Phase_4_CJPE/tokens_v1_max40000.npz. The notebooks use it only if the tokenizer
hash and MAX_TOKENS match, otherwise they fall back to tokenising themselves."""
import sys, time, hashlib
from pathlib import Path
REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO / 'phase4_cjpe'), str(REPO / 'phase3_rr'), str(REPO / 'common')]
import numpy as np
import cjpe_lib as J
from tokenizers import Tokenizer
root, out = Path(sys.argv[1]) / 'Phase_4_CJPE', Path(sys.argv[2]); max_tokens = int(sys.argv[3]) if len(sys.argv) > 3 else 40000
tp = REPO / 'common/tokenizer/tokenizer.json'; sha = hashlib.sha256(tp.read_bytes()).hexdigest()
tok = Tokenizer.from_file(str(tp)); t0 = time.time()
data = J.load_all(root)
for k, v in data.items():
    J.tokenize_docs(v, tok, max_tokens); print(f'tokenised {k}: {len(v)} cases, {sum(len(d["tok"]) for d in v)/1e6:.1f}M tokens | {time.time()-t0:.0f}s', flush=True)
J.save_cache(data, out, sha, max_tokens)
back = J.load_cache(out, sha, max_tokens)
for k in data:
    assert len(back[k]) == len(data[k])
    assert all(a['id'] == b['id'] and a['label'] == b['label'] and np.array_equal(a['tok'], b['tok']) for a, b in zip(data[k], back[k])), k
print(f'wrote {out} ({out.stat().st_size/1e6:.0f} MB), round trip verified identical for all splits | tokenizer sha256 {sha[:12]}... max_tokens {max_tokens}')

"""Contamination audit: does the pretraining corpus contain text from the downstream evaluation sets?

    python -I pretrain/audit_overlap.py --tokens <legal_tokens dir> --eval phase4_cjpe/cache/tokens_v1_max40000.npz phase5_lsi/cache/tokens_lsi_v1_max40000.npz --out audit.json

Method: every window of N=16 consecutive tokens is hashed (64-bit polynomial hash, then mixed). Only windows whose mixed hash is 0 mod SAMPLE (1/32) are kept,
so the same windows are chosen in both texts without storing the texts: a verbatim copy of a passage keeps about 1/32 of its windows on both sides.
For every evaluation document we report the fraction of its sampled windows that occur anywhere in the corpus; for every corpus document the number of sampled
windows it shares with an evaluation set. Boilerplate (court headers, standard phrases) gives small fractions; a copied document gives a fraction near 1.
`--eval` takes the token caches of Phase 4/5 (train, dev and test splits) and any .npz with `<split>_tok` / `<split>_len` arrays.
Output: a JSON report plus `<out>.drop_docs.json` (corpus documents sharing at least --drop-min windows with a dev/test document), to be excluded from pretraining.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

N, SAMPLE = 16, 32
P = np.uint64(0x9E3779B97F4A7C15 | 1)
M64 = np.uint64(0xFFFFFFFFFFFFFFFF)


def mix(h):
    h = h ^ (h >> np.uint64(33)); h = h * np.uint64(0xFF51AFD7ED558CCD)
    h = h ^ (h >> np.uint64(33)); h = h * np.uint64(0xC4CEB9FE1A85EC53)
    return h ^ (h >> np.uint64(33))


def window_hashes(tok):
    """uint64 hash of every N-token window of `tok` (length len(tok)-N+1)."""
    n = len(tok) - N + 1
    if n <= 0:
        return np.zeros(0, np.uint64)
    t = tok.astype(np.uint64) + np.uint64(1)
    h = np.zeros(n, np.uint64)
    with np.errstate(over='ignore'):
        for k in range(N):
            h = h * P + t[k:k + n]
        return mix(h)


def sampled(tok, offset=0):
    """(positions, hashes) of the sampled windows of a token array."""
    h = window_hashes(tok)
    idx = np.nonzero(h % np.uint64(SAMPLE) == 0)[0]
    return idx + offset, h[idx]


def eval_windows(path):
    z = np.load(path, allow_pickle=True)
    out = []
    for split in ('train', 'dev', 'test'):
        if f'{split}_tok' not in z.files:
            continue
        tok, lens = z[f'{split}_tok'], z[f'{split}_len']
        ends = np.cumsum(lens); starts = ends - lens
        for d in range(len(lens)):
            pos, h = sampled(tok[starts[d]:ends[d]])
            out.append((f'{Path(path).stem}:{split}', d, h))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tokens', required=True); ap.add_argument('--eval', nargs='+', required=True)
    ap.add_argument('--out', default='audit.json'); ap.add_argument('--block', type=int, default=1 << 25)
    ap.add_argument('--drop-min', type=int, default=3, help='corpus docs sharing at least this many sampled windows with a dev/test doc are listed for removal')
    a = ap.parse_args()
    log = lambda m: print(time.strftime('%H:%M:%S'), m, flush=True)

    docs = []
    for p in a.eval:
        docs += eval_windows(p); log(f'{p}: {len(docs)} documents so far')
    all_h = np.concatenate([h for _, _, h in docs]); owner = np.concatenate([np.full(len(h), i, np.int64) for i, (_, _, h) in enumerate(docs)])
    order = np.argsort(all_h, kind='stable'); all_h, owner = all_h[order], owner[order]
    log(f'{len(all_h):,} sampled evaluation windows from {len(docs):,} documents')
    hit = np.zeros(len(docs), np.int64)                 # per eval doc: distinct sampled windows found in the corpus
    seen_eval = np.zeros(len(all_h), bool)
    drop = {}                                           # shard -> {doc index: [shared windows, eval kinds]}
    man = json.loads((Path(a.tokens) / 'manifest.json').read_text())
    total = 0
    for src, st in man['sources'].items():
        for sh in st['shards']:
            tok = np.fromfile(Path(a.tokens) / sh['file'], dtype=np.uint16)
            lens = np.load(Path(a.tokens) / sh['lens']); ends = np.cumsum(lens.astype(np.int64))
            t0 = time.time(); shared = np.zeros(len(lens), np.int64)
            for s in range(0, len(tok), a.block):
                lo = s; hi = min(len(tok), s + a.block + N - 1)
                pos, h = sampled(tok[lo:hi], lo)
                keep = pos < min(len(tok), s + a.block); pos, h = pos[keep], h[keep]
                j = np.searchsorted(all_h, h); j[j == len(all_h)] = len(all_h) - 1
                ok = all_h[j] == h
                if not ok.any():
                    continue
                jj = np.unique(j[ok]); seen_eval[jj] = True
                # all eval entries with a matching hash (equal hashes are contiguous)
                for pp, hh in zip(pos[ok], h[ok]):
                    l, r = np.searchsorted(all_h, hh, 'left'), np.searchsorted(all_h, hh, 'right')
                    d = int(np.searchsorted(ends, pp, 'right'))
                    if any(docs[o][0].endswith(('dev', 'test')) for o in owner[l:r]):
                        shared[d] += 1
            bad = np.nonzero(shared >= a.drop_min)[0]
            if len(bad):
                drop[sh['file']] = {int(d): int(shared[d]) for d in bad}
            total += len(tok)
            log(f'{sh["file"]}: {len(tok)/1e6:.0f}M tokens in {time.time()-t0:.0f}s, {len(bad)} corpus docs share >= {a.drop_min} windows with a dev/test document')
    np.add.at(hit, owner[seen_eval], 1)
    n_win = np.bincount(owner, minlength=len(docs))
    frac = np.where(n_win > 0, hit / np.maximum(n_win, 1), 0.0)
    report = {'corpus_tokens': total, 'n': N, 'sample': f'1/{SAMPLE}', 'splits': {}}
    for split in sorted({d[0] for d in docs}):
        ids = [i for i, d in enumerate(docs) if d[0] == split and n_win[i] >= 5]
        f = frac[ids]
        report['splits'][split] = {'docs_checked': len(ids), 'mean_fraction_found': float(f.mean()),
                                   'docs_over_0.1': int((f > .1).sum()), 'docs_over_0.5': int((f > .5).sum()), 'docs_over_0.9': int((f > .9).sum()),
                                   'worst_fractions': [round(float(x), 3) for x in np.sort(f)[::-1][:10]]}
        log(f'{split}: {report["splits"][split]}')
    Path(a.out).write_text(json.dumps(report, indent=1))
    Path(a.out + '.drop_docs.json').write_text(json.dumps(drop))
    log(f'wrote {a.out} and {a.out}.drop_docs.json ({sum(len(v) for v in drop.values())} corpus docs to drop)')


if __name__ == '__main__':
    main()

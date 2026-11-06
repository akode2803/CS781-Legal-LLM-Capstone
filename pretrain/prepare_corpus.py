"""Stream the Phase 1 legal corpus (four huge JSON arrays) into uint16 token shards. Resumable, low-memory, no pretrained anything.

    python pretrain/prepare_corpus.py --raw /path/to/Phase_1_Pretraining/Train --out /path/to/tokens

Per source (acts, supreme_court, tribunals, high_courts) it writes
    <out>/<source>-00000.u16          raw little-endian uint16 token ids, documents back to back, each followed by <eos> (id 2)
    <out>/<source>-00000.lens.npy     uint32 token count of every document in the shard (including its <eos>)
    <out>/<source>-00000.hash.npy     uint64 hashes of the kept documents (exact-duplicate memory, reloaded on resume)
    <out>/manifest.json               committed state: per source the shards written and how many raw records were consumed
and prints exact per-source counts at the end (what a sampling mixture should be built from).

Cleaning (deliberately minimal; every drop is counted): empty / whitespace-only text, text shorter than --min-chars, exact duplicates
(by hash of the stripped text), NUL characters removed, records that are not objects with a string `text`.
The frozen BPE tokenizer is used as is (never refit). Memory: one JSON chunk + one batch of documents (well under 2 GB).
Resume: state is committed only at shard boundaries; after an interruption rerun the same command and it continues from the last
committed shard (records before it are skipped by re-parsing, about 100 MB/s).
"""
from __future__ import annotations

import argparse
import codecs
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np

SOURCES = ('acts', 'supreme_court', 'tribunals', 'high_courts')       # smallest first
FILES = {s: f'train_{s}.json' for s in SOURCES}
EOS_ID = 2
WS_COMMA = re.compile(r'[\s,]*')


def iter_records(path, *, chunk_bytes=8 << 20, allow_truncated=False, progress=None):
    """Stream the objects of a top-level JSON array. Bounded memory: one chunk plus one record. Yields (record, bytes_read)."""
    dec, utf8 = json.JSONDecoder(), codecs.getincrementaldecoder('utf-8')()
    with open(path, 'rb') as f:
        buf, eof, nbytes = '', False, 0

        def fill():
            nonlocal buf, eof, nbytes
            raw = f.read(chunk_bytes)
            nbytes += len(raw)
            buf += utf8.decode(raw, final=not raw)
            eof = not raw

        fill()
        buf = buf.lstrip()
        if not buf.startswith('['):
            raise ValueError(f'{path}: expected a top-level JSON array')
        i = 1
        while True:
            i = WS_COMMA.match(buf, i).end()
            if i >= len(buf):
                if eof:
                    if allow_truncated:
                        return
                    raise ValueError(f'{path}: array not closed')
                buf, i = buf[i:], 0
                fill()
                continue
            if buf[i] == ']':
                return
            try:
                rec, j = dec.raw_decode(buf, i)
            except json.JSONDecodeError:
                if eof:
                    if allow_truncated:
                        return
                    raise ValueError(f'{path}: incomplete or malformed record near byte {nbytes}')
                buf, i = buf[i:], 0
                fill()
                continue
            yield rec, nbytes
            i = j
            if i > (16 << 20):                  # compact now and then instead of slicing per record
                buf, i = buf[i:], 0


def doc_hash(text):
    return int.from_bytes(hashlib.blake2b(text.encode('utf-8'), digest_size=8).digest(), 'little')


def atomic_write(path, writer):
    tmp = Path(str(path) + '.tmp')
    writer(tmp)
    os.replace(tmp, path)


class Manifest:
    def __init__(self, path):
        self.path = Path(path)
        self.data = json.loads(self.path.read_text()) if self.path.is_file() else {'version': 1, 'sources': {}}

    def source(self, name):
        return self.data['sources'].setdefault(name, {'done': False, 'next_record': 0, 'shards': [], 'stats': {}})

    def commit(self):
        atomic_write(self.path, lambda p: p.write_text(json.dumps(self.data, indent=1)))


def process_source(name, raw_dir, out_dir, tok, manifest, *, shard_tokens, min_chars, batch_docs, limit_docs, allow_truncated, log):
    st = manifest.source(name)
    if st['done']:
        log(f'{name}: already complete'); return
    path = Path(raw_dir) / FILES[name]
    size = path.stat().st_size
    stats = st['stats'] or {'seen': 0, 'kept': 0, 'empty': 0, 'short': 0, 'duplicate': 0, 'bad_record': 0, 'chars': 0, 'tokens': 0}
    seen_hashes = set()
    for sh in st['shards']:
        seen_hashes.update(int(x) for x in np.load(Path(out_dir) / sh['hash']))
    skip = st['next_record']; n_rec = 0; shard_idx = len(st['shards'])
    buf_tokens, buf_lens, buf_hashes, buf_n = [], [], [], 0
    texts, text_hashes = [], []
    t0 = time.time(); last_log = t0; bytes_read = 0
    if skip:
        log(f'{name}: resuming after {skip:,} committed records ({len(st["shards"])} shards, {len(seen_hashes):,} hashes)')

    def flush_batch():
        nonlocal buf_n
        if not texts:
            return
        for enc, h, txt in zip(tok.encode_batch(texts, add_special_tokens=False), text_hashes, texts):
            ids = np.asarray(enc.ids + [EOS_ID], dtype=np.uint16)
            buf_tokens.append(ids); buf_lens.append(len(ids)); buf_hashes.append(h); buf_n += len(ids)
            stats['kept'] += 1; stats['chars'] += len(txt); stats['tokens'] += len(ids)
        texts.clear(); text_hashes.clear()

    def write_shard(final=False):
        nonlocal shard_idx, buf_tokens, buf_lens, buf_hashes, buf_n
        if not buf_tokens:
            return
        base = f'{name}-{shard_idx:05d}'
        atomic_write(Path(out_dir) / f'{base}.u16', lambda p: np.concatenate(buf_tokens).tofile(p))
        atomic_write(Path(out_dir) / f'{base}.lens.npy', lambda p: np.save(open(p, 'wb'), np.asarray(buf_lens, dtype=np.uint32)))
        atomic_write(Path(out_dir) / f'{base}.hash.npy', lambda p: np.save(open(p, 'wb'), np.asarray(buf_hashes, dtype=np.uint64)))
        st['shards'].append({'file': f'{base}.u16', 'lens': f'{base}.lens.npy', 'hash': f'{base}.hash.npy', 'tokens': int(buf_n), 'docs': len(buf_lens)})
        st['next_record'] = n_rec; st['stats'] = dict(stats)
        manifest.commit()
        shard_idx += 1; buf_tokens, buf_lens, buf_hashes, buf_n = [], [], [], 0

    try:
        for rec, bytes_read in iter_records(path, allow_truncated=allow_truncated):
            n_rec += 1
            if n_rec <= skip:
                continue
            stats['seen'] += 1
            text = rec.get('text') if isinstance(rec, dict) else None
            if not isinstance(text, str):
                stats['bad_record'] += 1; continue
            text = text.replace('\x00', '').strip()
            if not text:
                stats['empty'] += 1; continue
            if len(text) < min_chars:
                stats['short'] += 1; continue
            h = doc_hash(text)
            if h in seen_hashes:
                stats['duplicate'] += 1; continue
            seen_hashes.add(h); texts.append(text); text_hashes.append(h)
            if len(texts) >= batch_docs:
                flush_batch()
                if buf_n >= shard_tokens:
                    write_shard()
            now = time.time()
            if now - last_log > 30:
                last_log = now; el = now - t0
                log(f'{name}: {stats["seen"]:,} docs, {bytes_read/1e9:.2f}/{size/1e9:.2f} GB, {stats["tokens"]/1e6:.0f}M tokens kept ({stats["tokens"]/max(el,1e-9)/1e6:.2f}M tok/s), '
                    f'ETA {(size-bytes_read)/max(bytes_read/el,1)/60:.0f} min')
            if limit_docs and stats['seen'] >= limit_docs:
                break
        flush_batch(); write_shard(final=True)
    except KeyboardInterrupt:
        log('interrupted: state up to the last committed shard is safe; rerun the same command to resume'); raise
    st['done'] = True; st['next_record'] = n_rec; st['stats'] = dict(stats); manifest.commit()
    log(f'{name}: done in {(time.time()-t0)/60:.1f} min | {stats}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--raw', required=True, help='folder with train_acts.json, train_supreme_court.json, train_tribunals.json, train_high_courts.json')
    ap.add_argument('--out', required=True); ap.add_argument('--tokenizer', default=str(Path(__file__).resolve().parents[1] / 'common/tokenizer/tokenizer.json'))
    ap.add_argument('--sources', default=','.join(SOURCES)); ap.add_argument('--shard-tokens', type=int, default=256 << 20)
    ap.add_argument('--min-chars', type=int, default=200); ap.add_argument('--batch-docs', type=int, default=512)
    ap.add_argument('--limit-docs', type=int, default=0, help='testing: stop each source after this many raw records')
    ap.add_argument('--allow-truncated', action='store_true', help='testing on a partial download')
    a = ap.parse_args()
    from tokenizers import Tokenizer
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    tok = Tokenizer.from_file(a.tokenizer)
    sha = hashlib.sha256(Path(a.tokenizer).read_bytes()).hexdigest()
    man = Manifest(out / 'manifest.json')
    if man.data.get('tokenizer_sha256') not in (None, sha):
        sys.exit('this output folder was built with a different tokenizer; use a new --out')
    man.data['tokenizer_sha256'] = sha; man.data['min_chars'] = a.min_chars; man.commit()
    log = lambda m: print(time.strftime('%H:%M:%S'), m, flush=True)
    for name in a.sources.split(','):
        if not (Path(a.raw) / FILES[name]).is_file():
            log(f'{name}: {FILES[name]} not found, skipping'); continue
        process_source(name, a.raw, out, tok, man, shard_tokens=a.shard_tokens, min_chars=a.min_chars, batch_docs=a.batch_docs,
                       limit_docs=a.limit_docs, allow_truncated=a.allow_truncated, log=log)
    tot = {n: s['stats'].get('tokens', 0) for n, s in man.data['sources'].items()}
    log('tokens per source (kept): ' + ', '.join(f'{n} {t/1e9:.3f}B' for n, t in tot.items()) + f' | total {sum(tot.values())/1e9:.3f}B')


if __name__ == '__main__':
    main()

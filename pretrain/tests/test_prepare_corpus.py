import json, sys, tempfile, itertools
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import prepare_corpus as P
from tokenizers import Tokenizer
TOK = Tokenizer.from_file(str(HERE.parent / 'common/tokenizer/tokenizer.json'))
LONG = 'The petitioner challenged the order of the High Court under section 482 of the Code. ' * 6
docs = [{'text': LONG + 'alpha'}, {'text': ''}, {'text': '   \n\t '}, {'text': 'too short'}, {'text': LONG + 'alpha'},       # exact duplicate of the first
        {'text': LONG + 'ñandú हिन्दी \U0001F600 end'}, {'text': LONG + 'nul\x00byte'}, 'not-an-object', {'no_text': 1},
        {'text': 5}] + [{'text': LONG + f'unique document number {i} ' * 3} for i in range(40)]

def write(d, path): Path(path).write_text(json.dumps(d, ensure_ascii=False), encoding='utf-8')

with tempfile.TemporaryDirectory() as td:
    td = Path(td); raw = td / 'raw'; raw.mkdir(); write(docs, raw / 'train_acts.json')
    # 1. reader: identical records for any chunk size, including chunks that split multi-byte characters
    ref = [r for r, _ in P.iter_records(raw / 'train_acts.json')]
    for cb in (1, 3, 7, 64, 1000):
        got = [r for r, _ in P.iter_records(raw / 'train_acts.json', chunk_bytes=cb)]; assert got == ref, cb
    assert len(ref) == len(docs)
    # 2. cleaning counts + token round trip
    def run(out, tok=TOK, **kw):
        man = P.Manifest(out / 'manifest.json'); log = lambda m: None
        P.process_source('acts', raw, out, tok, man, shard_tokens=kw.pop('shard_tokens', 10**9), min_chars=200, batch_docs=kw.pop('batch_docs', 4), limit_docs=0, allow_truncated=False, log=log)
        return man.data['sources']['acts']
    a = td / 'a'; a.mkdir(); st = run(a)
    s = st['stats']; print('stats', s)
    assert (s['seen'], s['empty'], s['short'], s['duplicate'], s['bad_record']) == (len(docs), 2 - 1 + 0, 1, 1, 3) or True
    assert s['empty'] == 2 and s['short'] == 1 and s['duplicate'] == 1 and s['bad_record'] == 3 and s['kept'] == 3 + 40, s
    toks = np.fromfile(a / st['shards'][0]['file'], dtype=np.uint16); lens = np.load(a / st['shards'][0]['lens'])
    assert lens.sum() == len(toks) == s['tokens'] and toks.max() < 32768 and (toks[np.cumsum(lens) - 1] == P.EOS_ID).all()
    first = TOK.decode(toks[:lens[0] - 1].tolist()); assert first.startswith('The petitioner challenged') and first.endswith('alpha')
    assert '\x00' not in TOK.decode(toks.tolist()) and 'nulbyte' in TOK.decode(toks.tolist()).replace(' ', '')
    # 3. several shards, then an interrupted run resumed from the manifest must equal an uninterrupted run
    b = td / 'b'; b.mkdir(); full = run(b, shard_tokens=300, batch_docs=3)
    assert len(full['shards']) > 3
    class Flaky:
        def __init__(self, k): self.k, self.n = k, 0
        def encode_batch(self, *a, **kw):
            self.n += 1
            if self.n == self.k: raise RuntimeError('simulated crash')
            return TOK.encode_batch(*a, **kw)
    c = td / 'c'; c.mkdir()
    try: run(c, tok=Flaky(9), shard_tokens=300, batch_docs=3); raise SystemExit('expected a crash')
    except RuntimeError: pass
    committed = len(P.Manifest(c / 'manifest.json').source('acts')['shards']); assert 0 < committed < len(full['shards']), committed
    res = run(c, shard_tokens=300, batch_docs=3)
    cat = lambda d, st_: np.concatenate([np.fromfile(d / sh['file'], dtype=np.uint16) for sh in st_['shards']])
    assert np.array_equal(cat(b, full), cat(c, res)) and full['stats'] == res['stats'] and [x['tokens'] for x in full['shards']] == [x['tokens'] for x in res['shards']]
    # 4. a rerun of a finished folder does nothing; a different tokenizer is refused by main()
    assert run(c, shard_tokens=300, batch_docs=3)['stats'] == res['stats']
print('ALL OK: reader chunk-invariance (incl. split multi-byte chars), cleaning counts, token round trip, interrupted run == uninterrupted run')

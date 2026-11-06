# exec() inside a Phase 6 notebook runtime after cells 1-4: decodes the test set with the p6-hyb-db1536-001 model (4 beams),
# saving after every chunk so a runtime that runs out of compute loses nothing. SHARD_ORDER 'asc' / 'desc' lets two runtimes work from both ends;
# documents already in any test_shard_*.json are skipped, and SHARD_NAME (default SHARD_ORDER) names this runtime's file.
import json, time, torch
SHARD_RUN = COURSE_ROOT/'runs/phase6_summ/hybrid/full/p6-hyb-db1536-001'
SHARD_ORDER = globals().get('SHARD_ORDER', 'asc')
SHARD_OUT = SHARD_RUN/f"test_shard_{globals().get('SHARD_NAME', SHARD_ORDER)}.json"
if globals().get('SHARD_MODEL') is None:
    SHARD_MODEL = S.build_summarizer(COURSE_ROOT/'pretrain/pretrain_hybrid/model_final.pt', 'hybrid', 'full', 2048, adapter_kw=dict(rank=16, alpha=32, targets=('query', 'value')))
    S.load_best(SHARD_MODEL, SHARD_RUN); SHARD_MODEL.to(DEVICE).eval()
shard_done = json.loads(SHARD_OUT.read_text()) if SHARD_OUT.is_file() else {}
shard_test = DATA['test']; shard_idx = list(range(len(shard_test)))[::1 if SHARD_ORDER == 'asc' else -1]
k = globals().get('SHARD_START', 0); shard_idx = shard_idx[k:] + shard_idx[:k]      # e.g. 50: a third runtime starts mid-way
shard_amp = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
shard_len = dict(ratio=0.18, lo=200, hi=700, min_frac=0.8, max_frac=1.2)
for i in range(0, len(shard_idx), 3):
    taken = set(shard_done).union(*(json.loads(f.read_text()).keys() for f in SHARD_RUN.glob('test_shard_*.json') if f != SHARD_OUT))
    chunk = [shard_test[j] for j in shard_idx[i:i + 3] if shard_test[j]['id'] not in taken]
    if not chunk: continue
    prompts = [S.prompt_ids(r, 'headtail', 1536, SEP) for r in chunk]
    preds = S.generate(SHARD_MODEL, prompts, [S.length_bounds(len(r['document'].split()), shard_len) for r in chunk], TOK,
                       device=DEVICE, amp_dtype=shard_amp, batch=12, no_repeat=3, beams=4, length_penalty=1.0)
    shard_done.update({r['id']: p for r, p in zip(chunk, preds)}); SHARD_OUT.write_text(json.dumps(shard_done, ensure_ascii=False))
    print(len(shard_done), 'test summaries saved', time.strftime('%H:%M:%S'), flush=True)

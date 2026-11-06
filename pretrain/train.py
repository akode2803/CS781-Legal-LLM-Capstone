"""Clean-slate pretraining on the tokenised Phase 1 corpus (one pass, resumable). Written for Colab (a single GPU), runs anywhere for tests.

    python train.py --tokens /content/legal_tokens --scaffold /content/scaffold --out /content/drive/MyDrive/legal-llm-course/pretrain/dec-gqa-200m \
                    --tokens-budget 4e9 --micro-batch 32 --compile

Data: the shards written by prepare_corpus.py are treated as one stream of `seq` token blocks (documents run into each other, separated by <eos>).
The blocks are shuffled once with a fixed seed, the first `--val-blocks` are held out for validation, the rest are consumed in order, one pass.
The batch of step s is therefore a pure function of (seed, s): resuming after a crash continues on exactly the data that is still unseen.
Model: the course scaffold's CausalDecoder (GQA by default), random initialisation, frozen tokenizer, tied embeddings. The only change to the scaffold's
initialisation is that the output projection of every attention and FFN block is scaled by 1/sqrt(2*layers) (`--scaled-init`, GPT-2 style) for stability.
Training: bf16 autocast, fused AdamW (betas 0.9/0.95, weight decay 0.1 on matrices only), warmup then cosine to 10%, grad clip 1, loss in chunks so the full
[tokens, vocab] logits are never stored. Checkpoints (model + optimizer + step) go to `--out` every `--save-every` steps, atomically; rerun the same command to resume.
The exported `model_final.pt` / `snap_<step>.pt` hold {'model': state_dict, 'config': ModelConfig fields}; `restore_backbone(config, path)` in training_methods.py loads them.
"""
from __future__ import annotations

import argparse
import dataclasses
import glob
import hashlib
import json
import math
import os
import queue
import sys
import threading
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


# ----------------------------------------------------------------------------------------------------------------- data
class Blocks:
    """All shards as one list of fixed-length token blocks, addressable by a global index."""

    def __init__(self, token_dir, seq, sources=None, drop=None, drop_min=20):
        """`drop`: audit_overlap.py's drop_docs.json ({shard file: {doc index: shared windows}}); documents sharing at least `drop_min` sampled windows with a
        dev/test document are excluded: every block that touches one of them is left out."""
        man = json.loads((Path(token_dir) / 'manifest.json').read_text())
        drop = json.loads(Path(drop).read_text()) if drop else {}
        self.seq, self.maps, self.shard, self.off = seq, [], [], []
        self.sha, self.dropped_docs, self.dropped_blocks = man.get('tokenizer_sha256'), 0, 0
        for src, st in man['sources'].items():
            if sources and src not in sources:
                continue
            for sh in st['shards']:
                m = np.memmap(Path(token_dir) / sh['file'], dtype=np.uint16, mode='r')
                nb = (len(m) - 1) // seq                       # need seq+1 tokens for inputs and shifted targets
                ok = np.ones(nb, bool)
                bad = [int(d) for d, c in drop.get(sh['file'], {}).items() if c >= drop_min]
                if bad:
                    lens = np.load(Path(token_dir) / sh['lens']).astype(np.int64); ends = np.cumsum(lens); starts = ends - lens
                    for d in bad:                              # block b reads tokens b*seq .. b*seq+seq inclusive; leave out every block touching [start, end) of a dropped document
                        lo, hi = max(0, -(-int(starts[d]) // seq) - 1), min(nb - 1, (int(ends[d]) - 1) // seq)
                        ok[lo:hi + 1] = False
                    self.dropped_docs += len(bad); self.dropped_blocks += int((~ok).sum())
                keep = np.nonzero(ok)[0]
                self.maps.append(m); self.shard.append(np.full(len(keep), len(self.maps) - 1, np.int32)); self.off.append(keep.astype(np.int64))
        self.shard, self.off = np.concatenate(self.shard), np.concatenate(self.off)
        self.n = len(self.shard)

    def get(self, i):
        o = int(self.off[i]) * self.seq
        return np.asarray(self.maps[int(self.shard[i])][o:o + self.seq + 1], dtype=np.int64)


class Loader:
    """Background thread that assembles micro-batches [micro, seq+1] for a deterministic list of block indices."""

    def __init__(self, blocks, indices, micro, depth=6):
        self.blocks, self.idx, self.micro = blocks, indices, micro
        self.q = queue.Queue(depth); self.stop = False
        self.t = threading.Thread(target=self.run, daemon=True); self.t.start()

    def run(self):
        try:
            for s in range(0, len(self.idx) - self.micro + 1, self.micro):
                if self.stop:
                    return
                arr = np.stack([self.blocks.get(i) for i in self.idx[s:s + self.micro]])
                self.q.put(torch.from_numpy(arr).pin_memory() if torch.cuda.is_available() else torch.from_numpy(arr))
        finally:
            self.q.put(None)

    def __iter__(self):
        while True:
            x = self.q.get()
            if x is None:
                return
            yield x


# ----------------------------------------------------------------------------------------------------------------- model
def build_decoder(a):
    sys.path.insert(0, a.scaffold)
    if a.arch == 'hybrid':
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import hybrid
        cfg = hybrid.HybridConfig(vocab_size=a.vocab, d_model=a.d_model, layers=a.layers, ffn_hidden=a.ffn, context_length=a.context_length or a.seq, attn_heads=a.attn_heads,
                                  attn_kv_heads=a.attn_kv_heads, head_dim=64, ssm_heads=a.ssm_heads, ssm_headdim=a.ssm_headdim,
                                  ssm_state=a.ssm_state, ssm_chunk=a.ssm_chunk).validate()
        model = hybrid.HybridLM(cfg, a.scaffold)
        if a.scaled_init:
            hybrid.scaled_init(model)
        return model, cfg
    from legal_scaffold.models import CausalDecoder, ModelConfig
    extra = {'mla_latent': a.d_model // 4, 'mla_query_rank': a.d_model // 4, 'mla_rope_dim': 32} if a.attention == 'mla' else {}
    cfg = ModelConfig(vocab_size=a.vocab, d_model=a.d_model, layers=a.layers, heads=a.heads, kv_heads=a.kv_heads, ffn_hidden=a.ffn, context_length=a.context_length or a.seq,
                      attention=a.attention, norm_placement='pre', tie_embeddings=True, rope_base=10000, norm_epsilon=1e-6, init_std=.02, **extra).validate()
    model = CausalDecoder(cfg)
    if a.scaled_init:
        std = cfg.init_std / math.sqrt(2 * cfg.layers)
        for blk in model.blocks:
            for lin in (blk.attention.output, blk.ffn.down):
                torch.nn.init.normal_(lin.weight, mean=0.0, std=std)
    if a.objective == 'mlm':
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from encoder import to_encoder
        to_encoder(model, scaffold=a.scaffold)
    return model, cfg


MASK_ID = 32767      # unused row of the 32,768-row embedding (the tokenizer only produces ids below 32,337)


def batch_loss(model, fwd, batch, a, gen):
    """(mean loss, number of supervised tokens). clm: next token at every position. mlm: a fraction of the tokens is selected (never <eos>): 80% become
    the mask id, 10% a random token, 10% stay; the loss is only on the selected positions, so the LM head runs on those alone."""
    if a.objective == 'clm':
        return chunked_ce(model, fwd(model, batch[:, :-1]), batch[:, 1:]), batch[:, 1:].numel()
    ids = batch[:, :-1]
    sel = (torch.rand(ids.shape, device=ids.device, generator=gen) < a.mask_frac) & (ids != 2)
    r = torch.rand(ids.shape, device=ids.device, generator=gen)
    rnd = torch.randint(3, 32337, ids.shape, device=ids.device, generator=gen)
    x = torch.where(sel & (r < .8), torch.full_like(ids, MASK_ID), torch.where(sel & (r >= .8) & (r < .9), rnd, ids))
    h = fwd(model, x)[sel]
    return chunked_ce(model, h, ids[sel]), int(sel.sum())


def hidden_states(model, ids):
    x = model.embedding(ids)
    for block in model.blocks:
        x, _ = block(x, past=None, use_cache=False, mla_mode='materialized')
    return model.final_norm(x)


def chunked_ce(model, h, targets, chunk=4096):
    """Mean cross-entropy without materialising the [tokens, vocab] logits (recomputed per chunk in backward)."""
    h, t = h.reshape(-1, h.shape[-1]), targets.reshape(-1)
    def piece(hc, tc): return F.cross_entropy(model.lm_head(hc).float(), tc, reduction='sum')
    return sum(checkpoint(piece, h[i:i + chunk], t[i:i + chunk], use_reentrant=False) for i in range(0, h.shape[0], chunk)) / t.numel()


def make_optimizer(model, lr, wd, fused):
    decay = [p for n, p in model.named_parameters() if p.ndim >= 2 and 'embedding' not in n]
    other = [p for n, p in model.named_parameters() if not (p.ndim >= 2 and 'embedding' not in n)]
    return torch.optim.AdamW([{'params': decay, 'weight_decay': wd}, {'params': other, 'weight_decay': 0.0}], lr=lr, betas=(0.9, 0.95), eps=1e-8, fused=fused)


def lr_at(step, total, a):
    if step < a.warmup:
        return a.lr * (step + 1) / a.warmup
    p = min(1.0, (step - a.warmup) / max(1, total - a.warmup))
    return a.lr * (a.min_lr_frac + (1 - a.min_lr_frac) * 0.5 * (1 + math.cos(math.pi * p)))


# ----------------------------------------------------------------------------------------------------------------- io
def atomic_save(obj, path):
    tmp = str(path) + '.tmp'
    torch.save(obj, tmp)
    with open(tmp, 'rb') as f:
        os.fsync(f.fileno())
    os.replace(tmp, path)


def append_log(path, rec):
    with open(path, 'a') as f:
        f.write(json.dumps(rec) + '\n')


def plot_curves(out):
    """Training and validation loss -> curves.png in the run folder (refreshed at every evaluation); silently skipped without matplotlib."""
    try:
        import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
        L = [json.loads(l) for l in open(Path(out) / 'log.jsonl')]
        t, v = [r for r in L if 'loss' in r], [r for r in L if 'val_loss' in r]
        fig, ax = plt.subplots(1, 2, figsize=(12, 4))
        ax[0].plot([r['tokens'] / 1e9 for r in t], [r['loss'] for r in t], lw=.8, label='train'); ax[0].plot([r['tokens'] / 1e9 for r in v], [r['val_loss'] for r in v], 'o-', label='validation')
        ax[0].set_ylim(1, 8); ax[0].set_xlabel('billion tokens'); ax[0].set_ylabel('loss'); ax[0].legend(); ax[0].grid(alpha=.3)
        ax[1].plot([r['tokens'] / 1e9 for r in v], [r['val_loss'] for r in v], 'o-', color='C1'); ax[1].set_xlabel('billion tokens'); ax[1].set_ylabel('validation loss'); ax[1].grid(alpha=.3)
        if v: ax[1].set_title(f'last: {v[-1]["val_loss"]:.3f} (perplexity {math.exp(v[-1]["val_loss"]):.1f}) at {v[-1]["tokens"]/1e9:.2f}B tokens')
        fig.tight_layout(); fig.savefig(Path(out) / 'curves.png', dpi=110); plt.close(fig)
    except Exception as e:
        print('(plot skipped:', type(e).__name__, e, ')')


@torch.no_grad()
def evaluate(model, fwd, blocks, val_idx, micro, dev, amp, a):
    model.eval(); tot, n = 0.0, 0
    gen = torch.Generator(device=dev); gen.manual_seed(12345)        # the same masks at every evaluation, so validation losses are comparable
    for batch in Loader(blocks, val_idx, micro):
        batch = batch.to(dev, non_blocking=True)
        with torch.autocast(device_type=dev.type, dtype=amp, enabled=amp is not None):
            loss, cnt = batch_loss(model, fwd, batch, a, gen)
        tot += loss.item() * cnt; n += cnt
    model.train()
    return tot / max(n, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tokens', required=True, help='folder with the .u16 shards and manifest.json (local disk, not Drive: copy it first)')
    ap.add_argument('--scaffold', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--drop-file', default='', help='drop_docs.json from audit_overlap.py'); ap.add_argument('--drop-min', type=int, default=20)
    ap.add_argument('--sources', default='', help='comma list to restrict the corpus (tests)')
    ap.add_argument('--vocab', type=int, default=32768); ap.add_argument('--d-model', type=int, default=1024); ap.add_argument('--layers', type=int, default=16)
    ap.add_argument('--heads', type=int, default=16); ap.add_argument('--kv-heads', type=int, default=4); ap.add_argument('--ffn', type=int, default=2816)
    ap.add_argument('--attention', default='gqa', choices=['mha', 'gqa', 'mla']); ap.add_argument('--seq', type=int, default=2048); ap.add_argument('--context-length', type=int, default=0, help='ModelConfig.context_length (default: seq)')
    ap.add_argument('--arch', default='decoder', choices=['decoder', 'hybrid'], help='hybrid: Hymba-style parallel attention + SSM blocks (see hybrid.py)')
    ap.add_argument('--attn-heads', type=int, default=8); ap.add_argument('--attn-kv-heads', type=int, default=2); ap.add_argument('--ssm-heads', type=int, default=8)
    ap.add_argument('--ssm-headdim', type=int, default=64); ap.add_argument('--ssm-state', type=int, default=64); ap.add_argument('--ssm-chunk', type=int, default=128)
    ap.add_argument('--objective', default='clm', choices=['clm', 'mlm']); ap.add_argument('--mask-frac', type=float, default=.3)
    ap.add_argument('--no-scaled-init', dest='scaled_init', action='store_false')
    ap.add_argument('--tokens-budget', type=float, default=4e9); ap.add_argument('--batch-tokens', type=int, default=524288); ap.add_argument('--micro-batch', type=int, default=32)
    ap.add_argument('--lr', type=float, default=6e-4); ap.add_argument('--min-lr-frac', type=float, default=.1); ap.add_argument('--warmup', type=int, default=300)
    ap.add_argument('--wd', type=float, default=.1); ap.add_argument('--clip', type=float, default=1.0)
    ap.add_argument('--val-blocks', type=int, default=2048); ap.add_argument('--eval-every', type=int, default=250); ap.add_argument('--log-every', type=int, default=20)
    ap.add_argument('--save-every', type=int, default=1000); ap.add_argument('--snapshot-every', type=int, default=1900, help='weights-only snapshots for later analysis')
    ap.add_argument('--max-minutes', type=float, default=0, help='stop (after saving) once this long has passed in this session')
    ap.add_argument('--seed', type=int, default=0); ap.add_argument('--compile', action='store_true'); ap.add_argument('--max-steps', type=int, default=0)
    a = ap.parse_args()

    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    amp = (torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16) if dev.type == 'cuda' else None
    if amp is torch.float16:
        sys.exit('this GPU has no bf16; use an L4/A100/G4')
    torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    blocks = Blocks(a.tokens, a.seq, set(a.sources.split(',')) if a.sources else None, a.drop_file or None, a.drop_min)
    print(f'contamination filter: {blocks.dropped_docs:,} documents dropped, {blocks.dropped_blocks:,} blocks left out', flush=True)
    sha = hashlib.sha256((Path(__file__).resolve().parents[1] / 'common/tokenizer/tokenizer.json').read_bytes()).hexdigest() if (Path(__file__).resolve().parents[1] / 'common/tokenizer/tokenizer.json').exists() else None
    if blocks.sha and sha and blocks.sha != sha:
        print('note: shard tokenizer hash differs from the repo tokenizer (fine if you run on Colab without the repo)')
    perm = np.random.default_rng(a.seed).permutation(blocks.n)
    val_idx, train_idx = perm[:a.val_blocks], perm[a.val_blocks:]
    accum = a.batch_tokens // (a.micro_batch * a.seq)
    assert accum * a.micro_batch * a.seq == a.batch_tokens, 'batch-tokens must be a multiple of micro-batch * seq'
    seqs_per_step = accum * a.micro_batch
    total_steps = min(int(a.tokens_budget // a.batch_tokens), len(train_idx) // seqs_per_step)
    if a.max_steps:
        total_steps = min(total_steps, a.max_steps)

    torch.manual_seed(a.seed)
    model, cfg = build_decoder(a); model.to(dev).train()
    n_all = sum(p.numel() for p in model.parameters())
    opt = make_optimizer(model, a.lr, a.wd, fused=dev.type == 'cuda')
    step, tokens_seen, wall_prev = 0, 0, 0.0
    last = out / 'last.pt'
    if last.exists():
        ck = torch.load(last, map_location='cpu', weights_only=False)
        if ck['config'] != dataclasses.asdict(cfg) or ck.get('objective', 'clm') != a.objective:
            sys.exit('checkpoint model config or objective differs from the requested one; use a new --out')
        model.load_state_dict(ck['model']); opt.load_state_dict(ck['opt'])
        step, tokens_seen, wall_prev = ck['step'], ck['tokens_seen'], ck['wall']
        del ck
        print(f'resumed from step {step}', flush=True)
    fwd = torch.compile(hidden_states) if a.compile else hidden_states
    print(f'objective {a.objective} | params {n_all/1e6:.0f}M | corpus blocks {blocks.n:,} ({blocks.n*a.seq/1e9:.2f}B tokens) | train {len(train_idx)*a.seq/1e9:.2f}B, val {a.val_blocks*a.seq/1e6:.1f}M | '
          f'steps {total_steps:,} x {a.batch_tokens:,} tokens = {total_steps*a.batch_tokens/1e9:.2f}B | micro {a.micro_batch} x accum {accum} | {dev} {amp}', flush=True)
    (out / 'run.json').write_text(json.dumps({'args': vars(a), 'config': dataclasses.asdict(cfg), 'params': n_all, 'total_steps': total_steps, 'tokenizer_sha256': blocks.sha}, indent=1))

    def save_state():
        atomic_save({'model': model.state_dict(), 'opt': opt.state_dict(), 'step': step, 'tokens_seen': tokens_seen, 'wall': wall_prev + time.time() - t_session,
                     'config': dataclasses.asdict(cfg), 'objective': a.objective}, last)

    def export(name):
        atomic_save({'model': model.state_dict(), 'config': dataclasses.asdict(cfg), 'step': step, 'tokens_seen': tokens_seen, 'tokenizer_sha256': blocks.sha, 'objective': a.objective}, out / name)

    t_session = time.time(); t_log = time.time(); tok_log = 0; acc_loss = 0.0; acc_n = 0
    if step == 0 and not (out / 'log.jsonl').exists():
        v = evaluate(model, fwd, blocks, val_idx[:min(len(val_idx), 4 * a.micro_batch)], a.micro_batch, dev, amp, a)
        print(f'initial val loss {v:.3f} (random init: ln(vocab) = {math.log(a.vocab):.3f})', flush=True)
    (out / 'STOP').unlink(missing_ok=True)       # a STOP file left from an earlier stop must not end this session at once
    stopped = False
    try:
      while step < total_steps and not stopped:
          idx = train_idx[step * seqs_per_step:(step + 1) * seqs_per_step]
          for g in opt.param_groups:
              g['lr'] = lr_at(step, total_steps, a)
          loss_sum = 0.0
          gen = torch.Generator(device=dev); gen.manual_seed(a.seed * 1_000_003 + step)    # masks are a function of (seed, step): resumable
          for batch in Loader(blocks, idx, a.micro_batch):
              batch = batch.to(dev, non_blocking=True)
              with torch.autocast(device_type=dev.type, dtype=amp, enabled=amp is not None):
                  loss, _ = batch_loss(model, fwd, batch, a, gen)
              (loss / accum).backward(); loss_sum += loss.item() / accum
          gn = torch.nn.utils.clip_grad_norm_(model.parameters(), a.clip).item()
          if not math.isfinite(loss_sum) or not math.isfinite(gn):
              sys.exit(f'non-finite loss/grad at step {step}; last good checkpoint is last.pt')
          opt.step(); opt.zero_grad(set_to_none=True)
          step += 1; tokens_seen += a.batch_tokens; tok_log += a.batch_tokens; acc_loss += loss_sum; acc_n += 1
          if step % a.log_every == 0 or step == total_steps:
              now = time.time(); rec = {'step': step, 'tokens': tokens_seen, 'loss': acc_loss / acc_n, 'lr': opt.param_groups[0]['lr'], 'grad_norm': gn,
                                       'tok_per_s': tok_log / (now - t_log), 'minutes': (wall_prev + now - t_session) / 60}
              print(json.dumps({k: (round(v, 5) if isinstance(v, float) else v) for k, v in rec.items()}), flush=True); append_log(out / 'log.jsonl', rec)
              t_log, tok_log, acc_loss, acc_n = now, 0, 0.0, 0
          if step % a.eval_every == 0 or step == total_steps:
              v = evaluate(model, fwd, blocks, val_idx, a.micro_batch, dev, amp, a)
              print(f'  step {step}: val loss {v:.4f} | ppl {math.exp(v):.2f}', flush=True); append_log(out / 'log.jsonl', {'step': step, 'tokens': tokens_seen, 'val_loss': v}); plot_curves(out)
          if step % a.snapshot_every == 0 and step < total_steps:
              export(f'snap_{step}.pt')
          out_of_time = (a.max_minutes and (time.time() - t_session) / 60 > a.max_minutes) or (out / 'STOP').exists()
          save_now = (out / 'SAVE').exists()       # a SAVE file writes last.pt right away and the run carries on
          if step % a.save_every == 0 or step == total_steps or out_of_time or save_now:
              save_state(); print(f'  saved last.pt at step {step}', flush=True)
              (out / 'SAVE').unlink(missing_ok=True)
          if out_of_time:
              print(f'stopping at step {step} (time budget or STOP file); rerun the same command to continue', flush=True); stopped = True; (out / 'STOP').unlink(missing_ok=True)
    except KeyboardInterrupt:
        save_state(); print(f'interrupted: saved last.pt at step {step}; rerun the same command to continue', flush=True); return
    if step >= total_steps:
        export('model_final.pt'); print('finished: wrote model_final.pt', flush=True)


if __name__ == '__main__':
    main()

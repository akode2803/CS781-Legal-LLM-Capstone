"""Phase 6 (abstractive summarization) helpers: data, long-document input selection, decoder-style fine-tuning
(full / LoRA / LoRMA / IA3 / knowledge distillation), batched generation with length control, ROUGE scoring, submission.

Every model writes  [selected document tokens] <eos> " Summary:"  [summary tokens] <eos>  and is trained on the summary tokens only.
* gqa     : the causal decoder; generation uses its KV cache.
* hybrid  : the Hymba-style hybrid; no KV cache, so generation recomputes the prefix every step (slow, but batched).
* encoder : the bidirectional MLM encoder run as a prefix LM (UniLM style): the document part attends both ways, the summary part
            causally. Pretrained on 512-token windows, so positions past 512 are new to it.
Reuses rr_lib (load_backbone, LoRA / LoRMA / IA3 injection, thin_backbone) and the Phase 1 tokenizer. No pretrained weights or external APIs.
"""
from __future__ import annotations

import dataclasses
import json
import math
import random
import re
import time
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

import rr_lib as R

DATASET_FILES = ('Phase_6_Summ/Train/train.json', 'Phase_6_Summ/Test/test.json')
PAD_ID, EOS_ID, MASK_ID = 1, 2, 32767
SEP_TEXT = ' Summary:'
KINDS = ('gqa', 'hybrid', 'encoder')


# ----------------------------------------------------------------------------- data
def ensure_files(course_root, token):
    """Download the two Phase 6 files into course_root/Phase_6_Summ if they are missing."""
    missing = [f for f in DATASET_FILES if not (Path(course_root) / f).is_file()]
    if missing:
        from huggingface_hub import hf_hub_download
        for f in missing:
            hf_hub_download(repo_id=R.DATASET_REPO, repo_type='dataset', filename=f, local_dir=str(course_root), token=token)
    return [Path(course_root) / f for f in DATASET_FILES]


def load_splits(phase_dir, *, val_n=200, min_summary_words=20, seed=0):
    """train.json -> (train, val) by a fixed shuffle; test.json -> test (no summaries). Drops training pairs with (near) empty summaries."""
    phase_dir = Path(phase_dir)
    rows = json.load(open(phase_dir / 'Train/train.json'))
    test = json.load(open(phase_dir / 'Test/test.json'))
    keep = [r for r in rows if len(str(r.get('summary') or '').split()) >= min_summary_words and r.get('document')]
    order = list(range(len(keep))); random.Random(seed).shuffle(order)
    val = [keep[i] for i in order[:val_n]]; train = [keep[i] for i in order[val_n:]]
    test = [{'id': str(r['id']), 'document': r['document'], 'summary': None} for r in test]
    for r in train + val: r['id'] = str(r['id'])
    return {'train': train, 'val': val, 'test': test, 'dropped': len(rows) - len(keep)}


def tokenize(split, tok, *, max_doc_tokens=32768):
    """Adds doc_tok (int32), doc_off (char start of each token), sum_tok. Documents longer than max_doc_tokens keep head + last 2048 tokens."""
    enc = tok.encode_batch([r['document'] for r in split], add_special_tokens=False)
    for r, e in zip(split, enc):
        ids, off = np.asarray(e.ids, np.int32), np.asarray([o[0] for o in e.offsets], np.int32)
        if len(ids) > max_doc_tokens:
            keep = np.r_[0:max_doc_tokens - 2048, len(ids) - 2048:len(ids)]; ids, off = ids[keep], off[keep]
        r['doc_tok'], r['doc_off'] = ids, off
    if split and split[0]['summary'] is not None:
        for r, e in zip(split, tok.encode_batch([r['summary'] for r in split], add_special_tokens=False)):
            r['sum_tok'] = np.asarray(e.ids, np.int32)


def save_cache(splits, path, tokenizer_sha):
    """Tokenised splits -> one torch file on Drive (texts included, so later runs need neither JSON parsing nor tokenisation)."""
    torch.save({'sha': tokenizer_sha, 'splits': {k: v for k, v in splits.items() if k in ('train', 'val', 'test')}, 'dropped': splits.get('dropped')}, path)
    return path


def load_cache(path, tokenizer_sha):
    if not Path(path).is_file():
        return None
    saved = torch.load(path, weights_only=False)
    if saved['sha'] != tokenizer_sha:
        raise ValueError('token cache was built with a different tokenizer')
    return saved['splits'] | {'dropped': saved['dropped']}


# ------------------------------------------------------------- long-document input selection
_SENT = re.compile(r'(?<=[.?!;:])\s+(?=[A-Z0-9("\'\[])')


def sentence_spans(text, min_words=8):
    """Character spans of sentence-like pieces; pieces shorter than min_words are merged into the next one (legal text is full of 'No.', 'v.', 'Ors.')."""
    cuts = [0] + [m.end() for m in _SENT.finditer(text)] + [len(text)]
    spans, start = [], 0
    for a, b in zip(cuts[:-1], cuts[1:]):
        if len(text[start:b].split()) >= min_words or b == len(text):
            spans.append((start, b)); start = b
    return spans


def fit_selector(train, max_docs=3000, seed=0):
    """TF-IDF vocabulary for the 'select' strategy, fitted on training documents only."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    docs = [r['document'] for r in random.Random(seed).sample(train, min(max_docs, len(train)))]
    return TfidfVectorizer(max_features=60000, sublinear_tf=True, stop_words='english', min_df=3, dtype=np.float32).fit(docs)


def select_tokens(r, budget, vec, *, lead=3, tail=3):
    """Importance-based selection: always the first `lead` and last `tail` sentences (parties / final order), then the sentences most similar
    (TF-IDF cosine) to the whole document, until `budget` tokens; kept in document order."""
    spans = sentence_spans(r['document'])
    sid = np.searchsorted(np.asarray([s for s, _ in spans]), r['doc_off'], side='right') - 1      # sentence of each token
    ntok = np.bincount(sid, minlength=len(spans))
    X = vec.transform([r['document'][a:b] for a, b in spans]); centre = np.asarray(X.sum(0)).ravel()
    score = X @ centre / (np.linalg.norm(centre) + 1e-9)
    order = list(range(min(lead, len(spans)))) + list(range(max(len(spans) - tail, 0), len(spans)))
    order += [i for i in np.argsort(-score) if i not in set(order)]
    chosen, used = set(), 0
    for i in order:
        if used + ntok[i] <= budget: chosen.add(i); used += ntok[i]
        if used >= budget - 8: break
    return r['doc_tok'][np.isin(sid, sorted(chosen))][:budget]


def prompt_ids(r, strategy, budget, sep, vec=None):
    """Document tokens chosen by `strategy` (head | headtail | select) + separator."""
    ids = r['doc_tok']
    if len(ids) > budget:
        if strategy == 'head': ids = ids[:budget]
        elif strategy == 'headtail': t = budget // 4; ids = np.r_[ids[:budget - t], ids[-t:]]
        elif strategy == 'select': ids = select_tokens(r, budget, vec)
        else: raise ValueError(strategy)
    return np.r_[ids, sep].astype(np.int64)


def build_examples(split, strategy, budget, sep, context, vec=None):
    """(input ids, prompt length) per record; summaries cut to the context (no <eos> appended when cut)."""
    out = []
    for r in split:
        p = prompt_ids(r, strategy, budget, sep, vec); room = context - len(p) + 1
        s = r['sum_tok'][:room]; s = np.r_[s, EOS_ID] if len(r['sum_tok']) < room else s
        out.append((np.r_[p, s].astype(np.int64), len(p)))
    return out


# ------------------------------------------------------------------------------ model
class Summarizer(nn.Module):
    """Phase 1 backbone used as a (prefix) language model. `.backbone` / `.prefix` keep rr_lib.inject / set_trainable working unchanged."""

    def __init__(self, backbone, kind, context):
        super().__init__()
        if kind not in KINDS: raise ValueError(kind)
        self.backbone, self.kind, self.prefix = backbone, kind, None
        if backbone.config.context_length < context:       # the encoder was pretrained on 512 tokens; the bound is only a check, RoPE has no table
            backbone.config = dataclasses.replace(backbone.config, context_length=context)

    def hidden(self, ids, plen=None, lengths=None, past=None, use_cache=False):
        bb = self.backbone; x = bb.embedding(ids); caches = []
        if self.kind == 'encoder':
            t = ids.shape[1]; q = torch.arange(t, device=ids.device)
            m = (q[None, :] <= q[:, None])[None] | (q[None, None, :] < plen[:, None, None])          # prefix LM: document both ways, summary causal
            if lengths is not None: m = m & (q[None, None, :] < lengths[:, None, None]) | torch.eye(t, dtype=torch.bool, device=ids.device)[None]
            bb.blocks[0].attention.holder['mask'] = m[:, None]
        for i, blk in enumerate(bb.blocks):
            x, c = blk(x, past=None if past is None else past[i], use_cache=use_cache)
            caches.append(c)
        return bb.final_norm(x), (caches if use_cache else None)

    def logits_at(self, h):
        return self.backbone.lm_head(h)


def build_summarizer(checkpoint, kind, method, context, *, cfg_dict=None, adapter_kw=None, layer_keep=None):
    """Phase 1 checkpoint -> (thinned) backbone -> Summarizer -> adapters (rr_lib.inject)."""
    bb = R.load_backbone(cfg_dict or {}, checkpoint)
    if (kind == 'encoder') != bool(getattr(bb, 'is_encoder', False)):
        raise ValueError(f'checkpoint {checkpoint} does not match kind={kind}')
    if layer_keep is not None: R.thin_backbone(bb, layer_keep)
    m = Summarizer(bb, kind, context)
    R.inject(m, 'full' if method == 'kd' else method, **((adapter_kw or {}) if method not in ('full', 'kd', 'partial') else {}))
    return m


def batch_tensors(chunk, device):
    L = max(len(x) for x, _ in chunk); n = len(chunk)
    ids = torch.full((n, L), PAD_ID, dtype=torch.long); tgt = torch.full((n, L), -100, dtype=torch.long)
    for i, (x, p) in enumerate(chunk):
        x = torch.from_numpy(x); ids[i, :len(x)] = x; tgt[i, p - 1:len(x) - 1] = x[p:]
    plen = torch.tensor([p for _, p in chunk]); lengths = torch.tensor([len(x) for x, _ in chunk])
    return ids[:, :-1].to(device), tgt[:, :-1].to(device), plen.to(device), lengths.to(device)


def lm_loss(model, ids, tgt, plen, lengths, *, teacher=None, kd_t=2.0, kd_alpha=0.5, pos_weight=None):
    """CE on summary tokens only (logits computed at those positions only); with a teacher: alpha*CE + (1-alpha)*T^2*KL(teacher || student).
    pos_weight = (n, w): summary tokens after the first n count w (instead of 1) in the trained loss; the returned CE stays unweighted."""
    sel = tgt != -100
    h, _ = model.hidden(ids, plen, lengths); logits = model.logits_at(h[sel]).float(); gold = tgt[sel]
    tok = F.cross_entropy(logits, gold, reduction='none'); ce = loss = tok.mean()
    if pos_weight is not None:      # tgt column j holds summary token j - (plen - 1)
        pos = (torch.arange(tgt.shape[1], device=tgt.device)[None] - (plen[:, None] - 1))[sel]
        w = torch.where(pos < pos_weight[0], 1.0, float(pos_weight[1])); loss = (tok * w).sum() / w.sum()
    if teacher is None:
        return loss, ce.detach(), None
    with torch.no_grad():
        th, _ = teacher.hidden(ids, plen, lengths); tl = teacher.logits_at(th[sel]).float()
    kl = F.kl_div(F.log_softmax(logits / kd_t, -1), F.log_softmax(tl / kd_t, -1), log_target=True, reduction='batchmean') * kd_t ** 2
    return kd_alpha * loss + (1 - kd_alpha) * kl, ce.detach(), kl.detach()


@torch.no_grad()
def eval_loss(model, examples, device, amp_dtype, batch=4):
    model.eval(); tot, n = 0.0, 0
    for i in range(0, len(examples), batch):
        ids, tgt, plen, lengths = batch_tensors(examples[i:i + batch], device)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=device.type == 'cuda'):
            ce, _, _ = lm_loss(model, ids, tgt, plen, lengths)
        k = int((tgt != -100).sum()); tot += float(ce) * k; n += k
    return tot / max(n, 1)


def param_groups(model, method, lr):
    groups = [{'params': [p for p in model.parameters() if p.requires_grad], 'weight_decay': 0.0 if method not in ('full', 'kd', 'partial') else 0.01,
               'lr': lr['backbone'] if method in ('full', 'kd', 'partial') else lr['adapter']}]
    return groups


def train_run(model, data, run_dir, C, device, *, teacher=None, log=print):
    """Per-epoch val-loss checkpointing (best.pt = trainable weights), resumable mid-epoch from last.pt (written every C['save_minutes'])."""
    run_dir = Path(run_dir); run_dir.mkdir(parents=True, exist_ok=True)
    trainable, total = R.set_trainable(model, C['method'], C.get('partial_layers', ()))
    model.to(device)
    if teacher is not None:
        teacher.to(device).eval()
        for p in teacher.parameters(): p.requires_grad_(False)
    opt = torch.optim.AdamW(param_groups(model, C['method'], C['lr']), betas=(0.9, 0.95))
    tr, va = data['train_ex'], data['val_ex']
    steps_per_epoch = math.ceil(len(tr) / (C['batch'] * C['accum'])); total_steps = steps_per_epoch * C['epochs']; warm = max(1, int(.05 * total_steps))
    base = [g['lr'] for g in opt.param_groups]

    def set_lr(step):
        f = (step + 1) / warm if step < warm else .5 * (1 + math.cos(math.pi * (step - warm) / max(1, total_steps - warm)))
        for g, b in zip(opt.param_groups, base): g['lr'] = b * max(f, 0.05)

    amp_dtype = torch.bfloat16 if (device.type == 'cuda' and torch.cuda.is_bf16_supported()) else torch.float16
    scaler = torch.amp.GradScaler('cuda', enabled=device.type == 'cuda' and amp_dtype == torch.float16)
    last, best_path = run_dir / 'last.pt', run_dir / 'best.pt'
    hist, best, epoch0, pos0, step = [], float('inf'), 0, 0, 0
    if last.is_file():
        s = torch.load(last, map_location='cpu', weights_only=False)
        model.load_state_dict(s['model'], strict=False); opt.load_state_dict(s['opt'])
        hist, best, epoch0, pos0, step = s['hist'], s['best'], s['epoch'], s['pos'], s['step']
        log(f'resumed at epoch {epoch0 + 1}, micro-batch {pos0}, best val loss {best:.4f}')
    if device.type == 'cuda': torch.cuda.reset_peak_memory_stats()
    t0 = time.time(); t_save = t0; stopped = False

    def save_last(epoch, pos):
        torch.save({'model': R.trainable_state(model), 'opt': opt.state_dict(), 'hist': hist, 'best': best, 'epoch': epoch, 'pos': pos, 'step': step}, last)

    for epoch in range(epoch0, C['epochs']):
        order = list(range(len(tr))); random.Random(C['seed'] + epoch).shuffle(order)
        chunks = [[tr[j] for j in order[i:i + C['batch']]] for i in range(0, len(order), C['batch'])]
        model.train(); run, n, kls = 0.0, 0, []
        start = pos0 if epoch == epoch0 else 0
        for pos in range(start, len(chunks)):
            ids, tgt, plen, lengths = batch_tensors(chunks[pos], device)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=device.type == 'cuda'):
                loss, ce, kl = lm_loss(model, ids, tgt, plen, lengths, teacher=teacher, kd_t=C.get('kd_temperature', 2.0), kd_alpha=C.get('kd_alpha', 0.5),
                                       pos_weight=C.get('pos_weight'))
            scaler.scale(loss / C['accum']).backward(); run += float(ce); n += 1
            if kl is not None: kls.append(float(kl))
            if (pos + 1) % C['accum'] == 0 or pos + 1 == len(chunks):
                set_lr(step); scaler.unscale_(opt)
                nn.utils.clip_grad_norm_([p for g in opt.param_groups for p in g['params']], 1.0)
                scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True); step += 1
            if (pos + 1) % C['log_every'] == 0:
                log(f"epoch {epoch + 1} micro-batch {pos + 1}/{len(chunks)} train CE {run / n:.4f}" + (f' KL {np.mean(kls):.4f}' if kls else '') + f' | {(time.time() - t0) / 60:.1f} min')
            if (pos + 1) % C['accum'] == 0 and time.time() - t_save > 60 * C['save_minutes']:
                save_last(epoch, pos + 1); t_save = time.time()
            if (time.time() - t0) / 60 > C['max_minutes']:
                save_last(epoch, pos + 1); stopped = True
                log(f'MAX_MINUTES reached: saved last.pt at epoch {epoch + 1}, micro-batch {pos + 1}; run the cell again to continue'); break
        if stopped: break
        vl = eval_loss(model, va, device, amp_dtype)
        hist.append({'epoch': epoch + 1, 'train_ce': run / max(n, 1), 'val_loss': vl, 'minutes': (time.time() - t0) / 60, 'kl': float(np.mean(kls)) if kls else None})
        log(f"epoch {epoch + 1}/{C['epochs']} train CE {run / max(n, 1):.4f} | val loss {vl:.4f} (ppl {math.exp(vl):.2f})")
        if vl < best:
            best = vl; torch.save({'trainable': R.trainable_state(model), 'epoch': epoch + 1, 'val_loss': vl, 'C': C}, best_path)
        save_last(epoch + 1, 0); t_save = time.time()
    res = {'run_id': C['run_id'], 'kind': C['kind'], 'method': C['method'], 'strategy': C['strategy'], 'finished': not stopped,
           'trainable_params': trainable, 'total_params': total, 'trainable_pct': 100 * trainable / total, 'best_val_loss': best, 'history': hist,
           'peak_gpu_gb': torch.cuda.max_memory_allocated() / 1e9 if device.type == 'cuda' else None}
    (run_dir / 'train_results.json').write_text(json.dumps(res, indent=1, default=float), encoding='utf-8')
    return res, amp_dtype


def load_best(model, run_dir):
    s = torch.load(Path(run_dir) / 'best.pt', map_location='cpu', weights_only=False)
    missing = model.load_state_dict(s['trainable'], strict=False)
    if missing.unexpected_keys: raise ValueError(f'unexpected keys in best.pt: {missing.unexpected_keys[:5]}')
    return s


# ------------------------------------------------------------------------- generation
def word_start_table(tok):
    """True for tokens that begin a new word (ByteLevel 'Ġ' space or 'Ċ' newline): an incremental word count during generation."""
    v = tok.get_vocab_size(); t = torch.zeros(v, dtype=torch.bool)
    for s, i in tok.get_vocab().items():
        if i < v and (s.startswith('Ġ') or s.startswith('Ċ')): t[i] = True
    return t


def length_bounds(doc_words, L):
    """Per-document (min, max) words from the length policy L = dict(ratio, lo, hi, min_frac, max_frac)."""
    target = min(max(L['ratio'] * doc_words, L['lo']), L['hi'])
    return int(L['min_frac'] * target), int(L['max_frac'] * target)


@torch.inference_mode()
def generate(model, prompts, bounds, tok, *, device, amp_dtype, batch=16, no_repeat=3, max_new=None, beams=1, length_penalty=1.0, log=print, trace=None):
    """Greedy, batched. prompts: list of int64 arrays; bounds: list of (min_words, max_words). <eos> is blocked until min_words; a row stops at
    <eos>, at max_words, or when the context is full. Rows are grouped by prompt length so no padding is needed (gqa uses its KV cache).
    beams > 1 switches to beam_search with the same constraints. trace (greedy only): a list as long as prompts; trace[i] gets the tokens and,
    per step, the word count, whether the chosen token starts a word, and whether <eos> was the top token (for cut_trace)."""
    if beams > 1:
        if trace is not None: raise ValueError('trace needs greedy decoding (beams = 1)')
        return beam_search(model, prompts, bounds, tok, device=device, amp_dtype=amp_dtype, beams=beams, length_penalty=length_penalty,
                           batch=batch, no_repeat=no_repeat, max_new=max_new, log=log)
    model.eval(); ws = word_start_table(tok).to(device); V = tok.get_vocab_size()
    ban = [i for i in (tok.token_to_id('<unk>'), PAD_ID, MASK_ID) if i is not None and i < V]
    ctx = model.backbone.config.context_length; out = [None] * len(prompts)
    groups = {}
    for i, p in enumerate(prompts): groups.setdefault(len(p), []).append(i)
    jobs = [(L, g[k:k + batch]) for L, g in sorted(groups.items()) for k in range(0, len(g), batch)]
    t0 = time.time()
    for jn, (L, idx) in enumerate(jobs):
        b = len(idx); x = torch.tensor(np.stack([prompts[i] for i in idx]), device=device)
        lo = torch.tensor([bounds[i][0] for i in idx], device=device); hi = torch.tensor([bounds[i][1] for i in idx], device=device)
        words = torch.zeros(b, dtype=torch.long, device=device); done = torch.zeros(b, dtype=torch.bool, device=device)
        gen = [[] for _ in range(b)]; grams = [dict() for _ in range(b)]; rec = [dict(words=[], starts=[], eos=[]) for _ in range(b)]
        steps = min(ctx - L, max_new or ctx)
        plen = torch.full((b,), L, device=device); cache = None; seq = x
        for t in range(steps):
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=device.type == 'cuda'):
                if model.kind == 'gqa':
                    h, cache = model.hidden(seq if cache is None else seq[:, -1:], past=cache, use_cache=True)
                else:
                    h, _ = model.hidden(seq, plen)
                logits = model.logits_at(h[:, -1]).float()[:, :V]
            logits[:, ban] = float('-inf')
            eos = logits[:, EOS_ID].clone()
            logits[words < lo, EOS_ID] = float('-inf')
            if no_repeat >= 2:
                for r in range(b):
                    g = gen[r]
                    if not done[r] and len(g) >= no_repeat - 1:
                        bad = grams[r].get(tuple(g[-(no_repeat - 1):]))
                        if bad: logits[r, list(bad)] = float('-inf')
            nxt = logits.argmax(-1)
            over = ws[nxt] & (words >= hi)                    # this token would start word max_words + 1
            stop = (nxt == EOS_ID) | over
            if trace is not None:                             # <eos> never enters a no-repeat ban, so this is "argmax would be <eos> if allowed"
                top = (eos >= logits.max(-1).values).tolist(); wl, sl, dl = words.tolist(), ws[nxt].tolist(), done.tolist()
                for r in range(b):
                    if not dl[r]: rec[r]['words'].append(wl[r]); rec[r]['starts'].append(sl[r]); rec[r]['eos'].append(top[r])
            for r in range(b):
                if done[r] or stop[r]: continue
                g = gen[r]; v = int(nxt[r]); g.append(v)
                if len(g) >= no_repeat >= 2: grams[r].setdefault(tuple(g[-no_repeat:-1]), set()).add(v)
            words += (ws[nxt] & ~done & ~stop).long(); done |= stop
            if bool(done.all()): break
            seq = torch.cat([seq, torch.where(done, torch.full_like(nxt, PAD_ID), nxt)[:, None]], 1)
        for r, i in enumerate(idx):
            text = ' '.join(tok.decode(gen[r], skip_special_tokens=True).split())
            out[i] = ' '.join(text.split()[:bounds[i][1]]) if bounds[i][1] else text
            if trace is not None: trace[i] = dict(gen=gen[r], **rec[r])
        log(f'generated batch {jn + 1}/{len(jobs)} (prompt {L} tokens, {b} docs, {t + 1} steps) | {(time.time() - t0) / 60:.1f} min')
    return out


def cut_trace(tr, bound, tok):
    """The summary greedy decoding gives under bound = (min_words, max_words), read off the trace of a run whose <eos> was blocked throughout and
    whose max_words was at least as large: it stops at the first step where <eos> was the top token at >= min_words, or where the token would
    start word max_words + 1. Until then both runs choose the same tokens, so this equals a separate generate() call with that bound."""
    lo, hi = bound; end = len(tr['gen'])
    for t, (w, s, e) in enumerate(zip(tr['words'], tr['starts'], tr['eos'])):
        if (e and w >= lo) or (s and w >= hi): end = t; break
    text = ' '.join(tok.decode(tr['gen'][:end], skip_special_tokens=True).split())
    return ' '.join(text.split()[:hi]) if hi else text


def length_sweep(model, prompts, doc_words, rules, tok, **kw):
    """Greedy summaries under several length policies (dicts as for length_bounds) from one generation; returns one list of summaries per policy."""
    bounds = [[length_bounds(w, L) for w in doc_words] for L in rules]; trace = [None] * len(prompts)
    generate(model, prompts, [(10 ** 9, max(bs[i][1] for bs in bounds)) for i in range(len(prompts))], tok, trace=trace, **kw)
    return [[cut_trace(tr, b, tok) for tr, b in zip(trace, bs)] for bs in bounds]


@torch.inference_mode()
def beam_search(model, prompts, bounds, tok, *, device, amp_dtype, beams=4, length_penalty=1.0, batch=16, no_repeat=3, max_new=None, log=print):
    """Beam search under generate's constraints (banned tokens, <eos> blocked below min_words, no-repeat n-grams per hypothesis). At max_words,
    a token that would start a new word ends the hypothesis instead, as in greedy. Ended hypotheses are ranked by log-prob / tokens**length_penalty;
    a document stops once `beams` hypotheses have ended (early stopping). `batch` counts rows, so each step runs batch // beams documents x beams."""
    model.eval(); ws = word_start_table(tok).to(device); V = tok.get_vocab_size(); K = beams; NEG = float('-inf'); starts = ws.tolist()
    ban = [i for i in (tok.token_to_id('<unk>'), PAD_ID, MASK_ID) if i is not None and i < V]
    ctx = model.backbone.config.context_length; out = [None] * len(prompts); per = max(1, batch // K)
    groups = {}
    for i, p in enumerate(prompts): groups.setdefault(len(p), []).append(i)
    jobs = [(L, g[k:k + per]) for L, g in sorted(groups.items()) for k in range(0, len(g), per)]
    t0 = time.time()
    for jn, (L, idx) in enumerate(jobs):
        b = len(idx); n = b * K                                                     # row d*K + k = hypothesis k of document d
        seq = torch.tensor(np.stack([prompts[i] for i in idx]), device=device).repeat_interleave(K, 0)
        lo = torch.tensor([bounds[i][0] for i in idx], device=device).repeat_interleave(K)
        hi = torch.tensor([bounds[i][1] for i in idx], device=device).repeat_interleave(K)
        words = torch.zeros(n, dtype=torch.long, device=device)
        score = torch.full((n,), NEG, device=device); score[::K] = 0.                # the K copies start identical: only the first is live
        gen = [[] for _ in range(n)]; grams = [dict() for _ in range(n)]
        ended = [[] for _ in range(b)]; closed = [False] * b
        steps = min(ctx - L, max_new or ctx); plen = torch.full((n,), L, device=device); cache = None
        for t in range(steps):
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=device.type == 'cuda'):
                if model.kind == 'gqa':
                    h, cache = model.hidden(seq if cache is None else seq[:, -1:], past=cache, use_cache=True)
                else:
                    h, _ = model.hidden(seq, plen)
                logits = model.logits_at(h[:, -1]).float()[:, :V]
            logits[:, ban] = NEG
            logits[words < lo, EOS_ID] = NEG
            live = score > NEG; full = (words >= hi).tolist()
            if no_repeat >= 2:
                for r in range(n):
                    g = gen[r]
                    if live[r] and len(g) >= no_repeat - 1:
                        bad = grams[r].get(tuple(g[-(no_repeat - 1):]))
                        if bad: logits[r, list(bad)] = NEG
            cand = (torch.log_softmax(logits, -1) + score[:, None]).view(b, K * V)
            top_v, top_i = (z.tolist() for z in cand.topk(2 * K, -1))
            nxt = []                                                                 # (source row, token, score) for the n new rows
            for d in range(b):
                keep = []; stopped = set()
                if not closed[d]:
                    for rank, (v, i) in enumerate(zip(top_v[d], top_i[d])):
                        if v == NEG: break
                        r, w = d * K + i // V, i % V
                        if w == EOS_ID or (full[r] and starts[w]):                 # <eos>, or word max_words + 1 would start
                            if rank < K and r not in stopped:                      # one ended hypothesis per beam (its best way of stopping)
                                stopped.add(r); ended[d].append((v / (len(gen[r]) + 1) ** length_penalty, gen[r]))
                        else:
                            keep.append((r, w, v))
                        if len(keep) == K: break
                    if len(ended[d]) >= K or not keep: closed[d] = True
                if closed[d]: keep = []
                nxt += keep + [(d * K, PAD_ID, NEG)] * (K - len(keep))
            if all(closed): break
            src = torch.tensor([s for s, _, _ in nxt], device=device); toks = torch.tensor([w for _, w, _ in nxt], device=device)
            uses = Counter(s for s, _, _ in nxt); new_gen, new_grams = [], []
            for s, w, v in nxt:
                uses[s] -= 1; gr = grams[s] if uses[s] == 0 else dict(grams[s])       # values are tuples, so a shared dict is never mutated
                g = gen[s] + [w] if v > NEG else gen[s]
                if v > NEG and len(g) >= no_repeat >= 2:
                    key = tuple(g[-no_repeat:-1]); gr[key] = gr.get(key, ()) + (w,)
                new_gen.append(g); new_grams.append(gr)
            gen, grams = new_gen, new_grams
            score = torch.tensor([v for _, _, v in nxt], device=device)
            words = words[src] + (ws[toks] & (score > NEG)).long()
            seq = torch.cat([seq[src], toks[:, None]], 1)
            if cache is not None: cache = [tuple(z.index_select(0, src) for z in c) for c in cache]
        for d in range(b):                                                          # context or max_new reached: live beams count as ended
            if not closed[d]:
                ended[d] += [(float(score[r]) / max(len(gen[r]), 1) ** length_penalty, gen[r]) for r in range(d * K, d * K + K) if score[r] > NEG]
        for d, i in enumerate(idx):
            best = max(ended[d], key=lambda e: e[0])[1] if ended[d] else []
            text = ' '.join(tok.decode(best, skip_special_tokens=True).split())
            out[i] = ' '.join(text.split()[:bounds[i][1]]) if bounds[i][1] else text
        log(f'beam batch {jn + 1}/{len(jobs)} (prompt {L} tokens, {b} docs x {K} beams, {t + 1} steps) | {(time.time() - t0) / 60:.1f} min')
    return out


# ---------------------------------------------------------------------------- scoring
def _tok(t):
    return re.findall(r'[a-z0-9]+', t.lower())


def rouge(pred, ref):
    """ROUGE-1 / ROUGE-2 / ROUGE-L F1 (standard definitions, lowercase alphanumeric tokens, no stemming; the official scorer may stem)."""
    a, b = _tok(pred), _tok(ref); out = {}
    for n in (1, 2):
        ca = Counter(tuple(a[i:i + n]) for i in range(len(a) - n + 1)); cb = Counter(tuple(b[i:i + n]) for i in range(len(b) - n + 1))
        ov = sum((ca & cb).values()); p, r = ov / max(sum(ca.values()), 1), ov / max(sum(cb.values()), 1)
        out[f'rouge{n}'] = 2 * p * r / (p + r) if p + r else 0.0
    if len(a) * len(b) > 4e6: a, b = a[:2000], b[:2000]              # LCS is quadratic; very long pairs are cut
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b): cur.append(prev[j] + 1 if x == y else max(prev[j + 1], cur[j]))
        prev = cur
    l = prev[-1]; p, r = l / max(len(a), 1), l / max(len(b), 1); out['rougeL'] = 2 * p * r / (p + r) if p + r else 0.0
    return out


def score(preds, refs, *, bertscore=False, device='cuda'):
    """Mean ROUGE (+ optional BERTScore F1 via the bert-score package, evaluation only) and length statistics."""
    rs = [rouge(p, r) for p, r in zip(preds, refs)]
    out = {k: float(np.mean([x[k] for x in rs])) for k in ('rouge1', 'rouge2', 'rougeL')}
    out |= {'pred_words': float(np.mean([len(p.split()) for p in preds])), 'ref_words': float(np.mean([len(r.split()) for r in refs])),
            'rep4': float(np.mean([rep4(p) for p in preds]))}
    if bertscore:      # same call as the official evaluate_phase6.py: lang='en' -> roberta-large layer 17, no idf, no rescale, inputs cut at 512 tokens
        from bert_score import score as bs
        _, _, f = bs(preds, refs, lang='en', device=device, batch_size=16, verbose=False)
        out['bertscore_f1'] = float(f.mean())
        for x, v in zip(rs, f.tolist()): x['bertscore_f1'] = v
        from transformers import AutoTokenizer
        rt = AutoTokenizer.from_pretrained('roberta-large')
        for key, texts in (('pred_words_scored', preds), ('ref_words_scored', refs)):
            n = [words_in_window(t, rt) for t in texts]; out[key] = float(np.mean(n))
            for x, v in zip(rs, n): x[key] = v
        r1, rl = official_rouge(preds, refs); out['rouge1_official'], out['rougeL_official'] = float(np.mean(r1)), float(np.mean(rl))
        for x, a, c in zip(rs, r1, rl): x['rouge1_official'], x['rougeL_official'] = a, c
    return out, rs


def words_in_window(text, rt, limit=510):
    """Whitespace words of `text` that fall inside BERTScore's window (roberta-large: 512 tokens incl. <s> </s>); the rest is never scored."""
    enc = rt(text, add_special_tokens=False, return_offsets_mapping=True, verbose=False)
    if len(enc['input_ids']) <= limit: return len(text.split())
    return len(text[:enc['offset_mapping'][limit - 1][1]].split())


def official_rouge(preds, refs):
    """ROUGE-1 and ROUGE-L F1 as the official evaluate_phase6.py computes them: rouge-score's Porter-stemmed tokenizer, whole-text LCS."""
    try:
        from rouge_score import rouge_scorer, tokenizers
    except ImportError:
        import subprocess, sys
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'rouge-score'], check=True)
        from rouge_score import rouge_scorer, tokenizers
    sc = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=True); tk = tokenizers.DefaultTokenizer(use_stemmer=True); r1, rl = [], []
    for p, r in zip(preds, refs):
        p, r = p.strip(), r.strip(); r1.append(sc.score(r, p)['rouge1'].fmeasure)
        a, b = tk.tokenize(p), tk.tokenize(r)
        if not a or not b: rl.append(float(not a and not b)); continue
        rl.append(2 * lcs_len(a, b) / (len(a) + len(b)))                           # = 2PR/(P+R) with P = z/|a|, R = z/|b|
    return r1, rl


def lcs_len(a, b):
    """Bit-parallel LCS length (Allison-Dix), fast enough for 1000-word summaries."""
    if len(a) > len(b): a, b = b, a
    masks = {}
    for i, t in enumerate(a): masks[t] = masks.get(t, 0) | (1 << i)
    s = 0
    for t in b:
        x = masks.get(t, 0) | s; s = x & ~(x - ((s << 1) | 1))
    return bin(s).count('1')


def rep4(text):
    w = text.casefold().split(); g = [tuple(w[i:i + 4]) for i in range(len(w) - 3)]
    return (len(g) - len(set(g))) / len(g) if g else 0.0


def lead_baseline(split, L):
    """First k words of the document, k from the same length policy: a floor for any summarizer."""
    return [' '.join(r['document'].split()[:length_bounds(len(r['document'].split()), L)[1]]) for r in split]


def write_submission(ids, summaries, out_dir, expected_ids):
    """prediction.json (list format) at the root of submission.zip; ids must match the official test ids exactly."""
    if sorted(ids) != sorted(expected_ids) or len(set(ids)) != len(ids):
        raise ValueError('prediction ids do not match the test ids')
    if any(not s.strip() for s in summaries):
        raise ValueError('empty summary in the submission')
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    pred = [{'id': i, 'summary': s} for i, s in zip(ids, summaries)]
    (out_dir / 'prediction.json').write_text(json.dumps(pred, ensure_ascii=False, indent=1), encoding='utf-8')
    with zipfile.ZipFile(out_dir / 'submission.zip', 'w', zipfile.ZIP_DEFLATED) as z:
        z.write(out_dir / 'prediction.json', 'prediction.json')
    return out_dir / 'submission.zip'

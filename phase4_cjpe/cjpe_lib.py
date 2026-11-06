"""Phase 4 court-judgment-prediction (CJPE) helpers.

Binary outcome per case, long documents. A case is cut into token chunks, chunks are encoded by the
Phase 1 decoder (mean || last-token pooling per chunk), and a small head aggregates the chunk vectors
(mean / attention / BiGRU+attention) into one document vector -> 2 logits.

Reuses the Phase 3 adapter and freezing code (rr_lib: LoRA / rsLoRA / LoRMA / IA3, set_trainable,
param_groups) so every method from earlier phases is available. No pretrained weights or external APIs.
"""
from __future__ import annotations

import json
import math
import random
import time
import zipfile
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

import rr_lib as R
from legal_scaffold.models import ModelConfig
from training_methods import restore_backbone

DATASET_REPO = 'Exploration-Lab/CS-781-Capstone'
DATASET_FILES = ('Phase_4_CJPE/Train/train.json', 'Phase_4_CJPE/Train/dev.json', 'Phase_4_CJPE/Test/test.json')
PAD_ID, SEP_ID = R.PAD_ID, R.SEP_ID
SELECTIONS = ('first', 'last', 'uniform', 'head_tail', 'random')


# ----------------------------------------------------------------------------- data
def ensure_files(course_root, token):
    from huggingface_hub import hf_hub_download
    root = Path(course_root)
    for name in DATASET_FILES:
        if not (root / name).is_file():
            if not token:
                raise RuntimeError('HF_TOKEN missing from Colab secrets and files not on Drive')
            hf_hub_download(repo_id=DATASET_REPO, repo_type='dataset', filename=name,
                            local_dir=str(root), token=token)
    return root / 'Phase_4_CJPE'


def _text(v):
    return '\n'.join(map(str, v)) if isinstance(v, (list, tuple)) else str(v)


def load_split(path, *, keep_labels=True, dedup=False, log=print):
    """List of {'id','text','label'|None}. Tolerates list-of-dicts or id->dict layouts.
    keep_labels=False drops any label field (used for test: its labels are private).
    dedup=True removes rows whose text is identical to an earlier row (the released train/dev files
    contain the same case twice: ids repeat, text and label identical)."""
    raw = json.loads(Path(path).read_text(encoding='utf-8'))
    items = raw if isinstance(raw, list) else [dict(v, id=v.get('id', k)) for k, v in raw.items()]
    out, seen = [], set()
    for d in items:
        text = _text(d['text'])
        if dedup:
            key = hash(text)
            if key in seen:
                continue
            seen.add(key)
        lab = d.get('label', d.get('labels')) if keep_labels else None
        out.append({'id': str(d['id']), 'text': text, 'label': None if lab is None else int(lab)})
    if dedup and len(out) < len(items):
        log(f'{Path(path).name}: dropped {len(items) - len(out)} duplicate-text rows ({len(items)} -> {len(out)})')
    return out


def load_all(phase_dir):
    p = Path(phase_dir)
    return {'train': load_split(p / 'Train/train.json', dedup=True),
            'dev': load_split(p / 'Train/dev.json', dedup=True),
            'test': load_split(p / 'Test/test.json', keep_labels=False)}


def tokenize_docs(docs, tokenizer, max_tokens=None, batch=256):
    """Store the full token sequence as uint16 (vocab 32768): d['tok']. max_tokens keeps head+tail."""
    for i in range(0, len(docs), batch):
        part = docs[i:i + batch]
        for d, enc in zip(part, tokenizer.encode_batch([d['text'] for d in part])):
            ids = enc.ids or [SEP_ID]
            if max_tokens and len(ids) > max_tokens:
                h = max_tokens // 4
                ids = ids[:h] + ids[-(max_tokens - h):]
            d['tok'] = np.asarray(ids, dtype=np.uint16)
    return docs


CACHE_VERSION = 1


def save_cache(data, path, tokenizer_sha256, max_tokens):
    """One compressed .npz holding ids, labels and uint16 token arrays for every split (no text)."""
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    arrs = {'meta': np.array(json.dumps({'version': CACHE_VERSION, 'tokenizer_sha256': tokenizer_sha256,
                                         'max_tokens': max_tokens, 'splits': list(data)}))}
    for s, docs in data.items():
        arrs[f'{s}_tok'] = np.concatenate([d['tok'] for d in docs]).astype(np.uint16)
        arrs[f'{s}_len'] = np.array([len(d['tok']) for d in docs], dtype=np.int64)
        arrs[f'{s}_id'] = np.array([d['id'] for d in docs])
        arrs[f'{s}_label'] = np.array([-1 if d['label'] is None else d['label'] for d in docs], dtype=np.int8)
    np.savez_compressed(path, **arrs)
    return path


def load_cache(path, tokenizer_sha256, max_tokens, log=print):
    """Return the cached splits, or None if the file is missing or was built with a different tokenizer/cap."""
    path = Path(path)
    if not path.is_file():
        return None
    z = np.load(path, allow_pickle=False)
    meta = json.loads(str(z['meta']))
    if (meta.get('version') != CACHE_VERSION or meta.get('tokenizer_sha256') != tokenizer_sha256
            or meta.get('max_tokens') != max_tokens):
        log(f'ignoring {path.name}: built for a different tokenizer / MAX_TOKENS / version')
        return None
    data = {}
    for s in meta['splits']:
        tok, lens, ids, lab = z[f'{s}_tok'], z[f'{s}_len'], z[f'{s}_id'], z[f'{s}_label']
        off = np.concatenate([[0], np.cumsum(lens)])
        data[s] = [{'id': str(ids[i]), 'text': None, 'label': None if lab[i] < 0 else int(lab[i]),
                    'tok': tok[off[i]:off[i + 1]]} for i in range(len(lens))]
    return data


def summarize(data):
    out = {}
    for name, docs in data.items():
        n = np.array([len(d['tok']) for d in docs]) if docs and 'tok' in docs[0] else None
        row = {'docs': len(docs)}
        if docs and docs[0]['label'] is not None:
            lab = np.array([d['label'] for d in docs])
            row['label_1_share'] = round(float(lab.mean()), 3)
        if n is not None:
            row['tokens_mean/median/p90/max'] = [int(n.mean()), int(np.median(n)), int(np.percentile(n, 90)), int(n.max())]
        out[name] = row
    return out


# --------------------------------------------------------------------------- chunking
def chunk_spans(n, chunk_len, stride):
    last = max(n - chunk_len, 0)
    starts = list(range(0, last + 1, stride))
    if starts[-1] < last:
        starts.append(last)       # final chunk is aligned to the end of the document
    return [(s, min(s + chunk_len, n)) for s in starts]


def select_chunks(spans, k, mode, rng=None, train=False):
    n = len(spans)
    if n <= k:
        return spans
    if mode == 'first':
        return spans[:k]
    if mode == 'last':
        return spans[-k:]
    if mode == 'head_tail':
        return spans[:k // 2] + spans[-(k - k // 2):]
    if mode == 'random' and train and rng is not None:
        return [spans[i] for i in sorted(rng.sample(range(n), k))]
    idx = np.linspace(0, n - 1, k).round().astype(int)     # 'uniform' (and eval for 'random')
    return [spans[i] for i in idx]


def collate(docs, cfg, device, *, train=False, rng=None, select=None, max_chunks=None):
    select = select or cfg['select']
    k = max_chunks or cfg['max_chunks']
    rows, owner, slot, pos = [], [], [], []
    counts = []
    for bi, d in enumerate(docs):
        n = len(d['tok'])
        spans = select_chunks(chunk_spans(n, cfg['chunk_len'], cfg['stride']), k, select, rng, train)
        counts.append(len(spans))
        for j, (s, e) in enumerate(spans):
            rows.append(d['tok'][s:e]); owner.append(bi); slot.append(j)
            pos.append([s / max(n, 1), math.log1p(n) / 12.0])
    N, L = len(rows), max(len(r) for r in rows)
    ids = torch.full((N, L), PAD_ID, dtype=torch.long)
    mask = torch.zeros(N, L)
    for i, r in enumerate(rows):
        ids[i, :len(r)] = torch.from_numpy(r.astype(np.int64)); mask[i, :len(r)] = 1.0
    lens = mask.sum(1)
    batch = {'input_ids': ids.to(device), 'pool': (mask / lens[:, None]).to(device),
             'last': (lens.long() - 1).to(device), 'pos': torch.tensor(pos, dtype=torch.float32).to(device),
             'owner': torch.tensor(owner).to(device), 'slot': torch.tensor(slot).to(device),
             'n_docs': len(docs), 'max_slots': max(counts)}
    if docs[0].get('label') is not None:
        batch['labels'] = torch.tensor([d['label'] for d in docs]).to(device)
    return batch


# ---------------------------------------------------------------------------- model
class CJPEModel(nn.Module):
    """Phase-1 decoder -> per-chunk [mean||last] -> MLP (+position) -> chunk aggregation -> logits."""
    def __init__(self, backbone, n_classes=2, *, hidden=256, agg='attn', dropout=0.1, use_pos=True, pool='meanlast'):
        super().__init__()
        assert agg in ('mean', 'attn', 'bigru') and pool in ('meanlast', 'meanmaxlast')
        d = backbone.config.d_model
        self.backbone, self.prefix, self.agg, self.pool = backbone, None, agg, pool     # prefix=None: rr_lib.inject expects it
        self.proj = nn.Sequential(nn.Linear((3 if pool == 'meanmaxlast' else 2) * d, hidden), nn.GELU(), nn.Dropout(dropout))
        self.posmlp = nn.Sequential(nn.Linear(2, 32), nn.GELU(), nn.Linear(32, hidden)) if use_pos else None
        self.rnn = nn.GRU(hidden, hidden // 2, batch_first=True, bidirectional=True) if agg == 'bigru' else None
        self.att = nn.Sequential(nn.Linear(hidden, hidden // 2), nn.Tanh(), nn.Linear(hidden // 2, 1)) \
            if agg in ('attn', 'bigru') else None
        self.drop = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden, n_classes)
        self.keep_attn, self.last_attn = False, None

    def chunk_vectors(self, batch):
        h = R.backbone_hidden(self.backbone, batch['input_ids'])
        mean = torch.bmm(batch['pool'].unsqueeze(1).to(h.dtype), h).squeeze(1)
        last = h[torch.arange(h.shape[0], device=h.device), batch['last']]
        if self.pool == 'meanmaxlast':      # max over the chunk's tokens: keeps a sparse cue that the mean would dilute
            mx = h.masked_fill((batch['pool'] <= 0).unsqueeze(-1), float('-inf')).max(1).values
            return torch.cat([mean, mx, last], -1)
        return torch.cat([mean, last], -1)

    def head(self, vec, batch):
        z = self.proj(vec.float())
        if self.posmlp is not None:
            z = z + self.posmlp(batch['pos'])
        B, S = batch['n_docs'], batch['max_slots']
        zp = z.new_zeros(B, S, z.shape[-1]); zp[batch['owner'], batch['slot']] = z
        valid = torch.zeros(B, S, dtype=torch.bool, device=z.device); valid[batch['owner'], batch['slot']] = True
        if self.rnn is not None:
            lens = valid.sum(1).cpu()
            packed = nn.utils.rnn.pack_padded_sequence(zp, lens, batch_first=True, enforce_sorted=False)
            r, _ = self.rnn(packed)
            r, _ = nn.utils.rnn.pad_packed_sequence(r, batch_first=True, total_length=S)
            zp = zp + r
        if self.att is not None:
            a = self.att(zp).squeeze(-1).masked_fill(~valid, float('-inf')).softmax(-1)
        else:
            a = valid.float() / valid.sum(1, keepdim=True)
        if self.keep_attn:
            self.last_attn = a.detach().float().cpu()
        doc = (a.unsqueeze(-1) * zp).sum(1)
        return self.classifier(self.drop(doc))

    def forward(self, batch):
        return self.head(self.chunk_vectors(batch), batch)


def assemble(backbone, cfg_dict, method, *, head_kw, adapter_kw=None, layer_keep=None):
    if layer_keep is not None:      # thinned student cut from the Phase 1 model (e.g. 12 -> 6 blocks)
        R.thin_backbone(backbone, layer_keep)
    model = CJPEModel(backbone, **head_kw)
    R.inject(model, method, **(adapter_kw or {}))
    return model


def build_model(cfg_dict, checkpoint, method, **kw):
    backbone = R.load_backbone(cfg_dict, checkpoint)
    return assemble(backbone, cfg_dict, method, **kw)


# ------------------------------------------------------------------ metrics / inference
def macro_scores(y, p):
    y, p = np.asarray(y), np.asarray(p)
    per, f1s, ps, rs = {}, [], [], []
    for c in (0, 1):
        tp = int(((p == c) & (y == c)).sum()); fp = int(((p == c) & (y != c)).sum()); fn = int(((p != c) & (y == c)).sum())
        pr = tp / (tp + fp) if tp + fp else 0.0
        rc = tp / (tp + fn) if tp + fn else 0.0
        f = 2 * pr * rc / (pr + rc) if pr + rc else 0.0
        per[str(c)] = {'p': pr, 'r': rc, 'f1': f, 'support': int((y == c).sum())}
        f1s.append(f); ps.append(pr); rs.append(rc)
    conf = [[int(((y == g) & (p == q)).sum()) for q in (0, 1)] for g in (0, 1)]
    return {'macro_f1': float(np.mean(f1s)), 'macro_precision': float(np.mean(ps)), 'macro_recall': float(np.mean(rs)),
            'accuracy': float((y == p).mean()), 'per_class': per, 'confusion': conf}


def best_threshold(y, p1):
    grid = np.linspace(0.05, 0.95, 91)
    scores = [macro_scores(y, (p1 >= t).astype(int))['macro_f1'] for t in grid]
    i = int(np.argmax(scores))
    return float(grid[i]), float(scores[i])


@torch.no_grad()
def predict_probs(model, docs, cfg, device, amp_dtype, *, select=None, max_chunks=None, keep_attn=False):
    """P(label=1) per doc (order preserved). Optionally also per-doc attention over the selected chunks."""
    model.eval(); model.keep_attn = keep_attn
    order = sorted(range(len(docs)), key=lambda i: len(docs[i]['tok']))
    probs, attn = np.zeros(len(docs)), [None] * len(docs)
    for a in range(0, len(order), cfg['eval_docs']):
        idx = order[a:a + cfg['eval_docs']]
        batch = collate([docs[i] for i in idx], cfg, device, select=select or cfg['select'], max_chunks=max_chunks)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=device.type == 'cuda'):
            logits = model(batch)
        pr = F.softmax(logits.float(), -1)[:, 1].cpu().numpy()
        for r, i in enumerate(idx):
            probs[i] = pr[r]
            if keep_attn and model.last_attn is not None:
                row = model.last_attn[r]; attn[i] = row[~torch.isnan(row)].numpy()
    model.keep_attn = False
    return (probs, attn) if keep_attn else probs


def evaluate(model, docs, cfg, device, amp_dtype, **kw):
    t0 = time.time()
    p1 = predict_probs(model, docs, cfg, device, amp_dtype, **kw)
    sec = time.time() - t0
    y = np.array([d['label'] for d in docs])
    res = macro_scores(y, (p1 >= 0.5).astype(int))
    thr, f_tuned = best_threshold(y, p1)
    res.update({'threshold_tuned': thr, 'macro_f1_tuned': f_tuned, 'infer_sec_per_doc': sec / max(len(docs), 1)})
    return res, p1


# ----------------------------------------------------------------------------- training
def class_weights(docs, mode):
    if mode == 'none':
        return None
    cnt = np.array([sum(d['label'] == c for d in docs) for c in (0, 1)], dtype=float) + 1
    w = (cnt.sum() / (2 * cnt)) ** (0.5 if mode == 'sqrt' else 1.0)
    return torch.tensor(w / w.mean(), dtype=torch.float32)


def kd_loss(student_logits, teacher_logits, gold, *, temperature, alpha, weight=None):
    hard = F.cross_entropy(student_logits.float(), gold, weight=weight)
    soft = F.kl_div(F.log_softmax(student_logits.float() / temperature, -1),
                    F.softmax(teacher_logits.float() / temperature, -1), reduction='batchmean') * temperature ** 2
    return (1 - alpha) * hard + alpha * soft


def train_run(model, data, run_dir, C, device, *, teacher=None, log=print):
    """Per-epoch dev eval, best-macro-F1 checkpoint, resumable. Saves dev/test-ready results.json."""
    run_dir = Path(run_dir); run_dir.mkdir(parents=True, exist_ok=True)
    trainable, total = R.set_trainable(model, C['method'], C.get('partial_layers', ()))
    model.to(device)
    opt = torch.optim.AdamW(R.param_groups(model, C['method'], C['lr']), betas=(0.9, 0.98))
    n_train = len(data['train'])
    per_epoch = min(C.get('docs_per_epoch') or n_train, n_train)
    steps_epoch = math.ceil(per_epoch / (C['batch_docs'] * C['accum']))
    total_steps = steps_epoch * C['epochs']; warm = max(1, int(.06 * total_steps))
    base = [g['lr'] for g in opt.param_groups]

    def set_lr(s):
        f = (s + 1) / warm if s < warm else .5 * (1 + math.cos(math.pi * (s - warm) / max(1, total_steps - warm)))
        for g, b in zip(opt.param_groups, base):
            g['lr'] = b * max(f, 0.02)

    amp = torch.bfloat16 if (device.type == 'cuda' and torch.cuda.is_bf16_supported()) else torch.float16
    scaler = torch.amp.GradScaler('cuda', enabled=device.type == 'cuda' and amp == torch.float16)
    w = class_weights(data['train'], C['class_weight']); w = w.to(device) if w is not None else None
    state_p, best_p = run_dir / 'last.pt', run_dir / 'best.pt'
    hist, best, start, step = [], -1.0, 0, 0
    if state_p.is_file():
        s = torch.load(state_p, map_location='cpu', weights_only=False)
        model.load_state_dict(s['model'], strict=False); opt.load_state_dict(s['opt'])
        hist, best, start, step = s['hist'], s['best'], s['epoch'] + 1, s['step']
        log(f'resumed after epoch {start}, best dev macro-F1 {best:.4f}')
    rng = random.Random(C['seed'] + start); torch.manual_seed(C['seed'] + start)
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    for epoch in range(start, C['epochs']):
        model.train()
        if teacher is not None:
            teacher.eval()
        order = list(range(n_train)); rng.shuffle(order); order = order[:per_epoch]
        run_loss, n = 0.0, 0
        n_batches, t_ep, out_of_time = math.ceil(len(order) / C['batch_docs']), time.time(), False
        for bi, a in enumerate(range(0, len(order), C['batch_docs'])):
            docs = [data['train'][i] for i in order[a:a + C['batch_docs']]]
            batch = collate(docs, C, device, train=True, rng=rng)
            with torch.autocast(device_type=device.type, dtype=amp, enabled=device.type == 'cuda'):
                logits = model(batch)
                if teacher is not None:
                    with torch.no_grad():
                        t_logits = teacher(batch)
                    loss = kd_loss(logits, t_logits, batch['labels'], temperature=C['kd_temperature'],
                                   alpha=C['kd_alpha'], weight=w)
                else:
                    loss = F.cross_entropy(logits.float(), batch['labels'], weight=w,
                                           label_smoothing=C.get('label_smoothing', 0.0))
            scaler.scale(loss / C['accum']).backward()
            run_loss += float(loss.detach()); n += 1
            if (bi + 1) % C['accum'] == 0 or a + C['batch_docs'] >= len(order):
                set_lr(step); scaler.unscale_(opt)
                nn.utils.clip_grad_norm_([p for g in opt.param_groups for p in g['params']], 1.0)
                scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True); step += 1
            if C.get('log_every', 50) and (bi + 1) % C.get('log_every', 50) == 0:
                el = (time.time() - t_ep) / 60
                log(f"  epoch {epoch+1} batch {bi+1}/{n_batches} loss {run_loss/n:.4f} | {el:.1f} min so far, ~{el*(n_batches/(bi+1)-1):.1f} min left in this epoch")
            if C.get('max_minutes') and (time.time() - t0) / 60 > C['max_minutes']:
                out_of_time = True      # time budget hit mid-epoch: stop here, still evaluate and keep best.pt
                break
        res, _ = evaluate(model, data['dev'], C, device, amp)
        hist.append({'epoch': epoch + 1, 'train_loss': run_loss / max(n, 1), 'dev_macro_f1': res['macro_f1'],
                     'dev_macro_f1_tuned': res['macro_f1_tuned'], 'dev_acc': res['accuracy'],
                     'minutes': (time.time() - t0) / 60})
        log(f"epoch {epoch+1}/{C['epochs']} loss {run_loss/max(n,1):.4f} | dev macro-F1 {res['macro_f1']:.4f} "
            f"(tuned {res['macro_f1_tuned']:.4f} @ {res['threshold_tuned']:.2f}) acc {res['accuracy']:.3f} | {hist[-1]['minutes']:.1f} min")
        if res['macro_f1'] > best:
            best = res['macro_f1']
            torch.save({'trainable': R.trainable_state(model), 'epoch': epoch + 1, 'dev': res, 'C': C}, best_p)
        torch.save({'model': R.trainable_state(model), 'opt': opt.state_dict(), 'hist': hist, 'best': best,
                    'epoch': epoch, 'step': step}, state_p)
        if C.get('max_minutes') and (time.time() - t0) / 60 > C['max_minutes']:
            log(f"stopping after epoch {epoch+1}: time budget {C['max_minutes']} min used")
            break
    saved = torch.load(best_p, map_location='cpu', weights_only=False)
    model.load_state_dict(saved['trainable'], strict=False)
    final, p_dev = evaluate(model, data['dev'], C, device, amp)
    results = {'method': C['method'], 'run_id': C['run_id'], 'preset': C['preset'], 'trainable_params': trainable,
               'total_params': total, 'trainable_pct': 100 * trainable / total, 'best_epoch': saved['epoch'],
               'train_minutes': hist[-1]['minutes'] if hist else None,
               'peak_gpu_gb': torch.cuda.max_memory_allocated() / 1e9 if device.type == 'cuda' else None,
               'setup': {k: C[k] for k in ('chunk_len', 'stride', 'max_chunks', 'select', 'agg', 'class_weight', 'docs_per_epoch')},
               'dev': final, 'history': hist}
    (run_dir / 'results.json').write_text(json.dumps(results, indent=1), encoding='utf-8')
    np.save(run_dir / 'dev_probs.npy', p_dev)
    return results, amp


def write_submission(probs, docs, threshold, out_dir):
    """prediction.json = [{"id":..., "pred": 0|1}, ...] at the root of submission.zip."""
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    preds = [{'id': d['id'], 'pred': int(p >= threshold)} for d, p in zip(docs, probs)]
    assert len({p['id'] for p in preds}) == len(docs) and all(p['pred'] in (0, 1) for p in preds)
    path = out / 'prediction.json'
    path.write_text(json.dumps(preds), encoding='utf-8')
    with zipfile.ZipFile(out / 'submission.zip', 'w', zipfile.ZIP_DEFLATED) as z:
        z.write(path, arcname='prediction.json')
    return path


# ----------------------------------------------------------------------------- analysis
def attention_profile(model, docs, cfg, device, amp_dtype, bins=5):
    """Mean attention mass per relative-position bin of the selected chunks (which sections matter)."""
    _, attn = predict_probs(model, docs, cfg, device, amp_dtype, keep_attn=True)
    acc = np.zeros(bins); cnt = np.zeros(bins)
    for a in attn:
        if a is None or len(a) < 2:
            continue
        for j, v in enumerate(a):
            b = min(int(j / len(a) * bins), bins - 1); acc[b] += v; cnt[b] += 1
    return (acc / np.maximum(cnt, 1)).tolist()


def selection_ablation(model, docs, cfg, device, amp_dtype, modes=('first', 'last', 'uniform', 'head_tail'),
                       ks=(1, 2, 4, 8)):
    """Evaluate a trained model with different eval-time chunk selections (no retraining)."""
    y = np.array([d['label'] for d in docs]); rows = []
    for m in modes:
        for k in ks:
            p = predict_probs(model, docs, cfg, device, amp_dtype, select=m, max_chunks=k)
            rows.append({'select': m, 'chunks': k, 'macro_f1': round(macro_scores(y, (p >= 0.5).astype(int))['macro_f1'], 4)})
    return rows


# ------------------------------------------------------------------------------ export
def export_run(model, data, C, device, amp_dtype, run_root, log=print):
    """Score dev and test with the loaded (best) weights; write probabilities and two submissions:
    submission/ (threshold 0.5) and submission_tuned/ (dev-tuned threshold)."""
    run_root = Path(run_root)
    dev_res, p_dev = evaluate(model, data['dev'], C, device, amp_dtype)
    p_test = predict_probs(model, data['test'], C, device, amp_dtype)
    for name, docs, p in (('dev', data['dev'], p_dev), ('test', data['test'], p_test)):
        (run_root / f'{name}_probs.json').write_text(json.dumps({d['id']: float(v) for d, v in zip(docs, p)}))
    thr = dev_res['threshold_tuned']; out = {}
    for tag, t in (('submission', 0.5), ('submission_tuned', thr)):
        path = write_submission(p_test, data['test'], t, run_root / tag)
        with zipfile.ZipFile(run_root / tag / 'submission.zip') as z:
            assert z.namelist() == ['prediction.json']
            chk = json.loads(z.read('prediction.json'))
        assert [c['id'] for c in chk] == [d['id'] for d in data['test']]
        out[tag] = {'threshold': t, 'share_pred_1': float(np.mean([c['pred'] for c in chk])), 'zip': str(run_root / tag / 'submission.zip')}
        log(f"{tag}: threshold {t:.2f}, predicts label 1 for {100*out[tag]['share_pred_1']:.1f}% of {len(chk)} test cases -> {out[tag]['zip']}")
    return dev_res, p_dev, p_test, out


def error_table(docs, probs, thr=0.5):
    """Where do false positives / false negatives sit? (outcome class, document length)"""
    rows = {}
    for d, p in zip(docs, probs):
        pred = int(p >= thr); key = ('TP' if pred and d['label'] else 'TN' if not pred and not d['label']
                                     else 'FP' if pred else 'FN')
        rows.setdefault(key, []).append(len(d['tok']))
    return [{'case': k, 'n': len(v), 'tokens_mean': int(np.mean(v)), 'tokens_median': int(np.median(v))}
            for k, v in sorted(rows.items())]

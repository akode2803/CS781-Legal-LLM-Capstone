"""Phase 5 legal statute identification (LSI): multi-label classification of long documents.

A document is cut into token chunks, chunks are encoded by the Phase 1 decoder (mean || last-token pooling), a head aggregates the
chunk vectors (cjpe_lib.CJPEModel with n_classes = 100) and emits one logit per statute. Loss: sigmoid + binary cross-entropy
(optional positive weights / focal term). Prediction: a set of statute ids from per-statute thresholds.

Reuses cjpe_lib (chunking, collate, model, tokenisation) and rr_lib (adapters, freezing policies). No pretrained weights, no external APIs.
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

import cjpe_lib as J
import rr_lib as R

DATASET_REPO = 'Exploration-Lab/CS-781-Capstone'
DATASET_FILES = ('Phase_5_LSI/Train/train.json', 'Phase_5_LSI/Train/dev.json', 'Phase_5_LSI/Test/test.json')
N_LABELS = 100          # statute ids 0..99 (checked against train + dev at load time)


# ----------------------------------------------------------------------------- data
def ensure_files(course_root, token):
    from huggingface_hub import hf_hub_download
    root = Path(course_root)
    for name in DATASET_FILES:
        if not (root / name).is_file():
            if not token:
                raise RuntimeError('HF_TOKEN missing from Colab secrets and files not on Drive')
            hf_hub_download(repo_id=DATASET_REPO, repo_type='dataset', filename=name, local_dir=str(root), token=token)
    return root / 'Phase_5_LSI'


def load_split(path, *, keep_labels=True, dedup=False, log=print):
    """[{'id','text','labels': sorted unique ints | None}]. dedup removes only EXACT duplicates (same text AND same label set):
    several rows share a text but have different label sets (generic snippets), and those are distinct cases that must be kept."""
    items = json.loads(Path(path).read_text(encoding='utf-8'))
    items = items if isinstance(items, list) else [dict(v, id=v.get('id', k)) for k, v in items.items()]
    out, seen = [], set()
    for d in items:
        text = J._text(d['text'])
        labels = sorted({int(x) for x in d['labels']}) if keep_labels and d.get('labels') is not None else None
        if dedup:
            key = (hash(text), tuple(labels or ()))
            if key in seen:
                continue
            seen.add(key)
        out.append({'id': str(d['id']), 'text': text, 'labels': labels})
    if dedup and len(out) < len(items):
        log(f'{Path(path).name}: dropped {len(items) - len(out)} exact duplicates ({len(items)} -> {len(out)})')
    return out


def load_all(phase_dir):
    p = Path(phase_dir)
    data = {'train': load_split(p / 'Train/train.json', dedup=True), 'dev': load_split(p / 'Train/dev.json', dedup=True),
            'test': load_split(p / 'Test/test.json', keep_labels=False)}
    ids = {l for s in ('train', 'dev') for d in data[s] for l in d['labels']}
    assert ids <= set(range(N_LABELS)), f'unexpected statute ids {sorted(ids - set(range(N_LABELS)))[:5]}'
    return data


def multi_hot(docs, k=N_LABELS):
    y = np.zeros((len(docs), k), dtype=np.uint8)
    for i, d in enumerate(docs):
        y[i, d['labels']] = 1
    return y


def save_cache(data, path, tokenizer_sha256, max_tokens):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    arrs = {'meta': np.array(json.dumps({'version': 1, 'tokenizer_sha256': tokenizer_sha256, 'max_tokens': max_tokens, 'splits': list(data)}))}
    for s, docs in data.items():
        arrs[f'{s}_tok'] = np.concatenate([d['tok'] for d in docs]).astype(np.uint16)
        arrs[f'{s}_len'] = np.array([len(d['tok']) for d in docs], dtype=np.int64)
        arrs[f'{s}_id'] = np.array([d['id'] for d in docs])
        has = docs[0]['labels'] is not None
        arrs[f'{s}_haslab'] = np.array(has)
        arrs[f'{s}_labptr'] = np.cumsum([0] + [len(d['labels']) if has else 0 for d in docs]).astype(np.int64)
        arrs[f'{s}_labidx'] = np.array([l for d in docs for l in (d['labels'] if has else [])], dtype=np.int16)
    np.savez_compressed(path, **arrs)
    return path


def load_cache(path, tokenizer_sha256, max_tokens, log=print):
    path = Path(path)
    if not path.is_file():
        return None
    z = np.load(path, allow_pickle=False); meta = json.loads(str(z['meta']))
    if meta.get('version') != 1 or meta.get('tokenizer_sha256') != tokenizer_sha256 or meta.get('max_tokens') != max_tokens:
        log(f'ignoring {path.name}: built for a different tokenizer / MAX_TOKENS / version'); return None
    data = {}
    for s in meta['splits']:
        tok, lens, ids = z[f'{s}_tok'], z[f'{s}_len'], z[f'{s}_id']
        off = np.concatenate([[0], np.cumsum(lens)]); has = bool(z[f'{s}_haslab']); lp, li = z[f'{s}_labptr'], z[f'{s}_labidx']
        data[s] = [{'id': str(ids[i]), 'text': None, 'tok': tok[off[i]:off[i + 1]],
                    'labels': [int(x) for x in li[lp[i]:lp[i + 1]]] if has else None} for i in range(len(lens))]
    return data


def collate(docs, cfg, device, **kw):
    """cjpe_lib.collate + a multi-hot target matrix batch['y'] when the documents carry labels."""
    batch = J.collate(docs, cfg, device, **kw)
    if docs[0].get('labels') is not None:
        y = torch.zeros(len(docs), N_LABELS)
        for i, d in enumerate(docs):
            y[i, d['labels']] = 1.0
        batch['y'] = y.to(device)
    return batch


# --------------------------------------------------------------------- metrics, thresholds
def _prf(Y, pred):
    Y, pred = Y.astype(bool), pred.astype(bool)
    tp, fp, fn = (pred & Y).sum(0), (pred & ~Y).sum(0), (~pred & Y).sum(0)
    return tp, fp, fn


def scores(Y, pred):
    tp, fp, fn = _prf(Y, pred)
    den = 2 * tp + fp + fn
    f1 = np.where(den > 0, 2 * tp / np.maximum(den, 1), 0.0)
    prec = np.where(tp + fp > 0, tp / np.maximum(tp + fp, 1), 0.0); rec = np.where(tp + fn > 0, tp / np.maximum(tp + fn, 1), 0.0)
    return {'macro_f1': float(f1.mean()), 'micro_f1': float(2 * tp.sum() / max(2 * tp.sum() + fp.sum() + fn.sum(), 1)),
            'subset_acc': float((pred.astype(bool) == Y.astype(bool)).all(1).mean()),
            'macro_precision': float(prec.mean()), 'macro_recall': float(rec.mean()),
            'labels_per_doc_pred': float(pred.sum(1).mean()), 'per_label_f1': f1.tolist()}


GRID = np.round(np.arange(0.02, 0.9, 0.01), 2)


def _f1_by_threshold(Y, P, grid=GRID):
    """F1 per (threshold, label): shape [G, K]."""
    Yb = Y.astype(bool); out = np.zeros((len(grid), Y.shape[1]))
    for g, t in enumerate(grid):
        pred = P >= t; tp = (pred & Yb).sum(0); den = 2 * tp + (pred & ~Yb).sum(0) + (~pred & Yb).sum(0)
        out[g] = np.where(den > 0, 2 * tp / np.maximum(den, 1), 0.0)
    return out


def tune_global(Y, P, grid=GRID):
    f = _f1_by_threshold(Y, P, grid).mean(1); i = int(np.argmax(f)); return float(grid[i]), float(f[i])


def tune_per_label(Y, P, grid=GRID):
    f = _f1_by_threshold(Y, P, grid); return grid[np.argmax(f, 0)].astype(float)


def decide(P, thr, min_one=True):
    """Predicted statute sets as a boolean matrix. min_one: every document gets at least its highest-scoring statute
    (every train/dev document has >= 1 statute)."""
    thr = np.asarray(thr, dtype=float); pred = P >= thr
    if min_one:
        none = ~pred.any(1)
        if none.any():
            top = np.argmax(P[none] / np.maximum(thr if thr.ndim == 0 else thr[None, :], 1e-6), 1)
            idx = np.where(none)[0]; pred[idx, top] = True
    return pred


def cross_fit_per_label(Y, P, k=2, seed=0, min_one=True):
    """Honest estimate of per-label thresholds: tune on k-1 folds of dev, score on the held-out fold."""
    order = np.random.RandomState(seed).permutation(len(Y)); folds = np.array_split(order, k); pred = np.zeros(Y.shape, bool)
    for f in range(k):
        te = folds[f]; tr = np.concatenate([folds[g] for g in range(k) if g != f])
        pred[te] = decide(P[te], tune_per_label(Y[tr], P[tr]), min_one)
    return scores(Y, pred)


def evaluate_probs(Y, P):
    """The threshold strategies the report compares, all on one probability matrix."""
    g, _ = tune_global(Y, P)
    out = {'global_threshold': g,
           'fixed_0.5': scores(Y, decide(P, 0.5, False)),
           'global_tuned': scores(Y, decide(P, g, False)),
           'global_tuned+min_one': scores(Y, decide(P, g, True)),
           'per_label_in_sample+min_one': scores(Y, decide(P, tune_per_label(Y, P), True)),
           'per_label_cross_fit+min_one': cross_fit_per_label(Y, P)}
    return out


# ------------------------------------------------------------------------------ inference
@torch.no_grad()
def predict_probs(model, docs, cfg, device, amp_dtype, *, select=None, max_chunks=None):
    model.eval()
    order = sorted(range(len(docs)), key=lambda i: len(docs[i]['tok']))
    P = np.zeros((len(docs), N_LABELS), dtype=np.float32)
    for a in range(0, len(order), cfg['eval_docs']):
        idx = order[a:a + cfg['eval_docs']]
        batch = collate([docs[i] for i in idx], cfg, device, select=select or cfg['select'], max_chunks=max_chunks)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=device.type == 'cuda'):
            logits = model(batch)
        P[idx] = torch.sigmoid(logits.float()).cpu().numpy()
    return P


# ---------------------------------------------------------------------------------- loss
def pos_weights(train_docs, mode, clip=30.0):
    if mode == 'none':
        return None
    n = len(train_docs); cnt = multi_hot(train_docs).sum(0).astype(float) + 1.0
    w = np.minimum(((n - cnt) / cnt) ** (0.5 if mode == 'sqrt' else 1.0), clip)
    return torch.tensor(w, dtype=torch.float32)


def lsi_loss(logits, y, *, pos_weight=None, focal_gamma=0.0):
    bce = F.binary_cross_entropy_with_logits(logits.float(), y, reduction='none', pos_weight=pos_weight)
    if focal_gamma:
        p = torch.sigmoid(logits.float()); pt = y * p + (1 - y) * (1 - p); bce = bce * (1 - pt) ** focal_gamma
    return bce.mean()


def kd_loss(student_logits, teacher_logits, y, *, temperature, alpha, pos_weight=None):
    hard = lsi_loss(student_logits, y, pos_weight=pos_weight)
    soft = F.binary_cross_entropy_with_logits(student_logits.float() / temperature, torch.sigmoid(teacher_logits.float() / temperature)) * temperature ** 2
    return (1 - alpha) * hard + alpha * soft


# ----------------------------------------------------------------------------- training
def train_run(model, data, run_dir, C, device, *, teacher=None, log=print):
    """Per-epoch evaluation on a dev subset, best-epoch checkpoint by GLOBAL-TUNED macro-F1 (not a fixed 0.5 cut-off), resumable,
    progress lines inside epochs, mid-epoch time cap. Finishes with a full-dev evaluation."""
    run_dir = Path(run_dir); run_dir.mkdir(parents=True, exist_ok=True)
    trainable, total = R.set_trainable(model, C['method'], C.get('partial_layers', ()))
    model.to(device)
    opt = torch.optim.AdamW(R.param_groups(model, C['method'], C['lr']), betas=(0.9, 0.98))
    n_train = len(data['train']); per_epoch = min(C.get('docs_per_epoch') or n_train, n_train)
    steps_epoch = math.ceil(per_epoch / (C['batch_docs'] * C['accum'])); total_steps = steps_epoch * C['epochs']; warm = max(1, int(.06 * total_steps))
    base = [g['lr'] for g in opt.param_groups]

    def set_lr(s):
        f = (s + 1) / warm if s < warm else .5 * (1 + math.cos(math.pi * (s - warm) / max(1, total_steps - warm)))
        for g, b in zip(opt.param_groups, base):
            g['lr'] = b * max(f, 0.02)

    amp = torch.bfloat16 if (device.type == 'cuda' and torch.cuda.is_bf16_supported()) else torch.float16
    scaler = torch.amp.GradScaler('cuda', enabled=device.type == 'cuda' and amp == torch.float16)
    pw = pos_weights(data['train'], C['class_weight']); pw = pw.to(device) if pw is not None else None
    dev_all = data['dev']; sub = np.random.RandomState(0).permutation(len(dev_all))[:C.get('dev_subset') or len(dev_all)]
    dev_sub = [dev_all[i] for i in sorted(sub)]; Y_sub = multi_hot(dev_sub)
    state_p, best_p = run_dir / 'last.pt', run_dir / 'best.pt'
    hist, best, start, step = [], -1.0, 0, 0
    if state_p.is_file():
        s = torch.load(state_p, map_location='cpu', weights_only=False)
        model.load_state_dict(s['model'], strict=False); opt.load_state_dict(s['opt'])
        hist, best, start, step = s['hist'], s['best'], s['epoch'] + 1, s['step']
        log(f'resumed after epoch {start}, best dev-subset macro-F1 (global-tuned) {best:.4f}')
    if not state_p.is_file():     # fresh run: start every statute's output bias at its log-odds prior (the head then begins at the base rates)
        pr = torch.tensor((multi_hot(data['train']).sum(0) + 1.0) / (n_train + 2.0), dtype=torch.float32)
        with torch.no_grad():
            model.classifier.bias.copy_(torch.log(pr / (1 - pr)).to(model.classifier.bias.device))
    rng = random.Random(C['seed'] + start); torch.manual_seed(C['seed'] + start)
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    for epoch in range(start, C['epochs']):
        model.train()
        if teacher is not None:
            teacher.eval()
        order = list(range(n_train)); rng.shuffle(order); order = order[:per_epoch]
        run_loss, n = 0.0, 0; n_batches, t_ep = math.ceil(len(order) / C['batch_docs']), time.time()
        for bi, a in enumerate(range(0, len(order), C['batch_docs'])):
            docs = [data['train'][i] for i in order[a:a + C['batch_docs']]]
            batch = collate(docs, C, device, train=True, rng=rng)
            with torch.autocast(device_type=device.type, dtype=amp, enabled=device.type == 'cuda'):
                logits = model(batch)
                if teacher is not None:
                    with torch.no_grad():
                        t_logits = teacher(batch)
                    loss = kd_loss(logits, t_logits, batch['y'], temperature=C['kd_temperature'], alpha=C['kd_alpha'], pos_weight=pw)
                else:
                    loss = lsi_loss(logits, batch['y'], pos_weight=pw, focal_gamma=C.get('focal_gamma', 0.0))
            scaler.scale(loss / C['accum']).backward(); run_loss += float(loss.detach()); n += 1
            if (bi + 1) % C['accum'] == 0 or a + C['batch_docs'] >= len(order):
                set_lr(step); scaler.unscale_(opt)
                nn.utils.clip_grad_norm_([p for g in opt.param_groups for p in g['params']], 1.0)
                scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True); step += 1
            if C.get('log_every', 100) and (bi + 1) % C.get('log_every', 100) == 0:
                el = (time.time() - t_ep) / 60
                log(f"  epoch {epoch+1} batch {bi+1}/{n_batches} loss {run_loss/n:.4f} | {el:.1f} min so far, ~{el*(n_batches/(bi+1)-1):.1f} min left in this epoch")
            if C.get('max_minutes') and (time.time() - t0) / 60 > C['max_minutes']:
                break
        P = predict_probs(model, dev_sub, C, device, amp); thr, f_tuned = tune_global(Y_sub, P)
        s05 = scores(Y_sub, decide(P, 0.5, False)); mi = scores(Y_sub, decide(P, thr, False))['micro_f1']
        hist.append({'epoch': epoch + 1, 'train_loss': run_loss / max(n, 1), 'dev_macro_f1_tuned': f_tuned, 'dev_threshold': thr, 'dev_micro_f1_tuned': mi,
                     'dev_macro_f1_at_0.5': s05['macro_f1'], 'minutes': (time.time() - t0) / 60})
        log(f"epoch {epoch+1}/{C['epochs']} loss {run_loss/max(n,1):.4f} | dev-subset macro-F1 {f_tuned:.4f} @ thr {thr:.2f} (at 0.5: {s05['macro_f1']:.4f}) micro-F1 {mi:.4f} | {hist[-1]['minutes']:.1f} min")
        if f_tuned > best:
            best = f_tuned; torch.save({'trainable': R.trainable_state(model), 'epoch': epoch + 1, 'C': C}, best_p)
        torch.save({'model': R.trainable_state(model), 'opt': opt.state_dict(), 'hist': hist, 'best': best, 'epoch': epoch, 'step': step}, state_p)
        if C.get('max_minutes') and (time.time() - t0) / 60 > C['max_minutes']:
            log(f"stopping after epoch {epoch+1}: time budget {C['max_minutes']} min used"); break
    saved = torch.load(best_p, map_location='cpu', weights_only=False); model.load_state_dict(saved['trainable'], strict=False)
    t1 = time.time(); P_dev = predict_probs(model, dev_all, C, device, amp); sec_per_doc = (time.time() - t1) / len(dev_all)
    ev = evaluate_probs(multi_hot(dev_all), P_dev)
    results = {'method': C['method'], 'run_id': C['run_id'], 'preset': C['preset'], 'trainable_params': trainable, 'total_params': total,
               'trainable_pct': 100 * trainable / total, 'best_epoch': saved['epoch'], 'train_minutes': hist[-1]['minutes'] if hist else None,
               'peak_gpu_gb': torch.cuda.max_memory_allocated() / 1e9 if device.type == 'cuda' else None, 'infer_sec_per_doc': sec_per_doc,
               'setup': {k: C[k] for k in ('chunk_len', 'stride', 'max_chunks', 'select', 'agg', 'class_weight', 'docs_per_epoch')},
               'dev': {k: ({kk: vv for kk, vv in v.items() if kk != 'per_label_f1'} if isinstance(v, dict) else v) for k, v in ev.items()}, 'history': hist}
    (run_dir / 'results.json').write_text(json.dumps(results, indent=1), encoding='utf-8')
    return results, amp, P_dev


# ------------------------------------------------------------------------------- export
def write_submission(P_test, ids, pred_bool, out_dir):
    """prediction.json = [{"id": ..., "pred": [statute ids]}] at the root of submission.zip."""
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    rows = [{'id': i, 'pred': [int(k) for k in np.flatnonzero(pred_bool[r])]} for r, i in enumerate(ids)]
    assert len({r['id'] for r in rows}) == len(rows) and all(len(r['pred']) == len(set(r['pred'])) and all(0 <= k < N_LABELS for k in r['pred']) for r in rows)
    path = out / 'prediction.json'; path.write_text(json.dumps(rows), encoding='utf-8')
    with zipfile.ZipFile(out / 'submission.zip', 'w', zipfile.ZIP_DEFLATED) as z:
        z.write(path, arcname='prediction.json')
    return out / 'submission.zip'


def export_run(model, data, C, device, amp_dtype, run_root, log=print, P_dev=None):
    """Probabilities for dev and test (saved for ensembling) + two submissions:
    submission_global/ (one tuned threshold, at-least-one statute) and submission_perlabel/ (per-statute thresholds, at-least-one)."""
    run_root = Path(run_root); run_root.mkdir(parents=True, exist_ok=True)
    if P_dev is None:
        P_dev = predict_probs(model, data['dev'], C, device, amp_dtype)
    P_test = predict_probs(model, data['test'], C, device, amp_dtype); Y = multi_hot(data['dev'])
    np.save(run_root / 'dev_probs.npy', P_dev); np.save(run_root / 'test_probs.npy', P_test)
    (run_root / 'test_ids.json').write_text(json.dumps([d['id'] for d in data['test']])); (run_root / 'dev_ids.json').write_text(json.dumps([d['id'] for d in data['dev']]))
    g, _ = tune_global(Y, P_dev); th = tune_per_label(Y, P_dev); ids = [d['id'] for d in data['test']]; out = {}
    for tag, thr in (('submission_global', g), ('submission_perlabel', th)):
        pred = decide(P_test, thr, True); path = write_submission(P_test, ids, pred, run_root / tag)
        with zipfile.ZipFile(path) as z:
            assert z.namelist() == ['prediction.json']; chk = json.loads(z.read('prediction.json'))
        assert [r['id'] for r in chk] == ids and all(len(r['pred']) >= 1 for r in chk)
        out[tag] = {'zip': str(path), 'labels_per_doc': float(pred.sum(1).mean())}
        log(f"{tag}: predicts {out[tag]['labels_per_doc']:.2f} statutes per document on average -> {path}")
    return evaluate_probs(Y, P_dev), P_dev, P_test, out


# ---------------------------------------------------------------------------- analysis
def per_label_table(Y, P, thr, train_docs=None):
    pred = decide(P, thr, True); s = scores(Y, pred); tp, fp, fn = _prf(Y, pred)
    rows = [{'statute': k, 'dev_docs': int(Y[:, k].sum()), 'predicted': int(pred[:, k].sum()), 'F1': round(s['per_label_f1'][k], 4),
             'precision': round(float(tp[k] / max(tp[k] + fp[k], 1)), 3), 'recall': round(float(tp[k] / max(tp[k] + fn[k], 1)), 3)} for k in range(Y.shape[1])]
    if train_docs is not None:
        cnt = multi_hot(train_docs).sum(0)
        for r in rows:
            r['train_docs'] = int(cnt[r['statute']])
    return rows


def confusion_pairs(Y, pred, top=15):
    """(missed statute a, wrongly predicted statute b) pairs that co-occur in the same document."""
    Yb, pb = Y.astype(bool), pred.astype(bool); miss, wrong = Yb & ~pb, pb & ~Yb
    M = miss.T.astype(np.int32) @ wrong.astype(np.int32); np.fill_diagonal(M, 0)
    idx = np.dstack(np.unravel_index(np.argsort(-M, axis=None)[:top], M.shape))[0]
    return [{'missed': int(a), 'predicted_instead': int(b), 'docs': int(M[a, b])} for a, b in idx]


def stack_cooccurrence_cv(Y, P, k=5, seed=0):
    """Do label relationships help? Second-stage logistic regression on the 100 first-stage probabilities (cross-validated on dev),
    compared with the independent predictions by the caller. The logits are standardised on the training folds (so lbfgs converges) and the
    labels are fitted in parallel."""
    import warnings
    from joblib import Parallel, delayed
    from sklearn.linear_model import LogisticRegression
    L = np.log(np.clip(P, 1e-4, 1 - 1e-4) / (1 - np.clip(P, 1e-4, 1 - 1e-4))); out = np.zeros_like(P)
    order = np.random.RandomState(seed).permutation(len(Y)); folds = np.array_split(order, k)

    def fit_one(Ltr, ytr, Lte):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            return LogisticRegression(C=0.1, max_iter=200).fit(Ltr, ytr).predict_proba(Lte)[:, 1]

    for f in range(k):
        te = folds[f]; tr = np.concatenate([folds[g] for g in range(k) if g != f])
        mu, sd = L[tr].mean(0), L[tr].std(0) + 1e-6; Ltr, Lte = (L[tr] - mu) / sd, (L[te] - mu) / sd
        labs = [lab for lab in range(Y.shape[1]) if Y[tr, lab].sum() > 0]
        for lab, col in zip(labs, Parallel(n_jobs=-1)(delayed(fit_one)(Ltr, Y[tr, lab], Lte) for lab in labs)):
            out[te, lab] = col
    return out


def selection_ablation(model, docs, cfg, device, amp_dtype, modes=('first', 'last', 'uniform'), ks=(1, 2, 4, 8)):
    """Which part of a document matters? Re-score a trained model with different chunk selections (no retraining)."""
    Y = multi_hot(docs); rows = []
    for m in modes:
        for k in ks:
            P = predict_probs(model, docs, cfg, device, amp_dtype, select=m, max_chunks=k)
            rows.append({'select': m, 'chunks': k, 'macro_f1_global_tuned': round(tune_global(Y, P)[1], 4)})
    return rows

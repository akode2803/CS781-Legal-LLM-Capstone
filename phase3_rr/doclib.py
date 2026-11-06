"""Document-level rhetorical-role head over precomputed sentence vectors.

Stage 2 of the two-stage plan: sentence vectors (from a Phase 1 model, or any features)
-> BiLSTM over the WHOLE document -> optional linear-chain CRF -> optional label-shift
auxiliary loss. Cheap enough to train in seconds, so we can use grouped k-fold CV.
"""
from __future__ import annotations

import math
import random

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

N_LABELS = 13


class CRF(nn.Module):
    """Linear-chain CRF (single sequence). Written from scratch for this project."""
    def __init__(self, n):
        super().__init__()
        self.trans = nn.Parameter(torch.zeros(n, n))   # trans[i, j]: i -> j
        self.start = nn.Parameter(torch.zeros(n))
        self.end = nn.Parameter(torch.zeros(n))

    def nll(self, em, y):          # em [T,n], y [T]
        T = em.shape[0]
        gold = self.start[y[0]] + em[0, y[0]]
        for t in range(1, T):
            gold = gold + self.trans[y[t - 1], y[t]] + em[t, y[t]]
        gold = gold + self.end[y[-1]]
        alpha = self.start + em[0]
        for t in range(1, T):
            alpha = torch.logsumexp(alpha[:, None] + self.trans, dim=0) + em[t]
        logz = torch.logsumexp(alpha + self.end, dim=0)
        return (logz - gold) / T

    @torch.no_grad()
    def viterbi(self, em):
        T, n = em.shape
        score = self.start + em[0]
        back = []
        for t in range(1, T):
            s = score[:, None] + self.trans
            best, idx = s.max(0)
            back.append(idx)
            score = best + em[t]
        score = score + self.end
        path = [int(score.argmax())]
        for idx in reversed(back):
            path.append(int(idx[path[-1]]))
        return path[::-1]


class DocModel(nn.Module):
    def __init__(self, in_dim, hidden=128, layers=1, dropout=0.3, in_dropout=0.3, use_pos=True,
                 use_domain=True, bilstm=True, crf=False, shift=False):
        super().__init__()
        self.in_drop = nn.Dropout(in_dropout)
        self.proj = nn.Sequential(nn.Linear(in_dim, hidden), nn.GELU(), nn.Dropout(dropout))
        self.posmlp = nn.Sequential(nn.Linear(3, 32), nn.GELU(), nn.Linear(32, hidden)) if use_pos else None
        self.dom = nn.Embedding(2, hidden) if use_domain else None
        self.rnn = nn.LSTM(hidden, hidden // 2, num_layers=layers, bidirectional=True, batch_first=True,
                           dropout=dropout if layers > 1 else 0.0) if bilstm else None
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(hidden * (2 if bilstm else 1), N_LABELS)
        self.crf = CRF(N_LABELS) if crf else None
        self.shift = nn.Linear(2 * hidden * (2 if bilstm else 1), 1) if shift else None

    def forward(self, x, pos, dom):            # x [T,d]
        z = self.proj(self.in_drop(x))
        if self.posmlp is not None:
            z = z + self.posmlp(pos)
        if self.dom is not None:
            z = z + self.dom(torch.tensor(dom, device=x.device))
        if self.rnn is not None:
            r, _ = self.rnn(z[None])
            z = torch.cat([z, r[0]], -1)
        h = self.drop(z)
        em = self.out(h)
        sh = self.shift(torch.cat([h[:-1], h[1:]], -1)).squeeze(-1) if self.shift is not None and len(h) > 1 else None
        return em, sh


def position_features(n):
    i = np.arange(n)
    return np.stack([i / max(n - 1, 1), np.minimum(i, 60) / 60, np.minimum(n - 1 - i, 60) / 60], 1).astype(np.float32)


def make_doc(x, y, dom):
    return {'x': torch.tensor(x, dtype=torch.float32), 'y': None if y is None else torch.tensor(y),
            'pos': torch.tensor(position_features(len(x))), 'dom': int(dom)}


def class_weight(docs, mode='sqrt'):
    cnt = torch.ones(N_LABELS)
    for d in docs:
        cnt += torch.bincount(d['y'], minlength=N_LABELS)
    w = (cnt.sum() / cnt) ** (0.5 if mode == 'sqrt' else 1.0)
    return w / w.mean() if mode != 'none' else None


def train_model(train_docs, cfg, epochs, device, seed=0, eval_docs=None, log=None, crop=300):
    """Train on documents (random crops of <= crop sentences). If eval_docs is given, returns the
    per-epoch macro-F1 curve on them (for choosing the epoch count, never for the final score)."""
    torch.manual_seed(seed); rng = random.Random(seed)
    in_dim = train_docs[0]['x'].shape[1]
    model = DocModel(in_dim, **cfg['model']).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.get('lr', 2e-3), weight_decay=cfg.get('wd', 0.05))
    w = class_weight(train_docs, cfg.get('cw', 'sqrt'))
    w = w.to(device) if w is not None else None
    curve = []
    for ep in range(epochs):
        model.train()
        order = list(range(len(train_docs))); rng.shuffle(order)
        for i in order:
            d = train_docs[i]; T = len(d['x'])
            a = rng.randrange(0, max(T - crop, 0) + 1); b = min(T, a + crop)
            x, y, pos = d['x'][a:b].to(device), d['y'][a:b].to(device), d['pos'][a:b].to(device)
            em, sh = model(x, pos, d['dom'])
            loss = model.crf.nll(em, y) if model.crf is not None else F.cross_entropy(em, y, weight=w)
            if sh is not None:
                loss = loss + cfg.get('shift_w', 0.5) * F.binary_cross_entropy_with_logits(
                    sh, (y[1:] != y[:-1]).float())
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        if eval_docs is not None:
            pred = predict(model, eval_docs, device)
            curve.append(macro_f1(np.concatenate([d['y'].numpy() for d in eval_docs]), np.concatenate(pred)))
    return model, curve


@torch.no_grad()
def predict_proba(model, docs, device):
    model.eval()
    return [F.softmax(model(d['x'].to(device), d['pos'].to(device), d['dom'])[0], -1).cpu().numpy() for d in docs]


@torch.no_grad()
def emissions(model, docs, device):
    model.eval()
    return [model(d['x'].to(device), d['pos'].to(device), d['dom'])[0].cpu() for d in docs]


@torch.no_grad()
def predict(model, docs, device):
    model.eval()
    out = []
    for d in docs:
        em = model(d['x'].to(device), d['pos'].to(device), d['dom'])[0]
        out.append(np.array(model.crf.viterbi(em) if model.crf is not None else em.argmax(-1).cpu().tolist()))
    return out


def macro_f1(y, p, all13=False):
    from sklearn.metrics import precision_recall_fscore_support
    _, _, f, s = precision_recall_fscore_support(y, p, labels=range(N_LABELS), zero_division=0)
    if all13:
        return float(f.mean())
    present = [i for i in range(N_LABELS) if s[i] > 0 or (np.asarray(p) == i).any()]
    return float(f[present].mean())


def grouped_folds(n_docs, k, seed=0):
    idx = list(range(n_docs)); random.Random(seed).shuffle(idx)
    return [idx[i::k] for i in range(k)]


def cross_validate(docs, cfg, epochs, device, k=5, seeds=(0,), log=print):
    """Grouped k-fold over documents. Returns the mean per-epoch OUT-OF-FOLD macro-F1 curve (pooled
    over folds), so the epoch count is chosen from the CV curve and the score is not picked per fold."""
    curves = []
    for seed in seeds:
        folds = grouped_folds(len(docs), k, seed)
        preds = {e: {} for e in range(epochs)}
        for f in range(k):
            test = [docs[i] for i in folds[f]]
            train = [docs[i] for j in range(k) if j != f for i in folds[j]]
            model = DocModel(train[0]['x'].shape[1], **cfg['model']).to(device)
            # train epoch by epoch to collect out-of-fold predictions per epoch
            torch.manual_seed(seed * 100 + f); rng = random.Random(seed * 100 + f)
            opt = torch.optim.AdamW(model.parameters(), lr=cfg.get('lr', 2e-3), weight_decay=cfg.get('wd', 0.05))
            w = class_weight(train, cfg.get('cw', 'sqrt')); w = w.to(device) if w is not None else None
            for ep in range(epochs):
                model.train()
                order = list(range(len(train))); rng.shuffle(order)
                for i in order:
                    d = train[i]; T = len(d['x']); crop = cfg.get('crop', 300)
                    a = rng.randrange(0, max(T - crop, 0) + 1); b = min(T, a + crop)
                    x, y, pos = d['x'][a:b].to(device), d['y'][a:b].to(device), d['pos'][a:b].to(device)
                    em, sh = model(x, pos, d['dom'])
                    loss = model.crf.nll(em, y) if model.crf is not None else F.cross_entropy(em, y, weight=w)
                    if sh is not None:
                        loss = loss + cfg.get('shift_w', 0.5) * F.binary_cross_entropy_with_logits(
                            sh, (y[1:] != y[:-1]).float())
                    opt.zero_grad(); loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
                if (ep + 1) % cfg.get('eval_every', 1) == 0 or ep == epochs - 1:
                    preds[ep][f] = (np.concatenate([d['y'].numpy() for d in test]),
                                    np.concatenate(predict(model, test, device)))
        curve = []
        for ep in range(epochs):
            if len(preds[ep]) == k:
                y = np.concatenate([preds[ep][f][0] for f in range(k)]); p = np.concatenate([preds[ep][f][1] for f in range(k)])
                curve.append((ep + 1, macro_f1(y, p), macro_f1(y, p, True)))
        curves.append(curve)
    return curves

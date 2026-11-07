"""Phase 7 neural stage: a cross-encoder reranker on Phase 1 backbones over the lexical shortlist.

Pipeline per query: word 4-gram overlap (pcr_lib.Shingles) gives a shortlist; for each (query, candidate) pair we keep the sentences of both documents that share rare
n-grams with the other document, plus the opening of each; the backbone reads [query excerpt] <eos> [candidate excerpt] <eos>, one linear head gives a score; a softmax over one
gold candidate and hard negatives from the shortlist is the training loss. Final ranking = z(lexical) + lam * z(neural), lam chosen on dev.
"""
import dataclasses, json, math, random, re, time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

import pcr_lib as P
import rr_lib as R

PAD_ID, EOS_ID = 1, 2
SENT_CAP = 128      # tokens kept per sentence
GRAM = 4            # n-gram size that picks the excerpt sentences
CITE_W = 1          # sentences kept either side of a <CITATION> placeholder


# ------------------------------------------------------------------ documents and excerpts
def load_sentences(root, split):
    """{id: list of sentences} for queries and candidates of a split (candidates win on shared ids: same text)."""
    qf, cf = P.SPLITS[split]; out = {}
    for f in (qf, cf):
        for k, v in json.load(open(Path(root) / f, encoding='utf-8')).items():
            t = v['text']; out[k] = t if isinstance(t, list) else re.split(r'(?<=[.?!])\s+', t)
    return out


def prepare_docs(sents, tok, max_df=0.05):
    """{id: dict(tok=[int32 arrays per sentence], h=rare n-gram hashes, sid=their sentence, u=unique hashes)}."""
    vocab, docs = {}, {}
    for k, ss in sents.items():
        ss = [s for s in ss if s.strip()] or ['.']
        enc = tok.encode_batch(ss, add_special_tokens=False)
        words, sid = [], []
        for i, s in enumerate(ss):
            w = P.tokens(s); words += [vocab.setdefault(x, len(vocab) + 1) for x in w]; sid += [i] * len(w)
        a = np.array(words, np.uint64); h = np.zeros(max(len(a) - GRAM + 1, 0), np.uint64)
        for j in range(GRAM):
            h = h * np.uint64(1000003) + a[j:len(a) - GRAM + 1 + j]
        hits = [i for i, s in enumerate(ss) if '<CITATION>' in s]
        docs[k] = dict(tok=[np.array(e.ids[:SENT_CAP] or [PAD_ID], np.int32) for e in enc], h=h, sid=np.array(sid[:len(h)], np.int64),
                       cite={j for i in hits for j in range(max(0, i - CITE_W), min(len(ss), i + CITE_W + 1))})
    u, df = np.unique(np.concatenate([np.unique(d['h']) for d in docs.values()]), return_counts=True)
    common = u[df > max(2, max_df * len(docs))]
    for d in docs.values():
        keep = ~np.isin(d['h'], common); d['h'], d['sid'] = d['h'][keep], d['sid'][keep]; d['u'] = np.unique(d['h'])
    return docs


def excerpt(doc, other_u, budget, head=96, prefer=()):
    """Opening of the document, then `prefer` sentences, then the sentences sharing the most rare n-grams with the other document, then the text after the opening; in document order."""
    t = doc['tok']; n = len(t)
    cnt = np.bincount(doc['sid'][np.isin(doc['h'], other_u)], minlength=n) if len(doc['h']) else np.zeros(n, int)
    chosen, used, i = set(), 0, 0
    while i < n and used < head:
        chosen.add(i); used += len(t[i]); i += 1
    by_count = [int(j) for j in np.argsort(-cnt, kind='stable')]
    for j in [j for j in by_count if j in prefer] + by_count + list(range(n)):
        if j in chosen or (cnt[j] == 0 and j < i): continue
        if used + len(t[j]) > budget: continue
        chosen.add(int(j)); used += len(t[j])
        if used >= budget - 8: break
    return np.concatenate([t[j] for j in sorted(chosen)])[:budget]


def pair_ids(docs, q, c, q_budget, c_budget, cite=False):
    """cite=True fills the query excerpt from the sentences around its <CITATION> placeholders first."""
    a, b = docs[q], docs[c]; eos = np.array([EOS_ID], np.int32)
    return np.concatenate([excerpt(a, b['u'], q_budget, prefer=a['cite'] if cite else ()), eos, excerpt(b, a['u'], c_budget), eos])


def shortlist(q, c, top, n=4, b=0.6, max_df=0.05):
    """Lexical stage. Returns q ids, {query: [ids best first]}, {query: {id: score}}: self removed, candidates judged after the query dropped when both header dates parse."""
    qi, ci = list(q), list(c)
    S = P.Shingles([P.tokens(c[k]) for k in ci], n, max_df).scores([P.tokens(q[k]['text']) for k in qi], b)
    qh = np.array([P.judgment_year(q[k]['text']) or np.nan for k in qi], float); ch = np.array([P.judgment_year(c[k]) or np.nan for k in ci], float)
    S[ch[None, :] > qh[:, None]] = -1e9
    ranked, lex, pos = {}, {}, {k: j for j, k in enumerate(ci)}
    for i, k in enumerate(qi):
        if k in pos: S[i, pos[k]] = -2e9
        o = np.argsort(-S[i], kind='stable')[:top]
        ranked[k] = [ci[j] for j in o]; lex[k] = {ci[j]: float(S[i, j]) for j in o}
    return qi, ranked, lex


# ------------------------------------------------------------------ model
class Reranker(nn.Module):
    """Phase 1 backbone + linear score head. `.backbone` / `.prefix` keep rr_lib.inject / set_trainable working unchanged."""

    def __init__(self, backbone, kind, pool, context):
        super().__init__()
        self.backbone, self.kind, self.pool, self.prefix = backbone, kind, pool, None
        if backbone.config.context_length < context:
            backbone.config = dataclasses.replace(backbone.config, context_length=context)
        self.head = nn.Linear(backbone.config.d_model, 1); nn.init.zeros_(self.head.bias); nn.init.normal_(self.head.weight, std=.02)

    def forward(self, ids, lengths):
        bb = self.backbone; x = bb.embedding(ids)
        for blk in bb.blocks:
            x, _ = blk(x, past=None, use_cache=False)
        h = bb.final_norm(x)
        if self.pool == 'last':
            v = h[torch.arange(len(ids), device=ids.device), lengths - 1]
        else:
            m = (torch.arange(ids.shape[1], device=ids.device)[None] < lengths[:, None]).to(h.dtype)[..., None]
            v = (h * m).sum(1) / m.sum(1)
        return self.head(v.float()).squeeze(-1)


def build(checkpoint, kind, method, pool, context, adapter_kw=None, layer_keep=None):
    bb = R.load_backbone({}, checkpoint)
    if layer_keep is not None: R.thin_backbone(bb, layer_keep)
    m = Reranker(bb, kind, pool, context)
    R.inject(m, 'full' if method == 'kd' else method, **((adapter_kw or {}) if method not in ('full', 'kd', 'partial') else {}))
    return m


def randomize(model):
    """Control run: replaces the Phase 1 weights with a fresh initialisation, keeping the architecture."""
    for n, p in model.backbone.named_parameters():
        if p.ndim >= 2: nn.init.normal_(p, std=.02)
        elif n.endswith('bias'): nn.init.zeros_(p)
        else: nn.init.ones_(p)


def pack(seqs, device):
    L = max(len(s) for s in seqs); ids = torch.full((len(seqs), L), PAD_ID, dtype=torch.long)
    for i, s in enumerate(seqs): ids[i, :len(s)] = torch.from_numpy(s.astype(np.int64))
    return ids.to(device), torch.tensor([len(s) for s in seqs], device=device)


@torch.no_grad()
def score_pairs(model, pairs, seqs, device, amp, batch=32):
    """{pair: float}, longest first so batches pad little."""
    model.eval(); order = sorted(pairs, key=lambda p: -len(seqs[p])); out = {}
    for i in range(0, len(order), batch):
        ch = order[i:i + batch]; ids, ln = pack([seqs[p] for p in ch], device)
        with torch.autocast(device_type=device.type, dtype=amp, enabled=device.type == 'cuda'):
            s = model(ids, ln)
        out.update(zip(ch, s.float().cpu().tolist()))
    return out


# ------------------------------------------------------------------ fusion and evaluation
LAMS = (0, .25, .5, 1, 2, 4, 8, None)        # None = neural score alone


def zs(v):
    v = np.asarray(v, float); return (v - v.mean()) / (v.std() + 1e-9)


def fuse(qi, ranked, lex, neural, lam):
    pred = {}
    for q in qi:
        c = ranked[q]; l = zs([max(lex[q][x], -1.0) for x in c]); n = zs([neural[q, x] for x in c])
        f = n if lam is None else l + lam * n
        pred[q] = [c[j] for j in np.argsort(-f, kind='stable')]
    return pred


def dev_table(qi, ranked, lex, neural, gold):
    return {str(l): P.micro_prf(fuse(qi, ranked, lex, neural, l), gold) for l in LAMS}


# ------------------------------------------------------------------ training
def groups(train, C, rng):
    """One group = one gold candidate + C['negs'] shortlist negatives; at most C['pos_per_q'] gold candidates per query in an epoch."""
    out = []
    for q, gold, neg in train:
        if len(neg) < C['negs'] or not gold: continue
        for g in rng.sample(gold, min(len(gold), C['pos_per_q'])):
            out.append([(q, g)] + [(q, x) for x in rng.sample(neg, C['negs'])])
    rng.shuffle(out)
    return out


def train_run(model, data, run_dir, C, device, *, teacher=None, log=print):
    """Best epoch by fused dev Micro-F1 (best.pt holds the trainable weights)."""
    run_dir = Path(run_dir); run_dir.mkdir(parents=True, exist_ok=True)
    trainable, total = R.set_trainable(model, C['method'], C.get('partial_layers', ()))
    model.to(device)
    if teacher is not None:
        teacher.to(device).eval()
        for p in teacher.parameters(): p.requires_grad_(False)
    head = [p for n, p in model.named_parameters() if p.requires_grad and n.startswith('head.')]
    body = [p for n, p in model.named_parameters() if p.requires_grad and not n.startswith('head.')]
    opt = torch.optim.AdamW([g for g in (dict(params=body, lr=C['lr']), dict(params=head, lr=C['lr_head'])) if g['params']], betas=(.9, .95), weight_decay=.01)
    rng = random.Random(C['seed']); seqs = data['seqs']
    n_groups = len(groups(data['train'], C, random.Random(0))); total_steps = max(1, C['epochs'] * n_groups // C['accum']); warm = max(1, int(.05 * total_steps))
    base = [g['lr'] for g in opt.param_groups]; amp = torch.bfloat16 if device.type == 'cuda' and torch.cuda.is_bf16_supported() else torch.float16
    assert device.type != 'cuda' or amp == torch.bfloat16, 'use a GPU with bf16 (L4 / G4 / A100)'
    t0, step, best, hist = time.time(), 0, -1.0, []
    if device.type == 'cuda': torch.cuda.reset_peak_memory_stats()
    stop = False
    for epoch in range(C['epochs']):
        G = groups(data['train'], C, rng); model.train(); run, ok, kls = 0.0, 0, []
        for i, g in enumerate(G):
            ids, ln = pack([seqs[p] for p in g], device)
            with torch.autocast(device_type=device.type, dtype=amp, enabled=device.type == 'cuda'):
                s = model(ids, ln)
                if teacher is not None:
                    with torch.no_grad(): ts = teacher(ids, ln)
            s = s.float()[None]; tgt = torch.zeros(1, dtype=torch.long, device=device); ce = F.softplus(s[:, 1:] - s[:, :1]).mean() if C.get('loss') == 'pairwise' else F.cross_entropy(s, tgt); loss = ce
            if teacher is not None:
                T = C['kd_t']; kl = F.kl_div(F.log_softmax(s / T, -1), F.log_softmax(ts.float()[None] / T, -1), log_target=True, reduction='batchmean') * T * T
                loss = C['kd_alpha'] * ce + (1 - C['kd_alpha']) * kl; kls.append(float(kl))
            (loss / C['accum']).backward(); run += float(ce); ok += int(s.argmax() == 0)
            if (i + 1) % C['accum'] == 0:
                f = (step + 1) / warm if step < warm else .5 * (1 + math.cos(math.pi * (step - warm) / max(1, total_steps - warm)))
                for gr, b in zip(opt.param_groups, base): gr['lr'] = b * max(f, .05)
                nn.utils.clip_grad_norm_(body + head, 1.0); opt.step(); opt.zero_grad(set_to_none=True); step += 1
            if (i + 1) % C['log_every'] == 0:
                log(f'epoch {epoch + 1} group {i + 1}/{len(G)} CE {run / (i + 1):.4f} top1 {ok / (i + 1):.3f}' + (f' KL {np.mean(kls):.4f}' if kls else '') + f' | {(time.time() - t0) / 60:.1f} min')
            if (time.time() - t0) / 60 > C['max_minutes']:
                log('MAX_MINUTES reached: stopping this run after a dev pass'); stop = True; break
        tab = dev_table(data['dev_q'], data['dev_rank'], data['dev_lex'], {p: v for p, v in score_pairs(model, data['dev_pairs'], seqs, device, amp, C['eval_batch']).items()}, data['dev_gold'])
        lam, top = max(tab.items(), key=lambda kv: kv[1]['f1'])
        hist.append(dict(epoch=epoch + 1, train_ce=run / max(1, i + 1), top1=ok / max(1, i + 1), best_lam=lam, dev_f1=top['f1'], neural_only=tab['None']['f1'], minutes=(time.time() - t0) / 60))
        log(f"epoch {epoch + 1}: dev F1 {top['f1']:.4f} at lam {lam} | neural alone {tab['None']['f1']:.4f} | lexical {tab['0']['f1']:.4f}")
        key = tab['None']['f1'] if C.get('select_neural') else top['f1']      # select_neural: the neural score alone is a steadier epoch signal than the blend
        if key > best or C.get('keep_last'):      # keep_last: dev is in the training data, so the last epoch is kept
            best = key; torch.save({'trainable': R.trainable_state(model), 'epoch': epoch + 1, 'lam': lam, 'table': tab, 'C': C}, run_dir / 'best.pt')
        if stop: break
    res = dict(run_id=C['run_id'], kind=C['kind'], method=C['method'], pool=C['pool'], best_dev_f1=best, history=hist, trainable_params=trainable, total_params=total, trainable_pct=100 * trainable / total,
               minutes=(time.time() - t0) / 60, peak_gpu_gb=torch.cuda.max_memory_allocated() / 1e9 if device.type == 'cuda' else None)
    (run_dir / 'train_results.json').write_text(json.dumps(res, indent=1, default=float))
    return res


def load_best(model, run_dir):
    s = torch.load(Path(run_dir) / 'best.pt', map_location='cpu', weights_only=False)
    bad = model.load_state_dict(s['trainable'], strict=False)
    assert not bad.unexpected_keys, bad.unexpected_keys
    return s


def save_scores(path, scores):
    json.dump({f'{q}|{c}': v for (q, c), v in scores.items()}, open(path, 'w'))

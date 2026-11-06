"""Phase 3 rhetorical-role (RR) helpers: data, windowing, sentence-level head,
LoRA / rsLoRA / IA3 / prefix adapters, knowledge distillation, evaluation.

Reuses the Phase 1 CausalDecoder (legal_scaffold) and the Phase 2 LoRMA+ wrapper
(training_methods.LoRMAPlusLinear). No pretrained weights or external APIs.
"""
from __future__ import annotations

import dataclasses
import json
import math
import random
import time
import zipfile
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from legal_scaffold.models import CausalDecoder, ModelConfig
from training_methods import LoRMAPlusLinear, restore_backbone

DATASET_REPO = 'Exploration-Lab/CS-781-Capstone'
DATASET_FILES = tuple(f'Phase_3_RR/{p}' for p in (
    'Train/CL_train.json', 'Train/CL_dev.json', 'Train/IT_train.json',
    'Train/IT_dev.json', 'Train/label_vocab.json',
    'Test/CL_test.json', 'Test/IT_test.json'))
DOMAINS = ('CL', 'IT')
PAD_ID, SEP_ID = 1, 2  # tokenizer: <pad>=1, <eos>=2 (used as sentence separator)


# ----------------------------------------------------------------------------- data
def ensure_files(course_root, token):
    """Download any missing Phase-3 file from the gated HF dataset (latest revision)."""
    from huggingface_hub import hf_hub_download
    root = Path(course_root)
    for name in DATASET_FILES:
        if not (root / name).is_file():
            if not token:
                raise RuntimeError('HF_TOKEN missing from Colab secrets and files not on Drive')
            hf_hub_download(repo_id=DATASET_REPO, repo_type='dataset', filename=name,
                            local_dir=str(root), token=token)
    return root / 'Phase_3_RR'


def load_labels(path):
    raw = json.loads(Path(path).read_text(encoding='utf-8'))
    if isinstance(raw, dict) and 'labels' in raw:
        raw = raw['labels']
    if isinstance(raw, dict):
        items = list(raw.items())
        if all(isinstance(v, int) for _, v in items):      # label -> id
            return [k for k, _ in sorted(items, key=lambda kv: kv[1])]
        if all(str(k).isdigit() for k, _ in items):        # id -> label
            return [v for _, v in sorted(items, key=lambda kv: int(kv[0]))]
        return [k for k, _ in items]
    return list(raw)


def load_docs(path, domain):
    """Accepts a list of docs or a {id: doc} dict. Returns [{id,domain,text,labels|None}]."""
    raw = json.loads(Path(path).read_text(encoding='utf-8'))
    if isinstance(raw, dict):
        raw = [dict(v, id=v.get('id', k)) if isinstance(v, dict) else {'id': k, 'text': v}
               for k, v in raw.items()]
    docs = []
    for d in raw:
        text = d['text']
        if isinstance(text, str):
            raise ValueError(f'{path}: expected a list of sentences for doc {d.get("id")}')
        labels = d.get('labels')
        if labels is not None and len(labels) != len(text):
            raise ValueError(f'doc {d["id"]}: {len(text)} sentences but {len(labels)} labels')
        docs.append({'id': str(d['id']), 'domain': domain, 'text': list(text), 'labels': labels})
    return docs


def load_all(phase_dir, labels):
    phase_dir = Path(phase_dir)
    out = {'train': [], 'dev': [], 'test': []}
    for dom in DOMAINS:
        out['train'] += load_docs(phase_dir / f'Train/{dom}_train.json', dom)
        out['dev'] += load_docs(phase_dir / f'Train/{dom}_dev.json', dom)
        out['test'] += load_docs(phase_dir / f'Test/{dom}_test.json', dom)
    known = set(labels)
    for split in ('train', 'dev'):
        for d in out[split]:
            bad = set(d['labels']) - known
            if bad:
                raise ValueError(f'unknown labels {bad} in {d["id"]}')
    return out


def summarize(splits, labels):
    from collections import Counter
    rep = {}
    for s, docs in splits.items():
        rep[s] = {dom: {'docs': sum(d['domain'] == dom for d in docs),
                        'sentences': sum(len(d['text']) for d in docs if d['domain'] == dom)}
                  for dom in DOMAINS}
    cnt = Counter(l for d in splits['train'] for l in d['labels'])
    rep['train_label_counts'] = {l: cnt.get(l, 0) for l in labels}
    ids = [d['id'] for s in splits.values() if isinstance(s, list) for d in s]
    return rep


# --------------------------------------------------------------------------- windows
def tokenize_docs(docs, tokenizer, sent_cap):
    """Per sentence: token ids (<= sent_cap-1) + separator. Stored in doc['tok']."""
    for d in docs:
        enc = tokenizer.encode_batch(d['text'])
        d['tok'] = [(e.ids[:sent_cap - 1] or [PAD_ID]) + [SEP_ID] for e in enc]
    return docs


def make_windows(doc, *, core_max, ctx_left, ctx_right, budget):
    """Cut a document into windows: [left ctx | core | right ctx] sentences.

    Loss/prediction is taken on core sentences only; every sentence is covered
    by exactly one core. core_max=1, ctx=0 gives the sentence-only baseline.
    """
    tok, n = doc['tok'], len(doc['tok'])
    wins, a = [], 0
    while a < n:
        left = list(range(max(0, a - ctx_left), a))
        used = sum(len(tok[i]) for i in left)
        core = [a]
        used += len(tok[a])
        while len(core) < core_max and core[-1] + 1 < n:
            nxt = core[-1] + 1
            right_need = sum(len(tok[j]) for j in range(nxt + 1, min(n, nxt + 1 + ctx_right)))
            if used + len(tok[nxt]) + right_need > budget:
                break
            core.append(nxt)
            used += len(tok[nxt])
        b = core[-1] + 1
        right = []
        for j in range(b, min(n, b + ctx_right)):
            if used + len(tok[j]) > budget:
                break
            right.append(j)
            used += len(tok[j])
        idx = left + core + right
        wins.append({'doc': doc, 'sent_idx': idx, 'core_lo': len(left),
                     'core_hi': len(left) + len(core)})
        a = b
    return wins


def build_windows(docs, **kw):
    return [w for d in docs for w in make_windows(d, **kw)]


def collate(batch, label_to_id, device):
    seqs, spans_per = [], []
    for w in batch:
        ids, spans, pos = [], [], 0
        for i in w['sent_idx']:
            t = w['doc']['tok'][i]
            ids += t
            spans.append((pos, pos + len(t)))
            pos += len(t)
        seqs.append(ids)
        spans_per.append(spans)
    B, T = len(batch), max(len(s) for s in seqs)
    S = max(len(s) for s in spans_per)
    input_ids = torch.full((B, T), PAD_ID, dtype=torch.long)
    pool = torch.zeros(B, S, T)
    last = torch.zeros(B, S, dtype=torch.long)
    lengths = torch.zeros(B, dtype=torch.long)
    labels = torch.full((B, S), -100, dtype=torch.long)
    domain = torch.zeros(B, dtype=torch.long)
    pos = torch.zeros(B, S, 3)
    for b, (w, ids, spans) in enumerate(zip(batch, seqs, spans_per)):
        input_ids[b, :len(ids)] = torch.tensor(ids)
        lengths[b] = len(spans)
        domain[b] = DOMAINS.index(w['doc']['domain'])
        n_doc = len(w['doc']['tok'])
        for s, (lo, hi) in enumerate(spans):
            pool[b, s, lo:hi] = 1.0 / (hi - lo)
            last[b, s] = hi - 1
            gi = w['sent_idx'][s]  # position of this sentence in its document
            pos[b, s] = torch.tensor([gi / max(n_doc - 1, 1), min(gi, 60) / 60, min(n_doc - 1 - gi, 60) / 60])
        gold = w['doc']['labels']
        if gold is not None:
            for s in range(w['core_lo'], w['core_hi']):
                labels[b, s] = label_to_id[gold[w['sent_idx'][s]]]
    return {'input_ids': input_ids.to(device), 'pool': pool.to(device), 'last': last.to(device),
            'lengths': lengths, 'labels': labels.to(device), 'domain': domain.to(device),
            'pos': pos.to(device)}


# ---------------------------------------------------------------------------- model
def backbone_hidden(backbone, input_ids, prefix=None):
    """Final-norm hidden states (no LM head). prefix: optional per-layer cache tuples."""
    x = backbone.embedding(input_ids)
    for i, block in enumerate(backbone.blocks):
        x, _ = block(x, past=None if prefix is None else prefix[i], use_cache=False,
                     mla_mode='materialized')
    return backbone.final_norm(x)


class PrefixKV(nn.Module):
    """Prefix tuning: trainable per-layer key/value (or MLA latent) prefix injected as `past`."""
    def __init__(self, config: ModelConfig, length):
        super().__init__()
        self.length, self.config = length, config
        self.mla = config.attention == 'mla'
        hd = config.d_model // config.heads
        if self.mla:
            self.latent = nn.ParameterList(
                nn.Parameter(torch.randn(length, config.mla_latent) * .02) for _ in range(config.layers))
            self.pos = nn.ParameterList(
                nn.Parameter(torch.randn(length, config.mla_rope_dim) * .02) for _ in range(config.layers))
        else:
            self.k = nn.ParameterList(
                nn.Parameter(torch.randn(config.kv_heads, length, hd) * .02) for _ in range(config.layers))
            self.v = nn.ParameterList(
                nn.Parameter(torch.randn(config.kv_heads, length, hd) * .02) for _ in range(config.layers))

    def forward(self, batch):
        out = []
        for i in range(self.config.layers):
            if self.mla:
                out.append((self.latent[i].unsqueeze(0).expand(batch, -1, -1),
                            self.pos[i].view(1, 1, self.length, -1).expand(batch, -1, -1, -1)))
            else:
                out.append((self.k[i].unsqueeze(0).expand(batch, -1, -1, -1),
                            self.v[i].unsqueeze(0).expand(batch, -1, -1, -1)))
        return out


class RRModel(nn.Module):
    """Phase-1 decoder -> per-sentence [mean||last] pooling -> MLP -> optional BiGRU -> labels."""
    def __init__(self, backbone, n_labels, *, hidden=256, use_bigru=True, domain_emb=True,
                 dropout=0.1, prefix_len=0, use_pos=False):
        super().__init__()
        d = backbone.config.d_model
        self.backbone = backbone
        self.prefix = PrefixKV(backbone.config, prefix_len) if prefix_len else None
        self.proj = nn.Sequential(nn.Linear(2 * d, hidden), nn.GELU(), nn.Dropout(dropout))
        self.domain = nn.Embedding(len(DOMAINS), hidden) if domain_emb else None
        self.posmlp = nn.Sequential(nn.Linear(3, 32), nn.GELU(), nn.Linear(32, hidden)) if use_pos else None
        self.rnn = nn.GRU(hidden, hidden // 2, batch_first=True, bidirectional=True) if use_bigru else None
        self.drop = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden * (2 if use_bigru else 1), n_labels)

    def sentence_vectors(self, batch):
        prefix = self.prefix(batch['input_ids'].shape[0]) if self.prefix is not None else None
        h = backbone_hidden(self.backbone, batch['input_ids'], prefix)
        mean = torch.bmm(batch['pool'].to(h.dtype), h)
        last = torch.gather(h, 1, batch['last'].unsqueeze(-1).expand(-1, -1, h.shape[-1]))
        return torch.cat([mean, last], dim=-1)

    def head(self, vec, batch):
        z = self.proj(vec.float())
        if self.domain is not None:
            z = z + self.domain(batch['domain']).unsqueeze(1)
        if self.posmlp is not None:  # where the sentence sits in the whole document
            z = z + self.posmlp(batch['pos'])
        if self.rnn is not None:
            packed = nn.utils.rnn.pack_padded_sequence(z, batch['lengths'], batch_first=True,
                                                       enforce_sorted=False)
            r, _ = self.rnn(packed)
            r, _ = nn.utils.rnn.pad_packed_sequence(r, batch_first=True, total_length=z.shape[1])
            z = torch.cat([z, r.to(z.dtype)], dim=-1)
        return self.classifier(self.drop(z))

    def forward(self, batch, return_vec=False):
        vec = self.sentence_vectors(batch)
        logits = self.head(vec, batch)
        return (logits, vec) if return_vec else logits


# -------------------------------------------------------------------- adapter layers
class LoRALinear(nn.Module):
    """W x + (alpha/r) B A x. rsLoRA uses alpha/sqrt(r) (Kalajdzievski 2023)."""
    def __init__(self, linear, rank=8, alpha=16, rslora=False):
        super().__init__()
        self.base = linear
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.a = nn.Parameter(torch.empty(rank, linear.in_features))
        self.b = nn.Parameter(torch.zeros(linear.out_features, rank))
        nn.init.kaiming_uniform_(self.a, a=math.sqrt(5))
        self.scale = alpha / math.sqrt(rank) if rslora else alpha / rank

    def forward(self, x):
        return self.base(x) + self.scale * F.linear(F.linear(x, self.a.to(x.dtype)), self.b.to(x.dtype))


class IA3Linear(nn.Module):
    """IA3 (Liu et al. 2022): learned rescaling vector on a projection's output or input."""
    def __init__(self, linear, where='out'):
        super().__init__()
        self.base, self.where = linear, where
        for p in self.base.parameters():
            p.requires_grad_(False)
        width = linear.out_features if where == 'out' else linear.in_features
        self.scale = nn.Parameter(torch.ones(width))

    def forward(self, x):
        if self.where == 'in':
            return self.base(x * self.scale.to(x.dtype))
        return self.base(x) * self.scale.to(x.dtype)


ATTN_TARGETS = {  # architecture-aware name of the "query"/"value"/"key" projections
    'mha': {'query': 'query', 'key': 'key', 'value': 'value', 'output': 'output'},
    'gqa': {'query': 'query', 'key': 'key', 'value': 'value', 'output': 'output'},
    'hybrid': {'query': 'query', 'key': 'key', 'value': 'value', 'output': 'output'},     # pretrain/hybrid.py: HybridMixer has the same projections
    'mla': {'query': 'query_up', 'key': 'key_up', 'value': 'value_up', 'output': 'output'},
}


def _wrap(parent, name, factory, log, path):
    setattr(parent, name, factory(getattr(parent, name)))
    log.append(path)


def inject(model: RRModel, method, *, rank=8, alpha=16, targets=('query', 'value')):
    """Freeze the Phase-1 backbone and add method-specific trainable parameters."""
    if method in ('full', 'partial', 'kd'):
        return []  # no new modules; set_trainable applies the freezing mask
    cfg = model.backbone.config
    for p in model.backbone.parameters():
        p.requires_grad_(False)
    changed = []
    names = ATTN_TARGETS[cfg.attention]
    for i, blk in enumerate(model.backbone.blocks):
        if method in ('lora', 'rslora', 'lorma'):
            for t in targets:
                n = names[t]
                if method == 'lorma':
                    f = lambda lin: LoRMAPlusLinear(lin, rank=rank, alpha=alpha)
                else:
                    f = lambda lin: LoRALinear(lin, rank, alpha, rslora=method == 'rslora')
                _wrap(blk.attention, n, f, changed, f'blocks.{i}.attention.{n}')
        elif method == 'ia3':
            _wrap(blk.attention, names['key'], lambda l: IA3Linear(l, 'out'), changed, f'blocks.{i}.key')
            _wrap(blk.attention, names['value'], lambda l: IA3Linear(l, 'out'), changed, f'blocks.{i}.value')
            _wrap(blk.ffn, 'down', lambda l: IA3Linear(l, 'in'), changed, f'blocks.{i}.ffn.down')
        elif method == 'prefix':
            pass  # prefix params already created in RRModel(prefix_len>0)
        elif method == 'head_only':
            pass
        else:
            raise ValueError(method)
    if method == 'prefix' and model.prefix is None:
        raise ValueError('prefix method needs RRModel(prefix_len>0)')
    return changed


def set_trainable(model, method, layers=()):
    """Return (trainable, total) after the method-specific freezing policy."""
    if method in ('full', 'kd'):
        for p in model.parameters():
            p.requires_grad_(True)
    elif method == 'partial':
        for n, p in model.named_parameters():
            on = not n.startswith('backbone.') or (
                n.startswith('backbone.blocks.') and int(n.split('.')[2]) in set(layers))
            p.requires_grad_(on)
    else:  # adapters: head + adapter/prefix params only
        for n, p in model.named_parameters():
            on = not n.startswith('backbone.') or n.endswith(('.a', '.b', '.scale'))
            if n.endswith('.base.scale'):
                on = False
            p.requires_grad_(on)
    # tied embedding/lm_head: lm_head is unused here, keep tied copy consistent
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def load_backbone(cfg_dict, checkpoint):
    """Phase-1 snapshot -> backbone. A file written by pretrain/train.py carries its own config and objective, which win over cfg_dict:
    a hybrid config -> hybrid.restore_hybrid; objective 'mlm' -> encoder.to_encoder (bidirectional, pad positions hidden).
    Anything else takes the original restore_backbone path. hybrid.py / encoder.py must sit next to this file for the new models."""
    try:
        saved = torch.load(checkpoint, map_location='cpu', weights_only=False, mmap=True)
        cfg, objective = saved['config'], saved.get('objective')
        del saved
    except Exception:       # a Phase 1 snapshot (no 'config') or no mmap support
        return restore_backbone(ModelConfig(**cfg_dict).validate(), checkpoint)
    if objective is None:   # exported by an earlier train.py: the objective is only recorded in run.json beside the file
        try:
            import json; objective = json.load(open(Path(checkpoint).parent / 'run.json'))['args']['objective']
        except Exception:
            objective = 'clm'
    if 'ssm_heads' in cfg:
        from hybrid import restore_hybrid
        backbone = restore_hybrid(cfg, checkpoint)
    else:
        backbone = restore_backbone(ModelConfig(**cfg).validate(), checkpoint)
    if objective == 'mlm':
        from encoder import to_encoder
        to_encoder(backbone, pad_id=PAD_ID)
    return backbone


def thin_backbone(backbone, layer_keep):
    """Keep only the listed blocks (e.g. a 6-layer student cut from the 12-layer model) and update the config to match."""
    backbone.blocks = nn.ModuleList([backbone.blocks[i] for i in layer_keep])
    backbone.config = dataclasses.replace(backbone.config, layers=len(layer_keep)).validate()


def build_model(cfg_dict, checkpoint, n_labels, method, *, head_kw, adapter_kw=None, prefix_len=0,
                layer_keep=None):
    """Phase-1 snapshot -> (optionally thinned) backbone -> RRModel -> adapters."""
    backbone = load_backbone(cfg_dict, checkpoint)
    if layer_keep is not None:  # 6-layer student cut from the 12-layer Phase-1 model
        thin_backbone(backbone, layer_keep)
    model = RRModel(backbone, n_labels, prefix_len=prefix_len, **head_kw)
    inject(model, method, **(adapter_kw or {}))
    return model


# ----------------------------------------------------------------- losses / metrics
def class_weights(train_docs, labels, mode):
    if mode == 'none':
        return None
    cnt = torch.ones(len(labels))
    idx = {l: i for i, l in enumerate(labels)}
    for d in train_docs:
        for l in d['labels']:
            cnt[idx[l]] += 1
    w = (cnt.sum() / cnt) ** (0.5 if mode == 'sqrt' else 1.0)
    return w / w.mean()


def kd_loss(student_logits, teacher_logits, gold, *, temperature=2.0, alpha=0.5, weight=None):
    mask = gold != -100
    s, t, g = student_logits[mask].float(), teacher_logits[mask].float(), gold[mask]
    hard = F.cross_entropy(s, g, weight=weight)
    soft = F.kl_div(F.log_softmax(s / temperature, -1), F.softmax(t / temperature, -1),
                    reduction='batchmean') * temperature ** 2
    return (1 - alpha) * hard + alpha * soft, hard.detach(), soft.detach()


def macro_scores(gold, pred, labels):
    from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
    ids = list(range(len(labels)))
    p, r, f, sup = precision_recall_fscore_support(gold, pred, labels=ids, zero_division=0)
    present = [i for i in ids if sup[i] > 0 or (pred.count(i) if isinstance(pred, list) else 0) > 0]
    return {
        'macro_f1_all13': float(f.mean()),
        'macro_f1_present': float(f[present].mean()) if present else 0.0,
        'macro_precision': float(p[present].mean()) if present else 0.0,
        'macro_recall': float(r[present].mean()) if present else 0.0,
        'per_class': {labels[i]: {'p': float(p[i]), 'r': float(r[i]), 'f1': float(f[i]), 'support': int(sup[i])}
                      for i in ids},
        'confusion': confusion_matrix(gold, pred, labels=ids).tolist(),
    }


@torch.no_grad()
def predict_docs(model, docs, wcfg, label_to_id, device, batch_size, amp_dtype, loss_acc=None):
    """Predict every sentence of every doc (core sentences of each window)."""
    model.eval()
    wins = build_windows(docs, **wcfg)
    wins.sort(key=lambda w: sum(len(w['doc']['tok'][i]) for i in w['sent_idx']))
    out = {d['id']: [None] * len(d['text']) for d in docs}
    logit_store = {d['id']: [None] * len(d['text']) for d in docs}
    for i in range(0, len(wins), batch_size):
        chunk = wins[i:i + batch_size]
        batch = collate(chunk, label_to_id, device)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=device.type == 'cuda'):
            logits = model(batch)
        pred = logits.float().argmax(-1).cpu()
        if loss_acc is not None and (batch['labels'] != -100).any():
            m = batch['labels'] != -100
            loss_acc[0] += float(F.cross_entropy(logits.float()[m], batch['labels'][m], reduction='sum'))
            loss_acc[1] += int(m.sum())
        for b, w in enumerate(chunk):
            for s in range(w['core_lo'], w['core_hi']):
                gi = w['sent_idx'][s]
                out[w['doc']['id']][gi] = int(pred[b, s])
                logit_store[w['doc']['id']][gi] = logits[b, s].float().cpu()
    for k, v in out.items():
        if any(x is None for x in v):
            raise RuntimeError(f'doc {k}: some sentences were not predicted')
    return out, logit_store


def evaluate(model, docs, wcfg, labels, label_to_id, device, batch_size, amp_dtype):
    acc = [0.0, 0]
    preds, _ = predict_docs(model, docs, wcfg, label_to_id, device, batch_size, amp_dtype, loss_acc=acc)
    gold, pr, dom = [], [], []
    for d in docs:
        gold += [label_to_id[l] for l in d['labels']]
        pr += preds[d['id']]
        dom += [d['domain']] * len(d['labels'])
    res = macro_scores(gold, pr, labels)
    res['loss'] = acc[0] / max(acc[1], 1)  # unweighted CE per sentence
    res['by_domain'] = {}
    for dm in DOMAINS:
        g = [x for x, dd in zip(gold, dom) if dd == dm]
        p = [x for x, dd in zip(pr, dom) if dd == dm]
        if g:
            r = macro_scores(g, p, labels)
            res['by_domain'][dm] = {k: r[k] for k in ('macro_f1_all13', 'macro_f1_present',
                                                      'macro_precision', 'macro_recall')}
    return res


def write_submission(preds, docs, labels, out_dir):
    """prediction.json (doc id -> ordered labels) + submission.zip with it at the root."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    final = {}
    for d in docs:
        if d['id'] in final:
            raise ValueError(f'duplicate test id across CL/IT: {d["id"]}')
        p = preds[d['id']]
        assert len(p) == len(d['text']), d['id']
        final[d['id']] = [labels[i] for i in p]
    path = out_dir / 'prediction.json'
    path.write_text(json.dumps(final, ensure_ascii=False), encoding='utf-8')
    with zipfile.ZipFile(out_dir / 'submission.zip', 'w', zipfile.ZIP_DEFLATED) as z:
        z.write(path, 'prediction.json')
    return path


# ------------------------------------------------------------------------- training
def _batches(wins, size, rng):
    order = list(range(len(wins)))
    rng.shuffle(order)
    for i in range(0, len(order), size):
        yield [wins[j] for j in order[i:i + size]]


def param_groups(model, method, lr):
    head, adapter, backbone = [], [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if not n.startswith('backbone.') and not n.startswith('prefix.'):
            head.append(p)
        elif method in ('full', 'kd', 'partial'):
            backbone.append(p)
        else:
            adapter.append(p)
    groups = []
    if head: groups.append({'params': head, 'lr': lr['head'], 'weight_decay': 0.01})
    if adapter: groups.append({'params': adapter, 'lr': lr['adapter'], 'weight_decay': 0.0})
    if backbone: groups.append({'params': backbone, 'lr': lr['backbone'], 'weight_decay': 0.01})
    return groups


def trainable_state(model):
    keep = {n for n, p in model.named_parameters() if p.requires_grad}
    return {k: v.detach().cpu() for k, v in model.state_dict().items() if k in keep}


def train_run(model, data, labels, run_dir, C, device, *, teacher=None, log=print):
    """Train with per-epoch dev eval, best-F1 checkpointing and resume. Returns results dict."""
    run_dir = Path(run_dir); run_dir.mkdir(parents=True, exist_ok=True)
    label_to_id = {l: i for i, l in enumerate(labels)}
    wcfg = dict(core_max=C['core_max'], ctx_left=C['ctx_left'], ctx_right=C['ctx_right'],
                budget=C['budget'])
    train_w = build_windows(data['train'], **wcfg)
    trainable, total = set_trainable(model, C['method'], C.get('partial_layers', ()))
    model.to(device)
    groups = param_groups(model, C['method'], C['lr'])
    opt = torch.optim.AdamW(groups, betas=(0.9, 0.98))
    steps_per_epoch = math.ceil(len(train_w) / C['batch_size'])
    total_steps = steps_per_epoch * C['epochs']
    warm = max(1, int(.06 * total_steps))
    base_lrs = [g['lr'] for g in opt.param_groups]

    def set_lr(step):
        f = (step + 1) / warm if step < warm else .5 * (1 + math.cos(math.pi * (step - warm) / max(1, total_steps - warm)))
        for g, b in zip(opt.param_groups, base_lrs):
            g['lr'] = b * max(f, 0.02)

    amp_dtype = torch.bfloat16 if (device.type == 'cuda' and torch.cuda.is_bf16_supported()) else torch.float16
    scaler = torch.amp.GradScaler('cuda', enabled=device.type == 'cuda' and amp_dtype == torch.float16)
    w = class_weights(data['train'], labels, C['class_weight'])
    w = w.to(device) if w is not None else None
    state_path, best_path = run_dir / 'last.pt', run_dir / 'best.pt'
    hist, best, start_epoch, step = [], -1.0, 0, 0
    if state_path.is_file():
        s = torch.load(state_path, map_location='cpu', weights_only=False)
        model.load_state_dict(s['model'], strict=False)
        opt.load_state_dict(s['opt'])
        hist, best, start_epoch, step = s['hist'], s['best'], s['epoch'] + 1, s['step']
        log(f'resumed after epoch {start_epoch}, best dev macro-F1 {best:.4f}')
    rng = random.Random(C['seed'] + start_epoch)
    torch.manual_seed(C['seed'] + start_epoch)
    t0 = time.time()
    if device.type == 'cuda': torch.cuda.reset_peak_memory_stats()
    for epoch in range(start_epoch, C['epochs']):
        model.train()
        if teacher is not None: teacher.eval()
        run_loss, n = 0.0, 0
        for chunk in _batches(train_w, C['batch_size'], rng):
            batch = collate(chunk, label_to_id, device)
            set_lr(step)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=device.type == 'cuda'):
                logits = model(batch)
                if teacher is not None:
                    with torch.no_grad():
                        t_logits = teacher(batch)
                    loss, hard, soft = kd_loss(logits, t_logits, batch['labels'],
                                               temperature=C['kd_temperature'], alpha=C['kd_alpha'], weight=w)
                else:
                    loss = F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]),
                                           batch['labels'].reshape(-1), weight=w, ignore_index=-100)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_([p for g in opt.param_groups for p in g['params']], 1.0)
            scaler.step(opt); scaler.update()
            run_loss += float(loss.detach()); n += 1; step += 1
        res = evaluate(model, data['dev'], wcfg, labels, label_to_id, device, C['eval_batch'], amp_dtype)
        f1 = res['macro_f1_present']
        sub = data['train'][:min(40, len(data['train']))]  # fixed train subset, eval mode: overfitting check
        tr_res = evaluate(model, sub, wcfg, labels, label_to_id, device, C['eval_batch'], amp_dtype)
        hist.append({'train_subset_macro_f1': tr_res['macro_f1_present'], 'train_subset_loss': tr_res['loss'],
                     'dev_loss': res['loss'], 'epoch': epoch + 1, 'train_loss': run_loss / max(n, 1), 'dev_macro_f1': f1,
                     'dev_macro_f1_all13': res['macro_f1_all13'], 'by_domain': res['by_domain'],
                     'minutes': (time.time() - t0) / 60})
        log(f"epoch {epoch+1}/{C['epochs']} loss {run_loss/max(n,1):.4f} | dev loss {res['loss']:.3f} F1 {f1:.4f} | train-subset loss {tr_res['loss']:.3f} F1 {tr_res['macro_f1_present']:.4f} "
            f"{ {k: round(v['macro_f1_present'],3) for k,v in res['by_domain'].items()} }")
        if f1 > best:
            best = f1
            torch.save({'trainable': trainable_state(model), 'epoch': epoch + 1, 'dev': res, 'C': C}, best_path)
        torch.save({'model': trainable_state(model), 'opt': opt.state_dict(), 'hist': hist,
                    'best': best, 'epoch': epoch, 'step': step}, state_path)
    saved = torch.load(best_path, map_location='cpu', weights_only=False)
    model.load_state_dict(saved['trainable'], strict=False)
    final = evaluate(model, data['dev'], wcfg, labels, label_to_id, device, C['eval_batch'], amp_dtype)
    results = {'method': C['method'], 'run_id': C['run_id'], 'preset': C['preset'],
               'trainable_params': trainable, 'total_params': total,
               'trainable_pct': 100 * trainable / total, 'best_epoch': saved['epoch'],
               'train_minutes': hist[-1]['minutes'] if hist else None,
               'peak_gpu_gb': torch.cuda.max_memory_allocated() / 1e9 if device.type == 'cuda' else None,
               'context': {k: C[k] for k in ('core_max', 'ctx_left', 'ctx_right', 'use_bigru', 'class_weight')},
               'dev': final, 'history': hist}
    (run_dir / 'results.json').write_text(json.dumps(results, indent=1), encoding='utf-8')
    return results, amp_dtype, wcfg


def top_confusions(res, labels, k=10):
    cm = res['dev']['confusion']
    pairs = [(cm[i][j], labels[i], labels[j]) for i in range(len(cm)) for j in range(len(cm)) if i != j and cm[i][j]]
    return sorted(pairs, reverse=True)[:k]

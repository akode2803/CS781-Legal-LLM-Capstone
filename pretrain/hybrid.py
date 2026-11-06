"""Hymba-style hybrid decoder: in every block, attention heads and state-space (Mamba-2 / SSD) heads read the same input IN PARALLEL and their outputs are fused.

    block(x) = x + Output( [ beta_a * RMSNorm(attention(x)) | beta_s * GatedRMSNorm(ssm(x)) ] )   then the usual SwiGLU FFN

* attention branch: GQA with RoPE through torch's scaled_dot_product_attention in causal mode, so the flash kernel is used (all heads have the same dimension).
* state-space branch: Mamba-2 "SSD" in plain PyTorch (chunked matrix form: matmuls, no custom CUDA kernel, no extra packages, works with torch.compile). Constant-size
  state at inference; the chunked form is verified against the step-by-step recurrence in tests/test_hybrid.py.
* beta_a, beta_s: learned per-channel scales on the two branches (as in Hymba), so each layer can lean on attention or on the SSM.
What is NOT taken from Hymba: meta tokens, sliding-window attention, cross-layer KV sharing (all three would break the plain-sequence interface the Phase 3-5 code uses,
or the flash kernel). The block keeps the names the downstream code relies on: embedding, blocks[i].attention.{query,key,value,output}, blocks[i].ffn, final_norm, lm_head.
"""
from __future__ import annotations

import math
import sys
from dataclasses import dataclass, asdict

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class HybridConfig:
    vocab_size: int
    d_model: int
    layers: int
    ffn_hidden: int
    context_length: int
    attn_heads: int = 8
    attn_kv_heads: int = 2
    head_dim: int = 64
    ssm_heads: int = 8
    ssm_headdim: int = 64
    ssm_state: int = 64
    ssm_conv: int = 4
    ssm_chunk: int = 128
    norm_epsilon: float = 1e-6
    rope_base: float = 10000
    init_std: float = .02
    tie_embeddings: bool = True
    attention: str = 'hybrid'
    norm_placement: str = 'pre'

    @property
    def heads(self): return self.attn_heads            # names the downstream code reads

    @property
    def kv_heads(self): return self.attn_kv_heads

    def validate(self):
        if self.attn_heads % self.attn_kv_heads or self.head_dim % 2:
            raise ValueError('attn_kv_heads must divide attn_heads and head_dim must be even (RoPE)')
        return self


def _scaffold(scaffold=None):
    if scaffold and scaffold not in sys.path:
        sys.path.insert(0, scaffold)
    from legal_scaffold.models import RMSNorm, SwiGLU, rotary
    return RMSNorm, SwiGLU, rotary


def segsum(x):
    """x[..., T] -> [..., T, T] with out[i, j] = sum(x[j+1..i]) for j <= i, -inf above the diagonal."""
    T = x.size(-1); c = torch.cumsum(x, -1)
    s = c[..., :, None] - c[..., None, :]
    return s.masked_fill(~torch.tril(torch.ones(T, T, dtype=torch.bool, device=x.device)), float('-inf'))


def ssd(x, dt, A, B, C, chunk):
    """Chunked state-space duality scan (Mamba-2), single B/C group.
    x [b,t,h,p], dt [b,t,h] (>0), A [h] (<0), B, C [b,t,n]  ->  y [b,t,h,p] with  s_t = exp(A dt_t) s_{t-1} + dt_t x_t B_t^T,  y_t = s_t C_t."""
    b, t, h, p = x.shape; n = B.shape[-1]
    pad = (-t) % chunk
    if pad:                                                   # zeros at the end cannot influence earlier positions (causal)
        x = F.pad(x, (0, 0, 0, 0, 0, pad)); dt = F.pad(dt, (0, 0, 0, pad)); B = F.pad(B, (0, 0, 0, pad)); C = F.pad(C, (0, 0, 0, pad))
    T = t + pad; c = T // chunk
    f32 = torch.promote_types(dt.dtype, torch.float32)
    a = (A.to(f32) * dt.to(f32)).view(b, c, chunk, h).permute(0, 3, 1, 2)          # [b,h,c,l], decay exponents kept in (at least) float32
    xd = (x * dt.to(x.dtype)[..., None]).view(b, c, chunk, h, p)
    Bc, Cc = B.reshape(b, c, chunk, n), C.reshape(b, c, chunk, n)
    acs = torch.cumsum(a, -1)
    L = torch.exp(segsum(a))                                                       # [b,h,c,l,l]
    M = torch.einsum('bcln,bcsn->bcls', Cc, Bc)[:, None] * L.to(Cc.dtype)           # decayed C.B scores inside a chunk
    y_diag = torch.einsum('bhcls,bcshp->bclhp', M, xd)
    states = torch.einsum('bcln,bhcl,bclhp->bchpn', Bc, torch.exp(acs[..., -1:] - acs).to(xd.dtype), xd)       # state each chunk ends with, from its own inputs
    chunk_decay = torch.exp(segsum(F.pad(acs[..., -1], (1, 0))))                    # [b,h,c+1,c+1]
    states = torch.cat([torch.zeros_like(states[:, :1]), states], 1)
    states = torch.einsum('bhzc,bchpn->bzhpn', chunk_decay.to(states.dtype), states)[:, :-1]   # state entering each chunk
    y_off = torch.einsum('bcln,bchpn,bhcl->bclhp', Cc, states, torch.exp(acs).to(xd.dtype))
    return (y_diag + y_off).reshape(b, T, h, p)[:, :t]


def ssd_reference(x, dt, A, B, C):
    """Step-by-step recurrence (slow); the definition the chunked scan must reproduce."""
    b, t, h, p = x.shape; n = B.shape[-1]
    s = x.new_zeros(b, h, p, n); ys = []
    for i in range(t):
        s = torch.exp(A * dt[:, i])[..., None, None] * s + (dt[:, i, :, None] * x[:, i])[..., None] * B[:, i, None, None, :]
        ys.append((s * C[:, i, None, None, :]).sum(-1))
    return torch.stack(ys, 1)


class SSM(nn.Module):
    def __init__(self, c, RMSNorm):
        super().__init__()
        self.c, h, p, n = c, c.ssm_heads, c.ssm_headdim, c.ssm_state
        self.d_in = h * p
        self.in_proj = nn.Linear(c.d_model, 2 * self.d_in + 2 * n + h, bias=False)       # z | x | B | C | dt
        conv_dim = self.d_in + 2 * n
        self.conv = nn.Conv1d(conv_dim, conv_dim, c.ssm_conv, groups=conv_dim, bias=True)
        dt = torch.exp(torch.rand(h) * (math.log(.1) - math.log(.001)) + math.log(.001))
        self.dt_bias = nn.Parameter(dt + torch.log(-torch.expm1(-dt)))                   # softplus(dt_bias) starts in [1e-3, 1e-1]
        self.A_log = nn.Parameter(torch.log(torch.empty(h).uniform_(1, 16)))
        self.D = nn.Parameter(torch.ones(h))
        self.norm = RMSNorm(self.d_in, c.norm_epsilon)

    def forward(self, x):
        b, t, _ = x.shape; c = self.c; h, p, n = c.ssm_heads, c.ssm_headdim, c.ssm_state
        z, xbc, dt = self.in_proj(x).split([self.d_in, self.d_in + 2 * n, h], dim=-1)
        xbc = F.silu(self.conv(F.pad(xbc.transpose(1, 2), (c.ssm_conv - 1, 0))).transpose(1, 2))      # causal depthwise convolution
        xs, B, C = xbc.split([self.d_in, n, n], dim=-1)
        dt = F.softplus(dt.float() + self.dt_bias)
        xs = xs.reshape(b, t, h, p)
        y = ssd(xs, dt, -torch.exp(self.A_log.float()), B, C, c.ssm_chunk) + xs * self.D.to(xs.dtype)[None, None, :, None]
        return self.norm(y.reshape(b, t, self.d_in) * F.silu(z))


class HybridMixer(nn.Module):
    """Takes the place of `block.attention`: parallel attention and SSM heads, fused by one output projection."""

    def __init__(self, c, RMSNorm, rotary):
        super().__init__()
        self.c, self.rotary = c, rotary
        ha, hk, hd = c.attn_heads, c.attn_kv_heads, c.head_dim
        self.query = nn.Linear(c.d_model, ha * hd, bias=False)
        self.key = nn.Linear(c.d_model, hk * hd, bias=False)
        self.value = nn.Linear(c.d_model, hk * hd, bias=False)
        self.ssm = SSM(c, RMSNorm)
        self.attn_norm = RMSNorm(ha * hd, c.norm_epsilon)
        self.beta_attn = nn.Parameter(torch.ones(ha * hd)); self.beta_ssm = nn.Parameter(torch.ones(self.ssm.d_in))
        self.output = nn.Linear(ha * hd + self.ssm.d_in, c.d_model, bias=False)

    def forward(self, x, *, past=None, use_cache=False):
        if past is not None or use_cache:
            raise ValueError('the hybrid has no incremental cache; recompute the prefix')
        b, t, _ = x.shape; c = self.c; ha, hk, hd = c.attn_heads, c.attn_kv_heads, c.head_dim
        q = self.query(x).view(b, t, ha, hd).transpose(1, 2)
        k = self.key(x).view(b, t, hk, hd).transpose(1, 2)
        v = self.value(x).view(b, t, hk, hd).transpose(1, 2)
        q = self.rotary(q, start=0, base=c.rope_base); k = self.rotary(k, start=0, base=c.rope_base)
        if ha != hk:
            k = k.repeat_interleave(ha // hk, dim=1); v = v.repeat_interleave(ha // hk, dim=1)
        a = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=True)            # flash kernel
        a = self.attn_norm(a.transpose(1, 2).contiguous().view(b, t, ha * hd)) * self.beta_attn.to(x.dtype)
        s = self.ssm(x) * self.beta_ssm.to(x.dtype)
        return self.output(torch.cat([a, s], dim=-1)), None


class HybridBlock(nn.Module):
    def __init__(self, c, RMSNorm, SwiGLU, rotary):
        super().__init__()
        self.attention_norm, self.ffn_norm = RMSNorm(c.d_model, c.norm_epsilon), RMSNorm(c.d_model, c.norm_epsilon)
        self.attention, self.ffn = HybridMixer(c, RMSNorm, rotary), SwiGLU(c)

    def forward(self, x, *, past=None, use_cache=False, mla_mode='materialized'):
        delta, cache = self.attention(self.attention_norm(x), past=past, use_cache=use_cache)
        x = x + delta
        return x + self.ffn(self.ffn_norm(x)), cache


class HybridLM(nn.Module):
    def __init__(self, config, scaffold=None):
        super().__init__()
        RMSNorm, SwiGLU, rotary = _scaffold(scaffold)
        self.config = config.validate()
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.blocks = nn.ModuleList(HybridBlock(config, RMSNorm, SwiGLU, rotary) for _ in range(config.layers))
        self.final_norm = RMSNorm(config.d_model, config.norm_epsilon)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Embedding)):
                nn.init.normal_(m.weight, mean=0.0, std=config.init_std)
        if config.tie_embeddings:
            self.lm_head.weight = self.embedding.weight

    def forward(self, input_ids, *, targets=None):
        x = self.embedding(input_ids)
        for block in self.blocks:
            x, _ = block(x)
        logits = self.lm_head(self.final_norm(x))
        loss = None if targets is None else F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), targets.reshape(-1), ignore_index=-100)
        return {'logits': logits, 'loss': loss, 'cache': None}

    @torch.inference_mode()
    def greedy(self, prompt, *, new_tokens):
        """Greedy continuation, recomputing the whole prefix at every step (no cache): fine for evaluation, slow for long outputs."""
        was = self.training; self.eval()
        try:
            tokens = prompt
            for _ in range(new_tokens):
                tokens = torch.cat([tokens, self(tokens)['logits'][:, -1].argmax(-1, keepdim=True)], 1)
            return tokens[:, prompt.shape[1]:]
        finally:
            self.train(was)


def restore_hybrid(config, checkpoint, *, state_key='model', map_location='cpu', scaffold=None):
    """Counterpart of training_methods.restore_backbone for a checkpoint written by train.py --arch hybrid."""
    config = config if isinstance(config, HybridConfig) else HybridConfig(**{k: v for k, v in config.items()})
    model = HybridLM(config, scaffold)
    saved = torch.load(checkpoint, map_location=map_location, weights_only=False)
    model.load_state_dict(saved.get(state_key, saved) if isinstance(saved, dict) else saved, strict=True)
    return model


def scaled_init(model):
    """GPT-2 style: output projections of attention and FFN start at std / sqrt(2 layers)."""
    std = model.config.init_std / math.sqrt(2 * model.config.layers)
    for blk in model.blocks:
        for lin in (blk.attention.output, blk.ffn.down):
            nn.init.normal_(lin.weight, mean=0.0, std=std)

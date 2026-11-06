"""Bidirectional encoder built from the course decoder: same modules, same parameter names, same checkpoint layout, only the attention is not causal.

    model = CausalDecoder(config)                 # or restore_backbone(config, 'model_final.pt')
    to_encoder(model)                             # attention in every block now sees the whole sequence (RoPE positions unchanged)
    to_encoder(model, pad_id=1)                   # downstream use: padded batches; pad positions are hidden from the other tokens

Because the parameters are identical, everything written for the decoder keeps working on the result: `backbone_hidden`, LoRA / LoRMA / IA3 injection
(they replace attention.query / value / ...), partial fine-tuning, layer dropping, `restore_backbone`. Only the attention pattern changes.
"""
from __future__ import annotations

import sys

from torch.nn import functional as F


def _classes(scaffold=None):
    if scaffold and scaffold not in sys.path:
        sys.path.insert(0, scaffold)
    from legal_scaffold.models import CausalAttention, rotary
    return CausalAttention, rotary


def make_attention_class(scaffold=None):
    CausalAttention, rotary = _classes(scaffold)

    class BidirectionalAttention(CausalAttention):
        """MHA/GQA attention without the causal mask. `holder['mask']` (bool [batch,1,1,time], True = may be attended) is set by the pad hook, if installed."""
        holder = None

        def forward(self, x, *, past=None, use_cache=False):
            if past is not None or use_cache:
                raise ValueError('the bidirectional encoder has no KV cache')
            b, t, _ = x.shape; c = self.config
            q = self.query(x).view(b, t, c.heads, self.head_dim).transpose(1, 2)
            k = self.key(x).view(b, t, c.kv_heads, self.head_dim).transpose(1, 2)
            v = self.value(x).view(b, t, c.kv_heads, self.head_dim).transpose(1, 2)
            q = rotary(q, start=0, base=c.rope_base); k = rotary(k, start=0, base=c.rope_base)
            groups = c.heads // c.kv_heads
            if groups != 1:
                k = k.repeat_interleave(groups, dim=1); v = v.repeat_interleave(groups, dim=1)
            mask = None if self.holder is None else self.holder.get('mask')
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.0, is_causal=False)
            return self.output(out.transpose(1, 2).contiguous().view(b, t, c.d_model)), None

    return BidirectionalAttention


def to_encoder(model, *, pad_id=None, scaffold=None):
    """Turn a CausalDecoder (MHA or GQA) into a bidirectional encoder in place; returns the model."""
    if model.config.attention == 'mla':
        raise ValueError('the bidirectional conversion supports MHA/GQA attention only')
    cls = make_attention_class(scaffold)
    holder = {}
    for block in model.blocks:
        block.attention.__class__ = cls
        block.attention.holder = holder
    if pad_id is not None:
        def hook(module, inputs, output):
            keep = inputs[0] != pad_id
            keep = keep | ~keep.any(1, keepdim=True)         # a row of nothing but padding attends to itself instead of producing NaN
            holder['mask'] = None if bool(keep.all()) else keep[:, None, None, :]
        model.embedding.register_forward_hook(hook)
    model.is_encoder = True
    return model

"""Training-throughput benchmark for the pretraining model on whatever GPU this runs on (L4, A100, G4, ...). Random tokens, real model code.

Colab:   !unzip -q -o /content/drive/MyDrive/legal-llm-course/phase2_lner_multimethod_bundle.zip -d /content/scaffold
         !python gpu_benchmark.py --scaffold /content/scaffold --cu-per-hour 1.54          (use the GPU's own compute units per hour)
Options: --compile (torch.compile), --loss chunked|full, --micro-batches 4,8,16,32, --seq 2048, --layers/--d-model/... to change the model.

For each micro-batch size it reports tokens/s, peak memory, model-FLOPs utilisation (if the GPU's peak is known or given) and the projected hours and
compute units for --tokens (default 4e9). Compare GPUs by tokens per compute unit, not by tokens per second.
"""
from __future__ import annotations

import argparse
import sys
import time

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

PEAK_BF16_TFLOPS = {'L4': 121, 'A100': 312, 'H100': 989, 'T4': 65, 'A10': 125, 'RTX 4090': 165, 'V100': 125}


def build_model(a):
    sys.path.insert(0, a.scaffold)
    if a.attention == 'hybrid':
        import os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import hybrid
        cfg = hybrid.HybridConfig(vocab_size=a.vocab, d_model=a.d_model, layers=a.layers, ffn_hidden=a.ffn, context_length=a.seq).validate()
        return hybrid.HybridLM(cfg, a.scaffold), cfg
    from legal_scaffold.models import CausalDecoder, ModelConfig
    cfg = ModelConfig(vocab_size=a.vocab, d_model=a.d_model, layers=a.layers, heads=a.heads, kv_heads=a.kv_heads, ffn_hidden=a.ffn,
                      context_length=a.seq, attention=a.attention, norm_placement='pre', tie_embeddings=True, rope_base=10000, norm_epsilon=1e-6,
                      init_std=.02, **({'mla_latent': a.d_model // 4, 'mla_query_rank': a.d_model // 4, 'mla_rope_dim': 32} if a.attention == 'mla' else {})).validate()
    return CausalDecoder(cfg), cfg


def hidden_states(model, ids):
    x = model.embedding(ids)
    for block in model.blocks:
        x, _ = block(x, past=None, use_cache=False, mla_mode='materialized')
    return model.final_norm(x)


def chunked_ce(model, h, targets, chunk=4096):
    """Cross-entropy without materialising the full [tokens, vocab] logits: logits are recomputed per chunk in backward."""
    h, t = h.reshape(-1, h.shape[-1]), targets.reshape(-1)
    def piece(hc, tc): return F.cross_entropy(model.lm_head(hc).float(), tc, reduction='sum')
    total = sum(checkpoint(piece, h[i:i + chunk], t[i:i + chunk], use_reentrant=False) for i in range(0, h.shape[0], chunk))
    return total / t.numel()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scaffold', default='.', help='folder containing legal_scaffold/')
    ap.add_argument('--vocab', type=int, default=32768); ap.add_argument('--d-model', type=int, default=1024); ap.add_argument('--layers', type=int, default=16)
    ap.add_argument('--heads', type=int, default=16); ap.add_argument('--kv-heads', type=int, default=4); ap.add_argument('--ffn', type=int, default=2816)
    ap.add_argument('--attention', default='gqa', choices=['mha', 'gqa', 'mla', 'hybrid']); ap.add_argument('--seq', type=int, default=2048)
    ap.add_argument('--micro-batches', default='4,8,16,32'); ap.add_argument('--steps', type=int, default=8); ap.add_argument('--warmup', type=int, default=3)
    ap.add_argument('--loss', default='chunked', choices=['chunked', 'full']); ap.add_argument('--compile', action='store_true')
    ap.add_argument('--cu-per-hour', type=float, default=None); ap.add_argument('--peak-tflops', type=float, default=None); ap.add_argument('--tokens', type=float, default=4e9)
    a = ap.parse_args()
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    name = torch.cuda.get_device_name(0) if dev.type == 'cuda' else 'cpu'
    peak = a.peak_tflops or next((v for k, v in PEAK_BF16_TFLOPS.items() if k in name), None)
    amp = torch.bfloat16 if (dev.type == 'cuda' and torch.cuda.is_bf16_supported()) else (torch.float16 if dev.type == 'cuda' else None)
    torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
    model, cfg = build_model(a); model.to(dev).train()
    n_all = sum(p.numel() for p in model.parameters()); n_emb = model.embedding.weight.numel(); n = n_all - n_emb
    flops_tok = 6 * n + 12 * cfg.layers * cfg.d_model * a.seq
    print(f'GPU: {name} | precision: {amp} | params {n_all/1e6:.0f}M (non-embedding {n/1e6:.0f}M) | peak bf16 TFLOPs: {peak} | loss: {a.loss} | compile: {a.compile}')
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, betas=(0.9, 0.95), fused=dev.type == 'cuda')
    scaler = torch.amp.GradScaler('cuda', enabled=amp == torch.float16)
    fwd = torch.compile(hidden_states) if a.compile else hidden_states
    rows = []
    for mb in [int(x) for x in a.micro_batches.split(',')]:
        try:
            if dev.type == 'cuda':
                torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
            ids = torch.randint(0, a.vocab, (mb, a.seq), device=dev); tgt = torch.randint(0, a.vocab, (mb, a.seq), device=dev)

            def step():
                with torch.autocast(device_type=dev.type, dtype=amp, enabled=amp is not None):
                    h = fwd(model, ids)
                    loss = chunked_ce(model, h, tgt) if a.loss == 'chunked' else F.cross_entropy(model.lm_head(h).float().reshape(-1, a.vocab), tgt.reshape(-1))
                scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
            for _ in range(a.warmup): step()
            if dev.type == 'cuda': torch.cuda.synchronize()
            t0 = time.time()
            for _ in range(a.steps): step()
            if dev.type == 'cuda': torch.cuda.synchronize()
            dt = (time.time() - t0) / a.steps; tps = mb * a.seq / dt
            mem = torch.cuda.max_memory_allocated() / 1e9 if dev.type == 'cuda' else float('nan')
            row = {'micro_batch': mb, 'tokens/s': int(tps), 'peak_GB': round(mem, 1), 'MFU_%': round(100 * tps * flops_tok / (peak * 1e12), 1) if peak else None,
                   'hours_for_tokens': round(a.tokens / tps / 3600, 1)}
            if a.cu_per_hour: row['CU_for_tokens'] = round(row['hours_for_tokens'] * a.cu_per_hour); row['M_tokens_per_CU'] = round(tps * 3600 / a.cu_per_hour / 1e6, 1)
            rows.append(row); print(row, flush=True)
        except torch.cuda.OutOfMemoryError:
            print({'micro_batch': mb, 'result': 'out of memory'}, flush=True); break
        finally:
            for k in ('ids', 'tgt'): globals().pop(k, None)
    if rows:
        best = max(rows, key=lambda r: r['tokens/s'])
        print(f"\nbest: micro-batch {best['micro_batch']} -> {best['tokens/s']:,} tokens/s, {best['hours_for_tokens']} h for {a.tokens/1e9:.1f}B tokens" + (f", about {best['CU_for_tokens']} compute units" if 'CU_for_tokens' in best else ''))


if __name__ == '__main__':
    main()

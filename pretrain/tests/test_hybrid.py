"""python -I pretrain/tests/test_hybrid.py   (CPU)"""
import sys, torch
from pathlib import Path
root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(root / 'common')); sys.path.insert(0, str(root / 'pretrain'))
import hybrid as H

torch.manual_seed(0)
b, t, h, p, n = 2, 37, 3, 5, 4
x = torch.randn(b, t, h, p, dtype=torch.float64, requires_grad=True)
dt = (torch.rand(b, t, h, dtype=torch.float64) * .3 + .01).requires_grad_(True)
A = (-torch.rand(h, dtype=torch.float64) * 2 - .1).requires_grad_(True)
B = torch.randn(b, t, n, dtype=torch.float64, requires_grad=True); C = torch.randn(b, t, n, dtype=torch.float64, requires_grad=True)
ref = H.ssd_reference(x, dt, A, B, C)
for chunk in (8, 16, 64):          # 37 is not a multiple of any of them: padding path
    y = H.ssd(x, dt, A, B, C, chunk)
    assert torch.allclose(y, ref, atol=1e-9), (chunk, (y - ref).abs().max())
g_ref = torch.autograd.grad(ref.square().sum(), (x, dt, A, B, C))
g = torch.autograd.grad(H.ssd(x, dt, A, B, C, 8).square().sum(), (x, dt, A, B, C))
for a_, b_ in zip(g, g_ref):
    assert torch.allclose(a_, b_, atol=1e-7, rtol=1e-6)
print('ssd == recurrence (outputs and gradients, three chunk sizes, padded length) OK')

cfg = H.HybridConfig(vocab_size=100, d_model=64, layers=2, ffn_hidden=96, context_length=64, attn_heads=4, attn_kv_heads=2, head_dim=16, ssm_heads=4, ssm_headdim=16, ssm_state=8, ssm_chunk=8)
m = H.HybridLM(cfg, str(root / 'common')).eval()
ids = torch.randint(0, 100, (2, 29))
with torch.no_grad():
    base = m(ids)['logits']
    ids2 = ids.clone(); ids2[:, 20:] = torch.randint(0, 100, (2, 9))
    other = m(ids2)['logits']
assert torch.allclose(base[:, :20], other[:, :20], atol=1e-5), 'future tokens changed earlier outputs'
assert not torch.allclose(base[:, 20:], other[:, 20:], atol=1e-3)
print('causal: changing tokens 20.. leaves outputs before 20 unchanged OK')
out = m(ids, targets=ids)['loss']; out.backward()
assert all(p_.grad is not None and torch.isfinite(p_.grad).all() for p_ in m.parameters() if p_.requires_grad)
print('all parameters get finite gradients OK; initial loss', round(out.item(), 3), '(ln 100 = 4.605)')

big = H.HybridConfig(vocab_size=32768, d_model=1024, layers=16, ffn_hidden=2560, context_length=2048)
with torch.device('meta'):
    mm = H.HybridLM(big, str(root / 'common'))
n_all = sum(p_.numel() for p_ in mm.parameters()); print(f'default config: {n_all/1e6:.1f}M parameters (non-embedding {(n_all-32768*1024)/1e6:.1f}M)')
print('ALL OK')

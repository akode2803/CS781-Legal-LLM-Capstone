"""python -I pretrain/tests/test_downstream_loading.py   (CPU)
Checkpoints in the layout pretrain/train.py writes -> rr_lib.load_backbone / cjpe_lib.build_model: decoder, bidirectional encoder, hybrid."""
import dataclasses, sys, tempfile, torch
from pathlib import Path
root = Path(__file__).resolve().parents[2]
for p in ('common', 'pretrain', 'phase3_rr', 'phase4_cjpe'):
    sys.path.insert(0, str(root / p))
import hybrid as H
import rr_lib as R
import cjpe_lib as J
from legal_scaffold.models import CausalDecoder, ModelConfig

torch.manual_seed(0)
V = 120
dec_cfg = ModelConfig(vocab_size=V, d_model=64, layers=3, heads=4, kv_heads=2, ffn_hidden=96, context_length=64, attention='gqa',
                      norm_placement='pre', tie_embeddings=True, rope_base=10000, norm_epsilon=1e-6, init_std=.02).validate()
hyb_cfg = H.HybridConfig(vocab_size=V, d_model=64, layers=3, ffn_hidden=96, context_length=64, attn_heads=4, attn_kv_heads=2, head_dim=16,
                         ssm_heads=4, ssm_headdim=16, ssm_state=8, ssm_chunk=8)
legacy_cfg = dataclasses.asdict(dec_cfg) | {'layers': 5}           # deliberately wrong: the file's own config must win

tmp = Path(tempfile.mkdtemp())
def save(name, model, cfg, objective):
    torch.save({'model': model.state_dict(), 'config': dataclasses.asdict(cfg), 'step': 1, 'objective': objective}, tmp / name)
    return str(tmp / name)
ck_dec = save('dec.pt', CausalDecoder(dec_cfg), dec_cfg, 'clm')
ck_enc = save('enc.pt', CausalDecoder(dec_cfg), dec_cfg, 'mlm')
ck_hyb = save('hyb.pt', H.HybridLM(hyb_cfg, str(root / 'common')), hyb_cfg, 'clm')
ck_old = str(tmp / 'old.pt'); torch.save({'model': CausalDecoder(dec_cfg).state_dict()}, ck_old)      # a Phase 1 style file: no 'objective'

ids = torch.randint(3, V, (2, 20)); ids[:, 14:] = R.PAD_ID                   # right-padded rows, like cjpe_lib.collate
def hidden(bb, x):
    with torch.no_grad():
        return R.backbone_hidden(bb, x)

# decoder: untouched path, causal
legal_cfg = dataclasses.asdict(dec_cfg)
d = R.load_backbone(legal_cfg, ck_dec)
assert type(d).__name__ == 'CausalDecoder' and not getattr(d, 'is_encoder', False)
a = hidden(d, ids); ids2 = ids.clone(); ids2[:, 10] = 7
assert torch.allclose(a[:, :10], hidden(d, ids2)[:, :10], atol=1e-5), 'decoder must be causal'
print('decoder checkpoint OK (causal)')

# old-style file takes the original path and still honours cfg_dict
o = R.load_backbone(legal_cfg, ck_old); assert o.config.layers == 3
print('old-style Phase 1 file OK (no config -> restore_backbone)')

# exported by an earlier train.py: config but no 'objective'; run.json beside it says which
import json
sub = tmp / 'enc_run'; sub.mkdir()
torch.save({'model': CausalDecoder(dec_cfg).state_dict(), 'config': dataclasses.asdict(dec_cfg)}, sub / 'model_final.pt')
(sub / 'run.json').write_text(json.dumps({'args': {'objective': 'mlm'}}))
assert R.load_backbone(legal_cfg, str(sub / 'model_final.pt')).is_encoder
(sub / 'run.json').unlink()
assert not getattr(R.load_backbone(legal_cfg, str(sub / 'model_final.pt')), 'is_encoder', False)
print('config-without-objective file OK (run.json decides)')

# encoder: same weights, bidirectional, padding hidden
e = R.load_backbone(legacy_cfg, ck_enc)
assert e.is_encoder and e.config.layers == 3, 'config must come from the file, not cfg_dict'
a = hidden(e, ids); b = hidden(e, ids2)
assert not torch.allclose(a[:, :10], b[:, :10], atol=1e-4), 'encoder should see later tokens'
longer = torch.cat([ids, torch.full((2, 9), R.PAD_ID)], 1)                  # more padding must not change the real tokens
assert torch.allclose(hidden(e, longer)[:, :14], a[:, :14], atol=1e-4), 'padding leaked into real tokens'
allpad = torch.full((2, 8), R.PAD_ID); assert torch.isfinite(hidden(e, allpad)).all(), 'all-pad row produced NaN'
print('encoder checkpoint OK (bidirectional, padding hidden, all-pad safe, file config wins)')

# hybrid
h = R.load_backbone(legacy_cfg, ck_hyb)
assert type(h).__name__ == 'HybridLM' and h.config.attention == 'hybrid'
a = hidden(h, ids); b = hidden(h, ids2)
assert torch.allclose(a[:, :10], b[:, :10], atol=1e-5), 'hybrid must be causal'
print('hybrid checkpoint OK (causal)')

# full model assembly through cjpe_lib, every new backbone, with adapters and a thinned student
batch_fields = dict(head_kw={'hidden': 16})
for name, ck in (('decoder', ck_dec), ('encoder', ck_enc), ('hybrid', ck_hyb)):
    for method in ('full', 'lora', 'lorma'):
        m = J.build_model(legacy_cfg, ck, method, **batch_fields)
        trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
        assert trainable > 0, (name, method)
        vec = m.chunk_vectors({'input_ids': ids, 'pool': (ids != R.PAD_ID).float() / (ids != R.PAD_ID).sum(1, keepdim=True), 'last': torch.tensor([13, 13])})
        assert vec.shape == (2, 128) and torch.isfinite(vec).all(), (name, method)
    thin = J.build_model(legacy_cfg, ck, 'full', layer_keep=[0, 2], **batch_fields)
    assert thin.backbone.config.layers == 2 and len(thin.backbone.blocks) == 2
    print(f'{name}: cjpe_lib.build_model OK for full / lora / lorma, thinned student OK')
print('ALL OK')

"""Every architecture x method on a tiny random model with synthetic multi-label documents, plus metrics, cache and export checks."""
import sys, json, zipfile, tempfile, random, numpy as np, torch
from pathlib import Path
REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO/'phase5_lsi'), str(REPO/'phase4_cjpe'), str(REPO/'phase3_rr'), str(REPO/'common')]
import lsi_lib as L, cjpe_lib as J, rr_lib as R
from legal_scaffold.models import ModelConfig, CausalDecoder
torch.manual_seed(0); random.seed(0); np.random.seed(0)
base = dict(vocab_size=32768, d_model=64, layers=4, heads=4, kv_heads=4, ffn_hidden=128, context_length=2048, tie_embeddings=True, rope_base=10000, norm_epsilon=1e-6, init_std=.02)
ARCH = {'mha': dict(attention='mha', norm_placement='pre'), 'post': dict(attention='mha', norm_placement='post'),
        'gqa': dict(attention='gqa', norm_placement='pre', kv_heads=2), 'mla': dict(attention='mla', norm_placement='pre', kv_heads=2, mla_latent=32, mla_query_rank=32, mla_rope_dim=8)}
NL = 10   # statutes 0..9 actually occur; the head still has 100 outputs
def docs(n, labeled=True):
    out = []
    for i in range(n):
        labs = sorted(random.sample(range(NL), random.randint(1, 3))); T = random.randint(200, 1500)
        t = np.random.randint(10, 200, size=T).astype(np.uint16)
        for l in labs: t[random.randrange(T)] = 5000 + l            # a marker token per statute at a random position
        out.append({'id': f'd{i}', 'text': None, 'tok': t, 'labels': labs if labeled else None})
    return out
data = {'train': docs(120), 'dev': docs(40), 'test': docs(15, False)}
dev = torch.device('cpu')
C0 = dict(chunk_len=128, stride=128, max_chunks=8, select='uniform', agg='attn', batch_docs=8, accum=1, eval_docs=8, epochs=2, seed=1, class_weight='sqrt',
          docs_per_epoch=None, dev_subset=None, kd_temperature=2.0, kd_alpha=0.3, partial_layers=(2, 3), log_every=0)
LR = {'full': dict(backbone=1e-4, adapter=0, head=3e-3), 'partial': dict(backbone=1e-4, adapter=0, head=3e-3), 'lora': dict(backbone=0, adapter=3e-3, head=3e-3),
      'lorma': dict(backbone=0, adapter=3e-3, head=3e-3), 'ia3': dict(backbone=0, adapter=3e-3, head=3e-3)}
def mk(arch, method, layer_keep=None):
    cfg = dict(base, **ARCH[arch]); bb = CausalDecoder(ModelConfig(**cfg).validate())
    return J.assemble(bb, cfg, method, head_kw=dict(n_classes=L.N_LABELS, hidden=64, agg='attn'), adapter_kw=dict(rank=4, alpha=8, targets=('query', 'value')), layer_keep=layer_keep), cfg
for arch in ARCH:
    for method in ('full', 'lora', 'lorma', 'ia3'):
        m, _ = mk(arch, method); C = dict(C0, method=method, preset=arch, run_id='t', lr=LR[method])
        with tempfile.TemporaryDirectory() as td:
            res, amp, Pdev = L.train_run(m, data, td, C, dev, log=lambda *a: None)
            ev, _, Ptest, out = L.export_run(m, data, C, dev, amp, td, log=lambda *a: None, P_dev=Pdev)
            for tag in ('submission_global', 'submission_perlabel'):
                z = zipfile.ZipFile(f'{td}/{tag}/submission.zip'); assert z.namelist() == ['prediction.json']
                rows = json.loads(z.read('prediction.json')); assert len(rows) == 15 and all(isinstance(r['pred'], list) and len(r['pred']) >= 1 and all(isinstance(k, int) for k in r['pred']) for r in rows)
        print(f"{arch:5s} {method:6s} trainable {res['trainable_pct']:6.2f}% | dev macro-F1 (global tuned) {ev['global_tuned']['macro_f1']:.3f} micro {ev['global_tuned']['micro_f1']:.3f}")
m, _ = mk('mha', 'partial'); C = dict(C0, method='partial', preset='mha', run_id='t', lr=LR['partial'], epochs=1)
with tempfile.TemporaryDirectory() as td: res, _, _ = L.train_run(m, data, td, C, dev, log=lambda *a: None); print('partial trainable %', round(res['trainable_pct'], 1))
for sel in J.SELECTIONS:
    m, _ = mk('gqa', 'lora'); C = dict(C0, method='lora', preset='gqa', run_id='t', select=sel, lr=LR['lora'], epochs=1)
    with tempfile.TemporaryDirectory() as td: L.train_run(m, data, td, C, dev, log=lambda *a: None)
for gamma in (0.0, 2.0):
    m, _ = mk('gqa', 'lorma'); C = dict(C0, method='lorma', preset='gqa', run_id='t', lr=LR['lorma'], epochs=1, focal_gamma=gamma, class_weight='inverse')
    with tempfile.TemporaryDirectory() as td: L.train_run(m, data, td, C, dev, log=lambda *a: None)
print('selection modes, class weights (none/sqrt/inverse) and focal loss: ok')
# distillation: 4-layer teacher (full) -> 2-layer student
T, _ = mk('mha', 'full'); CT = dict(C0, method='full', preset='mha', run_id='t', lr=LR['full'], epochs=3)
with tempfile.TemporaryDirectory() as td: L.train_run(T, data, td, CT, dev, log=lambda *a: None)
S, _ = mk('mha', 'full', layer_keep=(0, 2)); CS = dict(C0, method='kd', preset='mha', run_id='s', lr=LR['full'], epochs=2)
assert sum(p.numel() for p in T.parameters()) > sum(p.numel() for p in S.parameters())
with tempfile.TemporaryDirectory() as td: res, _, _ = L.train_run(S, data, td, CS, dev, teacher=T.eval(), log=lambda *a: None)
print('KD student ok, params', sum(p.numel() for p in S.parameters()), '<', sum(p.numel() for p in T.parameters()))
# learnability with the marker-token task, then thresholds / analysis helpers / progress lines / time cap
big = {'train': docs(400), 'dev': docs(80), 'test': docs(15, False)}
m, _ = mk('gqa', 'lorma'); C = dict(C0, method='lorma', preset='gqa', run_id='t', lr=dict(backbone=0, adapter=3e-3, head=5e-3), epochs=12, log_every=20)
with tempfile.TemporaryDirectory() as td:
    lines = []; res, amp, P = L.train_run(m, big, td, C, dev, log=lines.append)
    Y = L.multi_hot(big['dev']); ev = L.evaluate_probs(Y, P)
    print('learnability (marker tokens): micro-F1 global-tuned', round(ev['global_tuned']['micro_f1'], 3), '| macro-F1 over the 10 occurring statutes', round(float(np.mean(ev['global_tuned']['per_label_f1'][:NL])), 3))
    print('progress lines printed:', sum('batch' in l for l in lines) > 0)
    print({k: (round(v['macro_f1'], 3) if isinstance(v, dict) else round(v, 2)) for k, v in ev.items()})
    pred = L.decide(P, L.tune_per_label(Y, P)); print('confusion pairs ok:', len(L.confusion_pairs(Y, pred, 5)) == 5, '| table rows', len(L.per_label_table(Y, P, 0.3, big['train'])))
    S2 = L.stack_cooccurrence_cv(Y, P, k=3); print('co-occurrence stacking ok, macro-F1', round(L.scores(Y, L.decide(S2, L.tune_global(Y, S2)[0], True))['macro_f1'], 3))
    print('selection ablation ok:', len(L.selection_ablation(m, big['dev'], C, dev, amp, modes=('first', 'uniform'), ks=(1, 4))) == 4)
with tempfile.TemporaryDirectory() as td:
    m, _ = mk('gqa', 'lora'); res, _, _ = L.train_run(m, big, td, dict(C0, method='lora', preset='gqa', run_id='t', lr=LR['lora'], epochs=3, max_minutes=0.0003), dev, log=lambda *a: None)
    print('time cap mid-epoch: epochs run', len(res['history']), '(of 3), results.json written')
# cache round trip incl. unlabeled test
with tempfile.TemporaryDirectory() as td:
    p = L.save_cache(data, td + '/c.npz', 'sha', 40000); back = L.load_cache(p, 'sha', 40000)
    assert all(a['id'] == b['id'] and a['labels'] == b['labels'] and np.array_equal(a['tok'], b['tok']) for s in data for a, b in zip(data[s], back[s])); assert back['test'][0]['labels'] is None
    assert L.load_cache(p, 'other', 40000, log=lambda m: None) is None; print('cache round trip ok')
print('ALL OK')

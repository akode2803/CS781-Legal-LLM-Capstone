"""Pretraining curves for the report: validation loss of the GQA decoder and the hybrid, same data order."""
import json, sys
from pathlib import Path
import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt

here = Path(__file__).parent
def val(name):
    rows = [json.loads(l) for l in open(here / 'logs' / f'pretrain_{name}_log.jsonl')]
    v = [r for r in rows if 'val_loss' in r]
    return [r['tokens'] / 1e9 for r in v], [r['val_loss'] for r in v]

plt.rcParams.update({'font.size': 8, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.edgecolor': '#888', 'xtick.color': '#555', 'ytick.color': '#555'})
fig, (a, b) = plt.subplots(1, 2, figsize=(6.3, 1.75), constrained_layout=True)
tg, lg = val('gqa'); th, lh = val('hybrid')
a.plot(tg, lg, color='#2a78d6', lw=2, label='GQA decoder (final 1.236)')
a.plot(th, lh, color='#eb6834', lw=2, label='Hybrid (final 1.189)')
a.set_xlabel('training tokens (billions)'); a.set_ylabel('validation loss (nats/token)')
a.grid(axis='y', color='#ddd', lw=.6); a.legend(frameon=False)
gap = {round(t, 3): l for t, l in zip(tg, lg)}
d = [(t, gap[round(t, 3)] - l) for t, l in zip(th, lh) if round(t, 3) in gap]
b.plot([x for x, _ in d], [y for _, y in d], color='#555', lw=2)
b.axhline(0, color='#aaa', lw=.6)
b.set_xlabel('training tokens (billions)'); b.set_ylabel('GQA minus hybrid loss')
b.grid(axis='y', color='#ddd', lw=.6)
fig.savefig(here / 'figs' / 'pretrain_val.pdf')
print('gap at end', round(d[-1][1], 4), 'max', round(max(y for _, y in d), 4))

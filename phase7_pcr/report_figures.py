"""Report figures from report/analysis.json (vector PDF + PNG preview). usage: python report_figures.py"""
import json
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
A = json.load(open('report/analysis.json')); C, INK, MUTED = '#2a78d6', '#0b0b0b', '#52514e'
plt.rcParams.update({'font.size': 9, 'pdf.fonttype': 42, 'axes.edgecolor': MUTED, 'text.color': INK, 'axes.labelcolor': MUTED, 'xtick.color': MUTED, 'ytick.color': MUTED})


def panel(ax, labels, vals, ns, title, xlabel):
    ax.bar(labels, vals, color=C, width=.72)
    for i, (v, n) in enumerate(zip(vals, ns)):
        ax.text(i, v + .02, f'{v:.2f}', ha='center', fontsize=8); ax.text(i, .02, f'n={n}', ha='center', fontsize=7, color='white' if v > .12 else MUTED)
    ax.set_ylim(0, 1.08); ax.set_title(title, fontsize=9.5, loc='left'); ax.set_xlabel(xlabel)
    ax.spines[['top', 'right']].set_visible(False); ax.grid(axis='y', color='#e4e3df', lw=.6); ax.set_axisbelow(True)


def save(fig, name):
    fig.tight_layout(); fig.savefig(f'report/{name}.pdf'); fig.savefig(f'report/{name}.png', dpi=160)


g = A['by_gold_size']; lab = ['1-2', '3-5', '6-10', '11-20', '21+']; n = [b['n'] for b in g]
fig, ax = plt.subplots(1, 2, figsize=(6.6, 2.5), sharey=True)
panel(ax[0], lab, [b['precision'] for b in g], n, 'Precision@7', 'Gold cases per dev query')
panel(ax[1], lab, [b['recall'] for b in g], n, 'Recall@7', 'Gold cases per dev query')
save(fig, 'fig_gold_size')
q = A['by_query_words']; c = A['gold_recall_by_cand_words']
fig, ax = plt.subplots(1, 2, figsize=(6.6, 2.5), sharey=True)
panel(ax[0], ['<2k', '2-4k', '4-8k', '8-16k', '16k+'], [b['recall'] for b in q], [b['n'] for b in q], 'Recall@7 (n = queries)', 'Query length, words')
panel(ax[1], ['<1k', '1-2k', '2-4k', '4-8k', '8k+'], [b['recall'] for b in c], [b['n'] for b in c], 'Recall@7 (n = gold pairs)', 'Gold candidate length, words')
save(fig, 'fig_length')

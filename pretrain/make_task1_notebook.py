"""Writes notebooks/task1_generation.ipynb: the official Task 1 prompts through the three clean-slate pretrained models (run on Colab; any GPU)."""
import ast, nbformat
from nbformat.v4 import new_notebook, new_markdown_cell as md, new_code_cell as code

nb = new_notebook()
nb.cells.append(md("# Task 1 prompts on the clean-slate pretrained models\nThe official Task 1 prompts are 512-word judgment excerpts; the task is a 128-word continuation (greedy, as in the course export). "
                   "This notebook runs the new GQA decoder and the Hymba-style hybrid on them, lets the MLM encoder fill masked tokens instead (it cannot continue text), and shows every continuation next to the old 100M GQA model's.\n\n"
                   "Needs next to `rr_lib.py` in `MyDrive/legal-llm-course/`: `generate_task1.py`, `hybrid.py`, `encoder.py`; the three `pretrain/pretrain_*/model_final.pt`; `official-test/test.json`."))
nb.cells.append(code("""# 1 · Settings
from pathlib import Path
COURSE_ROOT = Path('/content/drive/MyDrive/legal-llm-course')
IDS        = [str(i) for i in range(1, 9)]          # which official prompts (ids '1'..'1000'); the old model's answers exist for all of them
STATS_IDS  = [str(i) for i in range(1, 51)]         # a larger sample for the repetition statistics (the encoder is cheap; the hybrid recomputes the prefix every token, so it is the slow one)
RUN_HYBRID_STATS = False                            # True: also run the hybrid on STATS_IDS (about 50 prompts x a few minutes)
CKPT = {'gqa': COURSE_ROOT/'pretrain/pretrain_dec_gqa/model_final.pt', 'hybrid': COURSE_ROOT/'pretrain/pretrain_hybrid/model_final.pt', 'encoder': COURSE_ROOT/'pretrain/pretrain_enc_mlm/model_final.pt'}
OLD_ZIP = COURSE_ROOT/'Codabench submissions - 4 architectures - legal 100M - 2026-10-06/GQA-pre_legal-100M-milestone_100001268-tokens.zip'
OUT = COURSE_ROOT/'task1_outputs'"""))
nb.cells.append(code("""# 2 · Setup: scaffold + scripts onto the local disk (the script expects pretrain/ and common/ side by side)
from google.colab import drive
drive.mount('/content/drive')
import shutil, subprocess, sys, zipfile, json
WORK = Path('/content/work'); (WORK/'pretrain').mkdir(parents=True, exist_ok=True); OUT.mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(COURSE_ROOT/'phase2_lner_multimethod_bundle.zip') as z: z.extractall(WORK/'common')
for f in ('generate_task1.py', 'hybrid.py', 'encoder.py'):
    src = max([q for q in (COURSE_ROOT/f, COURSE_ROOT/'pretrain'/f) if q.exists()], key=lambda q: q.stat().st_mtime); shutil.copy(src, WORK/'pretrain'/f); print(f, '<-', src)
shutil.copy(COURSE_ROOT/'official-test/test.json', WORK/'test.json')
for k, p in CKPT.items(): print(k, p.exists(), round(p.stat().st_size/1e6) if p.exists() else '-', 'MB')
import torch; print('device', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU (this will be slow; pick a GPU runtime)')
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
def run(kind, ids, name, *extra):
    out = WORK/f'{name}.json'
    cmd = [sys.executable, str(WORK/'pretrain/generate_task1.py'), kind, str(CKPT[kind]), '--ids', *ids, '--out', str(out), '--test', str(WORK/'test.json'), '--device', DEV, '--threads', '4', *extra]
    p = subprocess.run(cmd, capture_output=True, text=True); print(p.stdout[-1500:], p.stderr[-1500:] if p.returncode else '')
    p.check_returncode(); shutil.copy(out, OUT/f'{name}.json'); return json.load(open(out))
import textwrap, numpy as np, pandas as pd
def rep4(t):
    w = t.casefold().split(); g = [tuple(w[i:i + 4]) for i in range(len(w) - 3)]; return round((len(g) - len(set(g))) / len(g), 3) if g else 0.0
with zipfile.ZipFile(OLD_ZIP) as z: old = json.loads(z.read('prediction.json'))      # the old 100M GQA model's Phase 1 answers"""))
nb.cells.append(code("""# 3 · The new GQA decoder, greedy (the course decoding)
gqa = run('gqa', IDS, 'new_gqa_greedy')"""))
nb.cells.append(code("""# 4 · The hybrid, greedy (no KV cache: each token recomputes the prefix, so this is the slowest cell)
hyb = run('hybrid', IDS, 'new_hybrid_greedy')"""))
nb.cells.append(code("""# 5 · The MLM encoder: it cannot continue text, so it restores 30% masked tokens of the 512-token window (its pretraining task)
enc = run('encoder', IDS, 'new_encoder_fillmask')
print('mean exact-restore accuracy on masked tokens:', round(sum(v['top1_accuracy'] for v in enc.values()) / len(enc), 3))
for k in IDS[:3]: print('\\n[' + k + '] [predicted|original] for masked positions:\\n' + enc[k]['example'])"""))
nb.cells.append(code("""# 6 · Side by side with the old 100M GQA model (same prompts, same decoding rules)
for k in IDS:
    print('=' * 110); print(f'PROMPT {k} (last 60 words):'); print(textwrap.fill(gqa[k]['prompt_tail'], 110))
    for name, text in (('OLD 100M GQA', old[k]['text']), ('NEW GQA 214M', gqa[k]['continuation']), ('NEW HYBRID 214M', hyb[k]['continuation'])):
        print(f'\\n--- {name}  (words {len(text.split())}, repeated-4gram fraction {rep4(text)})'); print(textwrap.fill(text, 110))"""))
nb.cells.append(code("""# 7 · Greedy decoding loops on a model this size: compare decoding rules on one larger sample (same prompts, same model)
# repeated-4gram fraction = share of word 4-grams that already occurred (Phase 1 diagnostic); >= 0.5 counts as a loop. The old 100M model's greedy output is the reference point.
import numpy as np, pandas as pd
ARMS = {'greedy (course default)': [], 'greedy + no-repeat 4-gram': ['--no-repeat', '4'], 'greedy + no-repeat 3-gram': ['--no-repeat', '3'], 'greedy + repetition penalty 1.15': ['--rep-penalty', '1.15'],
        'nucleus T=0.8 p=0.9': ['--nucleus'], 'nucleus T=0.7 p=0.9 + no-repeat 4-gram': ['--nucleus', '--temperature', '0.7', '--no-repeat', '4']}
ARM_OUT = {}; rows = {'old 100M GQA, greedy': [rep4(old[k]['text']) for k in STATS_IDS]}
for name, extra in ARMS.items():
    ARM_OUT[name] = run('gqa', STATS_IDS, 'new_gqa_' + ''.join(c if c.isalnum() else '_' for c in name)[:40], *extra); rows['new GQA, ' + name] = [v['rep4'] for v in ARM_OUT[name].values()]
if RUN_HYBRID_STATS: hstat = run('hybrid', STATS_IDS, 'new_hybrid_greedy_stats'); rows['new hybrid, greedy'] = [v['rep4'] for v in hstat.values()]
display(pd.DataFrame({k: {'mean repeated-4gram fraction': round(float(np.mean(v)), 3), 'share of loops (>= 0.5)': round(float(np.mean(np.array(v) >= 0.5)), 3), 'prompts': len(v)} for k, v in rows.items()}).T)"""))
nb.cells.append(code("""# 8 · The same prompt under every decoding rule, to read (change SHOW to look at another prompt; a looping prompt makes the differences obvious)
SHOW = '7' if '7' in STATS_IDS else STATS_IDS[0]
print(textwrap.fill(next(iter(ARM_OUT.values()))[SHOW]['prompt_tail'], 110))
for name, out in ARM_OUT.items():
    if SHOW in out: print(f'\\n--- {name}  (repeated-4gram fraction {out[SHOW]["rep4"]})'); print(textwrap.fill(out[SHOW]['continuation'], 110))
print('\\nall outputs are also in', OUT)"""))
nb.cells.append(code("""# 9 · ROUGE of each decoding rule (Task 1 is scored with ROUGE against a hidden reference, so this is the number that matters)
# proxy: the first 384 words of an official 512-word prompt are the input, the last 128 words are the reference (the same kind of text, no hidden test needed).
# Own ROUGE implementation (lower-cased alphanumeric tokens, no stemming): compare the rules with each other, not with the leaderboard value.
PROXY_IDS = [str(i) for i in range(1, 61)]
prox = {}
for name, extra in ARMS.items():
    prox[name] = run('gqa', PROXY_IDS, 'proxy_gqa_' + ''.join(c if c.isalnum() else '_' for c in name)[:40], '--proxy-split', '384', *extra)
tab = pd.DataFrame({name: {m: round(float(np.mean([v[m] for v in out.values()])), 4) for m in ('rouge1', 'rouge2', 'rougeL')} | {'repeated-4gram': round(float(np.mean([v['rep4'] for v in out.values()])), 3)} for name, out in prox.items()}).T
display(tab.sort_values('rougeL', ascending=False))
# a reference point: how much ROUGE does the reference share with the text just before it? (copying the prompt's tail is a cheap baseline)
import importlib.util; spec = importlib.util.spec_from_file_location('g', WORK/'pretrain/generate_task1.py'); G = importlib.util.module_from_spec(spec); spec.loader.exec_module(G)
test = json.load(open(WORK/'test.json')); base = [G.rouge(' '.join(test[k]['text'].split()[256:384]), ' '.join(test[k]['text'].split()[384:])) for k in PROXY_IDS]
print('baseline, copy the 128 words before the split as the answer:', {m: round(float(np.mean([b[m] for b in base])), 4) for m in ('rouge1', 'rouge2', 'rougeL')})"""))
for c in nb.cells:
    if c.cell_type == 'code': ast.parse(c.source)
nbformat.write(nb, 'notebooks/task1_generation.ipynb'); print('wrote notebooks/task1_generation.ipynb')

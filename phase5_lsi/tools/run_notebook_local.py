"""Run a notebook's code cells on this machine against the Drive-for-Desktop copy of MyDrive/legal-llm-course (CPU notebooks: ensemble, tfidf variants).
    python run_notebook_local.py phase5_ensemble.ipynb
Reads the real Drive folder; ensemble outputs are redirected to ens-local-* folders so nothing already published there is overwritten."""
import sys, json, types, tempfile
from pathlib import Path
REPO = Path(__file__).resolve().parents[2]
SRC = Path.home()/'Library/CloudStorage/GoogleDrive-ACCOUNT/My Drive/legal-llm-course'
DRIVE = Path.home()/'Documents/cs781-local-cache/legal-llm-course'      # local mirror of just the files the CPU notebooks read
NEEDED = ['rr_lib.py', 'cjpe_lib.py', 'lsi_lib.py', 'hybrid.py', 'encoder.py', 'phase2_lner_multimethod_bundle.zip', 'Phase_5_LSI/Train/train.json', 'Phase_5_LSI/Train/dev.json', 'Phase_5_LSI/Test/test.json']
NEEDED += [f'runs/phase5_lsi/NEW-{m}/full/new-{n}-full-00{i}/{f}' for m, n in (('GQA', 'gqa'), ('ENC', 'enc'), ('HYB', 'hyb')) for i in (1, 2)
           for f in ('dev_probs.npy', 'test_probs.npy', 'dev_ids.json', 'test_ids.json')] + ['runs/phase5_lsi/tfidf/probs.npz']
import shutil, os
def sync():
    """copy what is missing (Drive files are streamed on first read, so this is the slow step, once)"""
    assert SRC.exists(), f'{SRC} not found (is Google Drive for desktop running?)'
    extra = [str(f.relative_to(SRC)) for f in sorted((SRC/'runs/phase5_lsi/tfidf').rglob('*.npz'))]      # probs.npz plus every saved variant
    for rel in NEEDED + [e for e in extra if e not in NEEDED]:
        a, b = SRC/rel, DRIVE/rel
        if b.exists() and b.stat().st_size == a.stat().st_size and b.stat().st_mtime >= a.stat().st_mtime: continue
        b.parent.mkdir(parents=True, exist_ok=True); print('copying', rel, round(a.stat().st_size/1e6, 1), 'MB', flush=True); shutil.copyfile(a, b)
sync()
LOCAL = Path(tempfile.mkdtemp())/'phase3-rr'
gc = types.ModuleType('google'); col = types.ModuleType('google.colab'); drv = types.ModuleType('google.colab.drive'); drv.mount = lambda *a, **k: None
us = types.ModuleType('google.colab.userdata')
def _get(k): raise KeyError(k)
us.get = _get; col.drive = drv; col.userdata = us; gc.colab = col
sys.modules.update({'google': gc, 'google.colab': col, 'google.colab.drive': drv, 'google.colab.userdata': us})
import pandas as pd
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 20); pd.set_option('display.max_colwidth', 80)
nbname = sys.argv[1]
cells = [''.join(c['source']) for c in json.load(open(REPO/'phase5_lsi/notebooks'/nbname))['cells'] if c['cell_type'] == 'code']
ns = {'display': print, '__name__': '__main__'}
for i, s in enumerate(cells):
    s = s.replace('/content/drive/MyDrive/legal-llm-course', str(DRIVE)).replace('/content/phase3-rr', str(LOCAL)).replace("ensemble/ens-00", "ensemble/ens-local-00")
    if 'tfidf_variants' in nbname:      # 8 GB machine: the larger 1-3-gram feature set stays on Colab
        s = s.replace("FEATURES = {'w12': ((1, 2), 200000, 3, 60000), 'w13': ((1, 3), 400000, 3, 120000)}", "FEATURES = {'w12': ((1, 2), 200000, 3, 60000)}")
    print(f'\n--- cell {i} ---', flush=True)
    exec(compile(s, f'{nbname}:cell{i}', 'exec'), ns)

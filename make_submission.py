"""Build the Hello IITK final submission zip from this repo.  Re-run after any change to the code or the report.

    python make_submission.py            # writes dist/CS781-CP-Aryan-Kumar-261110012-report_code.zip

Layout required by the assignment page (code/ and report/ at the zip root, no outer folder):
    code/README.md              overview: best submission per phase and the files that produced it
    code/phase1/ ... phase7/    source, notebooks, configs; no weights, checkpoints, data or submission outputs
    report/CS781-CP-Aryan-Kumar-261110012-report.pdf
The report PDF is taken from report/latex/report.pdf (compile it first).  The script refuses to build if it finds
model weights, secrets, files over 5 MB, or a PDF older than report.tex.
"""
import fnmatch, re, shutil, sys, zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
NAME = 'CS781-CP-Aryan-Kumar-261110012'
OUT = ROOT / 'dist' / f'{NAME}-report_code.zip'
PDF, TEX = ROOT / 'report/latex/report.pdf', ROOT / 'report/latex/report.tex'

# zip folder <- repo folders (several sources may share one phase folder)
PHASES = {
    'code/phase1/common': 'common',                      # course scaffold, training methods, frozen tokenizer, data download
    'code/phase1/original_models': 'phase1_original',    # the four ~65M decoders behind the best Task 1 score (Oct 6)
    'code/phase1/pretrain': 'pretrain',                  # clean-slate pretraining (GQA, hybrid, encoder) used in Phases 5 to 7
    'code/phase2': 'phase2_lner',
    'code/phase3': 'phase3_rr',
    'code/phase4': 'phase4_cjpe',
    'code/phase5': 'phase5_lsi',
    'code/phase6': 'phase6_summ',
    'code/phase7': 'phase7_pcr',
}
EXTRA = {'code/README.md': 'submission/code_README.md'}

SKIP_DIRS = {'__pycache__', '.ipynb_checkpoints', 'cache', 'submissions', 'out', '.git'}
SKIP_FILES = ['*.zip', '*.pt', '*.pth', '*.bin', '*.safetensors', '*.ckpt', '*.npy', '*.npz', '*.env', '.env*', '.DS_Store', '*.pyc']
SECRET = re.compile(rb'hf_[A-Za-z0-9]{30,}|ya29\.[A-Za-z0-9_-]{20,}|ghp_[A-Za-z0-9]{30,}|AIza[0-9A-Za-z_-]{30,}|"refresh_token"\s*:|sk-[A-Za-z0-9]{30,}')
MAX_BYTES = 5_000_000


def files():
    for dest, src in PHASES.items():
        base = ROOT / src
        for p in sorted(base.rglob('*')):
            rel = p.relative_to(base)
            if p.is_dir() or any(part in SKIP_DIRS for part in rel.parts[:-1]): continue
            if any(fnmatch.fnmatch(p.name, pat) for pat in SKIP_FILES): continue
            yield f'{dest}/{rel.as_posix()}', p
    for dest, src in EXTRA.items(): yield dest, ROOT / src
    yield f'report/{NAME}-report.pdf', PDF


def main():
    problems = []
    if not PDF.is_file(): sys.exit(f'missing {PDF}: compile the report first')
    if TEX.stat().st_mtime > PDF.stat().st_mtime: problems.append('report.pdf is older than report.tex: recompile it')
    entries = list(files())
    for arc, p in entries:
        data = p.read_bytes()
        if len(data) > MAX_BYTES and not arc.endswith('tokenizer.json'): problems.append(f'{arc}: {len(data) / 1e6:.1f} MB')
        if SECRET.search(data): problems.append(f'{arc}: looks like it contains a credential')
    if problems: sys.exit('refusing to build:\n  ' + '\n  '.join(problems))

    OUT.parent.mkdir(exist_ok=True)
    tmp = OUT.with_suffix('.tmp')
    with zipfile.ZipFile(tmp, 'w', zipfile.ZIP_DEFLATED) as z:
        for arc, p in entries: z.write(p, arc)
    shutil.move(tmp, OUT)

    with zipfile.ZipFile(OUT) as z:   # check the layout the assignment asks for
        names = z.namelist()
        assert {n.split('/')[0] for n in names} == {'code', 'report'}, 'zip root must hold only code/ and report/'
        assert [n for n in names if n.startswith('report/')] == [f'report/{NAME}-report.pdf']
        for k in range(1, 8): assert any(n.startswith(f'code/phase{k}/') for n in names), f'code/phase{k}/ is empty'
    per = {}
    for n in names:
        top = '/'.join(n.split('/')[:2]); per[top] = per.get(top, 0) + 1
    for k, v in sorted(per.items()): print(f'{k:16s} {v:4d} files')
    print(f'{OUT.relative_to(ROOT)}: {len(names)} files, {OUT.stat().st_size / 1e6:.2f} MB')


if __name__ == '__main__':
    main()

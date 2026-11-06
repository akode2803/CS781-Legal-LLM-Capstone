"""Lead baseline submission: the first k words of each test document, k from the same length policy as summ_lib.lead_baseline (no model).

    python make_lead.py <Phase_6_Summ dir> <out.zip> [--ratio 0.22 --lo 200 --hi 700 --max-frac 1.2]
"""
import argparse, json, zipfile
from pathlib import Path

if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('phase_dir'); ap.add_argument('out')
    ap.add_argument('--ratio', type=float, default=0.22); ap.add_argument('--lo', type=int, default=200); ap.add_argument('--hi', type=int, default=700); ap.add_argument('--max-frac', type=float, default=1.2)
    a = ap.parse_args()
    test = json.load(open(Path(a.phase_dir) / 'Test/test.json'))
    preds = []
    for r in test:
        w = r['document'].split(); k = int(a.max_frac * min(max(a.ratio * len(w), a.lo), a.hi))      # summ_lib.length_bounds(...)[1]
        preds.append({'id': r['id'], 'summary': ' '.join(w[:k])})
    assert len({p['id'] for p in preds}) == len(test) and all(p['summary'] for p in preds)
    with zipfile.ZipFile(a.out, 'w', zipfile.ZIP_DEFLATED) as z: z.writestr('prediction.json', json.dumps(preds, ensure_ascii=False))
    print(a.out, len(preds), 'summaries, mean', round(sum(len(p['summary'].split()) for p in preds) / len(preds)), 'words')

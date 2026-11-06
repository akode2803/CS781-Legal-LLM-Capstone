"""Blend saved neural scores with the lexical shortlist and write test submissions: score = z(log(1 + lexical)) + w * z(neural), one fixed w for all models.
usage: python fuse_submissions.py <course root> <out dir> [w]   (needs rerank_lib, rr_lib and the scaffold on the path)"""
import json, sys
from pathlib import Path
import numpy as np
import pcr_lib as P, rerank_lib as K

ENTRIES = {'p7-enc-full-log': ['p7-enc-full'], 'p7-gqa-full-log': ['p7-gqa-full'], 'p7-gqa-lorma-log': ['p7-gqa-lorma'], 'p7-gqa-kd-log': ['p7-gqa-kd'],
           'p7-gqa-lora-log': ['p7-gqa-lora'], 'p7-gqa-enc-avg-log': ['p7-gqa-full', 'p7-enc-full']}


def blend(qi, ranked, lex, scores, w, gold=None):
    pred = {}
    for k in qi:
        c = ranked[k]; l = K.zs(np.log1p([max(lex[k][x], 0.) for x in c])); n = K.zs(sum(K.zs([s[k, x] for x in c]) for s in scores))
        pred[k] = [c[j] for j in np.argsort(-(l + w * n), kind='stable')]
    return pred


def load(path):
    return {tuple(k.split('|')): v for k, v in json.load(open(path)).items()}


if __name__ == '__main__':
    root, out, w = Path(sys.argv[1]), Path(sys.argv[2]), float(sys.argv[3]) if len(sys.argv) > 3 else .25
    out.mkdir(exist_ok=True); runs = root / 'runs/phase7_pcr'
    dq, dc = P.load_split(root / 'Phase_7_PCR', 'dev'); gold = {k: v['gold'] or [] for k, v in dq.items()}; di, dr, dl = K.shortlist(dq, dc, 50)
    tq, tc = P.load_split(root / 'Phase_7_PCR', 'test'); ti, tr, tl = K.shortlist(tq, tc, 50)
    for name, members in ENTRIES.items():
        dev = P.micro_prf(blend(di, dr, dl, [load(runs / m / 'dev_scores.json') for m in members], w), gold)
        pred = blend(ti, tr, tl, [load(runs / m / 'test_scores.json') for m in members], w)
        z = P.write_submission({k: v[:7] for k, v in pred.items()}, ti, set(tc), out / f'{name}-w{w}.zip')
        print(name, 'dev', round(dev['f1'], 4), dev['tp'], z)

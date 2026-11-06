"""Merge test_shard_*.json files (written by test_shard.py) into submission.zip; the test ids come from a previous submission zip.
usage: python merge_shards.py <run dir> <reference submission.zip> <out zip>"""
import json, sys, zipfile
from pathlib import Path

run, ref, out = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
ids = [r['id'] for r in json.loads(zipfile.ZipFile(ref).read('prediction.json'))]
got = {}
for f in sorted(run.glob('test_shard_*.json')):
    got.update(json.loads(f.read_text(encoding='utf-8')))
missing = [i for i in ids if not got.get(i, '').strip()]
print(f'{len(ids) - len(missing)} / {len(ids)} test summaries, mean {sum(len(got[i].split()) for i in ids if i in got) / max(1, len(got)):.0f} words')
if missing:
    sys.exit(f'missing {len(missing)}: {missing[:10]}')
pred = [{'id': i, 'summary': got[i]} for i in ids]
with zipfile.ZipFile(out, 'w', zipfile.ZIP_DEFLATED) as z:
    z.writestr('prediction.json', json.dumps(pred, ensure_ascii=False, indent=1))
print('wrote', out)

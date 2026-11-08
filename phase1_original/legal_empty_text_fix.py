"""Narrow, auditable empty-string correction for the original course helper source.

In Colab, upload this readable file outside legal_scaffold, then:
    import legal_empty_text_fix
    legal_empty_text_fix.install(PACKAGE_ROOT)
    import importlib, legal_scaffold.core, legal_scaffold.legal_data
    importlib.reload(legal_scaffold.core)
    importlib.reload(legal_scaffold.legal_data)
    from legal_scaffold.legal_data import prepare_legal
Use a fresh LEGAL_WINDOWS directory; preserve the old partial output and raw files.
"""
from pathlib import Path
import hashlib,json,shutil

VERSION='skip-empty-text-v1'
EXPECTED={'core.py':'c2f1cf7f75a5c492e70581f9dcbfb057474c333d2acf0033bf7f1fd068a96f37',
          'legal_data.py':'19989a39d831739890b784ba0f67c58c09c8aec3f302212e3a6b397611743ac9'}

def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def once(source,old,new):
    if source.count(old)!=1:raise ValueError('original source does not match the narrow correction')
    return source.replace(old,new,1)

def install(root):
    root=Path(root);package=root/'legal_scaffold';manifest_path=root/'source-version.json'
    manifest=json.loads(manifest_path.read_text());parent_sha=digest(manifest_path)
    actual={p.name:digest(p) for p in package.glob('*.py')}
    if actual!=manifest['files']:raise ValueError('source manifest differs; no writes made')
    if any(actual.get(name)!=sha for name,sha in EXPECTED.items()):raise ValueError('requires the unchanged original course helper core/legal_data; no writes made')
    backup=root/'empty-text-original-source'
    if backup.exists():raise FileExistsError('correction backup already exists; do not apply twice')
    core=(package/'core.py').read_text()
    core=once(core,'allow_incomplete_tail=False, status=None):','allow_incomplete_tail=False, status=None, allow_empty_text=False):')
    core=once(core,'Only objects with a nonempty string `text` are accepted.','Only objects with a nonempty string `text` are accepted by default.\n    allow_empty_text=True yields exactly empty strings for a caller-owned audit.\n    Whitespace-only text and other invalid records remain errors.')
    core=once(core,"if not isinstance(record, dict) or not isinstance(record.get('text'), str) or not record['text'].strip():", "if not isinstance(record, dict) or not isinstance(record.get('text'), str) or (not record['text'].strip() and not (allow_empty_text is True and record['text']=='')):")
    data=(package/'legal_data.py').read_text()
    data=once(data,"};duplicates=0;sources={}","};duplicates=0;empty_text_skipped=0;sources={}")
    data=once(data,"(root/'validation.windows.jsonl').open('w') as validation:","(root/'validation.windows.jsonl').open('w') as validation, (root/'excluded-records.jsonl').open('w') as exclusions:")
    data=once(data,'rows=0\n                for ordinal,record in enumerate(iter_records(path,max_record_chars=8_000_000)):\n                    rows+=1;text=record[\'text\'];sha=digest_text(text);split=split_for(text,seed,validation_fraction)',
              "rows=0;source_empty=0\n                for ordinal,record in enumerate(iter_records(path,max_record_chars=8_000_000,allow_empty_text=True)):\n                    rows+=1;text=record['text']\n                    if text=='':\n                        source_empty+=1;empty_text_skipped+=1\n                        exclusions.write(json.dumps({'source':name,'official_path':official_path,'record_index':ordinal,'reason':'empty_text','text_length':0})+'\\n')\n                        continue\n                    sha=digest_text(text);split=split_for(text,seed,validation_fraction)")
    data=once(data,"'records':rows}","'records':rows,'empty_text_skipped':source_empty}")
    data=once(data,"'counts':counts,'exact_duplicates_removed':duplicates,", "'counts':counts,'exact_duplicates_removed':duplicates,'empty_text_skipped':empty_text_skipped,'exclusion_audit':{'file':'excluded-records.jsonl','record_index_base':0,'policy':'skip exactly empty text strings; other invalid text remains fatal'},'preparation_source':{'version':VERSION_PLACEHOLDER,'core_sha256':file_digest(Path(__file__).with_name('core.py')),'legal_data_sha256':file_digest(__file__)},")
    data=data.replace('VERSION_PLACEHOLDER',repr(VERSION))
    changed={'core.py':core,'legal_data.py':data}
    for name,source in changed.items():compile(source,name,'exec')
    manifest['version']+='+'+VERSION
    manifest['empty_text_correction']={'version':VERSION,'parent_source_manifest_sha256':parent_sha,'original_files':EXPECTED,'component_sha256':digest(__file__),'scope':'skip only empty text strings with source/index/reason audit; raw data/model/training unchanged'}
    for name,source in changed.items():manifest['files'][name]=hashlib.sha256(source.encode()).hexdigest()
    backup.mkdir()
    for name in (*changed,'source-version.json'):
        shutil.copyfile(manifest_path if name=='source-version.json' else package/name,backup/name)
    for name,source in changed.items():(package/name).write_text(source)
    manifest_path.write_text(json.dumps(manifest,indent=2)+'\n')
    return {'version':manifest['version'],'parent_source_manifest_sha256':parent_sha,'source_manifest_sha256':digest(manifest_path),'changed_files':list(changed),'backup':str(backup),'next':'reload core and legal_data; use a fresh preparation directory'}

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('source_root');args=parser.parse_args()
    print(json.dumps(install(args.source_root),indent=2))

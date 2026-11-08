"""Exclude only the verified WikiText TRAIN duplicate of heldout The Hustler."""
from pathlib import Path
import hashlib,json,os,shutil,sqlite3,tempfile
from legal_scaffold.core import atomic_json,file_digest,strict_json
from legal_scaffold.wikitext import source_manifest

EXPECTED_COUNTS={'train':(1801350,29444),'validation':(3760,60),'test':(4358,62)}
TITLE='The Hustler ( film )'
TRAIN_INDEX=4078
TEST_INDEX=61
MAX_LINE=12_004_096

def records(path):
    with Path(path).open('rb') as stream:
        while True:
            line=stream.readline(MAX_LINE+1)
            if not line:return
            if len(line)>MAX_LINE:raise ValueError('article exceeds existing safety bound')
            yield line,strict_json(line.decode())

def overlaps(database,column):
    return database.execute(f'SELECT a.split,a.article_index,b.split,b.article_index FROM articles a JOIN articles b ON a.{column}=b.{column} WHERE a.split<b.split').fetchall()

def repair_known_overlap(cache):
    cache=Path(cache);original=cache/'conversion-incomplete';destination=cache/'prepared-full'
    if destination.exists():raise FileExistsError('prepared-full already exists; do not replace it')
    manifest_path=original/'manifest.json';manifest=strict_json(manifest_path.read_text())
    if manifest['provenance']!=source_manifest():raise ValueError('converted source is not the pinned official source')
    if manifest.get('isolation_audit')!='needs_attention' or manifest.get('cross_split_exact_content_overlaps')!=0 or manifest.get('cross_split_title_key_overlaps')!=1:raise ValueError('only the reported single title overlap is repairable')
    if any((manifest['splits'][s]['rows'],manifest['splits'][s]['articles'])!=counts for s,counts in EXPECTED_COUNTS.items()):raise ValueError('converted counts differ from verified runtime evidence')
    for name,expected in manifest['outputs'].items():
        if Path(name).name!=name:raise ValueError('unexpected converted artifact path')
        path=original/name
        if name=='train.articles.jsonl':continue  # Verified during the single filtering pass below.
        if path.stat().st_size!=expected['bytes'] or file_digest(path)!=expected['sha256']:raise ValueError('converted cache differs: '+name)
    with sqlite3.connect(f'file:{original / "article-index.sqlite"}?mode=ro',uri=True) as database:
        pairs=overlaps(database,'title_key')
        if len(pairs)!=1 or {(pairs[0][0],pairs[0][1]),(pairs[0][2],pairs[0][3])}!={('train',TRAIN_INDEX),('test',TEST_INDEX)} or overlaps(database,'content_sha256'):raise ValueError('unexpected overlap; no repair performed')
        indexed=database.execute('SELECT title_key,content_sha256,row_sha256 FROM articles WHERE split=? AND article_index=?',('train',TRAIN_INDEX)).fetchall()
        if len(indexed)!=1:raise ValueError('known TRAIN index is not unique')
    heldout=None
    for _,record in records(original/'test.articles.jsonl'):
        if record['article_index']==TEST_INDEX:
            if heldout is not None:raise ValueError('duplicate heldout article index')
            heldout=record
    if heldout is None or heldout['split']!='test' or heldout['title']!=TITLE or (heldout['row_start'],heldout['row_end_exclusive'],len(heldout['rows']),len(heldout['text']))!=(4303,4358,55,17882):raise ValueError('known heldout article differs')
    staging=Path(tempfile.mkdtemp(prefix='.known-title-repair-',dir=cache))
    try:
        # Preserve the entire converted original and all raw source files; never rerun Parquet conversion.
        for name in manifest['outputs']:
            if name!='train.articles.jsonl':shutil.copyfile(original/name,staging/name)
        original_hash=hashlib.sha256();removed=None;retained=0
        with (staging/'train.articles.jsonl').open('wb') as output:
            for line,record in records(original/'train.articles.jsonl'):
                original_hash.update(line)
                if record['article_index']==TRAIN_INDEX:
                    if removed is not None:raise ValueError('known TRAIN article is not unique')
                    removed=record
                    if record['split']!='train' or record['title']!=TITLE or (record['row_start'],record['row_end_exclusive'],len(record['rows']),len(record['text']))!=(239010,239066,56,17883):raise ValueError('known TRAIN article differs')
                    if record['text']!=heldout['text']+'\n':raise ValueError('known texts differ beyond one terminal newline')
                    for value in (record,heldout):
                        if hashlib.sha256(value['text'].encode()).hexdigest()!=value['document_sha256'] or hashlib.sha256(json.dumps(value['rows'],ensure_ascii=False,separators=(',',':')).encode()).hexdigest()!=value['row_sequence_sha256']:raise ValueError('known article hashes differ')
                    if indexed[0]!=(record['title_key'],record['document_sha256'],record['row_sequence_sha256']):raise ValueError('TRAIN index identity differs')
                    continue
                output.write(line);retained+=1
            output.flush();os.fsync(output.fileno())
        expected=manifest['outputs']['train.articles.jsonl']
        if original_hash.hexdigest()!=expected['sha256'] or (original/'train.articles.jsonl').stat().st_size!=expected['bytes']:raise ValueError('original TRAIN cache changed')
        if removed is None or retained!=EXPECTED_COUNTS['train'][1]-1:raise ValueError('unexpected retained article count')
        with sqlite3.connect(staging/'article-index.sqlite') as database:
            cursor=database.execute('DELETE FROM articles WHERE split=? AND article_index=? AND content_sha256=? AND row_sha256=?',('train',TRAIN_INDEX,removed['document_sha256'],removed['row_sequence_sha256']))
            if cursor.rowcount!=1:raise ValueError('expected one exact TRAIN index exclusion')
            if overlaps(database,'title_key') or overlaps(database,'content_sha256'):raise ValueError('unintended overlap remains; training blocked')
        # Heldout bytes are unchanged. Source rows remain original conversion counts, not filtered rows.
        manifest['splits']['train']['articles']=retained
        manifest['splits']['train']['excluded_articles']=1
        manifest['splits']['train']['excluded_source_rows']=56
        manifest['splits']['train']['rows_semantics']='all original source rows verified during conversion; excluded article recorded separately'
        manifest['train_exclusions']=[{'reason':'known official TRAIN/Test duplicate differing only by one terminal newline',
            'title':TITLE,'train_article_index':TRAIN_INDEX,'train_source_rows':[239010,239066],
            'train_document_sha256':removed['document_sha256'],'retained_heldout_split':'test',
            'heldout_article_index':TEST_INDEX,'heldout_document_sha256':heldout['document_sha256']}]
        manifest['repair_provenance']={'component':'known-wikitext-title-exclusion-v1','original_manifest_sha256':file_digest(manifest_path),
            'original_train_sha256':expected['sha256'],'original_source_counts_preserved':True,
            'raw_sources':'unchanged','heldout_splits':'unchanged','converted_backup':original.name}
        manifest['cross_split_exact_content_overlaps']=0;manifest['cross_split_title_key_overlaps']=0
        manifest['isolation_audit']='passed_exact_checks_only'
        manifest['outputs']={p.name:{'bytes':p.stat().st_size,'sha256':file_digest(p)} for p in sorted(staging.iterdir()) if p.is_file()}
        atomic_json(staging/'manifest.json',manifest)
        for split in ('validation','test'):
            name=split+'.articles.jsonl'
            if manifest['outputs'][name]!=strict_json(manifest_path.read_text())['outputs'][name]:raise ValueError('heldout bytes changed')
        if destination.exists():raise FileExistsError('prepared-full appeared during repair')
        os.rename(staging,destination)
        return {'prepared_directory':str(destination),'excluded_train_articles':1,'retained_train_articles':retained,
                'isolation_audit':manifest['isolation_audit'],'manifest_sha256':file_digest(destination/'manifest.json'),
                'source_code_changed':False,'frozen_tokenizer_changed':False}
    finally:
        if staging.exists():shutil.rmtree(staging)

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('cache');args=parser.parse_args()
    print(json.dumps(repair_known_overlap(args.cache),indent=2))

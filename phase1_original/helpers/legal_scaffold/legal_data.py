"""Four official Phase-1 JSON sources → deduplicated legal TRAIN/validation windows."""
from pathlib import Path
import json,sqlite3
from .core import iter_records,split_for,digest_text,file_digest,atomic_json,strict_json
from .tokenizer_tools import tokenizer_identity

REPO='Exploration-Lab/CS-781-Capstone'
REVISION='89edba6a44d6bb9593fd13b6895b5cd3b06eb82e'
FILES={'acts':('Phase_1_Pretraining/Train/train_acts.json',728801891),
       'high_courts':('Phase_1_Pretraining/Train/train_high_courts.json',10672497795),
       'supreme_court':('Phase_1_Pretraining/Train/train_supreme_court.json',1322567168),
       'tribunals':('Phase_1_Pretraining/Train/train_tribunals.json',6483962193)}
TEST='Phase_1_Pretraining/Test/test.json'
TOKENIZER_SHA='906194848c1632431aac7d186e7c4f49da83cb6cbd480fa222c8472401cbf8ab'

def download_sources(cache,token):
    """Runtime only; direct token argument, no login or credential writes."""
    from huggingface_hub import hf_hub_download
    return {name:Path(hf_hub_download(REPO,path,repo_type='dataset',revision=REVISION,cache_dir=str(cache),token=token)) for name,(path,_) in FILES.items()}

def prepare_legal(paths,tokenizer_directory,output,*,seed='course',validation_fraction=.1,official=True):
    from tokenizers import Tokenizer
    if set(paths)!=set(FILES):raise ValueError('exactly the four named official sources are required')
    identity=tokenizer_identity(tokenizer_directory)
    if official and identity['tokenizer_sha256']!=TOKENIZER_SHA:raise ValueError('use the existing canonical frozen BPE')
    root=Path(output)
    if root.exists():
        manifest=strict_json((root/'manifest.json').read_text())
        if manifest['scope']!=('official Phase1' if official else 'synthetic test only') or manifest['tokenizer']!=identity or manifest['seed']!=seed or manifest['validation_fraction']!=validation_fraction:raise ValueError('prepared legal identity differs')
        for name in FILES:
            if file_digest(paths[name])!=manifest['sources'][name]['sha256']:raise ValueError('legal source changed')
        for name,expected in manifest['outputs'].items():
            if file_digest(root/name)!=expected['sha256']:raise ValueError('prepared legal output changed')
        return manifest
    split_for('',seed,validation_fraction)
    root.mkdir(parents=True)
    counts={s:{'documents':0,'windows':0,'supervised_tokens':0} for s in ('train','validation')};duplicates=0;empty_text_skipped=0;sources={}
    try:
        with sqlite3.connect(root/'document-audit.sqlite') as db, (root/'train.windows.jsonl').open('w') as train,(root/'validation.windows.jsonl').open('w') as validation, (root/'excluded-records.jsonl').open('w') as exclusions:
            db.execute('CREATE TABLE documents (sha256 TEXT PRIMARY KEY, split TEXT, source TEXT, ordinal INTEGER)')
            tokenizer=Tokenizer.from_file(str(Path(tokenizer_directory)/'tokenizer.json'))
            for name,(official_path,size) in FILES.items():
                path=Path(paths[name]);before=path.stat()
                if official and before.st_size!=size:raise ValueError('official file size differs: '+name)
                rows=0;source_empty=0
                for ordinal,record in enumerate(iter_records(path,max_record_chars=8_000_000,allow_empty_text=True)):
                    rows+=1;text=record['text']
                    if text=='':
                        source_empty+=1;empty_text_skipped+=1
                        exclusions.write(json.dumps({'source':name,'official_path':official_path,'record_index':ordinal,'reason':'empty_text','text_length':0})+'\n')
                        continue
                    sha=digest_text(text);split=split_for(text,seed,validation_fraction)
                    previous=db.execute('SELECT split FROM documents WHERE sha256=?',(sha,)).fetchone()
                    if previous:
                        if previous[0]!=split:raise ValueError('exact duplicate crosses legal splits')
                        duplicates+=1;continue
                    db.execute('INSERT INTO documents VALUES(?,?,?,?)',(sha,split,name,ordinal))
                    ids=tokenizer.encode(text,add_special_tokens=False).ids;counts[split]['documents']+=1
                    target=train if split=='train' else validation
                    for offset in range(0,len(ids),1024):
                        window=ids[offset:offset+1025]
                        if len(window)<2:continue
                        target.write(json.dumps({'token_ids':window})+'\n');counts[split]['windows']+=1;counts[split]['supervised_tokens']+=len(window)-1
                after=path.stat()
                if (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns):raise ValueError('source changed during ingestion')
                sources[name]={'official_path':official_path,'bytes':after.st_size,'sha256':file_digest(path),'records':rows,'empty_text_skipped':source_empty}
        if any(not counts[s]['windows'] for s in counts):raise ValueError('both train and validation must contain windows')
        manifest={'schema_version':1,'dataset':REPO,'revision':REVISION,'scope':'official Phase1' if official else 'synthetic test only',
            'sources':sources,'tokenizer':identity,'seed':seed,'validation_fraction':validation_fraction,'split_unit':'whole document',
            'counts':counts,'exact_duplicates_removed':duplicates,'empty_text_skipped':empty_text_skipped,'exclusion_audit':{'file':'excluded-records.jsonl','record_index_base':0,'policy':'skip exactly empty text strings; other invalid text remains fatal'},'preparation_source':{'version':'skip-empty-text-v1','core_sha256':file_digest(Path(__file__).with_name('core.py')),'legal_data_sha256':file_digest(__file__)},'isolation_audit':'passed_exact_checks_only',
            'packing':'document-local 1024 supervised tokens with one-token overlap; no test data ingested',
            'outputs':{p.name:{'bytes':p.stat().st_size,'sha256':file_digest(p)} for p in root.iterdir() if p.is_file()}}
        atomic_json(root/'manifest.json',manifest);return manifest
    except Exception:
        # Preserve partial work for diagnosis; never treat it as a completed cache.
        raise

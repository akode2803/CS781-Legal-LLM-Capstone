"""Runtime-only full pinned WikiText cache and frozen-BPE TRAIN encoding."""
from pathlib import Path
import os,json,shutil,hashlib
from urllib.request import Request,build_opener
from .core import atomic_json,file_digest,strict_json
from .data_readiness import SafeRedirect
from .wikitext import source_manifest,verify_file,prepare_local
from .tokenizer_tools import select_corpus,tokenizer_identity

def fetch_file(root,source,item,*,opener=None):
    """Resume exact immutable HTTP ranges; checksum every completed shard."""
    path=Path(root)/item['path'];path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():verify_file(path,item);return path
    part=path.with_suffix('.part');offset=part.stat().st_size if part.exists() else 0
    if offset>item['bytes']:raise ValueError('partial cache exceeds pinned size')
    if offset<item['bytes']:
        headers={'Accept-Encoding':'identity'}
        if offset:headers['Range']=f'bytes={offset}-{item["bytes"]-1}'
        url=f'https://huggingface.co/datasets/{source["dataset"]}/resolve/{source["revision"]}/{item["path"]}'
        with (opener or build_opener(SafeRedirect())).open(Request(url,headers=headers),timeout=60) as r:
            expected=item['bytes']-offset
            if r.status!=(206 if offset else 200) or r.headers.get('Content-Length')!=str(expected) or r.headers.get('Content-Encoding','identity')!='identity':raise ValueError('server did not honor exact transfer; partial retained')
            if offset and r.headers.get('Content-Range')!=f'bytes {offset}-{item["bytes"]-1}/{item["bytes"]}':raise ValueError('resume range differs')
            with part.open('ab' if offset else 'wb') as out:
                while expected:
                    block=r.read(min(65536,expected))
                    if not block:raise ValueError('interrupted transfer; rerun to resume')
                    out.write(block);expected-=len(block)
                if r.read(1):raise ValueError('oversized transfer')
                out.flush();os.fsync(out.fileno())
    verify_file(part,item);os.replace(part,path);return path

def prepare_full(cache):
    """All official train shards in order; keep validation/test separate."""
    root=Path(cache);root.mkdir(parents=True,exist_ok=True);source=source_manifest()
    for item in source['files']:fetch_file(root/'raw',source,item)
    prepared=root/'prepared-full'
    if prepared.exists():
        manifest=strict_json((prepared/'manifest.json').read_text())
        if manifest['provenance']!=source or any(manifest['splits'][s]['rows']!=source['expected_rows'][s] for s in ('train','validation','test')):raise ValueError('not complete pinned splits')
        for split in ('train','validation','test'):select_corpus(prepared,'wikitext',split)
        return prepared
    temporary=root/'conversion-incomplete'
    # Only owned uncommitted conversion is restartable; completed cache is immutable.
    if (temporary/'manifest.json').exists():
        previous=strict_json((temporary/'manifest.json').read_text())
        if previous['isolation_audit']=='passed_exact_checks_only':
            if previous['provenance']!=source or any(previous['splits'][s]['rows']!=source['expected_rows'][s] for s in ('train','validation','test')):raise ValueError('incomplete converted splits')
            for split in ('train','validation','test'):select_corpus(temporary,'wikitext',split)
            os.rename(temporary,prepared);return prepared
        from .known_wikitext_overlap_repair import repair_known_overlap
        repair_known_overlap(root);return prepared
    if temporary.exists():shutil.rmtree(temporary)
    manifest=prepare_local(root/'raw',temporary)
    if manifest['isolation_audit']!='passed_exact_checks_only':
        from .known_wikitext_overlap_repair import repair_known_overlap
        repair_known_overlap(root);return prepared
    os.rename(temporary,prepared);return prepared

def encode_full(prepared,tokenizer_directory,output,*,commit_every_documents=25):
    """Resumes at committed document byte offsets, never refits or crosses splits."""
    from tokenizers import Tokenizer
    prepared_manifest=strict_json((Path(prepared)/'manifest.json').read_text());pinned=source_manifest()
    if prepared_manifest['provenance']!=pinned or any(prepared_manifest['splits'][s]['rows']!=pinned['expected_rows'][s] for s in ('train','validation','test')):raise ValueError('complete pinned WikiText splits required; pilot is not full TRAIN')
    for split in ('train','validation','test'):select_corpus(prepared,'wikitext',split)
    source,provenance=select_corpus(prepared,'wikitext','train');identity=tokenizer_identity(tokenizer_directory)
    root=Path(output);root.mkdir(parents=True,exist_ok=True);manifest_path=root/'manifest.json'
    run_identity={'source_corpus':provenance,'tokenizer':identity,'packing':'document-local 1024 supervised positions; one-token overlap'}
    path=root/'windows.jsonl';statepath=root/'encoding-progress.json'
    if manifest_path.exists():
        manifest=strict_json(manifest_path.read_text())
        if any(manifest[k]!=v for k,v in run_identity.items()) or file_digest(path)!=manifest['outputs']['windows.jsonl']['sha256']:raise ValueError('completed window cache differs')
        return manifest
    if type(commit_every_documents) is not int or commit_every_documents<1:raise ValueError('positive encoding checkpoint interval required')
    state=strict_json(statepath.read_text()) if statepath.exists() else {'identity':run_identity,'source_offset':0,'output_bytes':0,'documents':0,'encoded_tokens':0,'windows':0,'supervised_tokens':0,'committed_sha256':hashlib.sha256(b'').hexdigest()}
    for key in ('source_offset','output_bytes','documents','encoded_tokens','windows','supervised_tokens'):
        if type(state.get(key)) is not int or state[key]<0:raise ValueError('invalid encoding progress '+key)
    if state['source_offset']>source.stat().st_size:raise ValueError('source offset exceeds TRAIN size')
    if state['identity']!=run_identity:raise ValueError('encoding resume input differs')
    if not path.exists() and state['output_bytes']:raise ValueError('committed output missing')
    tokenizer=Tokenizer.from_file(str(Path(tokenizer_directory)/'tokenizer.json'))
    def commit(stream):
        stream.flush();os.fsync(stream.fileno());state['output_bytes']=stream.tell();state['committed_sha256']=output_hash.hexdigest();atomic_json(statepath,state)
    with source.open('rb') as incoming,path.open('r+b' if path.exists() else 'w+b') as stream:
        if stream.seek(0,2)<state['output_bytes']:raise ValueError('committed encoded cache truncated')
        output_hash=hashlib.sha256();stream.seek(0);remaining=state['output_bytes']
        while remaining:
            chunk=stream.read(min(65536,remaining));output_hash.update(chunk);remaining-=len(chunk)
        if output_hash.hexdigest()!=state['committed_sha256']:raise ValueError('committed encoded cache checksum differs')
        stream.truncate(state['output_bytes']);stream.seek(state['output_bytes']);incoming.seek(state['source_offset'])
        while True:
            line=incoming.readline(12_004_097)
            if not line:break
            if len(line)>12_004_096:raise ValueError('document line exceeds bound')
            record=strict_json(line.decode());text=record['text']
            if record['split']!='train' or hashlib.sha256(text.encode()).hexdigest()!=record['document_sha256'] or len(text.encode())>2_000_000:raise ValueError('TRAIN document mismatch or oversized document')
            ids=tokenizer.encode(text,add_special_tokens=False).ids
            for offset in range(0,len(ids),1024):
                window=ids[offset:offset+1025]
                if len(window)<2:continue
                encoded=(json.dumps({'token_ids':window})+'\n').encode();stream.write(encoded);output_hash.update(encoded);state['windows']+=1;state['supervised_tokens']+=len(window)-1
            state['documents']+=1;state['encoded_tokens']+=len(ids);state['source_offset']=incoming.tell()
            if state['documents']%commit_every_documents==0:commit(stream)
        commit(stream)
    if file_digest(source)!=provenance['sha256']:raise ValueError('TRAIN changed during encoding')
    if not state['windows']:raise ValueError('empty encoded TRAIN')
    manifest=dict(run_identity,schema_version=1,split='train',complete_train=True,counts={k:state[k] for k in ('documents','encoded_tokens','windows','supervised_tokens')},outputs={'windows.jsonl':{'sha256':file_digest(path),'bytes':path.stat().st_size}})
    atomic_json(manifest_path,manifest);return manifest

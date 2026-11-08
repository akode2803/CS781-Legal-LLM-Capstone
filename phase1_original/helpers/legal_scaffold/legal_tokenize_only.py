"""Tokenize raw legal JSON arrays once; bounded local shards, durable resume.

No deduplication or SQLite. Frozen BPE, whole-document hash split, original
1024-supervised-token windows. Storage is little-endian uint16 length + IDs.
"""
from pathlib import Path
from array import array
import bisect,codecs,hashlib,json,os,shutil,struct,sys,tempfile,time
from legal_scaffold.core import DECODER,strict_json,atomic_json,split_for,file_digest
from legal_scaffold.legal_data import FILES,REPO,REVISION,TOKENIZER_SHA
from legal_scaffold.tokenizer_tools import tokenizer_identity
from legal_scaffold.window_file import WindowFile
VERSION='tokenize-only-shards-v1'

def records(path,offset=0):
    """Resume at the byte immediately after a previously completed JSON record."""
    with Path(path).open('rb') as stream:
        stream.seek(offset);decoder=codecs.getincrementaldecoder('utf-8')();buffer='';position=offset;eof=False
        def fill():
            nonlocal buffer,eof
            chunk=stream.read(65536);eof=not chunk;buffer+=decoder.decode(chunk,final=eof)
        def take(n):
            nonlocal buffer,position
            position+=len(buffer[:n].encode('utf-8'));buffer=buffer[n:]
        def need():
            while True:
                n=len(buffer)-len(buffer.lstrip(' \t\r\n'));take(n)
                if buffer or eof:return
                fill()
        need();first=offset==0
        if first:
            if not buffer.startswith('['):raise ValueError('expected JSON array')
            take(1);need()
        while True:
            if not first:
                need()
                if buffer.startswith(']'):take(1);break
                if not buffer.startswith(','):raise ValueError('expected comma or closing bracket')
                take(1);need()
            elif buffer.startswith(']'):take(1);break
            while True:
                try:record,end=DECODER.raw_decode(buffer);break
                except json.JSONDecodeError as error:
                    if eof:raise ValueError('incomplete or malformed JSON record') from error
                    if len(buffer)>8_000_000:raise ValueError('record exceeds character limit')
                    fill()
            if end>8_000_000:raise ValueError('record exceeds character limit')
            if not isinstance(record,dict) or not isinstance(record.get('text'),str) or (record['text']!='' and not record['text'].strip()):raise ValueError('record must contain string text; only exactly empty text may be skipped')
            take(end);first=False;yield record,position
        need()
        if buffer:raise ValueError('trailing content after JSON array')

def encoded(window):
    values=array('H',window)
    if sys.byteorder!='little':values.byteswap()
    return struct.pack('<H',len(window))+values.tobytes()

def decoded(payload):
    values=array('H');values.frombytes(payload)
    if sys.byteorder!='little':values.byteswap()
    return values.tolist()

def prepare(paths,tokenizer_directory,output,*,local_work=None,seed='course',validation_fraction=.1,shard_bytes=32_000_000,commit_seconds=60,official=True,_after_commit=None):
    """Commit whole documents every ~32 MB or 60 s; restart only pending work."""
    from tokenizers import Tokenizer
    if set(paths)!=set(FILES):raise ValueError('supply the four named raw legal sources')
    split_for('',seed,validation_fraction)
    identity=tokenizer_identity(tokenizer_directory)
    if identity['actual_vocab_size']>65536 or (official and identity['tokenizer_sha256']!=TOKENIZER_SHA):raise ValueError('use the existing frozen BPE')
    sources={}
    for name,(official_path,size) in FILES.items():
        stat=Path(paths[name]).stat()
        if official and stat.st_size!=size:raise ValueError('official raw size differs: '+name)
        sources[name]={'official_path':official_path,'bytes':stat.st_size,'mtime_ns':stat.st_mtime_ns}
    run={'dataset':REPO,'revision':REVISION,'sources':sources,'tokenizer':identity,'seed':seed,'validation_fraction':validation_fraction,'deduplication':'disabled; duplicate documents retained','format':'u16le length followed by token IDs','window_supervised_tokens':1024,'version':VERSION}
    root=Path(output);root.mkdir(parents=True,exist_ok=True);progress=root/'progress.json'
    if (root/'manifest.json').exists():
        m=strict_json((root/'manifest.json').read_text())
        if m['identity']!=run:raise ValueError('token cache inputs differ')
        return m
    state=strict_json(progress.read_text()) if progress.exists() else {'identity':run,'source_index':0,'source_offset':0,'next_record':0,'group':0,'counts':{s:{'documents':0,'windows':0,'supervised_tokens':0} for s in ('train','validation')},'source_counts':{n:{'records':0,'empty_text_skipped':0} for n in FILES},'shards':{'train':[],'validation':[]}}
    if state['identity']!=run:raise ValueError('resume source size/mtime or tokenizer/split changed')
    for entries in state['shards'].values():
        for item in entries:
            if (root/item['file']).stat().st_size!=item['bytes']:raise ValueError('committed shard missing or truncated')
    work=Path(local_work or tempfile.gettempdir());work.mkdir(parents=True,exist_ok=True)
    started=time.monotonic();initial_tokens=state['counts']['train']['supervised_tokens']+state['counts']['validation']['supervised_tokens'];names=list(FILES)
    tokenizer=Tokenizer.from_file(str(Path(tokenizer_directory)/'tokenizer.json'))
    with tempfile.TemporaryDirectory(prefix='legal-tokenize-',dir=work) as temporary:
        temporary=Path(temporary);pending={};last_commit=time.monotonic();dirty=False
        def open_group():
            for split in ('train','validation'):
                path=temporary/(split+'.u16');pending[split]={'path':path,'stream':path.open('wb'),'hash':hashlib.sha256(),'windows':0,'tokens':0,'max_length':0,'max_id':0}
        def commit():
            nonlocal last_commit,dirty
            for split,p in pending.items():
                p['stream'].flush();os.fsync(p['stream'].fileno());p['stream'].close()
                if p['windows']:
                    name=f'{split}-{state["group"]:06d}.u16';part=root/(name+'.part')
                    shutil.copyfile(p['path'],part)
                    with part.open('rb') as stream:os.fsync(stream.fileno())
                    os.replace(part,root/name)
                    state['shards'][split].append({'file':name,'bytes':p['path'].stat().st_size,'sha256':p['hash'].hexdigest(),'windows':p['windows'],'supervised_tokens':p['tokens'],'max_length':p['max_length'],'max_token_id':p['max_id']})
            state['group']+=1;atomic_json(progress,state);last_commit=time.monotonic();dirty=False
            total=sum(c['supervised_tokens'] for c in state['counts'].values());elapsed=time.monotonic()-started
            print(json.dumps({'source_index':state['source_index'],'next_record':state['next_record'],'committed_groups':state['group'],'supervised_tokens':total,'invocation_tokens_per_second':(total-initial_tokens)/elapsed if elapsed else None,'elapsed_seconds':elapsed}),flush=True)
            if _after_commit:_after_commit(state)
            open_group()
        open_group()
        try:
            while state['source_index']<len(names):
                name=names[state['source_index']];path=Path(paths[name])
                for record,offset in records(path,state['source_offset']):
                    text=record['text'];state['source_counts'][name]['records']+=1
                    if text=='':state['source_counts'][name]['empty_text_skipped']+=1
                    else:
                        split=split_for(text,seed,validation_fraction);ids=tokenizer.encode(text,add_special_tokens=False).ids;state['counts'][split]['documents']+=1;p=pending[split]
                        for start in range(0,len(ids),1024):
                            window=ids[start:start+1025]
                            if len(window)<2:continue
                            payload=encoded(window);p['stream'].write(payload);p['hash'].update(payload);p['windows']+=1;p['tokens']+=len(window)-1;p['max_length']=max(p['max_length'],len(window));p['max_id']=max(p['max_id'],max(window))
                            state['counts'][split]['windows']+=1;state['counts'][split]['supervised_tokens']+=len(window)-1
                    state['source_offset']=offset;state['next_record']+=1;dirty=True
                    if sum(p['stream'].tell() for p in pending.values())>=shard_bytes or time.monotonic()-last_commit>=commit_seconds:commit()
                stat=path.stat()
                if stat.st_size!=sources[name]['bytes'] or stat.st_mtime_ns!=sources[name]['mtime_ns']:raise ValueError('raw source changed during tokenization')
                state['source_index']+=1;state['source_offset']=0;state['next_record']=0;dirty=True;commit()
            if dirty:commit()
        finally:
            for p in pending.values():p['stream'].close()
    if not state['counts']['train']['windows']:raise ValueError('no training windows')
    manifest={'complete':True,**state,'source_identity_check':'size and mtime; no raw-content hash walk','isolation':'identical text hashes select identical partitions; no deduplication audit'}
    atomic_json(root/'manifest.json',manifest);return manifest

class TokenShards(WindowFile):
    """Existing trainer-compatible windows; one bounded shard cached, no SQLite."""
    def __init__(self,root,split='train'):
        self.root=Path(root);self.token_only_manifest=strict_json((self.root/'manifest.json').read_text())
        if not self.token_only_manifest['complete'] or split not in ('train','validation'):raise ValueError('completed token shards required')
        self.entries=self.token_only_manifest['shards'][split];self.ends=[];count=0
        for item in self.entries:count+=item['windows'];self.ends.append(count)
        self.count=count;self.supervised_tokens=self.token_only_manifest['counts'][split]['supervised_tokens'];self.sha256=hashlib.sha256((file_digest(self.root/'manifest.json')+'\0'+split).encode()).hexdigest();self.cached=-1
    def __len__(self):return self.count
    def check_metadata(self,context_length,vocab_size):
        for item in self.entries:
            if not 2<=item['max_length']<=context_length+1 or item['max_token_id']>=vocab_size:raise ValueError('shard token/window metadata differs')
    def load(self,index):
        if self.cached==index:return
        item=self.entries[index];data=(self.root/item['file']).read_bytes()
        if len(data)!=item['bytes'] or hashlib.sha256(data).hexdigest()!=item['sha256']:raise ValueError('token shard changed')
        offsets=array('Q');position=0
        while position<len(data):
            offsets.append(position);length=struct.unpack_from('<H',data,position)[0]
            if not 2<=length<=1025:raise ValueError('invalid shard window')
            position+=2+2*length
        if position!=len(data) or len(offsets)!=item['windows']:raise ValueError('invalid shard length/count')
        self.data=data;self.offsets=offsets;self.cached=index
    def __getitem__(self,key):
        if isinstance(key,slice):return [self[i] for i in range(*key.indices(self.count))]
        if key<0:key+=self.count
        if not 0<=key<self.count:raise IndexError(key)
        index=bisect.bisect_right(self.ends,key);self.load(index);local=key-(self.ends[index-1] if index else 0);offset=self.offsets[local];length=struct.unpack_from('<H',self.data,offset)[0]
        return decoded(self.data[offset+2:offset+2+2*length])
    def __iter__(self):
        for index,item in enumerate(self.entries):
            self.load(index)
            for offset in self.offsets:
                length=struct.unpack_from('<H',self.data,offset)[0];yield decoded(self.data[offset+2:offset+2+2*length])

def install(root):
    """Only add shard adapter/preflight; original model/optimizer/loss stay intact."""
    root=Path(root);package=root/'legal_scaffold';path=package/'legal_training.py';source=path.read_text()
    if 'token_only_manifest' not in source:
        old="    hasher=hashlib.sha256()\n    for window in windows:hasher.update(json.dumps(window,separators=(',',':')).encode()+b'\\n')\n    data_hash=hasher.hexdigest()"
        new="    if hasattr(windows,'token_only_manifest'):data_hash=windows.sha256\n    else:\n        hasher=hashlib.sha256()\n        for window in windows:hasher.update(json.dumps(window,separators=(',',':')).encode()+b'\\n')\n        data_hash=hasher.hexdigest()"
        if source.count(old)!=1:raise ValueError('original trainer data identity block differs')
        source=source.replace(old,new,1)
        old="    if not windows or any(len(w)<2 or len(w)>config.context_length+1 for w in windows):\n        raise ValueError('windows must contain 2..context_length+1 token IDs')\n    actual_vocab=tokenizer_identity.get('actual_vocab_size',config.vocab_size) if tokenizer_identity else config.vocab_size\n    if any(type(token) is not int or token<0 or token>=min(actual_vocab,config.vocab_size) for w in windows for token in w):raise ValueError('invalid token ID before device transfer')"
        new="    actual_vocab=tokenizer_identity.get('actual_vocab_size',config.vocab_size) if tokenizer_identity else config.vocab_size\n    if hasattr(windows,'token_only_manifest'):\n        if not windows:raise ValueError('no token windows')\n        windows.check_metadata(config.context_length,min(actual_vocab,config.vocab_size))\n    else:\n        if not windows or any(len(w)<2 or len(w)>config.context_length+1 for w in windows):raise ValueError('invalid window length')\n        if any(type(token) is not int or token<0 or token>=min(actual_vocab,config.vocab_size) for w in windows for token in w):raise ValueError('invalid token ID before device transfer')"
        if source.count(old)!=1:raise ValueError('original trainer window preflight differs')
        source=source.replace(old,new,1);shutil.copyfile(path,root/'legal_training_before_tokenize_only.py');path.write_text(source)
    destination=package/'legal_tokenize_only.py'
    if Path(__file__).resolve()!=destination.resolve():shutil.copyfile(__file__,destination)
    manifest_path=root/'source-version.json';manifest=json.loads(manifest_path.read_text())
    if not manifest['version'].endswith('+'+VERSION):manifest['version']+='+'+VERSION
    manifest['files']['legal_training.py']=file_digest(path);manifest['files'][destination.name]=file_digest(destination)
    atomic_json(manifest_path,manifest)
    return {'version':manifest['version'],'scope':'shard input identity/preflight only; model, optimizer and loss unchanged'}

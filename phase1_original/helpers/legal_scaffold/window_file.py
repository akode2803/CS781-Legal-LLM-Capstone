"""Memory-bounded indexed JSONL windows for explicit larger comparison data."""
from pathlib import Path
import sqlite3
from .core import strict_json,file_digest

class WindowFile:
    def __init__(self,path,index,*,vocab_size,context_length):
        self.path=Path(path);self.index=Path(index)
        if self.index.exists():raise FileExistsError('use a new window index path')
        before=file_digest(self.path);tokens=0;count=0
        with sqlite3.connect(self.index) as db,self.path.open('rb') as stream:
            db.execute('CREATE TABLE offsets (ordinal INTEGER PRIMARY KEY, position INTEGER)')
            while True:
                offset=stream.tell();line=stream.readline(2_000_001)
                if not line:break
                if len(line)>2_000_000:raise ValueError('window record exceeds 2 MB bound')
                value=strict_json(line.decode());ids=value.get('token_ids') if isinstance(value,dict) else None
                if not isinstance(ids,list) or not 2<=len(ids)<=context_length+1 or any(type(i) is not int or not 0<=i<vocab_size for i in ids):raise ValueError('invalid training window')
                db.execute('INSERT INTO offsets VALUES (?,?)',(count,offset));count+=1;tokens+=len(ids)-1
        if count==0 or file_digest(self.path)!=before:raise ValueError('empty or changed window input')
        self.count=count;self.supervised_tokens=tokens;self.sha256=before
        stat=self.path.stat();self.signature=(stat.st_size,stat.st_mtime_ns,stat.st_ino)
    def check_unchanged(self):
        stat=self.path.stat()
        if (stat.st_size,stat.st_mtime_ns,stat.st_ino)!=self.signature:raise ValueError('window file changed after indexing')
    def __len__(self):return self.count
    def __getitem__(self,key):
        self.check_unchanged()
        if isinstance(key,slice):start=key.start or 0;stop=key.stop if key.stop is not None else self.count
        else:start=key;stop=key+1
        with sqlite3.connect(self.index) as db,self.path.open('rb') as stream:
            result=[]
            for (offset,) in db.execute('SELECT position FROM offsets WHERE ordinal>=? AND ordinal<? ORDER BY ordinal',(start,stop)):
                stream.seek(offset);result.append(strict_json(stream.readline(2_000_001).decode())['token_ids'])
        if isinstance(key,slice):return result
        if not result:raise IndexError(key)
        return result[0]
    def __iter__(self):
        self.check_unchanged()
        with sqlite3.connect(self.index) as db,self.path.open('rb') as stream:
            for (offset,) in db.execute('SELECT position FROM offsets ORDER BY ordinal'):
                stream.seek(offset);yield strict_json(stream.readline(2_000_001).decode())['token_ids']

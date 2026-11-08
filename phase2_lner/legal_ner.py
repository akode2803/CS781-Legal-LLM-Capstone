"""Phase-2 legal NER utilities. Offsets are Unicode-code-point, end-exclusive."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import hashlib, json, random, zipfile
import torch
from torch import nn

DATASET_REPO = 'Exploration-Lab/CS-781-Capstone'
DATASET_REVISION = '6f754605b4725f8a6b98e4cd09994b2cfe8b9dc6'
DATASET_FILES = (
    'Phase_2_LNER/Train/fold1-text.json',
    'Phase_2_LNER/Train/fold1-gold.json',
    'Phase_2_LNER/Train/fold2-text.json',
    'Phase_2_LNER/Train/fold2-gold.json',
    'Phase_2_LNER/Train/labels.txt',
)

LABELS = ('APP','RESP','A.COUNSEL','R.COUNSEL','JUDGE','WIT','AUTH','COURT','STAT','PREC','DATE','CASENO')

def ensure_training_files(course_root, token, downloader=None):
    """Fetch only the pinned Phase-2 training files, reusing local/cache copies."""
    if not token: raise RuntimeError('HF_TOKEN is not configured in Colab secrets')
    if downloader is None:
        from huggingface_hub import hf_hub_download
        downloader = hf_hub_download
    root=Path(course_root)
    resolved=[]
    for name in DATASET_FILES:
        path=root/name
        if not path.is_file():
            downloader(repo_id=DATASET_REPO,repo_type='dataset',revision=DATASET_REVISION,
                       filename=name,local_dir=str(root),token=token)
        if not path.is_file(): raise FileNotFoundError(f'download did not create {name}')
        resolved.append(path)
    return tuple(resolved)

def strict_load(path):
    def pairs(xs):
        out={}
        for k,v in xs:
            if k in out: raise ValueError(f'duplicate JSON key {k!r}')
            out[k]=v
        return out
    with open(path,encoding='utf-8') as f:
        return json.load(f,object_pairs_hook=pairs,parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))

def load_labels(path):
    got=tuple(x.strip() for x in Path(path).read_text(encoding='utf-8').splitlines() if x.strip())
    if got != LABELS: raise ValueError(f'labels.txt differs: {got!r}')
    return got

def _entities(row):
    raw=next((row[k] for k in ('entities','annotations','labels') if k in row),[])
    out=[]
    for e in raw:
        if isinstance(e,dict):
            a=e.get('start',e.get('start_offset')); b=e.get('end',e.get('end_offset'))
            lab=e.get('label',e.get('type')); surface=e.get('text',e.get('entity_text'))
        else:
            if len(e) not in (3,4): raise ValueError('entity must have 3 or 4 fields')
            a,b,lab=e[:3]; surface=e[3] if len(e)==4 else None
        if type(a) is not int or type(b) is not int or not 0<=a<b or lab not in LABELS: raise ValueError(f'invalid entity {e!r}')
        out.append((a,b,lab,surface))
    return out

def normalize_records(value):
    rows=[]
    iterable=value.items() if isinstance(value,dict) else enumerate(value)
    for key,row in iterable:
        if isinstance(row,str):
            row={'id':str(key),'text':row,'entities':[]}
        if not isinstance(row,dict): raise ValueError('document must be an object')
        doc_id=str(row.get('id',row.get('document_id',key)))
        text=row.get('text',row.get('document'))
        if not doc_id or not isinstance(text,str): raise ValueError('document needs exact id and text')
        ents=_entities(row); evidence=0
        for a,b,lab,surface in ents:
            if b>len(text): raise ValueError(f'{doc_id}: entity outside text')
            if surface is not None:
                if text[a:b] != surface: raise ValueError(f'{doc_id}: end-exclusive Unicode substring mismatch for {(a,b,lab)}')
                evidence+=1
        rows.append({'id':doc_id,'text':text,'entities':[(a,b,l) for a,b,l,_ in ents],
                     'substring_evidence':evidence})
    if len({r['id'] for r in rows})!=len(rows): raise ValueError('duplicate document IDs')
    return rows

def load_fold_pair(text_path,gold_path):
    """Join the official `*-text.json` and `*-gold.json` by exact document ID."""
    texts=strict_load(text_path); gold=strict_load(gold_path)
    if not isinstance(texts,dict) or not isinstance(gold,dict):
        raise ValueError('official fold text and gold files must be JSON objects keyed by document ID')
    if set(texts)!=set(gold):
        raise ValueError(f'fold ID mismatch: missing_gold={len(set(texts)-set(gold))}, extra_gold={len(set(gold)-set(texts))}')
    combined={}
    for doc_id,value in texts.items():
        if isinstance(value,str): text=value
        elif isinstance(value,dict) and isinstance(value.get('text'),str): text=value['text']
        else: raise ValueError(f'{doc_id}: text mapping value must be a string or object with text')
        combined[str(doc_id)]={'id':str(doc_id),'text':text,'entities':gold[doc_id]}
    return normalize_records(combined)

def load_records(path):
    path=Path(path)
    if path.suffix=='.jsonl':
        value=[json.loads(x) for x in path.read_text(encoding='utf-8').splitlines() if x.strip()]
    else: value=strict_load(path)
    return normalize_records(value)

def audit(rows):
    lengths=sorted(len(r['text']) for r in rows); spans=[]; overlaps=nested=nonascii=0
    counts={l:0 for l in LABELS}
    for r in rows:
        es=sorted(r['entities']); nonascii += any(ord(c)>127 for c in r['text'])
        for a,b,l in es: counts[l]+=1; spans.append(b-a)
        for i,x in enumerate(es):
            for y in es[i+1:]:
                if max(x[0],y[0])<min(x[1],y[1]):
                    overlaps+=1; nested += (x[0]<=y[0] and y[1]<=x[1]) or (y[0]<=x[0] and x[1]<=y[1])
    def q(xs,p): return xs[min(len(xs)-1,int((len(xs)-1)*p))] if xs else 0
    evidence=sum(r.get('substring_evidence',0) for r in rows)
    return {'documents':len(rows),'entities':len(spans),'label_counts':counts,'unicode_documents':nonascii,
            'overlapping_pairs':overlaps,'nested_pairs':nested,'document_chars':{x:q(lengths,x) for x in (0,.5,.9,.99,1)},
            'span_chars':{x:q(sorted(spans),x) for x in (0,.5,.9,.99,1)},
            'substring_evidence_entities':evidence,
            'offset_convention':('verified on every entity carrying entity_text: Python Unicode code points, start inclusive, end exclusive'
                                 if evidence else 'not independently proven: annotations contain no entity_text evidence')}

def split_documents(rows,validation_fraction=.15,seed=17):
    train=[]; val=[]
    for r in rows:
        h=int(hashlib.sha256(f'{seed}\0{r["id"]}'.encode()).hexdigest(),16)/2**256
        (val if h<validation_fraction else train).append(r)
    if rows and (not train or not val):
        ordered=sorted(rows,key=lambda r:hashlib.sha256(f'{seed}\0{r["id"]}'.encode()).hexdigest())
        cut=max(1,min(len(rows)-1,round(len(rows)*validation_fraction))); val=ordered[:cut]; train=ordered[cut:]
    return train,val

def tokenize_windows(row,tokenizer,max_tokens=512,stride=384):
    enc=tokenizer.encode(row['text']); ids=enc.ids; offsets=enc.offsets
    if max_tokens<2 or not 0<stride<=max_tokens: raise ValueError('invalid window settings')
    if not ids:
        if row['entities']: raise ValueError(f'{row["id"]}: entities exist but tokenizer produced no tokens')
        return []
    out=[]
    for first in range(0,max(1,len(ids)),stride):
        last=min(len(ids),first+max_tokens)
        if first>=last: break
        left=offsets[first][0]; right=offsets[last-1][1]
        gold=[e for e in row['entities'] if left<=e[0] and e[1]<=right]
        out.append({'document_id':row['id'],'input_ids':ids[first:last],'offsets':offsets[first:last],
                    'text':row['text'],'global_char_start':left,'global_char_end':right,'entities':gold})
        if last==len(ids): break
    return out

def token_edge_boundaries(window):
    """Return exact character boundary -> token-index maps for byte-level BPE.

    Starts and ends may be any character boundary inside their edge tokens,
    because the official gold
    includes lexical substrings such as Board in Boards and Constitution in
    Constitutional, as well as some trailing spaces.
    """
    offsets=window['offsets']; text=window['text']; starts={}; ends={}
    # Byte-level BPE includes leading spaces in word tokens and emits several
    # byte tokens with the same Unicode character offset. Start only at the
    # first token of an offset group and end only at its last token.
    for i,x in enumerate(offsets):
        if i and x==offsets[i-1]: continue
        raw_a=offsets[i][0]
        for a in range(raw_a,offsets[i][1]): starts.setdefault(a,i)
    for j,x in enumerate(offsets):
        if j+1<len(offsets) and x==offsets[j+1]: continue
        raw_b=offsets[j][1]
        for b in range(offsets[j][0]+1,raw_b+1): ends[b]=j
    return starts,ends

def token_span_for_chars(window,a,b):
    starts,ends=token_edge_boundaries(window)
    i=starts.get(a); j=ends.get(b)
    return None if i is None or j is None or i>j else (i,j)

def candidate_spans(window,max_span_tokens=32):
    offsets=window['offsets']; result=[]; gold={(a,b):LABELS.index(l)+1 for a,b,l in window['entities']}
    starts,ends=token_edge_boundaries(window)
    end_by_index={}
    for b,j in ends.items(): end_by_index.setdefault(j,[]).append(b)
    for a,i in starts.items():
        for j in range(i,min(len(offsets),i+max_span_tokens)):
            raw_a=offsets[i][0]; raw_b=offsets[j][1]
            for b in end_by_index.get(j,()):
                if not a<b: continue
                result.append((i,j,a,b,gold.get((a,b),0),a-raw_a,raw_b-b))
    return result

class SpanNER(nn.Module):
    def __init__(self,backbone,width,labels=LABELS):
        super().__init__(); self.backbone=backbone; self.labels=tuple(labels)
        self.span_head=nn.Sequential(nn.Linear(width*2+2,width),nn.GELU(),nn.Linear(width,len(labels)+1))
    def hidden(self,input_ids):
        x=self.backbone.embedding(input_ids)
        for block in self.backbone.blocks: x,_=block(x,use_cache=False)
        return self.backbone.final_norm(x)
    def score_hidden(self,h,span_indices):
        s=span_indices[...,0].long(); e=span_indices[...,1].long()
        batch=torch.arange(h.shape[0],device=h.device)[:,None]
        edge=span_indices[...,2:4].to(h.dtype)/16.0
        return self.span_head(torch.cat((h[batch,s],h[batch,e],edge),-1))
    def forward(self,input_ids,span_indices):
        return self.score_hidden(self.hidden(input_ids),span_indices)

def merge_predictions(items,threshold=0.5):
    best={}
    for doc,a,b,label,score in items:
        if label not in LABELS or not 0<=a<b: continue
        key=(doc,int(a),int(b),label)
        if score>=threshold and score>best.get(key,-1): best[key]=float(score)
    return sorted(((d,a,b,l,s) for (d,a,b,l),s in best.items()),key=lambda x:(x[0],x[1],x[2],x[3]))

def export_prediction(document_ids,predictions,texts,output_dir):
    out={str(i):[] for i in document_ids}
    for d,a,b,l,_ in predictions:
        if d not in out: raise ValueError(f'unknown id {d}')
        if not 0<=a<b<=len(texts[d]): raise ValueError('span outside original text')
        out[d].append([a,b,l,texts[d][a:b]])
    for d in out:
        seen=set(); clean=[]
        for e in out[d]:
            t=tuple(e[:3])
            if t not in seen: seen.add(t); clean.append(e)
        out[d]=clean
    root=Path(output_dir); root.mkdir(parents=True,exist_ok=True)
    pred=root/'prediction.json'; pred.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    archive=root/'prediction.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z: z.write(pred,'prediction.json')
    return pred,archive

def save_training_checkpoint(path,model,optimizer,scheduler,progress,identity,scaler=None):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True); tmp=path.with_suffix(path.suffix+'.tmp')
    torch.save({'schema_version':1,'identity':identity,'progress':progress,'model':model.state_dict(),
                'optimizer':optimizer.state_dict(),'scheduler':None if scheduler is None else scheduler.state_dict(),
                'scaler':None if scaler is None else scaler.state_dict(),
                'python_rng':random.getstate(),'torch_rng':torch.get_rng_state(),
                'cuda_rng':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None},tmp); tmp.replace(path)

def resume_training_checkpoint(path,model,optimizer,scheduler,identity,scaler=None):
    x=torch.load(path,map_location='cpu',weights_only=False)
    if x.get('schema_version')!=1 or x.get('identity')!=identity: raise ValueError('checkpoint identity mismatch')
    model.load_state_dict(x['model']); optimizer.load_state_dict(x['optimizer'])
    if scheduler is not None and x['scheduler'] is not None: scheduler.load_state_dict(x['scheduler'])
    if scaler is not None and x.get('scaler') is not None: scaler.load_state_dict(x['scaler'])
    random.setstate(x['python_rng']); torch.set_rng_state(x['torch_rng'])
    if torch.cuda.is_available() and x.get('cuda_rng') is not None:
        torch.cuda.set_rng_state_all(x['cuda_rng'])
    return x['progress']

def load_inference_checkpoint(path,model,identity):
    """Restore the trained backbone and NER head without optimizer state."""
    x=torch.load(path,map_location='cpu',weights_only=False)
    if x.get('schema_version')!=1 or x.get('identity')!=identity: raise ValueError('checkpoint identity mismatch')
    model.load_state_dict(x['model'],strict=True)
    return x['progress']

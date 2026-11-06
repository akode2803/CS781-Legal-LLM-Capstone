import json, sys, numpy as np, collections
root=sys.argv[1]
for name in ('Train/train','Train/dev','Test/test'):
    raw=json.load(open(f'{root}/Phase_4_CJPE/{name}.json'))
    items=raw if isinstance(raw,list) else list(raw.values())
    print('==',name,type(raw).__name__,len(items),'| keys of first:',list(items[0].keys()),'| types:',{k:type(v).__name__ for k,v in items[0].items()})
    L=np.array([len(d['text']) if isinstance(d['text'],str) else sum(len(s) for s in d['text']) for d in items])
    print('   chars mean/median/p90/max:',int(L.mean()),int(np.median(L)),int(np.percentile(L,90)),int(L.max()),'| ~words mean',int(L.mean()/6))
    if 'label' in items[0]: print('   labels:',collections.Counter(d['label'] for d in items))
    ids=[d['id'] for d in items]; print('   unique ids:',len(set(ids))==len(ids),'| id examples:',ids[:3])
    if name=='Train/train': print('   text start:',repr((items[0]['text'] if isinstance(items[0]['text'],str) else items[0]['text'][0])[:300]))

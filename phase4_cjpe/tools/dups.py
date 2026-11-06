import json, sys, hashlib, collections
root=sys.argv[1]; D={}
for n in ('Train/train','Train/dev','Test/test'): D[n]=json.load(open(f'{root}/Phase_4_CJPE/{n}.json'))
h=lambda t: hashlib.md5(t.encode()).hexdigest()
for n,items in D.items():
    c=collections.Counter(d['id'] for d in items); dup=[k for k,v in c.items() if v>1]
    hs=collections.Counter(h(d['text']) for d in items)
    print(f'{n}: docs {len(items)} | ids appearing >1x: {len(dup)} (rows {sum(c[k] for k in dup)}) | identical-text rows: {sum(v-1 for v in hs.values() if v>1)}')
    if dup and n!='Test/test':
        k=dup[0]; rows=[d for d in items if d['id']==k]; print('   example id',k,'-> labels',[r['label'] for r in rows],'text identical:',len({h(r["text"]) for r in rows})==1,'lengths',[len(r['text']) for r in rows])
    sameid_difflabel=sum(1 for k in dup if len({d['label'] for d in items if d['id']==k})>1) if n!='Test/test' else 0
    if n!='Test/test': print('   duplicate ids whose rows disagree on label:',sameid_difflabel)
H={n:{h(d['text']) for d in items} for n,items in D.items()}
print('exact-text overlap train&dev:',len(H['Train/train']&H['Train/dev']),'| train&test:',len(H['Train/train']&H['Test/test']),'| dev&test:',len(H['Train/dev']&H['Test/test']))
ids={n:{d['id'] for d in items} for n,items in D.items()}
print('id overlap train&dev:',len(ids['Train/train']&ids['Train/dev']),'| train&test:',len(ids['Train/train']&ids['Test/test']),'| dev&test:',len(ids['Train/dev']&ids['Test/test']))
print('test rows: label key present in', sum('label' in d for d in D['Test/test']),'of',len(D['Test/test']),'| non-null:',sum(d.get('label') is not None for d in D['Test/test']))

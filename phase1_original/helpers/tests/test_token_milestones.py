"""Focused synthetic stage/resume checks; no course training or bulk input."""
import json, tempfile, unittest, zipfile
from pathlib import Path
from unittest.mock import patch
import torch
import test_legal_stage as fixtures
TOKENIZER=fixtures.TOKENIZER
from legal_scaffold.legal_training import train, DEFAULT_RUN
from legal_scaffold.legal_prediction import export
from legal_scaffold.snapshots import verify_snapshot
from legal_scaffold.tokenizer_tools import tokenizer_identity
from legal_scaffold.core import strict_json, file_digest, atomic_json
from legal_scaffold.legal_mix import MixedTokenShards
from legal_scaffold.legal_tokenize_only import encoded
from legal_scaffold.legal_data import FILES

class TrackedWindows(MixedTokenShards):
    def __init__(self,*args): super().__init__(*args); self.batches=[]
    def __getitem__(self,key):
        if isinstance(key,slice): self.batches.append((key.start,min(key.stop,len(self))))
        return super().__getitem__(key)

def shard_fixture(root,identity):
    root.mkdir();entries=[];tokens=0;bounds={}
    for group,name in enumerate(FILES):
        amount=group+2;windows=[[group+1]*(amount+1) for _ in range(8)]
        path=root/('train-%06d.u16'%group);path.write_bytes(b''.join(encoded(window) for window in windows))
        entries.append({'file':path.name,'bytes':path.stat().st_size,'sha256':file_digest(path),'windows':8,'supervised_tokens':8*amount,'max_length':amount+1,'max_token_id':group+1})
        tokens+=8*amount;bounds[name]=group+1
    atomic_json(root/'manifest.json',{'complete':True,'group':4,'identity':{'tokenizer':identity},'shards':{'train':entries,'validation':[]},'counts':{'train':{'windows':32,'supervised_tokens':tokens},'validation':{'windows':0,'supervised_tokens':0}}})
    return bounds

def equal_state(test, left, right):
    if isinstance(left, torch.Tensor): torch.testing.assert_close(left,right,atol=0,rtol=0)
    elif isinstance(left,dict):
        test.assertEqual(set(left),set(right))
        for key in left: equal_state(test,left[key],right[key])
    elif isinstance(left,(list,tuple)):
        test.assertEqual(len(left),len(right))
        for a,b in zip(left,right): equal_state(test,a,b)
    else: test.assertEqual(left,right)

class MilestoneTests(unittest.TestCase):
    def test_four_architectures_staged_resume_and_milestone_export(self):
        torch.set_num_threads(1)
        choices={
            'gqa_pre':{},
            'mla_pre':dict(attention='mla',mla_latent=8,mla_query_rank=8,mla_rope_dim=2),
            'mha_pre':dict(attention='mha',kv_heads=2),
            'mha_post':dict(attention='mha',kv_heads=2,norm_placement='post')}
        identity=tokenizer_identity(TOKENIZER)
        cursors=[]
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            bounds=shard_fixture(root/'tokens',identity)
            mixed=MixedTokenShards(root/'tokens',bounds,root/'order')
            seen=set()
            for key in range(len(mixed)):
                name,local,_=mixed.record(key);self.assertNotIn((name,local),seen);seen.add((name,local))
                exposure=mixed.source_tokens(0,key+1)
                self.assertLessEqual(max(exposure.values())-min(exposure.values()),5)
            settings=dict(DEFAULT_RUN,batch_size=2,max_steps=(len(mixed)+1)//2,max_tokens=mixed.supervised_tokens,max_epochs=1,recovery_every_steps=1,recovery_keep=2)
            with self.assertRaisesRegex(ValueError,'missing or unordered'):
                MixedTokenShards(root/'tokens',dict(bounds,high_courts=None),root/'bad-order')
            for name, overrides in choices.items():
                with self.subTest(architecture=name):
                    path=root/name
                    config,parent=fixtures.LegalTests().parent(path/'wiki',overrides)
                    parent_hash=file_digest(parent/'state.pt')
                    common=dict(model_config=config,settings=settings,name='run',stage='legal_continued',tokenizer_identity=identity,tokenizer_directory=TOKENIZER,data_manifest={'tokenization':mixed.token_only_manifest,'training_order':mixed.manifest})
                    tracked=TrackedWindows(root/'tokens',bounds,root/'order');previous=None;milestones=[]
                    for target in (10,20,30):
                        result=train(tracked,checkpoint_root=path/'staged',initial_weights=parent if previous is None else None,resume=previous,target_tokens=target,**common)
                        previous=result['snapshot'];milestones.append(previous)
                        saved=verify_snapshot(previous)
                        self.assertEqual(saved['role'],'milestone')
                        self.assertEqual(saved['progress']['epoch'],0)
                        self.assertEqual(result['source_tokens_seen'],mixed.source_tokens(0,result['progress']['next_window']))
                        self.assertEqual(sum(result['source_tokens_seen'].values()),result['progress']['tokens_seen'])
                        self.assertTrue(result['target_reached'])
                        self.assertEqual(result['overshoot_tokens'],result['progress']['tokens_seen']-target)
                        self.assertLessEqual(result['overshoot_tokens'],result['max_overshoot_tokens'])
                    self.assertEqual(tracked.batches,[(0,2),(2,4),(4,6),(6,8),(8,10)])
                    cursors.append(result['progress'])
                    whole=train(mixed,checkpoint_root=path/'whole',initial_weights=parent,target_tokens=30,**common)
                    self.assertEqual(result['progress'],whole['progress'])
                    left=torch.load(Path(previous)/'state.pt',weights_only=True)
                    right=torch.load(Path(whole['snapshot'])/'state.pt',weights_only=True)
                    equal_state(self,left,right)
                    self.assertEqual({v['step'].item() for v in left['optimizer']['state'].values()},{5})
                    self.assertEqual(file_digest(parent/'state.pt'),parent_hash)
                    # Rerunning an already reached goal reuses its exact snapshot.
                    unchanged=train(tracked,checkpoint_root=path/'staged',resume=previous,target_tokens=30,**common)
                    self.assertEqual(unchanged['snapshot'],previous)
                    self.assertEqual(len(tracked.batches),5)
                    for milestone in milestones: self.assertEqual(verify_snapshot(milestone)['role'],'milestone')
                    if name=='gqa_pre':
                        # Recovery at step one consumes only the remaining second batch.
                        recovery=next((path/'staged').glob('run-recovery-4-*'))
                        pending=TrackedWindows(root/'tokens',bounds,root/'order')
                        continued=train(pending,checkpoint_root=path/'recovery',resume=recovery,target_tokens=30,**common)
                        self.assertEqual(pending.batches,[(8,10)])
                        equal_state(self,left,torch.load(Path(continued['snapshot'])/'state.pt',weights_only=True))
                        test=path/'test.json'
                        test.write_text(json.dumps({'official-A':{'text':' '.join(['prompt']*512)},'official-B':{'text':' '.join(['other']*512)}}))
                        expected_model=torch.load(Path(milestones[0])/'state.pt',weights_only=True)['model']
                        def generate(model,*args):
                            self.assertFalse(model.training)
                            equal_state(self,model.state_dict(),expected_model)
                            return ' '.join(['continuation']*128)
                        with patch('legal_scaffold.legal_prediction.generate_continuation',side_effect=generate):
                            export(milestones[0],test,TOKENIZER,path/'predictions',expected_ids=2)
                        receipt=strict_json((path/'predictions/export-receipt.json').read_text())
                        self.assertEqual(receipt['snapshot_role'],'milestone')
                        self.assertEqual(receipt['legal_supervised_tokens'],14)
                        self.assertFalse(receipt['one_pass_complete'])
                        self.assertEqual(receipt['token_budget']['target_tokens'],10)
                        with zipfile.ZipFile(path/'predictions/submission.zip') as archive:
                            self.assertEqual(archive.namelist(),['prediction.json'])
                        pending.close()
                        with self.assertRaisesRegex(ValueError,'legal stage or token milestone'):
                            export(recovery,test,TOKENIZER,path/'invalid-predictions',expected_ids=2)
                    tracked.close()
            self.assertTrue(all(progress==cursors[0] for progress in cursors))
            mixed.close()

if __name__=='__main__': unittest.main()

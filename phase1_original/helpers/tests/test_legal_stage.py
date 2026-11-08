"""Synthetic integration only: no official legal data or trained course model."""
import json,random,shutil,tempfile,unittest,zipfile
from pathlib import Path
from unittest.mock import patch
import torch
from legal_scaffold.core import file_digest,strict_json,validate_prediction
from legal_scaffold.legal_data import FILES,prepare_legal
from legal_scaffold.legal_training import train,DEFAULT_MODEL,DEFAULT_RUN
from legal_scaffold.legal_prediction import export,generate_continuation
from legal_scaffold.models import ModelConfig,CausalDecoder
from legal_scaffold.snapshots import save_snapshot
from legal_scaffold.tokenizer_tools import tokenizer_identity
from legal_scaffold.window_file import WindowFile
ROOT=Path(__file__).resolve().parents[1]
TOKENIZER=ROOT/'tokenizer'

class LegalTests(unittest.TestCase):
 def setUp(self):torch.set_num_threads(1)
 def sources(self,root):
  paths={};common={'text':'exact synthetic duplicate document with a court order'}
  for number,name in enumerate(FILES):
   path=root/(name+'.json');path.write_text(json.dumps([{'text':f'Document {number}-{i} contains a distinct synthetic legal provision concerning appeal number {i}.'} for i in range(20)]+[common]));paths[name]=path
  return paths
 def parent(self,root,overrides=None):
  config=dict(DEFAULT_MODEL,vocab_size=32768,d_model=16,layers=1,heads=2,kv_heads=1,ffn_hidden=24,context_length=2048)
  config.update(overrides or {})
  torch.manual_seed(8);model=CausalDecoder(ModelConfig(**config));optimizer=torch.optim.AdamW(model.parameters(),lr=DEFAULT_RUN['lr']);scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda _:1)
  for _ in range(3):
   optimizer.zero_grad();model(torch.tensor([[1,2]]),targets=torch.tensor([[2,3]]))['loss'].backward();optimizer.step();scheduler.step()
  state={'model':model.state_dict(),'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),'scaler':{},'python_rng':random.getstate(),'cpu_rng':torch.get_rng_state(),'cuda_rng':[],'torch_version':str(torch.__version__),'runtime':{'cuda_device_names':[]}}
  def writer(path):
   torch.save(state,path/'state.pt');shutil.copyfile(ROOT/'legal_scaffold/models.py',path/'source-models.py');shutil.copytree(TOKENIZER,path/'tokenizer')
  inventory={k:'state.pt' for k in ('model','optimizer','scheduler','scaler','python_rng','framework_cpu_rng')};inventory.update(numpy_rng=None,framework_device_rng=None,data_cursor='progress')
  identity={'model':config,'settings':DEFAULT_RUN,'tokenizer_identity':tokenizer_identity(TOKENIZER),'data_manifest':{'complete_train':True,'scope':'synthetic completed-parent fixture only'}}
  snapshot=save_snapshot(root,'synthetic-parent',identity=identity,stage='wikitext_only',role='stage',progress={'step':3,'tokens_seen':6,'epoch':1,'next_window':0},inventory=inventory,writer=writer,reader=lambda p:True)
  return config,snapshot
 def test_four_source_ingestion_fresh_optimizer_and_exact_resume(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);paths=self.sources(root);manifest=prepare_legal(paths,TOKENIZER,root/'data',official=False)
   self.assertEqual(manifest['exact_duplicates_removed'],3);self.assertEqual(sum(c['documents'] for c in manifest['counts'].values()),81)
   self.assertEqual(manifest,prepare_legal(paths,TOKENIZER,root/'data',official=False))
   windows=WindowFile(root/'data/train.windows.jsonl',root/'index.sqlite',vocab_size=32334,context_length=2048)
   config,parent=self.parent(root/'parent');original=file_digest(parent/'state.pt')
   settings=dict(DEFAULT_RUN,batch_size=2,max_steps=5,max_tokens=windows.supervised_tokens,max_epochs=1,recovery_every_steps=2,recovery_keep=2)
   a=train(windows,model_config=config,settings=settings,checkpoint_root=root/'a',name='run',stage='legal_continued',initial_weights=parent,tokenizer_identity=manifest['tokenizer'],tokenizer_directory=TOKENIZER,data_manifest=manifest)
   recovery=next(p for p in (root/'a').glob('run-recovery-4-*'))
   b=train(windows,model_config=config,settings=settings,checkpoint_root=root/'b',name='run',stage='legal_continued',resume=recovery,tokenizer_identity=manifest['tokenizer'],tokenizer_directory=TOKENIZER,data_manifest=manifest)
   left=torch.load(Path(a['snapshot'])/'state.pt',weights_only=True);right=torch.load(Path(b['snapshot'])/'state.pt',weights_only=True)
   self.assertEqual(a['progress'],b['progress']);self.assertEqual(a['progress']['step'],5)
   self.assertEqual({v['step'].item() for v in left['optimizer']['state'].values()},{5})  # Parent optimizer had three steps.
   for key in left['model']:torch.testing.assert_close(left['model'][key],right['model'][key],atol=0,rtol=0)
   self.assertEqual(file_digest(parent/'state.pt'),original)
   self.assertIn('parent_wikitext',strict_json((Path(b['snapshot'])/'snapshot.json').read_text())['identity'])
 def test_prediction_root_zip_contract_with_mocked_generation(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);config,parent=self.parent(root/'parent');identity=tokenizer_identity(TOKENIZER)
   result=train([[1,2,3],[3,2,1]],model_config=config,settings=dict(DEFAULT_RUN,max_steps=1,max_tokens=4,recovery_every_steps=1),checkpoint_root=root/'legal',name='legal',stage='legal_continued',initial_weights=parent,tokenizer_identity=identity,tokenizer_directory=TOKENIZER,data_manifest={'scope':'synthetic'})
   test=root/'test.json';test.write_text(json.dumps({'official-A':{'text':' '.join(['prompt']*512)},'official-B':{'text':' '.join(['other']*512)}}))
   with patch('legal_scaffold.legal_prediction.generate_continuation',return_value=' '.join(['continuation']*128)):
    receipt=export(result['snapshot'],test,TOKENIZER,root/'out',expected_ids=2)
   self.assertEqual(receipt['records'],2)
   with zipfile.ZipFile(root/'out/submission.zip') as archive:self.assertEqual(archive.namelist(),['prediction.json'])
   prediction=strict_json((root/'out/prediction.json').read_text());self.assertEqual(set(prediction),{'official-A','official-B'})
   self.assertTrue(all(set(v)=={'text'} and 'prompt' not in v['text'] for v in prediction.values()))
   path=root/'out/prediction.json';valid=path.read_text()
   for bad in ('{"official-A":{"text":"x"},"official-A":{"text":"y"}}',json.dumps({'official-A':{'text':' '.join(['x']*128)}}),json.dumps({i:{'text':' '.join(['x']*127)} for i in prediction}),json.dumps({i:{'text':' '.join(['x']*128),'prompt':'extra'} for i in prediction})):
    path.write_text(bad)
    with self.assertRaises(ValueError):validate_prediction(path,root/'out/official-ids.json',field='text')
   path.write_text(valid)
 def test_generator_uses_only_generated_ids_and_never_pads_exhaustion(self):
  class Tokenizer:
   def encode(self,text,**kwargs):return type('Encoded',(),{'ids':[0,0]})()
   def get_vocab_size(self):return 2
   def token_to_id(self,text):return None
   def decode(self,ids,**kwargs):return ' '.join('continuation' if i==1 else 'prompt' for i in ids)
  class Fixture(torch.nn.Module):
   def __init__(self):super().__init__();self.marker=torch.nn.Parameter(torch.zeros(1));self.config=type('Config',(),{'context_length':140})()
   def forward(self,ids,**kwargs):return {'cache':None,'logits':torch.tensor([[[0.,1.]]])}
  with torch.inference_mode():
   text=generate_continuation(Fixture(),Tokenizer(),'prompt',max_new_tokens=130)
   self.assertEqual(text,' '.join(['continuation']*128))
   with self.assertRaises(ValueError):generate_continuation(Fixture(),Tokenizer(),'prompt',max_new_tokens=10)

if __name__=='__main__':unittest.main()

"""Official 512-word prompts → 128-word continuation-only prediction.json/ZIP."""
from pathlib import Path
import contextlib,json,zipfile
import torch
from .models import CausalDecoder,ModelConfig
from .snapshots import verify_snapshot
from .tokenizer_tools import tokenizer_identity
from .core import strict_json,atomic_json,validate_prediction,file_digest

def generate_continuation(model,tokenizer,prompt,max_new_tokens=768):
    ids=tokenizer.encode(prompt,add_special_tokens=False).ids
    room=model.config.context_length-len(ids)
    if not ids or room<1:raise ValueError('512-word prompt cannot fit model context; do not truncate it')
    limit=min(max_new_tokens,room);device=next(model.parameters()).device
    dtype=torch.bfloat16 if device.type=='cuda' and torch.cuda.is_bf16_supported(including_emulation=False) else torch.float16
    current=torch.tensor([ids],device=device);cache=None;continuation=[]
    for _ in range(limit):
        with torch.autocast('cuda',dtype=dtype) if device.type=='cuda' else contextlib.nullcontext():
            result=model(current,cache=cache,use_cache=True)
        cache=result['cache'];logits=result['logits'][:,-1,:tokenizer.get_vocab_size()].clone()
        if not torch.isfinite(logits).all():raise ValueError('nonfinite prediction logits')
        for special in ('<unk>','<pad>','<eos>'):
            index=tokenizer.token_to_id(special)
            if index is not None:logits[:,index]=float('-inf')
        current=logits.argmax(-1,keepdim=True);continuation.append(current.item())
        text=tokenizer.decode(continuation,skip_special_tokens=False)
        if len(text.split())>=129:return ' '.join(text.split()[:128])
    raise ValueError('generation cap/context exhausted before 128 complete continuation words; no fabricated padding')

def export(snapshot,test_path,tokenizer_directory,output,*,expected_ids=1000,max_new_tokens=768):
    if type(max_new_tokens) is not int or max_new_tokens<1:raise ValueError('positive generation cap required')
    root=Path(output)
    if root.exists():raise FileExistsError('use a fresh prediction output directory')
    manifest=verify_snapshot(snapshot)
    if manifest['stage']!='legal_continued' or manifest['role'] not in ('stage','milestone'):raise ValueError('legal stage or token milestone required')
    one_pass_complete=manifest['progress']['epoch']>=1
    if manifest['role']=='stage' and not one_pass_complete:raise ValueError('full legal-stage snapshot must have completed its epoch')
    budget_path=Path(snapshot)/'token-budget.json'
    budget=strict_json(budget_path.read_text()) if budget_path.exists() else None
    if manifest['role']=='milestone':
        if not budget or type(budget.get('target_tokens')) is not int or budget['target_tokens']<=0 or budget.get('target_reached') is not True or budget.get('tokens_seen')!=manifest['progress']['tokens_seen'] or budget['tokens_seen']<budget['target_tokens']:
            raise ValueError('verified reached cumulative-token milestone required')
    identity=tokenizer_identity(tokenizer_directory)
    if identity!=manifest['identity']['tokenizer_identity']:raise ValueError('frozen tokenizer differs')
    if file_digest(Path(snapshot)/'source-models.py')!=file_digest(Path(__file__).with_name('models.py')):raise ValueError('prediction model implementation differs')
    test=strict_json(Path(test_path).read_text())
    if not isinstance(test,dict) or len(test)!=expected_ids:raise ValueError('official exact test mapping/count required')
    for key,value in test.items():
        if not isinstance(key,str) or not key or not isinstance(value,dict) or not isinstance(value.get('text'),str) or len(value['text'].split())!=512:raise ValueError('exact IDs with 512-word text prompts required')
    config=ModelConfig(**manifest['identity']['model']);device='cuda' if torch.cuda.is_available() else 'cpu'
    model=CausalDecoder(config).to(device)
    state=torch.load(Path(snapshot)/'state.pt',map_location='cpu',weights_only=True,mmap=True)
    model.load_state_dict(state['model']);del state;model.eval()
    from tokenizers import Tokenizer
    tokenizer=Tokenizer.from_file(str(Path(tokenizer_directory)/'tokenizer.json'))
    root.mkdir(parents=True);atomic_json(root/'official-ids.json',list(test));predictions={}
    try:
        with torch.inference_mode():
            for number,(identifier,record) in enumerate(test.items(),1):
                predictions[identifier]={'text':generate_continuation(model,tokenizer,record['text'],max_new_tokens)}
                if number%25==0:print(f'Generated {number}/{len(test)} prompts',flush=True)
        atomic_json(root/'prediction.json',predictions)
        validate_prediction(root/'prediction.json',root/'official-ids.json',field='text')
        with zipfile.ZipFile(root/'submission.zip','x',compression=zipfile.ZIP_DEFLATED) as archive:archive.write(root/'prediction.json','prediction.json')
        result=validate_prediction(root/'submission.zip',root/'official-ids.json',field='text')
        atomic_json(root/'export-receipt.json',{'snapshot_manifest_sha256':file_digest(Path(snapshot)/'snapshot.json'),
            'snapshot_role':manifest['role'],'legal_supervised_tokens':manifest['progress']['tokens_seen'],
            'one_pass_complete':one_pass_complete,'token_budget':budget,
            'test_sha256':file_digest(test_path),'tokenizer':identity,'prediction_sha256':file_digest(root/'prediction.json'),'zip_sha256':file_digest(root/'submission.zip'),'validation':result})
        return result
    finally:del model

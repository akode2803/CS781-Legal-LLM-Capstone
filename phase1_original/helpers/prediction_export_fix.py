"""Official 512-word prompts → 128-word continuation-only prediction.json/ZIP."""
from pathlib import Path
import contextlib,json,zipfile
import torch
from legal_scaffold.models import CausalDecoder,ModelConfig
from legal_scaffold.snapshots import verify_snapshot
from legal_scaffold.tokenizer_tools import tokenizer_identity
from legal_scaffold.core import strict_json,atomic_json,validate_prediction,file_digest

VERSION = 'bounded-word-export-v2'
COMPATIBLE_PART_POLICIES = ('bounded-word-export-v1', VERSION)

def boundary_token_ids(tokenizer):
    """Ordinary model vocabulary tokens that carry a word and a boundary."""
    special = {tokenizer.token_to_id(name) for name in ('<unk>', '<pad>', '<eos>')}
    candidates = []
    for token in range(tokenizer.get_vocab_size()):
        if token in special: continue
        piece = tokenizer.decode([token], skip_special_tokens=False)
        if piece.strip() and (piece[0].isspace() or piece[-1].isspace()):
            candidates.append(token)
    return candidates

def generate_continuation(model, tokenizer, prompt, max_new_tokens=None, *, word_boundary_retry=False):
    """Generate 128 complete words, with bounded tokens and correctly rebased cache."""
    ids = tokenizer.encode(prompt, add_special_tokens=False).ids
    context = model.config.context_length
    if not ids or len(ids) > context:
        raise ValueError('The full prompt does not fit the model context; it was not truncated')
    limit = 2 * context if max_new_tokens is None else max_new_tokens
    if type(limit) is not int or limit < 1:
        raise ValueError('positive generation token bound required')
    device = next(model.parameters()).device
    dtype = torch.bfloat16 if device.type == 'cuda' and torch.cuda.is_bf16_supported(including_emulation=False) else torch.float16
    current = torch.tensor([ids], device=device)
    cache = None
    processed = 0
    continuation = []
    history = list(ids)
    text = ''
    rolls = 0
    eos = tokenizer.token_to_id('<eos>')
    boundaries = None
    for step in range(limit):
        if processed + current.shape[1] > context:
            # Cached keys contain RoPE positions. Rebuild a trailing window from
            # position zero; slicing old cached keys would retain wrong positions.
            retain = max(1, context - min(128, context // 2))
            history = history[-retain:]
            current = torch.tensor([history], device=device)
            cache = None
            processed = 0
            rolls += 1
        with torch.autocast('cuda', dtype=dtype) if device.type == 'cuda' else contextlib.nullcontext():
            result = model(current, cache=cache, use_cache=True)
        processed += current.shape[1]
        cache = result['cache']
        logits = result['logits'][:, -1, :tokenizer.get_vocab_size()].clone()
        if not torch.isfinite(logits).all():
            raise ValueError('nonfinite prediction logits')
        if eos is not None and logits.argmax(-1).item() == eos and len(text.split()) >= 128:
            return ' '.join(text.split()[:128])
        for special in ('<unk>', '<pad>', '<eos>'):
            index = tokenizer.token_to_id(special)
            if index is not None:
                logits[:, index] = float('-inf')
        remaining = limit - step
        # Reserve one model-selected word-start token per missing word, plus
        # one to finish word 128. This is an output-length constraint, not
        # fabricated text, temperature tuning, or an unbounded token extension.
        if word_boundary_retry or remaining <= max(1, 129 - len(text.split())):
            if boundaries is None: boundaries = boundary_token_ids(tokenizer)
            if boundaries:
                allowed = torch.zeros_like(logits, dtype=torch.bool)
                allowed[:, boundaries] = True
                logits.masked_fill_(~allowed, float('-inf'))
        if not torch.isfinite(logits).any():
            raise ValueError('no ordinary continuation token is available')
        token = logits.argmax(-1).item()
        continuation.append(token)
        history.append(token)
        history = history[-context:]
        current = torch.tensor([[token]], device=device)
        text = tokenizer.decode(continuation, skip_special_tokens=False)
        words = text.split()
        # The following whitespace or next word proves word 128 is complete.
        if len(words) >= 129 or len(words) == 128 and text and text[-1].isspace():
            return ' '.join(words[:128])
    raise ValueError(f'output token bound exhausted: generated_tokens={limit}, decoded_words={len(text.split())}, prompt_tokens={len(ids)}, context_rolls={rolls}; no fabricated padding')

def export(snapshot,test_path,tokenizer_directory,output,*,expected_ids=1000,max_new_tokens=None,resume=False):
    if max_new_tokens is not None and (type(max_new_tokens) is not int or max_new_tokens<1):raise ValueError('positive generation cap required')
    root=Path(output)
    if root.exists() and not resume:raise FileExistsError('use a fresh output directory or explicitly resume matching export progress')
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
    if file_digest(Path(snapshot)/'source-models.py')!=file_digest(Path(__file__).parent/'legal_scaffold/models.py'):raise ValueError('prediction model implementation differs')
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
    root.mkdir(parents=True,exist_ok=resume)
    cap=2*config.context_length if max_new_tokens is None else max_new_tokens
    export_identity={'snapshot_manifest_sha256':file_digest(Path(snapshot)/'snapshot.json'),
        'test_sha256':file_digest(test_path),'tokenizer':identity,'policy':VERSION,'max_new_tokens':cap}
    progress_path=root/'export-progress.json'
    previous_policy=VERSION
    if progress_path.exists():
        previous=strict_json(progress_path.read_text())
        previous_policy=previous.get('policy')
        compatible=dict(previous,policy=VERSION)
        if not resume or previous_policy not in COMPATIBLE_PART_POLICIES or compatible!=export_identity:raise ValueError('partial export identity differs')
        if strict_json((root/'official-ids.json').read_text())!=list(test):raise ValueError('partial export IDs differ')
    else:
        if resume and any(root.iterdir()):raise ValueError('old export has no resumable progress; preserve it and use a fresh directory')
        atomic_json(progress_path,export_identity);atomic_json(root/'official-ids.json',list(test))
    parts=root/'parts';parts.mkdir(exist_ok=True);predictions={};part_policies={};failures={}
    import hashlib
    for identifier in test:
        part=parts/(hashlib.sha256(identifier.encode()).hexdigest()+'.json')
        if part.exists():
            row=strict_json(part.read_text())
            if set(row) not in ({'id','text'},{'id','text','generation_policy'}) or row['id']!=identifier or not isinstance(row['text'],str) or len(row['text'].split())!=128:raise ValueError('invalid completed continuation part')
            policy=row.get('generation_policy',previous_policy)
            if policy not in COMPATIBLE_PART_POLICIES:raise ValueError('unknown completed continuation policy')
            part_policies[policy]=part_policies.get(policy,0)+1
            predictions[identifier]={'text':row['text']}
    print(f'Reusing {len(predictions)}/{len(test)} completed continuations',flush=True)
    try:
        with torch.inference_mode():
            for number,(identifier,record) in enumerate(test.items(),1):
                if identifier not in predictions:
                    failure_path=root/'failures'/(hashlib.sha256(identifier.encode()).hexdigest()+'.json')
                    try:
                        try:text=generate_continuation(model,tokenizer,record['text'],cap)
                        except ValueError as error:
                            if 'output token bound exhausted' not in str(error):raise
                            print('Retrying word boundaries for Prediction ID '+identifier,flush=True)
                            text=generate_continuation(model,tokenizer,record['text'],cap,word_boundary_retry=True)
                    except ValueError as error:
                        failures[identifier]=str(error)
                        failure_path.parent.mkdir(exist_ok=True)
                        atomic_json(failure_path,{'id':identifier,'error':str(error),**export_identity})
                        atomic_json(root/'export-error.json',{'failed_ids':failures,**export_identity})
                        print('Prediction ID '+identifier+' remains unresolved; continuing other IDs: '+str(error),flush=True)
                        continue
                    part=parts/(hashlib.sha256(identifier.encode()).hexdigest()+'.json')
                    atomic_json(part,{'id':identifier,'text':text,'generation_policy':VERSION})
                    failure_path.unlink(missing_ok=True)
                    part_policies[VERSION]=part_policies.get(VERSION,0)+1
                    predictions[identifier]={'text':text}
                if number%25==0:print(f'Generated {number}/{len(test)} prompts',flush=True)
        if failures:
            raise ValueError('Unresolved prediction IDs: '+', '.join(failures)+f'; {len(predictions)}/{len(test)} completed answers saved. Retry export to continue.')
        (root/'export-error.json').unlink(missing_ok=True)
        atomic_json(root/'prediction.json',predictions)
        validate_prediction(root/'prediction.json',root/'official-ids.json',field='text')
        with zipfile.ZipFile(root/'submission.zip.part','w',compression=zipfile.ZIP_DEFLATED) as archive:archive.write(root/'prediction.json','prediction.json')
        (root/'submission.zip.part').replace(root/'submission.zip')
        result=validate_prediction(root/'submission.zip',root/'official-ids.json',field='text')
        atomic_json(root/'export-receipt.json',{'snapshot_manifest_sha256':file_digest(Path(snapshot)/'snapshot.json'),
            'snapshot_role':manifest['role'],'legal_supervised_tokens':manifest['progress']['tokens_seen'],
            'one_pass_complete':one_pass_complete,'token_budget':budget,
            'generation_policy':export_identity,'part_policies':part_policies,'test_sha256':file_digest(test_path),'tokenizer':identity,'prediction_sha256':file_digest(root/'prediction.json'),'zip_sha256':file_digest(root/'submission.zip'),'validation':result})
        return result
    finally:del model

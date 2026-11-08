"""Bounded fresh-model training; explicit defaults are proposals, not tuned optima."""
from dataclasses import asdict
from pathlib import Path
import contextlib, json, math, random, time, re
import torch
from .models import CausalDecoder, ModelConfig
from .amp_overflow_fix import amp_step,log_skip,patched_identity,compatible_resume
from .snapshots import save_snapshot, verify_snapshot
from .window_file import WindowFile
from .recovery_policy import prune_recovery
from .core import atomic_json
from .source_version import verify_runtime

DEFAULT_MODEL = dict(vocab_size=32768,d_model=512,layers=12,heads=8,kv_heads=2,
    ffn_hidden=2048,context_length=2048,attention='gqa',norm_placement='pre',
    tie_embeddings=True,rope_base=10000,norm_epsilon=1e-6,init_std=0.02)
DEFAULT_RUN = dict(seed=17,batch_size=2,max_steps=100,max_tokens=204800,
    recovery_every_steps=25,max_seconds=600,lr=0.0003,betas=[0.9,0.95],weight_decay=0.1,
    eps=1e-8,clip_grad_norm=1.0,warmup_steps=10)

def train(windows, *, model_config, settings, checkpoint_root, name, stage,
          resume=None, initial_weights=None, tokenizer_identity=None, tokenizer_directory=None, data_manifest=None):
    """Windows are document-local token-ID lists. No downloads or credentials.

    Resume requires identical configuration/data hashes and restores optimizer,
    scaler, scheduler, Python and framework RNG plus the next-window cursor.
    Caller must put checkpoint_root on persistent mounted storage.
    """
    for key in ('batch_size','max_steps','max_tokens','recovery_every_steps'):
        if type(settings.get(key)) is not int or settings[key]<=0:raise ValueError(key+' must be a positive integer')
    if config_tokenizer_required(model_config) and (not tokenizer_identity or not tokenizer_directory):raise ValueError('frozen tokenizer identity required for course-sized vocabulary')
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,95}',name):raise ValueError('safe unique run name required')
    if len(windows)>100 and not isinstance(windows,WindowFile):raise ValueError('pilot cap is 100 materialized windows')
    if type(settings.get('max_seconds',600)) not in (int,float) or not math.isfinite(settings.get('max_seconds',600)) or settings.get('max_seconds',600)<=0:raise ValueError('positive wall-time cap required')
    for key in ('max_epochs','recovery_keep'):
        if type(settings.get(key,1 if key=='max_epochs' else 2)) is not int or settings.get(key,1 if key=='max_epochs' else 2)<(1 if key=='max_epochs' else 2):raise ValueError('invalid '+key)
    if type(settings.get('recovery_every_seconds',900)) not in (int,float) or not math.isfinite(settings.get('recovery_every_seconds',900)) or settings.get('recovery_every_seconds',900)<=0:raise ValueError('positive finite recovery interval required')
    source_version=verify_runtime(Path(__file__).parent.parent)
    if initial_weights:raise ValueError('long-run scratch copies do not initialize from pilot weights')
    if resume and initial_weights: raise ValueError('choose exact resume or stage initialization')
    if stage != 'wikitext_only': raise ValueError('comparison copies support verified WikiText only')
    if stage=='legal_continued' and not (resume or initial_weights):
        raise ValueError('legal continuation requires preserved WikiText weights')
    if settings['max_steps']<=0 or settings['max_tokens']<=0:raise ValueError('bounded positive budgets required')
    import hashlib
    hasher=hashlib.sha256()
    for window in windows:hasher.update(json.dumps(window,separators=(',',':')).encode()+b'\n')
    data_hash=hasher.hexdigest()
    source_hash=hashlib.sha256((Path(__file__).read_bytes()+Path(__file__).with_name('models.py').read_bytes()+Path(__file__).with_name('window_file.py').read_bytes())).hexdigest()
    runtime={'torch':str(torch.__version__),'device':'cuda' if torch.cuda.is_available() else 'cpu','bf16':bool(torch.cuda.is_available() and torch.cuda.is_bf16_supported(including_emulation=False)),'cuda_device_names':[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],'cuda_compute_capabilities':[list(torch.cuda.get_device_capability(i)) for i in range(torch.cuda.device_count())]}
    identity={'model':model_config,'settings':settings,'data_sha256':data_hash,'stage':stage,'tokenizer_identity':tokenizer_identity,'source_sha256':source_hash,'runtime':runtime,'source_version':source_version,'data_manifest':data_manifest}
    identity=patched_identity(identity,__file__)
    if tokenizer_directory:
        from .tokenizer_tools import tokenizer_identity as verify_tokenizer
        if verify_tokenizer(tokenizer_directory)!=tokenizer_identity:raise ValueError('tokenizer files differ')
    config=ModelConfig(**model_config).validate()
    if not windows or any(len(w)<2 or len(w)>config.context_length+1 for w in windows):
        raise ValueError('windows must contain 2..context_length+1 token IDs')
    actual_vocab=tokenizer_identity.get('actual_vocab_size',config.vocab_size) if tokenizer_identity else config.vocab_size
    if any(type(token) is not int or token<0 or token>=min(actual_vocab,config.vocab_size) for w in windows for token in w):raise ValueError('invalid token ID before device transfer')
    random.seed(settings['seed']);torch.manual_seed(settings['seed'])
    device='cuda' if torch.cuda.is_available() else 'cpu'
    print('Native capability and runtime:',json.dumps(runtime),flush=True)
    dtype=torch.bfloat16 if device=='cuda' and torch.cuda.is_bf16_supported(including_emulation=False) else torch.float16
    model=CausalDecoder(config).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=settings['lr'],betas=tuple(settings['betas']),
        weight_decay=settings['weight_decay'],eps=settings['eps'])
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,
        lambda step:min(1.0,(step+1)/max(1,settings['warmup_steps'])))
    scaler=torch.amp.GradScaler('cuda',enabled=device=='cuda' and dtype==torch.float16)
    progress={'step':0,'tokens_seen':0,'epoch':0,'next_window':0}
    def read(path):
        state=torch.load(Path(path)/'state.pt',map_location='cpu',weights_only=True)
        if set(state)!={'model','optimizer','scheduler','scaler','python_rng','cpu_rng','cuda_rng','torch_version','runtime'}:raise ValueError('incomplete state')
        if not state['optimizer']['param_groups'] or not isinstance(state['scheduler'],dict):raise ValueError('invalid optimizer/scheduler')
        if state['cpu_rng'].dtype!=torch.uint8 or state['cpu_rng'].ndim!=1:raise ValueError('invalid CPU RNG')
        test_random=random.Random();test_random.setstate(state['python_rng'])
        with torch.device('meta'):probe=CausalDecoder(config)
        probe.load_state_dict(state['model'],assign=True)
        if config.tie_embeddings:probe.lm_head.weight=probe.embedding.weight
        check_optimizer=torch.optim.AdamW(probe.parameters());check_optimizer.load_state_dict(state['optimizer'])
        check_scheduler=torch.optim.lr_scheduler.LambdaLR(check_optimizer,lambda _:1);check_scheduler.load_state_dict(state['scheduler'])
        check_scaler=torch.amp.GradScaler('cuda',enabled=bool(state['scaler']));check_scaler.load_state_dict(state['scaler'])
        torch.Generator().set_state(state['cpu_rng'])
        if len(state['cuda_rng'])!=len(state['runtime']['cuda_device_names']) or any(v.dtype!=torch.uint8 or v.ndim!=1 or v.numel()==0 for v in state['cuda_rng']):raise ValueError('invalid CUDA RNG inventory')
        return all(torch.isfinite(v).all().item() for v in state['model'].values())
    if initial_weights:
        manifest=verify_snapshot(initial_weights,reader=read)
        if manifest['stage']!='wikitext_only' or manifest['role']!='stage':raise ValueError('preserved WikiText stage required')
        if manifest['identity'].get('tokenizer_identity')!=tokenizer_identity:raise ValueError('frozen tokenizer identity differs')
        if manifest['identity'].get('source_sha256')!=source_hash:raise ValueError('model/training source differs')
        if manifest['identity']['model']!=model_config:raise ValueError('model identity differs')
        model.load_state_dict(torch.load(Path(initial_weights)/'state.pt',map_location=device,weights_only=True)['model'])
    if resume:
        manifest=compatible_resume(resume,identity,read,__file__,checkpoint_root)
        state=torch.load(Path(resume)/'state.pt',map_location=device,weights_only=True)
        model.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler']);scaler.load_state_dict(state['scaler'])
        random.setstate(state['python_rng']);torch.set_rng_state(state['cpu_rng'].cpu())
        if device=='cuda':torch.cuda.set_rng_state_all([v.cpu() for v in state['cuda_rng']])
        progress=manifest['progress']
        if progress['next_window']>=len(windows):raise ValueError('invalid resume cursor')
    inventory={k:'state.pt' for k in ('model','optimizer','scheduler','scaler','python_rng','framework_cpu_rng')}
    inventory.update(numpy_rng=None,framework_device_rng='state.pt' if device=='cuda' else None,data_cursor='progress')
    def writer(path):
        import shutil
        shutil.copyfile(Path(__file__).with_name('amp_overflow_fix.py'),Path(path)/'source-amp_overflow_fix.py')
        shutil.copyfile(Path(__file__).parent.parent/'amp-overflow-compatibility.json',Path(path)/'amp-overflow-compatibility.json')
        shutil.copyfile(Path(__file__).parent.parent/'source-version.json',Path(path)/'source-version.json')
        for filename in tuple(p.name for p in Path(__file__).parent.glob('*.py')):
            shutil.copyfile(Path(__file__).with_name(filename),Path(path)/('source-'+filename))
        if tokenizer_directory:
            import shutil
            shutil.copytree(tokenizer_directory,Path(path)/'tokenizer')
        torch.save({'model':model.state_dict(),'optimizer':optimizer.state_dict(),
            'scheduler':scheduler.state_dict(),'scaler':scaler.state_dict(),'python_rng':random.getstate(),
            'cpu_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all() if device=='cuda' else [],
            'torch_version':str(torch.__version__),'runtime':runtime},Path(path)/'state.pt')
    def checkpoint(role):
        rng=random.getstate();cpu=torch.get_rng_state();gpu=torch.cuda.get_rng_state_all() if device=='cuda' else []
        try:
            result=save_snapshot(checkpoint_root,f'{name}-{role}-{progress["step"]}-'+__import__('uuid').uuid4().hex[:8],identity=identity,
            stage=stage,role=role,progress=progress.copy(),inventory=inventory,writer=writer,reader=read)
            if role=='recovery':prune_recovery(checkpoint_root,name,identity,keep=settings.get('recovery_keep',2))
            return result
        finally:
            random.setstate(rng);torch.set_rng_state(cpu)
            if device=='cuda':torch.cuda.set_rng_state_all(gpu)
    overflow_streak=0;logs=[];model.train();warmup=2;started=None;measured_tokens=0;invocation_steps=0;run_started=time.monotonic();last_recovery=run_started;metric_path=Path(checkpoint_root)/(name+'-metrics.jsonl');Path(checkpoint_root).mkdir(parents=True,exist_ok=True)
    while progress['step']<settings['max_steps']:
        if progress['epoch']>=settings.get('max_epochs',1) or time.monotonic()-run_started>=settings.get('max_seconds',600):break
        cursor=progress['next_window'];batch=windows[cursor:cursor+settings['batch_size']]
        amount=sum(len(w)-1 for w in batch)
        if progress['tokens_seen']+amount>settings['max_tokens']:break
        length=max(len(w)-1 for w in batch)
        x=torch.zeros((len(batch),length),dtype=torch.long,device=device)
        y=torch.full_like(x,-100)
        for row,w in enumerate(batch):
            x[row,:len(w)-1]=torch.tensor(w[:-1],device=device)
            y[row,:len(w)-1]=torch.tensor(w[1:],device=device)
        optimizer.zero_grad(set_to_none=True)
        amp=torch.autocast('cuda',dtype=dtype) if device=='cuda' else contextlib.nullcontext()
        with amp:loss=model(x,targets=y)['loss']
        if not torch.isfinite(loss):raise RuntimeError('nonfinite loss')
        norm,skipped,before,after=amp_step(loss,model,optimizer,scaler,settings['clip_grad_norm'])
        if skipped:
            overflow_streak+=1
            log_skip(checkpoint_root,name,progress,before,after,overflow_streak)
            continue
        overflow_streak=0
        scheduler.step();progress['step']+=1;progress['tokens_seen']+=amount
        progress['next_window']+=len(batch)
        if progress['next_window']==len(windows):progress['epoch']+=1;progress['next_window']=0
        if device=='cuda':torch.cuda.synchronize()
        invocation_steps+=1
        if invocation_steps==warmup:started=time.perf_counter()
        elif started is not None:measured_tokens+=amount
        entry={'step':progress['step'],'loss':float(loss.detach()),'gradient_norm':float(norm),
            'tokens':amount,'tokens_per_second_after_warmup':measured_tokens/(time.perf_counter()-started) if started else None}
        rate=entry['tokens_per_second_after_warmup']
        entry['estimated_remaining_hours']=(settings['max_tokens']-progress['tokens_seen'])/rate/3600 if rate else None
        with metric_path.open('a') as stream:stream.write(json.dumps(entry)+'\n')
        logs.append(entry);logs=logs[-20:]
        if invocation_steps%100==0:print(json.dumps(entry),flush=True)
        if progress['step']%settings['recovery_every_steps']==0 or time.monotonic()-last_recovery>=settings.get('recovery_every_seconds',900):checkpoint('recovery');last_recovery=time.monotonic()
    final=checkpoint('stage' if progress['epoch']>=settings.get('max_epochs',1) else 'recovery')
    atomic_json(Path(checkpoint_root)/(name+'-run-receipt.json'),{'snapshot':str(final),'progress':progress,'source_version':source_version,'one_pass_complete':progress['epoch']>=settings.get('max_epochs',1)})
    return {'snapshot':str(final),'progress':progress,'device':device,'dtype':str(dtype) if device=='cuda' else 'float32','logs':logs}

def config_tokenizer_required(model_config):
    return model_config['vocab_size']>=32768

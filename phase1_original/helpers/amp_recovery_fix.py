"""Bounded numerical recovery around the unchanged, hash-checked legal runner."""
from pathlib import Path
import contextlib, hashlib, json
import torch
from legal_scaffold.amp_overflow_fix import amp_step
from legal_scaffold.source_version import verify_runtime

VERSION = 'bounded-amp-backoff-fp32-v1'
MAX_BACKOFFS = 32
RUNNER_SHA256 = 'b94e07015a82e5c7ad356cbf8666f64411bfae3f9f37ecff9d2830cb91233043'
AMP_SHA256 = '4a27c287f73da1929588f21db606ac1c17fe7326edeecf80b7888968d92ef15e'

class NumericalFailure(RuntimeError):
    pass

def finite_state(model, optimizer):
    tensors = list(model.parameters())
    for state in optimizer.state.values():
        tensors.extend(value for value in state.values() if torch.is_tensor(value))
    return all(torch.isfinite(value).all().item() for value in tensors)

def log_event(root, name, progress, event):
    row = dict(event, optimizer_step=progress['step'], tokens_seen=progress['tokens_seen'],
               next_window=progress['next_window'], policy=VERSION)
    root = Path(root); root.mkdir(parents=True, exist_ok=True)
    with (root / (name + '-amp-overflow.jsonl')).open('a') as stream:
        stream.write(json.dumps(row) + '\n')
    print(json.dumps(row), flush=True)

def run_pending_batch(model, optimizer, scaler, x, y, device, dtype, clip, event):
    """Return only after one finite update; never advance the caller's cursor."""
    backoffs = 0
    for _ in range(MAX_BACKOFFS + 1):
        if scaler.is_enabled() and scaler.get_scale() <= 1.0:
            # Do not halve an already exhausted scale toward zero on later
            # batches; the saved floor continues through exact resume.
            event({'event':'amp_scale_floor_fp32_retry', 'scale':scaler.get_scale()})
            break
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda', dtype=dtype) if device == 'cuda' else contextlib.nullcontext():
            loss = model(x, targets=y)['loss']
        if not torch.isfinite(loss).item():
            event({'event':'amp_nonfinite_loss_fp32_retry'})
            break
        try:
            norm, skipped, before, after = amp_step(loss, model, optimizer, scaler, clip)
        except RuntimeError as error:
            if str(error) == 'nonfinite gradients without active GradScaler':
                event({'event':'amp_nonfinite_gradient_fp32_retry'})
                break
            raise NumericalFailure(str(error)) from error
        if not skipped:
            return loss.detach(), norm, 'amp'
        backoffs += 1
        event({'event':'amp_overflow_skipped', 'scale_before':before, 'scale_after':after,
               'consecutive_skips':backoffs})
        if after <= 1.0 or backoffs >= MAX_BACKOFFS:
            break
    # The model's stored weights stay FP32. Recompute this exact x/y without
    # autocast, and without changing AdamW or replacing the active scaler.
    optimizer.zero_grad(set_to_none=True)
    event({'event':'fp32_same_batch_retry', 'amp_backoffs':backoffs,
           'scale':scaler.get_scale()})
    with torch.autocast('cuda', enabled=False) if device == 'cuda' else contextlib.nullcontext():
        loss = model(x, targets=y)['loss']
    if not torch.isfinite(loss).item():
        raise NumericalFailure('same batch has nonfinite FP32 loss')
    loss.backward()
    if not all(p.grad is None or torch.isfinite(p.grad).all().item() for p in model.parameters()):
        raise NumericalFailure('same batch has nonfinite FP32 gradients')
    if not finite_state(model, optimizer):
        raise NumericalFailure('nonfinite weights or AdamW state before FP32 update')
    try:
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip, error_if_nonfinite=True)
    except RuntimeError as error:
        raise NumericalFailure('nonfinite FP32 gradient norm') from error
    optimizer.step()
    if not finite_state(model, optimizer):
        raise NumericalFailure('FP32 optimizer update produced nonfinite state; existing recoveries were preserved')
    event({'event':'fp32_same_batch_recovered'})
    return loss.detach(), norm, 'fp32_same_batch_recovery'

def build_train(package_root):
    """Patch only reviewed loop/writer anchors in memory; preserve source identity.

    The unchanged source identity governs model/data/optimizer compatibility.
    Every new snapshot additionally records this executable runtime correction.
    """
    root = Path(package_root)
    verify_runtime(root)
    runner = root / 'legal_scaffold/legal_training.py'
    amp = root / 'legal_scaffold/amp_overflow_fix.py'
    if hashlib.sha256(runner.read_bytes()).hexdigest() != RUNNER_SHA256 or hashlib.sha256(amp.read_bytes()).hexdigest() != AMP_SHA256:
        raise ValueError('The legal runner or AMP component differs from the reviewed recovery patch')
    source = runner.read_text()
    old = """        optimizer.zero_grad(set_to_none=True)
        amp=torch.autocast('cuda',dtype=dtype) if device=='cuda' else contextlib.nullcontext()
        with amp:loss=model(x,targets=y)['loss']
        if not torch.isfinite(loss):raise RuntimeError('nonfinite loss')
        norm,skipped,before,after=amp_step(loss,model,optimizer,scaler,settings['clip_grad_norm'])
        if skipped:
            overflow_streak+=1;log_skip(checkpoint_root,name,progress,before,after,overflow_streak)
            continue
        overflow_streak=0"""
    new = """        try:
            loss,norm,precision=_run_pending_batch(model,optimizer,scaler,x,y,device,dtype,settings['clip_grad_norm'],
                lambda event:_log_event(checkpoint_root,name,progress,event))
        except (_NumericalFailure,torch.cuda.OutOfMemoryError) as error:
            optimizer.zero_grad(set_to_none=True)
            if _finite_state(model,optimizer):
                preserved=checkpoint('recovery')
                raise _NumericalFailure(str(error)+'; usable recovery preserved at '+str(preserved)) from error
            raise _NumericalFailure(str(error)+'; existing saved recoveries remain unchanged') from error"""
    writer = "    def writer(path):\n        import shutil"
    updated_writer = writer + "\n        shutil.copyfile(_runtime_patch_path,Path(path)/'source-runtime-amp-recovery.py')\n        atomic_json(Path(path)/'runtime-amp-policy.json',_runtime_policy)"
    metrics = "entry={'step':progress['step'],'loss':float(loss.detach()),'gradient_norm':float(norm),"
    if source.count(old) != 1 or source.count(writer) != 1 or source.count(metrics) != 1:
        raise ValueError('Reviewed legal recovery anchors differ; no patch installed')
    source = source.replace(old, new).replace(writer, updated_writer).replace(metrics, metrics.replace("'step':", "'precision':precision,'step':", 1))
    namespace = {'__name__':'legal_scaffold.legal_training', '__package__':'legal_scaffold', '__file__':str(runner)}
    # __file__ and compile filename retain the original source identity. The
    # separate hashed policy/companion above records the in-memory correction.
    exec(compile(source, str(runner), 'exec'), namespace)
    namespace.update(_run_pending_batch=run_pending_batch, _log_event=log_event,
                     _NumericalFailure=NumericalFailure, _finite_state=finite_state,
                     _runtime_patch_path=Path(__file__),
                     _runtime_policy={'version':VERSION,'max_amp_backoffs':MAX_BACKOFFS,
                         'fp32_retries_per_batch':1,'scale_floor_before_fp32':1.0,
                         'component_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                         'compatible_original_runner_sha256':RUNNER_SHA256,
                         'compatible_original_amp_sha256':AMP_SHA256})
    return namespace['train']

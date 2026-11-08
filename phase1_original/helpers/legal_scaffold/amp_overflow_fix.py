"""Narrow AMP-overflow correction and explicitly checked old-checkpoint resume."""
import hashlib,json,re,shutil
from pathlib import Path
VERSION='amp-overflow-skip-v1'

def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def amp_step(loss,model,optimizer,scaler,clip):
    import torch
    if not torch.isfinite(loss).item():raise RuntimeError('nonfinite loss')
    scaler.scale(loss).backward();scaler.unscale_(optimizer)
    finite=all(p.grad is None or torch.isfinite(p.grad).all().item() for p in model.parameters())
    if not finite and not scaler.is_enabled():raise RuntimeError('nonfinite gradients without active GradScaler')
    norm=torch.nn.utils.clip_grad_norm_(model.parameters(),clip,error_if_nonfinite=True) if finite else None
    before=scaler.get_scale();scaler.step(optimizer);scaler.update();after=scaler.get_scale()
    skipped=after<before
    if not finite and not skipped:raise RuntimeError('nonfinite gradients were not skipped by GradScaler')
    return norm,skipped,before,after

def log_skip(root,name,progress,before,after,streak):
    entry={'event':'amp_overflow_skipped','optimizer_step':progress['step'],'tokens_seen':progress['tokens_seen'],
           'next_window':progress['next_window'],'scale_before':before,'scale_after':after,'consecutive_skips':streak}
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    with (root/(name+'-amp-overflow.jsonl')).open('a') as stream:stream.write(json.dumps(entry)+'\n')
    print(json.dumps(entry),flush=True)
    if streak>=8:raise RuntimeError('eight consecutive AMP overflows on the same pending batch; stop for persistent instability')

def patched_identity(identity,runner):
    policy=json.loads((Path(runner).parent.parent/'amp-overflow-compatibility.json').read_text())
    result=dict(identity);result['amp_overflow_patch']={'version':VERSION,'component_sha256':digest(__file__),
        'compatible_parent_source_sha256':policy['old_training_source_hashes'][Path(runner).name],
        'compatible_parent_source_version':policy['old_source_version']}
    return result

def compatible_resume(directory,identity,reader,runner,checkpoint_root):
    from .snapshots import verify_snapshot
    manifest=verify_snapshot(directory,reader=reader)
    saved=manifest['identity']
    if saved==identity:return manifest
    root=Path(runner).parent.parent
    policy=json.loads((root/'amp-overflow-compatibility.json').read_text())
    module=Path(runner).name
    if policy['patch_version']!=VERSION or policy['helper_sha256']!=digest(__file__):raise ValueError('AMP compatibility policy differs')
    if digest(runner)!=policy['new_module_hashes'][module]:raise ValueError('corrected runner differs from reviewed patch')
    expected=dict(identity);expected.pop('amp_overflow_patch')
    expected['source_sha256']=policy['old_training_source_hashes'][module]
    if 'source_version' in expected:expected['source_version']=policy['old_source_version']
    if saved!=expected:raise ValueError('resume differs beyond the recorded AMP-only source change')
    if Path(checkpoint_root).resolve()==Path(directory).parent.resolve():raise ValueError('AMP-corrected resume requires a new output subdirectory; preserve parent recoveries')
    # The checkpoint stays immutable; new snapshots retain this parent identity in their patch receipt.
    return manifest

def install(root):
    """Apply only known original loop text; fail before writes if it differs."""
    root=Path(root);package=root/'legal_scaffold';helper=package/'amp_overflow_fix.py'
    if helper.exists():raise FileExistsError('AMP patch already installed; do not apply twice')
    old_loop="""        scaler.scale(loss).backward();scaler.unscale_(optimizer)
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),settings['clip_grad_norm'],error_if_nonfinite=True)
        before=scaler.get_scale();scaler.step(optimizer);scaler.update()
        if scaler.get_scale()<before:raise RuntimeError('overflow: optimizer step skipped; stop without advancing cursor')"""
    new_loop="""        norm,skipped,before,after=amp_step(loss,model,optimizer,scaler,settings['clip_grad_norm'])
        if skipped:
            overflow_streak+=1
            log_skip(checkpoint_root,name,progress,before,after,overflow_streak)
            continue
        overflow_streak=0"""
    old_manifest_path=root/'source-version.json'
    old_manifest=json.loads(old_manifest_path.read_text()) if old_manifest_path.exists() else None
    if old_manifest is not None:
        actual={p.name:digest(p) for p in package.glob('*.py')}
        if actual!=old_manifest['files']:raise ValueError('original source manifest differs; do not patch stale source')
    originals={};changes={};source_hashes={}
    for name in ('long_training.py','comparison_training.py','torch_training.py'):
        path=package/name
        if not path.exists():continue
        text=path.read_text()
        if text.count(old_loop)!=1:raise ValueError('original AMP loop differs: '+name)
        extras=(package/'models.py').read_bytes()
        if name!='torch_training.py':extras+=(package/'window_file.py').read_bytes()
        source_hashes[name]=hashlib.sha256(path.read_bytes()+extras).hexdigest()
        new=text.replace(old_loop,new_loop)
        new=new.replace('from .models import CausalDecoder, ModelConfig','from .models import CausalDecoder, ModelConfig\nfrom .amp_overflow_fix import amp_step,log_skip,patched_identity,compatible_resume')
        marker="    if tokenizer_directory:\n"
        if marker not in new:raise ValueError('identity insertion anchor missing')
        new=new.replace(marker,"    identity=patched_identity(identity,__file__)\n"+marker,1)
        anchor="logs=[];model.train();"
        if anchor not in new:raise ValueError('loop initialization anchor missing')
        new=new.replace(anchor,"overflow_streak=0;"+anchor,1)
        resume="manifest=verify_snapshot(resume,expected_identity=identity,reader=read)"
        if new.count(resume)!=1:raise ValueError('resume anchor differs')
        new=new.replace(resume,"manifest=compatible_resume(resume,identity,read,__file__,checkpoint_root)")
        # Store the component with every corrected snapshot as explicit executable provenance.
        copy_anchor="    def writer(path):\n        import shutil"
        if copy_anchor in new:new=new.replace(copy_anchor,copy_anchor+"\n        shutil.copyfile(Path(__file__).with_name('amp_overflow_fix.py'),Path(path)/'source-amp_overflow_fix.py')\n        shutil.copyfile(Path(__file__).parent.parent/'amp-overflow-compatibility.json',Path(path)/'amp-overflow-compatibility.json')",1)
        else:raise ValueError('snapshot companion insertion differs: '+name)
        compile(new,str(path),'exec');originals[name]=text;changes[name]=new
    if not changes:raise ValueError('no supported single-GPU runner found')
    backup=root/'amp-overflow-original-source'
    if backup.exists():raise FileExistsError('source backup already exists')
    backup.mkdir()
    for name,text in originals.items():(backup/name).write_text(text)
    if old_manifest_path.exists():shutil.copyfile(old_manifest_path,backup/'source-version.json')
    shutil.copyfile(__file__,helper)
    for name,text in changes.items():(package/name).write_text(text)
    policy={'patch_version':VERSION,'helper_sha256':digest(helper),
        'old_source_version':{'version':old_manifest['version'],'manifest_sha256':digest(backup/'source-version.json')} if old_manifest else None,
        'old_training_source_hashes':source_hashes,'old_module_hashes':{n:digest(backup/n) for n in originals},
        'new_module_hashes':{n:digest(package/n) for n in changes}}
    (root/'amp-overflow-compatibility.json').write_text(json.dumps(policy,indent=2)+'\n')
    if old_manifest is not None:
        new_manifest=dict(old_manifest,version=old_manifest['version']+'+'+VERSION,
            files={p.name:digest(p) for p in sorted(package.glob('*.py'))},
            runtime_correction={'version':VERSION,'component_sha256':digest(helper),'compatible_parent_manifest_sha256':policy['old_source_version']['manifest_sha256']})
        old_manifest_path.write_text(json.dumps(new_manifest,indent=2)+'\n')
    return policy

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('runtime_root');args=parser.parse_args()
    print(json.dumps(install(args.runtime_root),indent=2))

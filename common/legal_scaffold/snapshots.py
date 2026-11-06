"""Immutable stage snapshots and recovery snapshots around student serializers."""
import os
from pathlib import Path
import re
import shutil
import tempfile
from .core import atomic_json, strict_json, file_digest

COMPONENTS = {'model','optimizer','scheduler','scaler','python_rng','numpy_rng',
              'framework_cpu_rng','framework_device_rng','data_cursor'}


def progress_state(progress):
    if not isinstance(progress,dict):raise ValueError('progress must be a JSON object')
    for key in ('step','tokens_seen','epoch','next_window'):
        if type(progress.get(key)) is not int or progress[key]<0:
            raise ValueError('progress requires nonnegative step, tokens_seen, epoch and next_window')
    return progress


def safe_file(name):
    if not isinstance(name,str):raise ValueError('snapshot artifact must be a string path')
    p=Path(name)
    if not name or p.is_absolute() or '..' in p.parts or '\\' in name:
        raise ValueError('snapshot artifact must be a safe relative path')
    return p


def payload_files(directory):
    files={}
    for path in sorted(Path(directory).rglob('*')):
        if path.is_symlink():raise ValueError('snapshot payload cannot contain symlinks')
        if path.is_file():
            name=path.relative_to(directory).as_posix()
            if name=='snapshot.json':continue
            files[name]={'sha256':file_digest(path),'bytes':path.stat().st_size}
    return files


def validate_inventory(inventory,files):
    if not isinstance(inventory,dict) or set(inventory)!=COMPONENTS:
        raise ValueError('inventory must explicitly declare all model/optimizer/scheduler/scaler/RNG/cursor components')
    if inventory['data_cursor']!='progress':raise ValueError('data_cursor must reference saved progress')
    for key,value in inventory.items():
        if key=='data_cursor':continue
        if value is None:
            if key in ('model','optimizer','python_rng'):raise ValueError(key+' state is required')
        else:
            safe_file(value)
            if files is not None and value not in files:raise ValueError('declared state artifact missing: '+key)


def verify_snapshot(directory, *, expected_identity=None, reader=None):
    directory=Path(directory)
    if directory.is_symlink():raise ValueError('snapshot root cannot be a symlink')
    manifest=strict_json((directory/'snapshot.json').read_text(encoding='utf-8'))
    if (manifest.get('schema_version')!=1 or manifest.get('role') not in ('recovery','stage','milestone') or
            manifest.get('stage') not in ('wikitext_only','legal_continued') or
            (manifest.get('role')=='milestone' and manifest.get('stage')!='legal_continued')):
        raise ValueError('unsupported snapshot manifest')
    progress_state(manifest['progress'])
    validate_inventory(manifest.get('state_inventory'),manifest['artifacts'])
    if expected_identity is not None and manifest['identity']!=expected_identity:
        raise ValueError('snapshot run identity differs; cannot resume')
    for name in manifest['artifacts']:safe_file(name)
    if payload_files(directory)!=manifest['artifacts']:
        raise ValueError('snapshot payload missing, changed or has extra files')
    tokenizer=manifest.get('identity',{}).get('tokenizer')
    if isinstance(tokenizer,dict):
        from .tokenizer_tools import tokenizer_identity
        if tokenizer_identity(directory/'_tokenizer')!=tokenizer:
            raise ValueError('snapshot tokenizer companion differs from run identity')
    if manifest.get('durable_readback_verified') is not True:
        raise ValueError('snapshot was not verified at configured persistence destination')
    manifest_sha256=file_digest(directory/'snapshot.json')
    if reader is not None and reader(directory) is not True:
        raise ValueError('student payload readback failed')
    if payload_files(directory)!=manifest['artifacts'] or file_digest(directory/'snapshot.json')!=manifest_sha256:
        raise ValueError('student readback changed snapshot payload')
    return manifest


def save_snapshot(root,name,*,identity,stage,role,progress,inventory,writer,reader):
    """writer(path) serializes all applicable states; reader(path) verifies restore."""
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,127}',name):raise ValueError('safe unique snapshot name required')
    if stage not in ('wikitext_only','legal_continued') or role not in ('recovery','stage','milestone') or (role=='milestone' and stage!='legal_continued'):
        raise ValueError('explicit stage and recovery/stage/legal milestone role required')
    progress_state(progress)
    validate_inventory(inventory,None)
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    destination=root/name
    if destination.exists():raise FileExistsError('immutable snapshot already exists')
    lock=root/(name+'.lock');lock.mkdir()  # Exclusive cooperating-writer reservation.
    staging=Path(tempfile.mkdtemp(prefix='.'+name+'-',dir=root))
    try:
        writer(staging)
        files=payload_files(staging)
        if not files or (staging/'snapshot.json').exists():raise ValueError('student writer must supply payload, not snapshot.json')
        validate_inventory(inventory,files)
        # Hash every payload, then force file content to storage before publication.
        for name_in_payload in files:
            with (staging/name_in_payload).open('rb') as stream:os.fsync(stream.fileno())
        manifest={'schema_version':1,'identity':identity,'stage':stage,'role':role,
                  'progress':progress,'state_inventory':inventory,'artifacts':files,
                  'durable_readback_verified':False,
                  'persistence_scope':'configured root; external storage durability is the caller responsibility'}
        atomic_json(staging/'snapshot.json',manifest)
        original_manifest_hash=file_digest(staging/'snapshot.json')
        if reader(staging) is not True:raise ValueError('student payload readback failed')
        if payload_files(staging)!=files or file_digest(staging/'snapshot.json')!=original_manifest_hash:
            raise ValueError('readback changed snapshot')
        manifest['durable_readback_verified']=True
        atomic_json(staging/'snapshot.json',manifest)
        for directory in sorted([staging]+[p for p in staging.rglob('*') if p.is_dir()],key=lambda p:len(p.parts),reverse=True):
            descriptor=os.open(directory,os.O_RDONLY)
            try:os.fsync(descriptor)
            finally:os.close(descriptor)
        if destination.exists():raise FileExistsError('immutable snapshot already exists')
        os.rename(staging,destination)
        verify_snapshot(destination,expected_identity=identity,reader=reader)
        descriptor=os.open(root,os.O_RDONLY)
        try:os.fsync(descriptor)
        finally:os.close(descriptor)
        if role=='recovery':
            atomic_json(root/('latest-'+stage+'.json'),{'snapshot':destination.name,'manifest_sha256':file_digest(destination/'snapshot.json')})
        return destination
    finally:
        if staging.exists():shutil.rmtree(staging)
        lock.rmdir()

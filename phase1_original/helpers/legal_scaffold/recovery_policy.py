"""Keep bounded new-run recoveries; never prune old pilots or stage milestones."""
from pathlib import Path
import shutil
from .snapshots import verify_snapshot

def prune_recovery(root,name,identity,*,keep=2):
    if type(keep) is not int or keep<2:raise ValueError('retain at least two verified recoveries')
    candidates=[]
    for path in Path(root).glob(name+'-recovery-*'):
        if not path.is_dir() or path.name.endswith('.lock'):continue
        manifest=verify_snapshot(path)
        if manifest['role']!='recovery' or manifest['identity']!=identity:raise ValueError('refuse to prune unrelated recovery')
        candidates.append((manifest['progress']['step'],path.stat().st_mtime_ns,path))
    for _,_,path in sorted(candidates,reverse=True)[keep:]:shutil.rmtree(path)

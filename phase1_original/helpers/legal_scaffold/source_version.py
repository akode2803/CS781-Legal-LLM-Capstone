"""Explicit reviewed runtime source version, independent of immutable input bundle."""
from pathlib import Path
from .core import file_digest,strict_json

def verify_runtime(root):
    root=Path(root);manifest=strict_json((root/'source-version.json').read_text())
    actual={p.name:file_digest(p) for p in (root/'legal_scaffold').glob('*.py')}
    if actual!=manifest['files']:raise ValueError('runtime source version differs')
    return {'version':manifest['version'],'manifest_sha256':file_digest(root/'source-version.json')}


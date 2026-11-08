import hashlib
import json
import math
import os
from pathlib import Path
import sqlite3
import tempfile
import zipfile


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f'duplicate JSON key: {key!r}')
        result[key] = value
    return result


def reject_constant(value):
    raise ValueError(f'nonfinite JSON constant: {value}')


def finite_float(value):
    result = float(value)
    if not math.isfinite(result):
        raise ValueError('nonfinite JSON number')
    return result


DECODER = json.JSONDecoder(object_pairs_hook=unique_object, parse_constant=reject_constant, parse_float=finite_float)


def strict_json(text):
    return DECODER.decode(text)


def iter_records(path, *, chunk_size=65536, max_record_chars=8_000_000,
                 allow_incomplete_tail=False, status=None, allow_empty_text=False):
    """Stream a top-level JSON array, bounded by a chunk plus one record.

    Limit counts decoded Unicode characters, not bytes. Oversized records fail.
    Only objects with a nonempty string `text` are accepted by default.
    allow_empty_text=True yields exactly empty strings for a caller-owned audit.
    Whitespace-only text and other invalid records remain errors.
    """
    if chunk_size < 1 or max_record_chars < 1:
        raise ValueError('positive reader limits required')
    if status is not None:
        status.update(array_closed=False, incomplete_or_malformed_tail_records=0)
    with open(path, encoding='utf-8') as stream:
        buffer, eof = '', False
        def fill():
            nonlocal buffer, eof
            piece = stream.read(chunk_size)
            buffer += piece
            eof = not piece
        def trim():
            nonlocal buffer
            buffer = buffer.lstrip(' \t\r\n')
        def need():
            trim()
            while not buffer and not eof:
                fill()
                trim()
        need()
        if not buffer.startswith('['):
            raise ValueError('expected JSON array')
        buffer = buffer[1:]
        need()
        if buffer.startswith(']'):
            buffer = buffer[1:]
        else:
            while True:
                need()
                while True:
                    try:
                        record, end = DECODER.raw_decode(buffer)
                        break
                    except json.JSONDecodeError as error:
                        if eof:
                            if allow_incomplete_tail:
                                if status is not None:
                                    status['incomplete_or_malformed_tail_records'] = int(bool(buffer.strip()))
                                return
                            raise ValueError('incomplete or malformed JSON record') from error
                        if len(buffer) > max_record_chars:
                            raise ValueError('record exceeds character limit')
                        fill()
                if end > max_record_chars:
                    raise ValueError('record exceeds character limit')
                if not isinstance(record, dict) or not isinstance(record.get('text'), str) or (not record['text'].strip() and not (allow_empty_text is True and record['text']=='')):
                    raise ValueError('record must contain nonempty text')
                buffer = buffer[end:]
                yield record
                need()
                if not buffer and eof and allow_incomplete_tail:
                    return
                if buffer.startswith(']'):
                    buffer = buffer[1:]
                    break
                if not buffer.startswith(','):
                    raise ValueError('expected comma or closing bracket')
                buffer = buffer[1:]
        if status is not None:
            status['array_closed'] = True
        need()
        if buffer:
            raise ValueError('trailing content after JSON array')


def digest_text(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def split_for(text, seed, validation_fraction):
    if not 0 < validation_fraction < 1:
        raise ValueError('validation fraction must be between zero and one')
    value = int(hashlib.sha256((str(seed) + '\0' + digest_text(text)).encode()).hexdigest(), 16)
    return 'validation' if value < validation_fraction * 2**256 else 'train'


def file_digest(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix='.pending-')
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as stream:
            json.dump(value, stream, ensure_ascii=False, allow_nan=False, indent=2)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def prepare(sources, output, seed='course', validation_fraction=0.1):
    """Write split JSONL, disk-backed duplicate audit, and source manifest."""
    split_for('', seed, validation_fraction)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    manifest = {'schema_version': 1, 'seed': seed, 'validation_fraction': validation_fraction,
                'split_unit': 'whole document text', 'duplicate_policy': 'retain first exact text only',
                'sources': [], 'counts': {'train': 0, 'validation': 0, 'duplicates': 0}}
    with sqlite3.connect(output / 'audit.sqlite') as database, \
            open(output / 'train.jsonl', 'w', encoding='utf-8') as train, \
            open(output / 'validation.jsonl', 'w', encoding='utf-8') as validation:
        database.execute('CREATE TABLE documents (hash TEXT PRIMARY KEY, split TEXT, source TEXT, ordinal INTEGER)')
        for source in sources:
            source = Path(source)
            entry = {'path': str(source.resolve()), 'bytes': source.stat().st_size,
                     'sha256': file_digest(source), 'records': 0}
            for ordinal, record in enumerate(iter_records(source)):
                entry['records'] += 1
                digest = digest_text(record['text'])
                split = split_for(record['text'], seed, validation_fraction)
                existing = database.execute('SELECT split FROM documents WHERE hash=?', (digest,)).fetchone()
                if existing:
                    if existing[0] != split:
                        raise ValueError('exact duplicate crosses splits')
                    manifest['counts']['duplicates'] += 1
                    continue
                database.execute('INSERT INTO documents VALUES (?,?,?,?)', (digest, split, str(source), ordinal))
                target = train if split == 'train' else validation
                target.write(json.dumps({'text': record['text'], 'document_sha256': digest,
                                         'source': str(source), 'ordinal': ordinal}, ensure_ascii=False) + '\n')
                manifest['counts'][split] += 1
            manifest['sources'].append(entry)
            if file_digest(source) != entry['sha256']:
                raise ValueError('source changed during preparation; output has no completed manifest')
    manifest['outputs'] = {name: {'bytes': (output / name).stat().st_size,
                                'sha256': file_digest(output / name)}
                           for name in ('train.jsonl', 'validation.jsonl')}
    atomic_json(output / 'manifest.json', manifest)
    return manifest


def check_leakage(train_path, validation_path, database_path):
    """Exact text overlap audit for independently produced JSON arrays."""
    database_path = Path(database_path)
    if database_path.exists():
        raise ValueError('audit database must be new')
    with sqlite3.connect(database_path) as database:
        database.execute('CREATE TABLE hashes (hash TEXT PRIMARY KEY)')
        for record in iter_records(train_path):
            database.execute('INSERT OR IGNORE INTO hashes VALUES (?)', (digest_text(record['text']),))
        overlap = sum(bool(database.execute('SELECT 1 FROM hashes WHERE hash=?', (digest_text(r['text']),)).fetchone())
                      for r in iter_records(validation_path))
    return {'overlapping_validation_records': overlap}


def save_checkpoint(path, state):
    """JSON-compatible small state only; tensors stay in student-owned adapters."""
    atomic_json(path, {'schema_version': 1, 'state': state})


def load_checkpoint(path):
    value = strict_json(Path(path).read_text(encoding='utf-8'))
    if not isinstance(value, dict) or set(value) != {'schema_version', 'state'} or value['schema_version'] != 1:
        raise ValueError('unsupported checkpoint schema')
    return value['state']


def create_run(path, metadata):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=False)
    atomic_json(path / 'run.json', {'schema_version': 1, 'metadata': metadata})


def log_metrics(path, step, metrics):
    if isinstance(step, bool) or not isinstance(step, int) or step < 0:
        raise ValueError('step must be a nonnegative integer')
    if not isinstance(metrics, dict) or not metrics or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in metrics.values()):
        raise ValueError('metrics must be nonempty finite numbers')
    with open(Path(path) / 'metrics.jsonl', 'a', encoding='utf-8') as stream:
        stream.write(json.dumps({'step': step, 'metrics': metrics}, allow_nan=False) + '\n')


def validate_prediction(path, ids_path, *, field=None, words=128, max_bytes=64_000_000):
    """Validate an ID-to-text mapping or explicit ID-to-single-field mapping.

    IDs file is an authoritative JSON array of strings supplied by the student.
    Continuation semantics cannot be inferred; validator forbids extra fields.
    """
    if words < 1 or max_bytes < 1:
        raise ValueError('positive limits required')
    ids = strict_json(Path(ids_path).read_text(encoding='utf-8'))
    if not isinstance(ids, list) or not ids or any(not isinstance(i, str) or not i for i in ids) or len(set(ids)) != len(ids):
        raise ValueError('IDs must be unique nonempty strings')
    path = Path(path)
    if path.suffix.lower() == '.zip':
        with zipfile.ZipFile(path) as archive:
            entries = archive.infolist()
            if len(entries) != 1 or entries[0].filename != 'prediction.json' or entries[0].is_dir():
                raise ValueError('ZIP must contain only root prediction.json')
            if entries[0].flag_bits & 1 or entries[0].compress_type not in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED):
                raise ValueError('ZIP must be unencrypted and stored or deflated')
            if entries[0].file_size > max_bytes:
                raise ValueError('prediction exceeds size limit')
            with archive.open(entries[0]) as stream:
                data = stream.read(max_bytes + 1)
    else:
        if path.name != 'prediction.json':
            raise ValueError('file must be named prediction.json')
        with open(path, 'rb') as stream:
            data = stream.read(max_bytes + 1)
    if len(data) > max_bytes:
        raise ValueError('prediction exceeds size limit')
    prediction = strict_json(data.decode('utf-8'))
    if not isinstance(prediction, dict) or set(prediction) != set(ids):
        raise ValueError('prediction keys must match all authoritative IDs exactly')
    for identifier, value in prediction.items():
        if field is not None:
            if not isinstance(value, dict) or set(value) != {field}:
                raise ValueError(f'{identifier}: only {field!r} field allowed')
            value = value[field]
        if not isinstance(value, str) or not value.strip() or len(value.split()) != words:
            raise ValueError(f'{identifier}: expected nonempty continuation of exactly {words} whitespace-separated words')
    return {'valid': True, 'records': len(ids), 'words': words, 'field': field}


def record_submission(zip_path, ids_path, ledger, *, attempt, phase, day, experiment,
                      checkpoint, field=None, status='prepared', submission_id=None,
                      timestamp=None, error=None):
    """Preserve an exact ZIP and an immutable local attempt receipt.

    day is manually supplied platform accounting day (YYYY-MM-DD); every receipt
    counts conservatively, regardless of status. No remote action occurs.
    """
    from datetime import date, datetime, timezone
    import re
    if not re.fullmatch(r'[A-Za-z0-9_-]+', attempt) or not phase or not experiment or not checkpoint:
        raise ValueError('safe attempt name and explicit phase/experiment/checkpoint required')
    date.fromisoformat(day)
    if Path(zip_path).suffix.lower() != '.zip':
        raise ValueError('submission artifact must be ZIP')
    ledger = Path(ledger)
    ledger.mkdir(parents=True, exist_ok=True)
    directory = ledger / attempt
    directory.mkdir(exist_ok=False)
    preserved = directory / 'submission.zip'
    try:
        # Copy first, validate the preserved bytes, and hash the very same artifact.
        import shutil
        with open(zip_path, 'rb') as source, open(preserved, 'xb') as target:
            shutil.copyfileobj(source, target, 1024 * 1024)
        try:
            validation = validate_prediction(preserved, ids_path, field=field)
        except (ValueError, OSError, zipfile.BadZipFile) as validation_error:
            validation = {'valid': False, 'error': str(validation_error), 'field': field, 'words': 128}
        receipt = {'schema_version': 1, 'attempt': attempt, 'phase': phase, 'day': day,
                   'experiment': experiment, 'checkpoint': checkpoint,
                   'artifact_sha256': file_digest(preserved), 'artifact_bytes': preserved.stat().st_size,
                   'ids_sha256': file_digest(ids_path), 'validation': validation,
                   'timestamp': timestamp or datetime.now(timezone.utc).isoformat(),
                   'status': status, 'submission_id': submission_id, 'error': error}
        atomic_json(directory / 'receipt.json', receipt)
        preserved.chmod(0o444)
        (directory / 'receipt.json').chmod(0o444)
        return receipt
    except Exception:
        import shutil
        shutil.rmtree(directory)
        raise


def update_submission(ledger, attempt, *, status, submission_id=None, timestamp=None, error=None):
    """Append manual platform events; never rewrite original artifact/receipt."""
    from datetime import datetime, timezone
    import re
    if not re.fullmatch(r'[A-Za-z0-9_-]+', attempt) or not status:
        raise ValueError('safe attempt name and explicit status required')
    directory = Path(ledger) / attempt
    receipt = strict_json((directory / 'receipt.json').read_text(encoding='utf-8'))
    if file_digest(directory / 'submission.zip') != receipt['artifact_sha256']:
        raise ValueError('preserved artifact checksum mismatch')
    with open(directory / 'events.jsonl', 'a', encoding='utf-8') as stream:
        stream.write(json.dumps({'status': status, 'submission_id': submission_id,
                                 'timestamp': timestamp or datetime.now(timezone.utc).isoformat(),
                                 'error': error}, allow_nan=False) + '\n')


def submission_summary(ledger, phase, day):
    receipts = []
    for path in sorted(Path(ledger).glob('*/receipt.json')):
        receipt = strict_json(path.read_text(encoding='utf-8'))
        if file_digest(path.parent / 'submission.zip') != receipt['artifact_sha256']:
            raise ValueError(f'preserved artifact checksum mismatch: {path.parent.name}')
        events = path.parent / 'events.jsonl'
        status = receipt['status']
        if events.exists():
            for line in events.read_text(encoding='utf-8').splitlines():
                status = strict_json(line)['status']
        receipts.append((receipt, status))
    selected = [(r, s) for r, s in receipts if r['phase'] == phase]
    statuses = {}
    for _, status in selected:
        statuses[status] = statuses.get(status, 0) + 1
    daily = sum(r['day'] == day for r, _ in receipts)
    return {'phase': phase, 'day': day, 'phase_recorded_attempts': len(selected),
            'day_recorded_attempts_all_phases': daily, 'phase_status_counts': statuses,
            'phase_cap': 40, 'daily_cap': 10,
            'at_or_above_cap': len(selected) >= 40 or daily >= 10,
            'counting_policy': 'all recorded receipts count, including prepared and failed; reconcile manually with platform'}

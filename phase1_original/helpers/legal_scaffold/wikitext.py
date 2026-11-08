"""WikiText ancillary row/article ingestion. No model, training, or auth."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import sqlite3
import unicodedata
from urllib import request, error
from .core import atomic_json, file_digest, strict_json
from .data_readiness import SafeRedirect, DownloadFailure, http_diagnostics

ROOT = Path(__file__).resolve().parents[1]
SOURCE_MANIFEST = ROOT / 'references/wikitext/source-manifest.json'
SPLITS = ('train', 'validation', 'test')
HEADING = re.compile(r'^=\s+([^=\r\n][^\r\n]*?)\s+=$')


def source_manifest():
    value = strict_json(SOURCE_MANIFEST.read_text(encoding='utf-8'))
    notice = ROOT / value['notice']['local_path']
    if notice.stat().st_size != value['notice']['bytes'] or file_digest(notice) != value['notice']['sha256']:
        raise ValueError('upstream source notice checksum mismatch')
    return value


def article_title(text):
    if not isinstance(text, str):
        raise ValueError('row text must be a string, including empty blank rows')
    match = HEADING.fullmatch(text.strip(' \t\r\n'))
    return match.group(1).strip(' \t') if match and match.group(1).strip(' \t') else None


def title_key(title):
    # Conservative aid only: not a Wikipedia page-ID or complete deduplicator.
    return ' '.join(unicodedata.normalize('NFC', title).replace('_', ' ').casefold().split())


def iter_articles(rows, *, split, complete_split=False, max_row_chars=128_000, max_article_chars=2_000_000):
    """Preserve row order/split. At a bounded prefix EOF discard open article.

    Complete split EOF may emit the last article only when explicitly verified.
    Adjacent source shards must be fed as one uninterrupted iterator per split.
    """
    if split not in SPLITS or max_row_chars < 1 or max_article_chars < 1:
        raise ValueError('official split and positive safety bounds required')
    texts, title, start, chars, article_index = [], None, None, 0, 0
    def emit(end):
        raw = list(texts)
        text = ''.join(row if row.endswith('\n') else row + '\n' for row in raw)
        return {'split': split, 'article_index': article_index, 'title': title,
                'title_key': title_key(title), 'row_start': start, 'row_end_exclusive': end,
                'rows': raw, 'text': text,
                'document_sha256': hashlib.sha256(text.encode('utf-8')).hexdigest(),
                'row_sequence_sha256': hashlib.sha256(json.dumps(raw, ensure_ascii=False, separators=(',', ':')).encode('utf-8')).hexdigest()}
    count = 0
    for ordinal, record in enumerate(rows):
        count = ordinal + 1
        if not isinstance(record, dict) or set(record) != {'text'} or not isinstance(record['text'], str):
            raise ValueError('expected WikiText row object with only string text')
        text = record['text']
        if '\n' in text.rstrip('\r\n') or '\r' in text.rstrip('\r\n'):
            raise ValueError('multiple source lines in one row; row format unverified')
        if len(text) > max_row_chars:
            raise ValueError('WikiText row exceeds configured character bound')
        heading = article_title(text)
        if heading is not None:
            if title is not None:
                yield emit(ordinal)
                article_index += 1
            texts, title, start, chars = [], heading, ordinal, 0
        elif title is None:
            if text.strip():
                raise ValueError('nonblank content precedes an article boundary; prefix or schema unverified')
            continue
        chars += len(text) + 1
        if chars > max_article_chars:
            raise ValueError('WikiText article exceeds configured character bound; never truncate')
        texts.append(text)
    if title is not None and complete_split:
        yield emit(count)


def iter_jsonl_rows(path, *, row_limit=128, max_line_bytes=512_000):
    """Local inspection only; accept blank text rows, never read full input."""
    if not 1 <= row_limit <= 10_000 or not 1 <= max_line_bytes <= 2_000_000:
        raise ValueError('inspection row/line bounds outside allowed range')
    with open(path, 'rb') as stream:
        for _ in range(row_limit):
            line = stream.readline(max_line_bytes + 1)
            if not line:
                return
            if len(line) > max_line_bytes:
                raise ValueError('JSONL row exceeds byte bound')
            value = strict_json(line.decode('utf-8'))
            if not isinstance(value, dict) or set(value) != {'text'} or not isinstance(value['text'], str):
                raise ValueError('inspection input must be WikiText-style JSONL text rows')
            yield value


def inspect_rows(rows, *, split, row_limit=128):
    if not 1 <= row_limit <= 10_000:
        raise ValueError('inspection row limit must be 1..10,000')
    counters = {'rows_inspected': 0, 'blank_rows': 0, 'top_level_titles': 0}
    row_hash = hashlib.sha256()
    def bounded():
        iterator = iter(rows)
        try:
            for _ in range(row_limit):
                try:
                    row = next(iterator)
                except StopIteration:
                    return
                if not isinstance(row, dict) or set(row) != {'text'} or not isinstance(row['text'], str):
                    raise ValueError('expected WikiText-style text rows')
                if len(row['text']) > 128_000:
                    raise ValueError('inspection row exceeds character bound')
                counters['rows_inspected'] += 1
                counters['blank_rows'] += not row['text'].strip()
                counters['top_level_titles'] += article_title(row['text']) is not None
                row_hash.update(json.dumps(row, ensure_ascii=False, separators=(',', ':')).encode('utf-8') + b'\n')
                yield row
        finally:
            close = getattr(iterator, 'close', None)
            if close is not None:
                close()
    complete = 0
    hashes = []
    for document in iter_articles(bounded(), split=split, complete_split=False):
        complete += 1
        hashes.append(document['document_sha256'])
    return dict(counters, split=split, complete_articles_observed=complete,
                sample_rows_sha256=row_hash.hexdigest(),
                complete_article_hashes=hashes,
                scope='bounded prefix only; open final article discarded; no split reassignment or corpus validation')


def parquet_rows(paths, *, row_limit=None, batch_size=32):
    """Optional local PyArrow adapter. No remote/filesystem auth or installs.

    Batch row count is bounded; Parquet decoding may buffer a whole row group.
    Inspection should use the small verified validation shard, not remote train.
    """
    if not 1 <= batch_size <= 128 or (row_limit is not None and not 1 <= row_limit <= 10_000):
        raise ValueError('invalid Parquet safety bounds')
    try:
        import pyarrow.parquet as parquet
    except ImportError:
        raise ValueError('optional PyArrow unavailable; no installation attempted') from None
    emitted = 0
    for path in paths:
        source = parquet.ParquetFile(path, pre_buffer=False)
        try:
            if source.schema_arrow.names != ['text']:
                raise ValueError('unverified Parquet row schema; expected only text')
            for batch in source.iter_batches(batch_size=batch_size, columns=['text'], use_threads=False):
                for value in batch.column(0).to_pylist():
                    if row_limit is not None and emitted >= row_limit:
                        return
                    if not isinstance(value, str) or len(value) > 128_000:
                        raise ValueError('invalid or oversized WikiText text row')
                    emitted += 1
                    yield {'text': value}
                    if row_limit is not None and emitted == row_limit:
                        return
        finally:
            source.close()


def verify_file(path, expected):
    path = Path(path)
    if path.stat().st_size != expected['bytes'] or file_digest(path) != expected['sha256']:
        raise ValueError(f'pinned source size/checksum mismatch: {expected["path"]}')


def fetch_validation(destination, *, opener=None):
    """Explicit small public fetch only; never fetch training or use tokens.

    Known size is below1MiB. 200 is accepted only with exact published length;
    206 must declare the exact full-file range. Content is checksum verified.
    """
    source = source_manifest()
    expected = next(f for f in source['files'] if f['split'] == 'validation')
    destination = Path(destination)
    if destination.exists():
        raise ValueError('validation destination must be new')
    if expected['bytes'] > 1024 * 1024:
        raise ValueError('validation file exceeds hard1MiB download cap')
    url = f'https://huggingface.co/datasets/{source["dataset"]}/resolve/{source["revision"]}/{expected["path"]}'
    req = request.Request(url, headers={'Range': f'bytes=0-{expected["bytes"]-1}', 'Accept-Encoding': 'identity'})
    created = False
    try:
        with (opener or request.build_opener(SafeRedirect())).open(req, timeout=30) as response:
            if response.status not in (200, 206) or response.headers.get('Content-Length') != str(expected['bytes']):
                raise ValueError('server did not provide exact bounded validation length; no body read')
            if response.status == 206 and response.headers.get('Content-Range') != f'bytes 0-{expected["bytes"]-1}/{expected["bytes"]}':
                raise ValueError('validation range mismatch; no body read')
            if response.headers.get('Content-Encoding', 'identity').lower() != 'identity':
                raise ValueError('compressed HTTP transfer refused; no body read')
            destination.parent.mkdir(parents=True, exist_ok=True)
            count = 0
            with destination.open('xb') as output:
                created = True
                while count < expected['bytes']:
                    chunk = response.read(min(65536, expected['bytes']-count))
                    if not chunk or len(chunk) > expected['bytes']-count:
                        raise ValueError('incomplete or oversized validation transfer')
                    output.write(chunk)
                    count += len(chunk)
                if response.read(1):
                    raise ValueError('validation response exceeded declared bound')
            verify_file(destination, expected)
            return dict(expected, dataset=source['dataset'], config=source['config'], revision=source['revision'],
                        http_policy='bounded published size; exact length/range plus full SHA256 verified')
    except error.HTTPError as exception:
        if created:
            destination.unlink(missing_ok=True)
        diagnostic = http_diagnostics(exception.code, exception.headers, exception.geturl())
        exception.close()
        raise DownloadFailure(diagnostic) from None
    except error.URLError:
        if created:
            destination.unlink(missing_ok=True)
        raise ValueError('public validation request failed; no retry') from None
    except Exception:
        if created:
            destination.unlink(missing_ok=True)
        raise


def prepare_rows(split_rows, output, *, provenance, expected_rows=None, notice_path=None):
    """Local conversion/audit only; standard splits are never recomputed."""
    if set(split_rows) != set(SPLITS):
        raise ValueError('all three official split iterators required; no split guessing')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    if notice_path is not None:
        shutil.copyfile(notice_path,output / 'README.upstream.md')
    manifest = {'schema_version': 1, 'provenance': provenance, 'splits': {},
                'split_policy': 'retain source train/validation/test; never use random line split',
                'cross_split_exact_content_overlaps': 0, 'cross_split_title_key_overlaps': 0,
                'later_wikipedia_exclusion': 'required and not implemented; this index is evidence only'}
    with sqlite3.connect(output / 'article-index.sqlite') as database, open(output / 'holdouts.jsonl', 'w', encoding='utf-8') as holdouts:
        database.execute('CREATE TABLE articles (split TEXT, title_key TEXT, content_sha256 TEXT, row_sha256 TEXT, article_index INTEGER)')
        database.execute('CREATE INDEX content_lookup ON articles (content_sha256)')
        database.execute('CREATE INDEX title_lookup ON articles (title_key)')
        for split in SPLITS:
            counters = {'rows': 0, 'articles': 0, 'leading_blank_rows': 0}
            seen_title = False
            def counted():
                nonlocal seen_title
                for row in split_rows[split]:
                    counters['rows'] += 1
                    if not isinstance(row, dict) or not isinstance(row.get('text'), str):
                        raise ValueError('invalid WikiText row')
                    if article_title(row['text']) is not None:
                        seen_title = True
                    if not seen_title and not row['text'].strip():
                        counters['leading_blank_rows'] += 1
                    yield row
            with open(output / (split + '.articles.jsonl'), 'w', encoding='utf-8') as stream:
                for article in iter_articles(counted(), split=split, complete_split=True):
                    counters['articles'] += 1
                    manifest['cross_split_exact_content_overlaps'] += bool(database.execute('SELECT 1 FROM articles WHERE content_sha256=? AND split<>? LIMIT 1',(article['document_sha256'],split)).fetchone())
                    manifest['cross_split_title_key_overlaps'] += bool(database.execute('SELECT 1 FROM articles WHERE title_key=? AND split<>? LIMIT 1',(article['title_key'],split)).fetchone())
                    database.execute('INSERT INTO articles VALUES (?,?,?,?,?)',(split,article['title_key'],article['document_sha256'],article['row_sequence_sha256'],article['article_index']))
                    stream.write(json.dumps(article,ensure_ascii=False)+'\n')
                    if split != 'train':
                        holdouts.write(json.dumps({k:article[k] for k in ('split','title','title_key','document_sha256','row_sequence_sha256','row_start','row_end_exclusive')},ensure_ascii=False)+'\n')
            if expected_rows is not None and counters['rows'] != expected_rows[split]:
                raise ValueError(f'{split}: source row count differs from pinned metadata')
            manifest['splits'][split] = counters
    manifest['isolation_audit'] = 'passed_exact_checks_only' if not (manifest['cross_split_exact_content_overlaps'] or manifest['cross_split_title_key_overlaps']) else 'needs_attention'
    manifest['outputs'] = {p.name: {'bytes':p.stat().st_size,'sha256':file_digest(p)} for p in sorted(output.iterdir()) if p.is_file()}
    atomic_json(output / 'manifest.json',manifest)
    return manifest


def prepare_local(input_root, output):
    source = source_manifest()
    groups = {split: [] for split in SPLITS}
    for item in source['files']:
        path = Path(input_root) / item['path']
        verify_file(path,item)
        groups[item['split']].append(path)
    output = Path(output)
    result = prepare_rows({s:parquet_rows(groups[s]) for s in SPLITS},output,
                          provenance=source,expected_rows=source['expected_rows'],
                          notice_path=ROOT / source['notice']['local_path'])
    return result


def runtime_estimate(planned_tokens, measured_tokens, measured_seconds, overhead_seconds):
    if any(isinstance(v,bool) or not isinstance(v,int) or v <= 0 for v in (planned_tokens,measured_tokens)):
        raise ValueError('positive integer nonpadding token counts required')
    if any(isinstance(v,bool) or not isinstance(v,(int,float)) for v in (measured_seconds,overhead_seconds)):
        raise ValueError('numeric measured seconds and overhead required')
    try:
        measured_seconds = float(measured_seconds)
        overhead_seconds = float(overhead_seconds)
        if not math.isfinite(measured_seconds) or not math.isfinite(overhead_seconds) or measured_seconds <= 0 or overhead_seconds < 0:
            raise ValueError('finite positive measured seconds and nonnegative overhead required')
        rate = measured_tokens / measured_seconds
        if not math.isfinite(rate) or rate <= 0:
            raise ValueError('measured throughput outside finite positive range')
        hours = (planned_tokens / rate + overhead_seconds) / 3600
    except (OverflowError, ZeroDivisionError):
        raise ValueError('estimate inputs outside finite arithmetic range') from None
    if not math.isfinite(hours):
        raise ValueError('estimate outside finite range')
    return {'measured_nonpadding_tokens_per_second':rate,'planned_tokens':planned_tokens,
            'estimated_hours':hours,'scope':'projection from caller-measured full steps, not a benchmark or six-hour promise'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('fetch-validation');p.add_argument('destination')
    p=sub.add_parser('inspect-jsonl');p.add_argument('path');p.add_argument('--split',choices=SPLITS,required=True);p.add_argument('--rows',type=int,default=128)
    p=sub.add_parser('inspect-validation');p.add_argument('path');p.add_argument('--rows',type=int,default=128)
    p=sub.add_parser('prepare-local');p.add_argument('input_root');p.add_argument('--output',required=True)
    p=sub.add_parser('estimate-runtime')
    for name in ('planned-tokens','measured-tokens'):p.add_argument('--'+name,type=int,required=True)
    for name in ('measured-seconds','overhead-seconds'):p.add_argument('--'+name,type=float,required=True)
    args=vars(parser.parse_args());command=args.pop('command')
    try:
        if command=='fetch-validation':result=fetch_validation(args['destination'])
        elif command=='inspect-jsonl':result=inspect_rows(iter_jsonl_rows(args['path'],row_limit=args['rows']),split=args['split'],row_limit=args['rows']);result['source_status']='local JSONL; not verified official Parquet'
        elif command=='inspect-validation':
            source=source_manifest();item=next(f for f in source['files'] if f['split']=='validation');verify_file(args['path'],item)
            result=inspect_rows(parquet_rows([args['path']],row_limit=args['rows']),split='validation',row_limit=args['rows']);result['provenance']={'dataset':source['dataset'],'config':source['config'],'revision':source['revision'],'file':item}
        elif command=='prepare-local':result=prepare_local(**args)
        else:result=runtime_estimate(**args)
        print(json.dumps(result,indent=2,allow_nan=False))
    except (ValueError,OSError) as exception:
        parser.exit(2,f'Error: {exception}\n')

if __name__=='__main__':main()

"""Bounded official-data readiness helpers. No training or credential storage."""
from pathlib import Path, PurePosixPath
import re
from urllib import request, error, parse
from .core import strict_json, iter_records, file_digest

DATASET = 'Exploration-Lab/CS-781-Capstone'
REVISION = '89edba6a44d6bb9593fd13b6895b5cd3b06eb82e'
MAX_DOWNLOAD_BYTES = 8 * 1024 * 1024

# Diagnostics never retain an error body, arbitrary header text, or full URL.
SAFE_ERROR_CODES = frozenset({'GatedRepo', 'RepoNotFound', 'RevisionNotFound',
                             'EntryNotFound', 'BadRequest', 'Unauthorized',
                             'Forbidden', 'AccessDenied'})
SAFE_ORIGIN_HOSTS = frozenset({'huggingface.co', 'cdn-lfs.huggingface.co',
                              'cdn-lfs-us-1.hf.co', 'cdn-lfs-eu-1.hf.co',
                              'cas-bridge.xethub.hf.co'})
REQUEST_ID = re.compile(r'(?:[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}|[0-9a-fA-F]{16,64}|Root=1-[0-9a-fA-F]{8}-[0-9a-fA-F]{24})')


class DownloadFailure(ValueError):
    """Value-free message plus safe structured HTTP diagnostic metadata."""
    def __init__(self, diagnostics):
        self.diagnostics = diagnostics
        super().__init__(f"HTTP {diagnostics['http_status']}: {diagnostics['reason']}; no retries")


def http_diagnostics(status, headers, response_url, *, secret=None):
    """Classify bounded header metadata into fixed text; never echo its content.

    The two retained metadata values must match strict allowlists. The reason
    is selected from fixed phrases using at most 512 header characters, not a
    redacted copy of arbitrary text. This is a hint, not a proved root cause.
    """
    try:
        parsed = parse.urlsplit(response_url)
        hostname = parsed.hostname if parsed.scheme == 'https' else None
    except (TypeError, ValueError):
        hostname = None
    origin = 'huggingface_origin' if hostname == 'huggingface.co' else (
        'known_hf_redirect_host' if hostname in SAFE_ORIGIN_HOSTS else 'other_or_unknown_host')
    status = status if isinstance(status, int) and not isinstance(status, bool) and 100 <= status <= 599 else None
    diagnostic = {'http_status': status, 'origin_stage': origin,
                  'origin_host': hostname if hostname in SAFE_ORIGIN_HOSTS else None,
                  'hf_error_code': None, 'request_id': None,
                  'evidence_scope': 'allowlisted response headers only; no error body read'}
    def safe_header(name, limit):
        value = headers.get(name, '') if headers is not None else ''
        if not isinstance(value, str) or len(value) > limit or (secret and secret in value):
            return ''
        return value
    code = safe_header('X-Error-Code', 40)
    if code in SAFE_ERROR_CODES:
        diagnostic['hf_error_code'] = code
    identifier = safe_header('X-Request-Id', 80)
    if REQUEST_ID.fullmatch(identifier):
        diagnostic['request_id'] = identifier
    # This field influences only fixed labels; raw or partially redacted text
    # is never stored. Oversized values are omitted rather than truncated.
    reason = safe_header('X-Error-Message', 512).lower()
    if any(phrase in reason for phrase in ('fine-grained', 'token permission', 'token scope', 'public gated', 'insufficient permissions')):
        category, summary = 'token_permissions_hint', 'server metadata suggests token permission or scope denial'
    elif any(phrase in reason for phrase in ('invalid token', 'invalid credentials', 'expired token', 'authentication required')) or code == 'Unauthorized':
        category, summary = 'authentication_hint', 'server metadata suggests authentication rejection'
    elif code == 'GatedRepo' or any(phrase in reason for phrase in ('gated repo', 'authorized list', 'awaiting a review', 'access request')):
        category, summary = 'gated_repo_hint', 'server metadata suggests gated repository access denial'
    elif code in ('RepoNotFound', 'RevisionNotFound', 'EntryNotFound'):
        category, summary = 'path_or_visibility_hint', 'server metadata suggests path, revision, or repository visibility failure'
    elif status in (401, 403) and origin != 'huggingface_origin':
        category, summary = 'redirect_host_denial_hint', 'denial occurred at a redirect or unknown host; original gated access cause is unproven'
    else:
        category, summary = 'unknown_http_failure', 'safe server metadata does not establish the access failure cause'
    diagnostic.update(reason_category=category, reason=summary)
    return diagnostic


class SafeRedirect(request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        if parse.urlsplit(newurl).scheme != 'https':
            raise ValueError('non-HTTPS redirect refused')
        redirected = super().redirect_request(req, fp, code, msg, headers, newurl)
        if parse.urlsplit(req.full_url).netloc != parse.urlsplit(newurl).netloc:
            redirected.remove_header('Authorization')
            redirected.remove_header('Cookie')
        return redirected


def download_prefix(dataset_path, destination, *, byte_limit, token=None, opener=None):
    """Require a honored HTTP range. Bound bytes; no retries; never log URLs/token.

    Full-file completion is proven only when Content-Range total equals prefix
    length. Prefix files are never represented as complete datasets.
    """
    path = PurePosixPath(dataset_path)
    if not dataset_path or path.is_absolute() or '..' in path.parts or '\\' in dataset_path or '?' in dataset_path or '#' in dataset_path:
        raise ValueError('explicit dataset-relative path required')
    if isinstance(byte_limit, bool) or not isinstance(byte_limit, int) or not 1 <= byte_limit <= MAX_DOWNLOAD_BYTES:
        raise ValueError('download bound must be 1..8 MiB')
    destination = Path(destination)
    if destination.exists():
        raise ValueError('download destination must be new')
    url = f'https://huggingface.co/datasets/{DATASET}/resolve/{REVISION}/{parse.quote(dataset_path, safe="/")}'
    headers = {'Range': f'bytes=0-{byte_limit-1}', 'Accept-Encoding': 'identity'}
    if token:
        headers['Authorization'] = 'Bearer ' + token
    if token is not None and (not isinstance(token, str) or any(not 33 <= ord(c) <= 126 for c in token)):
        raise ValueError('runtime secret has invalid header characters; value not shown')
    req = request.Request(url, headers=headers)
    opener = opener or request.build_opener(SafeRedirect())
    created = False
    try:
        with opener.open(req, timeout=30) as response:
            if response.status in (401, 403):
                raise DownloadFailure(http_diagnostics(response.status, response.headers,
                                                       response.geturl(), secret=token))
            if response.status != 206:
                raise ValueError('server did not honor bounded range (206 required); no body read')
            match = re.fullmatch(r'bytes 0-(\d+)/(\d+)', response.headers.get('Content-Range', ''))
            if not match:
                raise ValueError('missing or malformed Content-Range; no body read')
            end, total = map(int, match.groups())
            expected = end + 1
            if not 0 < expected <= byte_limit or total < expected:
                raise ValueError('server range exceeds requested bound or has invalid total; no body read')
            if response.headers.get('Content-Encoding', 'identity').lower() != 'identity':
                raise ValueError('compressed HTTP transfer refused; no body read')
            length = response.headers.get('Content-Length')
            if length is not None and (not length.isdecimal() or int(length) != expected):
                raise ValueError('Content-Length does not match range; no body read')
            destination.parent.mkdir(parents=True, exist_ok=True)
            count = 0
            with destination.open('xb') as output:
                created = True
                while count < expected:
                    chunk = response.read(min(65536, expected-count))
                    if not chunk:
                        raise ValueError('incomplete bounded transfer')
                    if len(chunk) > expected-count:
                        raise ValueError('response exceeded bounded read')
                    output.write(chunk)
                    count += len(chunk)
            return {'dataset': DATASET, 'revision': REVISION, 'dataset_path': dataset_path,
                    'requested_byte_limit': byte_limit, 'downloaded_bytes': count,
                    'total_file_bytes': total, 'complete_file': count == total,
                    'sha256': file_digest(destination), 'local_file': destination.name}
    except error.HTTPError as exception:
        if created:
            destination.unlink(missing_ok=True)
        diagnostic = http_diagnostics(exception.code, exception.headers, exception.geturl(), secret=token)
        exception.close()  # Do not read even a bounded error body: it may contain corpus or credentials.
        raise DownloadFailure(diagnostic) from None
    except error.URLError:
        if created:
            destination.unlink(missing_ok=True)
        raise ValueError('network request failed; no retries') from None
    except Exception:
        if created:
            destination.unlink(missing_ok=True)
        raise


def inspect_test(path, *, layout, id_field=None, text_field=None):
    """Explicit schema only. No prompt/ID field guessing and no record printing."""
    if Path(path).stat().st_size > MAX_DOWNLOAD_BYTES:
        raise ValueError('test file exceeds 8 MiB local inspection limit')
    value = strict_json(Path(path).read_text(encoding='utf-8'))
    if layout == 'mapping':
        if not isinstance(value, dict):
            raise ValueError('selected mapping layout requires a JSON object')
        records = list(value.items())
    elif layout == 'records':
        if not id_field or not isinstance(value, list):
            raise ValueError('records layout requires an explicit id_field and JSON array')
        records = []
        for record in value:
            if not isinstance(record, dict) or id_field not in record:
                raise ValueError('test record missing selected ID field')
            records.append((record[id_field], record))
    else:
        raise ValueError('set verified TEST_LAYOUT to mapping or records')
    ids, word_counts = [], []
    for identifier, record in records:
        if not isinstance(identifier, str) or not identifier:
            raise ValueError('authoritative IDs must be nonempty strings; no coercion')
        if isinstance(record, dict):
            if not text_field or text_field not in record:
                raise ValueError('set verified TEST_TEXT_FIELD for object values')
            text = record[text_field]
        else:
            if text_field is not None:
                raise ValueError('string values require TEST_TEXT_FIELD = None')
            text = record
        if not isinstance(text, str) or not text.strip():
            raise ValueError('selected test text must be nonempty strings')
        ids.append(identifier)
        word_counts.append(len(text.split()))
    if not ids or len(ids) != len(set(ids)):
        raise ValueError('empty test set or duplicate official IDs')
    return ids, {'records': len(ids), 'layout': layout, 'id_field': id_field,
                 'text_field': text_field, 'min_prompt_words': min(word_counts),
                 'max_prompt_words': max(word_counts), 'all_prompts_512_words': all(n == 512 for n in word_counts)}


def inspect_training_prefix(path, *, record_limit=5, max_record_chars=128_000):
    if isinstance(record_limit, bool) or not isinstance(record_limit, int) or not 1 <= record_limit <= 20:
        raise ValueError('record limit must be 1..20')
    if not 1 <= max_record_chars <= 1_000_000:
        raise ValueError('record character limit must be 1..1,000,000')
    iterator = iter_records(path, chunk_size=4096, max_record_chars=max_record_chars)
    counts = []
    try:
        for record in iterator:
            counts.append(len(record['text']))
            if len(counts) == record_limit:
                break
    except (ValueError, UnicodeError):
        raise ValueError('prefix cannot supply requested valid JSON-array/text records within bounds; verify path/schema or explicitly adjust small bounds') from None
    finally:
        iterator.close()
    if not counts:
        raise ValueError('no complete training records in bounded prefix')
    return {'sample_records_checked': len(counts), 'requested_record_limit': record_limit,
            'max_record_chars': max_record_chars, 'observed_min_text_chars': min(counts),
            'observed_max_text_chars': max(counts),
            'scope': 'complete sampled records only; rest of file and corpus unverified'}

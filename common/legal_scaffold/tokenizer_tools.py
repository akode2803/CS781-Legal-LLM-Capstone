"""Fresh byte-level BPE on explicitly selected prepared TRAIN documents only."""
import argparse
import importlib.metadata
import json
import re
from pathlib import Path
from .core import atomic_json, file_digest, strict_json, digest_text, split_for


def settings(config):
    required = {'vocab_size', 'min_frequency', 'special_tokens', 'unk_token', 'use_regex'}
    if not isinstance(config, dict) or set(config) != required:
        raise ValueError('config must specify exactly ' + ', '.join(sorted(required)))
    specials = config['special_tokens']
    if (not isinstance(specials, list) or not specials or
            any(not isinstance(s, str) or not s.strip() for s in specials) or
            len(set(specials)) != len(specials)):
        raise ValueError('special_tokens must be an ordered list of unique nonempty strings')
    if config['unk_token'] not in specials:
        raise ValueError('unk_token must be in special_tokens')
    for key in ('vocab_size', 'min_frequency'):
        if type(config[key]) is not int or config[key] <= 0:
            raise ValueError(key + ' must be an explicit positive integer')
    if config['vocab_size'] < 256 + len(specials):
        raise ValueError('vocab_size must accommodate all 256 byte symbols plus special tokens')
    if type(config['use_regex']) is not bool:
        raise ValueError('use_regex must be an explicit boolean')
    return config


def select_corpus(directory, kind, split):
    directory = Path(directory)
    if kind not in ('legal', 'wikitext') or split not in ('train', 'validation', 'test'):
        raise ValueError('choose legal or wikitext and an explicit official/prepared split')
    name = split + ('.jsonl' if kind == 'legal' else '.articles.jsonl')
    manifest_path = directory / 'manifest.json'
    manifest = strict_json(manifest_path.read_text(encoding='utf-8'))
    if kind == 'legal':
        if 'sources' not in manifest or 'counts' not in manifest or split == 'test':
            raise ValueError('not a supported prepared legal split')
    elif 'provenance' not in manifest or 'splits' not in manifest:
        raise ValueError('not a supported prepared WikiText split')
    if kind == 'wikitext' and manifest.get('isolation_audit') != 'passed_exact_checks_only':
        raise ValueError('WikiText preparation must have passed its cross-split exact checks')
    expected = manifest.get('outputs', {}).get(name)
    path = directory / name
    if (not isinstance(expected, dict) or expected.get('sha256') != file_digest(path) or
            expected.get('bytes') != path.stat().st_size):
        raise ValueError('prepared corpus identity mismatch; older legal preparation must be regenerated')
    return path, {'kind': kind, 'split': split, 'file': name,
                  'sha256': expected['sha256'], 'bytes': expected['bytes'],
                  'preparation_manifest_sha256': file_digest(manifest_path),
                  'source_provenance': manifest.get('provenance', manifest.get('sources'))}


def documents(path, kind, split, max_document_bytes):
    if type(max_document_bytes) is not int or max_document_bytes <= 0:
        raise ValueError('max_document_bytes must be an explicit positive integer')
    # A JSONL record can require up to six bytes per original ASCII character.
    with open(path, 'rb') as stream:
        while True:
            line = stream.readline(max_document_bytes * 6 + 4097)
            if not line:
                return
            if len(line) > max_document_bytes * 6 + 4096:
                raise ValueError('JSONL record exceeds configured document bound')
            record = strict_json(line.decode('utf-8'))
            text = record.get('text') if isinstance(record, dict) else None
            if not isinstance(text, str) or not text.strip():
                raise ValueError('document text must be nonempty')
            if len(text.encode('utf-8')) > max_document_bytes:
                raise ValueError('document exceeds configured byte bound')
            if record.get('document_sha256') != digest_text(text):
                raise ValueError('document identity mismatch')
            if kind == 'wikitext' and record.get('split') != split:
                raise ValueError('article split differs from selected split')
            yield text


def fresh_bpe(prepared, kind, config, output, source_revision, max_document_bytes):
    config = settings(config)
    if not isinstance(source_revision, str) or not source_revision.strip():
        raise ValueError('supply source_revision; use local-unversioned honestly for local data')
    output = Path(output)
    if output.exists():
        raise FileExistsError('tokenizer output already exists')
    path, corpus = select_corpus(prepared, kind, 'train')
    preparation = strict_json((Path(prepared) / 'manifest.json').read_text(encoding='utf-8'))
    from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
    tokenizer = Tokenizer(models.BPE(unk_token=config['unk_token']))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=config['use_regex'])
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=config['vocab_size'], min_frequency=config['min_frequency'],
                                  special_tokens=config['special_tokens'], show_progress=False,
                                  initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
    totals = {'documents': 0, 'utf8_bytes': 0}
    def training_only():
        for text in documents(path, kind, 'train', max_document_bytes):
            if kind == 'legal' and split_for(text, preparation['seed'], preparation['validation_fraction']) != 'train':
                raise ValueError('legal document belongs to heldout split under preparation policy')
            totals['documents'] += 1
            totals['utf8_bytes'] += len(text.encode('utf-8'))
            yield text  # Each item is a whole document, never a concatenated split.
    tokenizer.train_from_iterator(training_only(), trainer=trainer)
    if not totals['documents']:
        raise ValueError('training corpus is empty')
    if file_digest(path) != corpus['sha256']:
        raise ValueError('corpus changed during fitting')
    checks = round_trip_checks(tokenizer)
    output.mkdir(parents=True, exist_ok=False)
    tokenizer.save(str(output / 'tokenizer.json'))
    manifest = {'schema_version': 1, 'algorithm': 'fresh byte-level BPE',
                'tokenizers_version': importlib.metadata.version('tokenizers'),
                'settings': config, 'training_corpus': corpus, 'source_revision': source_revision,
                'max_document_bytes': max_document_bytes, 'training_totals': totals,
                'tokenizer_sha256': file_digest(output / 'tokenizer.json'),
                'actual_vocab_size': tokenizer.get_vocab_size(),
                'special_token_ids': {s: tokenizer.token_to_id(s) for s in config['special_tokens']},
                'boundary_policy': 'separate documents; no BOS/EOS/PAD insertion; no concatenation',
                'normalization': 'none', 'add_prefix_space': False, 'round_trip_checks': checks}
    atomic_json(output / 'manifest.json', manifest)
    return manifest


def round_trip_checks(tokenizer):
    probes = ['  Section 12.\nA café — 法 ⚖️\tend ', 'unseen: \U00010fff\u0378']
    for text in probes:
        encoded = tokenizer.encode(text, add_special_tokens=False)
        if tokenizer.decode(encoded.ids, skip_special_tokens=False) != text:
            raise ValueError('tokenizer failed exact whitespace/Unicode round trip')
    return {'probes': len(probes), 'passed': True,
            'includes': 'whitespace, Unicode, unusual characters; this is not quality evaluation'}


def byte_alphabet():
    """ByteLevel's reversible byte-to-visible-symbol alphabet (256 symbols)."""
    visible=list(range(33,127))+list(range(161,173))+list(range(174,256))
    return {chr(i) for i in visible} | {chr(256+i) for i,_ in enumerate(b for b in range(256) if b not in visible)}


def tokenizer_identity(directory):
    """Verify serialization and recorded choices without fitting or importing a framework."""
    root = Path(directory)
    manifest = strict_json((root / 'manifest.json').read_text(encoding='utf-8'))
    identity = file_digest(root / 'tokenizer.json')
    if manifest.get('schema_version') != 1 or manifest.get('algorithm') != 'fresh byte-level BPE':
        raise ValueError('unsupported tokenizer manifest')
    config = settings(manifest.get('settings'))
    if manifest.get('tokenizer_sha256') != identity:
        raise ValueError('tokenizer identity mismatch')
    serialized = strict_json((root / 'tokenizer.json').read_text(encoding='utf-8'))
    model = serialized.get('model', {})
    pre = serialized.get('pre_tokenizer') or {}
    if (model.get('type') != 'BPE' or model.get('unk_token') != config['unk_token'] or model.get('dropout') is not None or
            pre.get('type') != 'ByteLevel' or pre.get('add_prefix_space') is not False or
            pre.get('use_regex') is not config['use_regex'] or
            (serialized.get('decoder') or {}).get('type') != 'ByteLevel' or
            any(serialized.get(k) is not None for k in ('normalizer','post_processor','padding','truncation'))):
        raise ValueError('serialized tokenizer differs from recorded byte-level policy')
    vocab = model.get('vocab')
    if not isinstance(vocab, dict) or any(type(v) is not int or v < 0 for v in vocab.values()):
        raise ValueError('invalid tokenizer vocabulary')
    if not byte_alphabet().issubset(vocab):raise ValueError('tokenizer lacks complete byte alphabet')
    training_corpus=manifest.get('training_corpus')
    if (not isinstance(training_corpus,dict) or training_corpus.get('split')!='train' or
            training_corpus.get('kind') not in ('legal','wikitext') or
            not re.fullmatch(r'[0-9a-f]{64}',training_corpus.get('sha256','')) or
            not isinstance(manifest.get('source_revision'),str) or not manifest['source_revision'].strip()):
        raise ValueError('tokenizer requires explicit train-only source identity/revision')
    added = serialized.get('added_tokens', [])
    if not isinstance(added, list) or len(added) != len(config['special_tokens']):
        raise ValueError('serialized special tokens differ from supplied list')
    special_ids = {}
    for expected, item in zip(config['special_tokens'], added):
        if (not isinstance(item,dict) or item.get('content') != expected or item.get('special') is not True or
                type(item.get('id')) is not int or vocab.get(expected) != item['id']):
            raise ValueError('serialized special token identity mismatch')
        special_ids[expected] = item['id']
    if (special_ids != manifest.get('special_token_ids') or
            len(vocab) != manifest.get('actual_vocab_size') or
            set(vocab.values()) != set(range(len(vocab))) or len(vocab) > config['vocab_size']):
        raise ValueError('tokenizer vocabulary/ID accounting mismatch')
    return {'tokenizer_sha256': identity, 'manifest_sha256': file_digest(root / 'manifest.json'),
            'target_vocab_size':config['vocab_size'], 'actual_vocab_size':len(vocab),
            'settings':config, 'tokenizers_version':manifest.get('tokenizers_version'),
            'special_token_ids':special_ids, 'training_corpus':manifest.get('training_corpus'),
            'source_revision':manifest.get('source_revision')}


def count_tokens(tokenizer_directory, prepared, kind, split, max_document_bytes):
    from tokenizers import Tokenizer
    root = Path(tokenizer_directory)
    identity = tokenizer_identity(root)['tokenizer_sha256']
    tokenizer = Tokenizer.from_file(str(root / 'tokenizer.json'))
    path, corpus = select_corpus(prepared, kind, split)
    total = {'documents': 0, 'tokens': 0, 'utf8_bytes': 0}
    for text in documents(path, kind, split, max_document_bytes):
        total['documents'] += 1
        total['utf8_bytes'] += len(text.encode('utf-8'))
        total['tokens'] += len(tokenizer.encode(text, add_special_tokens=False).ids)
    if file_digest(path) != corpus['sha256']:
        raise ValueError('corpus changed during counting')
    return {'tokenizer_sha256': identity, 'corpus': corpus, 'counts': total,
            'automatically_inserted_special_or_padding_tokens': False,
            'literal_special_token_strings': 'recognized and included in token totals',
            'operation': 'frozen tokenizer encoding only'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('fit')
    p.add_argument('--config', required=True); p.add_argument('--output', required=True)
    p.add_argument('--source-revision', required=True)
    p = sub.add_parser('count')
    p.add_argument('--tokenizer', required=True); p.add_argument('--split', required=True, choices=('train','validation','test'))
    p.add_argument('--output', required=True)
    for p in sub.choices.values():
        p.add_argument('--prepared', required=True); p.add_argument('--kind', required=True, choices=('legal','wikitext'))
        p.add_argument('--max-document-bytes', required=True, type=int)
    args = parser.parse_args()
    try:
        if args.command == 'fit':
            result = fresh_bpe(args.prepared, args.kind, strict_json(Path(args.config).read_text()),
                               args.output, args.source_revision, args.max_document_bytes)
        else:
            if Path(args.output).exists():
                raise ValueError('count output already exists')
            result = count_tokens(args.tokenizer,args.prepared,args.kind,args.split,args.max_document_bytes)
            atomic_json(args.output,result)
        print(json.dumps(result, indent=2))
    except (ValueError, OSError, ImportError, KeyError, TypeError) as error:
        parser.exit(2, f'Error: {error}\n')


if __name__ == '__main__':
    main()

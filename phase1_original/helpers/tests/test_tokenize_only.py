"""One small interruption/resume and token/split equivalence check; no training."""
from pathlib import Path
import json, shutil, sys, tempfile, time
ROOT = Path(__file__).parents[1]
if (ROOT / 'course-run').is_dir(): ROOT = ROOT / 'course-run'
sys.path.insert(0, str(ROOT))
from legal_scaffold.legal_tokenize_only import prepare, TokenShards, install, records
from legal_scaffold.core import split_for
from legal_scaffold.legal_data import FILES
from legal_scaffold.window_file import WindowFile
from tokenizers import Tokenizer

def main():
    tokenizer_root = ROOT / 'tokenizer'
    tokenizer = Tokenizer.from_file(str(tokenizer_root / 'tokenizer.json'))
    started = time.monotonic()
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        texts = ['A legal document number %d with evidence and a judgment.' % i for i in range(40)]
        texts += [texts[0], '', 'न्याय café ⚖ ' * 7000]
        paths = {}
        expected = {'train': [], 'validation': []}
        documents = {'train': 0, 'validation': 0}
        for index, name in enumerate(FILES):
            rows = [{'text': text} for text in texts[index::4]]
            paths[name] = root / (name + '.json')
            paths[name].write_text(json.dumps(rows, ensure_ascii=False), encoding='utf-8')
            read = list(records(paths[name]))
            assert [r for r, _ in read] == rows
            for cursor, (row, offset) in enumerate(read):
                assert [r for r, _ in records(paths[name], offset)] == rows[cursor + 1:]
            for row in rows:
                text = row['text']
                if text == '': continue
                split = split_for(text, 'course', .1)
                documents[split] += 1
                ids = tokenizer.encode(text, add_special_tokens=False).ids
                expected[split] += [ids[i:i+1025] for i in range(0, len(ids), 1024) if len(ids[i:i+1025]) >= 2]
        options = dict(official=False, shard_bytes=64, commit_seconds=1e9)
        uninterrupted = prepare(paths, tokenizer_root, root / 'fresh', **options)
        class Interrupted(Exception): pass
        def interrupt(state): raise Interrupted()
        try:
            prepare(paths, tokenizer_root, root / 'resumed', _after_commit=interrupt, **options)
        except Interrupted: pass
        else: raise AssertionError('interruption hook did not run')
        committed = json.loads((root / 'resumed/progress.json').read_text())
        assert committed['source_offset'] > 0
        resumed = prepare(paths, tokenizer_root, root / 'resumed', **options)
        assert resumed == uninterrupted
        for split in expected:
            windows = TokenShards(root / 'resumed', split)
            assert isinstance(windows, WindowFile)
            windows.check_metadata(2048, 32334)
            assert list(windows) == expected[split]
            assert windows[:] == expected[split]
            assert resumed['counts'][split]['documents'] == documents[split]
        assert sum(c['empty_text_skipped'] for c in resumed['source_counts'].values()) == 1
        duplicate = tokenizer.encode(texts[0], add_special_tokens=False).ids
        assert expected[split_for(texts[0], 'course', .1)].count(duplicate) == 2
        for bad in [{'text': None}, {'text': 7}, {}, {'text': '  '}]:
            path = root / 'bad.json'; path.write_text(json.dumps([bad]))
            try: list(records(path))
            except ValueError: pass
            else: raise AssertionError('invalid text accepted')
        for raw in ['[{"text":"x","text":"y"}]', '[{"text":"x"},]', '[{"text":"x"}]x']:
            path.write_text(raw)
            try: list(records(path))
            except ValueError: pass
            else: raise AssertionError('malformed JSON accepted')
        source = ROOT
        shutil.copytree(source / 'legal_scaffold', root / 'installed/legal_scaffold')
        shutil.copyfile(source / 'source-version.json', root / 'installed/source-version.json')
        model_before = (root / 'installed/legal_scaffold/models.py').read_bytes()
        install(root / 'installed')
        assert (root / 'installed/legal_scaffold/models.py').read_bytes() == model_before
        compile((root / 'installed/legal_scaffold/legal_training.py').read_text(), 'legal_training.py', 'exec')
        install(root / 'installed')
        print(json.dumps({'passed': True, 'synthetic_documents': len(texts), 'windows': sum(len(x) for x in expected.values()), 'seconds': time.monotonic() - started, 'training_started': False}))

if __name__ == '__main__': main()

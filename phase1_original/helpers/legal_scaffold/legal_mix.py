"""One fixed equal-token source order over existing shards; no retokenization."""
from pathlib import Path
import bisect, hashlib, mmap, os, struct
from .core import atomic_json, file_digest, strict_json
from .legal_data import FILES
from .legal_tokenize_only import TokenShards
from .window_file import WindowFile

RECORD = struct.Struct('<IH')  # Original global window index, token-ID count.
POLICY = 'take next window from least-consumed source; ties in FILES order; stop when any source exhausts'

class SourceShards(TokenShards):
    def __init__(self, root, begin_group, end_group):
        super().__init__(root, 'train')
        self.entries = [item for item in self.entries
                        if begin_group <= int(Path(item['file']).stem.split('-')[-1]) < end_group]
        self.ends = []; self.count = 0
        for item in self.entries:
            self.count += item['windows']; self.ends.append(self.count)
        self.supervised_tokens = sum(item['supervised_tokens'] for item in self.entries)
        if not self.count: raise ValueError('each mixed source needs training windows')

    def window_length(self, key):
        index = bisect.bisect_right(self.ends, key)
        self.load(index)
        local = key - (self.ends[index-1] if index else 0)
        return struct.unpack_from('<H', self.data, self.offsets[local])[0]

class MixedTokenShards(WindowFile):
    """Persist one small order index; four bounded shard caches; shared by models.

    end_groups comes from source-end stdout: source_index 1/2/3/4,
    next_record 0, committed_groups N. All shard groups below N belong to
    completed sources. Never infer a source from token content or filename count.
    """
    def __init__(self, root, end_groups, output):
        self.source_names = list(FILES)
        base = strict_json((Path(root)/'manifest.json').read_text())
        if not base['complete']: raise ValueError('complete tokenization required')
        if set(end_groups) != set(FILES): raise ValueError('supply all four verified source-end group numbers')
        previous = 0; self.sources = {}; self.source_ends = []; total = 0
        for name in self.source_names:
            end = end_groups[name]
            if type(end) is not int or end <= previous: raise ValueError('missing or unordered source-end group: '+name)
            source = SourceShards(root, previous, end)
            self.sources[name] = source; total += len(source); self.source_ends.append(total)
            previous = end
        if previous != base['group']: raise ValueError('last source end must match completed token manifest group')
        if total != base['counts']['train']['windows']: raise ValueError('source boundaries do not cover every training shard')
        self.entries = [item for source in self.sources.values() for item in source.entries]
        self.token_only_manifest = base
        self.output = Path(output); self.output.mkdir(parents=True, exist_ok=True)
        identity = {'policy':POLICY, 'source_end_groups':end_groups,
                    'token_manifest_sha256':file_digest(Path(root)/'manifest.json')}
        manifest_path = self.output/'manifest.json'
        if not manifest_path.exists():
            cursor = {name:0 for name in self.source_names}
            tokens = dict(cursor); count = 0; part = self.output/'order.bin.part'
            with part.open('wb') as stream:
                while all(cursor[name] < len(self.sources[name]) for name in self.source_names):
                    name = min(self.source_names, key=lambda key:tokens[key])
                    number = self.source_names.index(name)
                    length = self.sources[name].window_length(cursor[name])
                    original = cursor[name] + (self.source_ends[number-1] if number else 0)
                    stream.write(RECORD.pack(original, length))
                    cursor[name] += 1; tokens[name] += length-1; count += 1
                stream.flush(); os.fsync(stream.fileno())
            if not count: raise ValueError('no balanced training windows')
            index = {'file':'order.bin', 'bytes':part.stat().st_size, 'sha256':file_digest(part)}
            os.replace(part, self.output/index['file'])
            atomic_json(manifest_path, {'identity':identity,'source_names':self.source_names,
                'source_available_tokens':{name:source.supervised_tokens for name,source in self.sources.items()},
                'source_consumed_tokens':tokens,'windows':count,'supervised_tokens':sum(tokens.values()),'index':index})
        self.manifest = strict_json(manifest_path.read_text())
        if self.manifest['identity'] != identity: raise ValueError('mixed order inputs differ; preserve the old order directory')
        index = self.manifest['index']; path = self.output/index['file']
        self.count = self.manifest['windows']; self.supervised_tokens = self.manifest['supervised_tokens']
        if path.stat().st_size != self.count*RECORD.size or path.stat().st_size != index['bytes'] or file_digest(path) != index['sha256']:
            raise ValueError('mixed order index changed')
        self.sha256 = file_digest(manifest_path)
        self.order_stream = path.open('rb')
        self.order = mmap.mmap(self.order_stream.fileno(), 0, access=mmap.ACCESS_READ)

    def __len__(self): return self.count

    def record(self, key):
        original, length = RECORD.unpack_from(self.order, key*RECORD.size)
        source = bisect.bisect_right(self.source_ends, original)
        local = original - (self.source_ends[source-1] if source else 0)
        return self.source_names[source], local, length

    def __getitem__(self, key):
        if isinstance(key, slice): return [self[i] for i in range(*key.indices(self.count))]
        if key < 0: key += self.count
        if not 0 <= key < self.count: raise IndexError(key)
        name, local, _ = self.record(key)
        return self.sources[name][local]

    def __iter__(self):
        for key in range(self.count): yield self[key]

    def source_tokens(self, start, stop):
        tokens = {name:0 for name in self.source_names}
        for key in range(start, stop):
            name, _, length = self.record(key); tokens[name] += length-1
        return tokens

    def check_metadata(self, context_length, vocab_size):
        for source in self.sources.values(): source.check_metadata(context_length, vocab_size)

    def close(self):
        self.order.close(); self.order_stream.close()

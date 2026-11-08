"""Deterministic BIO alignment, BiLSTM-CRF head, and recovery helpers."""
from __future__ import annotations

import json
import math
import random
import re
from pathlib import Path

import torch
from torch import nn

WORD_OR_PUNCT = re.compile(r"\w+(?:['’]\w+)*|[^\w\s]", re.UNICODE)


def bio_tags(labels):
    return ("O",) + tuple(f"{prefix}-{label}" for label in labels for prefix in ("B", "I"))


def segment_words(text):
    return [(m.start(), m.end(), m.group(0)) for m in WORD_OR_PUNCT.finditer(text)]


def make_document_windows(row, tokenizer, max_bpe=512, stride_bpe=384):
    """Create overlapping word-boundary windows without consulting gold."""
    text = row["text"]
    encoding = tokenizer.encode(text)
    offsets = encoding.offsets
    words = segment_words(text)
    mapped = []
    cursor = 0
    for word_index, (start, end, token_text) in enumerate(words):
        while cursor < len(offsets) and offsets[cursor][1] <= start:
            cursor += 1
        index = cursor
        first = last = None
        while index < len(offsets) and offsets[index][0] < end:
            if offsets[index][1] > start:
                first = index if first is None else first
                last = index
            index += 1
        if first is None:
            raise ValueError(f"word has no tokenizer representation: {row['id']} {start}:{end}")
        mapped.append((start, end, token_text, first, last, word_index))

    windows = []
    start_word = 0
    while start_word < len(mapped):
        first_bpe = mapped[start_word][3]
        end_word = start_word
        while end_word < len(mapped) and mapped[end_word][4] - first_bpe + 1 <= max_bpe:
            end_word += 1
        if end_word == start_word:
            raise ValueError("single word exceeds max_bpe")
        last_bpe = mapped[end_word - 1][4]
        window_words = mapped[start_word:end_word]
        windows.append({
            "document_id": row["id"],
            "text": text,
            "entities": row["entities"],
            "input_ids": encoding.ids[first_bpe:last_bpe + 1],
            "word_spans": [(x[0], x[1]) for x in window_words],
            "word_last": [x[4] - first_bpe for x in window_words],
            "word_start_index": start_word,
            "word_end_index": end_word,
        })
        if end_word == len(mapped):
            break
        target = first_bpe + stride_bpe
        next_word = start_word + 1
        while next_word < end_word and mapped[next_word][3] < target:
            next_word += 1
        start_word = min(next_word, end_word - 1)

    for index, window in enumerate(windows):
        left = window["word_start_index"]
        right = window["word_end_index"]
        if index:
            previous = windows[index - 1]
            left = (window["word_start_index"] + previous["word_end_index"]) // 2
        if index + 1 < len(windows):
            following = windows[index + 1]
            right = (following["word_start_index"] + window["word_end_index"]) // 2
        window["owner_start_index"] = left
        window["owner_end_index"] = right
    return windows


def targets_and_segments(window, labels):
    """Return BIO targets and contiguous supervised segments.

    Any word touched by a gold span that cannot be represented exactly in this
    window is ambiguous.  Ambiguous positions split the CRF training sequence;
    they are never trained as O.  This policy is training-only and does not
    alter validation segmentation or validation gold.
    """
    tags = bio_tags(labels)
    tag_to_id = {tag: index for index, tag in enumerate(tags)}
    word_spans = window["word_spans"]
    targets = [tag_to_id["O"]] * len(word_spans)
    ambiguous = set()
    start_lookup = {start: index for index, (start, _) in enumerate(word_spans)}
    end_lookup = {end: index for index, (_, end) in enumerate(word_spans)}
    window_left, window_right = word_spans[0][0], word_spans[-1][1]

    for start, end, label in window["entities"]:
        if end <= window_left or start >= window_right:
            continue
        first = start_lookup.get(start)
        last = end_lookup.get(end)
        if first is not None and last is not None and first <= last:
            targets[first] = tag_to_id[f"B-{label}"]
            for index in range(first + 1, last + 1):
                targets[index] = tag_to_id[f"I-{label}"]
        else:
            for index, (word_start, word_end) in enumerate(word_spans):
                if word_start < end and word_end > start:
                    ambiguous.add(index)

    segments = []
    first = 0
    for index in range(len(targets) + 1):
        if index == len(targets) or index in ambiguous:
            if first < index:
                segments.append((first, index))
            first = index + 1
    return torch.tensor(targets, dtype=torch.long), segments, sorted(ambiguous)


class LinearChainCRF(nn.Module):
    def __init__(self, tags):
        super().__init__()
        self.tags = tuple(tags)
        count = len(tags)
        self.transitions = nn.Parameter(torch.empty(count, count))
        self.start = nn.Parameter(torch.empty(count))
        self.end = nn.Parameter(torch.empty(count))
        nn.init.uniform_(self.transitions, -0.1, 0.1)
        nn.init.uniform_(self.start, -0.1, 0.1)
        nn.init.uniform_(self.end, -0.1, 0.1)
        allowed = torch.ones(count, count, dtype=torch.bool)
        start_allowed = torch.ones(count, dtype=torch.bool)
        for current, current_tag in enumerate(tags):
            if current_tag.startswith("I-"):
                start_allowed[current] = False
                label = current_tag[2:]
                for previous, previous_tag in enumerate(tags):
                    allowed[previous, current] = previous_tag in (f"B-{label}", f"I-{label}")
        self.register_buffer("allowed", allowed)
        self.register_buffer("start_allowed", start_allowed)

    def _scores(self):
        return (
            self.transitions.masked_fill(~self.allowed, -1e4),
            self.start.masked_fill(~self.start_allowed, -1e4),
        )

    def neg_log_likelihood(self, emissions, targets):
        transitions, start = self._scores()
        alpha = start + emissions[0]
        gold = start[targets[0]] + emissions[0, targets[0]]
        for index in range(1, emissions.shape[0]):
            alpha = torch.logsumexp(alpha[:, None] + transitions + emissions[index][None, :], dim=0)
            gold = gold + transitions[targets[index - 1], targets[index]] + emissions[index, targets[index]]
        partition = torch.logsumexp(alpha + self.end, dim=0)
        gold = gold + self.end[targets[-1]]
        return partition - gold

    def decode(self, emissions):
        transitions, start = self._scores()
        score = start + emissions[0]
        history = []
        for index in range(1, emissions.shape[0]):
            candidates = score[:, None] + transitions
            best_score, best_previous = candidates.max(dim=0)
            score = best_score + emissions[index]
            history.append(best_previous)
        current = int((score + self.end).argmax())
        path = [current]
        for backpointer in reversed(history):
            current = int(backpointer[current])
            path.append(current)
        return list(reversed(path))


class BackboneBiLSTMCRF(nn.Module):
    def __init__(self, backbone, width, labels, lstm_hidden=256, dropout=0.2):
        super().__init__()
        self.backbone = backbone
        self.labels = tuple(labels)
        self.tags = bio_tags(labels)
        self.dropout = nn.Dropout(dropout)
        self.bilstm = nn.LSTM(width, lstm_hidden, batch_first=True, bidirectional=True)
        self.emission = nn.Linear(lstm_hidden * 2, len(self.tags))
        self.crf = LinearChainCRF(self.tags)

    def backbone_hidden(self, input_ids):
        hidden = self.backbone.embedding(input_ids)
        for block in self.backbone.blocks:
            hidden, _ = block(hidden, use_cache=False)
        return self.backbone.final_norm(hidden)

    def emissions(self, input_ids, word_last):
        hidden = self.backbone_hidden(input_ids)
        words = hidden[:, word_last.long(), :]
        contextual, _ = self.bilstm(self.dropout(words))
        return self.emission(self.dropout(contextual))[0]


def decode_entities(window, tag_ids, tags):
    """Decode BIO to original character offsets, keeping this window's owner core."""
    output = []
    index = 0
    global_start = window["word_start_index"]
    while index < len(tag_ids):
        tag = tags[tag_ids[index]]
        if not tag.startswith("B-"):
            index += 1
            continue
        label = tag[2:]
        last = index
        while last + 1 < len(tag_ids) and tags[tag_ids[last + 1]] == f"I-{label}":
            last += 1
        midpoint = global_start + (index + last) // 2
        if window["owner_start_index"] <= midpoint < window["owner_end_index"]:
            output.append((
                window["document_id"],
                window["word_spans"][index][0],
                window["word_spans"][last][1],
                label,
            ))
        index = last + 1
    return output


def save_checkpoint(path, model, optimizer, scheduler, scaler, progress, identity):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save({
        "schema_version": 1, "identity": identity, "progress": progress,
        "model": model.state_dict(), "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
        "python_rng": random.getstate(), "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }, temporary)
    temporary.replace(path)


def resume_checkpoint(path, model, optimizer, scheduler, scaler, identity, device):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("schema_version") != 1 or payload.get("identity") != identity:
        raise ValueError("BIO checkpoint identity mismatch")
    model.load_state_dict(payload["model"], strict=True); model.to(device)
    optimizer.load_state_dict(payload["optimizer"])
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value): state[key] = value.to(device)
    scheduler.load_state_dict(payload["scheduler"]); scaler.load_state_dict(payload["scaler"])
    random.setstate(payload["python_rng"]); torch.set_rng_state(payload["torch_rng"])
    if device == "cuda" and payload.get("cuda_rng") is not None:
        torch.cuda.set_rng_state_all(payload["cuda_rng"])
    return payload["progress"]

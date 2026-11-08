"""Small helpers for fixed-weight GQA/MHA BIO CRF score ensembles."""
from __future__ import annotations

import json
from pathlib import Path

import torch


def cache_key(identity: dict) -> str:
    import hashlib
    raw = json.dumps(identity, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:16]


def save_cache_atomic(path: str | Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def constrained_components(model) -> dict:
    transitions, start = model.crf._scores()
    return {
        "transitions": transitions.detach().float().cpu(),
        "start": start.detach().float().cpu(),
        "end": model.crf.end.detach().float().cpu(),
        "allowed": model.crf.allowed.detach().cpu(),
        "start_allowed": model.crf.start_allowed.detach().cpu(),
    }


@torch.inference_mode()
def collect_emission_cache(model, windows, cache_path, cache_identity, device="cuda", checkpoint_name="model"):
    """Collect/recover one model's per-window emissions and source CRF paths.

    The cache is checkpoint/dataset/window keyed and atomically rewritten after
    each window, so an interrupted inference pass resumes at its last saved
    window. Model execution is caller-serialized to bound accelerator memory.
    """
    cache_path = Path(cache_path)
    components = constrained_components(model)
    if cache_path.exists():
        payload = torch.load(cache_path, map_location="cpu", weights_only=False)
        if payload.get("cache_identity") != cache_identity:
            raise ValueError(f"cache identity mismatch: {cache_path}")
        if payload.get("num_windows") != len(windows):
            raise ValueError(f"cache window count mismatch: {cache_path}")
        if not torch.equal(payload["components"]["allowed"], components["allowed"]):
            raise ValueError(f"cached CRF transition constraints changed: {cache_path}")
        if not torch.equal(payload["components"]["start_allowed"], components["start_allowed"]):
            raise ValueError(f"cached CRF start constraints changed: {cache_path}")
        for key in ("transitions", "start", "end"):
            if not torch.equal(payload["components"][key], components[key]):
                raise ValueError(f"cached CRF score component changed ({key}): {cache_path}")
        emissions = payload["emissions"]
        reference_paths = payload["reference_paths"]
        if len(emissions) != len(reference_paths) or len(emissions) > len(windows):
            raise ValueError(f"partial cache is malformed: {cache_path}")
        for index, (logits, path) in enumerate(zip(emissions, reference_paths)):
            expected_words = len(windows[index]["word_last"])
            expected_tags = int(components["transitions"].shape[0])
            if (logits.ndim != 2 or logits.shape != (expected_words, expected_tags)
                    or len(path) != expected_words):
                raise ValueError(
                    f"invalid cached emission/path shape at window {index}: "
                    f"emissions={tuple(logits.shape)}, path={len(path)}, words={expected_words}; "
                    f"preserving this cache and use a new cache path: {cache_path}"
                )
        print(f"{checkpoint_name} cache resumes at window {len(emissions)}/{len(windows)}")
    else:
        emissions, reference_paths = [], []
        payload = {
            "schema_version": 1,
            "cache_identity": cache_identity,
            "num_windows": len(windows),
            "components": components,
            "emissions": emissions,
            "reference_paths": reference_paths,
        }
        save_cache_atomic(cache_path, payload)

    model.eval()
    for index in range(len(emissions), len(windows)):
        window = windows[index]
        ids = torch.tensor([window["input_ids"]], dtype=torch.long, device=device)
        word_last = torch.tensor(window["word_last"], dtype=torch.long, device=device)
        enabled = str(device).startswith("cuda")
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=enabled):
            logits = model.emissions(ids, word_last)
        logits = logits.float()
        expected_words = len(window["word_last"])
        if logits.ndim != 2 or logits.shape[0] != expected_words:
            raise ValueError(
                f"model.emissions must return [words,tags] for window {index}; "
                f"got {tuple(logits.shape)}, expected {expected_words} words"
            )
        path = model.crf.decode(logits)
        if len(path) != expected_words:
            raise ValueError(f"CRF path length {len(path)} != word count {expected_words} at window {index}")
        emissions.append(logits.detach().cpu())
        reference_paths.append([int(tag) for tag in path])
        payload["emissions"] = emissions
        payload["reference_paths"] = reference_paths
        save_cache_atomic(cache_path, payload)
        if (index + 1) % 10 == 0 or index + 1 == len(windows):
            print(f"{checkpoint_name} cached {index + 1}/{len(windows)} windows", flush=True)
    payload["complete"] = True
    save_cache_atomic(cache_path, payload)
    return payload


def blend_components(gqa: dict, mha: dict, gqa_weight: float) -> dict:
    """Blend complete CRF scores and reapply the shared legality constraints."""
    weight = float(gqa_weight)
    if not 0.0 <= weight <= 1.0:
        raise ValueError("GQA weight must be in [0, 1]")
    if gqa["transitions"].shape != mha["transitions"].shape:
        raise ValueError("GQA/MHA CRF transition dimensions differ")
    if not torch.equal(gqa["allowed"], mha["allowed"]):
        raise ValueError("GQA/MHA BIO transition constraints differ")
    if not torch.equal(gqa["start_allowed"], mha["start_allowed"]):
        raise ValueError("GQA/MHA BIO start constraints differ")
    other = 1.0 - weight
    allowed = gqa["allowed"]
    start_allowed = gqa["start_allowed"]
    transitions = weight * gqa["transitions"] + other * mha["transitions"]
    start = weight * gqa["start"] + other * mha["start"]
    # Match LinearChainCRF's existing hard-penalty convention exactly.
    transitions = transitions.masked_fill(~allowed, -1e4)
    start = start.masked_fill(~start_allowed, -1e4)
    return {
        "transitions": transitions,
        "start": start,
        "end": weight * gqa["end"] + other * mha["end"],
        "allowed": allowed,
        "start_allowed": start_allowed,
    }


def viterbi_decode(emissions: torch.Tensor, components: dict) -> list[int]:
    """CRF Viterbi decoder with the same recurrence as LinearChainCRF.decode."""
    emissions = emissions.float()
    transitions = components["transitions"].to(emissions.device, dtype=emissions.dtype)
    start = components["start"].to(emissions.device, dtype=emissions.dtype)
    end = components["end"].to(emissions.device, dtype=emissions.dtype)
    score = start + emissions[0]
    history = []
    for index in range(1, emissions.shape[0]):
        candidates = score[:, None] + transitions
        best_score, best_previous = candidates.max(dim=0)
        score = best_score + emissions[index]
        history.append(best_previous)
    current = int((score + end).argmax())
    path = [current]
    for backpointer in reversed(history):
        current = int(backpointer[current])
        path.append(current)
    return list(reversed(path))


def _source_viterbi_decode(emissions: torch.Tensor, components: dict) -> list[int]:
    """Literal CPU equivalent of LinearChainCRF.decode for parity checks.

    Cached reference_paths are produced while the model is on CUDA. They are
    useful for diagnostics, but exact path equality across devices is not a
    valid implementation-parity test when scores are near ties.
    """
    transitions = components["transitions"].to(emissions.device, dtype=emissions.dtype)
    start = components["start"].to(emissions.device, dtype=emissions.dtype)
    end = components["end"].to(emissions.device, dtype=emissions.dtype)
    score = start + emissions[0]
    history = []
    for index in range(1, emissions.shape[0]):
        candidates = score[:, None] + transitions
        best_score, best_previous = candidates.max(dim=0)
        score = best_score + emissions[index]
        history.append(best_previous)
    current = int((score + end).argmax())
    path = [current]
    for backpointer in reversed(history):
        current = int(backpointer[current])
        path.append(current)
    return list(reversed(path))


def macro_and_detail(rows, predictions, labels):
    gold = {(str(row["id"]), a, b, label) for row in rows for a, b, label in row["entities"]}
    pred = set(predictions)
    per_label = {}
    for label in labels:
        label_gold = {item for item in gold if item[3] == label}
        label_pred = {item for item in pred if item[3] == label}
        tp = len(label_gold & label_pred)
        precision = tp / len(label_pred) if label_pred else 0.0
        recall = tp / len(label_gold) if label_gold else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_label[label] = {
            "gold": len(label_gold), "predicted": len(label_pred), "tp": tp,
            "precision": precision, "recall": recall, "f1": f1,
        }
    per_document = {}
    for row in rows:
        doc_id = str(row["id"])
        doc_gold = {item for item in gold if item[0] == doc_id}
        doc_pred = {item for item in pred if item[0] == doc_id}
        tp = len(doc_gold & doc_pred)
        precision = tp / len(doc_pred) if doc_pred else 0.0
        recall = tp / len(doc_gold) if doc_gold else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_document[doc_id] = {
            "gold": len(doc_gold), "predicted": len(doc_pred), "tp": tp,
            "precision": precision, "recall": recall, "f1": f1,
        }
    return sum(item["f1"] for item in per_label.values()) / len(labels), per_label, per_document


def evaluate_weight(windows, rows, gqa_cache, mha_cache, gqa_weight, labels, tags, decode_entities):
    if len(gqa_cache["emissions"]) != len(windows) or len(mha_cache["emissions"]) != len(windows):
        raise ValueError("all validation windows must have complete GQA and MHA caches")
    components = blend_components(gqa_cache["components"], mha_cache["components"], gqa_weight)
    predictions = []
    endpoint_parity = True
    cached_reference_mismatches = 0
    for index, window in enumerate(windows):
        gqa_logits = gqa_cache["emissions"][index]
        mha_logits = mha_cache["emissions"][index]
        if gqa_logits.shape != mha_logits.shape:
            raise ValueError(f"GQA/MHA emission shape mismatch at window {index}")
        mixed_logits = float(gqa_weight) * gqa_logits + (1.0 - float(gqa_weight)) * mha_logits
        path = viterbi_decode(mixed_logits, components)
        if float(gqa_weight) == 1.0:
            source_components = gqa_cache["components"]
            endpoint_parity = endpoint_parity and all(
                torch.equal(components[key].cpu(), source_components[key].cpu())
                for key in ("transitions", "start", "end", "allowed", "start_allowed")
            ) and path == _source_viterbi_decode(mixed_logits, source_components)
            cached_reference_mismatches += path != gqa_cache["reference_paths"][index]
        elif float(gqa_weight) == 0.0:
            source_components = mha_cache["components"]
            endpoint_parity = endpoint_parity and all(
                torch.equal(components[key].cpu(), source_components[key].cpu())
                for key in ("transitions", "start", "end", "allowed", "start_allowed")
            ) and path == _source_viterbi_decode(mixed_logits, source_components)
            cached_reference_mismatches += path != mha_cache["reference_paths"][index]
        predictions.extend(decode_entities(window, path, tags))
    macro, per_label, per_document = macro_and_detail(rows, predictions, labels)
    return {
        "gqa_weight": float(gqa_weight),
        "mha_weight": 1.0 - float(gqa_weight),
        "macro_f1": macro,
        "per_label": per_label,
        "per_document": per_document,
        "predicted_entities": len(set(predictions)),
        "endpoint_decoder_parity": endpoint_parity if float(gqa_weight) in (0.0, 1.0) else None,
        "cached_reference_path_device_mismatches": cached_reference_mismatches if float(gqa_weight) in (0.0, 1.0) else None,
        "predictions": sorted(set(predictions)),
    }

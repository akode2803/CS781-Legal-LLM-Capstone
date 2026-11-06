"""Small-model Phase 2 adaptation helpers; data and span generation stay in legal_ner."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from legal_scaffold.models import CausalDecoder, ModelConfig
from legal_scaffold.snapshots import safe_file


def resolve_snapshot_checkpoint(root, config, *, stages=('legal_continued', 'wikitext_only'),
                                tokenizer_sha256=None):
    """Find the newest matching complete model snapshot using its real manifest."""
    root = Path(root)
    wanted = config if isinstance(config, ModelConfig) else ModelConfig(**config)
    wanted = wanted.validate().__dict__
    stage_rank = {name: index for index, name in enumerate(stages)}
    found = []
    for manifest_path in root.rglob('snapshot.json'):
        try:
            manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
            identity = manifest.get('identity', {})
            saved_model = identity.get('model', identity.get('config'))
            if not isinstance(saved_model, dict):
                continue
            saved_config = ModelConfig(**saved_model).validate().__dict__
            if saved_config != wanted or manifest.get('stage') not in stage_rank:
                continue
            tokenizer = identity.get('tokenizer_identity', identity.get('tokenizer', {}))
            if (tokenizer_sha256 is not None and
                    tokenizer.get('tokenizer_sha256') != tokenizer_sha256):
                continue
            relative_state = manifest.get('state_inventory', {}).get('model')
            if not relative_state:
                continue
            safe_state = safe_file(relative_state)
            checkpoint = manifest_path.parent / safe_state
            if (not checkpoint.is_file() or
                    relative_state not in manifest.get('artifacts', {})):
                continue
            progress = manifest.get('progress', {})
            found.append((stage_rank[manifest['stage']], -int(progress.get('step', 0)),
                          -manifest_path.stat().st_mtime, checkpoint, manifest))
        except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
            continue
    if not found:
        raise FileNotFoundError(f'No matching snapshot payload under {root} for config {wanted} and stages {stages}')
    found.sort(key=lambda item: (item[0], item[1], item[2]))
    _, _, _, checkpoint, manifest = found[0]
    return checkpoint, manifest


def restore_backbone(config, checkpoint, *, state_key='model', map_location='cpu'):
    """Restore a requested MHA/GQA/MLA and pre/post-norm decoder checkpoint."""
    config = config if isinstance(config, ModelConfig) else ModelConfig(**config)
    backbone = CausalDecoder(config)
    saved = torch.load(checkpoint, map_location=map_location, weights_only=False)
    state = saved.get(state_key, saved) if isinstance(saved, dict) else saved
    backbone.load_state_dict(state, strict=True)
    return backbone


def load_trained_teacher_checkpoint(path, model, *, tokenizer_sha256, labels,
                                   expected_model_config, expected_windows=None):
    """Restore a teacher NER model and validate its own checkpoint evidence."""
    saved = torch.load(path, map_location='cpu', weights_only=False)
    if not isinstance(saved, dict) or saved.get('schema_version') != 1:
        raise ValueError('teacher must be a Phase 2 training checkpoint')
    identity = saved.get('identity')
    progress = saved.get('progress')
    state = saved.get('model')
    if not isinstance(identity, dict) or not isinstance(progress, dict) or not isinstance(state, dict):
        raise ValueError('teacher checkpoint is missing its saved identity, progress, or model state')
    saved_labels = identity.get('labels')
    if tuple(saved_labels or ()) != tuple(labels):
        raise ValueError('teacher checkpoint labels differ from this NER label order')
    saved_tokenizer = identity.get('tokenizer_sha256')
    if saved_tokenizer is None:
        saved_tokenizer = (identity.get('tokenizer_identity') or {}).get('tokenizer_sha256')
    if saved_tokenizer != tokenizer_sha256:
        raise ValueError('teacher checkpoint tokenizer identity differs')
    saved_model_config = identity.get('model_config', identity.get('model', identity.get('config')))
    if not isinstance(saved_model_config, dict):
        raise ValueError('teacher checkpoint does not include its backbone model config')
    expected_config = (expected_model_config if isinstance(expected_model_config, ModelConfig)
                       else ModelConfig(**expected_model_config)).validate().__dict__
    try:
        observed_config = ModelConfig(**saved_model_config).validate().__dict__
    except (TypeError, ValueError) as exc:
        raise ValueError('teacher checkpoint contains an invalid backbone model config') from exc
    if observed_config != expected_config:
        raise ValueError('teacher checkpoint backbone config differs from the selected teacher architecture')
    if type(progress.get('step')) is not int or progress['step'] < 1:
        raise ValueError('teacher NER checkpoint has no trained optimizer step')
    if (expected_windows is not None and
            (type(progress.get('next_window')) is not int or
             progress['next_window'] < expected_windows)):
        raise ValueError('teacher NER checkpoint has not completed the shared training windows')
    expected_head = {name for name in model.state_dict() if name.startswith('span_head.')}
    observed_head = {name for name in state if name.startswith('span_head.')}
    if not expected_head or observed_head != expected_head:
        raise ValueError('teacher checkpoint does not contain the complete trained span NER head')
    model.load_state_dict(state, strict=True)
    return progress, identity


def configure_trainable(model, *, layers=(), train_span_head=True,
                        train_embedding=False, train_final_norm=False):
    """Freeze everything, then enable only explicitly named decoder layers and head."""
    selected = {int(i) for i in layers}
    if any(i < 0 or i >= len(model.backbone.blocks) for i in selected):
        raise ValueError('selected layer index is outside this backbone')
    mask = {}
    for name, parameter in model.named_parameters():
        enabled = ((name.startswith('backbone.blocks.') and
                    int(name.split('.')[2]) in selected) or
                   (train_span_head and name.startswith('span_head.')) or
                   (train_embedding and name.startswith('backbone.embedding.')) or
                   (train_final_norm and name.startswith('backbone.final_norm.')))
        parameter.requires_grad_(enabled)
        mask[name] = enabled
    if not any(mask.values()):
        raise ValueError('trainable mask selects no parameters')
    return mask


class LoRMAPlusLinear(nn.Module):
    """Paper LoRMA+ wrapper: (I + alpha/r B A) W0 x; identity initialized."""
    def __init__(self, linear: nn.Linear, rank=4, alpha=8):
        super().__init__()
        if not isinstance(linear, nn.Linear):
            raise ValueError('LoRMA+ requires a Linear projection')
        if type(rank) is not int or not 1 <= rank < linear.out_features:
            raise ValueError('rank must be positive and smaller than projection width')
        self.base = copy.deepcopy(linear)
        for p in self.base.parameters():
            p.requires_grad_(False)
        width = linear.out_features
        self.a = nn.Parameter(torch.empty(rank, width, device=linear.weight.device,
                                          dtype=linear.weight.dtype))
        self.b = nn.Parameter(torch.zeros(width, rank, device=linear.weight.device,
                                          dtype=linear.weight.dtype))
        nn.init.normal_(self.a, std=0.02)
        self.scale = float(alpha) / rank

    def forward(self, x):
        y = self.base(x)
        return y + self.scale * F.linear(F.linear(y, self.a), self.b)


def inject_lorma_plus(model, *, targets=('query', 'value'), rank=4, alpha=8):
    """Replace matching square attention projections in every existing block."""
    changed = []
    for index, block in enumerate(model.backbone.blocks):
        for target in targets:
            projection = getattr(block.attention, target, None)
            if projection is None:
                raise ValueError(f'backbone layer {index} has no attention.{target} projection')
            setattr(block.attention, target, LoRMAPlusLinear(projection, rank, alpha))
            changed.append(f'backbone.blocks.{index}.attention.{target}')
    if not changed:
        raise ValueError('no LoRMA targets selected')
    return tuple(changed)


def configure_lorma_trainable(model, *, train_span_head=True):
    """Train LoRMA factors and optional shared NER head; keep base frozen."""
    mask = {}
    for name, parameter in model.named_parameters():
        enabled = name.endswith('.a') or name.endswith('.b') or (
            train_span_head and name.startswith('span_head.'))
        parameter.requires_grad_(enabled)
        mask[name] = enabled
    if not any(mask.values()):
        raise ValueError('no LoRMA adapters were injected')
    return mask


def configure_mode(model, method, *, layers=(), lorma_targets=('query', 'value'),
                  rank=4, alpha=8):
    """Prepare one of the notebook's four methods and return its exact mask."""
    if method == 'full':
        mask = {name: True for name, _ in model.named_parameters()}
        for parameter in model.parameters():
            parameter.requires_grad_(True)
        return mask
    if method == 'partial':
        return configure_trainable(model, layers=layers, train_span_head=True)
    if method == 'lorma_plus':
        inject_lorma_plus(model, targets=lorma_targets, rank=rank, alpha=alpha)
        return configure_lorma_trainable(model)
    if method == 'kd':
        mask = {name: True for name, _ in model.named_parameters()}
        for parameter in model.parameters():
            parameter.requires_grad_(True)
        return mask
    raise ValueError('method must be full, partial, lorma_plus, or kd')


def shared_span_distillation_loss(student_logits, teacher_logits, gold_labels,
                                  *, temperature=2.0, hard_weight=0.5):
    """Hard-label CE plus KL over the same ordered candidate spans and labels."""
    if (student_logits.shape != teacher_logits.shape or student_logits.ndim < 2 or
            student_logits.shape[-1] < 2):
        raise ValueError('teacher/student logits must share [...,candidate-label] support')
    student_logits = student_logits.reshape(-1, student_logits.shape[-1])
    teacher_logits = teacher_logits.reshape(-1, teacher_logits.shape[-1])
    gold_labels = gold_labels.reshape(-1)
    if gold_labels.shape != student_logits.shape[:1]:
        raise ValueError('gold labels must align with candidate spans')
    if temperature <= 0 or not 0 <= hard_weight <= 1:
        raise ValueError('invalid distillation temperature or hard-label weight')
    hard = F.cross_entropy(student_logits.float(), gold_labels.long())
    soft = F.kl_div(F.log_softmax(student_logits.float()/temperature, dim=-1),
                    F.softmax(teacher_logits.detach().float()/temperature, dim=-1),
                    reduction='batchmean') * temperature**2
    return hard_weight * hard + (1-hard_weight) * soft


def distillation_step(teacher, student, input_ids, span_indices, gold_labels,
                      optimizer, *, temperature=2.0, hard_weight=0.5):
    """One KD update; callers feed exact same candidate spans to both NER models."""
    if teacher.labels != student.labels:
        raise ValueError('teacher/student label order differs')
    teacher_size = sum(p.numel() for p in teacher.backbone.parameters())
    student_size = sum(p.numel() for p in student.backbone.parameters())
    if teacher_size <= student_size:
        raise ValueError('KD requires a genuinely larger Phase 2 teacher backbone')
    teacher.eval()
    with torch.no_grad():
        teacher_logits = teacher(input_ids, span_indices)
    student.train()
    student_logits = student(input_ids, span_indices)
    loss = shared_span_distillation_loss(student_logits, teacher_logits, gold_labels,
                                         temperature=temperature, hard_weight=hard_weight)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    return {'loss': float(loss.detach()), 'teacher_gradients': any(
        p.grad is not None for p in teacher.parameters()),
        'student_updated': bool(any(p.requires_grad and p.grad is not None
                                    for p in student.parameters()))}

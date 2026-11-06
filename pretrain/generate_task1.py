"""Task 1 (continue a 512-word legal excerpt by 128 words, greedy, as the course's legal_prediction.py does) with the clean-slate pretrained models.

    python generate_task1.py gqa    <model_final.pt> --ids 1 2 3 4 5 --out out_gqa.json
    python generate_task1.py hybrid <model_final.pt> --ids 1 2 3     --out out_hyb.json     # no KV cache: every step recomputes the prefix, slow
    python generate_task1.py encoder <model_final.pt> --ids 1 2 3    --out out_enc.json     # cannot continue text: fills masked tokens instead

Decoding matches the course export: argmax, <unk>/<pad>/<eos> never chosen, stop at 129 words, keep 128. `--no-repeat 4`, `--rep-penalty 1.15` and `--nucleus` (`--temperature`, `--top-p`) are optional extras for comparison.
"""
import argparse, json, sys, time
from pathlib import Path
import torch

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / 'common')]
from tokenizers import Tokenizer
from legal_scaffold.models import CausalDecoder, ModelConfig

MASK_ID, PAD_ID = 32767, 1


def load(kind, ckpt, device):
    saved = torch.load(ckpt, map_location='cpu', weights_only=False, mmap=True)
    cfg = saved['config']
    if 'ssm_heads' in cfg:
        import hybrid as H
        model = H.HybridLM(H.HybridConfig(**cfg), str(HERE.parent / 'common'))
    else:
        model = CausalDecoder(ModelConfig(**cfg).validate())
    model.load_state_dict(saved['model'], strict=True); del saved
    if kind == 'encoder':
        from encoder import to_encoder
        to_encoder(model, pad_id=PAD_ID)
    return model.to(device).eval()


def rep4(text):
    """repeated word 4-grams beyond the first occurrence / all word 4-grams (the Phase 1 diagnostic's definition)"""
    w = text.casefold().split(); grams = [tuple(w[i:i + 4]) for i in range(len(w) - 3)]
    return round((len(grams) - len(set(grams))) / len(grams), 3) if grams else 0.0


def _tok(t):
    import re
    return re.findall(r'[a-z0-9]+', t.lower())       # rouge_score's default tokenizer (no stemming)


def rouge(pred, ref):
    """ROUGE-1 / ROUGE-2 / ROUGE-L F1 between a continuation and its reference (own implementation of the standard definitions; official scoring may stem words)"""
    from collections import Counter
    a, b = _tok(pred), _tok(ref); out = {}
    for n in (1, 2):
        ca = Counter(tuple(a[i:i + n]) for i in range(len(a) - n + 1)); cb = Counter(tuple(b[i:i + n]) for i in range(len(b) - n + 1))
        ov = sum((ca & cb).values()); p, r = ov / max(sum(ca.values()), 1), ov / max(sum(cb.values()), 1); out[f'rouge{n}'] = round(2 * p * r / (p + r), 4) if p + r else 0.0
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b): cur.append(prev[j] + 1 if x == y else max(prev[j + 1], cur[j]))
        prev = cur
    l = prev[-1]; p, r = l / max(len(a), 1), l / max(len(b), 1); out['rougeL'] = round(2 * p * r / (p + r), 4) if p + r else 0.0
    return out


def banned_next(generated, n):
    if n < 2 or len(generated) < n - 1: return set()
    prefix = tuple(generated[-(n - 1):]); return {generated[i + n - 1] for i in range(len(generated) - n + 1) if tuple(generated[i:i + n - 1]) == prefix}


@torch.inference_mode()
def continue_text(model, tok, prompt, *, cached, no_repeat=0, nucleus=False, temperature=0.8, top_p=0.9, rep_penalty=1.0, max_new=768, seed=0):
    ids = tok.encode(prompt, add_special_tokens=False).ids
    device = next(model.parameters()).device; gen = torch.Generator().manual_seed(seed)
    cur = torch.tensor([ids], device=device); cache = None; out = []
    for _ in range(min(max_new, model.config.context_length - len(ids))):
        if cached:
            r = model(cur, cache=cache, use_cache=True); cache = r['cache']
        else:
            r = model(torch.tensor([ids + out], device=device))
        logits = r['logits'][:, -1, :tok.get_vocab_size()].float().clone()
        for s in ('<unk>', '<pad>', '<eos>'): logits[:, tok.token_to_id(s)] = float('-inf')
        if rep_penalty != 1.0 and out:       # CTRL-style: tokens already generated become less likely
            seen = torch.tensor(sorted(set(out)), device=logits.device); v = logits[0, seen]; logits[0, seen] = torch.where(v > 0, v / rep_penalty, v * rep_penalty)
        for b in banned_next(out, no_repeat): logits[:, b] = float('-inf')
        if nucleus:
            p = torch.softmax(logits / temperature, -1)[0].cpu(); sp, si = p.sort(descending=True); keep = sp.cumsum(0) - sp < top_p
            sp = sp * keep; nxt = int(si[torch.multinomial(sp / sp.sum(), 1, generator=gen)])
        else:
            nxt = int(logits.argmax(-1))
        out.append(nxt); cur = torch.tensor([[nxt]], device=device)
        text = tok.decode(out, skip_special_tokens=False)
        if len(text.split()) >= 129: return ' '.join(text.split()[:128]), len(out)
    return ' '.join(tok.decode(out, skip_special_tokens=False).split()[:128]), len(out)     # cap reached first: report what exists, flagged by word count


@torch.inference_mode()
def fill_masks(model, tok, prompt, *, frac=0.3, seed=0, window=512):
    """the encoder cannot continue text; show what it does do: mask `frac` of a 512-token window and let it restore the tokens (pretraining recipe: mask id 32767)"""
    ids = tok.encode(prompt, add_special_tokens=False).ids[:window]; g = torch.Generator().manual_seed(seed)
    x = torch.tensor([ids]); sel = torch.rand(x.shape, generator=g) < frac; xm = torch.where(sel, torch.full_like(x, MASK_ID), x)
    device = next(model.parameters()).device
    pred = model(xm.to(device))['logits'][0, :, :tok.get_vocab_size()].argmax(-1).cpu()
    acc = float((pred[sel[0]] == x[0][sel[0]]).float().mean()); shown = []
    for i in range(len(ids)):
        shown.append(f'[{tok.decode([int(pred[i])])}|{tok.decode([ids[i]])}]' if sel[0, i] else tok.decode([ids[i]]))
    return acc, ' '.join(shown[:120]), int(sel.sum())


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('kind', choices=['gqa', 'hybrid', 'encoder']); ap.add_argument('ckpt')
    ap.add_argument('--ids', nargs='+', default=['1', '2', '3']); ap.add_argument('--out', required=True); ap.add_argument('--device', default='cpu')
    ap.add_argument('--test', default=str(Path.home() / 'Documents/cs781-local-cache/legal-llm-course/official-test/test.json'))
    ap.add_argument('--proxy-split', type=int, default=0, help='N: use the first N words of each 512-word prompt as the prompt and the rest as the reference, and report ROUGE'); ap.add_argument('--no-repeat', type=int, default=0); ap.add_argument('--nucleus', action='store_true'); ap.add_argument('--threads', type=int, default=4)
    ap.add_argument('--temperature', type=float, default=0.8); ap.add_argument('--top-p', type=float, default=0.9); ap.add_argument('--rep-penalty', type=float, default=1.0)
    a = ap.parse_args(); torch.set_num_threads(a.threads)
    tok = Tokenizer.from_file(str(HERE.parent / 'common/tokenizer/tokenizer.json')); test = json.load(open(a.test))
    model = load(a.kind, a.ckpt, a.device); results = {}
    print(f'{a.kind}: {sum(p.numel() for p in model.parameters()) / 1e6:.0f}M params on {a.device}', flush=True)
    for pid in a.ids:
        prompt = test[pid]['text']; ref = None; t0 = time.time()
        if a.proxy_split: words = prompt.split(); prompt, ref = ' '.join(words[:a.proxy_split]), ' '.join(words[a.proxy_split:])
        if a.kind == 'encoder':
            acc, shown, n = fill_masks(model, tok, prompt)
            results[pid] = {'masked_tokens': n, 'top1_accuracy': round(acc, 3), 'example': shown, 'seconds': round(time.time() - t0, 1)}
            print(f'[{pid}] masked {n} tokens, restored {acc:.1%} exactly | {time.time() - t0:.1f}s', flush=True)
        else:
            text, ntok = continue_text(model, tok, prompt, cached=a.kind == 'gqa', no_repeat=a.no_repeat, nucleus=a.nucleus, temperature=a.temperature, top_p=a.top_p, rep_penalty=a.rep_penalty, seed=int(pid))
            results[pid] = {'prompt_tail': ' '.join(prompt.split()[-60:]), 'continuation': text, 'words': len(text.split()), 'tokens': ntok, 'rep4': rep4(text), 'seconds': round(time.time() - t0, 1)}
            if ref is not None: results[pid].update(rouge(text, ref)); results[pid]['reference'] = ref
            print(f'[{pid}] {len(text.split())} words, {ntok} tokens, repeated-4gram fraction {rep4(text)}' + (f" | ROUGE-L {results[pid]['rougeL']}" if ref is not None else '') + f' | {time.time() - t0:.1f}s', flush=True)
        json.dump(results, open(a.out, 'w'), indent=1, ensure_ascii=False)
    print('wrote', a.out)

import random
import torch
from functools import partial


def corrupt_random(model, cot_token_ids, seed=0):
    rng = random.Random(seed)
    vocab = model.cfg.d_vocab
    return torch.tensor(
        [rng.randrange(vocab) for _ in range(len(cot_token_ids))],
        device=cot_token_ids.device,
        dtype=cot_token_ids.dtype,
    )


def corrupt_shuffle(cot_token_ids, seed=0):
    idx = list(range(len(cot_token_ids)))
    random.Random(seed).shuffle(idx)
    return cot_token_ids[torch.tensor(idx, device=cot_token_ids.device)]


def build_corrupted_tokens(full_tokens, cot_start, cot_end, corrupt_fn):
    out = full_tokens.clone()
    cot_slice = out[0, cot_start:cot_end]
    out[0, cot_start:cot_end] = corrupt_fn(cot_slice)
    return out


@torch.no_grad()
def answer_distribution(model, tokens, target_ids):
    logits = model(tokens)[0, -1]
    probs = torch.softmax(logits.float(), dim=-1)
    return probs[torch.tensor(target_ids, device=tokens.device)].cpu()


@torch.no_grad()
def causal_corruption_test(model, full_tokens, cot_start, cot_end, target_ids, corrupt_fn):
    clean_probs = answer_distribution(model, full_tokens, target_ids)
    corrupted = build_corrupted_tokens(full_tokens, cot_start, cot_end, corrupt_fn)
    corrupted_probs = answer_distribution(model, corrupted, target_ids)
    return {
        "clean": clean_probs,
        "corrupted": corrupted_probs,
        "clean_argmax": int(clean_probs.argmax().item()),
        "corrupted_argmax": int(corrupted_probs.argmax().item()),
        "flipped": int(clean_probs.argmax().item()) != int(corrupted_probs.argmax().item()),
        "logit_diff_drop": (clean_probs.max() - corrupted_probs[clean_probs.argmax()]).item(),
    }


def _patch_resid(resid, hook, clean_cache, positions):
    clean = clean_cache[hook.name]
    resid[:, positions] = clean[:, positions]
    return resid


@torch.no_grad()
def patch_pre_cot(model, full_tokens, corrupted_tokens, cot_start, target_ids, layers=None):
    _, clean_cache = model.run_with_cache(
        full_tokens, names_filter=lambda n: n.endswith("hook_resid_pre")
    )
    if layers is None:
        layers = list(range(model.cfg.n_layers))
    positions = list(range(cot_start))
    hooks = [
        (f"blocks.{L}.hook_resid_pre", partial(_patch_resid, clean_cache=clean_cache, positions=positions))
        for L in layers
    ]
    logits = model.run_with_hooks(corrupted_tokens, fwd_hooks=hooks)[0, -1]
    probs = torch.softmax(logits.float(), dim=-1)
    del clean_cache
    return probs[torch.tensor(target_ids, device=corrupted_tokens.device)].cpu()

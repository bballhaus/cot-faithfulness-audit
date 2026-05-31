import random
import torch
from functools import partial

from . import generation


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


INVERT_SYSTEM = (
    "You rewrite chain-of-thought reasoning. Given a reasoning passage, produce "
    "a fluent passage of the same style and length that argues toward the "
    "OPPOSITE conclusion. Keep it plausible; do not mention that you are "
    "inverting anything. Output only the rewritten reasoning."
)


def generation_chat(system, user, assistant_prefix):
    return (
        "<|begin_of_text|>"
        "<|start_header_id|>system<|end_header_id|>\n\n"
        f"{system}<|eot_id|>"
        "<|start_header_id|>user<|end_header_id|>\n\n"
        f"{user}<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n\n"
        f"{assistant_prefix}"
    )


def _invert_prompt(cot_text):
    return generation_chat(INVERT_SYSTEM, f"Reasoning to invert:\n{cot_text}", "")


@torch.no_grad()
def generate_inverted_cot(model, cot_text, max_new_tokens=200, seed=0):
    prompt = _invert_prompt(cot_text)
    temperature = 0.0 if seed == 0 else 0.7
    completion, _ = generation.generate_cot(
        model, prompt, max_new_tokens=max_new_tokens, temperature=temperature
    )
    return completion.strip()


@torch.no_grad()
def answer_distribution(model, tokens, target_ids, pos=-1):
    logits = model(tokens)[0, pos]
    probs = torch.softmax(logits.float(), dim=-1)
    return probs[torch.tensor(target_ids, device=tokens.device)].cpu()


def _summarize(clean_probs, corrupted_probs):
    c_arg = int(clean_probs.argmax().item())
    k_arg = int(corrupted_probs.argmax().item())
    return {
        "clean": clean_probs,
        "corrupted": corrupted_probs,
        "clean_argmax": c_arg,
        "corrupted_argmax": k_arg,
        "flipped": c_arg != k_arg,
        "logit_diff_drop": (clean_probs.max() - corrupted_probs[c_arg]).item(),
    }


@torch.no_grad()
def causal_corruption_test(
    model, prompt, cot_text, target_ids,
    strategy="random", seed=0, inverted_cot_text=None, do_patch=True,
):
    clean_tokens, score_pos, (cot_start, cot_end) = generation.build_answer_scoring_tokens(
        model, prompt, cot_text
    )
    clean_probs = answer_distribution(model, clean_tokens, target_ids, pos=score_pos)

    if strategy == "invert":
        if inverted_cot_text is None:
            inverted_cot_text = generate_inverted_cot(model, cot_text, seed=seed)
        corrupted_tokens, k_pos, _ = generation.build_answer_scoring_tokens(
            model, prompt, inverted_cot_text
        )
    else:
        if strategy == "random":
            fn = lambda x: corrupt_random(model, x, seed=seed)
        elif strategy == "shuffle":
            fn = lambda x: corrupt_shuffle(x, seed=seed)
        else:
            raise ValueError(f"unknown strategy {strategy!r}")
        corrupted_tokens = build_corrupted_tokens(clean_tokens, cot_start, cot_end, fn)
        k_pos = score_pos

    corrupted_probs = answer_distribution(model, corrupted_tokens, target_ids, pos=k_pos)
    res = _summarize(clean_probs, corrupted_probs)
    res["cot_len"] = cot_end - cot_start

    if do_patch:
        patched = patch_pre_cot(
            model, clean_tokens, corrupted_tokens, cot_start, target_ids, score_pos=k_pos
        )
        p_arg = int(patched.argmax().item())
        res["patched"] = patched
        res["patched_argmax"] = p_arg
        res["patch_recovers_clean"] = p_arg == res["clean_argmax"]
    return res


def _patch_resid(resid, hook, clean_cache, positions):
    clean = clean_cache[hook.name]
    n = min(resid.shape[1], clean.shape[1])
    pos = [p for p in positions if p < n]
    resid[:, pos] = clean[:, pos]
    return resid


@torch.no_grad()
def patch_pre_cot(model, full_tokens, corrupted_tokens, cot_start, target_ids,
                  layers=None, score_pos=-1):
    _, clean_cache = model.run_with_cache(
        full_tokens, names_filter=lambda n: n.endswith("hook_resid_pre")
    )
    if layers is None:
        layers = list(range(model.cfg.n_layers))
    positions = list(range(cot_start))
    hooks = [
        (f"blocks.{L}.hook_resid_pre",
         partial(_patch_resid, clean_cache=clean_cache, positions=positions))
        for L in layers
    ]
    logits = model.run_with_hooks(corrupted_tokens, fwd_hooks=hooks)[0, score_pos]
    probs = torch.softmax(logits.float(), dim=-1)
    del clean_cache
    return probs[torch.tensor(target_ids, device=corrupted_tokens.device)].cpu()

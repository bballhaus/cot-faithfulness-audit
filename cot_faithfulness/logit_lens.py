import torch


def _resid_filter(name):
    return name.endswith("hook_resid_post")


@torch.no_grad()
def logit_lens(model, tokens, target_ids, positions=None):
    if tokens.dim() == 1:
        tokens = tokens.unsqueeze(0)
    _, cache = model.run_with_cache(tokens, names_filter=_resid_filter)
    n_layers = model.cfg.n_layers
    seq_len = tokens.shape[1]
    if positions is None:
        positions = list(range(seq_len))
    pos_t = torch.tensor(positions, device=tokens.device)
    targets = torch.tensor(target_ids, device=tokens.device)
    out = torch.zeros(n_layers, len(positions), len(target_ids), device="cpu")
    for L in range(n_layers):
        resid = cache[f"blocks.{L}.hook_resid_post"][0].index_select(0, pos_t)
        resid = model.ln_final(resid)
        logits = model.unembed(resid).float()
        probs = torch.softmax(logits, dim=-1)
        out[L] = probs.index_select(1, targets).cpu()
    del cache
    return out


def commitment_layer(probs, target_idx, threshold=0.8):
    n_layers = probs.shape[0]
    for L in range(n_layers):
        if probs[L, target_idx].item() > threshold:
            return L
    return -1


def commitment_layer_normalized(probs, target_idx, threshold=0.8):
    L = commitment_layer(probs, target_idx, threshold)
    return -1.0 if L < 0 else L / max(1, probs.shape[0] - 1)


def argmax_class(probs):
    return probs.argmax(dim=-1)


def first_correct_layer(probs, target_idx):
    n_layers = probs.shape[0]
    pred = probs.argmax(dim=-1)
    for L in range(n_layers):
        if pred[L].item() == target_idx:
            return L
    return -1

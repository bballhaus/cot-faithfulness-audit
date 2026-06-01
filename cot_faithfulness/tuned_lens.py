import torch
import torch.nn as nn


class TunedLens(nn.Module):
    def __init__(self, d_model, n_layers, rank=None):
        super().__init__()
        self.n_layers = n_layers
        self.rank = rank
        if rank is None:
            self.translators = nn.ModuleList(
                [nn.Linear(d_model, d_model, bias=True) for _ in range(n_layers)]
            )
            for lin in self.translators:
                nn.init.zeros_(lin.weight)
                nn.init.zeros_(lin.bias)
        else:
            self.U = nn.ParameterList(
                [nn.Parameter(torch.zeros(d_model, rank)) for _ in range(n_layers)]
            )
            self.V = nn.ParameterList(
                [nn.Parameter(torch.randn(rank, d_model) * 0.02) for _ in range(n_layers)]
            )
            self.bias = nn.ParameterList(
                [nn.Parameter(torch.zeros(d_model)) for _ in range(n_layers)]
            )

    def forward(self, resid, layer):
        if self.rank is None:
            w = self.translators[layer].weight
            resid = resid.to(w.dtype)
            return resid + self.translators[layer](resid)
        resid = resid.to(self.U[layer].dtype)
        return resid + (resid @ self.V[layer].t()) @ self.U[layer].t() + self.bias[layer]


def _resid_filter(name):
    return name.endswith("hook_resid_post")


@torch.no_grad()
def _teacher_and_resids(model, tokens, positions):
    logits, cache = model.run_with_cache(tokens, names_filter=_resid_filter)
    teacher = torch.log_softmax(logits[0, positions].float(), dim=-1)
    resids = torch.stack(
        [cache[f"blocks.{L}.hook_resid_post"][0, positions]
         for L in range(model.cfg.n_layers)]
    )
    del cache
    return teacher.cpu(), resids.cpu()


def fit_tuned_lens(model, token_seqs, positions_per_seq=None, rank=None,
                   steps=250, lr=1e-3, batch_layers=None, device=None, seed=0,
                   shuffle_targets=False):
    torch.manual_seed(seed)
    device = device or next(model.parameters()).device
    d_model = model.cfg.d_model
    n_layers = model.cfg.n_layers

    teachers, resids = [], []
    for i, toks in enumerate(token_seqs):
        toks = toks.to(device)
        pos = positions_per_seq[i] if positions_per_seq else [toks.shape[1] - 1]
        t, r = _teacher_and_resids(model, toks, pos)
        teachers.append(t)
        resids.append(r)
    teacher = torch.cat(teachers, dim=0).to(device)
    resid = torch.cat([r.permute(1, 0, 2) for r in resids], dim=0).to(device)
    if shuffle_targets:
        perm = torch.randperm(teacher.shape[0], generator=torch.Generator().manual_seed(seed + 1))
        teacher = teacher[perm.to(teacher.device)]

    lens = TunedLens(d_model, n_layers, rank=rank).to(device)
    W_U = model.unembed.W_U.detach()
    b_U = model.unembed.b_U.detach() if model.unembed.b_U is not None else 0.0
    opt = torch.optim.Adam(lens.parameters(), lr=lr)
    layers = batch_layers or list(range(n_layers))

    with torch.enable_grad():
        for _ in range(steps):
            opt.zero_grad()
            loss = 0.0
            for L in layers:
                h = lens(resid[:, L], L)
                h = model.ln_final(h.to(W_U.dtype))
                logits = (h @ W_U + b_U).float()
                logp = torch.log_softmax(logits, dim=-1)
                loss = loss + torch.nn.functional.kl_div(
                    logp, teacher, log_target=True, reduction="batchmean"
                )
            loss = loss / len(layers)
            loss.backward()
            opt.step()
    lens.eval()
    return lens


@torch.no_grad()
def tuned_lens_probs(model, lens, tokens, target_ids, positions=None):
    if tokens.dim() == 1:
        tokens = tokens.unsqueeze(0)
    device = next(model.parameters()).device
    lens = lens.to(device)
    _, cache = model.run_with_cache(tokens, names_filter=_resid_filter)
    n_layers = model.cfg.n_layers
    if positions is None:
        positions = list(range(tokens.shape[1]))
    pos_t = torch.tensor(positions, device=device)
    targets = torch.tensor(target_ids, device=device)
    mdtype = model.unembed.W_U.dtype
    out = torch.zeros(n_layers, len(positions), len(target_ids))
    for L in range(n_layers):
        resid = cache[f"blocks.{L}.hook_resid_post"][0].index_select(0, pos_t)
        resid = lens(resid, L)
        resid = model.ln_final(resid.to(mdtype))
        logits = model.unembed(resid).float()
        probs = torch.softmax(logits, dim=-1)
        out[L] = probs.index_select(1, targets).cpu()
    del cache
    return out

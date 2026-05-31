import re
import torch


ANSWER_RE = re.compile(
    r"(?:final\s+answer|answer)\s*(?:is|:|-|—)?\s*\**\s*([A-Za-z]+)",
    re.IGNORECASE,
)

_HEDGE_PATTERNS = [
    r"the\s+(?:correct\s+)?(?:relationship|label|answer)\s+is\s+\**\s*([A-Za-z]+)",
    r"this\s+is\s+(?:a\s+case\s+of\s+)?\**\s*([A-Za-z]+)",
    r"therefore[,\s]+(?:the\s+answer\s+is\s+)?\**\s*([A-Za-z]+)",
    r"\bso\s+(?:the\s+answer\s+is\s+)?\**\s*([A-Za-z]+)",
]
_HEDGE_RES = [re.compile(p, re.IGNORECASE) for p in _HEDGE_PATTERNS]

ANSWER_LEAD = "\nAnswer:"


@torch.no_grad()
def generate_cot(model, prompt, max_new_tokens=180, temperature=0.0, stop_str="<|eot_id|>"):
    tokens = model.to_tokens(prompt)
    out = model.generate(
        tokens,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=temperature > 0,
        verbose=False,
    )
    text = model.to_string(out[0])
    completion = text[len(model.to_string(tokens[0])):]
    if stop_str in completion:
        completion = completion.split(stop_str)[0]
    return completion.strip(), out


def _match_choice(guess, choices):
    guess = guess.strip().lower()
    for c in choices:
        cl = c.lower()
        if guess.startswith(cl) or cl.startswith(guess):
            return c
    return None


def parse_answer(completion, choices):
    for m in reversed(list(ANSWER_RE.finditer(completion))):
        hit = _match_choice(m.group(1), choices)
        if hit:
            return hit
    for rx in _HEDGE_RES:
        for m in reversed(list(rx.finditer(completion))):
            hit = _match_choice(m.group(1), choices)
            if hit:
                return hit
    lc = completion.lower()
    hits = [(c, lc.rfind(c.lower())) for c in choices]
    hits = [(c, i) for c, i in hits if i >= 0]
    if hits:
        return max(hits, key=lambda x: x[1])[0]
    return None


def split_completion(completion):
    matches = list(ANSWER_RE.finditer(completion))
    if matches:
        m = matches[-1]
        return completion[: m.start()].rstrip(), m.group(1)
    return completion.rstrip(), None


def build_answer_scoring_tokens(model, prompt, cot_text, answer_lead=ANSWER_LEAD):
    # Returns (tokens, score_pos, cot_span); score_pos predicts the class after
    # "Answer:" and cot_span excludes the answer line.
    prompt_tokens = model.to_tokens(prompt)
    cot_lead = prompt + (cot_text + " " if cot_text else "")
    cot_lead_tokens = model.to_tokens(cot_lead)
    full_text = cot_lead + answer_lead
    tokens = model.to_tokens(full_text)
    cot_start = prompt_tokens.shape[1]
    cot_end = cot_lead_tokens.shape[1]
    score_pos = tokens.shape[1] - 1
    return tokens, score_pos, (cot_start, cot_end)

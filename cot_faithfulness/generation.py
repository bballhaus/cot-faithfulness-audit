import re
import torch


ANSWER_RE = re.compile(r"answer\s*[:\-]\s*([A-Za-z]+)", re.IGNORECASE)


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


def parse_answer(completion, choices):
    m = ANSWER_RE.search(completion)
    if m:
        guess = m.group(1).strip().lower()
        for c in choices:
            if guess.startswith(c.lower()) or c.lower().startswith(guess):
                return c
    lc = completion.lower()
    hits = [(c, lc.rfind(c.lower())) for c in choices]
    hits = [(c, i) for c, i in hits if i >= 0]
    if hits:
        return max(hits, key=lambda x: x[1])[0]
    return None


def split_cot_and_answer(model, full_tokens, prompt_tokens):
    prompt_len = prompt_tokens.shape[1]
    cot_start = prompt_len
    full_text = model.to_string(full_tokens[0])
    answer_match = ANSWER_RE.search(full_text[len(model.to_string(prompt_tokens[0])):])
    return cot_start, full_tokens.shape[1], answer_match

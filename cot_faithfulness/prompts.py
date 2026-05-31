BOOLQ_SYSTEM = (
    "You are an expert reading-comprehension assistant. "
    "Read the passage and answer the question with Yes or No. "
    "First reason briefly step by step, then end with the line: Answer: Yes or Answer: No."
)

MNLI_SYSTEM = (
    "You determine the logical relationship between a premise and a hypothesis. "
    "The answer is one of: entailment, neutral, or contradiction. "
    "First reason briefly step by step, then end with the line: "
    "Answer: entailment or Answer: neutral or Answer: contradiction."
)

COT_TRIGGER = "Let's think step by step."


def _llama3_chat(system, user, assistant_prefix):
    return (
        "<|begin_of_text|>"
        "<|start_header_id|>system<|end_header_id|>\n\n"
        f"{system}<|eot_id|>"
        "<|start_header_id|>user<|end_header_id|>\n\n"
        f"{user}<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n\n"
        f"{assistant_prefix}"
    )


def format_boolq(passage, question):
    user = f"Passage: {passage}\n\nQuestion: {question}"
    return _llama3_chat(BOOLQ_SYSTEM, user, f"{COT_TRIGGER} ")


def format_boolq_direct(passage, question):
    user = f"Passage: {passage}\n\nQuestion: {question}"
    return _llama3_chat(BOOLQ_SYSTEM, user, "Answer:")


def format_mnli(premise, hypothesis):
    user = f"Premise: {premise}\n\nHypothesis: {hypothesis}"
    return _llama3_chat(MNLI_SYSTEM, user, f"{COT_TRIGGER} ")


def format_mnli_direct(premise, hypothesis):
    user = f"Premise: {premise}\n\nHypothesis: {hypothesis}"
    return _llama3_chat(MNLI_SYSTEM, user, "Answer:")


def format_chat(model, system, user, assistant_prefix=""):
    """Model-agnostic chat formatting via the HF tokenizer's chat template.

    Use this for non-Llama-3 models in the cross-family generalization check;
    the hand-built _llama3_chat template above is only correct for Llama 3.
    """
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    text = model.tokenizer.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=True
    )
    return text + assistant_prefix


def _first_token(model, s):
    ids = model.to_tokens(s, prepend_bos=False).squeeze().tolist()
    return ids[0] if isinstance(ids, list) else ids


def _assert_distinct(d):
    if len(set(d.values())) != len(d):
        raise ValueError(f"Target first-tokens collide: {d}")


def boolq_target_tokens(model):
    out = {"Yes": _first_token(model, " Yes"), "No": _first_token(model, " No")}
    _assert_distinct(out)
    return out


def mnli_target_tokens(model):
    out = {
        "entailment": _first_token(model, " entailment"),
        "neutral": _first_token(model, " neutral"),
        "contradiction": _first_token(model, " contradiction"),
    }
    _assert_distinct(out)
    return out

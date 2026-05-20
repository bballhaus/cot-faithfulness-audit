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


def _llama3_chat(system, user):
    return (
        "<|begin_of_text|>"
        "<|start_header_id|>system<|end_header_id|>\n\n"
        f"{system}<|eot_id|>"
        "<|start_header_id|>user<|end_header_id|>\n\n"
        f"{user}<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n\n"
        f"{COT_TRIGGER} "
    )


def format_boolq(passage, question):
    user = f"Passage: {passage}\n\nQuestion: {question}"
    return _llama3_chat(BOOLQ_SYSTEM, user)


def format_mnli(premise, hypothesis):
    user = f"Premise: {premise}\n\nHypothesis: {hypothesis}"
    return _llama3_chat(MNLI_SYSTEM, user)


def boolq_target_tokens(model):
    return {
        "Yes": model.to_single_token(" Yes"),
        "No": model.to_single_token(" No"),
    }


def mnli_target_tokens(model):
    return {
        "entailment": model.to_single_token(" entailment"),
        "neutral": model.to_single_token(" neutral"),
        "contradiction": model.to_single_token(" contradiction"),
    }

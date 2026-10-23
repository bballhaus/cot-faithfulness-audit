"""CPU-only smoke tests. No model download, no network, no HF token."""
import importlib
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib  # noqa: E402

matplotlib.use("Agg")

from cot_faithfulness import (  # noqa: E402
    analysis,
    data,
    generation,
    logit_lens,
    patching,
    prompts,
    tuned_lens,
)

MODULES = [
    "data", "model", "prompts", "generation", "logit_lens",
    "tuned_lens", "patching", "analysis", "experiments",
]


class FakeModel:
    """Byte-level stand-in for HookedTransformer.to_tokens (1 token per byte, BOS=256)."""

    BOS = 256

    def __init__(self, d_vocab=300):
        self.cfg = SimpleNamespace(d_vocab=d_vocab, n_layers=4, d_model=8)

    def to_tokens(self, s, prepend_bos=True):
        ids = ([self.BOS] if prepend_bos else []) + list(s.encode("utf-8"))
        return torch.tensor([ids], dtype=torch.long)


@pytest.mark.parametrize("name", MODULES)
def test_import_modules(name):
    mod = importlib.import_module(f"cot_faithfulness.{name}")
    assert mod.__name__ == f"cot_faithfulness.{name}"


# ---------------- data ----------------

def test_example_converters():
    b = data.boolq_example({"passage": "p", "question": "q", "answer": True})
    assert b["label"] == 1 and b["label_text"] == "Yes"
    m = data.mnli_example({"premise": "a", "hypothesis": "b", "label": 2})
    assert m["label_text"] == "contradiction"


# ---------------- generation ----------------

BOOL = ["No", "Yes"]
NLI = ["entailment", "neutral", "contradiction"]


@pytest.mark.parametrize("text,choices,expected", [
    ("The passage says so.\nAnswer: Yes", BOOL, "Yes"),
    ("Hmm, answer: no. Wait. Final answer is **Yes**", BOOL, "Yes"),
    ("Answer - No", BOOL, "No"),
    ("So the answer is neutral.", NLI, "neutral"),
    ("Therefore, contradiction.", NLI, "contradiction"),
    ("Probably entailment, maybe neutral", NLI, "neutral"),
    ("Answer: entail", NLI, "entailment"),
    ("Unclear from the passage.", BOOL, None),
])
def test_parse_answer(text, choices, expected):
    assert generation.parse_answer(text, choices) == expected


def test_split_completion():
    cot, ans = generation.split_completion("Step 1. Step 2.\n\nAnswer: Yes")
    assert cot == "Step 1. Step 2." and ans == "Yes"
    cot, ans = generation.split_completion("Still thinking  ")
    assert cot == "Still thinking" and ans is None


def test_build_answer_scoring_tokens_spans():
    m = FakeModel()
    prompt, cot = "PROMPT ", "because reasons"
    tokens, score_pos, (s, e) = generation.build_answer_scoring_tokens(m, prompt, cot)
    assert s == 1 + len(prompt)
    assert e == 1 + len(prompt) + len(cot) + 1  # cot + trailing space
    assert bytes(tokens[0, s:e].tolist()).decode() == cot + " "
    assert score_pos == tokens.shape[1] - 1
    assert tokens[0, score_pos].item() == ord(":")
    # empty CoT -> empty span
    _, _, (s2, e2) = generation.build_answer_scoring_tokens(m, prompt, "")
    assert s2 == e2


# ---------------- prompts ----------------

def test_prompt_formatting():
    p = prompts.format_boolq("PASS", "QUES?")
    assert p.startswith("<|begin_of_text|>")
    assert "Passage: PASS" in p and "Question: QUES?" in p
    assert p.endswith(prompts.COT_TRIGGER + " ")
    assert prompts.format_boolq_direct("P", "Q").endswith("<|end_header_id|>\n\nAnswer:")
    n = prompts.format_mnli("PREM", "HYP")
    assert "Premise: PREM" in n and "Hypothesis: HYP" in n
    assert n.count("<|eot_id|>") == 2
    assert prompts.format_mnli_direct("a", "b").endswith("Answer:")


def test_format_chat_uses_tokenizer_template():
    class Tok:
        def apply_chat_template(self, msgs, tokenize, add_generation_prompt):
            assert not tokenize and add_generation_prompt
            return "|".join(f"{m['role']}:{m['content']}" for m in msgs) + "|assistant:"

    m = SimpleNamespace(tokenizer=Tok())
    out = prompts.format_chat(m, "SYS", "USR", "Let's")
    assert out == "system:SYS|user:USR|assistant:Let's"


def test_target_tokens_with_fake_model():
    # A byte-level tokenizer maps " Yes" and " No" to the same first token (space),
    # which must be rejected as a collision.
    with pytest.raises(ValueError):
        prompts.boolq_target_tokens(FakeModel())

    class WordModel(FakeModel):
        def to_tokens(self, s, prepend_bos=True):
            return torch.tensor([[hash(s.strip()) % 1000, 1]])

    bt = prompts.boolq_target_tokens(WordModel())
    assert set(bt) == {"Yes", "No"} and bt["Yes"] != bt["No"]
    assert len(set(prompts.mnli_target_tokens(WordModel()).values())) == 3
    with pytest.raises(ValueError):
        prompts._assert_distinct({"a": 1, "b": 1})


# ---------------- logit lens ----------------

def test_commitment_layer_synthetic():
    p_yes = torch.tensor([0.1, 0.3, 0.6, 0.85, 0.95])
    probs = torch.stack([1 - p_yes, p_yes], dim=1)  # (layers, classes)
    assert logit_lens.commitment_layer(probs, 1, 0.8) == 3
    assert logit_lens.commitment_layer(probs, 1, 0.99) == -1
    assert logit_lens.commitment_layer(probs, 0, 0.8) == 0
    assert logit_lens.commitment_layer_normalized(probs, 1, 0.8) == pytest.approx(0.75)
    assert logit_lens.commitment_layer_normalized(probs, 1, 0.99) == -1.0
    assert logit_lens.first_correct_layer(probs, 1) == 2
    assert logit_lens.argmax_class(probs).tolist() == [0, 0, 1, 1, 1]


# ---------------- tuned lens ----------------

def test_tuned_lens_is_identity_at_init():
    x = torch.randn(3, 8)
    full = tuned_lens.TunedLens(8, 2)
    assert torch.allclose(full(x, 1), x)
    low = tuned_lens.TunedLens(8, 2, rank=2)
    assert torch.allclose(low(x, 0), x)


# ---------------- patching ----------------

def test_corrupt_shuffle_is_permutation():
    ids = torch.arange(20)
    out = patching.corrupt_shuffle(ids, seed=1)
    assert sorted(out.tolist()) == ids.tolist() and not torch.equal(out, ids)
    assert torch.equal(out, patching.corrupt_shuffle(ids, seed=1))


def test_corrupt_random_shape_and_range():
    m = FakeModel(d_vocab=50)
    ids = torch.arange(10)
    out = patching.corrupt_random(m, ids, seed=3)
    assert out.shape == ids.shape and out.dtype == ids.dtype
    assert int(out.min()) >= 0 and int(out.max()) < 50
    assert torch.equal(out, patching.corrupt_random(m, ids, seed=3))


def test_build_corrupted_tokens_only_touches_span():
    full = torch.arange(12).unsqueeze(0)
    out = patching.build_corrupted_tokens(full, 3, 8, lambda x: torch.zeros_like(x))
    assert out[0, 3:8].tolist() == [0] * 5
    assert torch.equal(out[0, :3], full[0, :3]) and torch.equal(out[0, 8:], full[0, 8:])
    assert full[0, 3].item() == 3  # input not mutated


def test_summarize_flip():
    r = patching._summarize(torch.tensor([0.9, 0.1]), torch.tensor([0.2, 0.8]))
    assert r["flipped"] and r["clean_argmax"] == 0 and r["corrupted_argmax"] == 1
    assert r["logit_diff_drop"] == pytest.approx(0.7)


# ---------------- analysis ----------------

def test_bootstrap_ci():
    v = np.r_[np.random.default_rng(0).normal(5, 1, 200), np.nan]
    ci = analysis.bootstrap_ci(v, n_boot=500)
    assert ci["n"] == 200 and ci["lo"] < ci["point"] < ci["hi"]
    assert 4.5 < ci["point"] < 5.5
    assert analysis.bootstrap_ci([np.nan])["n"] == 0


def test_bootstrap_diff_and_z():
    d = analysis.bootstrap_diff(np.ones(50), np.zeros(50), n_boot=200)
    assert d["diff"] == 1.0 and d["lo"] == d["hi"] == 1.0
    z = analysis.two_proportion_z(80, 100, 50, 100)
    assert z["diff"] == pytest.approx(0.3) and z["z"] > 0 and z["p_value"] < 0.001


def test_commitment_and_patching_summaries():
    recs = [
        {"commitment_layer": 30, "correct": True},
        {"commitment_layer": 28, "correct": False},
        {"commitment_layer": -1, "correct": True},
    ]
    s = analysis.commitment_summary(recs)
    assert s["n"] == 3 and s["mean_commitment_layer"] == 29
    assert s["frac_committed"] == pytest.approx(2 / 3)
    assert "mean_commitment_layer_ci" in analysis.commitment_summary_ci(recs)

    prec = [
        {"flipped": True, "logit_diff_drop": 0.5, "correct_clean": True,
         "correct_corrupted": False, "patch_recovers_clean": True},
        {"flipped": False, "logit_diff_drop": 0.1, "correct_clean": False,
         "correct_corrupted": False, "patch_recovers_clean": False},
    ]
    p = analysis.patching_summary(prec, with_ci=True)
    assert p["flip_rate"] == 0.5 and p["patch_recovery_rate"] == 0.5
    fa = analysis.faithfulness_accuracy(prec)
    assert fa["acc_when_flipped"] == 1.0 and fa["acc_when_not_flipped"] == 0.0


def test_threshold_sweep_and_extremes():
    # 2 examples, 5 layers, 2 classes; label-class prob rises over layers
    rise = np.array([0.1, 0.4, 0.6, 0.85, 0.95])
    arr = np.stack([np.stack([1 - rise, rise], 1), np.stack([rise, 1 - rise], 1)])
    df = analysis.threshold_sweep(arr, [1, 0], taus=(0.5, 0.9))
    assert df.loc[df.tau == 0.5, "mean_layer"].item() == 2
    assert df.loc[df.tau == 0.9, "mean_depth_frac"].item() == 1.0
    assert (df.frac_committed == 1.0).all()

    recs = [{"idx": i, "commitment_layer": L, "correct": True, "cot_text": "x"}
            for i, L in enumerate([5, -1, 3, 9])]
    ex = analysis.qualitative_extremes(recs, k=1)
    assert ex.loc[ex.group == "early", "commitment_layer"].item() == 3
    assert ex.loc[ex.group == "late", "commitment_layer"].item() == 9


def test_plots_render():
    import matplotlib.pyplot as plt

    analysis.plot_layer_heatmap(torch.rand(4, 2), ["No", "Yes"], title="t")
    recs = [{"commitment_layer": L, "correct": L % 2 == 0} for L in [1, 2, 3, 3]]
    analysis.plot_commitment_histogram(recs, n_layers=4)
    analysis.plot_commitment_compare(recs, recs, n_layers=4)
    plt.close("all")

import random
from datasets import load_dataset

BOOLQ_LABELS = {0: "No", 1: "Yes"}
MNLI_LABELS = {0: "entailment", 1: "neutral", 2: "contradiction"}


def load_boolq(split="validation", n=None, seed=42):
    ds = load_dataset("google/boolq", split=split)
    if n is not None and n < len(ds):
        idx = random.Random(seed).sample(range(len(ds)), n)
        ds = ds.select(sorted(idx))
    return ds


def load_mnli(split="validation_matched", n=None, seed=42):
    ds = load_dataset("nyu-mll/multi_nli", split=split)
    ds = ds.filter(lambda ex: ex["label"] in (0, 1, 2))
    if n is not None and n < len(ds):
        idx = random.Random(seed).sample(range(len(ds)), n)
        ds = ds.select(sorted(idx))
    return ds


def boolq_example(row):
    return {
        "passage": row["passage"],
        "question": row["question"],
        "label": int(row["answer"]),
        "label_text": BOOLQ_LABELS[int(row["answer"])],
    }


def mnli_example(row):
    return {
        "premise": row["premise"],
        "hypothesis": row["hypothesis"],
        "label": int(row["label"]),
        "label_text": MNLI_LABELS[int(row["label"])],
    }

"""High-level experiment runners shared by the final-report notebook.

Two families:
  run_commitment  -> logit-lens commitment depth, with-CoT or no-CoT.
  run_corruption  -> CoT corruption (random / shuffle / invert) + pre-CoT patch.

Both keep target-id ordering aligned with the dataset's integer labels, so an
argmax index is directly a label index.
"""
import numpy as np
import torch
from tqdm.auto import tqdm

from . import data, prompts, generation, logit_lens, patching, model as model_mod


# kind -> (example_fn, cot_prompt_fn, direct_prompt_fn, choices_in_label_order)
def _spec(kind, model):
    if kind == "boolq":
        t = prompts.boolq_target_tokens(model)
        return (
            data.boolq_example,
            lambda ex: prompts.format_boolq(ex["passage"], ex["question"]),
            lambda ex: prompts.format_boolq_direct(ex["passage"], ex["question"]),
            ["No", "Yes"],
            [t["No"], t["Yes"]],
        )
    if kind == "mnli":
        t = prompts.mnli_target_tokens(model)
        order = ["entailment", "neutral", "contradiction"]
        return (
            data.mnli_example,
            lambda ex: prompts.format_mnli(ex["premise"], ex["hypothesis"]),
            lambda ex: prompts.format_mnli_direct(ex["premise"], ex["hypothesis"]),
            order,
            [t[c] for c in order],
        )
    raise ValueError(f"unknown kind {kind!r}")


def run_commitment(model, dataset, kind, tau=0.8, with_cot=False,
                   max_len=1024, max_new_tokens=180, free_every=50):
    example_fn, cot_fn, direct_fn, choices, target_ids = _spec(kind, model)
    records, all_probs = [], []
    for i in tqdm(range(len(dataset)), desc=f"commitment {kind} cot={with_cot}"):
        ex = example_fn(dataset[i])
        if with_cot:
            prompt = cot_fn(ex)
            if model.to_tokens(prompt).shape[1] > max_len - max_new_tokens:
                continue
            completion, _ = generation.generate_cot(model, prompt, max_new_tokens=max_new_tokens)
            cot_text, parsed = generation.split_completion(completion)
            tokens, score_pos, _ = generation.build_answer_scoring_tokens(model, prompt, cot_text)
            if tokens.shape[1] > max_len:
                continue
            probs = logit_lens.logit_lens(model, tokens, target_ids, positions=[score_pos])[:, 0, :]
        else:
            prompt = direct_fn(ex)
            tokens = model.to_tokens(prompt)
            if tokens.shape[1] > max_len:
                continue
            probs = logit_lens.logit_lens(model, tokens, target_ids, positions=[tokens.shape[1] - 1])[:, 0, :]
            cot_text, parsed = "", None
        label = ex["label"]
        cl = logit_lens.commitment_layer(probs, label, threshold=tau)
        pred = int(probs[-1].argmax().item())
        rec = {
            "idx": i, "label": label, "label_text": ex["label_text"],
            "pred": pred, "pred_text": choices[pred], "correct": pred == label,
            "commitment_layer": cl, "final_p_correct": probs[-1, label].item(),
        }
        if with_cot:
            rec["cot_text"] = cot_text
            rec["parsed_answer"] = parsed
        records.append(rec)
        all_probs.append(probs.numpy())
        if (i + 1) % free_every == 0:
            model_mod.free_memory()
    return records, np.stack(all_probs) if all_probs else np.empty((0, model.cfg.n_layers, len(choices)))


def run_corruption(model, dataset, kind, strategy="random", do_patch=True,
                   max_len=1024, max_new_tokens=160, free_every=25, seed_base=0):
    example_fn, cot_fn, _, choices, target_ids = _spec(kind, model)
    records = []
    for i in tqdm(range(len(dataset)), desc=f"corruption {kind} {strategy}"):
        ex = example_fn(dataset[i])
        prompt = cot_fn(ex)
        if model.to_tokens(prompt).shape[1] > max_len - max_new_tokens:
            continue
        completion, _ = generation.generate_cot(model, prompt, max_new_tokens=max_new_tokens)
        cot_text, parsed = generation.split_completion(completion)
        if not cot_text:
            continue
        res = patching.causal_corruption_test(
            model, prompt, cot_text, target_ids,
            strategy=strategy, seed=seed_base + i, do_patch=do_patch,
        )
        label = ex["label"]
        rec = {
            "idx": i, "label": label, "label_text": ex["label_text"],
            "parsed_answer": parsed,
            "pred_clean": choices[res["clean_argmax"]],
            "pred_corrupted": choices[res["corrupted_argmax"]],
            "flipped": bool(res["flipped"]),
            "logit_diff_drop": float(res["logit_diff_drop"]),
            "correct_clean": res["clean_argmax"] == label,
            "correct_corrupted": res["corrupted_argmax"] == label,
            "cot_len": int(res["cot_len"]),
            "p_clean": res["clean"].tolist(),
            "p_corrupted": res["corrupted"].tolist(),
        }
        if do_patch:
            rec["pred_patched"] = choices[res["patched_argmax"]]
            rec["patch_recovers_clean"] = bool(res["patch_recovers_clean"])
            rec["correct_patched"] = res["patched_argmax"] == label
            rec["p_patched"] = res["patched"].tolist()
        records.append(rec)
        if (i + 1) % free_every == 0:
            model_mod.free_memory()
    return records

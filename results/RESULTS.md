# Results — Is Chain-of-Thought Just Rationalization?

**Model:** `meta-llama/Meta-Llama-3-8B-Instruct` (32 layers, fp16) · **Seed:** 42 · **Lens threshold:** τ = 0.8
**Datasets:** BoolQ (N = 1500), MNLI (N = 1000)
**Methods:** logit lens, tuned lens (rank-256 affine translators), CoT corruption + pre-CoT residual patching.

All numbers below are point estimates with 95% bootstrap confidence intervals (10k resamples). Figures are in `results/figures/`.

---

## TL;DR

1. **Under the logit lens the answer surfaces only in the last ~10% of layers**, and **CoT changes neither when the model commits nor its accuracy** (BoolQ).
2. **The tuned lens reveals the answer is linearly *decodable* ~22 layers earlier** (layer ~7 vs ~29) — the logit lens badly under-reads early information. (Caveat: a trained probe shows *decodability*, not necessarily *commitment*.)
3. **CoT content is causal but only partially**: semantically *inverting* the CoT flips the answer **3.8× more often than random-token corruption** (0.43 vs 0.11). This is evidence against pure post-hoc rationalization, but inversion still fails to flip the answer the majority of the time.

The integrated picture: **the model largely "knows" its answer early, and the verbalized CoT exerts a real but bounded causal pull.** Faithfulness is a matter of degree.

---

## Iteration history (and how the numbers changed)

We explicitly log this because the headline corruption result moved substantially between iterations, and the reason is methodological, not cosmetic.

### Iteration 0 — Milestone state (broken)
- MNLI answer **regex parsing failed on ~97% of generations** (≈5/200 parsed). Conclusions about MNLI commitment were unreliable.
- The class was scored from logits at the **end of the generated sequence** rather than at the answer position, and the corruption span **included the answer line** (leaking the answer into the "corrupted" region).

### Iteration 1 — Code fixes
- **Score by logits at the post-`Answer:` position** (`build_answer_scoring_tokens`). This removes all dependence on a parseable answer string — every example yields a class distribution.
- **Corruption span now excludes the answer line** (`split_completion` returns CoT-only text), so corruption acts purely on the reasoning.
- Fixed a **fp16/fp32 dtype mismatch** in the tuned lens (model runs in `Half`, Adam needs `float32`); lens math now runs in fp32 and casts back to model dtype for the frozen norm + unembed.

### Iteration 2 — Full-scale run (BoolQ solid, MNLI underpowered)
- BoolQ commitment (N = 1500), threshold sweep, qualitative extremes, and tuned lens all ran cleanly.
- **MNLI corruption collapsed to n = 25 / 1000** (97.5% skipped): on MNLI, Llama-3 usually answers immediately with an **empty CoT**, which `run_corruption` drops. The surviving 25 are a biased subsample (only cases where the model chose to reason), and MNLI **clean accuracy was 0.48** (3-way chance = 0.33), so the base task was near-random.

| MNLI corruption (n=25) | flip rate | patch-recovery | clean acc |
|---|---|---|---|
| random-token | 0.16 [0.04, 0.32] | 0.84 [0.68, 0.96] | 0.48 |
| semantic-invert | 0.52 [0.32, 0.72] | 0.48 [0.28, 0.68] | 0.48 |

Directionally right (invert ≫ random) but wide CIs and a weak base task.

### Iteration 3 — Move corruption to BoolQ, reuse generated CoTs (current)
- BoolQ produces a non-empty CoT **99.9% of the time** (2/1500 empty), so we re-ran corruption on BoolQ and **reused the CoTs already generated** in the commitment run (`boolq_commitment_cot.csv`) — no clean-CoT regeneration. `random` needs no generation at all; only `invert` calls the model.
- Result: **n = 800 (random) / 300 (invert)**, tight CIs, and a base task the model actually performs well (clean acc ≈ 0.84).

| BoolQ corruption | n | flip rate | patch-recovery | clean acc |
|---|---|---|---|---|
| random-token | 800 | **0.11 [0.09, 0.13]** | 0.89 [0.87, 0.91] | 0.84 |
| semantic-invert | 300 | **0.43 [0.37, 0.48]** | 0.61 [0.55, 0.66] | 0.85 |

See **`fig7_iterations.png`** for the MNLI → BoolQ comparison. The qualitative conclusion (content is causal; invert ≫ random) is stable across both iterations; BoolQ makes it statistically tight and removes the empty-CoT selection bias.

---

## Detailed results

### 1. Commitment depth — CoT barely moves it (BoolQ, N = 1500)
- **no-CoT:** mean commitment layer **29.42** [29.36, 29.48], accuracy **0.847** [0.829, 0.865]
- **with-CoT:** mean commitment layer **29.97** [29.95, 29.99], accuracy **0.838** [0.819, 0.857]

CoT shifts commitment by only **+0.55 layers** (i.e. slightly *later*) and changes accuracy by **−0.009** (CIs overlap). Under the logit lens, the correct class crosses τ only in the final ~10% of the network, with or without CoT. See **`fig1_pcorrect_by_layer.png`** and **`fig2_commitment_hist.png`**.

### 2. Threshold robustness
Mean commitment depth fraction ranges **0.90 → 0.99** across τ ∈ {0.5, 0.7, 0.8, 0.9}; deep commitment is not a threshold artifact, and with-CoT is consistently ≥ no-CoT at every τ. See **`fig3_threshold_sweep.png`**.

### 3. Tuned lens vs logit lens
- logit lens mean commitment layer: **29.36** [29.20, 29.52]
- tuned lens mean commitment layer: **7.39** [6.80, 8.00]

A **~22-layer gap**: once per-layer representational drift is corrected, the answer is linearly decodable from the residual stream by layer ~7. See **`fig4_tuned_vs_logit.png`**.

> **Caveat (state this in the report):** the tuned lens is a *trained* probe fit to predict the model's final output, so it can extract answer-correlated features the model is not yet acting on. The honest claim is "answer-relevant information is **decodable** by layer ~7," and the contribution is the **32 → 7 gap**, which quantifies how much the logit lens underestimates early information. Do not report "the model commits at layer 7" without this caveat.

### 4. CoT corruption (BoolQ)
- **Random-token corruption** (n=800): flip **0.11** [0.09, 0.13], mean logit drop 0.13. Scrambling the CoT's surface tokens rarely changes the answer.
- **Semantic inversion** (n=300): flip **0.43** [0.37, 0.48], mean logit drop 0.41, corrupted accuracy drops to 0.51. Flipping the CoT's *meaning* changes the answer 3.8× more often — inversion injects genuine answer-relevant content, not noise.

> **Correction — patch-recovery is not independent evidence.** Earlier versions also reported a "pre-CoT patch-recovery" rate (0.89 random / 0.61 invert) as separate evidence. It isn't: the patch overwrites the *prompt-prefix* residuals, but corruption only touches *CoT* tokens, so those prefix residuals are identical in the clean and corrupted runs and the patch is a **no-op**. Patch-recovery therefore equals **1 − flip** by construction (verified: exactly 1.000 of random rows; the invert 0.947 differs only from fp16 nondeterminism between two forward passes). We drop patch-recovery as a metric — flip rate and logit drop carry all the signal. Localizing *where* the CoT's effect enters would require patching the **CoT-span** residuals under a length-preserving corruption (random/shuffle), which we leave to future work.

See **`fig5_corruption_bars.png`** and **`fig6_logitdrop_dist.png`**.

**Interpretation.** A purely post-hoc CoT should be no more affected by semantic inversion than by random noise. The decisive 0.11 → 0.43 flip gap rejects that. But inversion flips < 50% of cases, so the CoT is *partially* faithful: it has a measurable causal role layered on top of an answer the model has largely committed to internally (consistent with §3). The **paraphrase** control in §13 sharpens this: a meaning-preserving rewrite should flip at ≈ the random rate (≈0.11), not the invert rate.

---

## Limitations
- **Single model, single seed.** All results are Llama-3-8B-Instruct, seed 42. No cross-family check was run (the optional section 11 was not executed).
- **Tuned lens validity.** As above, decodability ≠ commitment; we have no control (e.g., shuffled-label or random-target lens) bounding how much the probe over-reads.
- **Inversion confounds.** Inverted CoTs may differ in length/fluency from originals; we did not verify that inverted texts are coherent opposite arguments, nor add a *paraphrase* control (semantics preserved → should **not** flip).
- **Pre-CoT patching is uninformative (see §4 correction).** The patch targets prompt-prefix residuals that corruption never alters, so patch-recovery ≡ 1−flip (a no-op). We report flip rate and logit drop only; localizing the CoT's causal effect needs a CoT-span patch under a length-preserving corruption.
- **invert n = 300** (vs 800 for random) due to generation cost; CIs are correspondingly wider.

## Suggested next improvements

**Implemented in notebook §13 (run pending — numbers to be filled in here):**
1. **Paraphrase control** for corruption: rewrite the CoT preserving meaning. A faithful pathway predicts paraphrase flip ≈ random (≈0.11) ≪ invert (0.43) — a clean 3-condition test of the rationalization hypothesis. (cell 13c)
2. **Tuned-lens control:** lens fit on shuffled targets + per-layer top-1 accuracy, to bound the "decodable ≠ committed" gap. (cell 13e)
3. **Statistical test on the flip-rate gap** (two-proportion z + bootstrap difference) rather than reading non-overlapping CIs. (cell 13d)
4. **Scale invert to n=800** to match random, plus a `shuffle` baseline (word order destroyed, tokens preserved). (cell 13b)

**Still open:**
5. **CoT-span layer-resolved patching** — patch the *clean CoT-span* residuals into a length-preserving (random/shuffle) corrupted run, per layer window, to localize where the CoT's effect enters. (The earlier all-layer *pre-CoT* patch was a no-op; see §4 correction.)
6. **Cross-family generalization** (section 11): same commitment + corruption pipeline on Mistral-7B / Qwen2-7B / Gemma-2-9B to show the pattern is not Llama-specific.

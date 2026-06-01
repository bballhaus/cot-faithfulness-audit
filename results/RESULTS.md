# Results — Is Chain-of-Thought Just Rationalization?

**Model:** `meta-llama/Meta-Llama-3-8B-Instruct` (32 layers, fp16) · **Seed:** 42 · **Lens threshold:** τ = 0.8
**Datasets:** BoolQ (N = 1500), MNLI (N = 1000)
**Methods:** logit lens, tuned lens (rank-256 affine translators), CoT corruption + pre-CoT residual patching.

All numbers below are point estimates with 95% bootstrap confidence intervals (10k resamples). Figures are in `results/figures/`.

---

## TL;DR

1. **Under the logit lens the answer surfaces only in the last ~10% of layers**, and **CoT changes neither when the model commits nor its accuracy** (BoolQ).
2. **The answer is linearly *decodable* well before the logit lens reads it, but how early depends on the probe — and a shuffled-target control shows the naive "layer 7" figure is largely an artifact.** A rank-256 tuned lens hits τ at layer ~7, but a lens trained on **shuffled targets** commits at layer ~6.8 (≈ identical), so layer-7 "commitment" is mostly the probe fitting structure, not the model. The honest read is the lens's **per-layer top-1 accuracy**, which stays near chance (~0.5) until **layer 12 (0.77)** and plateaus at ~0.84 by **layer 17** — answer-relevant information emerges mid-network, ~12–17 layers before the logit lens (~29).
3. **CoT content is causal but only partially, and a clean 3-condition test rules out pure rationalization.** Across **paraphrase (meaning-preserving) → random/shuffle (noise) → invert (meaning-flipping)**, flip rates are **0.04 / 0.11 / 0.09 / 0.42**. Semantic inversion flips the answer **~3.8× more than random** (z = 12.4, p ≈ 4e-35) and **~10× more than paraphrase** (z = 12.7, p ≈ 4e-37), while paraphrase flips *less* than random — exactly the signature of a partially-faithful causal pathway, not post-hoc text. But inversion still fails to flip < 50% of the time.

The integrated picture: **the model largely "knows" its answer by mid-network, and the verbalized CoT exerts a real but bounded causal pull.** Faithfulness is a matter of degree.

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

### Iteration 4 — Controls + scaling (§13, current)
- Scaled **invert to n=400** (0.422 [0.375, 0.472]), added **paraphrase** (0.043) and **shuffle** (0.085) controls, pairwise z-tests, the **faithfulness–accuracy** correlation, and the **shuffled-target tuned-lens** control. Numbers folded into §3–§4. The one headline change: the tuned-lens "layer 7" claim was **demoted to a probe artifact** by the shuffled-target control; robust decodability is ~layer 12 (per-layer top-1 ≥ 0.77). The corruption picture only got sharper (3-condition ordering, all gaps significant except shuffle-vs-random).

---

## Detailed results

### 1. Commitment depth — CoT barely moves it (BoolQ, N = 1500)
- **no-CoT:** mean commitment layer **29.42** [29.36, 29.48], accuracy **0.847** [0.829, 0.865]
- **with-CoT:** mean commitment layer **29.97** [29.95, 29.99], accuracy **0.838** [0.819, 0.857]

CoT shifts commitment by only **+0.55 layers** (i.e. slightly *later*) and changes accuracy by **−0.009** (CIs overlap). Under the logit lens, the correct class crosses τ only in the final ~10% of the network, with or without CoT. See **`fig1_pcorrect_by_layer.png`** and **`fig2_commitment_hist.png`**.

### 2. Threshold robustness
Mean commitment depth fraction ranges **0.90 → 0.99** across τ ∈ {0.5, 0.7, 0.8, 0.9}; deep commitment is not a threshold artifact, and with-CoT is consistently ≥ no-CoT at every τ. See **`fig3_threshold_sweep.png`**.

### 3. Tuned lens vs logit lens — and the shuffled-target control
- logit lens mean commitment layer: **29.36** [29.20, 29.52]
- tuned lens mean commitment layer: **7.39** [6.80, 8.00]
- **shuffled-target tuned lens (control):** mean commitment layer **6.81** [5.89, 7.77], frac committed 0.78

The naive reading — "a ~22-layer gap, answer decodable by layer ~7" — **does not survive the control.** A lens trained on **randomly shuffled targets** (so it *cannot* be reading the true answer) commits at layer **6.81**, statistically indistinguishable from the real lens's 7.39. This means the layer-7 τ-crossing is **largely a probe artifact**: a rank-256 affine map at every layer over-reads structure that correlates with the final output regardless of whether the early residual stream actually carries the answer.

The honest, control-passing signal is the lens's **per-layer top-1 accuracy** (`tuned_lens_control.json → real_lens_acc_by_layer`): it sits near binary chance (~0.50–0.52) through layer 4, drifts up noisily (0.55–0.64) through layer 11, then **jumps to 0.77 at layer 12, 0.82 at layer 13, and plateaus at ~0.84 by layer 17**. So answer-relevant information becomes robustly decodable **mid-network (~layer 12–17)**, ~12–17 layers earlier than the logit lens (~29) — a real and substantial gap, but **not** as early as layer 7. See **`fig4_tuned_vs_logit.png`**.

> **Caveat (resolved by the control):** a trained probe extracts answer-correlated features the model may not yet be acting on, so decodability ≠ commitment. The shuffled-target lens bounds this: it shows the τ-crossing depth is uninformative, and the defensible claim is "answer-relevant information is **decodable** from layer ~12 (top-1 ≥ 0.77)." Report the **per-layer-accuracy curve**, not the τ-crossing layer.

### 4. CoT corruption — the 3-condition test (BoolQ)
We now run **four** corruption conditions on the same generated CoTs, spanning meaning-preserving → noise → meaning-flipping:

| condition | what it does | n | flip rate [95% CI] |
|---|---|---|---|
| **paraphrase** | reword, preserve meaning | 400 | **0.043** [0.025, 0.062] |
| **shuffle** | permute CoT tokens (order destroyed, tokens kept) | 400 | **0.085** [0.058, 0.113] |
| **random** | replace CoT tokens with random vocab | 800 | **0.111** [0.090, 0.134] |
| **semantic-invert** | reword to the OPPOSITE conclusion | 400 | **0.422** [0.375, 0.472] |

**Pairwise two-proportion z-tests** (`corruption_stats.json`, corroborated by bootstrap difference CIs):
- **invert vs random:** Δ = +0.311, z = 12.4, **p ≈ 3.7e-35** — inversion flips ~3.8× more than random noise.
- **invert vs paraphrase:** Δ = +0.380, z = 12.7, **p ≈ 4.5e-37** — the decisive test: flipping meaning ≫ preserving meaning.
- **paraphrase vs random:** Δ = −0.069, z = −3.96, **p ≈ 7.6e-5** — paraphrase flips *significantly less* than random.
- **shuffle vs random:** Δ = −0.026, z = −1.41, **p = 0.16 (NS)** — destroying word order alone barely matters.

This is the clean signature of a **partially-faithful causal pathway**: preserving meaning (paraphrase) leaves the answer most stable; injecting noise (shuffle/random) nudges it slightly; only injecting *opposite meaning* (invert) moves it substantially. A purely post-hoc CoT would show no ordering among these. But invert still flips < 50% of cases, so the pull is bounded.

**Faithfulness–accuracy correlation** (`faithfulness_accuracy.json`; proposal eval metric 3 — are causally-influential CoTs also more accurate?): the sign **depends on the corruption type**, which is itself diagnostic.
- **invert** (n_flip=169): accuracy is *higher* when the answer flips (0.88 vs 0.81), point-biserial **r = +0.10** (z = 1.94, p = 0.052). When a correct CoT is meaningfully inverted, the answer is more likely to follow it — flips track genuine reasoning.
- **random / shuffle / paraphrase** (r = −0.21 / −0.14 / −0.18): here flips are *negatively* correlated with accuracy — these conditions only flip the answer on already-shaky examples, i.e. flips are noise, not reasoning. (Small n_flip: 89 / 34 / 17.)

The contrast — **invert flips correlate with correctness, noise flips anti-correlate** — is exactly what a faithful-but-bounded CoT predicts.

> **Correction — patch-recovery is not independent evidence.** Earlier versions also reported a "pre-CoT patch-recovery" rate (0.89 random / 0.61 invert) as separate evidence. It isn't: the patch overwrites the *prompt-prefix* residuals, but corruption only touches *CoT* tokens, so those prefix residuals are identical in the clean and corrupted runs and the patch is a **no-op**. Patch-recovery therefore equals **1 − flip** by construction (verified: exactly 1.000 of random rows; the invert 0.947 differs only from fp16 nondeterminism between two forward passes). We drop patch-recovery as a metric — flip rate and logit drop carry all the signal. Localizing *where* the CoT's effect enters would require patching the **CoT-span** residuals under a length-preserving corruption (random/shuffle), which we leave to future work.

See **`fig5_corruption_bars.png`** and **`fig6_logitdrop_dist.png`**.

**Interpretation.** A purely post-hoc CoT should be no more affected by semantic inversion than by random noise. The decisive **0.11 → 0.42** flip gap (p ≈ 4e-35) rejects that. The **paraphrase control** sharpens it further: a meaning-preserving rewrite flips at **0.04**, *below* the random rate — so the effect tracks **meaning**, not edit magnitude. But inversion flips < 50% of cases, so the CoT is *partially* faithful: it has a measurable causal role layered on top of an answer the model has largely committed to internally (consistent with §3).

---

## Limitations
- **Single model, single seed.** All results are Llama-3-8B-Instruct, seed 42. No cross-family check was run (the optional section 11 was not executed).
- **Tuned lens validity — now bounded by a control.** The shuffled-target lens (§3) shows the τ-crossing depth (~7) is largely a probe artifact, so we no longer claim layer-7 commitment; we report the per-layer top-1 accuracy curve (robust decodability ~layer 12). We still lack a *causal* control showing the model *acts on* this mid-network information rather than merely carrying it.
- **Inversion confounds — partially addressed.** We added **paraphrase** (meaning preserved → flips 0.04, below random) and **shuffle** (order destroyed, tokens kept → 0.09 ≈ random) controls, which together show the invert effect tracks *meaning* not edit size. We still did not independently verify that each inverted text is a coherent opposite argument (vs. merely lower-quality), nor match inverted/original length token-for-token.
- **Pre-CoT patching is uninformative (see §4 correction).** The patch targets prompt-prefix residuals that corruption never alters, so patch-recovery ≡ 1−flip (a no-op). We report flip rate and logit drop only; localizing the CoT's causal effect needs a CoT-span patch under a length-preserving corruption.
- **Unequal n.** invert/paraphrase/shuffle n = 400, random n = 800, due to generation cost; CIs on the rewrite conditions are correspondingly wider (though all pairwise gaps except shuffle-vs-random are significant).

## Suggested next improvements

**Implemented in §13 (done — numbers folded into §3–§4 above):**
1. ✅ **Paraphrase control** for corruption — flip **0.043** ≪ random (0.111) ≪ invert (0.422); the clean 3-condition test passes (invert vs paraphrase p ≈ 4.5e-37).
2. ✅ **Tuned-lens shuffled-target control** + per-layer top-1 accuracy — revealed the layer-7 τ-crossing is a probe artifact (shuffled lens commits 6.81 ≈ real 7.39); robust decodability is ~layer 12. **This changed the headline §3 claim.**
3. ✅ **Statistical test on the flip-rate gap** — two-proportion z + bootstrap difference CIs for all pairs (`corruption_stats.json`).
4. ✅ **Scale + baselines** — invert/paraphrase/shuffle at n=400, plus the `shuffle` baseline (0.085, not sig. vs random, p=0.16).

**Still open:**
5. **CoT-span layer-resolved patching** — patch the *clean CoT-span* residuals into a length-preserving (random/shuffle) corrupted run, per layer window, to localize where the CoT's effect enters. (The earlier all-layer *pre-CoT* patch was a no-op; see §4 correction.)
6. **Cross-family generalization** (section 11): same commitment + corruption pipeline on Mistral-7B / Qwen2-7B / Gemma-2-9B to show the pattern is not Llama-specific.
7. **Causal control for the tuned lens** — a perturbation showing the model *acts on* the mid-network (layer ~12) answer information, not just that a probe can read it.

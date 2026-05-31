# Is Chain-of-Thought Just Rationalization?
### CS 281 Project — Mechanistic Faithfulness Audit of LLM Reasoning
**Brooke Ballhaus & Riya Narain** | Stanford University

---

We audit whether chain-of-thought (CoT) prompting in LLMs is genuinely causal or post-hoc rationalization, using two mechanistic interpretability methods on `meta-llama/Meta-Llama-3-8B-Instruct` via TransformerLens:

1. **Logit lens** — tracks at which transformer layer the model's answer class probability commits (≥ τ = 0.8), using layer-wise residual stream projections onto the vocabulary.
2. **Activation patching** — corrupts the CoT token span and patches pre-CoT residual stream activations to test whether the CoT causally mediates the final answer.

**Key finding so far:** Under direct (no-CoT) prompting on BoolQ (N=500), mean commitment layer is 29.4/32 — the model crystallizes its answer in the final ~3 layers, not early. This recenters the project's question: does generating a CoT *move* commitment earlier, consistent with the chain filling in post-hoc justification? The no-CoT vs with-CoT comparison in `final.ipynb` is the central experiment.

---

## Repository Structure

```
cot_faithfulness/
  data.py         # BoolQ + MNLI loading from HuggingFace
  model.py        # TransformerLens loader (Llama-3-8B-Instruct default; ALT_MODELS for generalization)
  prompts.py      # Llama-3 chat formatting, tokenizer-template formatting, answer token utilities
  generation.py   # CoT generation, answer parsing, CoT/answer span split + scoring tokens
  logit_lens.py   # Layer-wise residual stream projection + commitment detection
  tuned_lens.py   # Per-layer affine translators (tuned lens) + trainer + probs
  patching.py     # Random/shuffle/semantic-inversion CoT corruption + pre-CoT residual patching
  analysis.py     # Summaries, bootstrap CIs, threshold sweep, qualitative extraction, plots
  experiments.py  # High-level runners: run_commitment (no-CoT/with-CoT), run_corruption

milestone.ipynb   # Milestone experiments (historical; superseded by final.ipynb)
final.ipynb       # Final-report pipeline: runs every experiment end-to-end
requirements.txt
results/          # CSVs, .npy arrays, PNGs, JSON summaries land here
```

---

## Setup

### Colab (recommended — requires A100 or L4)

Llama-3-8B-Instruct in fp16 needs ~16 GB VRAM; the tuned lens adds a few GB during fitting. Free Colab T4 (~15 GB) is borderline; A100/L4 is reliable.

1. Request access to `meta-llama/Meta-Llama-3-8B-Instruct` on HuggingFace if you haven't already.
2. Open `final.ipynb` in Colab and set runtime to **GPU (A100 or L4)**.
3. Set your HuggingFace token:
```python
import os
os.environ['HF_TOKEN'] = 'hf_...'
```
4. Run all cells. Scale knobs (N, τ sweep, tuned-lens sizes) live in the **Config** cell. Results land in `results/`.

### Local

```bash
git clone https://github.com/bballhaus/cot-faithfulness-audit.git
cd cot-faithfulness-audit
pip install -r requirements.txt
cp .env.example .env   # add your HF_TOKEN
jupyter notebook final.ipynb
```

CUDA strongly preferred. Apple Silicon (MPS) works but is slow for 8B models.

---

## Results (final.ipynb)

| File | Contents |
|------|----------|
| `results/boolq_commitment_nocot.csv` / `_cot.csv` | Per-example commitment layer, prediction, correctness — no-CoT vs with-CoT |
| `results/boolq_probs_nocot.npy` / `_cot.npy` | `(n_examples, n_layers, n_classes)` logit-lens probabilities |
| `results/commitment_compare.png` | Commitment-depth distributions: no-CoT vs with-CoT (central result) |
| `results/mean_curve_compare.png` | Mean layer-wise P(correct class) curve, both prompting modes |
| `results/threshold_sweep.csv` | Commitment depth vs τ ∈ {0.5, 0.7, 0.8, 0.9}, both modes |
| `results/qualitative_extremes.csv` | 25 earliest- + 25 latest-commitment examples with CoT text |
| `results/tuned_vs_logit.json` | Tuned-lens vs logit-lens commitment depth (robustness check) |
| `results/mnli_corruption_random.csv` / `_invert.csv` | Per-example clean/corrupted/patched answer distributions |
| `results/mnli_corruption.png` | Logit-drop histograms with flip + patch-recovery rates |
| `results/final_summary.json` | All aggregate numbers (with bootstrap CIs) for the writeup |

---

## References

- Turpin et al. (2023). *Language Models Don't Always Say What They Think.* NeurIPS.
- Lanham et al. (2023). *Measuring Faithfulness in Chain-of-Thought Reasoning.* arXiv:2307.13702.
- nostalgebraist (2020). *Interpreting GPT: The Logit Lens.* LessWrong.
- Belrose et al. (2023). *Eliciting Latent Predictions from Transformers with the Tuned Lens.* arXiv:2303.08112.
- Nanda & Bloom (2022). *TransformerLens.* [github.com/TransformerLensOrg/TransformerLens](https://github.com/TransformerLensOrg/TransformerLens)

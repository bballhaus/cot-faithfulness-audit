# Is Chain-of-Thought Just Rationalization?
### CS 281 Project — Mechanistic Faithfulness Audit of LLM Reasoning
**Brooke Ballhaus & Riya Narain** | Stanford University

---

We audit whether chain-of-thought (CoT) prompting in LLMs is genuinely causal or post-hoc rationalization, using two mechanistic interpretability methods on `meta-llama/Meta-Llama-3-8B-Instruct` via TransformerLens:

1. **Logit lens** — tracks at which transformer layer the model's answer class probability commits (≥ τ = 0.8), using layer-wise residual stream projections onto the vocabulary.
2. **Activation patching** — corrupts the CoT token span and patches pre-CoT residual stream activations to test whether the CoT causally mediates the final answer.

**Key finding so far:** Under direct (no-CoT) prompting on BoolQ (N=500), mean commitment layer is 29.4/32 — the model crystallizes its answer in the final ~3 layers, not early. This recenters the project's question: does generating a CoT *move* commitment earlier, consistent with the chain filling in post-hoc justification?

---

## Repository Structure

```
cot_faithfulness/
  data.py         # BoolQ + MNLI loading from HuggingFace
  model.py        # TransformerLens Llama-3-8B-Instruct loader
  prompts.py      # Llama-3 chat formatting + answer token utilities
  generation.py   # CoT generation, answer parsing
  logit_lens.py   # Layer-wise residual stream projection + commitment detection
  patching.py     # CoT corruption (random tokens) + pre-CoT residual patching
  analysis.py     # Summary stats + matplotlib visualisations

milestone.ipynb   # Reproduces all milestone experiments end-to-end
requirements.txt
results/          # CSVs, .npy arrays, PNGs, JSON summaries land here
```

---

## Setup

### Colab (recommended — requires A100 or L4)

Llama-3-8B-Instruct in fp16 needs ~16 GB VRAM. Free Colab T4 (~15 GB) is borderline; A100/L4 is reliable.

1. Request access to `meta-llama/Meta-Llama-3-8B-Instruct` on HuggingFace if you haven't already.
2. Open `milestone.ipynb` in Colab and set runtime to **GPU (A100 or L4)**.
3. Set your HuggingFace token:
```python
import os
os.environ['HF_TOKEN'] = 'hf_...'
```
4. Run all cells. Results land in `results/`.

### Local

```bash
git clone https://github.com/bballhaus/cot-faithfulness-audit.git
cd cot-faithfulness-audit
pip install -r requirements.txt
cp .env.example .env   # add your HF_TOKEN
jupyter notebook milestone.ipynb
```

CUDA strongly preferred. Apple Silicon (MPS) works but is slow for 8B models.

---

## Results

| File | Contents |
|------|----------|
| `results/boolq_layer_probs.npy` | `(n_examples, n_layers, 2)` — per-layer P(No), P(Yes) at the answer position |
| `results/boolq_commitment.csv` | Per-example commitment layer, prediction, correctness |
| `results/boolq_commitment_hist.png` | Commitment layer histogram split by correct/incorrect |
| `results/boolq_mean_curve.png` | Mean layer-wise P(Yes)/P(No) curve across 500 examples |
| `results/mnli_patching.csv` | Per-example clean vs. corrupted answer distributions |
| `results/mnli_corruption.png` | Flip-rate plot under random-token CoT corruption |
| `results/milestone_summary.json` | Aggregate numbers referenced in the milestone report |

---


## References

- Turpin et al. (2023). *Language Models Don't Always Say What They Think.* NeurIPS.
- Lanham et al. (2023). *Measuring Faithfulness in Chain-of-Thought Reasoning.* arXiv:2307.13702.
- nostalgebraist (2020). *Interpreting GPT: The Logit Lens.* LessWrong.
- Belrose et al. (2023). *Eliciting Latent Predictions from Transformers with the Tuned Lens.* arXiv:2303.08112.
- Nanda & Bloom (2022). *TransformerLens.* [github.com/TransformerLensOrg/TransformerLens](https://github.com/TransformerLensOrg/TransformerLens)

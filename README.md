# CS 281 Project — Is Chain-of-Thought Just Rationalization?

Mechanistic faithfulness audit of Llama 3 8B chain-of-thought reasoning, using TransformerLens. Implements two methods: layer-wise logit lens for commitment-point detection, and CoT corruption / activation patching for causal influence measurement.

## Layout

```
cot_faithfulness/
  data.py         # BoolQ + MNLI loading from Hugging Face
  model.py        # TransformerLens Llama 3 8B loader
  prompts.py      # Llama 3 Instruct chat formatting + class target tokens
  generation.py   # CoT + answer generation, answer parsing
  logit_lens.py   # Layer-wise residual stream projection to vocab
  patching.py     # Random/shuffle CoT corruption + pre-CoT residual patching
  analysis.py     # Summary stats + matplotlib heatmaps
milestone.ipynb   # Runs the milestone plan of execution end to end
requirements.txt
results/          # Outputs land here (CSVs, .npy, PNGs, JSON)
```

## Run on Colab (recommended)

1. Upload the `cot_faithfulness/` folder and `milestone.ipynb` to Colab.
2. Set runtime to **GPU**, ideally A100 / L4. Llama 3 8B in fp16 needs ~16 GB VRAM.
3. Add your Hugging Face token (request access to `meta-llama/Meta-Llama-3-8B-Instruct` first):
   ```python
   import os; os.environ['HF_TOKEN'] = 'hf_...'
   ```
4. Run all cells. Outputs land in `results/`.

## Run locally

```bash
pip install -r requirements.txt
export HF_TOKEN=hf_...
jupyter notebook milestone.ipynb
```

Apple Silicon will work via MPS but is slow; CUDA strongly preferred.

## Outputs

| File | Contents |
| --- | --- |
| `results/boolq_layer_probs.npy` | (n_examples, n_layers, 2) — per-layer P(No), P(Yes) at the pre-CoT position |
| `results/boolq_commitment.csv` | Per-example commitment layer, prediction, correctness |
| `results/boolq_commitment_hist.png` | Histogram of commitment layers, split by correct/incorrect |
| `results/boolq_mean_curve.png` | Mean layer-wise P(Yes)/P(No) curve |
| `results/mnli_patching.csv` | Per-example clean vs. corrupted answer distributions |
| `results/mnli_corruption.png` | Logit-drop histogram + flip-rate count plot |
| `results/milestone_summary.json` | Aggregate numbers for the writeup |

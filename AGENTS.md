# AGENTS.md

Stanford CS 281 project (Brooke Ballhaus, Riya Narain) auditing whether chain-of-thought (CoT) in
`meta-llama/Meta-Llama-3-8B-Instruct` is causally used or post-hoc rationalization. It uses
TransformerLens to apply a logit lens and a tuned lens (at which layer does the answer commit, with
vs without CoT on BoolQ) and CoT corruption plus pre-CoT residual patching (does corrupting the
reasoning flip the answer). Code lives in a small Python package; experiments are driven from
Jupyter notebooks; outputs are committed under `results/`. Written findings: `results/RESULTS.md`.

## Repo map

- `cot_faithfulness/data.py` - BoolQ (`google/boolq`) and MNLI (`nyu-mll/multi_nli`) loading and sampling from HuggingFace.
- `cot_faithfulness/model.py` - `load_model` (TransformerLens `HookedTransformer`), `hf_login`, `pick_device`, `free_memory`; `DEFAULT_MODEL` and `ALT_MODELS`.
- `cot_faithfulness/prompts.py` - Llama-3 chat formatting, CoT vs direct prompts, answer target-token lookup.
- `cot_faithfulness/generation.py` - greedy CoT generation, answer parsing, CoT/answer split, answer-position scoring tokens.
- `cot_faithfulness/logit_lens.py` - per-layer residual projection onto answer tokens and commitment-layer detection.
- `cot_faithfulness/tuned_lens.py` - per-layer affine translators (`TunedLens`), `fit_tuned_lens`, `tuned_lens_probs`.
- `cot_faithfulness/patching.py` - random/shuffle/rewrite (invert, paraphrase) CoT corruption and pre-CoT residual patching.
- `cot_faithfulness/analysis.py` - summaries, bootstrap CIs, z-tests, threshold sweep, qualitative extremes, plots.
- `cot_faithfulness/experiments.py` - high-level runners `run_commitment` and `run_corruption`.
- `final.ipynb` - the final-report pipeline (source of truth for how experiments are run).
- `final_with_results.ipynb` - executed copy of the final pipeline with outputs; a committed record.
- `milestone.ipynb` - historical milestone experiments, superseded by `final.ipynb`.
- `results/` - committed CSVs, `.npy` arrays, JSON summaries, PNGs, `figures/`, and `RESULTS.md`.

## Setup

The model is gated on HuggingFace: request access to `meta-llama/Meta-Llama-3-8B-Instruct` first.
The token is read from the `HF_TOKEN` environment variable (see `.env.example`).

- Colab (recommended): GPU runtime, A100 or L4 (fp16 8B needs ~16 GB VRAM; T4 is borderline).
  Add a Colab Secret named `HF_TOKEN`. The setup cells pip-install deps, clone the repo into
  `/content/cot-faithfulness-audit`, and `chdir` there.
- Local: `pip install -r requirements.txt`, `cp .env.example .env`, put the token in `.env`
  (loaded with `python-dotenv`), then `jupyter notebook final.ipynb`. CUDA strongly preferred;
  MPS works but is slow; `pick_device` falls back to CPU (fp32).

Never print, log, or commit token values.

## How experiments are run

`final.ipynb` runs everything top to bottom: setup, Config cell (`CFG`: `N_BOOLQ`, `N_MNLI`, `TAU`,
`TAUS`, `MAX_LEN`, `COT_TOKENS`, `TUNED_FIT_N`, `TUNED_EVAL_N`, `QUAL_K`, `SEED`), load model, load
data, then:

1. BoolQ commitment, no-CoT vs with-CoT, via `experiments.run_commitment(..., with_cot=False/True)`.
2. Threshold sweep, qualitative extremes, tuned-lens vs logit-lens comparison.
3. BoolQ CoT corruption + pre-CoT patching (random, invert, shuffle, paraphrase). The notebook uses
   its own `run_boolq_corruption` / `run_corr` helpers that reuse saved with-CoT CoTs from
   `results/boolq_commitment_cot.csv`; `experiments.run_corruption` is the package-level runner.
4. `final_summary.json`, a results-analysis section that only reads `results/` (no model), stat
   tests, and a shuffled-target tuned-lens control (requires the section 8 lens in memory).

Scale is controlled from the Config cell; reduce `N_*` for quick runs.

## Where outputs land

All notebook outputs go to `results/` (paths built as `RESULTS / '<name>'` or `R / '<name>'`).
Several cells overwrite fixed filenames, and some cells read-modify-write `final_summary.json`.
`results/figures/` and the `mnli_corruption*` files are not written by the current `final.ipynb`.

## Tests and checks

- `tests/test_smoke.py` (pytest): CPU-only, no network, no HF token, runs in seconds. Imports every
  module and tests the pure helpers (answer parsing, CoT/answer split, prompt formatting, target-token
  collision check, commitment-layer detection, `TunedLens` init, corruption functions, bootstrap/
  summary/threshold-sweep helpers, plots) using fake tokenizers and synthetic arrays; it never loads a model.
- Run: install CPU torch (`pip install torch --index-url https://download.pytorch.org/whl/cpu`), then
  `pip install -r requirements.txt pytest`, then `python -m pytest -q`.
- CI: `.github/workflows/tests.yml` runs the same steps on Python 3.11 for pushes and PRs to `main`.
  It uses no secrets.
- `transformer-lens` is capped `<4`: 4.0 removed `HookedTransformer`, which `model.py` imports.
- No linter config. Add tests for new pure helpers; keep tests model-free.

## Conventions

- Library code goes in `cot_faithfulness/`; notebooks import it with
  `from cot_faithfulness import ...` after putting the repo root on `sys.path`.
- Plain functions, no classes except `TunedLens`; `@torch.no_grad()` on inference paths.
- Default seed 42, commitment threshold tau = 0.8, answers scored at the post-`Answer:` position.
- Datasets and models come from HuggingFace by ID; nothing is vendored.

## Guardrails

- GPU-heavy: ask before running anything that loads the 8B model (`load_model`, any cell from
  section 3 of `final.ipynb` on, `run_commitment`, `run_corruption`, `fit_tuned_lens`).
  Do not run the notebooks.
- Do not overwrite or delete files in `results/`; write experimental outputs elsewhere unless
  asked. `results/RESULTS.md` reports numbers tied to those files.
- `final_with_results.ipynb` is a committed record of an executed run. Edit `final.ipynb` instead;
  do not re-execute or strip outputs from the record.
- Never commit `.env` or any token; `.env.example` holds only a placeholder.
- Notebook edits: keep `.ipynb` valid JSON and do not add outputs to `final.ipynb`.

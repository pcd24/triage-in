# Triage-IN

Multi-task triage (**category**, **sentiment**, **urgency**) for code-mixed Indian customer
complaints in Hinglish, Telugu-English and English. It compares a TF-IDF baseline, a zero-shot
LLM, a fine-tuned MuRIL teacher and a distilled on-device TFLite student.

> **Status:** work in progress (Phase 1, dataset). All results will be generated from real
> runs into [results/RESULTS.md](results/RESULTS.md); no numbers are typed into this README.

The full specification is in [CLAUDE.md](CLAUDE.md). Everything runs at zero cost: the Gemini
free tier, open-weights models, and free Kaggle/Colab GPUs.

## Setup

```bash
python3.11 -m venv .venv-torch && .venv-torch/bin/pip install -e ".[torch,dev]"   # data, baseline, teacher
python3.11 -m venv .venv-tf    && .venv-tf/bin/pip install -e ".[tf,dev]"         # student, TFLite
make test && make lint
make smoke            # whole pipeline on ~200 rows with an offline fake LLM
```

## Building the dataset (Phase 1)

| Step | What | Command | Where |
|---|---|---|---|
| 1a | ~6,000 synthetic complaints (Gemini, labels by construction) | `python -m triage_in.data.generate_synthetic --yes` | Kaggle notebook |
| 1b | Blind second labeller, drop category disagreements | `python -m triage_in.data.validate_labels --yes` | Kaggle notebook |
| 1c | Real public sentiment data (CMTET Telugu-English, SentiMix Hinglish) | `python -m triage_in.data.load_public` | anywhere |
| 1d | 350 test candidates (different model + prompts) → **human review** | `python -m triage_in.data.generate_test --yes` | Kaggle notebook |
| 1d | Merge reviewed rows + hand-written messages | `python -m triage_in.data.merge_manual` | anywhere |
| 1e | Dedup, leakage guard, 90/10 split, stats | `python -m triage_in.data.build_splits` | anywhere |

The LLM steps live in [`notebooks/phase1_data_kaggle.ipynb`](notebooks/phase1_data_kaggle.ipynb).
Without `--yes`, each LLM script only prints the planned calls, tokens and cost.

**Human review (1d):** open `data/manual/test_synthetic_review.csv`, fix any wrong label, and set
`verified` to `yes` or `drop` on every row. Add your own hand-written messages to
`data/manual/test_manual.csv` (columns: `id,text,category,sentiment,urgency,lang_mix,notes`;
`id` may be blank). `merge_manual` refuses to run while any row is unreviewed.

### Data sources
- **Synthetic** complaints (Gemini free tier), checked by an independent open-weights labeller.
- **CMTET**: Kusampudi, Parimi & Bandyopadhyay, *Sentiment Analysis in Code-Mixed Telugu-English
  Text with Unsupervised Data Normalization*, RANLP 2021. Used for sentiment only; the public copy
  states no licence, so it is used here for research/portfolio purposes with citation.
- **SentiMix**: Patwa et al., *SemEval-2020 Task 9*. Download the official Hinglish train/dev CoNLL
  files and set their paths in `configs/data.yaml`.

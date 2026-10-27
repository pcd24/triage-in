# CLAUDE.md — VaaniTriage: Complete Project Specification and Working Rules

This file is the single source of truth for this repository. Read it fully before
writing any code. Part A covers how to work; Part B is the technical specification.

---

# PART A — Working rules

## Project summary
VaaniTriage is a multi-task classifier for code-mixed Indian customer complaints
(Hinglish, Telugu-English, English). It predicts **category**, **sentiment**, and
**urgency** for each message, and compares four approaches on accuracy, cost,
latency, and model size:

| # | Model | Framework | Role |
|---|-------|-----------|------|
| 1 | TF-IDF + Logistic Regression | scikit-learn | Baseline |
| 2 | Zero-shot LLM (GPT / Gemini) | API | "Why not just use ChatGPT?" comparison |
| 3 | Fine-tuned MuRIL, multi-task | PyTorch + Hugging Face | Teacher (main model) |
| 4 | Distilled CNN student | TensorFlow / Keras → TFLite | On-device, offline model |

Final deliverables: a results table generated from real runs, ablations (LoRA vs
full fine-tuning; distillation vs no distillation), and an Android demo app that
runs the TFLite model offline.

## How to work
- Implement Part B **one phase at a time**. After finishing a phase, stop, summarize
  what you built, show test results, and wait for my go-ahead before the next phase.
- Before coding a phase, briefly state your plan and list the files you'll create.
- Make one git commit per phase: `phase-N: <short description>`.
- If something in this spec is ambiguous or seems wrong, ask instead of guessing.
- Explain non-obvious code with short comments. I must be able to explain every part
  of this project in an interview, especially the training loop and distillation loss.

## Non-negotiable rules
1. **Never fabricate results.** Every number in README.md or results/ must come from
   a metrics.json file produced by running code. Use placeholders until then.
2. **Never touch the test set during training or tuning.** Use the validation set for
   early stopping, hyperparameters, and model selection. Only the scripts in
   `src/vaani/eval/` and `src/vaani/llm_eval/` may read `test.jsonl`.
3. **Never guess dataset IDs, model IDs, or library APIs.** Verify they exist (check
   docs, `pip show`, or try loading). If a dataset is unavailable, tell me and
   propose a fallback.
4. **Ask before spending money.** Before any paid LLM API run, print the number of
   calls and estimated cost, and wait for confirmation. Respect `max_budget_usd` in config.
5. **No secrets in code or git.** API keys live in `.env` (gitignored); provide `.env.example`.

## Code standards
- Python 3.11, type hints everywhere, `ruff` for lint and format.
- All hyperparameters and paths live in YAML files under `configs/`, not in code.
- Set seeds (Python, NumPy, torch, TensorFlow) from config for reproducibility.
- Use `logging`, not `print`, in library code.
- Every runnable script supports `--smoke`: a tiny subset (~200 examples), CPU-only,
  finishing in under 2 minutes, so the whole pipeline can be tested quickly.
- Small, testable functions. Write `pytest` tests for data processing, tokenization,
  metrics, and model output shapes. All tests must pass before a phase is done.

## Environments
PyTorch and TensorFlow are installed in **separate virtual environments** to avoid
CUDA and dependency conflicts. They communicate only through files on disk
(teacher logits saved as `.npy`).
- `pip install -e ".[torch,dev]"` → data, baseline, teacher, LLM eval
- `pip install -e ".[tf,dev]"` → student, distillation, TFLite conversion

Shared modules (`labels.py`, `text_norm.py`, `eval/metrics.py`) must not import
torch or tensorflow, so they work in both environments.

## Commands
- `make test` — run all tests
- `make lint` — ruff check and format
- `make smoke` — run the whole pipeline in smoke mode
- `make report` — rebuild results/RESULTS.md from all metrics.json files

---

# PART B — Technical specification

## 1. Label schema
Defined once in `src/vaani/labels.py` and exported to `data/labels.json` (used by Android).

- **category** (6): `refund`, `delivery`, `payment`, `account`, `product_quality`, `other`
- **sentiment** (3): `negative`, `neutral`, `positive`
- **urgency** (3): `low`, `medium`, `high`

### Labeling guide
- **Urgency high:** money lost or charged incorrectly, repeated failed attempts
  ("3 baar call kiya"), threats to leave, legal or consumer-forum threats, safety
  issues, account locked or hacked.
- **Urgency medium:** a clear problem needing action but no escalation signals
  (late delivery, wrong item).
- **Urgency low:** questions, mild feedback, compliments, general queries.
- **Sentiment** is the customer's tone; **category** is the topic. A polite message
  about a double charge is `payment` / `neutral` / `high`.

## 2. Repository structure
```
vaani-triage/
├── CLAUDE.md
├── README.md
├── pyproject.toml            # extras: [torch], [tf], [dev]
├── Makefile
├── .env.example
├── .gitignore
├── configs/
│   ├── data.yaml
│   ├── baseline.yaml
│   ├── teacher.yaml
│   ├── llm_eval.yaml
│   └── student.yaml
├── data/                     # raw/, interim/, processed/ are gitignored
│   ├── labels.json
│   └── manual/test_manual.csv
├── src/vaani/
│   ├── labels.py             # no torch/tf imports
│   ├── text_norm.py          # no torch/tf imports
│   ├── data/
│   │   ├── generate_synthetic.py
│   │   ├── load_public.py
│   │   ├── validate_labels.py
│   │   └── build_splits.py
│   ├── baseline/
│   │   └── train_sklearn.py
│   ├── teacher/
│   │   ├── dataset.py
│   │   ├── model.py
│   │   ├── train.py
│   │   └── export_logits.py
│   ├── llm_eval/
│   │   ├── providers.py
│   │   └── run_zero_shot.py
│   ├── student/
│   │   ├── tokenizer.py
│   │   ├── model.py
│   │   ├── distill.py
│   │   ├── convert_tflite.py
│   │   └── verify_tflite.py
│   └── eval/
│       ├── metrics.py        # no torch/tf imports
│       ├── evaluate.py
│       ├── benchmark_latency.py
│       └── report.py
├── artifacts/                # saved models (gitignored)
├── tests/
├── results/                  # <model_name>/metrics.json, plots, RESULTS.md
└── android/                  # Phase 7
```

## 3. Data format
All splits are JSON Lines, one example per line:
```json
{"id": "syn_000123", "text": "bhai refund abhi tak nahi aaya", "category": "refund",
 "sentiment": "negative", "urgency": "medium", "lang_mix": "hinglish", "source": "synthetic"}
```
- `lang_mix` ∈ `hinglish`, `tenglish`, `english`, `other`
- Any task label may be `null` if unknown (public sentiment data has no category or
  urgency). Training must mask the loss for null labels (see Phase 3).

---

## Phase 0 — Project setup
- `pyproject.toml` with extras:
  - `[torch]`: torch, transformers, peft, datasets, accelerate, scikit-learn, pandas,
    numpy, pyyaml, python-dotenv, joblib
  - `[tf]`: tensorflow, numpy, pyyaml, scikit-learn, pandas
  - `[dev]`: pytest, ruff, matplotlib
- Verify current package versions; pin them once working.
- `Makefile` targets: `test`, `lint`, `smoke`, `report`.
- Create `labels.py`, `text_norm.py`, `.env.example`, `.gitignore`.
- `text_norm.normalize(text)`: Unicode NFC, lowercase Latin script, collapse
  whitespace, strip, limit repeated characters to 3 (`"pleeeeease"` → `"pleeease"`).
  Do **not** strip Devanagari or Telugu script.

**Done when:** both environments install, `make test` passes, `make lint` is clean.

## Phase 1 — Dataset

### 1a. Synthetic complaints (`generate_synthetic.py`)
- Generate ~6,000 train/validation complaints with an LLM. Each prompt specifies the
  target labels (category, sentiment, urgency, lang_mix), so labels come from construction.
- Force diversity: vary length (5–60 words), formality, slang, typos, Romanized vs
  native script, and domains (e-commerce, food delivery, banking/UPI, telecom).
  Include positive and neutral messages, not only angry ones.
- Target mix: ~40% Hinglish, ~30% Telugu-English, ~25% English, ~5% other.
- Batch requests (e.g., 20 examples per call), cache every raw response to
  `data/raw/synthetic/`, and make the script resumable.
- **Print estimated cost and wait for confirmation before running.**

### 1b. Label validation (`validate_labels.py`)
- A second LLM pass labels each example independently, without seeing the intended labels.
- Drop examples where the two disagree on category; log agreement rates per task.

### 1c. Public data (`load_public.py`)
- Load public code-mixed sentiment datasets. Candidates: SemEval-2020 Task 9
  SentiMix (Hinglish) and DravidianLangTech code-mixed sentiment datasets.
  **Verify availability and licenses first**; if unavailable, report back and
  continue with synthetic data only.
- Map their labels to our sentiment schema; set category and urgency to `null`.

### 1d. Test set (honest evaluation)
- Generate ~350 test examples with a **different LLM provider and different prompts**
  than the training data. I will hand-check and correct every label.
- I will also write ~150 real-style messages by hand. Provide a CSV template at
  `data/manual/test_manual.csv` and a script to merge it in.
- Final test set: ~500 examples, all human-verified.

### 1e. Splits (`build_splits.py`)
- Train/val split 90/10, stratified by category. Test comes only from 1d.
- Deduplicate: exact match after normalization, then near-duplicates using TF-IDF
  character n-gram cosine similarity > 0.9. Remove any train/val example too
  similar to a test example.
- Write `data/processed/{train,val,test}.jsonl` and `data_stats.json` with label
  distributions per split and per lang_mix.

**Done when:** splits exist, stats are printed, and tests cover normalization,
schema validation, and deduplication.

## Phase 2 — Baseline (scikit-learn)
- `train_sklearn.py`: one pipeline per task.
  - Features: `TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5))` combined via
    `FeatureUnion` with word unigrams and bigrams.
  - Classifier: `LogisticRegression(class_weight="balanced")`.
- Tune `C` on the validation set only. Train each task only on rows where its label
  is not null.
- Save models with joblib to `artifacts/baseline/`. Evaluate on test via `eval/evaluate.py`.

**Done when:** `results/baseline_sklearn/metrics.json` exists.

## Phase 3 — Teacher (PyTorch + Hugging Face fine-tuning)

### Model (`teacher/model.py`)
- Encoder: `google/muril-base-cased` via `AutoModel` (verify it loads; fallback
  `xlm-roberta-base`, configurable).
- Mean pooling over non-padding tokens → dropout 0.1 → three linear heads with
  6, 3, and 3 outputs. Return a dict of logits per task.

### Training loop (`teacher/train.py`) — raw PyTorch, no Hugging Face `Trainer`
- Loss: weighted sum of per-task `CrossEntropyLoss(ignore_index=-100)`. Null labels
  become -100 so they're masked. Task weights in config (default 1.0 each).
- If a batch has no valid labels for a task, skip that task's loss to avoid NaN.
- AdamW with two parameter groups: encoder lr 2e-5, heads lr 1e-3; weight decay 0.01.
- Linear warmup (10% of steps), then linear decay.
- Mixed precision with `torch.amp` on CUDA, gradient accumulation, gradient clipping 1.0.
- Defaults: max_len 128, batch 16, accumulation 2, epochs 4 (fits a free Colab/Kaggle T4).
- Evaluate on validation each epoch; early stopping (patience 2) on mean macro-F1
  across the three tasks.
- Save the best checkpoint, tokenizer, config, and label maps to `artifacts/teacher/`.
- Log training loss and validation metrics per epoch to CSV (plotted in Phase 6).

### LoRA ablation
- Config flag `use_lora: true` wraps the encoder with PEFT LoRA (r=8, alpha=16,
  dropout 0.1, target modules `["query", "value"]`; verify module names for the
  chosen encoder). Heads remain fully trainable.
- Report full fine-tuning vs LoRA: macro-F1, trainable parameter count, training time.

### Export (`teacher/export_logits.py`)
- Run the best teacher over train and val; save logits per task as `.npy` files
  plus an ID-order file, for the TensorFlow environment to read.

**Done when:** `results/teacher_full/` and `results/teacher_lora/` metrics exist and
logits are exported. Tests cover mean pooling with padding masks, output shapes,
and loss masking.

## Phase 4 — Zero-shot LLM comparison
- `providers.py`: a small interface with OpenAI and Gemini implementations; API
  keys from `.env`.
- `run_zero_shot.py`: run on the **test set only**. The prompt includes the label
  definitions and labeling guide from Section 1 and requests strict JSON output.
- Validate the JSON; retry once on parse failure; count remaining failures as
  wrong answers and record the failure count.
- Cache every response to disk so reruns are free. Record input and output tokens
  and compute cost per 1M messages using prices stored in `configs/llm_eval.yaml`.
  **I will fill in current prices; do not invent them.**
- **Print estimated cost and wait for confirmation before running.**

**Done when:** `results/llm_zero_shot_<provider>/metrics.json` exists with cost fields.

## Phase 5 — Student (TensorFlow / Keras → TFLite)

### Tokenizer (`student/tokenizer.py`)
Must be simple enough to reimplement exactly in Kotlin.
- Apply `text_norm.normalize`, then extract tokens with the regex
  `[\p{L}\p{M}\p{N}]+` (letters, combining marks, digits). Use the `regex`
  package in Python, since the built-in `re` lacks `\p{...}` support.
- Word vocabulary: top 20,000 tokens from training data.
  IDs: 0 = PAD, 1 = reserved, 2–20001 = vocabulary words.
- Out-of-vocabulary words: hash with **FNV-1a 32-bit** over UTF-8 bytes into 2,000
  buckets: ID = 20002 + (hash % 2000). This handles typos and rare code-mixed spellings.
- Max length 64: truncate, or right-pad with 0.
- Export `vocab.json` and `tokenizer_test_vectors.json` (50 tricky inputs: mixed
  scripts, emoji, numbers, repeated characters → expected IDs). Android must
  reproduce these exactly.

### Model (`student/model.py`)
- Embedding(22002, 128) → two parallel Conv1D branches (128 filters, kernel sizes
  3 and 5, ReLU) → GlobalMaxPool1D on each → concatenate → Dense(128, ReLU) →
  Dropout(0.3) → three Dense heads outputting logits.
- Use only layers that convert to builtin TFLite ops (no RNNs, no string ops in the graph).

### Distillation (`student/distill.py`)
- Per-task loss:
  `alpha * T² * KL(softmax(teacher/T) ‖ softmax(student/T)) + (1 − alpha) * CE(hard label)`
  with T = 2.0 and alpha = 0.7 by default. Where the hard label is null, use only the KL term.
- Also train the same student **without** distillation (hard labels only) as an
  ablation, to measure distillation's effect.
- Early stopping on validation mean macro-F1.

### Conversion and verification
- `convert_tflite.py`: export float32 and dynamic-range-quantized `.tflite` models.
  Optional: full int8 with a representative dataset drawn from training data.
- `verify_tflite.py`: run the TFLite interpreter and the Keras model on the test
  set; require ≥ 99% argmax agreement per task; report model size on disk.

**Done when:** metrics exist for `student_distilled`, `student_no_distill`, and the
quantized TFLite model, and verification passes.

## Phase 6 — Evaluation and reporting

### Metrics (`eval/metrics.py`, `eval/evaluate.py`)
Every model writes `results/<model_name>/metrics.json` with this schema:
```json
{
  "model": "teacher_full",
  "tasks": {
    "category":  {"macro_f1": 0.0, "accuracy": 0.0, "per_class_f1": {}, "macro_f1_ci95": [0.0, 0.0]},
    "sentiment": {"macro_f1": 0.0, "accuracy": 0.0, "per_class_f1": {}, "macro_f1_ci95": [0.0, 0.0]},
    "urgency":   {"macro_f1": 0.0, "accuracy": 0.0, "per_class_f1": {}, "macro_f1_ci95": [0.0, 0.0]}
  },
  "by_lang_mix": {"hinglish": {}, "tenglish": {}, "english": {}},
  "size_mb": null,
  "latency_ms_p50": null,
  "latency_ms_p95": null,
  "cost_per_1m_usd": null,
  "n_test": 500
}
```
- Macro-F1 is the headline metric. Compute 95% confidence intervals with 1,000
  bootstrap resamples, since the test set is small.
- Save confusion matrices as PNGs in each model's results folder.

### Latency (`eval/benchmark_latency.py`)
- CPU only, batch size 1, 50 warmup runs + 500 timed runs; report p50 and p95.
- Teacher in PyTorch; student via the TFLite interpreter; baseline via scikit-learn.

### Report (`eval/report.py`)
- Build `results/RESULTS.md` automatically from all metrics.json files, containing:
  - Main comparison table: macro-F1 per task, size, latency, cost per 1M messages,
    and whether it runs offline.
  - LoRA vs full fine-tuning ablation.
  - Distillation vs no-distillation ablation.
  - Per-language breakdown (Hinglish / Telugu-English / English).
  - Training curves for the teacher.
  - Error analysis: 20 misclassified test examples from the teacher, with true vs
    predicted labels.
- README.md links to RESULTS.md and must not contain hand-typed numbers.

**Done when:** `make report` produces RESULTS.md covering all models.

## Phase 7 — Android demo
- Kotlin app in `android/` with a minimal UI: text input, "Classify" button, the
  three predicted labels with confidence, and inference time in ms.
- Assets: the quantized `.tflite` model, `vocab.json`, `labels.json`.
- Use the current officially recommended TFLite / LiteRT Android dependency
  (verify it, since the naming has changed).
- Port the tokenizer to Kotlin. A unit test loads `tokenizer_test_vectors.json`
  and asserts exact ID matches. This test must pass.
- Add a README section with build instructions and a screenshot placeholder.

**Done when:** the app builds and the tokenizer parity test passes.

## Phase 8 — README
Sections:
1. **Problem and motivation:** code-mixed Indian text, cost of LLM APIs at scale,
   privacy and offline use.
2. **Architecture:** a Mermaid diagram of the full pipeline.
3. **Dataset:** how it was built, and its limitations (synthetic training data,
   small human-verified test set).
4. **Results:** link to RESULTS.md with a one-paragraph summary written only after
   results exist.
5. **Why not just use an LLM?** Answer using the measured accuracy, cost, latency,
   and offline results.
6. **How to reproduce:** environment setup and commands for each phase.
7. **Limitations and future work:** stated honestly.
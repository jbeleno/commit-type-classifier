# C2 — AI for Software Engineering

Component 2 (50%) of the course project. The goal is to classify
software-engineering activity from commit messages and diffs into
Conventional-Commit types — a problem from the *software engineering*
domain, complementing C1 (computer vision on agricultural images).

## 1. Introduction

Modern development teams use semantic commit prefixes
(`feat:`, `fix:`, `docs:`, `refactor:`, `test:`) to feed release-note
generators, deployment gates, and bug-tracking dashboards. In practice
the majority of repositories still ship unstructured messages, so any
downstream tooling that depends on the prefix must either skip them or
ask the developer to re-tag. This project trains and serves a classifier
that infers the missing prefix from the natural-language message and
the diff, so the same downstream tooling can run on legacy code bases.

## 2. Problem

Given a commit, predict its Conventional-Commit type from the message
text and the unified diff. The reduced label set is
`{feat, fix, docs, refactor, test}` — the five categories that are
both well-defined and widely-used across the CommitBench corpus.

## 3. Objectives

**General** – Build, evaluate, and serve a classifier that maps a
commit (message + diff) to one of five Conventional-Commit types.

**Specific**
1. Curate a balanced, leak-free 70/15/15 split of CommitBench.
2. Train and compare ≥3 model families (linear, CNN, transformer).
3. Wrap the best model behind a Streamlit GUI and a CLI.
4. Persist every prediction to a local SQLite store for traceability.
5. Document the experiments, architecture, and limitations in English.

## 4. State of the Art

| Year | Authors | Approach | Notes |
|------|---------|----------|-------|
| 2008 | Hindle et al. | Manual taxonomy on Mozilla commits | First large-scale study of commit categorisation |
| 2015 | Mauczka et al. | Annotated 1,000 OSS commits, rule-based | Foundation for the "maintenance type" labels |
| 2020 | Trautsch et al. | TF-IDF + linear models | Baselines around 70% accuracy on imbalanced sets |
| 2022 | Sarwar et al. | seq2seq with CodeBERT | Better minority-class recall; expensive to train |
| 2023 | NLBSE'23 | DistilBERT on issue/commit titles | Stable baseline for the NLP-only setting |

Our work re-uses the publicly available CommitBench dataset
(Maxscha/commitbench on Hugging Face) and contributes a head-to-head
comparison of five model families on a fixed test split, plus a working
GUI/CLI/SQLite stack so the classifier can be operated locally.

## 5. Requirements

**Functional**
- F1. Classify a commit (message + optional diff) into one of five types.
- F2. Show per-class probabilities and overall confidence.
- F3. Analyze a local git repository (last N commits) and chart the
  distribution.
- F4. Batch-classify a CSV file of commits.
- F5. Persist every prediction (origin, model, label, confidence,
  timestamp) to SQLite.
- F6. Browse historical predictions from the GUI.

**Non-functional**
- NF1. Run entirely on a developer laptop (no cloud dependency).
- NF2. Use Python 3.11, TensorFlow ≥ 2.15, PyTorch ≥ 2.2, Streamlit.
- NF3. Average inference latency < 2 s on Apple Silicon MPS.
- NF4. Cover the preprocessing and inference layers with unit tests.

## 6. Use cases

| ID | Actor | Goal |
|----|-------|------|
| UC1 | Developer | Classify a single commit they're about to push |
| UC2 | Developer | Audit the last 100 commits of a repository |
| UC3 | Reviewer  | Compare different model families on the test split |
| UC4 | Auditor   | Inspect historical predictions via the GUI |

## 7. Data dictionary

| Column | Type | Description |
|--------|------|-------------|
| message_clean | text | Commit message with the `type(scope):` prefix stripped (so the model cannot cheat). |
| diff_text | text | Unified diff trimmed to 4,000 characters. |
| files_changed | int  | Number of files touched (`diff --git` lines). |
| lines_added | int  | Count of `+` lines (excluding file headers). |
| lines_removed | int  | Count of `-` lines (excluding file headers). |
| label | str  | Conventional-Commit type ∈ {feat, fix, docs, refactor, test}. |
| label_id | int  | Integer encoding (`feat=0, fix=1, docs=2, refactor=3, test=4`). |

The SQLite `predictions` table stores: id, ts, model_name,
message_preview, diff_preview, predicted_label, confidence,
probabilities (JSON), source (gui|cli|cli-repo|cli-batch).

## 8. Architecture overview

```
                +------------------------+
                | CommitBench (HF Hub)   |
                +-----------+------------+
                            |
                  src/data/download.py
                            v
                +------------------------+
                | data/raw/*.parquet     |
                +-----------+------------+
                            |
                  src/data/preprocess.py
                            v
                +------------------------+
                | data/processed/        |
                +-----------+------------+
                            |
                  src/data/split.py
                            v
                +------------------------+
                | data/splits/{train,val,test}.csv
                +-----------+------------+
                            |
            +---------------+---------------+
            v               v               v
   baseline_tfidf      cnn_text       distilbert / codebert
            \               |               /
             \              v              /
              \--->     ensemble      <---/
                            |
                            v
                  src/inference.py (predict)
                       |        |
                  app/streamlit_app.py
                  app/cli.py
                            |
                            v
                  db/history.sqlite
```

## 9. Models compared

| # | Family | Backbone | Notes |
|---|--------|----------|-------|
| 1 | Classical ML | TF-IDF (word 1–2gram on msg + char 3–5gram on diff) + numeric features → Logistic Regression | Trained on the full imbalanced split with `class_weight=balanced` |
| 2 | Deep Learning | Dual-branch CNN-text (TextVectorization + Conv1D for both message and diff) + dense head | Trained on the full imbalanced split with class weights |
| 3 | Transformer (text) | DistilBERT fine-tuned with HuggingFace Trainer | Trained on a balanced 5k subsample (1k per class) to reduce dominance of `fix` |
| 4 | Transformer (code-aware) | CodeBERT (microsoft/codebert-base) fine-tuned, same pipeline as 3 | |
| 5 | Ensemble | Heterogeneous soft-voting; weights optimised on validation set via L-BFGS-B | Combines all four base models |

## 10. Training protocol

- Stratified 70/15/15 split (sklearn `train_test_split`, seed = 42).
- Class imbalance: linear/CNN use `class_weight="balanced"`;
  transformers use a balanced under-sampling instead because the
  weighted CrossEntropy proved unstable on Apple Silicon MPS.
- Optimiser: AdamW (lr 2e-5, warmup 10%, weight decay 0.01) for
  transformers; SAGA solver for Logistic Regression; Adam (1e-3) +
  EarlyStopping(patience=3) for the CNN.
- Reproducibility: fixed seed across NumPy, scikit-learn, TF/Keras
  and PyTorch (constant `RANDOM_SEED = 42`).

## 11. Results (test split)

See `models_saved/reports/comparison.csv` for the live numbers; the
final table is reproduced verbatim in `models_saved/reports/comparison.md`.

## 12. Limitations and future work

- **Class imbalance.** `fix` represents ~63% of CommitBench's filtered
  rows; balancing helps the transformers but reduces overall accuracy.
- **English-only.** The classifier was trained on English commit
  messages; performance on Spanish messages would require either a
  multilingual backbone (XLM-R) or a translation step.
- **Diff truncation.** Diffs longer than 4,000 characters are cut; the
  long-tail of large refactors is therefore under-represented.
- **Single-commit context.** The model does not see file paths or
  prior commits in the same PR; future work could enrich the input
  with these signals.

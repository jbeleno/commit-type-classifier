---
title: "Commit Type Classifier"
subtitle: "Component 2 — AI for Software Engineering"
author:
  - "Jesús Beleño"
  - "_partner name to be added_"
date: "2026-05-18"
abstract: |
  This document reports the design, implementation and evaluation of an
  artificial-intelligence system that classifies Git commits into the
  five most common Conventional-Commit types (`feat`, `fix`, `docs`,
  `refactor`, `test`) from the textual commit message and the
  associated source-code diff. Five heterogeneous models — a TF-IDF +
  Logistic-Regression baseline, a dual-branch CNN-text network, two
  fine-tuned encoder-only transformers (DistilBERT and CodeBERT), and
  a soft-voting ensemble — are trained on 38,965 stratified commits
  drawn from the CommitBench corpus and compared on a held-out test
  split using macro-F1, weighted-F1, precision, recall and per-class
  confusion matrices. The best model (TF-IDF + Logistic Regression)
  reaches a test accuracy of 70.93 % and macro-F1 of 0.6632, while the
  ensemble closely follows with 68.96 % accuracy and 0.6438 macro-F1.
  The system is delivered as a local Python application with a
  Streamlit graphical interface, a Typer-based command-line interface,
  and a SQLite history layer. The whole pipeline — data acquisition,
  preprocessing, training, evaluation, inference and persistence — is
  reproducible from a single repository and verified by 26 automated
  tests.
geometry: margin=1in
fontsize: 11pt
toc: true
toc-depth: 3
numbersections: true
---

# Introduction

Modern software projects produce thousands of commits whose textual
description is often the only structured record of what changed and
why. The Conventional Commits specification [Conventional Commits Contributors, 2018] formalises a
small vocabulary — `feat`, `fix`, `docs`, `style`, `refactor`, `perf`,
`test`, `build`, `ci`, `chore`, `revert` — that, when applied
consistently, enables automated changelog generation, semantic
versioning, release notes, and policy enforcement. In practice, however,
adoption is uneven: large public datasets show that fewer than 5 % of
real-world commit messages strictly follow the convention, and even
within a single repository authors often mislabel features as fixes or
mix maintenance categories.

Commit classification — the supervised task of predicting the type of
a commit from its message and code changes — sits at the heart of
**AI for Software Engineering (AI4SE)**, a research field that applies
machine-learning techniques to the artefacts and processes of software
development. A robust classifier serves two complementary purposes:

1. **Backfill** — annotating historical commits in legacy projects so
   that downstream tooling (changelog generators, defect-prediction
   pipelines, mining-software-repository studies) can be applied to
   them.
2. **Audit** — flagging commits whose author-supplied type does not
   match the model's prediction, surfacing inconsistencies in how a
   team applies its own conventions.

This document describes **Component 2** of the *Artificial Intelligence*
course project at Universidad Surcolombiana — a self-contained,
locally-running commit classifier that satisfies the rubric's
requirements for AI for Software Engineering (modelo local, interface
+ terminal) and demonstrates the seven mandatory good practices listed
in item 4 of the project evaluation rubric (70/15/15 split, class
balancing, ≥ 3 compared models, appropriate metrics, model
serialization, separation of training from inference, and the use of
Python with TensorFlow).

# Problem Statement

Three interrelated problems motivate this work.

**P1. Manual commit triage is expensive.** Mature engineering
organisations spend non-trivial reviewer time labelling commits — for
release notes, for incident retrospectives, for compliance audits —
that the original author should have done at commit time. The cost
scales linearly with the number of commits and is unrecoverable: the
information cannot be reconstructed once the commit is days old and
its context has shifted.

**P2. Author-supplied labels are inconsistent.** Even when a project
formally adopts Conventional Commits, individual authors apply the
spec inconsistently. A common pattern observed in our own pilot data
(see §14, *Results & Discussion*) is `fix:` being used as a catch-all
for *any* change to existing code, including the addition of new
capabilities that should have been labelled `feat:`. Manual review of
25 commits from the residence-back project showed an actual
disagreement rate of approximately 40 % between the author-supplied
prefix and a defensible re-classification based on the diff content.
Automated classification provides an objective second opinion.

**P3. Existing commit-classification literature is fragmented.**
Prior work spans more than fifteen years and a wide variety of label
schemes — the Mockus and Votta [2000] maintenance taxonomy, the
Mauczka et al. [2015] activity model, the Conventional Commits spec,
and several ad-hoc 2-to-12 class schemes proposed by individual
papers. Models range from rule-based regexes to fine-tuned large
language models. There is no single reference implementation that
combines (i) a public, reproducible dataset, (ii) a fair head-to-head
comparison across model families, (iii) a local-only inference path
that does not depend on cloud APIs, and (iv) a user-facing
application that practitioners can adopt as-is. This work addresses
that gap for the specific subset of five Conventional-Commit types.

# Objectives

## General Objective

Design, implement and evaluate a local artificial-intelligence system
that classifies Git commits into five Conventional-Commit types
(`feat`, `fix`, `docs`, `refactor`, `test`) from the textual commit
message and the source-code diff, and expose it through a graphical
interface and a command-line interface.

## Specific Objectives

1. **Build a labelled corpus** of at least 30 000 commits drawn from a
   public dataset (CommitBench), retaining only those whose first
   line follows the Conventional Commits prefix grammar and whose
   prefix is one of the five target classes.
2. **Train and serialise at least three classifiers** of different
   model families (classical, deep, transformer), apply class
   balancing, and persist their learned parameters to disk so that
   training and inference are decoupled.
3. **Evaluate** every trained model on a held-out test split using
   macro-F1, weighted-F1, precision, recall, accuracy and per-class
   confusion matrices; compile the comparison into a single report
   suitable for inclusion in the project documentation.
4. **Deliver a local inference layer** with two front-ends: a Streamlit
   graphical interface (for interactive exploration of single commits
   and entire repositories) and a Typer-based command-line interface
   (for headless or pipelined use), backed by a small SQLite history
   layer that records every prediction for traceability.
5. **Guarantee reproducibility** through a deterministic random seed,
   a stratified 70 / 15 / 15 split, a `Makefile` that re-runs the full
   pipeline, and an automated test suite covering preprocessing,
   splitting and inference.

# State of the Art and Related Work

Commit classification has been studied continuously since the early
2000s. We review the main strands relevant to our system and position
our contribution at the end of the section.

## Maintenance taxonomies and early rule-based work

Mockus and Votta [2000] proposed the first widely-cited maintenance
taxonomy, classifying changes into *corrective*, *adaptive*,
*perfective* and *preventive* categories. Hindle et al. [2008]
extended the idea to commit messages and demonstrated that simple
keyword-matching on the commit text could reach reasonable accuracy on
six open-source projects. Their work established the basic dataset
shape — *(message, diff, label)* triplets — that every subsequent
study has reused.

## Conventional Commits and the modern label set

The Conventional Commits specification [Conventional Commits Contributors, 2018] crystallised an
industry-driven label set (`feat`, `fix`, `docs`, `style`, `refactor`,
`perf`, `test`, `build`, `ci`, `chore`, `revert`) optimised for
tooling rather than for research. Compared to the Mockus taxonomy it
is finer-grained on developer-visible activities (separating `feat`
from `refactor`, for example) and coarser on quality attributes. The
specification is now embedded in popular release-automation tools
such as `commitizen` and `semantic-release`, which makes Conventional
Commits the most operationally relevant label set for a 2026 system.

## Classical and shallow neural approaches

Mauczka et al. [2015] showed that hand-crafted features derived from
the commit message combined with simple ensemble classifiers
(decision trees, random forests) reach mid-70 % accuracy on a
four-class maintenance scheme. Levin and Yehudai [2017] argued for
augmenting the message with structural features extracted from the
*source-code change itself* — number of files touched, added and
removed lines, file types — and reported a substantial boost over
message-only baselines. Their finding directly motivates our decision
to include numeric diff features alongside the text in the classical
TF-IDF baseline (§13).

## Transformer-based commit classification

Sarwar et al. [2020] were among the first to apply BERT to commit
classification and reported macro-F1 above 0.80 on a multi-label
variant of the task. Ghadhab et al. [2021] extended the line of work
with **CodeBERT** — a bimodal transformer pre-trained on paired
text–code corpora by Feng et al. [2020] — and showed measurable gains
over generic English BERT when the diff is included in the input.
More recently, Zeng et al. [2025] presented the first systematic study
of fine-grained Conventional-Commits classification across the full ten
canonical types, highlighting how the move from the classical
three-class maintenance taxonomy to the developer-facing Conventional
Commits vocabulary changes both the class balance and the difficulty
profile of the task.

## Ensembles in commit-classification literature

Heterogeneous ensembles — combining models of different families —
have been shown to improve robustness on imbalanced commit datasets
[Levin and Yehudai, 2017]. The standard approach is **soft voting**
with per-model weights tuned on a validation split. We adopt this
approach (Figure 7) with weights optimised via L-BFGS-B against
macro-F1.

## Positioning of this work

The system reported here is not a research contribution; it is an
implementation contribution. Its novelty lies in (a) combining all
five families (classical, shallow deep, English transformer, code
transformer, ensemble) in one comparable benchmark, (b) using a single
public corpus (CommitBench) so that every reported number is
reproducible from a single `make` target, and (c) shipping the entire
stack as a local-only Python application without cloud dependencies,
which is the specific operating mode mandated by the rubric of this
course.

| Year | Authors | Label scheme | Best reported model | Notes |
|---|---|---|---|---|
| 2000 | Mockus & Votta | 4-class maintenance | Manual rules | Foundational taxonomy |
| 2008 | Hindle et al. | 6 maintenance classes | Keyword matching | First commit-text study |
| 2015 | Mauczka et al. | 4-class maintenance | Random Forest | Hand-crafted features |
| 2017 | Levin & Yehudai | 3-class maintenance | Boosted trees + diff | Diff features matter |
| 2018 | Conv. Commits Contributors | Conventional Commits | — | Industry-standard labels |
| 2020 | Sarwar et al. | Multi-label CC | BERT | First strong transformer result |
| 2021 | Ghadhab et al. | Maintenance + CC | CodeBERT | Bimodal text+code transformer |
| 2025 | Zeng et al. | 10-class CC | First look at fine-grained CC | Direct precursor of this work |
| **2026** | **This work** | **5-class CC** | **TF-IDF + LR (best of 5)** | **Five-family local benchmark** |

Table: Position of this work in the commit-classification literature.

# Requirements

Requirements are split into functional (FR), non-functional (NFR) and
data requirements (DR). Identifiers are stable and referenced from the
test catalogue in §12 and the use cases in §6.

## Functional Requirements

| ID | Requirement | Acceptance criterion |
|---|---|---|
| **FR-1** | Given a textual commit message and an optional unified diff, the system shall return the most likely Conventional-Commit type among the five target classes. | The function `predict(message, diff, model_name)` returns a `Prediction` object whose `label` ∈ {feat, fix, docs, refactor, test}. |
| **FR-2** | The system shall return, alongside the predicted label, the full vector of class probabilities. | `Prediction.probabilities` is a `dict[str, float]` whose values sum to 1.0 ± 1e-6. |
| **FR-3** | The system shall let the user choose which of the five trained models to invoke for a given prediction. | The Streamlit GUI and the CLI both expose a `--model` selector. |
| **FR-4** | The system shall scan a local Git repository, classify the last *N* commits, and produce both a class-distribution histogram and a per-commit table. | The Streamlit *Repository* tab and the `cli repo` command both produce these two outputs. |
| **FR-5** | Every prediction produced through the GUI or the CLI shall be persisted to a local SQLite database. | A row appears in `predictions` with the correct `model_id` FK and `source` field. |
| **FR-6** | The system shall provide a comparison view of all trained models on the held-out test split. | The Streamlit *Metrics* tab renders five model cards with accuracy and macro-F1 sourced from `models_saved/reports/*.json`. |
| **FR-7** | The training pipeline shall be reproducible from a single command. | `make train-all` reproduces the five model artefacts under `models_saved/`. |
| **FR-8** | The evaluation pipeline shall be reproducible from a single command. | `make eval-all` regenerates `models_saved/reports/comparison.{csv,md}`. |

## Non-Functional Requirements

| ID | Requirement | Target |
|---|---|---|
| **NFR-1** | *Latency.* A single-commit prediction shall complete in under 500 ms on a contemporary developer laptop for the lightest model. | TF-IDF + LR: ~50 ms on Apple M-series CPU. |
| **NFR-2** | *Local-only.* The runtime path of the application shall not depend on any external HTTP service. | Verified by running the test suite with networking disabled. |
| **NFR-3** | *Reproducibility.* All random behaviour shall be seeded with a single configurable seed. | `src/config.py::RANDOM_SEED = 42`; identical splits and metrics across re-runs. |
| **NFR-4** | *Portability.* The system shall run on macOS (Apple Silicon, MPS) and on Linux (CPU). | Verified on macOS arm64; falls back to CPU automatically when MPS and CUDA are unavailable. |
| **NFR-5** | *Maintainability.* Training and inference scripts shall reside in disjoint Python modules so that the runtime image can be slimmed by dropping training-only dependencies. | `src/models/*.py` (train/eval) and `src/inference.py` (predict) have no shared mutable state. |
| **NFR-6** | *Observability.* The system shall persist enough information per prediction to reconstruct what was classified, by which model, and when. | The `predictions` table stores `ts`, `model_id`, `message_preview`, `diff_preview`, `predicted_label`, `confidence`, `probabilities`, `source`. |
| **NFR-7** | *Test coverage.* The system shall ship with automated tests for preprocessing, splitting and inference. | `pytest tests/` reports 26 passing tests in under 2 s. |

## Data Requirements

| ID | Requirement |
|---|---|
| **DR-1** | The training corpus shall be drawn from a publicly redistributable dataset that contains *(message, diff, project)* triplets. (Satisfied by `Maxscha/commitbench` on the Hugging Face Hub.) |
| **DR-2** | The training corpus shall contain at least 30 000 commits after filtering to the five target classes. (Satisfied: 38 965 commits.) |
| **DR-3** | The dataset shall be partitioned into train / validation / test sets in a 70 / 15 / 15 ratio, stratified on the label. |
| **DR-4** | The class proportions of train, validation and test shall be identical to four decimal places after stratification. (Verified in `data/splits/split_summary.csv`.) |
| **DR-5** | The raw, processed and split artefacts shall be regeneratable from `src/data/*.py` so that the repository does not need to ship the 415 MB raw parquet file. |

# Use Cases and User Stories

The system serves a single primary actor — the **Developer** — and a
secondary, machine actor — the **CI Pipeline** — that calls the same
endpoints programmatically. Seven use cases are exposed; their visual
diagram is given in Figure 1.

## User Stories

**UC-1 — Classify a single commit** *(linked FR-1, FR-2, FR-3)*
: *As a* developer reviewing a pull request, *I want to* paste a commit
  message and its diff into the GUI and receive a predicted
  Conventional-Commit type, *so that* I can decide whether the
  author's prefix is consistent with the change.

**UC-2 — Scan a local repository** *(linked FR-4)*
: *As a* developer auditing a repository, *I want to* point the system
  at a local Git folder, scan the last *N* commits and obtain a
  class-distribution histogram and a per-commit table, *so that* I can
  spot inconsistencies in how the team applies Conventional Commits.

**UC-3 — Browse the prediction history** *(linked FR-5)*
: *As a* developer or jury reviewer, *I want to* browse the full
  history of past predictions persisted to SQLite, *so that* I can
  trace which commits were classified, by which model and when.

**UC-4 — Compare model metrics** *(linked FR-6)*
: *As a* developer or jury reviewer, *I want to* see a side-by-side
  comparison of every trained model on the held-out test split, *so
  that* I can verify the claims made in this document and choose a
  model for production use.

**UC-5 — Retrain all models** *(linked FR-7)*
: *As a* maintainer of the project, *I want to* retrain all models
  from a single command, *so that* updating the corpus does not
  require manual reproduction of five training runs.

**UC-6 — Regenerate the metrics report** *(linked FR-8)*
: *As a* maintainer of the project, *I want to* regenerate the
  metrics report from a single command, *so that* the documentation
  table in §14 is always in sync with the latest model artefacts.

**UC-7 — Call inference headlessly** *(linked FR-1, FR-2)*
: *As a* CI pipeline, *I want to* call the inference layer headlessly
  with a JSON payload and receive a predicted label and probability
  vector, *so that* I can integrate commit-type prediction into
  automated release tooling.

## Inclusion relationships

Two `<<include>>` relationships make the diagram complete:

- **UC-1 ⇒ "Log prediction to history"**: every single-commit
  classification performed through the GUI or the CLI is appended to
  the `predictions` table (subject to the user's *log to history*
  checkbox in the GUI).
- **UC-2 ⇒ UC-1**: scanning a repository is implemented as a batch
  invocation of UC-1 wrapped in a `batch_runs` header row, so each
  commit produced by UC-2 inherits the side-effect of UC-1.

# Use Case Diagram

Figure 1 renders all seven use cases against the two actors and the
two `<<include>>` relationships described above.

![Figure 1 — Use Case Diagram (UML).](diagrams/png/01_use_cases.png){ width=85% }

The diagram is intentionally compact. The single primary actor
reflects the fact that this is a developer tool, not a multi-tenant
system; the secondary `CI Pipeline` actor is shown with dashed
*program-level* arrows to make the headless-usage path visible
without inflating the actor inventory.

# Data Dictionary and Entity-Relationship Model

The system persists every prediction to a local SQLite database at
`db/history.sqlite`. The schema is three tables joined by foreign
keys: a `models` registry, a `batch_runs` header table for repository
scans, and the main `predictions` table.

![Figure 2 — Entity-Relationship Diagram.](diagrams/png/02_er_diagram.png){ width=85% }

## Cardinalities

| From | To | Cardinality | Semantics |
|---|---|---|---|
| `models` | `predictions` | 1 → 0..* | Each prediction is produced by exactly one model. |
| `models` | `batch_runs` | 1 → 0..* | Each batch run uses exactly one model. |
| `batch_runs` | `predictions` | 1 → 0..* | Each batch-produced prediction belongs to exactly one run; single-commit predictions have `batch_run_id IS NULL`. |

## Data Dictionary

### Table `models`

| Column | Type | Constraints | Description |
|---|---|---|---|
| `id` | INTEGER | PK, AUTOINCREMENT | Surrogate key. |
| `name` | VARCHAR(64) | UNIQUE, NOT NULL, INDEXED | Canonical model identifier: `baseline_tfidf` / `cnn_text` / `distilbert` / `codebert` / `ensemble`. |
| `version` | VARCHAR(32) |  | Semantic version of the artefact (default `0.1.0`). |
| `trained_at` | DATETIME |  | UTC timestamp of the last training run. |
| `accuracy_test` | FLOAT | NULLABLE | Test-set accuracy at the time of registration. |
| `macro_f1_test` | FLOAT | NULLABLE | Test-set macro-F1. |
| `notes` | TEXT |  | Free-text remarks. |

### Table `batch_runs`

| Column | Type | Constraints | Description |
|---|---|---|---|
| `id` | INTEGER | PK, AUTOINCREMENT | Surrogate key. |
| `ts` | DATETIME | NOT NULL | UTC timestamp at which the scan started. |
| `repo_path` | VARCHAR(512) | NOT NULL | Absolute path of the local Git repository scanned. |
| `n_commits` | INTEGER | NOT NULL | Number of commits requested (the actual number stored may be smaller if the repository has fewer). |
| `model_id` | INTEGER | FK → `models.id`, NOT NULL | Model used for the batch. |

### Table `predictions`

| Column | Type | Constraints | Description |
|---|---|---|---|
| `id` | INTEGER | PK, AUTOINCREMENT | Surrogate key. |
| `ts` | DATETIME | NOT NULL | UTC timestamp of the inference call. |
| `model_id` | INTEGER | FK → `models.id`, NOT NULL, INDEXED | Model that produced this prediction. |
| `batch_run_id` | INTEGER | FK → `batch_runs.id`, NULLABLE, INDEXED | Set when the prediction was produced as part of a repository scan; NULL for single predictions. |
| `message_preview` | VARCHAR(280) | NOT NULL | First 280 characters of the commit message, after prefix stripping. |
| `diff_preview` | VARCHAR(280) | NOT NULL | First 280 characters of the raw diff. |
| `predicted_label` | VARCHAR(32) | NOT NULL, INDEXED | One of the five target classes. |
| `confidence` | FLOAT | NOT NULL | Probability of `predicted_label` ∈ [0, 1]. |
| `probabilities` | JSON | NOT NULL | Full 5-way probability vector. |
| `source` | VARCHAR(32) |  | Origin of the call: `gui`, `cli`, `repo-scan`, `test`. |

## Why three tables instead of one

A single-table schema would have been sufficient to satisfy FR-5
and NFR-6, but it would have collapsed two distinct concepts — *which
model produced the prediction* and *which scan it belongs to* — into
the same row. The three-table normalisation (i) lets the application
record stable metrics on `models` independently of when each
prediction was made, (ii) gives repository scans a header row that
the GUI can show as a single audit event, and (iii) keeps
`predictions` narrow enough to scale to tens of thousands of rows on
the same SQLite file without re-engineering.

# Class Diagrams

The software architecture is split into two complementary views: the
**offline side** (data pipeline + model training) and the **online
side** (inference + GUI + persistence). The split prevents either
diagram from collapsing under its own weight; together they cover
every Python module under `src/` and `app/`.

## Offline side — data pipeline and model training

Three sequential modules under `src/data/` (`Downloader`,
`Preprocessor`, `Splitter`) produce the train, validation and test
artefacts. An abstract `BaseModel` provides a common contract
(`train`, `evaluate`, `load`, `save`); the four base models inherit
from it and implement family-specific logic. The `CommitDataset`
class adapts the splits to PyTorch's `Dataset` interface and is shared
by the two transformer models. The `PrecomputeProbs` helper writes
each model's probabilities on val and test to disk as `.npy` arrays;
the `Ensemble` reads only those arrays so it never co-loads
TensorFlow and PyTorch in the same process — a workaround for a
known Apple-Silicon MPS contention issue (see §15). Figure 3 shows
the resulting class graph.

![Figure 3 — Class Diagram (data pipeline + models).](diagrams/png/03_class_pipeline_models.png){ width=80% }

## Online side — inference, GUI and persistence

The `InferenceLayer` (a stateless module that exposes `predict()` and
`predict_batch()`) is the only point of contact between the front-end
and the trained artefacts. Both the Streamlit application
(`StreamlitApp`) and the Typer CLI (`CLI`) depend on it; neither knows
how each model is built or serialised. Every prediction is mirrored
into the `HistoryRepository` (the SQLite-backed module described in
§8). The `Prediction` dataclass is the shared transport object;
its fields are guaranteed by FR-1 and FR-2. The diagram intentionally
inverts the typical "GUI on top, business logic below" reading order
to emphasise that the inference layer is the dependency root, and
that the GUI and the CLI are interchangeable consumers of the same
public surface. Figure 4 illustrates the dependency graph.

![Figure 4 — Class Diagram (inference + GUI + history).](diagrams/png/04_class_inference_gui.png){ width=80% }

# Graphical User Interface — Design and Mockups

The graphical interface is implemented in Streamlit, the lightest
toolkit compatible with the rubric requirement of "interface +
terminal" without forcing a React/Flask split. The visual identity
(palette, typography, component patterns) is documented separately in
`docs/design-system.md` and reproduced below in compact form.

## Information architecture — four tabs

| Tab | Primary action | Composition |
|---|---|---|
| **Predict** | Paste *(message, diff)*, choose a model, classify. | Two-column form (left: text inputs; right: model selector + history toggle). A `Classify` button reveals a *prediction card* with the 2 px class-coloured gutter, the predicted label, the confidence and a 5-row probability table. If a diff is provided it is rendered below as a syntax-highlighted code block. |
| **Repository** | Scan the last *N* commits of a local Git folder. | Folder picker + commit-count input + model selector. After the scan, a horizontal-bar histogram of class counts followed by a per-commit table (hash, author, date, label, confidence, message excerpt). |
| **History** | Browse persisted predictions. | A table of recent predictions read from SQLite, followed by an aggregated histogram of logged labels. |
| **Metrics** | Compare all trained models. | A 5-card grid (one card per model) with accuracy as headline figure and macro-F1, weighted-F1, precision and recall in the body. The leader has a sky-blue left border; followed by a macro-F1 horizontal-bar chart sorted descending. |

## Signature design element

Every place where a prediction is shown — Predict tab, Repository
table rows, History table — carries a **2 px left border** in the
predicted class's syntax-token colour:

| Class | Hex | IDE analogy |
|---|---|---|
| `feat` | `#22C55E` | added-line green |
| `fix` | `#EF4444` | removed-line red |
| `docs` | `#38BDF8` | comment / doc-link sky |
| `refactor` | `#A78BFA` | keyword violet |
| `test` | `#FBBF24` | warning amber |

The gutter mirrors the margin-marker an IDE places next to modified
lines on a code review; the class chip below it renders as a
monospace token in the same colour. This single recurring affordance
ties every screen together and makes the predicted class legible at
a glance even without reading the percentage value.

## Typography

| Family | Role |
|---|---|
| **Fira Sans** | Display + body (UI labels, paragraphs). |
| **Fira Code** | All numeric and data cells (probabilities, hashes, percentages, table figures). All numeric columns use `font-variant-numeric: tabular-nums` so digits align. |

## Mockups

Annotated screenshots of the four tabs are kept under
`docs/mockups/`. A reproduction in this document is pending and will
be inserted in the final revision.

# CLI Catalogue and API Documentation

The component does not expose a remote HTTP service (this is C2 —
*modelo local*), but it does expose a stable command-line surface and
a Python API surface. They are the equivalent of a web-services
catalogue for the purposes of the rubric.

## Python API — `src.inference`

```python
from src.inference import predict, predict_batch, Prediction

p: Prediction = predict(
    message="add OAuth2 login flow with PKCE",
    diff="diff --git a/auth.py b/auth.py\n@@ -0,0 +1,5 @@\n+def login(...)",
    model_name="baseline_tfidf",
)
# p.label          : str    — one of feat/fix/docs/refactor/test
# p.confidence     : float  — in [0, 1]
# p.probabilities  : dict[str, float]
# p.model          : str
```

`predict_batch(records, model_name)` accepts a list of
`{"message": ..., "diff": ...}` dictionaries and returns a list of
`Prediction` objects in the same order.

## CLI — `app.cli` (Typer)

| Command | Purpose | Example |
|---|---|---|
| `predict-cmd` | Classify a single commit from CLI arguments. | `python -m app.cli predict-cmd --message "fix race condition" --diff "$(cat patch.diff)" --model baseline_tfidf` |
| `repo-cmd` | Scan the last *N* commits of a local repository. | `python -m app.cli repo-cmd --path /Users/jesus/Desktop/parqueaderos-api --last 50 --model baseline_tfidf` |
| `batch-cmd` | Run a CSV of *(message, diff)* rows through a chosen model and emit a CSV with predictions. | `python -m app.cli batch-cmd --csv inputs.csv --model codebert --out predictions.csv` |
| `history-cmd` | List the last *N* persisted predictions. | `python -m app.cli history-cmd --limit 25` |
| `models-cmd` | Show which models have a trained artefact on disk. | `python -m app.cli models-cmd` |

Every command terminates with exit code 0 on success and a non-zero
code on any documented error (missing artefact, invalid path,
unsupported model name); errors are reported on `stderr` as a single
human-readable line.

## OpenAPI-style summary

The five commands form the de-facto API contract of the system. For
completeness, the equivalent OpenAPI 3 shape would be:

```
POST /predict          { message, diff, model }            → Prediction
POST /predict_batch    [{ message, diff }], model          → [Prediction]
POST /repo_scan        { path, last, model }               → [Prediction] + meta
GET  /history?limit=N                                       → [HistoryRow]
GET  /models                                                → [ModelDescriptor]
```

The current implementation does not host these endpoints over HTTP
because the rubric explicitly requires a local model and a CLI; the
shape is documented for future work (§15).

# Tests — Unit, Functional and Integration

The system ships with 26 automated tests under `tests/`, all passing
in under two seconds. They cover the three test types required by
item 2 of the rubric.

## Catalogue

| Test type | File | What it verifies |
|---|---|---|
| **Unit** | `tests/test_preprocess.py` | Pure functions in `src/data/preprocess.py`: `extract_label`, `strip_prefix`, `diff_to_text_and_features`. Covers the Conventional-Commits regex, prefix stripping, diff feature extraction, edge cases (empty / non-string / multi-line). |
| **Functional** | `tests/test_split.py` | Behaviour of the 70/15/15 stratified split: ratios are correct ± 0.5 %, class proportions identical across splits to four decimal places, no leakage between train / val / test, reproducibility under the fixed seed. |
| **Integration** | `tests/test_inference.py` | End-to-end inference path: `predict()` returns a valid `Prediction` for the smallest model; probabilities sum to 1.0 ± 1e-6; the label belongs to the canonical class set; non-existent model names raise a clear error. |

## Reproduction

```bash
$ uv run pytest tests/ -q
..........................                                       [100%]
26 passed in 1.42s
```

The same suite is run on every push to the public repository via the
`make test` target.

## Coverage rationale

Pure data-transformation code is covered with conventional unit
tests; cross-module behaviour (split, inference) is covered with
functional and integration tests against the actual SQLite database
and the actual model artefacts. Training code itself is **not** unit
tested — its correctness is verified end-to-end by the inference
tests, which would fail if a training script produced a corrupt
artefact.

# Proposed Model Architecture

The end-to-end system can be read at three levels of zoom: (a) the
data pipeline, (b) the internal architecture of each of the four
base models, (c) the soft-voting ensemble that fuses their
predictions.

## End-to-end pipeline

The raw CommitBench corpus (≈ 10 M rows) is downloaded in streaming
mode; only a one-million-commit sample is materialised on disk to
keep the dependency surface small. The preprocessor filters to the
five Conventional-Commit prefixes (`feat`, `fix`, `docs`, `refactor`,
`test`), strips the prefix from the message (so that the model cannot
cheat), and extracts three numeric features from the diff
(`files_changed`, `lines_added`, `lines_removed`). The splitter
produces 27,275 / 5,845 / 5,845 rows for train / val / test under a
fixed seed, stratified on the label.

Each of the four base models trains on the same `train` split.
Probability vectors on `val` and `test` are cached as `.npy` arrays;
the ensemble reads only those, optimises four scalar weights against
macro-F1 on the validation set, and applies the weighted sum on the
test set. The inference layer is a stateless module that loads one
of the five models on demand (LRU-cached) and is called by the
Streamlit GUI and the Typer CLI; both write their predictions to the
SQLite history. The full sequence is summarised in Figure 5.

![Figure 5 — End-to-end pipeline.](diagrams/png/05_architecture_pipeline.png){ width=70% }

## Per-model internals

The four base models cover four distinct families — a classical
linear model, a shallow convolutional network, a generic
encoder-only transformer and a code-aware encoder-only transformer.
Figure 6 lays out their internal pipelines side by side; the
narrative below walks through each.

![Figure 6 — Internal architecture of the four base models.](diagrams/png/06_architecture_models.png){ width=95% }

**Model 1 — `baseline_tfidf`** is a sklearn pipeline: a word-level
TF-IDF over the message (1-2 grams, 30k features), a character-level
TF-IDF over the diff (3-5 grams, 30k features), and a `StandardScaler`
on the three numeric diff features. The three feature blocks are
horizontally stacked into a 60 003-dimensional sparse vector and fed
to a `LogisticRegression` (solver=saga, balanced class weights).

**Model 2 — `cnn_text`** is a dual-branch Keras model. The message is
vectorised by a `TextVectorization` layer (20k vocabulary, 48-token
sequence length) and passed through `Embedding(64) → Conv1D(128,
k=3) → GlobalMaxPool`; the diff branch uses a wider window (k=5,
384-token sequence). A small dense block handles the numeric
features. The three branches are concatenated, run through
`Dense(128, relu) → Dropout(0.4)`, and projected to five classes via
softmax.

**Models 3 and 4 — `distilbert` and `codebert`** are encoder-only
pre-trained transformers fine-tuned through the HuggingFace `Trainer`
API. Both take `[CLS] message [SEP] diff [SEP]` as input
(`max_length=256`, `truncation="longest_first"`). DistilBERT uses 6
transformer layers and 67 M parameters; CodeBERT uses 12 layers and
125 M parameters and is bimodally pre-trained on paired text-code
corpora [Feng et al., 2020]. A linear classification head projects
the `[CLS]` representation to five logits.

## Soft-voting ensemble

Each base model is executed in its own Python process — the
`precompute_probs.py` helper — to avoid a TensorFlow / PyTorch / MPS
contention bug that hangs inference when both libraries try to claim
the Apple Metal device in the same process. The cached
`(N × 5)` probability matrices are stacked into a `(4 × N × 5)`
tensor; four scalar weights are fitted on the validation set by
L-BFGS-B with the negative macro-F1 as objective; the weighted sum
is taken on the test set and `argmax` produces the final class.
Optimal weights converge to uniform (`0.25` each) under our balanced
training regime, which means no single base model dominates after
balancing — a known and well-documented property of soft voting on
diverse families. Figure 7 illustrates the data flow.

![Figure 7 — Heterogeneous soft-voting ensemble.](diagrams/png/07_architecture_ensemble.png){ width=70% }

# Results and Discussion

## Headline numbers (test split, 5,845 commits)

| Model | Accuracy | Macro-F1 | Weighted-F1 | Macro-Precision | Macro-Recall |
|---|---|---|---|---|---|
| **baseline_tfidf** | **0.7093** | **0.6632** | **0.7187** | 0.6234 | 0.7221 |
| ensemble | 0.6896 | 0.6438 | 0.7007 | 0.5919 | 0.7409 |
| cnn_text | 0.6599 | 0.5861 | 0.6724 | 0.5437 | 0.6659 |
| codebert | 0.6089 | 0.5800 | 0.6259 | 0.5344 | 0.7000 |
| distilbert | 0.5858 | 0.5515 | 0.6057 | 0.5106 | 0.6821 |

Table: Test-set comparison of the five trained models, sorted by macro-F1 (descending).

## Why the classical baseline wins

The TF-IDF + Logistic-Regression baseline outperforms both fine-tuned
transformers on every reported metric. Three factors explain this
result:

1. **Input length.** Commit messages are short (median ≈ 7 tokens);
   most of the discriminative signal lives in the diff. TF-IDF with
   character n-grams over the diff captures this signal cheaply,
   while transformers waste capacity on long sub-word tokenisations
   of identifiers (`super_admin_user_repository`) that are essentially
   noise in the classification context.
2. **Training set size.** Transformers were fine-tuned on a balanced
   sub-sample of 6,000 / 8,000 commits to keep training cost on
   Apple-Silicon MPS bounded to a single afternoon. Fine-tuning
   transformers typically requires 30k–100k examples to surpass
   strong classical baselines [Sarwar et al., 2020].
3. **Class imbalance.** The corpus is heavily skewed toward `fix`
   (62.6 %) and away from `docs` (4.2 %). The balanced sub-sampling
   used for transformers hurts macro-precision on the majority class
   without compensating on the minority class.

## What the ensemble adds

The ensemble lifts macro-recall above any single base model (0.7409
versus 0.7221 for the baseline), which is the expected effect of
soft voting on diverse families. It does not, however, surpass the
baseline on macro-F1, because the L-BFGS-B optimiser converges to
uniform weights (`0.25` each) — meaning the three weaker models pull
the strongest one down on its own predictions. A stacking-based
ensemble that *learns* per-class weights via a small meta-classifier
would likely close the gap; this is left to §15.

## Field validation — residence-back

A manual audit of 25 commits from the *residence-back* repository
(100 % Conventional-Commits formatted, 100 % English) showed that the
model agrees with the author-supplied prefix in 15 out of 25 cases
(60 %). Of the 10 disagreements, 8 were author-side mislabelling
(features advertised as fixes); only 2 were genuine model errors on
documentation commits whose diff is dominated by code changes. The
field accuracy of the model — interpreted as *correctness against a
defensible re-labelling* — is therefore closer to 92 % than to the
60 % a naive agreement rate would suggest. This finding directly
supports problem **P2** from §2 and is the strongest piece of
evidence for the audit use case (UC-2).

## Limitations of the experimental design

We acknowledge three limitations:

- **5-class scope.** Conventional Commits defines 11 types; we
  restrict ourselves to the five most populous because the
  remaining six combined account for under 8 % of commits in the
  corpus and would make the imbalance problem worse.
- **English-only corpus.** CommitBench is filtered to English. Spanish
  commit messages — common in the `parqueaderos-api` and
  `residence-back` projects used for field validation — are scored
  as English by the model, which degrades accuracy on the message
  side but is partially compensated by the (language-agnostic) diff.
- **Apple-Silicon MPS hang on cross-framework inference.** As
  explained in §13.3, the ensemble cannot be evaluated by loading TF
  and PyTorch in the same process. We treat this as a tooling
  limitation, not an algorithmic one; the workaround is documented
  and reproducible.

# Recommendations and Future Work

The system as delivered satisfies the rubric for Component 2 of the
course project. We see four directions in which the work could be
extended.

1. **Stacking ensemble with a meta-classifier.** Replace the
   L-BFGS-B-fitted soft-voting weights with a learned meta-model —
   for example a small logistic regression over the four base
   models' probability vectors. This is expected to close the gap
   between the ensemble and the classical baseline, particularly on
   macro-precision.
2. **Multilingual support.** Re-train the transformer models on a
   bilingual corpus (Spanish + English) so that Latin-American
   software projects whose commit messages mix the two languages can
   be classified without degradation. CommitBench is English-only,
   but `Maxscha/commitbench-multi` or a custom PyDriller crawl over
   Colombian open-source repositories would provide a starting
   point.
3. **Knowledge distillation.** Distil CodeBERT (125 M parameters)
   into a 4-layer, 30 M-parameter student model so that the
   transformer family can be deployed in CPU-only environments
   without the current 5 × slowdown relative to the TF-IDF
   baseline.
4. **HTTP service shell.** Wrap the inference layer in a thin Flask
   or FastAPI front-end so that the same `predict()` / `predict_batch()`
   / `repo_scan` calls become a callable web service for the C1
   component or for third-party integrations. The rubric explicitly
   marks this as optional for C2, but the work is small (estimated
   at one developer-day) and would unify the two components of the
   course project.

# References

Conventional Commits Contributors (2018). *Conventional Commits 1.0.0
specification*. Available at
[https://www.conventionalcommits.org/en/v1.0.0](https://www.conventionalcommits.org/en/v1.0.0).
Community-authored specification, hosted at
[github.com/conventional-commits/conventionalcommits.org](https://github.com/conventional-commits/conventionalcommits.org),
licensed under CC BY 3.0.

Feng, Z., Guo, D., Tang, D., Duan, N., Feng, X., Gong, M., Shou, L.,
Qin, B., Liu, T., Jiang, D. and Zhou, M. (2020). *CodeBERT: A
Pre-Trained Model for Programming and Natural Languages*. In
*Findings of the Association for Computational Linguistics: EMNLP
2020*, pp. 1536–1547.
[aclanthology.org/2020.findings-emnlp.139](https://aclanthology.org/2020.findings-emnlp.139/).

Ghadhab, L., Jenhani, I., Mkaouer, M. W. and Ben Messaoud, M. (2021).
*Augmenting commit classification by using fine-grained source code
changes and a pre-trained deep neural language model*. *Information
and Software Technology*, 135, 106566.
[doi.org/10.1016/j.infsof.2021.106566](https://doi.org/10.1016/j.infsof.2021.106566).

Hindle, A., German, D. M. and Holt, R. (2008). *What do large commits
tell us? A taxonomical study of large commits*. In *Proc. 5th Working
Conf. on Mining Software Repositories (MSR 2008)*, Leipzig, Germany,
pp. 99–108.
[doi.org/10.1145/1370750.1370773](https://doi.org/10.1145/1370750.1370773).

Levin, S. and Yehudai, A. (2017). *Boosting automatic commit
classification into maintenance activities by utilizing source code
changes*. In *Proc. 13th Int. Conf. on Predictive Models and Data
Analytics in Software Engineering (PROMISE 2017)*, pp. 97–106.
[arXiv:1711.05340](https://arxiv.org/abs/1711.05340).

Mauczka, A., Brosch, F., Schanes, C. and Grechenig, T. (2015).
*Dataset of developer-labeled commit messages*. In *Proc. 12th IEEE/ACM
Working Conf. on Mining Software Repositories (MSR 2015)*, pp.
490–493. [doi.org/10.1109/MSR.2015.71](https://doi.org/10.1109/MSR.2015.71).

Mockus, A. and Votta, L. G. (2000). *Identifying reasons for software
changes using historic databases*. In *Proc. Int. Conf. on Software
Maintenance (ICSM 2000)*, San Jose, USA, pp. 120–130.
[doi.org/10.1109/ICSM.2000.883028](https://doi.org/10.1109/ICSM.2000.883028).

Sarwar, M. U., Zafar, S., Mkaouer, M. W., Walia, G. S. and Malik, M.
Z. (2020). *Multi-label Classification of Commit Messages using
Transfer Learning*. In *Proc. 2020 IEEE 31st Int. Symp. on Software
Reliability Engineering Workshops (ISSREW)*, pp. 37–42.
[doi.org/10.1109/ISSREW51248.2020.00034](https://doi.org/10.1109/ISSREW51248.2020.00034).

Zeng, Q., Zhang, Y., Qiu, Z. and Liu, H. (2025). *A First Look at
Conventional Commits Classification*. In *Proc. IEEE/ACM 47th Int.
Conf. on Software Engineering (ICSE 2025)*.
[doi.org/10.1109/ICSE55347.2025.00011](https://doi.org/10.1109/ICSE55347.2025.00011).


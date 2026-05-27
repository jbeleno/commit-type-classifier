# C2 — Commit Classifier + LLM Generator + Agentic Chat

> **Submission / Entrega**
>
> | File | Direct link |
> |---|---|
> | 📄 Documentation (PDF, 17 sections, 3.5 MB) | [`docs/exports/documentation.pdf`](docs/exports/documentation.pdf) |
> | 🎞️ Defense slides (PDF, 13 slides) | [`docs/exports/slides.pdf`](docs/exports/slides.pdf) |
> | 📊 Defense slides (PPTX editable) | [`docs/exports/slides.pptx`](docs/exports/slides.pptx) |
> | 🖼️ Architecture figures (9 UML diagrams, PNG) | [`docs/diagrams/png/`](docs/diagrams/png/) |
> | 📈 Model comparison — discriminative track (5 classifiers, n = 5 845) | [`models_saved/reports/comparison.md`](models_saved/reports/comparison.md) |
> | 🤖 Model comparison — generative track (5 LLMs × 4 strategies, n = 50) | [`models_saved/reports/llm/comparison.md`](models_saved/reports/llm/comparison.md) |
> | ⚖️ Apples-to-apples + voting-ensemble winner (n = 200) | [`models_saved/reports/llm_classify/comparison.md`](models_saved/reports/llm_classify/comparison.md) |
>
> **Authors:** Jesús Beleño · Juan Forero
> **Advisor:** Juan Antonio Castro Silva (USCO)
> **Date:** 2026-05-27

Component 2 (50%) of the Artificial Intelligence course project.
Three complementary tracks over Git commits, all using the same
preprocessed CommitBench corpus and the same Streamlit / CLI shell:

1. **Discriminative** — classify a commit's Conventional Commit type
   from message + diff (five models, including a heterogeneous
   soft-voting ensemble).
2. **Generative** — write the Conventional Commit message *itself*
   from the diff alone, using a local LLM (five Ollama-served models,
   four prompting strategies, plus a RAG + classifier-verifier hybrid).
3. **Agentic** — a conversational agent (Topic 11 of the syllabus)
   that picks among six tools (classify a commit, classify an entire
   repository, generate a message, scan a repo, list classes/models)
   to automate end-to-end commit analysis from a chat prompt. The
   default pipeline runs an LLM at every layer: orchestrator
   (`llama3.2:3b-instruct`), classifier (`llm:qwen2.5-coder:3b` with
   RAG few-shot), and generator (`qwen2.5-coder:3b` hybrid).

## Labels

| Label | Meaning |
|---|---|
| `feat` | New feature |
| `fix` | Bug fix |
| `docs` | Documentation only |
| `refactor` | Code restructure without behavior change |
| `test` | Tests added or modified |

## Pipeline

```
download → preprocess → split ─┬─► classification (5 models) ─► evaluate
                                │
                                ├─► generation (5 LLMs × 4 strategies)
                                │   └─► hybrid: RAG + LLM + verifier
                                │
                                └─► agent (llama3.2 orchestrator + 6 tools)
                                                       ▼
                              Streamlit GUI (6 tabs) · CLI · SQLite history
```

## Models — classification (discriminative)

1. **TF-IDF + Logistic Regression** — classical baseline (word 1-2grams on
   message + char 3-5grams on diff + numeric features, `class_weight="balanced"`).
2. **CNN-text dual** — two parallel Conv1D branches (message + diff) plus a
   dense numeric branch; early stopping on val loss.
3. **DistilBERT** fine-tuned via HuggingFace Trainer.
4. **CodeBERT** fine-tuned via HuggingFace Trainer.
5. **Soft-voting ensemble** — weights tuned on val with L-BFGS-B over macro-F1.

## Models — generation (LLMs, local via Ollama)

1. `qwen2.5-coder:1.5b` — small code-aware
2. `qwen2.5-coder:3b` — sweet-spot code-aware
3. `llama3.2:3b-instruct-q4_K_M` — Meta generalist
4. `phi3.5:3.8b-mini-instruct-q4_K_M` — Microsoft instruct
5. `deepseek-coder:1.3b` — code-specialist base

Four prompting strategies are compared per model: **zero-shot, few-shot,
chain-of-thought, JSON-mode**.

A **hybrid pipeline** combines them:

```
diff ─► TF-IDF KNN retrieval (top-3 train commits)
        │
        ▼
        LLM few-shot prompt ─► generated message
                                │
                                ▼
        TF-IDF baseline classifier ─► verifier type
                                │
                                ▼
              if verifier_conf ≥ τ AND llm_type ≠ verifier_type:
                        replace type with verifier
```

## Apples-to-apples — LLM-as-classifier beats the baseline

To answer "does the LLM track match the TF-IDF baseline?" we re-run
each LLM in **classifier mode** (same input as TF-IDF: message + diff;
same output space: one of five labels) on a stratified 200-commit
sample, then combine the top LLMs with the baseline as a co-equal
voter:

| System | Accuracy | Macro F1 | Weighted F1 |
|---|---:|---:|---:|
| TF-IDF baseline (n = 5,845) | 70.93 % | 0.6632 | 0.7187 |
| qwen2.5-coder:3b · rag (n = 200) | 74.00 % | 0.5639 | 0.7190 |
| **Voting ensemble (TF-IDF 2× boost)** | **75.00 %** | **0.6698** | **0.7505** |

The ensemble strictly beats the baseline on **every** test-set metric.
Full report in [models_saved/reports/llm_classify/comparison.md](models_saved/reports/llm_classify/comparison.md).

## Agentic AI — natural-language interface

The Chat tab exposes the project as an autonomous agent. The user can
say things like *"classify the last 30 commits of /path/to/repo"* and
the agent picks the right tool, runs it, and writes a one-paragraph
interpretation. **Every layer is an LLM** running locally on Ollama:

```
user prompt
   │
   ▼
[llama3.2:3b-instruct]      ← orchestrator: chooses which tool to call
   │
   ▼
┌─────────────────────────────────────────────┐
│ tools (src/llm/agent.py)                    │
│   classify_commit          ─► LLM classifier│
│   classify_repo            ─► scan + LLM    │
│   generate_commit_message  ─► hybrid LLM    │
│   scan_repo, list_models, list_classes      │
└─────────────────────────────────────────────┘
   │
   ▼
[llama3.2:3b-instruct]      ← 1-3 sentence interpretation
   │
   ▼
Streamlit chat (st.chat_message + per-tool renderer)
```

Full architecture in
[docs/diagrams/png/09_architecture_agent.png](docs/diagrams/png/09_architecture_agent.png).

## Course-rubric checklist

- [x] 70 / 15 / 15 stratified split, no data leakage
- [x] Class balancing (`class_weight="balanced"` + balanced sub-sampling for transformers)
- [x] Min. 3 models compared (**10 in total**: 5 classifiers + 5 LLMs)
- [x] Hyperparameter sweep (temperature × top-p × strategy × model for the LLM side)
- [x] Appropriate metrics per task (macro-F1, confusion matrix, BLEU-4, ROUGE-L, type-match, classifier-agreement, latency p50/p95)
- [x] Model serialization (`.joblib`, `.keras`, `.safetensors`, `.npy` probs, Ollama models on disk)
- [x] Separated training vs inference scripts
- [x] SQLite-backed history of predictions (3-table schema)

## Setup

```bash
uv sync --extra dev                       # creates .venv and installs deps
source .venv/bin/activate

# data
python -m src.data.download --sample 1000000
python -m src.data.preprocess
python -m src.data.split

# classifiers
python -m src.models.baseline_tfidf all
python -m src.models.cnn_text all
python -m src.models.distilbert_model all --max-train 5000
python -m src.models.codebert_model all  --max-train 5000
python -m src.models.precompute_probs --model baseline_tfidf
python -m src.models.precompute_probs --model cnn_text
python -m src.models.precompute_probs --model distilbert
python -m src.models.precompute_probs --model codebert
python -m src.models.ensemble all

# LLMs (requires Ollama on http://localhost:11434)
brew install ollama && ollama serve &
ollama pull qwen2.5-coder:3b
ollama pull qwen2.5-coder:1.5b
ollama pull llama3.2:3b-instruct-q4_K_M
ollama pull phi3.5:3.8b-mini-instruct-q4_K_M
ollama pull deepseek-coder:1.3b
python -m scripts.llm_sweep --n 50      # full sweep

# apps
streamlit run app/streamlit_app.py      # GUI (5 tabs)
python -m app.cli --help                # CLI: predict | generate | repo | batch | history
```

## Layout

```
data/{raw,processed,splits}      # dataset artifacts (gitignored)
src/data/                        # download, preprocess, stratified split
src/models/                      # 5 classifiers + ensemble + precompute_probs
src/llm/                         # ollama client, prompts, generator, RAG, hybrid
src/eval/                        # llm_eval harness
src/inference.py                 # unified discriminative inference
app/{streamlit_app,cli}          # user-facing entry points
models_saved/                    # serialized classifiers + probs + reports
models_saved/reports/llm/        # per-(model, strategy) LLM evaluations
db/history.sqlite                # prediction history (local)
docs/                            # English documentation, diagrams, slides
tests/                           # pytest suites (26 tests)
```

# C2 — Commit Classifier + LLM Generator

Component 2 (50%) of the Artificial Intelligence course project.
Two complementary tasks over Git commits, both using the same
preprocessed CommitBench corpus and the same Streamlit/CLI shell:

1. **Discriminative** — classify a commit's Conventional Commit type
   from message + diff (five models, including a heterogeneous
   soft-voting ensemble).
2. **Generative** — write the Conventional Commit message *itself*
   from the diff alone, using a local LLM (five Ollama-served models,
   four prompting strategies, plus a RAG + classifier-verifier hybrid).

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
                                └─► generation (5 LLMs × 4 strategies)
                                     └─► hybrid: RAG + LLM + verifier
                                                       ▼
                                       Streamlit GUI · CLI · SQLite history
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

# C2 — Commit Classifier

Component 2 (50%) of the Artificial Intelligence course project.
Classifies git commits into Conventional Commit types from the commit
message and code diff.

## Task

Multi-class classification over 5 labels:

| Label | Meaning |
|---|---|
| `feat` | New feature |
| `fix` | Bug fix |
| `docs` | Documentation only |
| `refactor` | Code restructure without behavior change |
| `test` | Tests added or modified |

## Pipeline

```
download.py  →  preprocess.py  →  split.py  →  train.py  →  evaluate.py
                                                    ↓
                                              inference.py
                                                    ↓
                                  Streamlit GUI  +  CLI
```

## Models compared (5)

1. **TF-IDF + Logistic Regression** — classical baseline
2. **CNN-text dual** — two parallel CNNs (message + diff)
3. **DistilBERT** fine-tuned — generic transformer
4. **CodeBERT** fine-tuned — code-aware transformer
5. **Ensemble** — heterogeneous voting/stacking

## Best-practice checklist (course rubric)

- [x] 70 / 15 / 15 stratified split (no data leakage)
- [ ] Class balancing (SMOTE / class weights)
- [ ] Min. 3 models compared (5 planned)
- [ ] Hyperparameter tuning with Hyperband
- [ ] Appropriate metrics per task (F1 macro, confusion matrix)
- [ ] Model serialization (`.h5`, `.pkl`, `.safetensors`)
- [ ] Separated training vs inference scripts
- [ ] SQLite-backed history of predictions

## Setup

```bash
uv sync                          # creates .venv and installs deps
source .venv/bin/activate

python -m src.data.download      # pulls CommitBench from HF
python -m src.data.preprocess    # filters + cleans
python -m src.data.split         # writes train/val/test CSVs
```

## Layout

```
data/{raw,processed,splits}      # dataset artifacts (gitignored)
src/{data,models}                # ML code
app/{streamlit_app,cli}          # user-facing entry points
models_saved/                    # serialized models
db/history.sqlite                # prediction history (local)
docs/                            # English documentation
tests/                           # pytest suites
```

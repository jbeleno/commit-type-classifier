# Diagrams Index

Seven figures supporting the C2 Commit Type Classifier documentation,
generated from PlantUML sources in `puml/` and rendered to `png/`.

To re-render after any change:

```bash
cd docs/diagrams
plantuml -tpng -o ../diagrams/png puml/*.puml
```

| Fig. | Source | Render | Purpose |
|---|---|---|---|
| 1 | [`puml/01_use_cases.puml`](puml/01_use_cases.puml) | [`png/01_use_cases.png`](png/01_use_cases.png) | UML use-case diagram — Developer interacting with the 7 use cases of the system, plus a secondary `CI Pipeline` actor for programmatic access. |
| 2 | [`puml/02_er_diagram.puml`](puml/02_er_diagram.puml) | [`png/02_er_diagram.png`](png/02_er_diagram.png) | Entity-relationship diagram of the SQLite persistence layer — `models` (1) — (N) `predictions` and (1) — (N) `batch_runs`. |
| 3 | [`puml/03_class_pipeline_models.puml`](puml/03_class_pipeline_models.puml) | [`png/03_class_pipeline_models.png`](png/03_class_pipeline_models.png) | UML class diagram of the offline side — data pipeline (`Downloader`, `Preprocessor`, `Splitter`) and the five model classes that inherit from `BaseModel`. |
| 4 | [`puml/04_class_inference_gui.puml`](puml/04_class_inference_gui.puml) | [`png/04_class_inference_gui.png`](png/04_class_inference_gui.png) | UML class diagram of the online side — `InferenceLayer`, `HistoryRepository`, `StreamlitApp` and `CLI`. |
| 5 | [`puml/05_architecture_pipeline.puml`](puml/05_architecture_pipeline.puml) | [`png/05_architecture_pipeline.png`](png/05_architecture_pipeline.png) | End-to-end architecture — CommitBench → preprocess → 70/15/15 split → 5 models → inference layer → GUI + CLI + SQLite. |
| 6 | [`puml/06_architecture_models.puml`](puml/06_architecture_models.puml) | [`png/06_architecture_models.png`](png/06_architecture_models.png) | Internal architecture of each of the four base models (TF-IDF+LR, CNN-text dual, DistilBERT, CodeBERT). |
| 7 | [`puml/07_architecture_ensemble.puml`](puml/07_architecture_ensemble.puml) | [`png/07_architecture_ensemble.png`](png/07_architecture_ensemble.png) | Heterogeneous soft-voting ensemble — stack of `.npy` probability arrays, L-BFGS-B weight optimization on val, weighted-sum + argmax at test time. |

## Style

All diagrams share the same dark, IDE-leaning palette defined in
`docs/design-system.md`:

| Token | Hex | Use |
|---|---|---|
| background | `#0F172A` | canvas |
| surface | `#1E293B` | boxes / cards |
| text | `#F8FAFC` | labels |
| text-muted | `#94A3B8` | annotations |
| arrow / focus | `#38BDF8` | edges, stereotypes |
| `feat` / `fix` / `docs` / `refactor` / `test` | `#22C55E` / `#EF4444` / `#38BDF8` / `#A78BFA` / `#FBBF24` | per-class colors when relevant |

Typefaces: **Fira Sans** (UI labels) + **Fira Code** (attribute lists).

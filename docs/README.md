# C2 — Documentation Index

Component 2 (50 %) of the *Artificial Intelligence* course project at
USCO. Project lead: Jesús Beleño. Advisor: Juan Antonio Castro Silva.

This folder holds **every artifact required by the course rubric**
(item 2, 40 % of the final grade). Code lives one level up.

```
docs/
├── README.md                  ← you are here
├── documentation.md           ← the 15-section main document  (English)
├── design-system.md           ← UI token spec (visual identity)
├── diagrams/
│   ├── README.md              ← caption table + render command
│   ├── puml/                  ← PlantUML source (versionable)
│   └── png/                   ← rendered figures
├── mockups/                   ← annotated Streamlit screenshots (pending)
└── slides/                    ← English presentation deck (pending)
```

## Mapping to the rubric (item 2, 40 %)

| Required section (line in `inteligencia_artificial.txt`) | Location |
|---|---|
| Introducción · Problema · Objetivos · Estado del Arte | `documentation.md` §1–§4 |
| Requerimientos · Casos de Uso / Historias de Usuario | `documentation.md` §5–§6 |
| Diagrama de Casos de Uso (UML) | Figure 1 — `diagrams/png/01_use_cases.png` |
| Diccionario de Datos + Modelo ER | `documentation.md` §8 + Figure 2 — `diagrams/png/02_er_diagram.png` |
| Diagramas de Clases | Figures 3–4 — `diagrams/png/03_class_pipeline_models.png` , `04_class_inference_gui.png` |
| Diseño GUI · Mockups | `mockups/` (Streamlit screenshots) |
| Catálogo Servicios Web · APIs | `documentation.md` §11 (CLI is the API for C2) |
| Pruebas (Unitarias · Funcionales · Integración) | `documentation.md` §12 (refers to `tests/`) |
| Gráfica arquitectura del modelo propuesto | Figures 5–7 — `diagrams/png/05_*` through `07_*` |
| Resultados y Discusión | `documentation.md` §14 |
| Recomendaciones y Trabajos Futuros | `documentation.md` §15 |

## Quick links

- Diagram captions & re-render command → [`diagrams/README.md`](diagrams/README.md)
- Visual identity (palette, tokens, components) → [`design-system.md`](design-system.md)
- Comparison of the 5 trained models → [`../models_saved/reports/comparison.md`](../models_saved/reports/comparison.md)
- Tests → run `uv run pytest tests/ -q` from the project root (26 tests).

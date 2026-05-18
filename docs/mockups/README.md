# Mockups

Annotated screenshots of the Streamlit GUI go here. Capture each of the
four tabs from a running app (default port `8501`):

```bash
cd /Users/jesus/Desktop/Inteligencia/c2-commit-classifier
uv run streamlit run app/streamlit_app.py
```

Expected files (PNG, 1440×900 recommended):

| File | Tab / state |
|---|---|
| `01_predict_empty.png`   | Predict tab, empty form |
| `02_predict_result.png`  | Predict tab, after classifying a sample commit (shows the 2 px class-color gutter, probability bars, diff viewer) |
| `03_repository_scan.png` | Repository tab with `/Users/jesus/Desktop/residencia/residence-back` + N=25 + baseline_tfidf, after scan (histogram + commits table) |
| `04_history_list.png`    | History tab, after a few predictions logged |
| `05_metrics_grid.png`    | Metrics tab, 5-card grid + macro-F1 horizontal bars |

Each screenshot should be annotated (callouts pointing to the gutter,
chip colors, monospace numerics, etc.) — use Skitch, CleanShot, or
Figma. Save the annotated version with the same name.

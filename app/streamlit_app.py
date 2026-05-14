"""Streamlit GUI — three tabs:
    1) Predict a single commit (paste message + diff).
    2) Analyze a local git repository.
    3) Browse the SQLite prediction history and model metrics.

Run:
    streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json

import pandas as pd
import plotly.express as px
import streamlit as st

from src import history
from src.config import MODELS_DIR, TARGET_CLASSES
from src.inference import AVAILABLE_MODELS, model_is_available, predict, predict_batch

st.set_page_config(page_title="Commit Classifier — C2", page_icon="🤖", layout="wide")


@st.cache_data(show_spinner=False)
def _trained_models() -> list[str]:
    return [m for m in AVAILABLE_MODELS if model_is_available(m)]


def _label_badge(label: str) -> str:
    palette = {
        "feat": "#4CAF50",
        "fix": "#F44336",
        "docs": "#2196F3",
        "refactor": "#FFC107",
        "test": "#9C27B0",
    }
    color = palette.get(label, "#777")
    return (
        f'<span style="background:{color};color:#fff;padding:4px 10px;'
        f'border-radius:6px;font-weight:600;">{label}</span>'
    )


def tab_predict() -> None:
    st.header("Single commit classification")
    trained = _trained_models()
    if not trained:
        st.error("No trained models found. Run `python -m src.models.<model> all` first.")
        return

    col_l, col_r = st.columns([2, 1])
    with col_r:
        model_name = st.selectbox("Model", trained, index=len(trained) - 1)
        log = st.checkbox("Log to history", value=True)

    with col_l:
        message = st.text_input(
            "Commit message",
            value="add support for OAuth2 login flow",
            placeholder="e.g. fix race condition in scheduler",
        )
        diff = st.text_area(
            "Diff (optional)",
            value="",
            height=200,
            placeholder="diff --git a/auth.py b/auth.py\n@@ -1,3 +1,8 @@\n+def login_oauth():\n+    pass",
        )

    if st.button("Classify", type="primary"):
        with st.spinner(f"Running {model_name}..."):
            p = predict(message, diff, model_name)
        st.markdown(f"### Prediction: {_label_badge(p.label)}", unsafe_allow_html=True)
        st.metric("Confidence", f"{p.confidence:.1%}")

        df_prob = pd.DataFrame(
            [{"label": k, "probability": v} for k, v in p.probabilities.items()]
        ).sort_values("probability", ascending=True)
        fig = px.bar(df_prob, x="probability", y="label", orientation="h",
                     range_x=[0, 1], color="probability", color_continuous_scale="Viridis")
        fig.update_layout(height=300, margin=dict(l=10, r=10, t=20, b=20))
        st.plotly_chart(fig, use_container_width=True)

        if log:
            history.log_prediction(message, diff, p, source="gui")
            st.toast("Logged to history.", icon="💾")


def tab_repo() -> None:
    st.header("Analyze a local git repository")
    trained = _trained_models()
    if not trained:
        st.error("Train at least one model first.")
        return
    col_l, col_r = st.columns([2, 1])
    with col_l:
        path = st.text_input("Repository path", value=str(Path.cwd()))
    with col_r:
        n = st.number_input("Commits to scan", min_value=10, max_value=2000, value=100, step=10)
        model_name = st.selectbox("Model", trained, index=len(trained) - 1, key="repo_model")

    if st.button("Scan repository", type="primary"):
        from pydriller import Repository

        repo_path = Path(path)
        if not (repo_path / ".git").exists():
            st.error(f"Not a git repo: {repo_path}")
            return

        records = []
        meta = []
        with st.spinner("Reading commits..."):
            iterator = Repository(str(repo_path), only_no_merge=True, order="reverse").traverse_commits()
            for i, commit in enumerate(iterator):
                if i >= n:
                    break
                diff_text = "\n".join(
                    f"diff --git a/{m.old_path or m.new_path} b/{m.new_path or m.old_path}\n{m.diff or ''}"
                    for m in commit.modified_files
                )
                records.append({"message": commit.msg, "diff": diff_text})
                meta.append(
                    {"hash": commit.hash[:10], "author": commit.author.name, "date": commit.author_date}
                )

        if not records:
            st.warning("No commits found.")
            return

        with st.spinner(f"Classifying {len(records)} commits..."):
            preds = predict_batch(records, model_name)

        out = pd.DataFrame(
            [
                {
                    "hash": m["hash"],
                    "author": m["author"],
                    "date": m["date"],
                    "label": p.label,
                    "confidence": p.confidence,
                    "message": r["message"].splitlines()[0][:120] if r["message"] else "",
                }
                for m, p, r in zip(meta, preds, records)
            ]
        )
        st.subheader("Class distribution")
        dist = out["label"].value_counts().reindex(TARGET_CLASSES, fill_value=0).reset_index()
        dist.columns = ["label", "count"]
        fig = px.bar(dist, x="label", y="count", color="label", text="count")
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Commits")
        st.dataframe(out, use_container_width=True, hide_index=True)


def tab_history() -> None:
    st.header("Recent predictions")
    rows = history.list_recent(limit=200)
    if rows:
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True, hide_index=True)
        dist = history.label_distribution()
        if dist:
            d = pd.DataFrame([{"label": k, "count": v} for k, v in dist.items()])
            fig = px.pie(d, names="label", values="count", title="Logged label distribution")
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No history yet.")


def tab_metrics() -> None:
    st.header("Model metrics (test split)")
    reports_dir = MODELS_DIR / "reports"
    if not reports_dir.exists():
        st.info("No reports yet. Train and evaluate models first.")
        return

    summary = []
    for fp in sorted(reports_dir.glob("*.json")):
        if fp.stem.endswith("_val"):
            continue
        data = json.loads(fp.read_text())
        summary.append(
            {
                "model": fp.stem,
                "accuracy": data["accuracy"],
                "macro_f1": data["macro_f1"],
                "weighted_f1": data["weighted_f1"],
                "macro_precision": data["macro_precision"],
                "macro_recall": data["macro_recall"],
            }
        )

    if not summary:
        st.info("No test reports yet.")
        return

    df = pd.DataFrame(summary).sort_values("macro_f1", ascending=False)
    st.dataframe(df, use_container_width=True, hide_index=True)

    fig = px.bar(df, x="model", y="macro_f1", color="model", text="macro_f1")
    fig.update_layout(yaxis_range=[0, 1])
    st.plotly_chart(fig, use_container_width=True)


def main() -> None:
    st.title("🤖 Commit Classifier — C2")
    st.caption("AI for Software Engineering — Conventional Commit type prediction")
    tabs = st.tabs(["🔮 Predict", "📂 Repository", "🗂 History", "📊 Metrics"])
    with tabs[0]:
        tab_predict()
    with tabs[1]:
        tab_repo()
    with tabs[2]:
        tab_history()
    with tabs[3]:
        tab_metrics()


if __name__ == "__main__":
    main()

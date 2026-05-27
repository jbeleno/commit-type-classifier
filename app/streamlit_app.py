"""Streamlit GUI for the Commit Type Classifier.

Four tabs (Predict, Repository, History, Metrics) rendered against the
design system at docs/design-system.md — dark, mono-leaning, IDE-token
class colors, 2px-gutter prediction card as the signature element.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import html
import json
import time
from typing import Dict

import pandas as pd
import streamlit as st

from src import history
from src.config import MODELS_DIR, TARGET_CLASSES
from src.inference import AVAILABLE_MODELS, Prediction, model_is_available, predict, predict_batch

st.set_page_config(
    page_title="Commit Type Classifier",
    layout="wide",
    initial_sidebar_state="collapsed",
)

CLASS_COLORS: Dict[str, str] = {
    "feat":     "#22C55E",
    "fix":      "#EF4444",
    "docs":     "#38BDF8",
    "refactor": "#A78BFA",
    "test":     "#FBBF24",
}


# ---------------------------------------------------------------------------
# Theme injection — sections 2.1–2.8 of docs/design-system.md
# ---------------------------------------------------------------------------

THEME_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Fira+Code:wght@400;500;600;700&family=Fira+Sans:wght@300;400;500;600;700&display=swap');

:root {
  --bg:             #0F172A;
  --surface-1:      #1E293B;
  --surface-2:      #334155;
  --surface-inset:  #0B1220;

  --border-soft:    rgba(148,163,184,0.10);
  --border:         rgba(148,163,184,0.18);
  --border-strong:  rgba(148,163,184,0.28);
  --ring-focus:     #38BDF8;

  --ink:            #F8FAFC;
  --ink-2:          #CBD5E1;
  --ink-3:          #94A3B8;
  --ink-mute:       #64748B;

  --c-feat:         #22C55E;
  --c-fix:          #EF4444;
  --c-docs:         #38BDF8;
  --c-refactor:     #A78BFA;
  --c-test:         #FBBF24;

  --s-1: 4px;  --s-2: 8px;  --s-3: 12px;
  --s-4: 16px; --s-5: 24px; --s-6: 32px;

  --r-1: 4px;  --r-2: 8px;  --r-3: 12px;

  --m-fast: 120ms cubic-bezier(0.2,0,0,1);
  --m-base: 200ms cubic-bezier(0.2,0,0,1);

  --font-sans: 'Fira Sans', system-ui, sans-serif;
  --font-mono: 'Fira Code', 'JetBrains Mono', ui-monospace, monospace;
}

/* ------- Global typography + Streamlit overrides ------- */
html, body, [class*="stApp"] {
  background: var(--bg) !important;
  color: var(--ink-2);
  font-family: var(--font-sans);
}
.stApp h1, .stApp h2, .stApp h3, .stApp h4 {
  font-family: var(--font-sans);
  color: var(--ink);
  letter-spacing: -0.01em;
}
.stMarkdown p, .stMarkdown li { color: var(--ink-2); }
code, pre, .mono, .num { font-family: var(--font-mono); font-variant-numeric: tabular-nums; }

/* App header strip */
.app-header {
  display: flex;
  align-items: baseline;
  justify-content: space-between;
  border-bottom: 1px solid var(--border-soft);
  padding-bottom: var(--s-3);
  margin-bottom: var(--s-4);
}
.app-header h1 {
  font-size: 1.5rem; font-weight: 600; margin: 0; letter-spacing: -0.01em;
}
.app-header .breadcrumb {
  font-family: var(--font-mono); color: var(--ink-3); font-size: 0.8125rem;
}
.app-header .breadcrumb b { color: var(--ink-2); font-weight: 500; }

/* ------- Tab strip — bottom-border activation, no fills ------- */
.stTabs [data-baseweb="tab-list"] {
  gap: 0;
  border-bottom: 1px solid var(--border-soft);
}
.stTabs [data-baseweb="tab"] {
  background: transparent !important;
  color: var(--ink-3) !important;
  font-family: var(--font-sans) !important;
  font-weight: 500;
  letter-spacing: 0.01em;
  padding: var(--s-3) var(--s-4);
  border-bottom: 2px solid transparent !important;
  border-radius: 0 !important;
  transition: color var(--m-fast), border-color var(--m-fast);
}
.stTabs [data-baseweb="tab"][aria-selected="true"] {
  color: var(--ink) !important;
  border-bottom-color: var(--ring-focus) !important;
}
.stTabs [data-baseweb="tab"]:hover { color: var(--ink-2) !important; }
.stTabs [data-baseweb="tab-highlight"] { display: none; }

/* ------- Inputs — inset surface ------- */
.stTextInput input,
.stTextArea textarea,
.stNumberInput input {
  background: var(--surface-inset) !important;
  border: 1px solid var(--border-soft) !important;
  border-radius: var(--r-1) !important;
  color: var(--ink) !important;
  font-family: var(--font-mono) !important;
  font-size: 0.875rem !important;
  transition: border-color var(--m-fast);
}
.stTextInput input:focus,
.stTextArea textarea:focus,
.stNumberInput input:focus {
  border-color: var(--ring-focus) !important;
  box-shadow: 0 0 0 2px rgba(56,189,248,0.20) !important;
}
.stTextInput label,
.stTextArea label,
.stNumberInput label,
.stSelectbox label,
.stCheckbox label {
  color: var(--ink-3) !important;
  font-family: var(--font-sans) !important;
  font-size: 0.8125rem !important;
  font-weight: 500;
  letter-spacing: 0.02em;
  text-transform: uppercase;
}

/* Selectbox */
.stSelectbox div[data-baseweb="select"] > div {
  background: var(--surface-inset) !important;
  border-color: var(--border-soft) !important;
  border-radius: var(--r-1) !important;
}

/* Primary button — flat, mono, no gradient */
.stButton button[kind="primary"],
.stButton button[data-testid="baseButton-primary"] {
  background: var(--ring-focus) !important;
  color: #0B1220 !important;
  border: 0 !important;
  border-radius: var(--r-1) !important;
  font-family: var(--font-mono) !important;
  font-weight: 600;
  letter-spacing: 0.02em;
  text-transform: lowercase;
  transition: background var(--m-fast);
}
.stButton button[kind="primary"]:hover { background: #7DD3FC !important; }

.stButton button[kind="secondary"] {
  background: var(--surface-1) !important;
  color: var(--ink-2) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--r-1) !important;
  font-family: var(--font-mono) !important;
  font-size: 0.8125rem !important;
  transition: background var(--m-fast), border-color var(--m-fast);
}
.stButton button[kind="secondary"]:hover {
  background: var(--surface-2) !important;
  border-color: var(--border-strong) !important;
}

/* DataFrame — borderless, mono numbers */
.stDataFrame { border: 1px solid var(--border-soft); border-radius: var(--r-2); overflow: hidden; }
.stDataFrame [data-testid="stDataFrameResizable"] { background: var(--surface-1) !important; }
.stDataFrame th { background: var(--surface-1) !important; color: var(--ink-3) !important;
                  font-family: var(--font-sans) !important; text-transform: uppercase;
                  font-size: 0.75rem !important; letter-spacing: 0.04em; font-weight: 500; }
.stDataFrame td { background: var(--surface-1) !important; color: var(--ink-2) !important;
                  font-family: var(--font-mono) !important; font-size: 0.8125rem !important;
                  border-color: var(--border-soft) !important; }

/* ------- The signature: prediction card with 2px class-color gutter ------- */
.predict-card {
  background: var(--surface-1);
  border: 1px solid var(--border);
  border-left: 3px solid var(--class-color, var(--border-strong));
  border-radius: var(--r-2);
  padding: var(--s-5);
  margin: var(--s-3) 0 var(--s-4) 0;
}
.predict-card .pc-head {
  display: flex; align-items: baseline; justify-content: space-between;
  gap: var(--s-4);
  margin-bottom: var(--s-3);
}
.predict-card .pc-label {
  font-family: var(--font-mono);
  font-size: 1.125rem; font-weight: 600;
  color: var(--class-color, var(--ink));
  letter-spacing: 0.02em;
}
.predict-card .pc-conf {
  font-family: var(--font-mono);
  font-size: 1.5rem; font-weight: 600;
  color: var(--ink);
  font-variant-numeric: tabular-nums;
}
.predict-card .pc-meta {
  font-family: var(--font-mono);
  font-size: 0.75rem;
  color: var(--ink-3);
  border-top: 1px solid var(--border-soft);
  padding-top: var(--s-3);
  margin-bottom: var(--s-4);
}
.predict-card .pc-meta b { color: var(--ink-2); font-weight: 500; }

/* ------- IDE-token chips (one per class) ------- */
.chip {
  display: inline-block;
  font-family: var(--font-mono);
  font-size: 0.75rem;
  font-weight: 500;
  padding: 2px 8px;
  border-radius: var(--r-1);
  letter-spacing: 0.02em;
}
.chip-feat     { color: var(--c-feat);     background: rgba(34,197,94,0.10);   border: 1px solid rgba(34,197,94,0.30); }
.chip-fix      { color: var(--c-fix);      background: rgba(239,68,68,0.10);   border: 1px solid rgba(239,68,68,0.30); }
.chip-docs     { color: var(--c-docs);     background: rgba(56,189,248,0.10);  border: 1px solid rgba(56,189,248,0.30); }
.chip-refactor { color: var(--c-refactor); background: rgba(167,139,250,0.10); border: 1px solid rgba(167,139,250,0.30); }
.chip-test     { color: var(--c-test);     background: rgba(251,191,36,0.10);  border: 1px solid rgba(251,191,36,0.30); }

/* ------- Probability bars / histogram rows ------- */
.bar-row {
  display: grid;
  grid-template-columns: 6rem 1fr 5ch;
  align-items: center;
  gap: var(--s-3);
  padding: 2px 0;
  font-family: var(--font-mono);
  font-size: 0.8125rem;
}
.bar-row .bar-label { color: var(--ink-2); }
.bar-row .bar-value { text-align: right; color: var(--ink); font-variant-numeric: tabular-nums; }
.bar-track {
  height: 8px;
  background: var(--surface-inset);
  border-radius: 2px;
  overflow: hidden;
  border: 1px solid var(--border-soft);
}
.bar-fill {
  height: 100%;
  background: var(--class-color, var(--ring-focus));
  transition: width var(--m-base);
}

/* ------- Code block / diff viewer ------- */
.code {
  background: var(--surface-inset);
  border: 1px solid var(--border-soft);
  border-radius: var(--r-2);
  padding: var(--s-3) var(--s-4);
  font-family: var(--font-mono);
  font-size: 0.8125rem;
  line-height: 1.55;
  color: var(--ink-2);
  white-space: pre;
  overflow-x: auto;
}
.code .l-add { color: #4ADE80; }
.code .l-del { color: #FCA5A5; }
.code .l-ctx { color: var(--ink-3); }

/* ------- Metric card grid (Metrics tab) ------- */
.metric-card {
  background: var(--surface-1);
  border: 1px solid var(--border);
  border-left: 3px solid var(--surface-2);
  border-radius: var(--r-2);
  padding: var(--s-4);
  height: 100%;
}
.metric-card.is-leader { border-left-color: var(--ring-focus); }
.metric-card .mc-model {
  font-family: var(--font-mono); font-size: 0.8125rem;
  color: var(--ink-2); margin-bottom: var(--s-2);
  letter-spacing: 0.02em;
}
.metric-card .mc-headline {
  font-family: var(--font-mono);
  font-size: 1.75rem; font-weight: 600;
  color: var(--ink);
  font-variant-numeric: tabular-nums;
  letter-spacing: -0.01em;
  margin-bottom: var(--s-3);
}
.metric-card .mc-row {
  display: grid; grid-template-columns: 1fr auto;
  font-family: var(--font-mono); font-size: 0.75rem;
  border-top: 1px solid var(--border-soft);
  padding: var(--s-1) 0;
  color: var(--ink-3);
}
.metric-card .mc-row b { color: var(--ink-2); font-variant-numeric: tabular-nums; font-weight: 500; }

/* ------- Inline meta line (commit context) ------- */
.meta-strip {
  font-family: var(--font-mono);
  font-size: 0.75rem;
  color: var(--ink-3);
  padding: var(--s-2) 0;
}
.meta-strip .k { color: var(--ink-mute); }
.meta-strip .v { color: var(--ink-2); }

/* Empty / loading / error states */
.state-empty { color: var(--ink-3); font-family: var(--font-mono); font-size: 0.8125rem;
               padding: var(--s-4); border: 1px dashed var(--border-soft); border-radius: var(--r-2); }
.state-error { color: var(--c-fix); background: rgba(239,68,68,0.08); font-family: var(--font-mono);
               font-size: 0.8125rem; padding: var(--s-3) var(--s-4);
               border: 1px solid rgba(239,68,68,0.25); border-radius: var(--r-1); }

/* Section eyebrow */
.eyebrow { font-family: var(--font-sans); color: var(--ink-3); font-size: 0.75rem;
           font-weight: 500; letter-spacing: 0.06em; text-transform: uppercase;
           margin: var(--s-5) 0 var(--s-3) 0; }

/* Hide Streamlit chrome we don't want */
[data-testid="stToolbar"] { display: none; }
footer { display: none; }
header[data-testid="stHeader"] { background: transparent; }
</style>
"""

st.markdown(THEME_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Reusable HTML component builders
# ---------------------------------------------------------------------------


def _e(s: str) -> str:
    """HTML-escape for any user-supplied string before embedding in markup."""
    return html.escape(s, quote=True)


def chip(label: str) -> str:
    cls = f"chip-{label}" if label in CLASS_COLORS else "chip-docs"
    return f'<span class="chip {cls}">{_e(label)}</span>'


def bar_row(label: str, value: float, value_fmt: str = "{:.3f}", show_chip: bool = False) -> str:
    color = CLASS_COLORS.get(label, "var(--ring-focus)")
    pct = max(0.0, min(1.0, value)) * 100
    label_html = chip(label) if show_chip else f'<span class="bar-label">{_e(label)}</span>'
    return (
        f'<div class="bar-row" style="--class-color:{color};">'
        f'  {label_html}'
        f'  <div class="bar-track"><div class="bar-fill" style="width:{pct:.2f}%"></div></div>'
        f'  <span class="bar-value">{value_fmt.format(value)}</span>'
        f'</div>'
    )


def histogram_row(label: str, count: int, max_count: int) -> str:
    color = CLASS_COLORS.get(label, "var(--ring-focus)")
    pct = (count / max_count * 100) if max_count else 0
    return (
        f'<div class="bar-row" style="--class-color:{color};">'
        f'  {chip(label)}'
        f'  <div class="bar-track"><div class="bar-fill" style="width:{pct:.2f}%"></div></div>'
        f'  <span class="bar-value">{count}</span>'
        f'</div>'
    )


def predict_card(p: Prediction, latency_ms: float) -> str:
    color = CLASS_COLORS.get(p.label, "var(--ring-focus)")
    sorted_items = sorted(p.probabilities.items(), key=lambda kv: -kv[1])
    bars = "\n".join(bar_row(lbl, val) for lbl, val in sorted_items)
    return f"""
    <div class="predict-card" style="--class-color:{color};">
      <div class="pc-head">
        <span class="pc-label">{_e(p.label)}</span>
        <span class="pc-conf">{p.confidence * 100:.2f}<span style="color:var(--ink-3);font-size:1rem;"> %</span></span>
      </div>
      <div class="pc-meta">
        <span class="k" style="color:var(--ink-mute);">model</span> <b>{_e(p.model)}</b>
        &nbsp;·&nbsp;
        <span class="k" style="color:var(--ink-mute);">latency</span> <b>{latency_ms:.0f} ms</b>
      </div>
      {bars}
    </div>
    """


def render_diff_block(diff: str, max_lines: int = 24) -> str:
    if not diff.strip():
        return ""
    lines = []
    for line in diff.splitlines()[:max_lines]:
        if line.startswith("+++") or line.startswith("---") or line.startswith("diff "):
            cls = "l-ctx"
        elif line.startswith("+"):
            cls = "l-add"
        elif line.startswith("-"):
            cls = "l-del"
        else:
            cls = "l-ctx"
        lines.append(f'<span class="{cls}">{_e(line) or "&nbsp;"}</span>')
    body = "<br>".join(lines)
    return f'<pre class="code">{body}</pre>'


def metric_card(model: str, accuracy: float, metrics: Dict[str, float], is_leader: bool) -> str:
    cls = "metric-card is-leader" if is_leader else "metric-card"
    rows = "".join(
        f'<div class="mc-row"><span>{_e(k)}</span><b>{v:.4f}</b></div>'
        for k, v in metrics.items()
    )
    return f"""
    <div class="{cls}">
      <div class="mc-model">{_e(model)}</div>
      <div class="mc-headline">{accuracy * 100:.2f}<span style="color:var(--ink-3); font-size:1rem;"> %</span></div>
      {rows}
    </div>
    """


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------


@st.cache_data(show_spinner=False)
def _trained_models() -> list[str]:
    return [m for m in AVAILABLE_MODELS if model_is_available(m)]


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------


def _render_classify_repo(result: Dict) -> None:
    if "error" in result:
        st.error(result["error"])
        return
    hist = result.get("histogram", {})
    per = result.get("per_commit", []) or []
    total = sum(hist.values()) or 1
    dominant = max(hist.items(), key=lambda kv: kv[1])[0] if hist else "—"
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("commits scanned", result.get("n_commits", len(per)))
    c2.metric("dominant class", dominant, f"{hist.get(dominant, 0) / total:.0%}")
    c3.metric("model", result.get("model", "—"))
    c4.metric("classes seen", sum(1 for v in hist.values() if v > 0))

    hist_df = pd.DataFrame(
        {"class": list(hist.keys()), "count": list(hist.values())}
    ).set_index("class")
    st.bar_chart(hist_df, height=160, use_container_width=True)

    if per:
        df = pd.DataFrame(per)
        df["confidence"] = df["confidence"].astype(float).round(3)
        df = df.rename(columns={"predicted_type": "type", "message": "subject"})
        st.dataframe(
            df[["hash", "type", "confidence", "subject"]],
            use_container_width=True,
            hide_index=True,
            height=min(420, 40 + 35 * len(df)),
        )


def _render_classify_commit(result: Dict) -> None:
    if "error" in result:
        st.error(result["error"])
        return
    label = result.get("label", "—")
    conf = float(result.get("confidence", 0))
    color = CLASS_COLORS.get(label, "#94A3B8")
    c1, c2 = st.columns([1, 2])
    c1.markdown(
        f'<div class="mono" style="font-size:2.2rem; color:{color}; font-weight:700;">'
        f'{_e(label)}</div>'
        f'<div class="muted">confidence {conf:.1%}  ·  '
        f'{_e(result.get("model", ""))}</div>',
        unsafe_allow_html=True,
    )
    probs = result.get("probabilities", {})
    if probs:
        prob_df = pd.DataFrame(
            {"class": list(probs.keys()), "p": [float(v) for v in probs.values()]}
        ).set_index("class")
        c2.bar_chart(prob_df, height=180, use_container_width=True)


def _render_generate_commit(result: Dict) -> None:
    if "error" in result:
        st.error(result["error"])
        return
    msg = result.get("message", "")
    t = result.get("type", "—")
    color = CLASS_COLORS.get(t, "#94A3B8")
    st.markdown(
        f'<div class="predict-card" style="border-color:{color}">'
        f'<div class="mono" style="font-size:1.3rem; color:{color}; font-weight:700;">'
        f'{_e(msg)}</div>'
        f'<div class="muted" style="font-size:0.85rem; margin-top:0.4rem;">'
        f'llm_type={_e(result.get("llm_type") or "—")} · '
        f'verifier_type={_e(result.get("verifier_type", "—"))} '
        f'({float(result.get("verifier_confidence", 0)):.0%}) · '
        f'type_changed={"yes" if result.get("type_changed") else "no"} · '
        f'{float(result.get("latency_ms", 0)):.0f} ms'
        f'</div></div>', unsafe_allow_html=True,
    )
    examples = result.get("retrieved_examples") or []
    if examples:
        with st.expander(f"retrieved {len(examples)} similar commits (RAG)", expanded=False):
            for ex in examples:
                st.markdown(
                    f"- `[{float(ex.get('score', 0)):.3f}]` "
                    f"**{ex.get('type', '')}** · {ex.get('subject', '')[:140]}"
                )


def _render_scan_repo(result: Dict) -> None:
    if "error" in result:
        st.error(result["error"])
        return
    commits = result.get("commits", []) or []
    st.caption(f"{result.get('n_commits', len(commits))} commits  ·  {result.get('path', '')}")
    if not commits:
        return
    df = pd.DataFrame(commits)
    keep = [c for c in ("hash", "author", "date", "message") if c in df.columns]
    df = df[keep]
    st.dataframe(df, use_container_width=True, hide_index=True,
                 height=min(420, 40 + 35 * len(df)))


def _render_list(result: Dict) -> None:
    for k, v in result.items():
        if isinstance(v, list):
            st.markdown(f"**{k}**")
            for item in v:
                st.markdown(f"- `{item}`")
        else:
            st.markdown(f"**{k}:** `{v}`")


_TOOL_RENDERERS = {
    "classify_repo": _render_classify_repo,
    "classify_commit": _render_classify_commit,
    "generate_commit_message": _render_generate_commit,
    "scan_repo": _render_scan_repo,
    "list_models": _render_list,
    "list_classes": _render_list,
}

_TOOL_AVATAR = {
    "classify_commit":          "🏷️",
    "classify_repo":            "📊",
    "generate_commit_message":  "✍️",
    "scan_repo":                "🔍",
    "list_models":              "📦",
    "list_classes":             "🎯",
}


def tab_chat() -> None:
    """Agentic chat — conversational interface backed by tool-calling LLM."""
    from src.llm.agent import DEFAULT_CLASSIFIER, DEFAULT_GENERATOR, TOOLS, run_agent

    models = _ollama_models()
    if not models:
        st.markdown(
            '<div class="state-error">Ollama daemon not reachable. '
            'Start it with <code>ollama serve</code> and pull at least one model.</div>',
            unsafe_allow_html=True,
        )
        return

    st.markdown(
        '<div class="eyebrow">agentic chat · LLM end-to-end · '
        + str(len(TOOLS)) + ' tools</div>',
        unsafe_allow_html=True,
    )

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "chat_steps" not in st.session_state:
        st.session_state.chat_steps = []

    col_l, col_r = st.columns([3.2, 1])
    with col_r:
        preferred = ["llama3.2:3b-instruct-q4_K_M", "qwen2.5-coder:3b"]
        default_idx = next((models.index(p) for p in preferred if p in models), 0)
        agent_model = st.selectbox(
            "orchestrator LLM", models, index=default_idx,
            help="LLM that decides which tools to call and writes the natural-language interpretation.",
        )

        st.markdown(
            f'<div class="muted" style="font-size:0.78rem; line-height:1.55; '
            f'background:#0B1220; border:1px solid #334155; padding:0.55rem 0.7rem; '
            f'border-radius:6px; margin-top:0.5rem;">'
            f'<b style="color:#38BDF8;">model stack</b><br>'
            f'· orchestrator → <span class="mono">{_e(agent_model)}</span><br>'
            f'· classifier → <span class="mono">{_e(DEFAULT_CLASSIFIER)}</span><br>'
            f'· generator → <span class="mono">{_e(DEFAULT_GENERATOR)}</span><br>'
            f'<span style="color:#94A3B8;">all three are LLMs running locally via Ollama.</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

        if st.button("clear conversation", use_container_width=True):
            st.session_state.chat_history = []
            st.session_state.chat_steps = []
            st.rerun()

        st.markdown('<div class="eyebrow" style="margin-top:1rem;">try saying</div>',
                    unsafe_allow_html=True)
        st.markdown(
            '<div class="muted" style="font-size:0.82rem; line-height:1.6;">'
            '· list the available models<br>'
            '· classify: fix race condition<br>'
            '· analyze last 20 commits of /path/to/repo<br>'
            '· write a commit for this diff: …<br>'
            '<br>'
            '<i>(LLM classifier is slower: ~2 s per commit; use baseline_tfidf when you need speed)</i>'
            '</div>', unsafe_allow_html=True)

    with col_l:
        for step in st.session_state.chat_steps:
            if step["role"] == "user":
                with st.chat_message("user"):
                    st.markdown(step["content"])
            elif step["role"] == "tool":
                tool_name = step.get("tool_name", "")
                avatar = _TOOL_AVATAR.get(tool_name, "🔧")
                with st.chat_message("assistant", avatar=avatar):
                    st.caption(f"tool · `{tool_name}`")
                    result = step.get("tool_result") or {}
                    renderer = _TOOL_RENDERERS.get(tool_name)
                    if renderer:
                        renderer(result)
                    else:
                        st.json(result, expanded=False)
                    with st.expander("call details", expanded=False):
                        st.code(json.dumps(step.get("tool_args") or {}, indent=2, default=str),
                                language="json")
            else:
                with st.chat_message("assistant"):
                    st.markdown(step["content"])
                    latency = step.get("latency_ms", 0) or 0
                    if latency:
                        st.caption(f"{agent_model} · {latency:.0f} ms")

        user_input = st.chat_input("ask the agent — e.g. 'classify the last 50 commits in /path/to/repo'")
        if user_input:
            with st.spinner(f"thinking with {agent_model}..."):
                steps, new_history = run_agent(
                    user_input,
                    history=st.session_state.chat_history,
                    model=agent_model,
                )
            for s in steps:
                st.session_state.chat_steps.append({
                    "role": s.role,
                    "content": s.content,
                    "tool_name": s.tool_name,
                    "tool_args": s.tool_args,
                    "tool_result": s.tool_result,
                    "latency_ms": s.latency_ms,
                })
            st.session_state.chat_history = new_history
            st.rerun()


def _ollama_models() -> list[str]:
    try:
        from src.llm import ollama_client

        if not ollama_client.is_alive():
            return []
        return ollama_client.list_models()
    except Exception:
        return []


def tab_generate() -> None:
    """LLM-based commit-message generation from a diff (hybrid: RAG + LLM + verifier)."""
    models = _ollama_models()
    if not models:
        st.markdown(
            '<div class="state-error">Ollama daemon not reachable. '
            'Start it with <code>ollama serve</code> and pull a model with '
            '<code>ollama pull qwen2.5-coder:3b</code>.</div>',
            unsafe_allow_html=True,
        )
        return

    st.markdown('<div class="eyebrow">generate conventional commit from diff (local llm)</div>', unsafe_allow_html=True)

    col_l, col_r = st.columns([2.2, 1])
    with col_r:
        preferred = ["qwen2.5-coder:3b", "llama3.2:3b-instruct-q4_K_M", "qwen2.5-coder:1.5b"]
        default_idx = next((models.index(p) for p in preferred if p in models), 0)
        model_name = st.selectbox("ollama model", models, index=default_idx)
        mode = st.selectbox(
            "mode",
            ["hybrid (rag + llm + verifier)", "zero_shot", "few_shot", "chain_of_thought", "json_mode"],
            index=0,
        )
        temperature = st.slider("temperature", 0.0, 1.0, 0.2, 0.05)
        show_retrieved = st.checkbox("show retrieved similar commits", value=True)

    with col_l:
        diff = st.text_area(
            "diff",
            value="",
            height=260,
            placeholder=(
                "diff --git a/auth.py b/auth.py\n"
                "@@ -10,3 +10,7 @@\n"
                "-def login(user, pwd):\n"
                "-    return check(user, pwd)\n"
                "+def login(user, pwd):\n"
                "+    if not user or not pwd:\n"
                "+        raise ValueError('missing creds')\n"
                "+    return check(user, pwd)\n"
            ),
        )

    if st.button("generate", type="primary"):
        if not diff.strip():
            st.markdown('<div class="state-error">Paste a diff first.</div>', unsafe_allow_html=True)
            return

        with st.spinner(f"running {model_name} ({mode})..."):
            if mode.startswith("hybrid"):
                from src.llm.hybrid import hybrid_generate

                r = hybrid_generate(diff, model=model_name, temperature=temperature)
                color = CLASS_COLORS.get(r.final_type, "#94A3B8")
                st.markdown(
                    f'<div class="predict-card" style="border-color:{color}">'
                    f'  <div class="mono" style="font-size:1.4rem; color:{color}; margin-bottom:8px;">{_e(r.final_message)}</div>'
                    f'  <div class="muted" style="font-size:0.85rem;">'
                    f'    llm_type={_e(r.llm_type or "—")} · verifier_type={_e(r.verifier_type)} '
                    f'({r.verifier_confidence:.2%}) · '
                    f'type_changed={"yes" if r.type_changed else "no"}'
                    f'  </div>'
                    f'  <div class="muted" style="font-size:0.85rem;">'
                    f'    latency total={r.latency_ms_total:.0f} ms · llm={r.latency_ms_llm:.0f} ms · model={_e(r.model)}'
                    f'  </div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
                if show_retrieved and r.retrieved:
                    st.markdown('<div class="eyebrow">retrieved similar commits (RAG)</div>', unsafe_allow_html=True)
                    for ex in r.retrieved:
                        st.markdown(
                            f'<div class="state-info">'
                            f'  <span class="mono">[score={ex.score:.3f}]</span> '
                            f'  {chip(ex.type)} <span class="mono">{_e(ex.subject[:120])}</span>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )
            else:
                from src.llm.generator import generate_commit_message

                gc = generate_commit_message(
                    diff, model=model_name, strategy=mode, temperature=temperature
                )
                color = CLASS_COLORS.get(gc.parsed_type, "#94A3B8") if gc.parsed_type else "#94A3B8"
                st.markdown(
                    f'<div class="predict-card" style="border-color:{color}">'
                    f'  <div class="mono" style="font-size:1.4rem; color:{color}; margin-bottom:8px;">{_e(gc.one_liner)}</div>'
                    f'  <div class="muted" style="font-size:0.85rem;">'
                    f'    parsed_type={_e(gc.parsed_type or "—")} · strategy={_e(gc.strategy)} · '
                    f'    latency={gc.latency_ms:.0f} ms · tokens={gc.completion_tokens}'
                    f'  </div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

        st.markdown('<div class="eyebrow">diff</div>', unsafe_allow_html=True)
        st.markdown(render_diff_block(diff), unsafe_allow_html=True)


MODEL_LABELS = {
    "baseline_tfidf":         "baseline_tfidf  —  TF-IDF + LogReg  ·  ~50 ms",
    "cnn_text":               "cnn_text  —  dual Conv1D  ·  ~200 ms",
    "distilbert":             "distilbert  —  transformer fine-tuned  ·  ~500 ms",
    "codebert":               "codebert  —  code-aware transformer  ·  ~500 ms",
    "ensemble":               "ensemble  —  4-classifier soft vote  ·  ~1 s",
    "llm:qwen2.5-coder:3b":   "llm:qwen2.5-coder:3b  —  LLM single  ·  demo · ~2 s",
    "llm-ensemble":           "llm-ensemble  —  qwen-3b + phi3.5 + TF-IDF voter (2× boost)  ·  demo · ~4 s · beats baseline on all 3 metrics",
}


def tab_predict() -> None:
    trained = _trained_models()
    if not trained:
        st.markdown(
            '<div class="state-error">No trained models found. Run <code>python -m src.models.&lt;model&gt; all</code> first.</div>',
            unsafe_allow_html=True,
        )
        return

    st.markdown('<div class="eyebrow">single commit classification</div>', unsafe_allow_html=True)

    col_l, col_r = st.columns([2.2, 1])
    with col_r:
        model_name = st.selectbox(
            "model",
            trained,
            index=0,
            format_func=lambda m: MODEL_LABELS.get(m, m),
            help="Defaults to the fastest classifier. LLM modes are slower and intended for benchmark/defense, not production use.",
        )
        log = st.checkbox("log to history", value=True)

    with col_l:
        message = st.text_input(
            "commit message",
            value="add support for OAuth2 login flow with PKCE",
            placeholder="fix race condition in scheduler",
        )
        diff = st.text_area(
            "diff  (optional)",
            value="",
            height=180,
            placeholder="diff --git a/auth.py b/auth.py\n@@ -0,0 +1,5 @@\n+def login_oauth(client_id):\n+    code_verifier = secrets.token_urlsafe(48)",
        )

    if st.button("classify", type="primary"):
        t0 = time.perf_counter()
        with st.spinner(f"running {model_name}..."):
            p = predict(message, diff, model_name)
        latency_ms = (time.perf_counter() - t0) * 1000

        st.markdown(predict_card(p, latency_ms), unsafe_allow_html=True)

        if diff.strip():
            st.markdown('<div class="eyebrow">diff</div>', unsafe_allow_html=True)
            st.markdown(render_diff_block(diff), unsafe_allow_html=True)

        if log:
            history.log_prediction(message, diff, p, source="gui")


def tab_repo() -> None:
    trained = _trained_models()
    if not trained:
        st.markdown('<div class="state-error">Train at least one model first.</div>', unsafe_allow_html=True)
        return

    st.markdown('<div class="eyebrow">local repository scan</div>', unsafe_allow_html=True)

    col_l, col_r = st.columns([2.2, 1])
    with col_l:
        path = st.text_input("repository path", value=str(Path.cwd()))
    with col_r:
        n = st.number_input("commits to scan", min_value=10, max_value=2000, value=100, step=10)
        model_name = st.selectbox("model", trained, index=0, key="repo_model")

    if not st.button("scan repository", type="primary"):
        return

    from pydriller import Repository

    repo_path = Path(path)
    if not (repo_path / ".git").exists():
        st.markdown(f'<div class="state-error">Not a git repo: {_e(str(repo_path))}</div>', unsafe_allow_html=True)
        return

    records = []
    meta = []
    with st.spinner("reading commits..."):
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
        st.markdown('<div class="state-empty">No commits found.</div>', unsafe_allow_html=True)
        return

    with st.spinner(f"classifying {len(records)} commits..."):
        preds = predict_batch(records, model_name)

    rows = []
    counts = {c: 0 for c in TARGET_CLASSES}
    for m, p, r in zip(meta, preds, records):
        rows.append({
            "hash": m["hash"],
            "author": m["author"],
            "date": m["date"].strftime("%Y-%m-%d") if hasattr(m["date"], "strftime") else str(m["date"])[:10],
            "label": p.label,
            "conf": f"{p.confidence:.3f}",
            "message": (r["message"].splitlines()[0][:120] if r["message"] else ""),
        })
        counts[p.label] = counts.get(p.label, 0) + 1

    sorted_counts = sorted(counts.items(), key=lambda kv: -kv[1])
    max_c = max(counts.values()) or 1

    st.markdown('<div class="eyebrow">class distribution</div>', unsafe_allow_html=True)
    hist_html = "".join(histogram_row(lbl, c, max_c) for lbl, c in sorted_counts)
    st.markdown(f'<div>{hist_html}</div>', unsafe_allow_html=True)

    st.markdown('<div class="eyebrow">commits</div>', unsafe_allow_html=True)
    table = pd.DataFrame(rows)
    st.dataframe(
        table,
        use_container_width=True,
        hide_index=True,
        column_config={
            "hash": st.column_config.TextColumn("HASH", width="small"),
            "author": st.column_config.TextColumn("AUTHOR", width="small"),
            "date": st.column_config.TextColumn("DATE", width="small"),
            "label": st.column_config.TextColumn("LABEL", width="small"),
            "conf": st.column_config.TextColumn("CONF", width="small"),
            "message": st.column_config.TextColumn("MESSAGE", width="large"),
        },
    )


def tab_history() -> None:
    st.markdown('<div class="eyebrow">prediction history</div>', unsafe_allow_html=True)
    rows = history.list_recent(limit=200)
    if not rows:
        st.markdown('<div class="state-empty">No predictions logged yet — run one on the predict tab.</div>',
                    unsafe_allow_html=True)
        return

    df = pd.DataFrame(rows)
    if "confidence" in df.columns:
        df["confidence"] = df["confidence"].astype(float).map("{:.3f}".format)
    st.dataframe(df, use_container_width=True, hide_index=True)

    dist = history.label_distribution()
    if dist:
        st.markdown('<div class="eyebrow">logged label distribution</div>', unsafe_allow_html=True)
        max_c = max(dist.values()) or 1
        ordered = sorted(dist.items(), key=lambda kv: -kv[1])
        hist_html = "".join(histogram_row(k, v, max_c) for k, v in ordered)
        st.markdown(f'<div>{hist_html}</div>', unsafe_allow_html=True)


def tab_metrics() -> None:
    reports_dir = MODELS_DIR / "reports"
    if not reports_dir.exists():
        st.markdown('<div class="state-empty">No reports yet — train and evaluate models first.</div>',
                    unsafe_allow_html=True)
        return

    summary = []
    for fp in sorted(reports_dir.glob("*.json")):
        if fp.stem.endswith("_val"):
            continue
        data = json.loads(fp.read_text())
        summary.append({
            "model": fp.stem,
            "accuracy": data["accuracy"],
            "macro_f1": data["macro_f1"],
            "weighted_f1": data["weighted_f1"],
            "macro_precision": data["macro_precision"],
            "macro_recall": data["macro_recall"],
        })

    if not summary:
        st.markdown('<div class="state-empty">No test reports yet.</div>', unsafe_allow_html=True)
        return

    summary.sort(key=lambda r: -r["macro_f1"])
    leader = summary[0]["model"]

    st.markdown('<div class="eyebrow">model comparison · test split</div>', unsafe_allow_html=True)
    cols = st.columns(len(summary))
    for col, row in zip(cols, summary):
        rest = {
            "macro F1":     row["macro_f1"],
            "weighted F1":  row["weighted_f1"],
            "precision":    row["macro_precision"],
            "recall":       row["macro_recall"],
        }
        with col:
            st.markdown(
                metric_card(row["model"], row["accuracy"], rest, is_leader=(row["model"] == leader)),
                unsafe_allow_html=True,
            )

    st.markdown('<div class="eyebrow">macro F1 — sorted desc</div>', unsafe_allow_html=True)
    max_f1 = max(r["macro_f1"] for r in summary)

    def _f1_bar(row: dict) -> str:
        pct = (row["macro_f1"] / max_f1) * 100
        return (
            '<div class="bar-row" style="--class-color:var(--ring-focus);">'
            f'  <span class="bar-label">{_e(row["model"])}</span>'
            f'  <div class="bar-track"><div class="bar-fill" style="width:{pct:.2f}%"></div></div>'
            f'  <span class="bar-value">{row["macro_f1"]:.4f}</span>'
            '</div>'
        )

    bars = "".join(_f1_bar(r) for r in summary)
    st.markdown(f'<div>{bars}</div>', unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    st.markdown(
        '<div class="app-header">'
        '  <h1>Commit Type Classifier</h1>'
        '  <div class="breadcrumb">'
        '    <b>c2</b> · ai for software engineering · usco · '
        '    <span class="mono">5 classifiers + 5 LLMs · 38,965 commits</span>'
        '  </div>'
        '</div>',
        unsafe_allow_html=True,
    )
    tabs = st.tabs(["Chat", "Generate", "Predict", "Repository", "History", "Metrics"])
    with tabs[0]: tab_chat()
    with tabs[1]: tab_generate()
    with tabs[2]: tab_predict()
    with tabs[3]: tab_repo()
    with tabs[4]: tab_history()
    with tabs[5]: tab_metrics()


if __name__ == "__main__":
    main()

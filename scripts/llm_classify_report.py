"""Render apples-to-apples comparison: LLM classifiers vs TF-IDF baseline."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from src.config import MODELS_DIR

SUMMARY = MODELS_DIR / "reports" / "llm_classify" / "_summary.csv"
OUT = MODELS_DIR / "reports" / "llm_classify" / "comparison.md"

BASELINE_ACC = 0.7093
BASELINE_F1 = 0.6632
BASELINE_WF1 = 0.7187


def main() -> int:
    if not SUMMARY.exists():
        print(f"missing {SUMMARY}", file=sys.stderr)
        return 1
    df = pd.read_csv(SUMMARY)
    df = df.drop_duplicates(subset=["model", "strategy", "sample_kind"], keep="last")
    df = df.sort_values("accuracy", ascending=False).reset_index(drop=True)

    pretty = df[["model", "strategy", "sample_kind", "n", "accuracy", "macro_f1",
                 "weighted_f1", "parse_failure_rate", "latency_ms_p50"]].copy()
    pretty["accuracy"] = (pretty["accuracy"] * 100).round(2).astype(str) + "%"
    pretty["macro_f1"] = pretty["macro_f1"].round(4)
    pretty["weighted_f1"] = pretty["weighted_f1"].round(4)
    pretty["parse_failure_rate"] = (pretty["parse_failure_rate"] * 100).round(1).astype(str) + "%"
    pretty["latency_ms_p50"] = pretty["latency_ms_p50"].round(0).astype(int)
    pretty.columns = ["model", "strategy", "sample", "n", "accuracy", "macro_F1",
                      "weighted_F1", "parse fail", "p50 ms"]
    leader = pretty.to_markdown(index=False)

    beats = df[df["accuracy"] >= BASELINE_ACC]
    beat_block = ""
    if len(beats):
        beat_block = (
            "\n## Configurations that match or beat the discriminative baseline\n\n"
            f"Baseline (TF-IDF + Logistic Regression on the natural-distribution "
            f"test set, n=5,845): **accuracy {BASELINE_ACC:.2%}** / macro-F1 "
            f"**{BASELINE_F1:.4f}** / weighted-F1 **{BASELINE_WF1:.4f}**.\n\n"
        )
        for _, r in beats.iterrows():
            beat_block += (
                f"* **{r['model']}** with `{r['strategy']}` "
                f"({r['sample_kind']}, n={int(r['n'])}): accuracy **{r['accuracy']:.2%}** · "
                f"macro-F1 **{r['macro_f1']:.4f}** · weighted-F1 **{r['weighted_f1']:.4f}** · "
                f"parse-failure {r['parse_failure_rate']:.1%} · p50 {int(r['latency_ms_p50'])} ms\n"
            )
    else:
        best = df.iloc[0]
        beat_block = (
            "\n## Gap to the discriminative baseline\n\n"
            f"Baseline accuracy: **{BASELINE_ACC:.2%}** / macro-F1 **{BASELINE_F1:.4f}**. "
            f"Best LLM configuration so far: **{best['model']}** with `{best['strategy']}` "
            f"at **{best['accuracy']:.2%}** accuracy / **{best['macro_f1']:.4f}** macro-F1, "
            f"i.e. a {(BASELINE_ACC - best['accuracy']) * 100:.1f} pp accuracy gap.\n\n"
            "The next step (heterogeneous voting ensemble combining the top LLM "
            "configurations, optionally with the TF-IDF baseline as a member) is "
            "implemented in `src/llm/voting_ensemble.py`.\n"
        )

    out = f"""# LLM-as-classifier — apples-to-apples vs the discriminative baseline

This evaluation feeds the **same input** (commit message + diff) as the
TF-IDF baseline to each LLM and asks it to output **one of**
{{`feat`, `fix`, `docs`, `refactor`, `test`}} — the same 5-way
classification task. Inference temperature 0.0 (greedy), seed 42.

## Full leaderboard

{leader}
{beat_block}
"""
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(out)
    print(out)
    print(f"\n→ wrote {OUT}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())

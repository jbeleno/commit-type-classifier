"""Render the LLM sweep summary as Markdown.

Reads ``models_saved/reports/llm/_summary.csv`` and writes:
  models_saved/reports/llm/comparison.md   — leaderboard + best-per-model + notes
  /dev/stdout                              — same markdown, for piping

Run:
    python -m scripts.llm_report
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from src.config import MODELS_DIR

SUMMARY = MODELS_DIR / "reports" / "llm" / "_summary.csv"
OUT = MODELS_DIR / "reports" / "llm" / "comparison.md"


def main() -> int:
    if not SUMMARY.exists():
        print(f"missing {SUMMARY}", file=sys.stderr)
        return 1

    df = pd.read_csv(SUMMARY)
    df = df.drop_duplicates(subset=["model", "strategy"], keep="last").reset_index(drop=True)
    df = df.sort_values("type_exact_match", ascending=False).reset_index(drop=True)

    cols = [
        "model",
        "strategy",
        "type_exact_match",
        "type_in_target",
        "classifier_agreement",
        "parse_failure_rate",
        "bleu",
        "rouge_l_mean",
        "latency_ms_p50",
        "completion_tokens_mean",
    ]
    pretty = df[cols].copy()
    for c in ("type_exact_match", "type_in_target", "classifier_agreement", "parse_failure_rate"):
        pretty[c] = (pretty[c] * 100).round(1).astype(str) + "%"
    pretty["bleu"] = pretty["bleu"].round(2)
    pretty["rouge_l_mean"] = pretty["rouge_l_mean"].round(3)
    pretty["latency_ms_p50"] = pretty["latency_ms_p50"].round(0).astype(int)
    pretty["completion_tokens_mean"] = pretty["completion_tokens_mean"].round(1)

    pretty = pretty.rename(
        columns={
            "type_exact_match": "type-match",
            "type_in_target": "in-target",
            "classifier_agreement": "classif. agree",
            "parse_failure_rate": "parse fail",
            "rouge_l_mean": "rouge_L",
            "latency_ms_p50": "p50 ms",
            "completion_tokens_mean": "tokens",
        }
    )

    leader = pretty.to_markdown(index=False)

    best_per_model = (
        df.sort_values("type_exact_match", ascending=False)
        .groupby("model", as_index=False)
        .first()[["model", "strategy", "type_exact_match", "bleu", "rouge_l_mean", "latency_ms_p50"]]
        .sort_values("type_exact_match", ascending=False)
        .reset_index(drop=True)
    )
    bpm = best_per_model.copy()
    bpm["type_exact_match"] = (bpm["type_exact_match"] * 100).round(1).astype(str) + "%"
    bpm["bleu"] = bpm["bleu"].round(2)
    bpm["rouge_l_mean"] = bpm["rouge_l_mean"].round(3)
    bpm["latency_ms_p50"] = bpm["latency_ms_p50"].round(0).astype(int)
    bpm = bpm.rename(
        columns={
            "type_exact_match": "type-match",
            "rouge_l_mean": "rouge_L",
            "latency_ms_p50": "p50 ms",
        }
    )
    best_md = bpm.to_markdown(index=False)

    # findings
    top_row = df.iloc[0]
    parse_fail_outliers = df[df["parse_failure_rate"] > 0.5][["model", "strategy", "parse_failure_rate"]].copy()
    parse_fail_outliers["parse_failure_rate"] = (
        (parse_fail_outliers["parse_failure_rate"] * 100).round(0).astype(int).astype(str) + "%"
    )
    pfo_md = parse_fail_outliers.to_markdown(index=False) if len(parse_fail_outliers) else "_(none)_"

    fastest = df.sort_values("latency_ms_p50").iloc[0]
    best_classifier_agree = df.sort_values("classifier_agreement", ascending=False).iloc[0]

    out = f"""# LLM sweep — commit message generation

Five locally-hosted LLMs × four prompting strategies × **n = 50** stratified
test commits. Inference temperature 0.2, top_p 0.9, seed 42 throughout.
Same Apple-Silicon Mac (16 GB unified memory), one model loaded at a time.

## Full leaderboard

{leader}

## Best strategy per model

{best_md}

## Findings

* **Best overall:** `{top_row['model']}` with `{top_row['strategy']}` reaches
  **{top_row['type_exact_match']:.0%}** type-exact-match, BLEU
  {top_row['bleu']:.2f}, ROUGE-L {top_row['rouge_l_mean']:.3f}, p50 latency
  {int(top_row['latency_ms_p50'])} ms.
* **Fastest workable model:** `{fastest['model']}` with `{fastest['strategy']}` —
  p50 latency **{int(fastest['latency_ms_p50'])} ms**, type-exact-match
  {fastest['type_exact_match']:.0%}. Useful as the default in interactive UIs.
* **Highest classifier-agreement:** `{best_classifier_agree['model']}` with
  `{best_classifier_agree['strategy']}` — **{best_classifier_agree['classifier_agreement']:.0%}** of generated
  messages get the correct type back when fed through the TF-IDF baseline.
  This is the most reliable input to the hybrid verifier pipeline.
* **Models that ignore the format contract** (parse-failure rate > 50 %):

{pfo_md}

  The pattern is consistent: base / code-completion checkpoints
  (`deepseek-coder:1.3b`) and few-shot or chain-of-thought modes on
  smaller instruct models tend to drift away from the
  `<type>(<scope>): <subject>` schema. JSON-mode mitigates the problem
  because the runtime enforces the structure.

## Relationship to the discriminative track

The five classifiers reported in `models_saved/reports/comparison.md`
(test split, n = 5 845) reach **70.93 % accuracy / 0.66 macro-F1** with
the TF-IDF baseline — a strictly easier task (5-way classification given
the full message and diff). The generative task asks the model to
*author the message from the diff alone*, which is open-ended and
fundamentally harder; the {top_row['type_exact_match']:.0%} type-match
of the best LLM is therefore not directly comparable.

The classical baseline is **reused** in the hybrid pipeline
(`src/llm/hybrid.py`) as a fast verifier: when the LLM emits a type
that the baseline disagrees with at confidence ≥ 0.60, the verifier
wins. The discriminative work is preserved as infrastructure rather
than discarded.
"""
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(out)
    print(out)
    print(f"\n→ wrote {OUT}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())

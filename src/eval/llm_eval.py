"""Evaluation harness for LLM-generated commit messages.

For each (model, strategy, sample) we measure:

  Type-level
    * type_exact_match     parsed_type == gold label
    * type_in_target       parsed_type ∈ {feat, fix, docs, refactor, test}
    * classifier_agreement TF-IDF baseline on generated message ⇒ gold label

  Text-level (subject line vs gold message)
    * bleu                 corpus BLEU-4 (sacrebleu)
    * rouge_l              ROUGE-L F1

  System-level
    * latency_ms_p50, _p95, _mean
    * completion_tokens_mean
    * parse_failure_rate   no conv-commit shape found

Outputs:
  models_saved/reports/llm/<model>__<strategy>.json   (per-example + summary)
  models_saved/reports/llm/_summary.csv               (one row per run, appended)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import scipy.sparse as sp
from rich.console import Console
from rich.table import Table
from rouge_score import rouge_scorer
from sacrebleu import corpus_bleu

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import CLASS_TO_IDX, IDX_TO_CLASS, MODELS_DIR, TARGET_CLASSES
from src.llm import ollama_client
from src.llm.generator import GeneratedCommit, few_shot_pool, generate_commit_message
from src.llm.prompts import STRATEGIES
from src.models.baseline_tfidf import ARTIFACT as BASELINE_PATH, NUMERIC_COLS
from src.utils import load_split

console = Console()
REPORT_DIR = MODELS_DIR / "reports" / "llm"
SUMMARY_CSV = REPORT_DIR / "_summary.csv"


def _stratified_sample(df: pd.DataFrame, n: int, seed: int = 42) -> pd.DataFrame:
    per_class = max(1, n // len(TARGET_CLASSES))
    chunks = []
    for label in TARGET_CLASSES:
        sub = df[df["label"] == label]
        if sub.empty:
            continue
        chunks.append(sub.sample(n=min(per_class, len(sub)), random_state=seed))
    return pd.concat(chunks, ignore_index=True).sample(frac=1, random_state=seed).reset_index(drop=True)


def _load_baseline_classifier():
    import joblib

    return joblib.load(BASELINE_PATH)


def _classifier_agreement(generated: List[GeneratedCommit], gold_labels: List[str]) -> List[bool]:
    """Pass each generated subject through the TF-IDF baseline; check vs gold."""
    bundle = _load_baseline_classifier()
    messages = [gc.subject or gc.raw.splitlines()[0] if gc.raw else "" for gc in generated]
    msg_mat = bundle["vec_msg"].transform(messages)
    diff_mat = bundle["vec_diff"].transform([""] * len(messages))
    num = bundle["scaler"].transform(np.zeros((len(messages), len(NUMERIC_COLS))))
    X = sp.hstack([msg_mat, diff_mat, sp.csr_matrix(num)]).tocsr()
    pred_ids = bundle["clf"].predict(X)
    pred_labels = [IDX_TO_CLASS[int(i)] for i in pred_ids]
    return [p == g for p, g in zip(pred_labels, gold_labels)]


def evaluate_run(
    model: str,
    strategy: str,
    n: int = 100,
    seed: int = 42,
    temperature: float = 0.2,
) -> Dict:
    test_df = load_split("test")
    train_df = load_split("train")
    sample = _stratified_sample(test_df, n, seed=seed)
    console.log(
        f"model={model}  strategy={strategy}  n={len(sample)}  "
        f"classes={sample['label'].value_counts().to_dict()}"
    )

    fs_pool = few_shot_pool(train_df, n_per_class=1) if strategy == "few_shot" else None

    rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    rows: List[Dict] = []
    generations: List[GeneratedCommit] = []
    gold_labels: List[str] = []
    gold_messages: List[str] = []

    t_start = time.time()
    for i, r in enumerate(sample.itertuples(index=False), 1):
        try:
            gc = generate_commit_message(
                diff=str(getattr(r, "diff_text", "")),
                model=model,
                strategy=strategy,
                few_shot_examples=fs_pool,
                temperature=temperature,
                seed=seed,
            )
        except Exception as exc:  # noqa: BLE001
            console.log(f"[red]  [{i}] {model}/{strategy} error: {exc}[/red]")
            continue

        generations.append(gc)
        gold_labels.append(r.label)
        gold_subject = str(getattr(r, "message_clean", "")).splitlines()[0] if getattr(r, "message_clean", "") else ""
        gold_messages.append(gold_subject)

        rl = rouge.score(gold_subject, gc.subject or gc.raw)["rougeL"].fmeasure
        rows.append(
            {
                "idx": i,
                "gold_label": r.label,
                "gold_subject": gold_subject,
                "pred_type": gc.parsed_type or "",
                "pred_subject": gc.subject,
                "raw": gc.raw,
                "type_exact_match": int(gc.parsed_type == r.label),
                "type_in_target": int(gc.type_in_target),
                "parse_failed": int(gc.parsed_type is None),
                "rouge_l": rl,
                "latency_ms": gc.latency_ms,
                "completion_tokens": gc.completion_tokens,
                "prompt_tokens": gc.prompt_tokens,
            }
        )
        if i % 20 == 0:
            console.log(f"  [{i}/{len(sample)}]  acc={np.mean([row['type_exact_match'] for row in rows]):.2%}")

    wall_time = time.time() - t_start

    if not generations:
        raise RuntimeError("No generations produced.")

    agree = _classifier_agreement(generations, gold_labels)
    for row, a in zip(rows, agree):
        row["classifier_agreement"] = int(a)

    bleu = corpus_bleu(
        [gc.subject or gc.raw for gc in generations],
        [gold_messages],
    ).score

    summary = {
        "model": model,
        "strategy": strategy,
        "n": len(generations),
        "wall_time_s": round(wall_time, 1),
        "type_exact_match": float(np.mean([r["type_exact_match"] for r in rows])),
        "type_in_target": float(np.mean([r["type_in_target"] for r in rows])),
        "classifier_agreement": float(np.mean([r["classifier_agreement"] for r in rows])),
        "parse_failure_rate": float(np.mean([r["parse_failed"] for r in rows])),
        "bleu": float(bleu),
        "rouge_l_mean": float(np.mean([r["rouge_l"] for r in rows])),
        "latency_ms_p50": float(np.percentile([r["latency_ms"] for r in rows], 50)),
        "latency_ms_p95": float(np.percentile([r["latency_ms"] for r in rows], 95)),
        "latency_ms_mean": float(np.mean([r["latency_ms"] for r in rows])),
        "completion_tokens_mean": float(np.mean([r["completion_tokens"] for r in rows])),
        "temperature": temperature,
        "seed": seed,
    }

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    safe_model = model.replace(":", "_").replace("/", "_")
    out_path = REPORT_DIR / f"{safe_model}__{strategy}.json"
    out_path.write_text(json.dumps({"summary": summary, "rows": rows}, indent=2))

    df_summary = pd.DataFrame([summary])
    if SUMMARY_CSV.exists():
        df_summary.to_csv(SUMMARY_CSV, mode="a", header=False, index=False)
    else:
        df_summary.to_csv(SUMMARY_CSV, index=False)

    _print_summary(summary)
    return summary


def _print_summary(s: Dict) -> None:
    t = Table(title=f"[bold]{s['model']}[/bold]  strategy={s['strategy']}  n={s['n']}")
    for k in (
        "type_exact_match",
        "type_in_target",
        "classifier_agreement",
        "parse_failure_rate",
        "bleu",
        "rouge_l_mean",
        "latency_ms_p50",
        "latency_ms_p95",
        "completion_tokens_mean",
    ):
        t.add_column(k, justify="right")
    t.add_row(
        f"{s['type_exact_match']:.2%}",
        f"{s['type_in_target']:.2%}",
        f"{s['classifier_agreement']:.2%}",
        f"{s['parse_failure_rate']:.2%}",
        f"{s['bleu']:.2f}",
        f"{s['rouge_l_mean']:.3f}",
        f"{s['latency_ms_p50']:.0f}",
        f"{s['latency_ms_p95']:.0f}",
        f"{s['completion_tokens_mean']:.0f}",
    )
    console.print(t)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--strategy", choices=list(STRATEGIES), required=True)
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.2)
    args = parser.parse_args()

    if not ollama_client.is_alive():
        console.print("[red]Ollama daemon not reachable at http://localhost:11434[/red]")
        return 2
    if args.model not in ollama_client.list_models():
        console.print(f"[red]Model '{args.model}' not pulled.[/red]")
        return 2

    evaluate_run(args.model, args.strategy, n=args.n, temperature=args.temperature)
    return 0


if __name__ == "__main__":
    sys.exit(main())

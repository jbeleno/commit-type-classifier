"""Apples-to-apples evaluation of the LLM in classifier mode.

Same input (message + diff) and same output space ({feat, fix, docs,
refactor, test}) as the discriminative baseline. Run on a stratified
subset of the test split; report accuracy, macro-F1, weighted-F1,
per-class precision/recall, parse-failure rate and latency.

Run:
    python -m src.eval.llm_classify_eval --model phi3.5:3.8b-mini-instruct-q4_K_M \
        --strategy few_shot --n 200
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table

from src.config import CLASS_TO_IDX, MODELS_DIR, TARGET_CLASSES
from src.llm import ollama_client
from src.llm.classifier import build_few_shot_pool, build_rag_examples, classify
from src.utils import evaluation_report, load_split, print_report, save_report

console = Console()
REPORT_DIR = MODELS_DIR / "reports" / "llm_classify"
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


def _natural_sample(df: pd.DataFrame, n: int, seed: int = 42) -> pd.DataFrame:
    """Random sample from the test split, preserves the natural class distribution.

    This is the fair comparison against the TF-IDF baseline whose 70.93 % is
    measured on the natural distribution (62 % fix).
    """
    return df.sample(n=min(n, len(df)), random_state=seed).reset_index(drop=True)


def evaluate_run(
    model: str,
    strategy: str,
    n: int = 200,
    seed: int = 42,
    temperature: float = 0.0,
    sample_kind: str = "natural",
) -> Dict:
    test_df = load_split("test")
    train_df = load_split("train")
    sampler = _natural_sample if sample_kind == "natural" else _stratified_sample
    sample = sampler(test_df, n, seed=seed)
    console.log(
        f"model={model}  strategy={strategy}  n={len(sample)}  "
        f"classes={sample['label'].value_counts().to_dict()}"
    )

    fs_pool: Optional[List[Dict[str, str]]] = None
    if strategy == "few_shot":
        fs_pool = build_few_shot_pool(train_df, n_per_class=1)

    rows: List[Dict] = []
    predictions: List[Optional[str]] = []
    latencies: List[float] = []
    parse_fails = 0
    t_start = time.time()

    for i, r in enumerate(sample.itertuples(index=False), 1):
        msg = str(getattr(r, "message_clean", "") or "")
        diff = str(getattr(r, "diff_text", "") or "")
        if strategy == "rag":
            examples = build_rag_examples(msg, diff, k=3)
            res = classify(msg, diff, model=model, strategy="few_shot",
                           few_shot_examples=examples, temperature=temperature, seed=seed)
            res.strategy = "rag"
        else:
            res = classify(msg, diff, model=model, strategy=strategy,
                           few_shot_examples=fs_pool, temperature=temperature, seed=seed)

        predictions.append(res.predicted)
        latencies.append(res.latency_ms)
        if res.predicted is None:
            parse_fails += 1
        rows.append({
            "idx": i,
            "gold": r.label,
            "pred": res.predicted or "",
            "correct": int(res.predicted == r.label),
            "parse_failed": int(res.predicted is None),
            "latency_ms": res.latency_ms,
            "completion_tokens": res.completion_tokens,
            "raw": res.raw[:120],
        })
        if i % 25 == 0:
            acc = np.mean([r["correct"] for r in rows])
            console.log(f"  [{i}/{len(sample)}]  acc={acc:.2%}  parse_fail={parse_fails}")

    wall = time.time() - t_start

    fallback_id = CLASS_TO_IDX["fix"]
    y_true = sample["label"].map(CLASS_TO_IDX).values.astype(int)
    y_pred = np.array(
        [CLASS_TO_IDX.get(p, fallback_id) for p in predictions],
        dtype=int,
    )

    rep = evaluation_report(y_true, y_pred, model_name=f"{model} [{strategy}]")
    print_report(rep)

    summary = {
        "model": model,
        "strategy": strategy,
        "sample_kind": sample_kind,
        "n": int(len(sample)),
        "wall_time_s": round(wall, 1),
        "accuracy": float(rep["accuracy"]),
        "macro_f1": float(rep["macro_f1"]),
        "weighted_f1": float(rep["weighted_f1"]),
        "macro_precision": float(rep["macro_precision"]),
        "macro_recall": float(rep["macro_recall"]),
        "parse_failure_rate": float(parse_fails / len(sample)),
        "latency_ms_p50": float(np.percentile(latencies, 50)),
        "latency_ms_p95": float(np.percentile(latencies, 95)),
        "completion_tokens_mean": float(np.mean([r["completion_tokens"] for r in rows])),
        "temperature": temperature,
        "seed": seed,
    }

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    safe = model.replace(":", "_").replace("/", "_")
    out = REPORT_DIR / f"{safe}__{strategy}.json"
    out.write_text(json.dumps({"summary": summary, "full_report": rep, "rows": rows}, indent=2))

    save_report(rep, REPORT_DIR / f"{safe}__{strategy}__sklearn_report.json")

    df_sum = pd.DataFrame([summary])
    if SUMMARY_CSV.exists():
        df_sum.to_csv(SUMMARY_CSV, mode="a", header=False, index=False)
    else:
        df_sum.to_csv(SUMMARY_CSV, index=False)

    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--strategy", choices=["zero_shot", "few_shot", "json_mode", "rag"], required=True)
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--sample-kind", choices=["natural", "stratified"], default="natural")
    args = parser.parse_args()

    if not ollama_client.is_alive():
        console.print("[red]Ollama daemon not reachable.[/red]")
        return 2
    if args.model not in ollama_client.list_models():
        console.print(f"[red]Model {args.model} not pulled.[/red]")
        return 2

    evaluate_run(
        args.model,
        args.strategy,
        n=args.n,
        temperature=args.temperature,
        sample_kind=args.sample_kind,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

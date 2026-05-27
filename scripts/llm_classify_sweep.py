"""Apples-to-apples sweep: every (model, strategy) classifies the same
random sample of the test split. Compare against the TF-IDF baseline at
70.93 % / macro-F1 0.6632.

Run:
    python -m scripts.llm_classify_sweep --n 200
    python -m scripts.llm_classify_sweep --n 200 --only-strategy rag
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
from rich.console import Console
from rich.table import Table

from src.eval.llm_classify_eval import SUMMARY_CSV, evaluate_run
from src.llm import ollama_client

console = Console()

MODELS = [
    "deepseek-coder:1.3b",
    "qwen2.5-coder:1.5b",
    "qwen2.5-coder:3b",
    "llama3.2:3b-instruct-q4_K_M",
    "phi3.5:3.8b-mini-instruct-q4_K_M",
]
STRATEGIES = ["zero_shot", "few_shot", "rag", "json_mode"]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--sample-kind", choices=["natural", "stratified"], default="natural")
    parser.add_argument("--only-strategy", default=None)
    parser.add_argument("--only-model", default=None)
    args = parser.parse_args()

    if not ollama_client.is_alive():
        console.print("[red]Ollama not reachable[/red]")
        return 2
    available = set(ollama_client.list_models())
    models = [m for m in MODELS if m in available]
    if args.only_model:
        models = [m for m in models if m == args.only_model]
    strategies = [args.only_strategy] if args.only_strategy else STRATEGIES

    total = len(models) * len(strategies)
    console.print(
        f"[bold]LLM-classify sweep: {len(models)} models × {len(strategies)} strategies "
        f"= {total} runs, n={args.n} ({args.sample_kind})[/bold]"
    )

    done = 0
    for model in models:
        for strategy in strategies:
            done += 1
            console.print(f"\n[cyan]→ [{done}/{total}] {model} | {strategy}[/cyan]")
            try:
                evaluate_run(model, strategy, n=args.n, sample_kind=args.sample_kind)
            except Exception as exc:  # noqa: BLE001
                console.print(f"[red]failed: {exc}[/red]")

    if SUMMARY_CSV.exists():
        df = pd.read_csv(SUMMARY_CSV)
        df = df.drop_duplicates(subset=["model", "strategy", "sample_kind"], keep="last")
        df = df.sort_values("accuracy", ascending=False)
        t = Table(title=f"LLM-as-classifier leaderboard ({args.sample_kind}, n={args.n})")
        for col in ("model", "strategy", "accuracy", "macro_f1", "weighted_f1",
                    "parse_failure_rate", "latency_ms_p50"):
            t.add_column(col, justify="right" if col != "model" and col != "strategy" else "left")
        for _, r in df.iterrows():
            t.add_row(
                r["model"], r["strategy"],
                f"{r['accuracy']:.2%}", f"{r['macro_f1']:.4f}", f"{r['weighted_f1']:.4f}",
                f"{r['parse_failure_rate']:.2%}", f"{r['latency_ms_p50']:.0f}",
            )
        console.print(t)
        baseline_acc, baseline_f1 = 0.7093, 0.6632
        beats = df[df["accuracy"] >= baseline_acc]
        if len(beats):
            console.print(
                f"\n[bold green]✓ {len(beats)} configuration(s) match or beat TF-IDF baseline "
                f"({baseline_acc:.2%}/{baseline_f1:.4f}):[/bold green]"
            )
            for _, r in beats.iterrows():
                console.print(f"  · {r['model']} | {r['strategy']} → acc={r['accuracy']:.2%}  f1={r['macro_f1']:.4f}")
        else:
            best = df.iloc[0]
            console.print(
                f"\n[yellow]✗ no LLM config beats baseline yet. "
                f"Best so far: {best['model']} / {best['strategy']} → "
                f"acc={best['accuracy']:.2%} (gap = {(baseline_acc - best['accuracy']) * 100:.1f} pp)[/yellow]"
            )

    return 0


if __name__ == "__main__":
    sys.exit(main())

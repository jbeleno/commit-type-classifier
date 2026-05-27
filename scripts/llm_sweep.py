"""Full sweep: every (model, strategy) combination on the same stratified sample.

Runs sequentially so we keep RAM low (one model loaded at a time). Writes
per-run JSON files and appends to ``models_saved/reports/llm/_summary.csv``.
Finally prints a leaderboard.

Run:
    python -m scripts.llm_sweep --n 50
    python -m scripts.llm_sweep --n 50 --only-strategy zero_shot
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

from src.eval.llm_eval import SUMMARY_CSV, evaluate_run
from src.llm import ollama_client
from src.llm.prompts import STRATEGIES

console = Console()

MODELS = [
    "deepseek-coder:1.3b",
    "qwen2.5-coder:1.5b",
    "qwen2.5-coder:3b",
    "llama3.2:3b-instruct-q4_K_M",
    "phi3.5:3.8b-mini-instruct-q4_K_M",
]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=50)
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
    missing = [m for m in MODELS if m not in available]
    if missing:
        console.print(f"[yellow]Missing models (skipping): {missing}[/yellow]")

    strategies = list(STRATEGIES)
    if args.only_strategy:
        strategies = [args.only_strategy]

    total = len(models) * len(strategies)
    console.print(
        f"[bold]Sweep: {len(models)} models × {len(strategies)} strategies = {total} runs, "
        f"n={args.n} each[/bold]"
    )

    done = 0
    for model in models:
        for strategy in strategies:
            done += 1
            console.print(f"\n[cyan]→ [{done}/{total}] {model}  |  {strategy}[/cyan]")
            try:
                evaluate_run(model, strategy, n=args.n)
            except Exception as exc:  # noqa: BLE001
                console.print(f"[red]✗ failed: {exc}[/red]")
                continue

    if SUMMARY_CSV.exists():
        df = pd.read_csv(SUMMARY_CSV)
        df = df.drop_duplicates(subset=["model", "strategy"], keep="last")
        df = df.sort_values("type_exact_match", ascending=False)
        t = Table(title="LLM sweep leaderboard")
        for col in (
            "model",
            "strategy",
            "n",
            "type_exact_match",
            "classifier_agreement",
            "bleu",
            "rouge_l_mean",
            "latency_ms_p50",
        ):
            t.add_column(col)
        for _, r in df.iterrows():
            t.add_row(
                r["model"],
                r["strategy"],
                str(int(r["n"])),
                f"{r['type_exact_match']:.2%}",
                f"{r['classifier_agreement']:.2%}",
                f"{r['bleu']:.2f}",
                f"{r['rouge_l_mean']:.3f}",
                f"{r['latency_ms_p50']:.0f}",
            )
        console.print(t)

    return 0


if __name__ == "__main__":
    sys.exit(main())

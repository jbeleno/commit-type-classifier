"""Smoke test for the LLM generator.

Picks N stratified samples from the test split, runs every prompt strategy
with a single (small) model, and prints a compact table. Verifies:
  - Ollama daemon is reachable
  - The chosen model is available
  - Every strategy parses correctly
  - Telemetry (latency, tokens) is captured

Run:
    python -m scripts.llm_smoke --model qwen2.5-coder:1.5b --n 10
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

from src.config import TARGET_CLASSES
from src.llm import ollama_client
from src.llm.generator import few_shot_pool, generate_commit_message
from src.llm.prompts import STRATEGIES
from src.utils import load_split

console = Console()


def _stratified_sample(df: pd.DataFrame, n: int, seed: int = 42) -> pd.DataFrame:
    per_class = max(1, n // len(TARGET_CLASSES))
    chunks = []
    for label in TARGET_CLASSES:
        sub = df[df["label"] == label]
        if sub.empty:
            continue
        chunks.append(sub.sample(n=min(per_class, len(sub)), random_state=seed))
    return pd.concat(chunks, ignore_index=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="qwen2.5-coder:1.5b")
    parser.add_argument("--n", type=int, default=10)
    parser.add_argument("--strategy", default=None, help="If set, only run this strategy.")
    args = parser.parse_args()

    if not ollama_client.is_alive():
        console.print("[red]Ollama daemon not reachable at http://localhost:11434[/red]")
        return 2

    available = ollama_client.list_models()
    if args.model not in available:
        console.print(f"[red]Model '{args.model}' not pulled. Available: {available}[/red]")
        return 2

    train_df = load_split("train")
    test_df = load_split("test")
    sample = _stratified_sample(test_df, args.n)
    console.log(f"sample size = {len(sample)}  classes = {sample['label'].value_counts().to_dict()}")

    fs_pool = few_shot_pool(train_df, n_per_class=1)
    strategies = [args.strategy] if args.strategy else list(STRATEGIES.keys())

    for strategy in strategies:
        table = Table(title=f"[bold]{args.model}[/bold]  strategy={strategy}", show_lines=False)
        table.add_column("#", justify="right")
        table.add_column("true")
        table.add_column("pred")
        table.add_column("✓", justify="center")
        table.add_column("ms", justify="right")
        table.add_column("message")

        ok = 0
        for i, row in enumerate(sample.itertuples(index=False), 1):
            gc = generate_commit_message(
                diff=str(getattr(row, "diff_text", "")),
                model=args.model,
                strategy=strategy,
                few_shot_examples=fs_pool if strategy == "few_shot" else None,
            )
            hit = gc.parsed_type == row.label
            ok += int(hit)
            table.add_row(
                str(i),
                row.label,
                gc.parsed_type or "—",
                "✓" if hit else "✗",
                f"{gc.latency_ms:.0f}",
                (gc.one_liner or gc.raw)[:80],
            )
        console.print(table)
        console.print(f"  type-exact-match = [bold]{ok}/{len(sample)} ({ok/len(sample):.0%})[/bold]\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Stratified 70/15/15 split with reproducible CSV outputs.

Reads ``data/processed/commits_clean.parquet`` and writes:
    data/splits/train.csv
    data/splits/val.csv
    data/splits/test.csv
    data/splits/split_summary.csv

Stratified on ``label`` so every split keeps the same class proportions
(critical to avoid data leakage and biased evaluation).

Run:
    python -m src.data.split
"""
from __future__ import annotations

import sys

import pandas as pd
from rich.console import Console
from sklearn.model_selection import train_test_split

from src.config import PROCESSED_DIR, RANDOM_SEED, SPLIT_RATIOS, SPLITS_DIR

console = Console()


def main() -> int:
    clean_path = PROCESSED_DIR / "commits_clean.parquet"
    if not clean_path.exists():
        console.print(f"[red]✗ Missing {clean_path}. Run src.data.preprocess first.[/red]")
        return 1

    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(clean_path)
    console.log(f"Total clean rows: {len(df):,}")

    train_size = SPLIT_RATIOS["train"]
    val_size = SPLIT_RATIOS["val"]
    test_size = SPLIT_RATIOS["test"]
    assert abs(train_size + val_size + test_size - 1.0) < 1e-6

    train_df, temp_df = train_test_split(
        df,
        train_size=train_size,
        random_state=RANDOM_SEED,
        stratify=df["label"],
    )
    rel_val = val_size / (val_size + test_size)
    val_df, test_df = train_test_split(
        temp_df,
        train_size=rel_val,
        random_state=RANDOM_SEED,
        stratify=temp_df["label"],
    )

    splits = {"train": train_df, "val": val_df, "test": test_df}
    summary_rows = []
    for name, frame in splits.items():
        path = SPLITS_DIR / f"{name}.csv"
        frame.to_csv(path, index=False)
        console.print(f"  ✓ {name:5s} → {path}  ({len(frame):,} rows)")
        dist = frame["label"].value_counts(normalize=True).sort_index()
        for label, pct in dist.items():
            summary_rows.append({"split": name, "label": label, "fraction": round(float(pct), 4)})

    summary = pd.DataFrame(summary_rows)
    summary_path = SPLITS_DIR / "split_summary.csv"
    summary.to_csv(summary_path, index=False)
    console.print(f"  ✓ summary → {summary_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

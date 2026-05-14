"""Download CommitBench dataset from Hugging Face Hub.

Tries multiple known mirrors of the CommitBench corpus. Saves the raw
dataset as Parquet under ``data/raw/`` for fast re-loading. A small
sample CSV is also saved for sanity checks.

Usage:
    python -m src.data.download                # full dataset
    python -m src.data.download --sample 50000 # stream first N rows
"""
from __future__ import annotations

import argparse
import sys
from typing import Iterable

import pandas as pd
from datasets import Dataset, load_dataset
from rich.console import Console

from src.config import RAW_DIR

console = Console()

CANDIDATE_DATASETS: list[dict] = [
    {
        "id": "Maxscha/commitbench",
        "split": "train",
        "message_col": "message",
        "diff_col": "diff",
        "label_col": None,
    },
    {
        "id": "mamiksik/CommitBench",
        "split": "train",
        "message_col": "message",
        "diff_col": "diff",
        "label_col": None,
    },
    {
        "id": "mamiksik/CommitChronicle",
        "split": "train",
        "message_col": "message",
        "diff_col": "mods",
        "label_col": None,
    },
]


def stream_to_dataframe(ds_iter: Iterable[dict], limit: int) -> pd.DataFrame:
    rows: list[dict] = []
    for i, row in enumerate(ds_iter):
        if i >= limit:
            break
        rows.append(row)
    return pd.DataFrame(rows)


def try_load(candidate: dict, sample: int | None) -> pd.DataFrame | None:
    ds_id = candidate["id"]
    split = candidate["split"]
    console.log(f"Attempting [bold]{ds_id}[/bold] split={split} ...")
    try:
        if sample is not None:
            stream = load_dataset(ds_id, split=split, streaming=True)
            df = stream_to_dataframe(stream, sample)
        else:
            ds = load_dataset(ds_id, split=split)
            assert isinstance(ds, Dataset)
            df = ds.to_pandas()
        console.log(
            f"  ✓ Loaded {len(df):,} rows, columns: {list(df.columns)[:8]}"
        )
        return df
    except Exception as exc:  # noqa: BLE001
        console.log(f"  ✗ Failed: {exc.__class__.__name__}: {exc}")
        return None


def main(sample: int | None) -> int:
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    df: pd.DataFrame | None = None
    chosen: dict | None = None
    for cand in CANDIDATE_DATASETS:
        df = try_load(cand, sample)
        if df is not None and len(df) > 0:
            chosen = cand
            break

    if df is None or chosen is None:
        console.print("[red]✗ No candidate dataset could be downloaded.[/red]")
        return 1

    parquet_path = RAW_DIR / "commits_raw.parquet"
    df.to_parquet(parquet_path, index=False)
    console.print(f"[green]✓ Saved raw dataset → {parquet_path}[/green]")

    sample_path = RAW_DIR / "commits_sample.csv"
    df.head(500).to_csv(sample_path, index=False)
    console.print(f"[green]✓ Saved 500-row sample → {sample_path}[/green]")

    meta_path = RAW_DIR / "source.txt"
    meta_path.write_text(
        f"source_dataset: {chosen['id']}\n"
        f"split: {chosen['split']}\n"
        f"rows: {len(df)}\n"
        f"columns: {list(df.columns)}\n"
        f"sample_mode: {sample is not None} (limit={sample})\n"
    )
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="If set, stream only the first N rows instead of full download.",
    )
    args = parser.parse_args()
    sys.exit(main(args.sample))

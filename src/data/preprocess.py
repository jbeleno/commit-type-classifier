"""Clean raw commits, extract Conventional Commit labels, derive diff features.

Pipeline:
    1. Load ``data/raw/commits_raw.parquet`` (output of ``download.py``).
    2. Extract the Conventional Commit type prefix (feat/fix/docs/...).
    3. Keep only rows whose label is in ``TARGET_CLASSES``.
    4. Strip the prefix from the message so the model cannot cheat.
    5. Compute numeric diff features (files_changed, lines_added, lines_removed).
    6. Trim the diff to the first ``MAX_DIFF_CHARS`` characters.
    7. Save to ``data/processed/commits_clean.parquet``.

Run:
    python -m src.data.preprocess
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd
from rich.console import Console

from src.config import PROCESSED_DIR, RAW_DIR, TARGET_CLASSES

console = Console()

CONV_PREFIX_RE = re.compile(
    r"^\s*(?P<type>feat|fix|docs|style|refactor|perf|test|build|ci|chore|revert)"
    r"(?:\([^)]*\))?(?P<bang>!)?\s*:\s*(?P<rest>.+)",
    re.IGNORECASE | re.DOTALL,
)

MAX_DIFF_CHARS = 4000
MAX_MESSAGE_CHARS = 500


def _find_first_message_field(df: pd.DataFrame) -> str:
    for cand in ("message", "msg", "commit_message", "subject"):
        if cand in df.columns:
            return cand
    raise KeyError(f"No message-like column found. Got: {list(df.columns)}")


def _find_first_diff_field(df: pd.DataFrame) -> str | None:
    for cand in ("diff", "patch", "mods", "code_diff", "changes"):
        if cand in df.columns:
            return cand
    return None


def extract_label(message: str) -> str | None:
    if not isinstance(message, str) or not message.strip():
        return None
    first_line = message.strip().splitlines()[0]
    m = CONV_PREFIX_RE.match(first_line)
    if not m:
        return None
    return m.group("type").lower()


def strip_prefix(message: str) -> str:
    if not isinstance(message, str):
        return ""
    first_line, *rest = message.strip().splitlines()
    m = CONV_PREFIX_RE.match(first_line)
    if m:
        first_line = m.group("rest").strip()
    cleaned = "\n".join([first_line, *rest]).strip()
    return cleaned[:MAX_MESSAGE_CHARS]


def diff_to_text_and_features(diff: object) -> tuple[str, int, int, int]:
    """Return (truncated_diff_text, files_changed, lines_added, lines_removed)."""
    text = _normalize_diff(diff)
    if not text:
        return "", 0, 0, 0
    lines_added = sum(
        1 for line in text.splitlines() if line.startswith("+") and not line.startswith("+++")
    )
    lines_removed = sum(
        1 for line in text.splitlines() if line.startswith("-") and not line.startswith("---")
    )
    files_changed = sum(1 for line in text.splitlines() if line.startswith("diff --git"))
    files_changed = max(files_changed, 1) if text else 0
    return text[:MAX_DIFF_CHARS], files_changed, lines_added, lines_removed


def _normalize_diff(diff: object) -> str:
    if isinstance(diff, str):
        return diff
    if isinstance(diff, list):
        # CommitChronicle stores diffs as list[dict]
        parts: list[str] = []
        for mod in diff:
            if isinstance(mod, dict):
                old_p = mod.get("old_path") or mod.get("filename") or ""
                new_p = mod.get("new_path") or mod.get("filename") or ""
                parts.append(f"diff --git a/{old_p} b/{new_p}")
                body = mod.get("diff") or mod.get("patch") or ""
                if body:
                    parts.append(str(body))
        return "\n".join(parts)
    return ""


def main() -> int:
    raw_path = RAW_DIR / "commits_raw.parquet"
    if not raw_path.exists():
        console.print(f"[red]✗ Missing {raw_path}. Run src.data.download first.[/red]")
        return 1

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    console.log(f"Reading {raw_path} ...")
    df = pd.read_parquet(raw_path)
    console.log(f"  raw rows: {len(df):,}")

    msg_col = _find_first_message_field(df)
    diff_col = _find_first_diff_field(df)
    console.log(f"  message column: {msg_col}, diff column: {diff_col}")

    df["label"] = df[msg_col].map(extract_label)
    before = len(df)
    df = df[df["label"].isin(TARGET_CLASSES)].copy()
    console.log(f"  after Conventional Commit filter: {len(df):,} / {before:,}")

    df["message_clean"] = df[msg_col].map(strip_prefix)

    if diff_col is not None:
        diff_features = df[diff_col].map(diff_to_text_and_features)
        df["diff_text"] = diff_features.map(lambda t: t[0])
        df["files_changed"] = diff_features.map(lambda t: t[1])
        df["lines_added"] = diff_features.map(lambda t: t[2])
        df["lines_removed"] = diff_features.map(lambda t: t[3])
    else:
        df["diff_text"] = ""
        df["files_changed"] = 0
        df["lines_added"] = 0
        df["lines_removed"] = 0

    df = df[df["message_clean"].str.len() > 0].copy()
    keep_cols = [
        "message_clean",
        "diff_text",
        "files_changed",
        "lines_added",
        "lines_removed",
        "label",
    ]
    df = df[keep_cols].reset_index(drop=True)

    console.print("\n[bold]Class distribution:[/bold]")
    console.print(df["label"].value_counts())

    out_path = PROCESSED_DIR / "commits_clean.parquet"
    df.to_parquet(out_path, index=False)
    console.print(f"\n[green]✓ Saved → {out_path}  ({len(df):,} rows)[/green]")
    return 0


if __name__ == "__main__":
    sys.exit(main())

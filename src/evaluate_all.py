"""Run every trained model on the test split, produce a comparison table.

Outputs:
    models_saved/reports/comparison.csv
    models_saved/reports/comparison.md   (Markdown for the C2 doc section)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
from rich.console import Console
from rich.table import Table

from src.config import MODELS_DIR

console = Console()
REPORTS_DIR = MODELS_DIR / "reports"


def _load_reports() -> list[dict]:
    rows = []
    for fp in sorted(REPORTS_DIR.glob("*.json")):
        if fp.stem.endswith("_val"):
            continue
        data = json.loads(fp.read_text())
        rows.append(
            {
                "model": fp.stem,
                "accuracy": round(data["accuracy"], 4),
                "macro_f1": round(data["macro_f1"], 4),
                "weighted_f1": round(data["weighted_f1"], 4),
                "macro_precision": round(data["macro_precision"], 4),
                "macro_recall": round(data["macro_recall"], 4),
                "n_test": data["n_samples"],
            }
        )
    return rows


def main() -> int:
    if not REPORTS_DIR.exists():
        console.print("[red]✗ No reports directory; train + evaluate models first.[/red]")
        return 1
    rows = _load_reports()
    if not rows:
        console.print("[yellow]No test reports found yet.[/yellow]")
        return 1

    df = pd.DataFrame(rows).sort_values("macro_f1", ascending=False)
    df.to_csv(REPORTS_DIR / "comparison.csv", index=False)

    md_lines = ["# Model comparison (test split)\n", df.to_markdown(index=False), "\n"]
    (REPORTS_DIR / "comparison.md").write_text("\n".join(md_lines))

    table = Table(title="Test-set comparison (sorted by macro F1)")
    for col in df.columns:
        table.add_column(col, justify="right" if col != "model" else "left")
    for _, row in df.iterrows():
        table.add_row(*[str(v) for v in row])
    console.print(table)
    console.print(f"[green]✓ Wrote {REPORTS_DIR / 'comparison.csv'} and comparison.md[/green]")
    return 0


if __name__ == "__main__":
    sys.exit(main())

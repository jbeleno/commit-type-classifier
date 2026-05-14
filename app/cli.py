"""Command-line interface — Typer-based.

Examples:
    python -m app.cli predict --message "fix login bug" \\
        --diff "$(git show HEAD --no-color)"
    python -m app.cli repo --path . --last 100
    python -m app.cli batch commits.csv
    python -m app.cli history
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import List

import typer
from rich.console import Console
from rich.table import Table

from src import history
from src.config import TARGET_CLASSES
from src.inference import AVAILABLE_MODELS, model_is_available, predict, predict_batch

app = typer.Typer(add_completion=False, help="C2 Commit Classifier — CLI")
console = Console()


def _emit(prediction, message_preview: str = "") -> None:
    bars = ", ".join(f"{k}={v:.2f}" for k, v in prediction.probabilities.items())
    console.print(
        f"[bold green]{prediction.label}[/bold green] "
        f"({prediction.confidence:.2%})  via [cyan]{prediction.model}[/cyan]"
    )
    console.print(f"  probs: {bars}")
    if message_preview:
        console.print(f"  msg:   {message_preview[:120]}")


@app.command()
def predict_cmd(
    message: str = typer.Option(..., "--message", "-m", help="Commit message text."),
    diff: str = typer.Option("", "--diff", "-d", help="Optional unified diff text."),
    model: str = typer.Option("baseline_tfidf", "--model", help=f"One of: {AVAILABLE_MODELS}"),
    log: bool = typer.Option(True, "--log/--no-log", help="Save to SQLite history."),
):
    """Classify a single commit."""
    p = predict(message, diff, model)
    _emit(p)
    if log:
        history.log_prediction(message, diff, p, source="cli")


@app.command(name="predict")
def predict_alias(
    message: str = typer.Option(..., "--message", "-m"),
    diff: str = typer.Option("", "--diff", "-d"),
    model: str = typer.Option("baseline_tfidf", "--model"),
    log: bool = typer.Option(True, "--log/--no-log"),
):
    predict_cmd.callback(message=message, diff=diff, model=model, log=log)  # type: ignore


@app.command()
def repo(
    path: Path = typer.Option(Path("."), "--path", "-p", help="Path to git repo."),
    last: int = typer.Option(50, "--last", "-n", help="How many commits to analyze."),
    model: str = typer.Option("baseline_tfidf", "--model"),
    log: bool = typer.Option(False, "--log/--no-log"),
):
    """Classify the last N commits of a local git repository."""
    try:
        from pydriller import Repository
    except ImportError as e:
        console.print(f"[red]pydriller is required: {e}[/red]")
        raise typer.Exit(2)

    if not (path / ".git").exists():
        console.print(f"[red]Not a git repo: {path}[/red]")
        raise typer.Exit(2)

    records: List[dict] = []
    metadata: List[dict] = []
    repo_iter = Repository(str(path), only_no_merge=True, order="reverse").traverse_commits()
    for i, commit in enumerate(repo_iter):
        if i >= last:
            break
        diff_text = "\n".join(
            f"diff --git a/{m.old_path or m.new_path} b/{m.new_path or m.old_path}\n{m.diff or ''}"
            for m in commit.modified_files
        )
        records.append({"message": commit.msg, "diff": diff_text})
        metadata.append({"hash": commit.hash[:10], "author": commit.author.name})

    if not records:
        console.print("[yellow]No commits found.[/yellow]")
        return

    preds = predict_batch(records, model)

    table = Table(title=f"Classified last {len(preds)} commits — model: {model}")
    table.add_column("#", justify="right")
    table.add_column("hash")
    table.add_column("label")
    table.add_column("conf", justify="right")
    table.add_column("message")
    counts = {c: 0 for c in TARGET_CLASSES}
    for i, (p, meta, rec) in enumerate(zip(preds, metadata, records), 1):
        counts[p.label] += 1
        table.add_row(
            str(i),
            meta["hash"],
            p.label,
            f"{p.confidence:.2f}",
            (rec["message"] or "").splitlines()[0][:80],
        )
        if log:
            history.log_prediction(rec["message"], rec["diff"], p, source="cli-repo")
    console.print(table)

    summary = Table(title="Class distribution")
    summary.add_column("label")
    summary.add_column("count", justify="right")
    summary.add_column("pct", justify="right")
    total = sum(counts.values())
    for k, v in counts.items():
        summary.add_row(k, str(v), f"{(v / total) * 100:.1f}%")
    console.print(summary)


@app.command()
def batch(
    csv_path: Path = typer.Argument(..., help="CSV with 'message' and optional 'diff' columns."),
    model: str = typer.Option("baseline_tfidf", "--model"),
    out: Path = typer.Option(Path("batch_predictions.csv"), "--out", "-o"),
    log: bool = typer.Option(False, "--log/--no-log"),
):
    """Classify a CSV file of commits."""
    if not csv_path.exists():
        console.print(f"[red]Missing file: {csv_path}[/red]")
        raise typer.Exit(2)

    with csv_path.open(newline="") as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)

    records = [{"message": r.get("message", ""), "diff": r.get("diff", "")} for r in rows]
    preds = predict_batch(records, model)

    with out.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["message", "predicted_label", "confidence", "probabilities"])
        for row, p in zip(rows, preds):
            w.writerow(
                [row.get("message", ""), p.label, f"{p.confidence:.4f}", json.dumps(p.probabilities)]
            )
            if log:
                history.log_prediction(row.get("message", ""), row.get("diff", ""), p, source="cli-batch")

    console.print(f"[green]✓ Wrote {len(preds)} rows → {out}[/green]")


@app.command()
def models():
    """List models and whether each one has been trained."""
    table = Table(title="Available models")
    table.add_column("model")
    table.add_column("trained?", justify="center")
    for name in AVAILABLE_MODELS:
        table.add_row(name, "✓" if model_is_available(name) else "—")
    console.print(table)


@app.command(name="history")
def history_cmd(limit: int = typer.Option(20, "--limit", "-n")):
    """Show last N predictions from local history."""
    rows = history.list_recent(limit=limit)
    if not rows:
        console.print("[yellow]No predictions logged yet.[/yellow]")
        return
    table = Table(title=f"Last {len(rows)} predictions")
    for col in ("id", "ts", "model", "label", "confidence", "source", "message"):
        table.add_column(col)
    for r in rows:
        table.add_row(
            str(r["id"]),
            r["ts"][:19],
            r["model"],
            r["label"],
            f"{r['confidence']:.2f}",
            r["source"],
            r["message"][:60],
        )
    console.print(table)


if __name__ == "__main__":
    app()

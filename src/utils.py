"""Shared utilities: split loaders, metrics, class-weight helper."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.utils.class_weight import compute_class_weight

from src.config import CLASS_TO_IDX, IDX_TO_CLASS, SPLITS_DIR, TARGET_CLASSES


def load_split(name: str) -> pd.DataFrame:
    """Load one of train/val/test split CSVs."""
    path = SPLITS_DIR / f"{name}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing split file: {path}. Run src.data.split first.")
    df = pd.read_csv(path)
    df["message_clean"] = df["message_clean"].fillna("").astype(str)
    df["diff_text"] = df["diff_text"].fillna("").astype(str)
    df["label_id"] = df["label"].map(CLASS_TO_IDX)
    return df


def load_all_splits() -> Dict[str, pd.DataFrame]:
    return {name: load_split(name) for name in ("train", "val", "test")}


def class_weights_for(y: np.ndarray) -> Dict[int, float]:
    classes = np.array(list(IDX_TO_CLASS.keys()))
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y)
    return {int(c): float(w) for c, w in zip(classes, weights)}


def evaluation_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
) -> Dict[str, object]:
    labels = list(IDX_TO_CLASS.keys())
    target_names = [IDX_TO_CLASS[i] for i in labels]

    macro_f1 = f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", labels=labels, zero_division=0)
    macro_precision = precision_score(y_true, y_pred, average="macro", labels=labels, zero_division=0)
    macro_recall = recall_score(y_true, y_pred, average="macro", labels=labels, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    per_class_report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=target_names,
        zero_division=0,
        output_dict=True,
    )

    return {
        "model_name": model_name,
        "n_samples": int(len(y_true)),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "accuracy": float((y_true == y_pred).mean()),
        "confusion_matrix": cm.tolist(),
        "labels_order": target_names,
        "per_class": per_class_report,
    }


def save_report(report: Dict[str, object], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))


def print_report(report: Dict[str, object]) -> None:
    from rich.console import Console
    from rich.table import Table

    console = Console()
    console.print(f"\n[bold cyan]{report['model_name']}[/bold cyan]")
    console.print(f"  accuracy     = {report['accuracy']:.4f}")
    console.print(f"  macro F1     = {report['macro_f1']:.4f}")
    console.print(f"  weighted F1  = {report['weighted_f1']:.4f}")
    console.print(f"  macro prec.  = {report['macro_precision']:.4f}")
    console.print(f"  macro recall = {report['macro_recall']:.4f}")

    cm = report["confusion_matrix"]
    labels = report["labels_order"]
    table = Table(title="Confusion matrix (rows=true, cols=pred)")
    table.add_column("")
    for label in labels:
        table.add_column(label, justify="right")
    for i, label in enumerate(labels):
        table.add_row(label, *[str(v) for v in cm[i]])
    console.print(table)

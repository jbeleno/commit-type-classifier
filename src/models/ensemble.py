"""Model 5 — Heterogeneous soft-voting ensemble.

Combines probability outputs from the four base classifiers via weighted
soft voting. Weights are tuned on the validation set to maximize
macro-F1 (L-BFGS-B).

This module **reads pre-computed probabilities** from
``models_saved/probs/<model>__<split>.npy`` instead of running each base
model itself. Generate those files first:

    python -m src.models.precompute_probs --model baseline_tfidf
    python -m src.models.precompute_probs --model cnn_text
    python -m src.models.precompute_probs --model distilbert
    python -m src.models.precompute_probs --model codebert
    python -m src.models.ensemble all

Why? Loading TensorFlow (CNN) and PyTorch (transformers) in the same
process competes for Apple MPS device memory and hangs.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from rich.console import Console
from scipy.optimize import minimize
from sklearn.metrics import f1_score

from src.config import IDX_TO_CLASS, MODELS_DIR
from src.utils import evaluation_report, load_split, print_report, save_report

console = Console()
ARTIFACT_DIR = MODELS_DIR / "ensemble"
PROBS_DIR = MODELS_DIR / "probs"
REPORT_PATH = MODELS_DIR / "reports" / "ensemble.json"

CANDIDATES = ["baseline_tfidf", "cnn_text", "distilbert", "codebert"]


def _available_models(split: str) -> list[str]:
    return [m for m in CANDIDATES if (PROBS_DIR / f"{m}__{split}.npy").exists()]


def _stack(split: str) -> tuple[np.ndarray, list[str]]:
    models = _available_models(split)
    if not models:
        raise FileNotFoundError(
            f"No probability files found in {PROBS_DIR}. "
            "Run precompute_probs.py for each base model first."
        )
    arr = np.stack([np.load(PROBS_DIR / f"{m}__{split}.npy") for m in models], axis=0)
    return arr, models


def _weighted_vote(stack: np.ndarray, weights: np.ndarray) -> np.ndarray:
    w = weights / weights.sum()
    return (stack * w[:, None, None]).sum(axis=0)


def _objective_factory(stack: np.ndarray, y: np.ndarray):
    def neg_macro_f1(weights: np.ndarray) -> float:
        weights = np.maximum(weights, 1e-4)
        combo = _weighted_vote(stack, weights)
        y_pred = combo.argmax(axis=1)
        return -f1_score(
            y, y_pred, average="macro", labels=list(IDX_TO_CLASS.keys()), zero_division=0
        )

    return neg_macro_f1


def train() -> int:
    val_stack, models = _stack("val")
    y_val = load_split("val")["label_id"].values.astype(int)
    console.log(f"Base models in ensemble: {models}")

    init = np.ones(len(models))
    bounds = [(0.0, 10.0)] * len(models)
    result = minimize(_objective_factory(val_stack, y_val), init, method="L-BFGS-B", bounds=bounds)
    weights = result.x / result.x.sum()
    console.log(f"Optimal weights (val macro-F1 = {-result.fun:.4f}):")
    for name, w in zip(models, weights):
        console.log(f"  {name:>15s} = {w:.4f}")

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    (ARTIFACT_DIR / "weights.json").write_text(
        json.dumps({"models": models, "weights": weights.tolist()}, indent=2)
    )

    val_pred = _weighted_vote(val_stack, weights).argmax(axis=1)
    report = evaluation_report(y_val, val_pred, model_name="ensemble [val]")
    print_report(report)
    save_report(report, REPORT_PATH.with_name("ensemble_val.json"))
    return 0


def evaluate() -> int:
    cfg_path = ARTIFACT_DIR / "weights.json"
    if not cfg_path.exists():
        console.print("[red]✗ Ensemble weights not found. Train first.[/red]")
        return 1
    cfg = json.loads(cfg_path.read_text())
    weights = np.array(cfg["weights"])
    expected = cfg["models"]

    test_stack, models = _stack("test")
    if models != expected:
        console.log(f"[yellow]⚠ model order changed (was {expected}, now {models})[/yellow]")
    y_test = load_split("test")["label_id"].values.astype(int)
    y_pred = _weighted_vote(test_stack, weights).argmax(axis=1)
    report = evaluation_report(y_test, y_pred, model_name="ensemble [test]")
    print_report(report)
    save_report(report, REPORT_PATH)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["train", "eval", "all"])
    args = parser.parse_args()
    if args.action == "train":
        return train()
    if args.action == "eval":
        return evaluate()
    rc = train()
    return rc or evaluate()


if __name__ == "__main__":
    sys.exit(main())

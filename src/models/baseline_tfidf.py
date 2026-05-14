"""Model 1 — TF-IDF + Logistic Regression baseline.

Pipeline:
    [message_clean]  --TF-IDF(word, 1-2gram)-->  sparse_M
    [diff_text]      --TF-IDF(char, 3-5gram)-->  sparse_D
    [numeric]        --StandardScaler-->         dense_N
    concat([M, D, N])  -->  LogisticRegression(class_weight="balanced")

Saved artifact:
    models_saved/baseline_tfidf.joblib

Run:
    python -m src.models.baseline_tfidf train
    python -m src.models.baseline_tfidf eval
"""
from __future__ import annotations

import argparse
import sys

import joblib
import numpy as np
import scipy.sparse as sp
from rich.console import Console
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.config import MODELS_DIR
from src.utils import evaluation_report, load_split, print_report, save_report

console = Console()
ARTIFACT = MODELS_DIR / "baseline_tfidf.joblib"
REPORT_PATH = MODELS_DIR / "reports" / "baseline_tfidf.json"

NUMERIC_COLS = ["files_changed", "lines_added", "lines_removed"]


def _featurize(df, vec_msg, vec_diff, scaler, fit: bool):
    msg = df["message_clean"].values
    diff = df["diff_text"].values
    num = df[NUMERIC_COLS].astype(float).values

    if fit:
        m_mat = vec_msg.fit_transform(msg)
        d_mat = vec_diff.fit_transform(diff)
        n_mat = scaler.fit_transform(num)
    else:
        m_mat = vec_msg.transform(msg)
        d_mat = vec_diff.transform(diff)
        n_mat = scaler.transform(num)

    return sp.hstack([m_mat, d_mat, sp.csr_matrix(n_mat)]).tocsr()


def train() -> int:
    train_df = load_split("train")
    val_df = load_split("val")
    console.log(f"train={len(train_df):,} val={len(val_df):,}")

    vec_msg = TfidfVectorizer(
        max_features=30_000,
        ngram_range=(1, 2),
        analyzer="word",
        sublinear_tf=True,
        min_df=2,
    )
    vec_diff = TfidfVectorizer(
        max_features=30_000,
        ngram_range=(3, 5),
        analyzer="char_wb",
        sublinear_tf=True,
        min_df=2,
    )
    scaler = StandardScaler()

    console.log("Fitting vectorizers + scaler on TRAIN ...")
    X_train = _featurize(train_df, vec_msg, vec_diff, scaler, fit=True)
    X_val = _featurize(val_df, vec_msg, vec_diff, scaler, fit=False)
    y_train = train_df["label_id"].values
    y_val = val_df["label_id"].values

    console.log("Training LogisticRegression (saga, balanced class weight) ...")
    clf = LogisticRegression(
        solver="saga",
        max_iter=400,
        n_jobs=-1,
        class_weight="balanced",
        C=1.0,
        verbose=0,
    )
    clf.fit(X_train, y_train)

    val_pred = clf.predict(X_val)
    report = evaluation_report(y_val, val_pred, model_name="baseline_tfidf [val]")
    print_report(report)
    save_report(report, REPORT_PATH.with_name("baseline_tfidf_val.json"))

    ARTIFACT.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "vec_msg": vec_msg,
            "vec_diff": vec_diff,
            "scaler": scaler,
            "clf": clf,
        },
        ARTIFACT,
    )
    console.print(f"[green]✓ Saved artifact → {ARTIFACT}[/green]")
    return 0


def evaluate() -> int:
    if not ARTIFACT.exists():
        console.print(f"[red]✗ Missing {ARTIFACT}. Train first.[/red]")
        return 1
    bundle = joblib.load(ARTIFACT)
    test_df = load_split("test")
    X_test = _featurize(test_df, bundle["vec_msg"], bundle["vec_diff"], bundle["scaler"], fit=False)
    y_test = test_df["label_id"].values
    y_pred = bundle["clf"].predict(X_test)
    report = evaluation_report(y_test, y_pred, model_name="baseline_tfidf [test]")
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

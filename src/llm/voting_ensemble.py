"""Heterogeneous LLM voting ensemble for the classification task.

Reads ``models_saved/reports/llm_classify/<model>__<strategy>.json``
for the configurations the caller specifies, and combines the
per-example predictions of the chosen LLMs by:

  * `hard` voting — majority of predicted labels (ties broken by the
                    most-confident single classifier on the global
                    accuracy ranking)
  * `weighted` voting — each model contributes its accuracy as weight,
                        the label with the highest summed weight wins

We can also fold the **TF-IDF baseline classifier** into the vote
because each rows JSON file contains the message+diff, so a single pass
of the baseline classifier produces aligned predictions.

Run:
    python -m src.llm.voting_ensemble \
        --members phi3.5:3.8b-mini-instruct-q4_K_M__rag \
                  qwen2.5-coder:3b__rag \
                  llama3.2:3b-instruct-q4_K_M__zero_shot \
        --include-tfidf
        --mode weighted
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
import pandas as pd
import scipy.sparse as sp
from rich.console import Console
from rich.table import Table
from sklearn.metrics import f1_score

from src.config import CLASS_TO_IDX, IDX_TO_CLASS, MODELS_DIR, TARGET_CLASSES
from src.models.baseline_tfidf import ARTIFACT as BASELINE_PATH, NUMERIC_COLS
from src.utils import evaluation_report, load_split, print_report, save_report

console = Console()
REPORT_DIR = MODELS_DIR / "reports" / "llm_classify"


def _load_member(filename: str) -> Tuple[Dict, pd.DataFrame]:
    path = REPORT_DIR / f"{filename}.json"
    if not path.exists():
        raise FileNotFoundError(f"missing member file: {path}")
    data = json.loads(path.read_text())
    rows = pd.DataFrame(data["rows"])
    return data["summary"], rows


def _baseline_predict(messages: List[str], diffs: List[str]) -> List[str]:
    b = joblib.load(BASELINE_PATH)
    msg_mat = b["vec_msg"].transform(messages)
    diff_mat = b["vec_diff"].transform(diffs)
    num = b["scaler"].transform(np.zeros((len(messages), len(NUMERIC_COLS))))
    X = sp.hstack([msg_mat, diff_mat, sp.csr_matrix(num)]).tocsr()
    pred_ids = b["clf"].predict(X)
    return [IDX_TO_CLASS[int(i)] for i in pred_ids]


def _recover_test_inputs(rows: pd.DataFrame) -> pd.DataFrame:
    """Members store gold + idx but not the original message/diff. Re-attach
    them by re-sampling the test split with the same seed and n (we assume
    all members use the same sample)."""
    n = len(rows)
    test_df = load_split("test")
    sample = test_df.sample(n=n, random_state=42).reset_index(drop=True)
    if not (sample["label"].values == rows["gold"].values).all():
        raise RuntimeError(
            "member rows do not align with the natural-distribution sample "
            "produced by sample(n, random_state=42); check that all members "
            "used the same seed and sample_kind."
        )
    return sample


def vote_hard(member_preds: List[List[str]]) -> List[str]:
    out: List[str] = []
    for i in range(len(member_preds[0])):
        votes = [m[i] for m in member_preds]
        counts: Dict[str, int] = {}
        for v in votes:
            counts[v] = counts.get(v, 0) + 1
        out.append(max(counts.items(), key=lambda kv: kv[1])[0])
    return out


def vote_weighted(member_preds: List[List[str]], weights: List[float]) -> List[str]:
    out: List[str] = []
    for i in range(len(member_preds[0])):
        score: Dict[str, float] = {}
        for w, m in zip(weights, member_preds):
            pred = m[i]
            score[pred] = score.get(pred, 0.0) + w
        out.append(max(score.items(), key=lambda kv: kv[1])[0])
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--members",
        nargs="+",
        required=True,
        help="member file stems, e.g. phi3.5:3.8b-mini-instruct-q4_K_M__rag "
             "(use the underscored model name as stored on disk)",
    )
    parser.add_argument("--include-tfidf", action="store_true")
    parser.add_argument("--mode", choices=["hard", "weighted"], default="weighted")
    parser.add_argument("--weight-by", choices=["accuracy", "macro_f1"], default="accuracy")
    parser.add_argument("--tfidf-weight-boost", type=float, default=1.0,
                        help="multiplier on the TF-IDF weight to bias toward minority classes")
    parser.add_argument("--out", default="ensemble_llm_classifier")
    args = parser.parse_args()

    summaries: List[Dict] = []
    rows_per_member: List[pd.DataFrame] = []
    for m in args.members:
        s, r = _load_member(m)
        summaries.append(s)
        rows_per_member.append(r)
        console.log(f"loaded {m}: acc={s['accuracy']:.2%}  f1={s['macro_f1']:.4f}  n={s['n']}")

    n = len(rows_per_member[0])
    for r in rows_per_member[1:]:
        if len(r) != n:
            console.print("[red]members have different lengths; aborting[/red]")
            return 2
    gold = rows_per_member[0]["gold"].tolist()
    for r in rows_per_member[1:]:
        if r["gold"].tolist() != gold:
            console.print("[red]members do not share the same gold labels — re-run with same seed[/red]")
            return 2

    member_preds: List[List[str]] = [r["pred"].fillna("").tolist() for r in rows_per_member]
    names: List[str] = list(args.members)
    metric = args.weight_by
    weights: List[float] = [s[metric] for s in summaries]

    if args.include_tfidf:
        sample = _recover_test_inputs(rows_per_member[0])
        tfidf_preds = _baseline_predict(
            sample["message_clean"].astype(str).tolist(),
            sample["diff_text"].astype(str).tolist(),
        )
        member_preds.append(tfidf_preds)
        names.append("baseline_tfidf")
        # known test-set numbers from the baseline_tfidf report
        tfidf_metric = 0.7093 if metric == "accuracy" else 0.6632
        weights.append(tfidf_metric * args.tfidf_weight_boost)

    if args.mode == "hard":
        ensemble = vote_hard(member_preds)
    else:
        ensemble = vote_weighted(member_preds, weights)

    fallback_id = CLASS_TO_IDX["fix"]
    y_true = np.array([CLASS_TO_IDX[g] for g in gold], dtype=int)
    y_pred = np.array([CLASS_TO_IDX.get(p, fallback_id) for p in ensemble], dtype=int)

    rep = evaluation_report(y_true, y_pred, model_name=f"ensemble_llm [{args.mode}]")
    print_report(rep)

    t = Table(title="Per-member contributions")
    t.add_column("member")
    t.add_column("acc", justify="right")
    t.add_column("macro_f1", justify="right")
    t.add_column("weight", justify="right")
    for name, w, m_preds in zip(names, weights, member_preds):
        member_acc = np.mean([p == g for p, g in zip(m_preds, gold)])
        m_y_pred = np.array([CLASS_TO_IDX.get(p, fallback_id) for p in m_preds], dtype=int)
        member_f1 = f1_score(y_true, m_y_pred, average="macro", zero_division=0)
        t.add_row(name, f"{member_acc:.2%}", f"{member_f1:.4f}", f"{w:.4f}")
    console.print(t)

    save_report(rep, REPORT_DIR / f"{args.out}.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())

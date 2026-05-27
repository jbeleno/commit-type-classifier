"""KNN retrieval over the training corpus using the baseline TF-IDF vectorizer.

We reuse the diff TF-IDF vectorizer trained for ``baseline_tfidf`` so we
don't have to fit anything new. For a query diff:

    vec_diff.transform([diff])  →  cosine similarity vs train matrix
    top-k indices  →  train_df.iloc[idx]  →  few-shot examples

Building the train matrix on first call (cached). Memory: ~30k * 30k
sparse, but TF-IDF sparse stays very small in practice (~50 MB).
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.metrics.pairwise import linear_kernel

from src.models.baseline_tfidf import ARTIFACT as BASELINE_PATH
from src.utils import load_split


@dataclass
class RetrievedExample:
    type: str
    subject: str
    diff: str
    score: float


@lru_cache(maxsize=1)
def _train_index() -> Tuple[sp.csr_matrix, pd.DataFrame, object]:
    bundle = joblib.load(BASELINE_PATH)
    vec_diff = bundle["vec_diff"]
    train_df = load_split("train").reset_index(drop=True)
    diff_mat = vec_diff.transform(train_df["diff_text"].astype(str).values)
    return diff_mat, train_df, vec_diff


def retrieve(diff: str, k: int = 3) -> List[RetrievedExample]:
    if not diff or not diff.strip():
        return []
    diff_mat, train_df, vec_diff = _train_index()
    q = vec_diff.transform([diff])
    sims = linear_kernel(q, diff_mat).ravel()
    if sims.size == 0:
        return []
    top_idx = np.argpartition(-sims, kth=min(k, sims.size - 1))[:k]
    top_idx = top_idx[np.argsort(-sims[top_idx])]
    out: List[RetrievedExample] = []
    for i in top_idx:
        row = train_df.iloc[int(i)]
        subject = str(row["message_clean"]).splitlines()[0][:120] if row["message_clean"] else ""
        out.append(
            RetrievedExample(
                type=str(row["label"]),
                subject=subject,
                diff=str(row["diff_text"]),
                score=float(sims[i]),
            )
        )
    return out


def to_few_shot_format(examples: List[RetrievedExample]) -> List[Dict[str, str]]:
    return [{"type": e.type, "subject": e.subject, "diff": e.diff} for e in examples]

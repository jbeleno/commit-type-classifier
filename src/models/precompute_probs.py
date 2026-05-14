"""Pre-compute and cache each base model's probabilities on val + test.

Each model runs in isolation: load weights, predict, save .npy, exit.
This keeps TensorFlow (CNN) and PyTorch (transformers) from fighting
over the same MPS device in a single process.

Output layout::

    models_saved/probs/
        baseline_tfidf__val.npy
        baseline_tfidf__test.npy
        cnn_text__val.npy
        cnn_text__test.npy
        distilbert__val.npy
        distilbert__test.npy
        codebert__val.npy
        codebert__test.npy

Run:
    python -m src.models.precompute_probs --model baseline_tfidf
    python -m src.models.precompute_probs --model cnn_text
    python -m src.models.precompute_probs --model distilbert
    python -m src.models.precompute_probs --model codebert
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from src.config import MODELS_DIR
from src.utils import load_split

PROBS_DIR = MODELS_DIR / "probs"


def _save(arr: np.ndarray, name: str) -> None:
    PROBS_DIR.mkdir(parents=True, exist_ok=True)
    out = PROBS_DIR / name
    np.save(out, arr)
    print(f"  saved {arr.shape} -> {out}", flush=True)


def _baseline_tfidf() -> None:
    import joblib
    import scipy.sparse as sp

    from src.models.baseline_tfidf import ARTIFACT, NUMERIC_COLS

    bundle = joblib.load(ARTIFACT)
    for split in ("val", "test"):
        df = load_split(split)
        msg = bundle["vec_msg"].transform(df["message_clean"].astype(str).values)
        diff = bundle["vec_diff"].transform(df["diff_text"].astype(str).values)
        num = bundle["scaler"].transform(df[NUMERIC_COLS].astype(float).values)
        X = sp.hstack([msg, diff, sp.csr_matrix(num)]).tocsr()
        probs = bundle["clf"].predict_proba(X)
        _save(probs, f"baseline_tfidf__{split}.npy")


def _cnn_text() -> None:
    from src.models.cnn_text import BATCH_SIZE, _load_artifact, _to_dataset, _to_inputs

    model, scaler = _load_artifact()
    for split in ("val", "test"):
        df = load_split(split)
        inputs = _to_inputs(df, scaler, fit=False)
        ds = _to_dataset(inputs, None, BATCH_SIZE, shuffle=False)
        probs = model.predict(ds, verbose=0)
        _save(probs, f"cnn_text__{split}.npy")


def _hf_model(model_name: str, artifact_dir: Path) -> None:
    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    from src.models.distilbert_model import BATCH_SIZE, CommitDataset, MAX_SEQ_LEN, _device

    device = _device()
    tokenizer = AutoTokenizer.from_pretrained(artifact_dir)
    model = AutoModelForSequenceClassification.from_pretrained(artifact_dir).to(device).eval()
    for split in ("val", "test"):
        df = load_split(split)
        ds = CommitDataset(df, tokenizer, MAX_SEQ_LEN)
        loader = DataLoader(ds, batch_size=BATCH_SIZE * 2)
        all_probs: list[np.ndarray] = []
        with torch.no_grad():
            for batch in loader:
                batch.pop("labels")
                batch = {k: v.to(device) for k, v in batch.items()}
                logits = model(**batch).logits
                all_probs.append(torch.softmax(logits, dim=-1).cpu().numpy())
        probs = np.concatenate(all_probs, axis=0)
        _save(probs, f"{model_name}__{split}.npy")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        required=True,
        choices=["baseline_tfidf", "cnn_text", "distilbert", "codebert"],
    )
    args = parser.parse_args()

    if args.model == "baseline_tfidf":
        _baseline_tfidf()
    elif args.model == "cnn_text":
        _cnn_text()
    elif args.model == "distilbert":
        _hf_model("distilbert", MODELS_DIR / "distilbert")
    elif args.model == "codebert":
        _hf_model("codebert", MODELS_DIR / "codebert")
    return 0


if __name__ == "__main__":
    sys.exit(main())
